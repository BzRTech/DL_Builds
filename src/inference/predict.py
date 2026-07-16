"""Inferência sobre uma ortofoto inteira via sliding window com blending.

Percorre a imagem em janelas sobrepostas, roda o modelo em cada uma e combina as
probabilidades com uma janela de pesos (cosseno) para evitar costuras nas bordas.
Salva uma máscara de probabilidade georreferenciada (float32, 0-1).

Uso:
    python -m src.inference.predict --config configs/buildings.yaml --city cidade_b
    python -m src.inference.predict --config configs/buildings.yaml \
        --image data/processed/cidade_b.tif --output outputs/masks/cidade_b_prob.tif
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from src.train.model import build_model
from src.utils import ensure_dir, load_config


def _cosine_window(size: int) -> np.ndarray:
    """Janela 2D de pesos (produto de meias-ondas cosseno) para blending suave."""
    w1d = np.hanning(size + 2)[1:-1]  # evita zeros nas extremidades
    w1d = np.clip(w1d, 1e-3, None)
    return np.outer(w1d, w1d).astype(np.float32)


def _iter_windows(width: int, height: int, size: int, stride: int):
    cols = list(range(0, max(width - size, 0) + 1, stride))
    rows = list(range(0, max(height - size, 0) + 1, stride))
    if not cols or cols[-1] != width - size:
        cols.append(max(width - size, 0))
    if not rows or rows[-1] != height - size:
        rows.append(max(height - size, 0))
    for r in rows:
        for c in cols:
            yield c, r


def load_checkpoint(checkpoint: str, device: torch.device):
    """Carrega modelo + normalização a partir do checkpoint salvo no treino."""
    ckpt = torch.load(checkpoint, map_location=device)
    model = build_model(ckpt["config_model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = np.array(ckpt["norm_mean"], dtype=np.float32)
    std = np.array(ckpt["norm_std"], dtype=np.float32)
    return model, mean, std


def predict_image(image_path: str, output_path: str, cfg: dict) -> str:
    """Roda a inferência completa e salva a máscara de probabilidade."""
    icfg = cfg["inference"]
    bands = cfg["data_prep"]["bands"]
    size = icfg["window"]
    stride = size - icfg["overlap"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, mean, std = load_checkpoint(icfg["checkpoint"], device)
    weight = _cosine_window(size)

    # Acumuladores em disco (memmap) — a imagem pode ter dezenas de bilhões de
    # pixels, o que estouraria a RAM se fossem arrays em memória.
    tmp_dir = tempfile.mkdtemp(prefix="predict_", dir=Path(output_path).parent)
    try:
        with rasterio.open(image_path) as src:
            width, height = src.width, src.height
            profile = src.profile.copy()

            prob_sum = np.memmap(Path(tmp_dir) / "prob_sum.dat", dtype=np.float32,
                                 mode="w+", shape=(height, width))
            weight_sum = np.memmap(Path(tmp_dir) / "weight_sum.dat", dtype=np.float32,
                                   mode="w+", shape=(height, width))

            windows = list(_iter_windows(width, height, size, stride))
            print(f"[predict] {image_path}: {len(windows)} janelas "
                  f"({size}px, stride {stride}).")

            batch_imgs, batch_pos = [], []
            done = 0

            def flush():
                nonlocal done
                if not batch_imgs:
                    return
                x = torch.from_numpy(np.stack(batch_imgs)).to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device.type,
                                        enabled=(device.type == "cuda")):
                        probs = torch.sigmoid(model(x)).float().cpu().numpy()[:, 0]
                for (c_off, r_off), p in zip(batch_pos, probs):
                    prob_sum[r_off:r_off + size, c_off:c_off + size] += p * weight
                    weight_sum[r_off:r_off + size, c_off:c_off + size] += weight
                done += len(batch_imgs)
                print(f"\r[predict] janelas processadas: {done}/{len(windows)}",
                      end="", flush=True)
                batch_imgs.clear()
                batch_pos.clear()

            for c_off, r_off in windows:
                window = Window(c_off, r_off, size, size)
                raw = src.read(bands, window=window)              # (C,H,W) uint8
                if not raw.any():                                 # pula áreas pretas (sem dado)
                    continue
                img = raw.astype(np.float32) / 255.0
                img = np.transpose(img, (1, 2, 0))                # (H,W,C)
                img = (img - mean) / std
                batch_imgs.append(np.transpose(img, (2, 0, 1)).copy())
                batch_pos.append((c_off, r_off))
                if len(batch_imgs) >= icfg["batch_size"]:
                    flush()
            flush()
            print()  # quebra de linha após a barra de progresso

        # Grava a máscara de probabilidade em blocos de linhas (sem materializar
        # a imagem inteira em memória).
        ensure_dir(Path(output_path).parent)
        profile.update(driver="GTiff", count=1, dtype="float32", nodata=None,
                       compress="DEFLATE", tiled=True, blockxsize=512,
                       blockysize=512, BIGTIFF="IF_SAFER")
        block = 2048
        with rasterio.open(output_path, "w", **profile) as dst:
            for r0 in range(0, height, block):
                r1 = min(r0 + block, height)
                ps = np.asarray(prob_sum[r0:r1])
                ws = np.asarray(weight_sum[r0:r1])
                out = np.where(ws > 0, ps / np.maximum(ws, 1e-6), 0.0).astype(np.float32)
                dst.write(out, 1, window=Window(0, r0, width, r1 - r0))

        del prob_sum, weight_sum
        print(f"[predict] Máscara de probabilidade salva: {output_path}")
        return output_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_from_config(config_path: str, city: str | None,
                     image: str | None, output: str | None) -> None:
    cfg = load_config(config_path)
    if image and output:
        predict_image(image, output, cfg)
        return

    processed_dir = Path(cfg["data_prep"]["processed_dir"])
    out_dir = ensure_dir(cfg["inference"]["out_mask_dir"])
    targets = [c for c in cfg["cities"] if city is None or c["name"] == city]
    if not targets:
        raise SystemExit(f"Cidade '{city}' não encontrada na config.")
    for c in targets:
        img = processed_dir / f"{c['name']}.tif"
        out = out_dir / f"{c['name']}_prob.tif"
        predict_image(str(img), str(out), cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inferência sliding-window de edificações.")
    ap.add_argument("--config", required=True, help="Config YAML.")
    ap.add_argument("--city", help="Nome da cidade da config (default: todas).")
    ap.add_argument("--image", help="Raster avulso de entrada.")
    ap.add_argument("--output", help="Máscara de probabilidade de saída (avulso).")
    args = ap.parse_args()
    _run_from_config(args.config, args.city, args.image, args.output)


if __name__ == "__main__":
    main()
