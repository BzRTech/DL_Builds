"""Atribui cada tile a train/val/test e calcula estatísticas de normalização.

Regras:
  - Cidades com split de cidade == 'test'  -> todos os tiles viram 'test'
    (isto é o teste leave-one-city-out: generalização para uma cidade nova).
  - Cidades com split de cidade == 'train' -> split ESPACIAL em train/val:
    reserva-se a faixa de colunas mais à direita (val_fraction) para validação,
    evitando vazamento entre tiles vizinhos sobrepostos.

Também calcula média/desvio por banda sobre os tiles de TREINO e salva em
data/tiles/norm_stats.json (usado pelo dataset no treino/inferência).

Uso:
    python -m src.data_prep.split --config configs/buildings.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio

from src.utils import load_config, set_seed


def _assign_splits(rows: list[dict], val_fraction: float) -> None:
    """Preenche o campo 'split' de cada linha do manifest (in-place)."""
    by_city: dict[str, list[dict]] = {}
    for r in rows:
        by_city.setdefault(r["city"], []).append(r)

    for city, city_rows in by_city.items():
        city_split = city_rows[0]["city_split"]
        if city_split == "test":
            for r in city_rows:
                r["split"] = "test"
            continue

        # Split espacial: corta pela coordenada de coluna (eixo x em pixels).
        col_offs = sorted({int(r["col_off"]) for r in city_rows})
        n_val_cols = max(1, int(round(len(col_offs) * val_fraction)))
        val_cols = set(col_offs[-n_val_cols:])  # faixa mais à direita p/ validação
        for r in city_rows:
            r["split"] = "val" if int(r["col_off"]) in val_cols else "train"


def _compute_norm_stats(rows: list[dict], bands: int) -> dict:
    """Média/desvio por banda (escala 0-1) sobre os tiles de treino."""
    train_rows = [r for r in rows if r["split"] == "train"]
    if not train_rows:
        raise RuntimeError("Nenhum tile de treino para calcular normalização.")

    n_pixels = 0
    channel_sum = np.zeros(bands, dtype=np.float64)
    channel_sq_sum = np.zeros(bands, dtype=np.float64)
    for r in train_rows:
        with rasterio.open(r["image"]) as src:
            arr = src.read().astype(np.float64) / 255.0  # (C, H, W)
        c = arr.shape[0]
        flat = arr.reshape(c, -1)
        channel_sum += flat.sum(axis=1)
        channel_sq_sum += (flat ** 2).sum(axis=1)
        n_pixels += flat.shape[1]

    mean = channel_sum / n_pixels
    var = channel_sq_sum / n_pixels - mean ** 2
    std = np.sqrt(np.clip(var, 1e-8, None))
    return {"mean": mean.tolist(), "std": std.tolist()}


def _run_from_config(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    tiles_dir = Path(cfg["data_prep"]["tiles_dir"])
    manifest_path = tiles_dir / "manifest.csv"

    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    _assign_splits(rows, cfg["data_prep"]["val_fraction"])

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    print(f"[split] Distribuição: {counts}")

    stats = _compute_norm_stats(rows, bands=len(cfg["data_prep"]["bands"]))
    stats_path = tiles_dir / "norm_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[split] Normalização (treino): mean={stats['mean']} std={stats['std']}")
    print(f"[split] Salvo em {stats_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Atribui splits e calcula normalização.")
    ap.add_argument("--config", required=True, help="Config YAML.")
    args = ap.parse_args()
    _run_from_config(args.config)


if __name__ == "__main__":
    main()
