"""Classifica o pavimento de cada trecho de logradouro com o modelo treinado.

Para cada trecho: amostra patches ao longo da linha, classifica cada um e faz
votação → classe do trecho (+ confiança). Salva o shapefile de logradouros com
as colunas 'PAV_PRED' (classe prevista) e 'PAV_CONF' (fração de votos).

Uso:
    python -m src.road_surface.predict --config configs/road_surface.yaml --city Tabira
    python -m src.road_surface.predict --config configs/road_surface.yaml \
        --roads data/raw/NovaCidade/logradouros.shp \
        --image data/processed/NovaCidade.tif --out outputs/NovaCidade_vias.gpkg
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from shapely.geometry import LineString, MultiLineString

from src.road_surface.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.road_surface.train import build_classifier
from src.utils import ensure_dir, load_config


def _iter_lines(geom):
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        yield from geom.geoms


def _load_model(checkpoint: str, device):
    ckpt = torch.load(checkpoint, map_location=device)
    model = build_classifier(ckpt["arch"], len(ckpt["classes"]), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["classes"], ckpt["input_size"]


def _segment_patches(line, src, transform, patch_px, bands, size, min_valid_frac, step_m):
    width, height = src.width, src.height
    out = []
    for ln in _iter_lines(line):
        if ln is None or ln.is_empty:
            continue
        dists = list(np.arange(0, max(ln.length, 1e-6), step_m)) or [ln.length / 2]
        for d in dists:
            pt = ln.interpolate(float(d))
            r, c = rasterio.transform.rowcol(transform, pt.x, pt.y)
            r0, c0 = int(r) - patch_px // 2, int(c) - patch_px // 2
            if r0 < 0 or c0 < 0 or r0 + patch_px > height or c0 + patch_px > width:
                continue
            arr = src.read(bands, window=Window(c0, r0, patch_px, patch_px))
            if arr.shape[1] != patch_px or arr.shape[2] != patch_px:
                continue
            img = np.transpose(arr, (1, 2, 0))
            if img.any(axis=2).mean() < min_valid_frac:
                continue
            out.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
    return out


@torch.no_grad()
def classify_roads(image_cog: str, roads_path: str, out_path: str, cfg: dict) -> str:
    dp = cfg["data_prep"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, size = _load_model(cfg["predict"]["checkpoint"], device)

    gdf = gpd.read_file(roads_path)
    with rasterio.open(image_cog) as src:
        transform = src.transform
        if gdf.crs is not None and src.crs is not None and gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        patch_px = max(8, int(round(dp["patch_size_m"] / abs(transform.a))))

        preds, confs = [], []
        for _, row in gdf.iterrows():
            patches = _segment_patches(row.geometry, src, transform, patch_px,
                                       dp["bands"], size, dp["min_valid_frac"], dp["step_m"])
            if not patches:
                preds.append(None); confs.append(0.0); continue
            batch = np.stack(patches).astype(np.float32) / 255.0
            batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
            batch = torch.from_numpy(np.transpose(batch, (0, 3, 1, 2)).copy()).to(device)
            logits = model(batch)
            idxs = logits.argmax(1).cpu().numpy()
            vote, n = Counter(idxs).most_common(1)[0]
            preds.append(classes[vote]); confs.append(n / len(idxs))

    gdf["PAV_PRED"] = preds
    gdf["PAV_CONF"] = [round(c, 3) for c in confs]
    ensure_dir(Path(out_path).parent)
    gdf.to_file(out_path, driver="GPKG", layer="logradouros")
    n_ok = sum(p is not None for p in preds)
    print(f"[road_predict] {n_ok}/{len(gdf)} trechos classificados -> {out_path}")
    dist = Counter(p for p in preds if p is not None)
    print(f"[road_predict] distribuição prevista: {dict(dist)}")
    return out_path


def _run(config_path, city, roads, image, out):
    cfg = load_config(config_path)
    if roads and image and out:
        classify_roads(image, roads, out, cfg)
        return
    targets = [c for c in cfg["cities"] if city is None or c["name"] == city]
    if not targets:
        raise SystemExit(f"Cidade '{city}' não encontrada na config.")
    for c in targets:
        out_v = cfg["predict"]["out_vector"] if len(targets) == 1 else \
            str(Path(cfg["predict"]["out_vector"]).with_name(
                f"logradouros_{c['name']}.gpkg"))
        classify_roads(c["image"], c["roads"], out_v, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Classifica pavimento dos logradouros.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--city", help="Cidade da config (default: todas).")
    ap.add_argument("--roads", help="Shapefile de logradouros avulso.")
    ap.add_argument("--image", help="COG avulso.")
    ap.add_argument("--out", help="GeoPackage de saída avulso.")
    args = ap.parse_args()
    _run(args.config, args.city, args.roads, args.image, args.out)


if __name__ == "__main__":
    main()
