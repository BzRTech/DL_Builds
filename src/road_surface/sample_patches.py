"""Amostra patches de imagem ao longo dos trechos de logradouro, rotulados pela
classe de pavimento (campo STATUS). Gera o dataset para o classificador.

Para cada trecho (LineString), caminha ao longo da linha a cada `step_m` metros e
recorta um patch de `patch_size_m` (convertido para pixels pela resolução do
raster e redimensionado para um tamanho fixo). O split treino/validação é feito
por TRECHO (patches do mesmo trecho ficam no mesmo split, sem vazamento).

Saída em data/road_patches/: patches.npy, labels.npy, splits.npy, meta.json.

Uso:
    python -m src.road_surface.sample_patches --config configs/road_surface.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import LineString, MultiLineString

from src.utils import ensure_dir, load_config

INPUT_SIZE = 96  # tamanho fixo do patch para o modelo (px)


def _iter_lines(geom):
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        yield from geom.geoms


def sample_city(image_cog: str, roads_path: str, class_field: str,
                class_to_idx: dict, patch_size_m: float, step_m: float,
                bands: list[int], min_valid_frac: float, val_fraction: float,
                seed: int):
    """Retorna listas (patches, labels, splits) para uma cidade."""
    gdf = gpd.read_file(roads_path)
    with rasterio.open(image_cog) as src:
        transform = src.transform
        width, height = src.width, src.height
        raster_crs = src.crs
        if gdf.crs is not None and raster_crs is not None and gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)
        pixel_size = abs(transform.a)
        patch_px = max(8, int(round(patch_size_m / pixel_size)))
        rng = random.Random(seed)

        patches, labels, splits = [], [], []
        per_class = {c: 0 for c in class_to_idx}
        for _, row in gdf.iterrows():
            cls = row.get(class_field)
            if cls not in class_to_idx:
                continue
            split = "val" if rng.random() < val_fraction else "train"
            label = class_to_idx[cls]
            for line in _iter_lines(row.geometry):
                if line is None or line.is_empty:
                    continue
                length = line.length
                dists = list(np.arange(0, max(length, 1e-6), step_m)) or [length / 2]
                for d in dists:
                    pt = line.interpolate(float(d))
                    r, c = rasterio.transform.rowcol(transform, pt.x, pt.y)
                    r0, c0 = int(r) - patch_px // 2, int(c) - patch_px // 2
                    if r0 < 0 or c0 < 0 or r0 + patch_px > height or c0 + patch_px > width:
                        continue
                    win = Window(c0, r0, patch_px, patch_px)
                    arr = src.read(bands, window=win)              # (C, h, w)
                    if arr.shape[1] != patch_px or arr.shape[2] != patch_px:
                        continue
                    img = np.transpose(arr, (1, 2, 0))             # (h, w, C)
                    if (img.any(axis=2).mean()) < min_valid_frac:  # muito preto
                        continue
                    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE),
                                     interpolation=cv2.INTER_AREA)
                    patches.append(img.astype(np.uint8))
                    labels.append(label)
                    splits.append(split)
                    per_class[cls] += 1
        print(f"[sample] {roads_path}: {len(patches)} patches | patch={patch_px}px | "
              f"por classe: {per_class}")
    return patches, labels, splits


def _run_from_config(config_path: str) -> None:
    cfg = load_config(config_path)
    dp = cfg["data_prep"]
    classes = cfg["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    all_p, all_l, all_s = [], [], []
    for city in cfg["cities"]:
        p, l, s = sample_city(
            image_cog=city["image"],
            roads_path=city["roads"],
            class_field=cfg["class_field"],
            class_to_idx=class_to_idx,
            patch_size_m=dp["patch_size_m"],
            step_m=dp["step_m"],
            bands=dp["bands"],
            min_valid_frac=dp["min_valid_frac"],
            val_fraction=dp["val_fraction"],
            seed=cfg["project"]["seed"],
        )
        all_p.extend(p); all_l.extend(l); all_s.extend(s)

    if not all_p:
        raise SystemExit("Nenhum patch amostrado — verifique class_field/classes e os dados.")

    out = ensure_dir(dp["patches_dir"])
    np.save(out / "patches.npy", np.stack(all_p))
    np.save(out / "labels.npy", np.array(all_l, dtype=np.int64))
    np.save(out / "splits.npy", np.array(all_s))
    counts = {c: int(np.sum(np.array(all_l) == i)) for i, c in enumerate(classes)}
    meta = {"classes": classes, "input_size": INPUT_SIZE, "counts": counts,
            "n_total": len(all_p),
            "n_train": int(np.sum(np.array(all_s) == "train")),
            "n_val": int(np.sum(np.array(all_s) == "val"))}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    print(f"[sample] Total: {meta['n_total']} patches "
          f"(train={meta['n_train']}, val={meta['n_val']}) por classe: {counts}")
    print(f"[sample] Salvo em {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Amostra patches de pavimento das vias.")
    ap.add_argument("--config", required=True, help="Config YAML.")
    args = ap.parse_args()
    _run_from_config(args.config)


if __name__ == "__main__":
    main()
