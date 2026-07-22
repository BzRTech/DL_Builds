"""Recorta cada COG + máscara em tiles pareados (imagem/máscara) para treino.

Gera:
  data/tiles/images/<cidade>_<row>_<col>.tif
  data/tiles/masks/<cidade>_<row>_<col>.tif
  data/tiles/manifest.csv   (registro de cada tile: cidade, split, posição, cobertura)

O manifest guarda a posição geográfica (row/col em pixels na origem da cidade) para
permitir o split espacial em split.py.

Uso:
    python -m src.data_prep.tile --config configs/buildings.yaml
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from src.utils import ensure_dir, load_config, target_name


def _iter_windows(width: int, height: int, size: int, stride: int):
    """Gera janelas (col_off, row_off) cobrindo toda a imagem, incluindo bordas."""
    cols = list(range(0, max(width - size, 0) + 1, stride))
    rows = list(range(0, max(height - size, 0) + 1, stride))
    if cols[-1] != width - size and width > size:
        cols.append(width - size)
    if rows[-1] != height - size and height > size:
        rows.append(height - size)
    for r in rows:
        for c in cols:
            yield c, r


def tile_city(
    name: str,
    image_cog: str,
    mask_tif: str,
    images_dir: Path,
    masks_dir: Path,
    bands: list[int],
    size: int,
    overlap: int,
    min_coverage: float,
) -> list[dict]:
    """Recorta uma cidade em tiles e retorna as linhas do manifest."""
    stride = size - overlap
    rows_manifest: list[dict] = []

    with rasterio.open(image_cog) as img_src, rasterio.open(mask_tif) as mask_src:
        assert img_src.width == mask_src.width and img_src.height == mask_src.height, (
            f"Imagem e máscara de '{name}' têm dimensões diferentes."
        )
        width, height = img_src.width, img_src.height

        for col_off, row_off in _iter_windows(width, height, size, stride):
            window = Window(col_off, row_off, size, size)
            img = img_src.read(bands, window=window)
            mask = mask_src.read(1, window=window)

            # Descarta tiles quase totalmente sem dados (todas as bandas zeradas).
            if not img.any():
                continue

            coverage = float(np.count_nonzero(mask)) / mask.size
            if coverage < min_coverage:
                continue

            tile_id = f"{name}_{row_off}_{col_off}"
            img_profile = img_src.profile.copy()
            img_transform = img_src.window_transform(window)
            img_profile.update(
                height=size, width=size, count=len(bands),
                transform=img_transform, compress="DEFLATE",
            )
            with rasterio.open(images_dir / f"{tile_id}.tif", "w", **img_profile) as dst:
                dst.write(img)

            mask_profile = mask_src.profile.copy()
            mask_profile.update(
                height=size, width=size, count=1, dtype="uint8",
                transform=img_transform, compress="DEFLATE",
            )
            with rasterio.open(masks_dir / f"{tile_id}.tif", "w", **mask_profile) as dst:
                dst.write(mask.astype("uint8"), 1)

            rows_manifest.append({
                "tile_id": tile_id,
                "city": name,
                "row_off": row_off,
                "col_off": col_off,
                "coverage": round(coverage, 4),
                "image": str(images_dir / f"{tile_id}.tif"),
                "mask": str(masks_dir / f"{tile_id}.tif"),
                "split": "",  # preenchido por split.py
            })

    print(f"[tile] {name}: {len(rows_manifest)} tiles gerados.")
    return rows_manifest


FIELDNAMES = ["tile_id", "city", "city_split", "split", "row_off",
              "col_off", "coverage", "image", "mask"]


def _run_from_config(config_path: str, only: str | None = None) -> None:
    cfg = load_config(config_path)
    dp = cfg["data_prep"]
    processed_dir = Path(dp["processed_dir"])
    tiles_dir = ensure_dir(dp["tiles_dir"])
    images_dir = ensure_dir(tiles_dir / "images")
    masks_dir = ensure_dir(tiles_dir / "masks")
    manifest_path = tiles_dir / "manifest.csv"

    target = target_name(cfg)
    city_split = {c["name"]: c.get("split", "train") for c in cfg["cities"]}
    cities = cfg["cities"]
    if only:
        cities = [c for c in cities if c["name"] == only]
        if not cities:
            raise SystemExit(f"Cidade '{only}' não está na config.")

    # Modo incremental (--only): preserva as linhas das outras cidades já tiladas.
    existing_rows: list[dict] = []
    if only and manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            existing_rows = [r for r in csv.DictReader(f) if r["city"] != only]

    new_rows: list[dict] = []
    for city in cities:
        cog = processed_dir / f"{city['name']}.tif"
        mask = processed_dir / f"{city['name']}_{target}_mask.tif"
        rows = tile_city(
            name=city["name"],
            image_cog=str(cog),
            mask_tif=str(mask),
            images_dir=images_dir,
            masks_dir=masks_dir,
            bands=dp["bands"],
            size=dp["tile_size"],
            overlap=dp["overlap"],
            min_coverage=dp["min_building_coverage"],
        )
        # Marca o split de cidade (train/test); split.py separa train->train/val.
        for r in rows:
            r["city_split"] = city_split[r["city"]]
        new_rows.extend(rows)

    all_rows = existing_rows + new_rows
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    extra = f" (+{len(existing_rows)} preservados)" if only else ""
    print(f"[tile] Manifest escrito: {manifest_path} ({len(all_rows)} tiles{extra}).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Recorta COGs+máscaras em tiles pareados.")
    ap.add_argument("--config", required=True, help="Config YAML.")
    ap.add_argument("--only", help="Tila apenas esta cidade e anexa ao manifest "
                                   "existente (não re-tila as demais).")
    args = ap.parse_args()
    _run_from_config(args.config, only=args.only)


if __name__ == "__main__":
    main()
