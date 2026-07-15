"""Rasteriza os shapefiles de edificações em máscaras binárias alinhadas ao raster.

A máscara resultante tem exatamente o mesmo CRS, resolução, transform e dimensões
do COG da cidade, para que imagem e máscara possam ser recortadas em pares.

Uso:
    python -m src.data_prep.rasterize_labels --config configs/buildings.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from src.utils import ensure_dir, load_config


def rasterize_city(image_cog: str, labels_path: str, out_mask: str) -> str:
    """Gera uma máscara binária uint8 (1=edificação, 0=fundo) alinhada ao COG."""
    with rasterio.open(image_cog) as src:
        transform = src.transform
        out_shape = (src.height, src.width)
        raster_crs = src.crs
        profile = src.profile

    gdf = gpd.read_file(labels_path)
    if gdf.empty:
        raise ValueError(f"Shapefile de rótulos vazio: {labels_path}")

    # Reprojeta os polígonos para o CRS do raster, se necessário.
    if gdf.crs is None:
        raise ValueError(
            f"'{labels_path}' não tem CRS definido. Defina o CRS antes de rasterizar."
        )
    if raster_crs is not None and gdf.crs != raster_crs:
        print(f"[rasterize] Reprojetando rótulos {gdf.crs} -> {raster_crs}")
        gdf = gdf.to_crs(raster_crs)

    shapes = ((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty)
    mask = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )

    profile.update(count=1, dtype="uint8", nodata=0, compress="DEFLATE")
    ensure_dir(Path(out_mask).parent)
    with rasterio.open(out_mask, "w", **profile) as dst:
        dst.write(mask, 1)

    cov = float(np.count_nonzero(mask)) / mask.size
    print(f"[rasterize] {out_mask}  (cobertura de edificação: {cov:.2%})")
    return out_mask


def _run_from_config(config_path: str) -> None:
    cfg = load_config(config_path)
    processed_dir = Path(cfg["data_prep"]["processed_dir"])
    for city in cfg["cities"]:
        cog = processed_dir / f"{city['name']}.tif"
        out_mask = processed_dir / f"{city['name']}_mask.tif"
        rasterize_city(str(cog), city["labels"], str(out_mask))
    print("[rasterize] Concluído.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rasteriza rótulos de edificações em máscaras.")
    ap.add_argument("--config", required=True, help="Config YAML.")
    args = ap.parse_args()
    _run_from_config(args.config)


if __name__ == "__main__":
    main()
