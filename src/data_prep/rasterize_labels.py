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
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform

from src.utils import ensure_dir, load_config, target_name


def rasterize_city(image_cog: str, labels_path: str, out_mask: str,
                   block: int = 2048) -> str:
    """Gera uma máscara binária uint8 (1=alvo, 0=fundo) alinhada ao COG.

    Rasteriza em blocos de linhas (com filtro espacial por bloco), de modo que a
    memória fica limitada mesmo em ortofotos gigantes (dezenas de gigapixels).
    """
    with rasterio.open(image_cog) as src:
        transform = src.transform
        width, height = src.width, src.height
        raster_crs = src.crs
        profile = src.profile.copy()

    gdf = gpd.read_file(labels_path)
    if gdf.empty:
        raise ValueError(f"Shapefile de rótulos vazio: {labels_path}")
    if gdf.crs is None:
        raise ValueError(
            f"'{labels_path}' não tem CRS definido. Defina o CRS antes de rasterizar."
        )
    if raster_crs is not None and gdf.crs != raster_crs:
        print(f"[rasterize] Reprojetando rótulos {gdf.crs} -> {raster_crs}")
        gdf = gdf.to_crs(raster_crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    sindex = gdf.sindex  # índice espacial para filtrar por bloco

    profile.update(count=1, dtype="uint8", nodata=0, compress="DEFLATE",
                   driver="GTiff", tiled=True, blockxsize=512, blockysize=512,
                   BIGTIFF="IF_SAFER")
    ensure_dir(Path(out_mask).parent)

    total_pos = 0
    with rasterio.open(out_mask, "w", **profile) as dst:
        for r0 in range(0, height, block):
            r1 = min(r0 + block, height)
            win = Window(0, r0, width, r1 - r0)
            win_tf = window_transform(win, transform)
            left, bottom, right, top = window_bounds(win, transform)
            idx = list(sindex.intersection((left, bottom, right, top)))
            if idx:
                sub = gdf.iloc[idx]
                block_mask = rasterize(
                    ((g, 1) for g in sub.geometry),
                    out_shape=(r1 - r0, width),
                    transform=win_tf,
                    fill=0,
                    dtype="uint8",
                    all_touched=False,
                )
            else:
                block_mask = None
            if block_mask is not None:
                dst.write(block_mask, 1, window=win)
                total_pos += int(block_mask.sum())

    cov = total_pos / float(width * height)
    print(f"[rasterize] {out_mask}  (cobertura do alvo: {cov:.2%})")
    return out_mask


def _run_from_config(config_path: str) -> None:
    cfg = load_config(config_path)
    processed_dir = Path(cfg["data_prep"]["processed_dir"])
    target = target_name(cfg)
    for city in cfg["cities"]:
        cog = processed_dir / f"{city['name']}.tif"
        out_mask = processed_dir / f"{city['name']}_{target}_mask.tif"
        rasterize_city(str(cog), city["labels"], str(out_mask))
    print("[rasterize] Concluído.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rasteriza rótulos de edificações em máscaras.")
    ap.add_argument("--config", required=True, help="Config YAML.")
    args = ap.parse_args()
    _run_from_config(args.config)


if __name__ == "__main__":
    main()
