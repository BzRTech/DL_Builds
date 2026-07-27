"""Quadras a partir da rede viária: quadra = área urbana − ruas (bufferizadas).

Abordagem geométrica (sem deep learning): bufferiza as linhas de logradouro pela
metade da largura da via e subtrai da área de interesse; o que sobra são os
blocos (quadras), recortados pelas ruas reais.

Uso:
    python -m src.cadastral.quadras_from_roads --config configs/cadastral.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union

from src.utils import ensure_dir, load_config


def build_quadras(roads_path: str, boundary_path: str | None, crs: str,
                  street_buffer_m: float, min_area_m2: float) -> gpd.GeoDataFrame:
    roads = gpd.read_file(roads_path).to_crs(crs)
    streets = unary_union(roads.buffer(street_buffer_m))

    if boundary_path and Path(boundary_path).exists():
        aoi = unary_union(gpd.read_file(boundary_path).to_crs(crs).geometry)
    else:
        aoi = unary_union(roads.geometry).convex_hull  # fallback: envelope das ruas

    blocks = aoi.difference(streets)
    geoms = [g for g in getattr(blocks, "geoms", [blocks])
             if isinstance(g, Polygon) and g.area >= min_area_m2]
    gdf = gpd.GeoDataFrame(
        {"quadra_id": range(1, len(geoms) + 1),
         "area_m2": [round(g.area, 1) for g in geoms]},
        geometry=geoms, crs=crs)
    print(f"[quadras_vias] {roads_path}: {len(gdf)} quadras (buffer {street_buffer_m} m)")
    return gdf


def _run(config_path: str) -> None:
    cfg = load_config(config_path)
    p = cfg["params"]
    for city in cfg["cities"]:
        gdf = build_quadras(city["roads"], city.get("boundary"), cfg["crs"],
                            p["street_buffer_m"], p["min_quadra_area_m2"])
        out = ensure_dir(Path(cfg["out_dir"]) / city["name"]) / "quadras_vias.gpkg"
        gdf.to_file(out, driver="GPKG", layer="quadras")
        print(f"[quadras_vias] -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Quadras a partir da rede viária.")
    ap.add_argument("--config", required=True, help="Config YAML.")
    args = ap.parse_args()
    _run(args.config)


if __name__ == "__main__":
    main()
