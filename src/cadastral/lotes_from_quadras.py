"""Lotes a partir das quadras: subdivide cada quadra por proximidade às
edificações (diagrama de Voronoi das construções, recortado pela quadra).

Premissa: ~1 lote por edificação. Onde vale (maioria das casas) fica limpo;
lotes vagos são absorvidos pelo vizinho e vários prédios num lote viram lotes
separados. É uma reconstrução aproximada (não os limites cadastrais legais).

Requer as quadras já geradas (quadras_from_roads) e uma camada de edificações.

Uso:
    python -m src.cadastral.lotes_from_quadras --config configs/cadastral.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import voronoi_diagram

from src.utils import ensure_dir, load_config


def _split_quadra(quadra: Polygon, seeds: list) -> list[Polygon]:
    """Divide uma quadra em lotes por Voronoi das sementes (centroides de prédios)."""
    if len(seeds) <= 1:
        return [quadra]
    try:
        cells = voronoi_diagram(MultiPoint(seeds), envelope=quadra)
    except Exception:  # noqa: BLE001  (colinear/duplicado etc.)
        return [quadra]
    lots = []
    for cell in cells.geoms:
        part = cell.intersection(quadra)
        for g in getattr(part, "geoms", [part]):
            if isinstance(g, Polygon) and not g.is_empty:
                lots.append(g)
    return lots or [quadra]


def build_lotes(quadras_path: str, buildings_path: str, crs: str,
                min_area_m2: float) -> gpd.GeoDataFrame:
    quadras = gpd.read_file(quadras_path).to_crs(crs)
    builds = gpd.read_file(buildings_path).to_crs(crs)
    b_centroids = builds.geometry.centroid
    sindex = b_centroids.sindex

    lots = []
    for quadra in quadras.geometry:
        idx = list(sindex.query(quadra, predicate="contains"))
        seeds = [(p.x, p.y) for p in b_centroids.iloc[idx]] if idx else []
        lots.extend(_split_quadra(quadra, seeds))

    lots = [g for g in lots if g.area >= min_area_m2]
    gdf = gpd.GeoDataFrame(
        {"lote_id": range(1, len(lots) + 1),
         "area_m2": [round(g.area, 1) for g in lots]},
        geometry=lots, crs=crs)
    print(f"[lotes_vias] {len(gdf)} lotes de {len(quadras)} quadras "
          f"({len(builds)} edificações-semente)")
    return gdf


def _run(config_path: str) -> None:
    cfg = load_config(config_path)
    p = cfg["params"]
    for city in cfg["cities"]:
        quadras_path = Path(cfg["out_dir"]) / city["name"] / "quadras_vias.gpkg"
        if not quadras_path.exists():
            raise SystemExit(f"Rode quadras_from_roads antes: falta {quadras_path}")
        gdf = build_lotes(str(quadras_path), city["buildings"], cfg["crs"],
                          p["min_lote_area_m2"])
        out = ensure_dir(Path(cfg["out_dir"]) / city["name"]) / "lotes_vias.gpkg"
        gdf.to_file(out, driver="GPKG", layer="lotes")
        print(f"[lotes_vias] -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Lotes por subdivisão das quadras (Voronoi).")
    ap.add_argument("--config", required=True, help="Config YAML.")
    args = ap.parse_args()
    _run(args.config)


if __name__ == "__main__":
    main()
