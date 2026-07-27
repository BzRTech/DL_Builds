"""Lotes a partir das quadras: subdivide cada quadra em lotes.

Dois métodos (config params.method):
  - "rect" (padrão): corta a quadra em TIRAS RETANGULARES perpendiculares à rua,
    nas posições das edificações — reproduz o padrão urbano típico (2 fileiras
    frente/fundo, lotes retangulares). É o que se parece com o cadastro real.
  - "voronoi": partição por proximidade às edificações (células irregulares).

Premissa: ~1 lote por edificação. Reconstrução aproximada (não os limites legais).
Requer as quadras já geradas (quadras_from_roads) e uma camada de edificações.

Uso:
    python -m src.cadastral.lotes_from_quadras --config configs/cadastral.yaml
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
from shapely.affinity import rotate
from shapely.geometry import MultiPoint, Polygon, box
from shapely.ops import voronoi_diagram

from src.utils import ensure_dir, load_config


def _polys(geom):
    for g in getattr(geom, "geoms", [geom]):
        if isinstance(g, Polygon) and not g.is_empty:
            yield g


def _mrr_angle(poly: Polygon) -> float:
    """Ângulo (graus) do lado mais longo do retângulo mínimo — orientação da quadra."""
    cs = list(poly.minimum_rotated_rectangle.exterior.coords)[:5]
    best = max(((cs[i], cs[i + 1]) for i in range(len(cs) - 1)),
               key=lambda e: (e[1][0] - e[0][0]) ** 2 + (e[1][1] - e[0][1]) ** 2)
    return math.degrees(math.atan2(best[1][1] - best[0][1], best[1][0] - best[0][0]))


def _split_voronoi(quadra: Polygon, seeds: list) -> list[Polygon]:
    if len(seeds) <= 1:
        return [quadra]
    try:
        cells = voronoi_diagram(MultiPoint(seeds), envelope=quadra)
    except Exception:  # noqa: BLE001
        return [quadra]
    lots = [g for cell in cells.geoms for g in _polys(cell.intersection(quadra))]
    return lots or [quadra]


def _row_bounds(xs: list, lo: float, hi: float, min_w: float) -> list[float]:
    """Posições de corte (incluindo lo e hi) nos meios entre prédios; se min_w>0,
    descarta cortes que gerariam tiras mais estreitas que min_w (funde os finos)."""
    cuts = [(xs[i] + xs[i + 1]) / 2 for i in range(len(xs) - 1)]
    if min_w <= 0:
        return [lo] + cuts + [hi]
    bounds = [lo]
    for c in cuts:
        if c - bounds[-1] >= min_w and hi - c >= min_w:
            bounds.append(c)
    bounds.append(hi)
    return bounds


def _split_rects(quadra: Polygon, seeds: list, min_area: float,
                 min_row_depth: float = 8.0, min_lot_width: float = 0.0) -> list[Polygon]:
    """Corta a quadra em tiras retangulares perpendiculares à via, nas posições
    das edificações; 2 fileiras (frente/fundo) se a quadra for funda o bastante."""
    if len(seeds) < 2:
        return [quadra]
    ang = _mrr_angle(quadra)
    origin = quadra.centroid
    qr = rotate(quadra, -ang, origin=origin)                       # alinha eixo longo em x
    pts = rotate(MultiPoint(seeds), -ang, origin=origin)
    spts = [(p.x, p.y) for p in pts.geoms]

    minx, miny, maxx, maxy = qr.bounds
    ymid = (miny + maxy) / 2
    depth = maxy - miny
    bands = [(miny, ymid), (ymid, maxy)] if depth >= 2 * min_row_depth else [(miny, maxy)]

    lots_rot = []
    for y0, y1 in bands:
        band = box(minx, y0, maxx, y1).intersection(qr)
        if band.is_empty:
            continue
        xs = sorted(x for x, y in spts if y0 <= y < y1)
        xb = _row_bounds(xs, minx, maxx, min_lot_width) if xs else [minx, maxx]
        for i in range(len(xb) - 1):
            lots_rot.extend(_polys(box(xb[i], y0, xb[i + 1], y1).intersection(band)))

    lots_rot = [g for g in lots_rot if g.area >= min_area]
    if not lots_rot:
        return [quadra]
    return [rotate(g, ang, origin=origin) for g in lots_rot]        # volta à orientação real


def build_lotes(quadras_path: str, buildings_path: str, crs: str, min_area_m2: float,
                method: str = "rect", min_row_depth: float = 8.0,
                min_lot_width: float = 0.0) -> gpd.GeoDataFrame:
    quadras = gpd.read_file(quadras_path).to_crs(crs)
    builds = gpd.read_file(buildings_path).to_crs(crs)
    b_centroids = builds.geometry.centroid
    sindex = b_centroids.sindex

    lots = []
    for quadra in quadras.geometry:
        idx = list(sindex.query(quadra, predicate="contains"))
        seeds = [(p.x, p.y) for p in b_centroids.iloc[idx]] if idx else []
        if method == "voronoi":
            lots.extend(_split_voronoi(quadra, seeds))
        else:
            lots.extend(_split_rects(quadra, seeds, min_area_m2, min_row_depth,
                                     min_lot_width))

    lots = [g for g in lots if g.area >= min_area_m2]
    gdf = gpd.GeoDataFrame(
        {"lote_id": range(1, len(lots) + 1),
         "area_m2": [round(g.area, 1) for g in lots]},
        geometry=lots, crs=crs)
    print(f"[lotes_vias] {len(gdf)} lotes de {len(quadras)} quadras "
          f"({len(builds)} edificações-semente) | método={method}")
    return gdf


def _run(config_path: str) -> None:
    cfg = load_config(config_path)
    p = cfg["params"]
    method = p.get("method", "rect")
    min_row_depth = float(p.get("lot_min_depth_m", 8.0))
    min_lot_width = float(p.get("min_lot_width_m", 0.0))
    for city in cfg["cities"]:
        quadras_path = Path(cfg["out_dir"]) / city["name"] / "quadras_vias.gpkg"
        if not quadras_path.exists():
            raise SystemExit(f"Rode quadras_from_roads antes: falta {quadras_path}")
        gdf = build_lotes(str(quadras_path), city["buildings"], cfg["crs"],
                          p["min_lote_area_m2"], method, min_row_depth, min_lot_width)
        out = ensure_dir(Path(cfg["out_dir"]) / city["name"]) / "lotes_vias.gpkg"
        gdf.to_file(out, driver="GPKG", layer="lotes")
        print(f"[lotes_vias] -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Lotes por subdivisão das quadras.")
    ap.add_argument("--config", required=True, help="Config YAML.")
    args = ap.parse_args()
    _run(args.config)


if __name__ == "__main__":
    main()
