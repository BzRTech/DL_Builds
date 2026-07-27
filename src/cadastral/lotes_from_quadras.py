"""Lotes a partir das quadras + vínculo com as edificações.

Método "rect" (padrão): corta a quadra em tiras retangulares perpendiculares à
rua, colocando os cortes NO VÃO entre as edificações (usando o contorno) — assim
cada edificação fica INTEIRA dentro de um lote. 2 fileiras (frente/fundo). Lotes
são disjuntos (não se sobrepõem). Cada edificação recebe o lote_id que a contém
e cada lote a contagem de edificações.

Método "voronoi": partição por proximidade (células irregulares) — alternativo.

Saídas por cidade:
  outputs/<cidade>/lotes_vias.gpkg     (lotes; coluna n_edif)
  outputs/<cidade>/cadastro.gpkg       (camadas: lotes + edificacoes com lote_id)

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
from shapely.ops import unary_union, voronoi_diagram

from src.utils import ensure_dir, load_config


def _polys(geom):
    for g in getattr(geom, "geoms", [geom]):
        if isinstance(g, Polygon) and not g.is_empty:
            yield g


def _mrr_angle(poly: Polygon) -> float:
    cs = list(poly.minimum_rotated_rectangle.exterior.coords)[:5]
    best = max(((cs[i], cs[i + 1]) for i in range(len(cs) - 1)),
               key=lambda e: (e[1][0] - e[0][0]) ** 2 + (e[1][1] - e[0][1]) ** 2)
    return math.degrees(math.atan2(best[1][1] - best[0][1], best[1][0] - best[0][0]))


def dedupe_overlaps(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove SOBREPOSIÇÕES entre edificações recortando a parte sobreposta do
    polígono menor (mantém o maior). Não funde prédios que apenas se encostam."""
    geoms = list(gdf.geometry)
    order = sorted(range(len(geoms)), key=lambda i: geoms[i].area, reverse=True)
    sindex = gpd.GeoSeries(geoms, crs=gdf.crs).sindex
    kept, taken = [], set()
    for i in order:
        g = geoms[i]
        for j in sindex.query(g, predicate="intersects"):
            if j != i and j in taken:                    # já colocado (maior/igual)
                g = g.difference(geoms[j])
                if g.is_empty:
                    break
        for p in _polys(g):
            kept.append(p)
        taken.add(i)
    return gpd.GeoDataFrame({"edif_id": range(1, len(kept) + 1)}, geometry=kept, crs=gdf.crs)


def _split_voronoi(quadra: Polygon, seeds: list) -> list[Polygon]:
    if len(seeds) <= 1:
        return [quadra]
    try:
        cells = voronoi_diagram(MultiPoint(seeds), envelope=quadra)
    except Exception:  # noqa: BLE001
        return [quadra]
    return [g for cell in cells.geoms for g in _polys(cell.intersection(quadra))] or [quadra]


def _split_rects(quadra: Polygon, builds: list, min_area: float,
                 min_row_depth: float, min_lot_width: float) -> list[Polygon]:
    """Tiras retangulares com cortes NO VÃO entre as edificações (não as cruzam)."""
    if len(builds) < 2:
        return [quadra]
    ang = _mrr_angle(quadra)
    origin = quadra.centroid
    qr = rotate(quadra, -ang, origin=origin)
    br = [rotate(b, -ang, origin=origin) for b in builds]          # prédios alinhados

    minx, miny, maxx, maxy = qr.bounds
    ymid = (miny + maxy) / 2
    bands = ([(miny, ymid), (ymid, maxy)] if (maxy - miny) >= 2 * min_row_depth
             else [(miny, maxy)])

    lots_rot = []
    for y0, y1 in bands:
        band = box(minx, y0, maxx, y1).intersection(qr)
        if band.is_empty:
            continue
        # prédios cujo centro cai nesta fileira, ordenados pela extensão em x
        exts = sorted((b.bounds[0], b.bounds[2]) for b in br if y0 <= b.centroid.y < y1)
        bounds = [minx]
        for i in range(len(exts) - 1):
            # corte na DIVISA entre duas casas consecutivas (fim de uma / início da
            # próxima) -> 1 lote fino por edificação, como no cadastro oficial.
            c = (exts[i][1] + exts[i + 1][0]) / 2
            if c - bounds[-1] > max(min_lot_width, 0.05) and maxx - c > max(min_lot_width, 0.05):
                bounds.append(c)
        bounds.append(maxx)
        for i in range(len(bounds) - 1):
            lots_rot.extend(_polys(box(bounds[i], y0, bounds[i + 1], y1).intersection(band)))

    lots_rot = [g for g in lots_rot if g.area >= min_area]
    return [rotate(g, ang, origin=origin) for g in lots_rot] or [quadra]


def build_cadastro(quadras_path: str, buildings_path: str, crs: str, min_area_m2: float,
                   method: str, min_row_depth: float, min_lot_width: float,
                   clean: bool):
    quadras = gpd.read_file(quadras_path).to_crs(crs)
    builds = gpd.read_file(buildings_path).to_crs(crs)
    builds = builds[builds.geometry.notna() & ~builds.geometry.is_empty]
    if clean:
        n0 = len(builds)
        builds = dedupe_overlaps(builds)
        print(f"[cadastro] limpeza de edificações: {n0} -> {len(builds)} (sem sobreposição)")
    builds = builds.reset_index(drop=True)
    b_geom = builds.geometry
    sindex = b_geom.sindex

    lots = []
    for quadra in quadras.geometry:
        idx = [i for i in sindex.query(quadra, predicate="intersects")
               if b_geom.iloc[i].representative_point().within(quadra)]
        binq = [b_geom.iloc[i] for i in idx]
        if method == "voronoi":
            lots.extend(_split_voronoi(quadra, [(g.centroid.x, g.centroid.y) for g in binq]))
        else:
            lots.extend(_split_rects(quadra, binq, min_area_m2, min_row_depth, min_lot_width))
    lots = [g for g in lots if g.area >= min_area_m2]

    lotes = gpd.GeoDataFrame({"lote_id": range(1, len(lots) + 1),
                              "area_m2": [round(g.area, 1) for g in lots]},
                             geometry=lots, crs=crs)

    # vincula cada edificação ao lote que a contém (ponto representativo dentro)
    bpts = gpd.GeoDataFrame({"edif_i": range(len(builds))},
                            geometry=list(b_geom.representative_point()), crs=crs)
    j = gpd.sjoin(bpts, lotes[["lote_id", "geometry"]], how="left", predicate="within")
    j = j.drop_duplicates("edif_i").set_index("edif_i")
    builds_out = builds.copy()
    builds_out["lote_id"] = j["lote_id"].reindex(range(len(builds))).values
    counts = j["lote_id"].value_counts()
    lotes["n_edif"] = lotes["lote_id"].map(counts).fillna(0).astype(int)

    print(f"[cadastro] {len(lotes)} lotes | {len(builds_out)} edificações "
          f"({int((builds_out['lote_id'].notna()).sum())} vinculadas a um lote)")
    return lotes, builds_out


def _run(config_path: str) -> None:
    cfg = load_config(config_path)
    p = cfg["params"]
    method = p.get("method", "rect")
    min_row_depth = float(p.get("lot_min_depth_m", 8.0))
    min_lot_width = float(p.get("min_lot_width_m", 0.0))
    clean = bool(p.get("clean_buildings", False))
    for city in cfg["cities"]:
        quadras_path = Path(cfg["out_dir"]) / city["name"] / "quadras_vias.gpkg"
        if not quadras_path.exists():
            raise SystemExit(f"Rode quadras_from_roads antes: falta {quadras_path}")
        lotes, builds = build_cadastro(str(quadras_path), city["buildings"], cfg["crs"],
                                       p["min_lote_area_m2"], method, min_row_depth,
                                       min_lot_width, clean)
        out = ensure_dir(Path(cfg["out_dir"]) / city["name"])
        lotes.to_file(out / "lotes_vias.gpkg", driver="GPKG", layer="lotes")
        lotes.to_file(out / "cadastro.gpkg", driver="GPKG", layer="lotes")
        builds.to_file(out / "cadastro.gpkg", driver="GPKG", layer="edificacoes")
        print(f"[cadastro] -> {out/'lotes_vias.gpkg'} e {out/'cadastro.gpkg'} "
              f"(camadas lotes + edificacoes)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Lotes + vínculo com edificações.")
    ap.add_argument("--config", required=True, help="Config YAML.")
    args = ap.parse_args()
    _run(args.config)


if __name__ == "__main__":
    main()
