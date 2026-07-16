"""Converte a máscara de probabilidade em polígonos vetoriais de edificações.

Passos: threshold -> poligonização -> filtro de área mínima -> preenchimento de
buracos pequenos -> simplificação -> (opcional) regularização ortogonal.
Exporta GeoPackage no CRS do raster.

Uso:
    python -m src.postprocess.vectorize --config configs/buildings.yaml --city cidade_b
    python -m src.postprocess.vectorize --config configs/buildings.yaml \
        --prob outputs/masks/cidade_b_prob.tif --output outputs/cidade_b.gpkg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes as rio_shapes
from rasterio.transform import rowcol
from rasterio.windows import Window, transform as window_transform
from scipy import ndimage
from shapely.geometry import shape
from shapely.geometry.polygon import Polygon
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from src.utils import ensure_dir, load_config


def _fill_small_holes(poly: Polygon, min_hole_area: float) -> Polygon:
    """Remove buracos internos com área menor que min_hole_area (unidades do CRS)."""
    if not poly.interiors:
        return poly
    keep = [ring for ring in poly.interiors if Polygon(ring).area >= min_hole_area]
    return Polygon(poly.exterior, keep)


def _orthogonalize(poly: Polygon, rect_ratio: float = 0.90) -> Polygon:
    """Regularização leve: se o polígono é quase retangular, substitui pelo seu
    retângulo mínimo rotacionado. Caso contrário, mantém a geometria simplificada."""
    mrr = poly.minimum_rotated_rectangle
    if mrr.area <= 0:
        return poly
    if poly.area / mrr.area >= rect_ratio:
        return mrr
    return poly


def _split_instances(poly: Polygon, transform, min_peak_distance_px: int,
                     pad: int = 2) -> list[Polygon]:
    """Divide um blob (que pode conter vários prédios encostados) em instâncias.

    Rasteriza o blob numa janela local, aplica watershed sobre a transformada de
    distância (um "pico" por prédio) e devolve um polígono por instância. Trabalha
    só na janela do blob, então é leve mesmo em ortofotos gigantes.
    """
    minx, miny, maxx, maxy = poly.bounds
    row_top, col_left = rowcol(transform, minx, maxy)   # linha cresce para baixo
    row_bot, col_right = rowcol(transform, maxx, miny)
    row0, col0 = max(int(row_top) - pad, 0), max(int(col_left) - pad, 0)
    row1, col1 = int(row_bot) + pad + 1, int(col_right) + pad + 1
    h, w = row1 - row0, col1 - col0
    if h < 3 or w < 3:
        return [poly]

    local_tf = window_transform(Window(col0, row0, w, h), transform)
    mask = rasterize([(poly, 1)], out_shape=(h, w), transform=local_tf, dtype="uint8")
    if mask.sum() == 0:
        return [poly]

    dist = ndimage.distance_transform_edt(mask)
    coords = peak_local_max(dist, min_distance=min_peak_distance_px, labels=mask)
    if len(coords) <= 1:
        return [poly]                       # um pico só -> não separa

    markers = np.zeros(dist.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    labels = watershed(-dist, markers, mask=mask.astype(bool))

    result: list[Polygon] = []
    for geom, _ in rio_shapes(labels.astype(np.int32), mask=labels > 0,
                              transform=local_tf):
        g = shape(geom)
        if isinstance(g, Polygon) and not g.is_empty:
            result.append(g)
    return result if result else [poly]


def vectorize(prob_path: str, output_path: str, cfg: dict) -> str:
    """Vetoriza a máscara de probabilidade e salva GeoPackage."""
    pcfg = cfg["postprocess"]
    threshold = cfg["inference"]["threshold"]

    with rasterio.open(prob_path) as src:
        prob = src.read(1)
        transform = src.transform
        crs = src.crs
        pixel_area = abs(transform.a * transform.e)  # m² por pixel (CRS projetado)

    binary = (prob >= threshold).astype(np.uint8)
    if binary.sum() == 0:
        print("[vectorize] AVISO: máscara vazia após threshold — nenhum polígono.")

    instance_sep = pcfg.get("instance_separation", False)
    min_peak = int(pcfg.get("min_peak_distance_px", 25))
    min_split_area = float(pcfg.get("min_split_area_m2", 40.0))

    geoms = []
    n_blobs = 0
    for geom, val in rio_shapes(binary, mask=binary.astype(bool), transform=transform):
        if val != 1:
            continue
        poly = shape(geom)
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        if poly.area < pcfg["min_area_m2"]:
            continue
        n_blobs += 1

        # Separa blobs grandes em prédios individuais (watershed).
        if instance_sep and poly.area >= min_split_area:
            candidates = _split_instances(poly, transform, min_peak)
        else:
            candidates = [poly]

        for cand in candidates:
            cand = _fill_small_holes(cand, pcfg["fill_holes_m2"])
            cand = cand.simplify(pcfg["simplify_tolerance_m"], preserve_topology=True)
            if pcfg.get("orthogonalize", False):
                cand = _orthogonalize(cand)
            if cand.is_valid and not cand.is_empty and cand.area >= pcfg["min_area_m2"]:
                geoms.append(cand)

    if instance_sep:
        print(f"[vectorize] separação por instância: {n_blobs} blobs -> {len(geoms)} prédios")

    gdf = gpd.GeoDataFrame(
        {"id": range(1, len(geoms) + 1),
         "area_m2": [round(g.area, 2) for g in geoms]},
        geometry=geoms,
        crs=crs,
    )
    ensure_dir(Path(output_path).parent)
    gdf.to_file(output_path, driver="GPKG", layer="edificacoes")
    print(f"[vectorize] {len(gdf)} edificações -> {output_path}")
    return output_path


def _run_from_config(config_path: str, city: str | None,
                     prob: str | None, output: str | None) -> None:
    cfg = load_config(config_path)
    if prob and output:
        vectorize(prob, output, cfg)
        return

    mask_dir = Path(cfg["inference"]["out_mask_dir"])
    out_vector = cfg["postprocess"]["out_vector"]
    targets = [c for c in cfg["cities"] if city is None or c["name"] == city]
    for c in targets:
        prob_path = mask_dir / f"{c['name']}_prob.tif"
        # Uma camada por cidade: sufixa o nome quando há mais de uma.
        out = out_vector if len(targets) == 1 else \
            str(Path(out_vector).with_name(f"{Path(out_vector).stem}_{c['name']}.gpkg"))
        vectorize(str(prob_path), out, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Vetoriza máscara de probabilidade.")
    ap.add_argument("--config", required=True, help="Config YAML.")
    ap.add_argument("--city", help="Cidade da config (default: todas).")
    ap.add_argument("--prob", help="Máscara de probabilidade avulsa.")
    ap.add_argument("--output", help="GeoPackage de saída (avulso).")
    args = ap.parse_args()
    _run_from_config(args.config, args.city, args.prob, args.output)


if __name__ == "__main__":
    main()
