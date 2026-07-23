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


def _instances_tiled(binary: np.ndarray, transform, min_peak_distance_px: int,
                     tile: int = 4096, overlap: int = 512):
    """Separa prédios encostados via watershed, varrendo a máscara em tiles de
    tamanho fixo (com sobreposição). Assim a memória fica limitada mesmo quando a
    máscara forma um único componente conectado gigante.

    Cada polígono é atribuído ao tile cujo "miolo" (tile menos as margens de
    sobreposição) contém o seu ponto representativo, evitando contagem dupla nas
    bordas. Devolve uma lista de polígonos (geo).
    """
    height, width = binary.shape
    step = tile - overlap
    margin = overlap // 2
    polys: list[Polygon] = []

    for r0 in range(0, height, step):
        for c0 in range(0, width, step):
            r1, c1 = min(r0 + tile, height), min(c0 + tile, width)
            sub = binary[r0:r1, c0:c1]
            if not sub.any():
                continue

            dist = ndimage.distance_transform_edt(sub)
            coords = peak_local_max(dist, min_distance=min_peak_distance_px, labels=sub)
            if len(coords) == 0:
                labels, _ = ndimage.label(sub)          # sem picos: 1 rótulo por componente
            else:
                markers = np.zeros(sub.shape, dtype=np.int32)
                for i, (rr, cc) in enumerate(coords, start=1):
                    markers[rr, cc] = i
                labels = watershed(-dist, markers, mask=sub.astype(bool))

            local_tf = window_transform(Window(c0, r0, c1 - c0, r1 - r0), transform)
            # limites do "miolo" em pixels globais (nas bordas da imagem vai até a ponta)
            core_r0 = r0 + margin if r0 > 0 else 0
            core_c0 = c0 + margin if c0 > 0 else 0
            core_r1 = r1 - margin if r1 < height else height
            core_c1 = c1 - margin if c1 < width else width

            for geom, _ in rio_shapes(labels.astype(np.int32), mask=labels > 0,
                                      transform=local_tf):
                g = shape(geom)
                if not isinstance(g, Polygon) or g.is_empty:
                    continue
                rp = g.representative_point()
                rr, cc = rowcol(transform, rp.x, rp.y)
                if core_r0 <= rr < core_r1 and core_c0 <= cc < core_c1:
                    polys.append(g)
    return polys


def vectorize(prob_path: str, output_path: str, cfg: dict) -> str:
    """Vetoriza a máscara de probabilidade e salva GeoPackage."""
    pcfg = cfg["postprocess"]
    threshold = cfg["inference"]["threshold"]

    with rasterio.open(prob_path) as src:
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width
        # Constrói a binária em blocos (uint8), evitando carregar o float32
        # inteiro na RAM — essencial em ortofotos gigantes (dezenas de gigapixels).
        binary = np.zeros((height, width), dtype=np.uint8)
        blk = 4096
        for r0 in range(0, height, blk):
            r1 = min(r0 + blk, height)
            prob_block = src.read(1, window=Window(0, r0, width, r1 - r0))
            binary[r0:r1] = (prob_block >= threshold).astype(np.uint8)

    if binary.sum() == 0:
        print("[vectorize] AVISO: máscara vazia após threshold — nenhum polígono.")

    instance_sep = pcfg.get("instance_separation", False)
    # Distância mínima entre "picos" (centros de prédios). Em METROS é consistente
    # entre cidades com resoluções (GSD) diferentes; converte para pixels pela
    # resolução do raster. Cai para _px se _m não estiver definido.
    pixel_size = abs(transform.a)  # m por pixel (CRS projetado)
    if pcfg.get("min_peak_distance_m") is not None:
        min_peak = max(1, int(round(float(pcfg["min_peak_distance_m"]) / pixel_size)))
    else:
        min_peak = int(pcfg.get("min_peak_distance_px", 25))
    tile_px = int(pcfg.get("instance_tile_px", 4096))
    overlap_px = int(pcfg.get("instance_overlap_px", 512))

    # Polígonos-base: separados por instância (watershed em tiles) ou direto por
    # componente conectado (rio_shapes).
    if instance_sep:
        raw_polys = _instances_tiled(binary, transform, min_peak, tile_px, overlap_px)
    else:
        raw_polys = []
        for geom, val in rio_shapes(binary, mask=binary.astype(bool),
                                    transform=transform):
            if val != 1:
                continue
            g = shape(geom)
            if isinstance(g, Polygon) and not g.is_empty:
                raw_polys.append(g)

    geoms = []
    for poly in raw_polys:
        if poly.is_empty or poly.area < pcfg["min_area_m2"]:
            continue
        poly = _fill_small_holes(poly, pcfg["fill_holes_m2"])
        poly = poly.simplify(pcfg["simplify_tolerance_m"], preserve_topology=True)
        if pcfg.get("orthogonalize", False):
            poly = _orthogonalize(poly)
        if poly.is_valid and not poly.is_empty and poly.area >= pcfg["min_area_m2"]:
            geoms.append(poly)

    if instance_sep:
        print(f"[vectorize] separação por instância -> {len(geoms)} prédios")

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
