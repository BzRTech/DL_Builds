"""Inspeção rápida dos dados de uma cidade (Fase 0).

Rode como script ou converta em notebook. Verifica: leitura da imagem (ECW/COG),
CRS, resolução (GSD), bandas, extensão, leitura do shapefile e alinhamento visual
imagem x rótulos num recorte.

Uso:
    python notebooks/00_inspecao_dados.py --image data/raw/cidade_a/ortofoto.ecw \
        --labels data/raw/cidade_a/edificacoes.shp
"""
from __future__ import annotations

import argparse

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.windows import Window

from src.utils import has_ecw_driver


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="outputs/inspecao.png")
    args = ap.parse_args()

    print("Driver ECW disponível no GDAL:", has_ecw_driver())

    with rasterio.open(args.image) as src:
        print("\n== Imagem ==")
        print("  CRS       :", src.crs)
        print("  Tamanho   :", src.width, "x", src.height, "px")
        print("  Bandas    :", src.count, "| dtypes:", src.dtypes)
        print("  Resolução :", src.res, "(unidades do CRS por pixel)")
        print("  Extensão  :", src.bounds)

        win = Window(0, 0, min(1024, src.width), min(1024, src.height))
        rgb = src.read([1, 2, 3], window=win)
        rgb_img = reshape_as_image(np.clip(rgb, 0, 255).astype("uint8"))
        win_transform = src.window_transform(win)
        win_bounds = rasterio.windows.bounds(win, src.transform)

    gdf = gpd.read_file(args.labels)
    print("\n== Rótulos ==")
    print("  CRS       :", gdf.crs)
    print("  Feições   :", len(gdf))
    print("  Colunas   :", list(gdf.columns))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb_img, extent=[win_bounds[0], win_bounds[2],
                               win_bounds[1], win_bounds[3]])
    try:
        clip = gdf.to_crs(gdf.crs).cx[win_bounds[0]:win_bounds[2],
                                      win_bounds[1]:win_bounds[3]]
        clip.boundary.plot(ax=ax, color="red", linewidth=0.8)
    except Exception as e:  # noqa: BLE001
        print("  (não foi possível sobrepor rótulos:", e, ")")
    ax.set_title("Recorte: imagem + limites de edificações (vermelho)")
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"\nFigura de inspeção salva em: {args.out}")


if __name__ == "__main__":
    main()
