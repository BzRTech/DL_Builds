"""Converte ortofotos (ECW ou outro raster) para Cloud-Optimized GeoTIFF (COG).

Se o raster de entrada já for GeoTIFF/COG, ainda vale rodar para padronizar
compressão, tiling e overviews.

Uso:
    python -m src.data_prep.ecw_to_cog --config configs/buildings.yaml
    python -m src.data_prep.ecw_to_cog --input a.ecw --output a.tif
"""
from __future__ import annotations

import argparse
from pathlib import Path

from osgeo import gdal

from src.utils import ensure_dir, has_ecw_driver, load_config

gdal.UseExceptions()


def convert(input_path: str, output_path: str, blocksize: int = 512) -> str:
    """Converte um raster para COG (GeoTIFF, compressão DEFLATE, com overviews)."""
    in_p = Path(input_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Raster de entrada não encontrado: {in_p}")
    if in_p.suffix.lower() == ".ecw" and not has_ecw_driver():
        raise RuntimeError(
            f"'{in_p}' é ECW mas o GDAL instalado não tem o driver de leitura ECW.\n"
            "Converta antes para GeoTIFF/COG usando o QGIS, ou instale um GDAL com "
            "suporte a ECW. Ver README para detalhes."
        )

    ensure_dir(Path(output_path).parent)
    creation = [
        f"BLOCKSIZE={blocksize}",
        "COMPRESS=DEFLATE",
        "PREDICTOR=2",
        "NUM_THREADS=ALL_CPUS",
        "BIGTIFF=IF_SAFER",
    ]
    print(f"[ecw_to_cog] {in_p}  ->  {output_path}")
    gdal.Translate(
        output_path,
        str(in_p),
        format="COG",
        creationOptions=creation,
    )
    return output_path


def _run_from_config(config_path: str) -> None:
    cfg = load_config(config_path)
    processed_dir = ensure_dir(cfg["data_prep"]["processed_dir"])
    for city in cfg["cities"]:
        out = processed_dir / f"{city['name']}.tif"
        convert(city["image"], str(out))
    print("[ecw_to_cog] Concluído.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Converte raster (ECW/…) para COG GeoTIFF.")
    ap.add_argument("--config", help="Config YAML: converte a imagem de cada cidade.")
    ap.add_argument("--input", help="Raster de entrada (uso avulso).")
    ap.add_argument("--output", help="COG de saída (uso avulso).")
    args = ap.parse_args()

    if args.config:
        _run_from_config(args.config)
    elif args.input and args.output:
        convert(args.input, args.output)
    else:
        ap.error("Informe --config OU (--input e --output).")


if __name__ == "__main__":
    main()
