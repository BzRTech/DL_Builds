"""Aplica o modelo já treinado a uma NOVA ortofoto (sem rótulos), em um comando.

Faz inferência (sliding window) + vetorização e gera o GeoPackage de edificações.
Não precisa de shapefile nem de re-treino — usa o checkpoint indicado na config
(inference.checkpoint).

Uso:
    python -m src.predict_city --image data/raw/NovaCidade/ortofoto.tif \
        --out outputs/NovaCidade.gpkg

Se a ortofoto for ECW e o GDAL não tiver o driver, converta antes para GeoTIFF
no QGIS (ver README).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.inference.predict import predict_image
from src.postprocess.vectorize import vectorize
from src.utils import ensure_dir, has_ecw_driver, load_config


def run(image: str, out_gpkg: str, config: str = "configs/buildings.yaml") -> str:
    """Inferência + vetorização de uma ortofoto nova. Retorna o caminho do GeoPackage."""
    if str(image).lower().endswith(".ecw") and not has_ecw_driver():
        raise SystemExit(
            "A entrada é ECW e o GDAL instalado não lê ECW. "
            "Converta a ortofoto para GeoTIFF no QGIS primeiro (ver README)."
        )
    cfg = load_config(config)
    out = Path(out_gpkg)
    ensure_dir(out.parent)
    prob = out.with_name(out.stem + "_prob.tif")

    print(f"[predict_city] 1/2 Inferência: {image}")
    predict_image(str(image), str(prob), cfg)
    print(f"[predict_city] 2/2 Vetorização -> {out}")
    vectorize(str(prob), str(out), cfg)
    print(f"[predict_city] Pronto! Edificações em: {out}")
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extrai edificações de uma nova ortofoto (inferência + vetorização)."
    )
    ap.add_argument("--image", required=True,
                    help="Ortofoto GeoTIFF/COG. Se for ECW, converta no QGIS antes.")
    ap.add_argument("--out", required=True, help="GeoPackage de saída.")
    ap.add_argument("--config", default="configs/buildings.yaml", help="Config YAML.")
    args = ap.parse_args()
    run(args.image, args.out, args.config)


if __name__ == "__main__":
    main()
