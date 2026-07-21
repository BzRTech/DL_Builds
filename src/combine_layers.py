"""Junta as camadas de saída (edificações, quadras, lotes) em UM único GeoPackage.

Cada alvo vira uma camada (layer) dentro do mesmo arquivo .gpkg — pronto para
abrir no QGIS/ArcGIS com tudo junto.

Uso:
    python -m src.combine_layers --out outputs/malta_resultado.gpkg \
        --layer edificacoes=outputs/edificacoes.gpkg \
        --layer quadras=outputs/quadras.gpkg \
        --layer lotes=outputs/lotes.gpkg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

from src.utils import ensure_dir


def combine(out_path: str, layers: list[tuple[str, str]]) -> str:
    ensure_dir(Path(out_path).parent)
    written = 0
    for name, path in layers:
        if not Path(path).exists():
            print(f"[combine] AVISO: pulando '{name}' — não encontrado: {path}")
            continue
        gdf = gpd.read_file(path)
        gdf.to_file(out_path, driver="GPKG", layer=name)
        print(f"[combine] camada '{name}': {len(gdf)} feições  <-  {path}")
        written += 1
    if written:
        print(f"[combine] {written} camadas -> {out_path}")
    else:
        print("[combine] Nenhuma camada escrita (nenhum arquivo de entrada existe).")
    return out_path


def _parse_layer(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--layer deve ser nome=caminho (recebido: {spec})")
    name, path = spec.split("=", 1)
    return name.strip(), path.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Junta camadas em um único GeoPackage.")
    ap.add_argument("--out", required=True, help="GeoPackage de saída.")
    ap.add_argument("--layer", action="append", type=_parse_layer, required=True,
                    help="nome=caminho (repita para cada camada).")
    args = ap.parse_args()
    combine(args.out, args.layer)


if __name__ == "__main__":
    main()
