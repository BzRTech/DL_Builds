"""Classifica o pavimento das vias de uma cidade NOVA em um comando.

Prepara o COG a partir da ortofoto (se ainda não existir) e roda o classificador
já treinado sobre os logradouros informados — sem re-treino e sem rótulos.

Uso:
    python -m src.classify_roads_city --city NovaCidade \
        --image data/raw/NovaCidade/ortofoto.tif \
        --roads data/raw/NovaCidade/logradouros.shp

Saída: outputs/<cidade>/vias.gpkg, com as colunas 'PAV_PRED' (classe prevista)
e 'PAV_CONF' (fração de votos dos patches do trecho).

Se a ortofoto for ECW e o GDAL instalado não tiver o driver de leitura, converta
antes para GeoTIFF no QGIS (ver README).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import geopandas as gpd

from src.road_surface.predict import classify_roads
from src.utils import ensure_dir, has_ecw_driver, load_config

PROCESSED_DIR = Path("data/processed")


def prepare_cog(image: str, city: str, force: bool = False) -> str:
    """Devolve o caminho do COG a usar, convertendo a ortofoto só quando preciso."""
    src = Path(image)
    if not src.exists():
        raise SystemExit(f"Ortofoto não encontrada: {src}")
    if src.suffix.lower() == ".ecw" and not has_ecw_driver():
        raise SystemExit(
            f"'{src}' é ECW e o GDAL instalado não lê ECW.\n"
            "Abra a ortofoto no QGIS e exporte como GeoTIFF (o QGIS embute o driver);\n"
            "depois rode este comando apontando --image para o .tif exportado."
        )

    target = PROCESSED_DIR / f"{city}.tif"
    if target.exists() and src.resolve() == target.resolve():
        return str(src)
    if target.exists() and not force:
        print(f"[vias] COG já existe, reaproveitando: {target}")
        print("       (use --force-cog para gerar de novo)")
        return str(target)

    from src.data_prep.ecw_to_cog import convert  # importa GDAL só quando necessário

    ensure_dir(PROCESSED_DIR)
    print(f"[vias] Gerando COG: {src} -> {target}")
    return convert(str(src), str(target))


def _resumo(out_path: str) -> None:
    """Imprime o panorama por classe: nº de trechos e extensão (quando em metros)."""
    gdf = gpd.read_file(out_path)
    if "PAV_PRED" not in gdf.columns or gdf.empty:
        return
    metrico = gdf.crs is not None and not gdf.crs.is_geographic
    n, ext = defaultdict(int), defaultdict(float)
    for classe, geom in zip(gdf["PAV_PRED"], gdf.geometry):
        rotulo = classe if classe is not None else "(sem classificação)"
        n[rotulo] += 1
        if metrico and geom is not None:
            ext[rotulo] += geom.length

    print(f"[vias] Resumo de {len(gdf)} trechos:")
    for rotulo in sorted(n, key=lambda k: -n[k]):
        linha = f"       {rotulo:<24} {n[rotulo]:>6} trechos"
        if metrico:
            linha += f"   {ext[rotulo] / 1000:>8.2f} km"
        print(linha)
    if not metrico:
        print("       (CRS geográfico — extensão em km omitida)")


def run(image: str, roads: str, city: str, out: str | None = None,
        config: str = "configs/road_surface.yaml", force_cog: bool = False,
        skip_cog: bool = False) -> str:
    """Prepara o COG (se preciso) e classifica o pavimento dos logradouros."""
    if not Path(roads).exists():
        raise SystemExit(f"Shapefile de logradouros não encontrado: {roads}")

    cog = str(image) if skip_cog else prepare_cog(image, city, force_cog)
    out_path = Path(out) if out else Path("outputs") / city / "vias.gpkg"
    ensure_dir(out_path.parent)

    print(f"[vias] Classificando o pavimento: {roads}")
    classify_roads(cog, roads, str(out_path), load_config(config))
    _resumo(str(out_path))
    print(f"[vias] Pronto! Camada em: {out_path}")
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Classifica o pavimento das vias de uma cidade nova (1 comando).")
    ap.add_argument("--image", required=True,
                    help="Ortofoto (GeoTIFF/COG; ECW só se o GDAL tiver o driver).")
    ap.add_argument("--roads", required=True, help="Shapefile de logradouros (linhas).")
    ap.add_argument("--city", required=True, help="Nome da cidade (define a saída).")
    ap.add_argument("--out", help="GeoPackage de saída (default: outputs/<cidade>/vias.gpkg).")
    ap.add_argument("--config", default="configs/road_surface.yaml", help="Config YAML.")
    ap.add_argument("--force-cog", action="store_true",
                    help="Regera o COG mesmo que data/processed/<cidade>.tif já exista.")
    ap.add_argument("--skip-cog", action="store_true",
                    help="Usa --image direto, sem gerar COG.")
    args = ap.parse_args()
    run(args.image, args.roads, args.city, args.out, args.config,
        args.force_cog, args.skip_cog)


if __name__ == "__main__":
    main()
