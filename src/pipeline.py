"""Fluxo completo por cidade: dado a ortofoto (e os logradouros), roda tudo na
ordem correta e organiza a saída em outputs/<cidade>/.

Ordem (respeita as dependências):
    1) Edificações  (segmentação)                    -> edificacoes.gpkg
    2) Quadras      (recorte pela rede viária)        -> quadras_vias.gpkg   [precisa logradouros]
    3) Lotes        (subdivide quadras P/ edificações)-> lotes_vias.gpkg + cadastro.gpkg
    4) Vias         (classificação de pavimento)      -> vias.gpkg           [precisa logradouros]

Uso (CLI):
    python -m src.pipeline --image data/processed/Cidade.tif --city Cidade \
        --roads data/raw/Cidade/logradouros.shp
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.cadastral.lotes_from_quadras import build_cadastro
from src.cadastral.quadras_from_roads import build_quadras
from src.inference.predict import predict_image
from src.postprocess.vectorize import vectorize
from src.road_surface.predict import classify_roads
from src.utils import ensure_dir, load_config


def run_city(image: str, city: str, roads: str | None = None,
             boundary: str | None = None, log=print) -> list[tuple[str, str]]:
    """Executa o fluxo completo para uma cidade. Retorna [(rótulo, caminho), ...]."""
    out = ensure_dir(Path("outputs") / city)
    tmp = ensure_dir(out / "_tmp")
    results: list[tuple[str, str]] = []

    # 1) Edificações
    log("🏠 Edificações — inferência (pode demorar em ortofotos grandes)…")
    bcfg = load_config("configs/buildings.yaml")
    prob = str(tmp / "edif_prob.tif")
    predict_image(image, prob, bcfg)
    log("🏠 Edificações — vetorização…")
    edif = str(out / "edificacoes.gpkg")
    vectorize(prob, edif, bcfg)
    results.append(("Edificações", edif))

    if roads:
        cad = load_config("configs/cadastral.yaml")
        crs, p = cad["crs"], cad["params"]

        # 2) Quadras (pela rede viária)
        log("🧩 Quadras — recorte pela rede viária…")
        q = build_quadras(roads, boundary, crs, p["street_buffer_m"], p["min_quadra_area_m2"])
        qpath = str(out / "quadras_vias.gpkg")
        q.to_file(qpath, driver="GPKG", layer="quadras")
        results.append(("Quadras", qpath))

        # 3) Lotes (subdivide as quadras usando as edificações EXTRAÍDAS)
        log("📐 Lotes — subdivisão das quadras pelas edificações…")
        lotes, builds = build_cadastro(
            qpath, edif, crs, p["min_lote_area_m2"], p.get("method", "rect"),
            float(p.get("lot_min_depth_m", 8.0)), float(p.get("min_lot_width_m", 0.0)),
            bool(p.get("clean_buildings", True)))
        lpath = str(out / "lotes_vias.gpkg")
        lotes.to_file(lpath, driver="GPKG", layer="lotes")
        cadpath = str(out / "cadastro.gpkg")
        lotes.to_file(cadpath, driver="GPKG", layer="lotes")
        builds.to_file(cadpath, driver="GPKG", layer="edificacoes")
        results.append(("Lotes", lpath))
        results.append(("Cadastro (lotes + edificações)", cadpath))

        # 4) Vias (pavimento)
        log("🛣️ Vias — classificação do pavimento…")
        vpath = str(out / "vias.gpkg")
        classify_roads(image, roads, vpath, load_config("configs/road_surface.yaml"))
        results.append(("Vias (pavimento)", vpath))
    else:
        log("(sem logradouros — pulei quadras, lotes e vias)")

    shutil.rmtree(tmp, ignore_errors=True)
    log("✅ Concluído.")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Fluxo completo por cidade.")
    ap.add_argument("--image", required=True, help="Ortofoto (COG/GeoTIFF).")
    ap.add_argument("--city", required=True, help="Nome da cidade (pasta de saída).")
    ap.add_argument("--roads", help="Shapefile de logradouros (para quadras/lotes/vias).")
    ap.add_argument("--boundary", help="Perímetro urbano (opcional).")
    args = ap.parse_args()
    res = run_city(args.image, args.city, args.roads, args.boundary)
    print("\nSaídas:")
    for label, path in res:
        print(f"  - {label}: {path}")


if __name__ == "__main__":
    main()
