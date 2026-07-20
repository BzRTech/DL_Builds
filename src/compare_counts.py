"""Comparação rápida de quantidade e área entre rótulos (verdade) e predição.

Não faz correspondência por objeto (isso é o eval.evaluate) — só dá o panorama
geral: quantos polígonos e qual a área total de cada lado. Útil para um sanity
check rápido do modelo.

Uso:
    python -m src.compare_counts --pred outputs/edificacoes.gpkg \
        --truth data/raw/Malta/edificacoes_merged.gpkg
"""
from __future__ import annotations

import argparse

import geopandas as gpd


def compare(pred_path: str, truth_path: str) -> dict:
    pred = gpd.read_file(pred_path)
    truth = gpd.read_file(truth_path)
    if truth.crs is not None and pred.crs is not None and pred.crs != truth.crs:
        truth = truth.to_crs(pred.crs)

    n_pred, n_truth = len(pred), len(truth)
    area_pred = float(pred.geometry.area.sum())
    area_truth = float(truth.geometry.area.sum())

    print(f"VERDADE : {n_truth:>7} polígonos | área total {area_truth/1e4:>10.1f} ha")
    print(f"PREVISTO: {n_pred:>7} polígonos | área total {area_pred/1e4:>10.1f} ha")
    print("-" * 52)
    print(f"razão de quantidade (prev/verdade): {n_pred / max(n_truth, 1):.2f}")
    print(f"razão de área       (prev/verdade): {area_pred / max(area_truth, 1e-9):.2f}")
    print("\nLeitura: razão de ÁREA perto de 1.0 = footprint bem capturado.")
    print("         razão de QUANTIDADE muito >1 = super-segmentação;")
    print("         muito <1 = prédios fundidos (sub-segmentação).")
    return {"n_pred": n_pred, "n_truth": n_truth,
            "area_pred_ha": area_pred / 1e4, "area_truth_ha": area_truth / 1e4}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara quantidade/área verdade vs. predição.")
    ap.add_argument("--pred", required=True, help="GeoPackage/Shapefile previsto.")
    ap.add_argument("--truth", required=True, help="GeoPackage/Shapefile de verdade.")
    args = ap.parse_args()
    compare(args.pred, args.truth)


if __name__ == "__main__":
    main()
