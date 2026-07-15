"""Avalia os polígonos preditos contra a verdade-terreno (por objeto).

Faz correspondência por IoU entre polígonos preditos e de referência e reporta
precision/recall/F1 por limiar de IoU, além das contagens. Salva um JSON de relatório.

Uso:
    python -m src.eval.evaluate --config configs/buildings.yaml --city cidade_b
    python -m src.eval.evaluate --config configs/buildings.yaml \
        --pred outputs/cidade_b.gpkg --truth data/raw/cidade_b/edificacoes.shp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd

from src.utils import ensure_dir, load_config


def _match_and_score(pred: gpd.GeoDataFrame, truth: gpd.GeoDataFrame,
                     iou_threshold: float) -> dict:
    """Correspondência gulosa por IoU; retorna métricas por objeto."""
    if truth.crs is not None and pred.crs != truth.crs:
        truth = truth.to_crs(pred.crs)

    pred = pred.reset_index(drop=True)
    truth = truth.reset_index(drop=True)

    # Candidatos por interseção espacial (índice espacial acelera).
    joined = gpd.sjoin(pred[["geometry"]], truth[["geometry"]],
                       how="inner", predicate="intersects")

    pairs = []  # (iou, pred_idx, truth_idx)
    for pred_idx, row in joined.iterrows():
        t_idx = row["index_right"]
        pg, tg = pred.geometry.iloc[pred_idx], truth.geometry.iloc[t_idx]
        inter = pg.intersection(tg).area
        if inter <= 0:
            continue
        union = pg.area + tg.area - inter
        iou = inter / union if union > 0 else 0.0
        if iou >= iou_threshold:
            pairs.append((iou, pred_idx, t_idx))

    # Matching guloso: maiores IoUs primeiro, sem reutilizar pred/truth.
    pairs.sort(reverse=True)
    used_pred, used_truth = set(), set()
    tp = 0
    for iou, p_idx, t_idx in pairs:
        if p_idx in used_pred or t_idx in used_truth:
            continue
        used_pred.add(p_idx)
        used_truth.add(t_idx)
        tp += 1

    fp = len(pred) - tp
    fn = len(truth) - tp
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "iou_threshold": iou_threshold,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_pred": len(pred), "n_truth": len(truth),
    }


def evaluate(pred_path: str, truth_path: str, iou_thresholds: list[float],
             report_path: str) -> dict:
    """Roda a avaliação e salva o relatório JSON."""
    pred = gpd.read_file(pred_path)
    truth = gpd.read_file(truth_path)
    print(f"[eval] preditos={len(pred)}  verdade={len(truth)}")

    results = [_match_and_score(pred, truth, t) for t in iou_thresholds]
    report = {"pred": pred_path, "truth": truth_path, "metrics": results}

    ensure_dir(Path(report_path).parent)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    for r in results:
        print(f"[eval] IoU>={r['iou_threshold']}: "
              f"P={r['precision']} R={r['recall']} F1={r['f1']} "
              f"(TP={r['tp']} FP={r['fp']} FN={r['fn']})")
    print(f"[eval] Relatório salvo: {report_path}")
    return report


def _run_from_config(config_path: str, city: str | None,
                     pred: str | None, truth: str | None) -> None:
    cfg = load_config(config_path)
    thresholds = cfg["eval"]["iou_thresholds"]
    report_dir = Path(cfg["eval"]["report_dir"])

    if pred and truth:
        evaluate(pred, truth, thresholds, str(report_dir / "report.json"))
        return

    # Avalia as cidades de teste (leave-one-city-out).
    out_vector = cfg["postprocess"]["out_vector"]
    test_cities = [c for c in cfg["cities"]
                   if (city is None and c.get("split") == "test") or c["name"] == city]
    if not test_cities:
        raise SystemExit("Nenhuma cidade de teste na config (split: test).")
    for c in test_cities:
        n_targets = sum(1 for x in cfg["cities"] if x.get("split") == "test")
        pred_path = out_vector if n_targets == 1 and city is None else \
            str(Path(out_vector).with_name(f"{Path(out_vector).stem}_{c['name']}.gpkg"))
        evaluate(pred_path, c["labels"], thresholds,
                 str(report_dir / f"report_{c['name']}.json"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Avalia polígonos preditos vs. verdade.")
    ap.add_argument("--config", required=True, help="Config YAML.")
    ap.add_argument("--city", help="Cidade da config (default: cidades de teste).")
    ap.add_argument("--pred", help="GeoPackage/Shapefile de predições (avulso).")
    ap.add_argument("--truth", help="Shapefile de verdade-terreno (avulso).")
    args = ap.parse_args()
    _run_from_config(args.config, args.city, args.pred, args.truth)


if __name__ == "__main__":
    main()
