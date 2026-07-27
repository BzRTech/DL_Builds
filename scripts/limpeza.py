"""Relatório e limpeza de disco do projeto.

Sem argumentos: apenas RELATA tamanhos (nada é apagado).
Com --apply: apaga os intermediários REGENERÁVEIS (tiles, máscaras de
probabilidade, patches, temporários). NUNCA toca em data/raw (fonte), nos
resultados .gpkg, nos modelos (runs/) nem nos COGs (data/processed/*.tif),
a menos que você passe --cogs (aí também apaga os COGs, que são regeneráveis
a partir de data/raw via ecw_to_cog).

Uso:
    python scripts/limpeza.py            # só relatório
    python scripts/limpeza.py --apply    # apaga intermediários regeneráveis
    python scripts/limpeza.py --apply --cogs   # + apaga os COGs de data/processed
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _gb(n: int) -> float:
    return round(n / 1e9, 2)


def _collect() -> dict[str, list[Path]]:
    """Itens REGENERÁVEIS, seguros para apagar."""
    groups: dict[str, list[Path]] = {
        "Tiles de treino": [Path(d) for d in
                            ("data/tiles", "data/tiles_quadras", "data/tiles_lotes")],
        "Patches de vias": [Path("data/road_patches")],
        "Máscaras de probabilidade": [Path(d) for d in
                            ("outputs/masks", "outputs/masks_quadras", "outputs/masks_lotes")],
    }
    globs = {
        "Máscaras soltas (*_prob.tif)": list(Path("outputs").rglob("*_prob.tif")),
        "Máscaras rasterizadas (data/processed/*_mask.tif)":
            list(Path("data/processed").glob("*_mask.tif")) if Path("data/processed").exists() else [],
        "Temporários": list(Path("outputs").rglob("_tmp")) + list(Path("outputs").rglob("predict_*")),
    }
    out: dict[str, list[Path]] = {}
    for k, v in groups.items():
        ex = [p for p in v if p.exists()]
        if ex:
            out[k] = ex
    for k, v in globs.items():
        ex = [p for p in v if p.exists()]
        if ex:
            out[k] = ex
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Relatório e limpeza de disco.")
    ap.add_argument("--apply", action="store_true", help="Apaga os intermediários regeneráveis.")
    ap.add_argument("--cogs", action="store_true",
                    help="Também apaga os COGs de data/processed (regeneráveis via ecw_to_cog).")
    args = ap.parse_args()

    print("=== MANTIDOS (não serão apagados) ===")
    for p in ["data/raw", "data/processed", "runs"]:
        if Path(p).exists():
            print(f"  {p:22s} {_gb(_size(Path(p))):>8} GB")
    gpkgs = list(Path("outputs").rglob("*.gpkg")) if Path("outputs").exists() else []
    print(f"  {'resultados (*.gpkg)':22s} {_gb(sum(_size(p) for p in gpkgs)):>8} GB")

    groups = _collect()
    if args.cogs and Path("data/processed").exists():
        cogs = [p for p in Path("data/processed").glob("*.tif") if not p.name.endswith("_mask.tif")]
        if cogs:
            groups["COGs (data/processed/*.tif)"] = cogs

    print("\n=== REGENERÁVEIS (apagáveis) ===")
    total = 0
    for k, paths in groups.items():
        s = sum(_size(p) for p in paths)
        total += s
        print(f"  {k:42s} {_gb(s):>8} GB  ({len(paths)} item(ns))")
    print(f"  {'-'*42} {'-'*8}")
    print(f"  {'TOTAL apagável':42s} {_gb(total):>8} GB")

    if not args.apply:
        print("\n(Relatório apenas.) Para apagar de verdade: "
              "python scripts/limpeza.py --apply")
        return

    print("\nApagando…")
    freed = 0
    for _, paths in groups.items():
        for p in paths:
            try:
                freed += _size(p)
                shutil.rmtree(p) if p.is_dir() else p.unlink()
                print(f"  removido: {p}")
            except Exception as e:  # noqa: BLE001
                print(f"  ERRO ao remover {p}: {e}")
    print(f"\n✅ Liberado ~{_gb(freed)} GB.")
    print("Obs.: tiles/máscaras se regeneram com o pipeline; COGs, com ecw_to_cog.")


if __name__ == "__main__":
    main()
