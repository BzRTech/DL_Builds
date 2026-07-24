"""Normalização de rótulos de classe (ignora acento/caixa/espaços).

Cidades diferentes grafam o mesmo valor de forma diferente (ex.: 'NÃO
PAVIMENTADA' vs 'NAO PAVIMENTADA', 'ASFÁLTICO' vs 'ASFALTICO'). Normalizar
permite casar todos com as classes canônicas da config.
"""
from __future__ import annotations

import unicodedata


def norm_label(s) -> str | None:
    if s is None:
        return None
    t = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return " ".join(t.upper().split())
