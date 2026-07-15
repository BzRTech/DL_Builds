"""Utilitários compartilhados: carregamento de config, seeds e I/O geoespacial."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Carrega o YAML de configuração como dicionário."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Fixa as seeds para reprodutibilidade (numpy, random e, se disponível, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str | os.PathLike) -> Path:
    """Cria o diretório (e pais) se não existir e retorna o Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def has_ecw_driver() -> bool:
    """Indica se o GDAL instalado consegue ler ECW (driver proprietário da ERDAS)."""
    try:
        from osgeo import gdal
    except ImportError:
        return False
    return gdal.GetDriverByName("ECW") is not None
