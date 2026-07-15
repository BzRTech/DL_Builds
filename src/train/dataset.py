"""Dataset PyTorch de tiles pareados (imagem/máscara) com augmentation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import albumentations as A
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


def load_manifest(tiles_dir: str | Path, split: str) -> list[dict]:
    """Carrega as linhas do manifest para um split ('train' | 'val' | 'test')."""
    manifest = Path(tiles_dir) / "manifest.csv"
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    if not rows:
        raise RuntimeError(f"Nenhum tile no split '{split}' em {manifest}.")
    return rows


def load_norm_stats(tiles_dir: str | Path) -> tuple[list[float], list[float]]:
    """Lê média/desvio salvos por split.py (fallback: ImageNet)."""
    path = Path(tiles_dir) / "norm_stats.json"
    if path.exists():
        stats = json.loads(path.read_text(encoding="utf-8"))
        return stats["mean"], stats["std"]
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_augmentation(train: bool) -> A.Compose:
    """Pipeline de augmentation (só geometria/fotometria; normalização à parte)."""
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                               border_mode=0, p=0.3),
        ])
    return A.Compose([])


class BuildingTileDataset(Dataset):
    """Retorna (imagem_tensor CxHxW normalizada, máscara_tensor 1xHxW em {0,1})."""

    def __init__(self, tiles_dir: str | Path, split: str,
                 mean: list[float], std: list[float], augment: bool = False):
        self.rows = load_manifest(tiles_dir, split)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.aug = build_augmentation(train=augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        with rasterio.open(row["image"]) as src:
            img = src.read().astype(np.float32) / 255.0  # (C, H, W)
        with rasterio.open(row["mask"]) as src:
            mask = src.read(1).astype(np.float32)         # (H, W)

        img_hwc = np.transpose(img, (1, 2, 0))            # -> (H, W, C) para albumentations
        augmented = self.aug(image=img_hwc, mask=mask)
        img_hwc, mask = augmented["image"], augmented["mask"]

        img_hwc = (img_hwc - self.mean) / self.std
        img_chw = np.transpose(img_hwc, (2, 0, 1)).copy()
        mask = (mask > 0.5).astype(np.float32)[None, ...]  # (1, H, W)

        return torch.from_numpy(img_chw), torch.from_numpy(mask)
