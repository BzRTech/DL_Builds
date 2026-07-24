"""Dataset em memória de patches de pavimento para classificação."""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _aug(train: bool) -> A.Compose:
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        ])
    return A.Compose([])


class RoadPatchDataset(Dataset):
    def __init__(self, patches_dir: str | Path, split: str, augment: bool = False):
        d = Path(patches_dir)
        patches = np.load(d / "patches.npy")            # (N,H,W,3) uint8
        labels = np.load(d / "labels.npy")              # (N,)
        splits = np.load(d / "splits.npy")              # (N,)
        sel = splits == split
        self.patches = patches[sel]
        self.labels = labels[sel]
        if len(self.patches) == 0:
            raise RuntimeError(f"Nenhum patch no split '{split}' em {d}.")
        self.aug = _aug(augment)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        img = self.aug(image=self.patches[idx])["image"].astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1)).copy()
        return torch.from_numpy(img), int(self.labels[idx])
