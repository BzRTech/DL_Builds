"""Construção do modelo de segmentação e loss/métricas."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(cfg_model: dict) -> nn.Module:
    """Cria um modelo SMP (Unet/DeepLabV3Plus) a partir da config."""
    arch = cfg_model["arch"]
    factory = {
        "Unet": smp.Unet,
        "DeepLabV3Plus": smp.DeepLabV3Plus,
    }
    if arch not in factory:
        raise ValueError(f"Arquitetura não suportada: {arch}. Use {list(factory)}.")
    return factory[arch](
        encoder_name=cfg_model["encoder"],
        encoder_weights=cfg_model.get("encoder_weights", "imagenet"),
        in_channels=cfg_model["in_channels"],
        classes=cfg_model["classes"],
    )


class DiceBCELoss(nn.Module):
    """Loss combinada Dice + BCE (opera sobre logits). Robusta a desbalanceamento."""

    def __init__(self, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        target_f = target.reshape(target.size(0), -1)
        inter = (probs * target_f).sum(dim=1)
        dice = 1 - (2 * inter + self.smooth) / (
            probs.sum(dim=1) + target_f.sum(dim=1) + self.smooth
        )
        dice = dice.mean()
        return (1 - self.dice_weight) * bce + self.dice_weight * dice


@torch.no_grad()
def binary_metrics(logits: torch.Tensor, target: torch.Tensor,
                   threshold: float = 0.5) -> dict[str, float]:
    """IoU, F1, precision e recall para segmentação binária (por batch)."""
    pred = (torch.sigmoid(logits) > threshold).float()
    t = (target > 0.5).float()
    tp = (pred * t).sum().item()
    fp = (pred * (1 - t)).sum().item()
    fn = ((1 - pred) * t).sum().item()
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}
