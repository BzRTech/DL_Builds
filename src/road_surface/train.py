"""Treina o classificador de pavimento (3 classes) sobre os patches das vias.

Uso:
    python -m src.road_surface.train --config configs/road_surface.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from src.road_surface.dataset import RoadPatchDataset
from src.utils import ensure_dir, load_config, set_seed


def build_classifier(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    weights = "IMAGENET1K_V1" if pretrained else None
    factory = {"resnet18": torchvision.models.resnet18,
               "resnet34": torchvision.models.resnet34}
    model = factory[arch](weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _class_weights(patches_dir: str, num_classes: int) -> torch.Tensor:
    labels = np.load(Path(patches_dir) / "labels.npy")
    splits = np.load(Path(patches_dir) / "splits.npy")
    tr = labels[splits == "train"]
    counts = np.bincount(tr, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1
    w = counts.sum() / (num_classes * counts)     # inverso da frequência
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def _evaluate(model, loader, device, num_classes):
    model.eval()
    correct = total = 0
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for x, y in loader:
        x = x.to(device)
        pred = model(x).argmax(1).cpu().numpy()
        y = y.numpy()
        correct += int((pred == y).sum()); total += len(y)
        for t, p in zip(y, pred):
            conf[t, p] += 1
    acc = correct / max(total, 1)
    recalls = [conf[i, i] / max(conf[i].sum(), 1) for i in range(num_classes)]
    return acc, recalls, conf


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    tcfg = cfg["train"]
    classes = cfg["classes"]
    num_classes = len(classes)
    patches_dir = cfg["data_prep"]["patches_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() and
                          tcfg["device"] == "cuda" else "cpu")

    tr = RoadPatchDataset(patches_dir, "train", augment=True)
    va = RoadPatchDataset(patches_dir, "val", augment=False)
    print(f"[road_train] train={len(tr)} val={len(va)} classes={classes}")
    tr_dl = DataLoader(tr, batch_size=tcfg["batch_size"], shuffle=True,
                       num_workers=tcfg["num_workers"], pin_memory=True, drop_last=True)
    va_dl = DataLoader(va, batch_size=tcfg["batch_size"], shuffle=False,
                       num_workers=tcfg["num_workers"], pin_memory=True)

    model = build_classifier(cfg["model"]["arch"], num_classes,
                             cfg["model"]["pretrained"]).to(device)
    weights = _class_weights(patches_dir, num_classes).to(device)
    print(f"[road_train] pesos de classe (inv. freq.): {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                                  weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])

    out_dir = ensure_dir(tcfg["out_dir"])
    best_acc, no_improve = -1.0, 0
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        acc, recalls, _ = _evaluate(model, va_dl, device, num_classes)
        rec_str = " ".join(f"{c[:6]}={r:.2f}" for c, r in zip(classes, recalls))
        print(f"[epoch {epoch:03d}] val_acc={acc:.4f} | recall: {rec_str}")

        if acc > best_acc:
            best_acc, no_improve = acc, 0
            torch.save({"model_state": model.state_dict(),
                        "arch": cfg["model"]["arch"], "classes": classes,
                        "input_size": json.loads(
                            (Path(patches_dir) / "meta.json").read_text(
                                encoding="utf-8"))["input_size"],
                        "val_acc": best_acc, "epoch": epoch}, out_dir / "best.pt")
            print(f"          -> novo melhor modelo salvo (val_acc={best_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= tcfg["early_stop_patience"]:
                print(f"[road_train] Early stopping em {epoch}.")
                break
    print(f"[road_train] Concluído. Melhor val_acc={best_acc:.4f}. Modelo: {out_dir/'best.pt'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Treina o classificador de pavimento.")
    ap.add_argument("--config", required=True, help="Config YAML.")
    args = ap.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
