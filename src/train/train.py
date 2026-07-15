"""Treina o modelo de segmentação de edificações.

Salva o melhor checkpoint (maior IoU de validação) em <out_dir>/best.pt e logs
para TensorBoard em <out_dir>.

Uso:
    python -m src.train.train --config configs/buildings.yaml
    tensorboard --logdir runs/buildings
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.train.dataset import BuildingTileDataset, load_norm_stats
from src.train.model import DiceBCELoss, binary_metrics, build_model
from src.utils import ensure_dir, load_config, set_seed


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    """Roda uma época. Se optimizer=None, é validação (sem grad)."""
    train = optimizer is not None
    model.train(train)
    totals = {"loss": 0.0, "iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    n = 0

    for imgs, masks in tqdm(loader, leave=False, desc="train" if train else "val"):
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(imgs)
                loss = criterion(logits, masks)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        m = binary_metrics(logits.detach().float(), masks)
        bs = imgs.size(0)
        totals["loss"] += loss.item() * bs
        for k in ("iou", "f1", "precision", "recall"):
            totals[k] += m[k] * bs
        n += bs

    return {k: v / max(n, 1) for k, v in totals.items()}


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    tcfg = cfg["train"]
    device = torch.device(tcfg["device"] if torch.cuda.is_available()
                          or tcfg["device"] == "cpu" else "cpu")
    if device.type == "cpu":
        print("[train] AVISO: rodando em CPU — treino será lento. Use GPU se possível.")

    tiles_dir = cfg["data_prep"]["tiles_dir"]
    mean = tcfg["norm_mean"] or load_norm_stats(tiles_dir)[0]
    std = tcfg["norm_std"] or load_norm_stats(tiles_dir)[1]

    train_ds = BuildingTileDataset(tiles_dir, "train", mean, std, augment=True)
    val_ds = BuildingTileDataset(tiles_dir, "val", mean, std, augment=False)
    print(f"[train] tiles: train={len(train_ds)}  val={len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                          num_workers=tcfg["num_workers"], pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False,
                        num_workers=tcfg["num_workers"], pin_memory=True)

    model = build_model(cfg["model"]).to(device)
    criterion = DiceBCELoss(dice_weight=tcfg["dice_weight"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                                  weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])
    amp = bool(tcfg["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    out_dir = ensure_dir(tcfg["out_dir"])
    writer = SummaryWriter(log_dir=str(out_dir))

    best_iou = -1.0
    epochs_no_improve = 0
    for epoch in range(1, tcfg["epochs"] + 1):
        tr = _run_epoch(model, train_dl, criterion, device, optimizer, scaler, amp)
        va = _run_epoch(model, val_dl, criterion, device)
        scheduler.step()

        for k, v in tr.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in va.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        print(f"[epoch {epoch:03d}] train_loss={tr['loss']:.4f} iou={tr['iou']:.4f} | "
              f"val_loss={va['loss']:.4f} iou={va['iou']:.4f} f1={va['f1']:.4f}")

        if va["iou"] > best_iou:
            best_iou = va["iou"]
            epochs_no_improve = 0
            ckpt = {
                "model_state": model.state_dict(),
                "config_model": cfg["model"],
                "norm_mean": mean,
                "norm_std": std,
                "val_iou": best_iou,
                "epoch": epoch,
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"          -> novo melhor modelo salvo (val IoU={best_iou:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= tcfg["early_stop_patience"]:
                print(f"[train] Early stopping em {epoch} (sem melhora há "
                      f"{epochs_no_improve} épocas).")
                break

    writer.close()
    print(f"[train] Concluído. Melhor val IoU={best_iou:.4f}. Modelo: {out_dir/'best.pt'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Treina o modelo de edificações.")
    ap.add_argument("--config", required=True, help="Config YAML.")
    args = ap.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
