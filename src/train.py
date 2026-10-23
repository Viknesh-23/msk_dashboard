"""Module 3 - train the baseline 2D U-Net on Dataset A.

    python -m src.train --epochs 30 --batch 8

The baseline deliberately uses only GEOMETRIC augmentation (--aug_level basic) so that the
intensity/noise augmentation in Module 6 is a genuine, measurable adaptation method.
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config as C
from .dataset import MuscleDataset
from .inference import get_device
from .metrics import DiceBCELoss, batch_scores
from .model import UNet


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_epoch(model, loader, device, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    tot_loss, tot_dice, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            b = x.size(0)
            tot_loss += loss.item() * b
            tot_dice += batch_scores(logits, y)["dice"].sum().item()
            n += b
    return tot_loss / max(n, 1), tot_dice / max(n, 1)


def save_ckpt(path, model, a, epoch, val_dice, method="baseline", aug_level="basic"):
    torch.save({"model": model.state_dict(), "img_size": a.size, "norm": a.norm, "clahe": a.clahe,
                "base": a.base, "epoch": epoch, "val_dice": val_dice, "method": method,
                "aug_level": aug_level, "group_regex": a.group_regex}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DATASET_A))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=C.IMG_SIZE)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--norm", default="zscore", choices=["zscore", "minmax", "none"])
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--aug_level", default="basic", choices=["none", "basic", "strong"])
    ap.add_argument("--group_regex", default=None,
                    help=r"regex whose group(1) is the patient/video id, e.g. '^(P\d+)_' (prevents split leakage)")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    if a.size % 16:
        raise SystemExit("--size must be divisible by 16 (4 pooling levels).")

    seed_everything(C.SEED)
    device = get_device()
    kw = dict(img_size=a.size, norm=a.norm, clahe=a.clahe, seed=C.SEED, group_regex=a.group_regex)
    train_ds = MuscleDataset(a.data, "train", aug_level=a.aug_level, **kw)
    val_ds = MuscleDataset(a.data, "val", aug_level="none", **kw)
    if len(train_ds) == 0:
        raise SystemExit(f"No image/mask pairs found in {a.data}. See README for the folder layout.")
    print(f"device={device}  train={len(train_ds)}  val={len(val_ds)}  aug={a.aug_level}")

    tl = DataLoader(train_ds, a.batch, shuffle=True, num_workers=a.workers, drop_last=len(train_ds) > a.batch)
    vl = DataLoader(val_ds, a.batch, shuffle=False, num_workers=a.workers)

    model = UNet(base=a.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loss_fn = DiceBCELoss()

    hist, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        trl, trd = run_epoch(model, tl, device, loss_fn, opt)
        vll, vld = run_epoch(model, vl, device, loss_fn) if len(val_ds) else (trl, trd)
        sched.step(vll)
        hist.append({"epoch": ep, "train_loss": trl, "train_dice": trd, "val_loss": vll, "val_dice": vld,
                     "lr": opt.param_groups[0]["lr"]})
        print(f"ep {ep:03d}  loss {trl:.4f}/{vll:.4f}  dice {trd:.4f}/{vld:.4f}  ({time.time() - t0:.1f}s)")
        if vld > best:
            best = vld
            save_ckpt(C.CKPT_BASELINE, model, a, ep, vld, "baseline", a.aug_level)
        C.HISTORY_BASELINE.write_text(json.dumps(hist, indent=2))
    print(f"best val dice {best:.4f} -> {C.CKPT_BASELINE}")


if __name__ == "__main__":
    main()
