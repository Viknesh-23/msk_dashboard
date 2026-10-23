"""Module 6 - Domain adaptation (Dataset A -> Dataset B).

    python -m src.adapt --method adabn
    python -m src.adapt --method clahe
    python -m src.adapt --method aug
    python -m src.adapt --method coral      --lambda_coral 100
    python -m src.adapt --method coral_aug  --lambda_coral 100

Methods (all start from the baseline checkpoint):
  adabn      re-estimate BatchNorm statistics on unlabeled Dataset-B images (no training, no labels)
  clahe      fine-tune on A with CLAHE contrast equalisation + per-image z-score
  aug        fine-tune on A with STRONG intensity/speckle/noise/blur/depth augmentation
  coral      fine-tune on A with Deep-CORAL: align feature covariance of A and (unlabeled) B
  coral_aug  CORAL + strong augmentation

Fair protocol: Dataset-B MASKS are never used for training. Only B's images from the 'train' pool
(70%) are used as unlabeled target data; evaluate on B 'heldout' (30%) to avoid transductive bias.
"""
import argparse
import itertools
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from .dataset import MuscleDataset
from .inference import get_device
from .metrics import DiceBCELoss, batch_scores
from .model import UNet
from .train import run_epoch, save_ckpt, seed_everything

METHODS = ["adabn", "clahe", "aug", "coral", "coral_aug"]


def coral_loss(fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
    """Deep CORAL between feature maps (B,C,H,W): spatial positions are treated as samples so the
    C x C covariance is estimated from B*H*W vectors (stable even for batch size 4-8)."""
    def cov(f):
        x = f.permute(0, 2, 3, 1).reshape(-1, f.size(1))
        x = x - x.mean(0, keepdim=True)
        return x.t() @ x / max(x.size(0) - 1, 1)
    d = fs.size(1)
    return ((cov(fs) - cov(ft)) ** 2).sum() / (4 * d * d)


@torch.no_grad()
def adabn(model, loader, device):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None          # cumulative moving average over all target batches
    model.train()
    for x, _, _ in loader:             # labels are ignored
        model(x.to(device))
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--init", default=str(C.CKPT_BASELINE))
    ap.add_argument("--a", default=str(C.DATASET_A))
    ap.add_argument("--b", default=str(C.DATASET_B))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda_coral", type=float, default=100.0)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()

    seed_everything(C.SEED)
    device = get_device()
    ck = torch.load(a.init, map_location=device)
    model = UNet(base=ck["base"]).to(device)
    model.load_state_dict(ck["model"])

    # args-like object reused by save_ckpt
    a.size, a.norm, a.base = ck["img_size"], ck["norm"], ck["base"]
    a.clahe, a.group_regex = bool(ck["clahe"]) or a.method == "clahe", ck.get("group_regex")
    aug_level = "strong" if a.method in ("aug", "coral_aug") else ck.get("aug_level", "basic")
    out = C.OUT_DIR / f"unet_{a.method}.pt"

    kw = dict(img_size=a.size, norm=a.norm, clahe=a.clahe, seed=C.SEED, group_regex=a.group_regex)
    tgt_ds = MuscleDataset(a.b, "train", aug_level="none", **kw)     # unlabeled target pool (masks never read for loss)
    if len(tgt_ds) == 0:
        raise SystemExit("Dataset B target pool is empty.")
    tgt_loader = DataLoader(tgt_ds, a.batch, shuffle=True, num_workers=a.workers)

    if a.method == "adabn":
        adabn(model, tgt_loader, device)
        save_ckpt(out, model, a, 0, ck.get("val_dice", 0.0), a.method, aug_level)
        print(f"AdaBN done ({len(tgt_ds)} unlabeled target images) -> {out}")
        return

    src_ds = MuscleDataset(a.a, "train", aug_level=aug_level, **kw)
    val_ds = MuscleDataset(a.a, "val", aug_level="none", **kw)
    src_loader = DataLoader(src_ds, a.batch, shuffle=True, num_workers=a.workers, drop_last=len(src_ds) > a.batch)
    val_loader = DataLoader(val_ds, a.batch, shuffle=False, num_workers=a.workers)
    use_coral = a.method.startswith("coral")
    tgt_iter = itertools.cycle(tgt_loader)
    loss_fn = DiceBCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    print(f"method={a.method} device={device} src={len(src_ds)} tgt(unlabeled)={len(tgt_ds)} aug={aug_level}")

    hist = []
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        tot_seg = tot_coral = tot_dice = n = 0.0
        for xs, ys, _ in src_loader:
            xs, ys = xs.to(device), ys.to(device)
            logits, fs = model(xs, return_features=True)
            seg = loss_fn(logits, ys)
            loss, c = seg, torch.zeros((), device=device)
            if use_coral:
                xt = next(tgt_iter)[0].to(device)
                _, ft = model(xt, return_features=True)
                c = coral_loss(fs, ft)
                loss = seg + a.lambda_coral * c
            opt.zero_grad()
            loss.backward()
            opt.step()
            b = xs.size(0)
            tot_seg += seg.item() * b
            tot_coral += c.item() * b
            tot_dice += batch_scores(logits.detach(), ys)["dice"].sum().item()
            n += b
        vl, vd = run_epoch(model, val_loader, device, loss_fn) if len(val_ds) else (0.0, 0.0)
        hist.append({"epoch": ep, "train_loss": tot_seg / n, "train_dice": tot_dice / n, "val_loss": vl,
                     "val_dice": vd, "coral": tot_coral / n})
        print(f"ep {ep:03d}  seg {tot_seg / n:.4f}  coral {tot_coral / n:.5f}  dice {tot_dice / n:.4f}/{vd:.4f}  "
              f"({time.time() - t0:.1f}s)")
        (C.OUT_DIR / f"history_{a.method}.json").write_text(json.dumps(hist, indent=2))

    # keep the FINAL epoch: selecting by Dataset-A validation would favour epoch 1 (before adaptation acts)
    save_ckpt(out, model, a, a.epochs, hist[-1]["val_dice"], a.method, aug_level)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
