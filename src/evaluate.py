"""Module 5 - Cross-dataset evaluation + Generalization Gap.

    python -m src.evaluate
Writes outputs/results_baseline.json and outputs/per_image_baseline.csv
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config as C
from .dataset import MuscleDataset
from .inference import load_model
from .metrics import batch_scores, generalization_gap


@torch.no_grad()
def evaluate_dataset(model, ds, device, thr=C.THRESHOLD, batch=8):
    rows = []
    for x, y, names in DataLoader(ds, batch, shuffle=False):
        s = batch_scores(model(x.to(device)), y.to(device), thr)
        for i, nm in enumerate(names):
            rows.append({"image": nm, **{k: float(v[i]) for k, v in s.items()}})
    df = pd.DataFrame(rows)
    summary = {k: float(df[k].mean()) for k in ("dice", "iou", "precision", "recall")}
    summary.update({f"{k}_std": float(df[k].std(ddof=0)) for k in ("dice", "iou")})
    summary["n_images"] = len(df)
    return summary, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(C.CKPT_BASELINE))
    ap.add_argument("--a", default=str(C.DATASET_A))
    ap.add_argument("--b", default=str(C.DATASET_B))
    ap.add_argument("--out", default=str(C.RESULTS_BASELINE))
    ap.add_argument("--csv", default=str(C.OUT_DIR / "per_image_baseline.csv"))
    ap.add_argument("--thr", type=float, default=C.THRESHOLD)
    a = ap.parse_args()

    model, meta = load_model(a.ckpt)
    device = next(model.parameters()).device
    kw = dict(img_size=meta["img_size"], norm=meta["norm"], clahe=bool(meta["clahe"]), augment_data=False)
    ds_a = MuscleDataset(a.a, "test", seed=C.SEED, **kw)   # internal test split of Dataset A
    ds_b = MuscleDataset(a.b, "all", **kw)                 # everything in Dataset B is unseen
    if len(ds_a) == 0 or len(ds_b) == 0:
        raise SystemExit("Dataset A test split or Dataset B is empty - check the data folders.")

    sa, dfa = evaluate_dataset(model, ds_a, device, a.thr)
    sb, dfb = evaluate_dataset(model, ds_b, device, a.thr)
    gap = generalization_gap(sa, sb)

    result = {"checkpoint": a.ckpt, "threshold": a.thr, "dataset_A_test": sa, "dataset_B": sb, "gap": gap}
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    pd.concat([dfa.assign(dataset="A"), dfb.assign(dataset="B")]).to_csv(a.csv, index=False)

    print(f"Dataset A (test):  Dice {sa['dice']:.4f}  IoU {sa['iou']:.4f}  (n={sa['n_images']})")
    print(f"Dataset B (unseen): Dice {sb['dice']:.4f}  IoU {sb['iou']:.4f}  (n={sb['n_images']})")
    print(f"Generalization gap: Dice {gap['dice_gap']:.4f} ({gap['dice_gap_pct']:.1f}%)")


if __name__ == "__main__":
    main()
