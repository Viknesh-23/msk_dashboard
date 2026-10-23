"""Module 5 - Cross-dataset evaluation + Generalization Gap.

    python -m src.evaluate                       # baseline -> outputs/results_baseline.json
    python -m src.evaluate --ckpt outputs/unet_coral.pt --out outputs/x.json

Reports three numbers per model:
  dataset_A_test    internal test split of Dataset A
  dataset_B         ALL of Dataset B (unseen labels)
  dataset_B_heldout the part of B NOT used as unlabeled adaptation pool (fair before/after comparison)
"""
import argparse
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config as C
from .dataset import MuscleDataset
from .inference import load_model
from .metrics import batch_scores, generalization_gap


@torch.no_grad()
def evaluate_dataset(model, ds, device, thr=C.THRESHOLD, batch=8):
    model.eval()
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


def evaluate_ckpt(ckpt, a_root=C.DATASET_A, b_root=C.DATASET_B, thr=C.THRESHOLD):
    model, meta = load_model(ckpt)
    device = next(model.parameters()).device
    kw = dict(img_size=meta["img_size"], norm=meta["norm"], clahe=bool(meta["clahe"]), aug_level="none",
              seed=C.SEED, group_regex=meta.get("group_regex"))
    ds_a = MuscleDataset(a_root, "test", **kw)
    ds_b = MuscleDataset(b_root, "all", **kw)
    ds_bh = MuscleDataset(b_root, "heldout", **kw)
    if len(ds_a) == 0 or len(ds_b) == 0:
        raise SystemExit("Dataset A test split or Dataset B is empty - check the data folders.")
    sa, dfa = evaluate_dataset(model, ds_a, device, thr)
    sb, dfb = evaluate_dataset(model, ds_b, device, thr)
    sbh, _ = evaluate_dataset(model, ds_bh, device, thr)
    result = {"checkpoint": str(ckpt), "method": meta["method"], "threshold": thr,
              "dataset_A_test": sa, "dataset_B": sb, "dataset_B_heldout": sbh,
              "gap": generalization_gap(sa, sb), "gap_heldout": generalization_gap(sa, sbh)}
    return result, pd.concat([dfa.assign(dataset="A"), dfb.assign(dataset="B")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(C.CKPT_BASELINE))
    ap.add_argument("--a", default=str(C.DATASET_A))
    ap.add_argument("--b", default=str(C.DATASET_B))
    ap.add_argument("--out", default=str(C.RESULTS_BASELINE))
    ap.add_argument("--csv", default=str(C.OUT_DIR / "per_image_baseline.csv"))
    ap.add_argument("--thr", type=float, default=C.THRESHOLD)
    a = ap.parse_args()

    res, per_image = evaluate_ckpt(a.ckpt, a.a, a.b, a.thr)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    per_image.to_csv(a.csv, index=False)

    sa, sb, g = res["dataset_A_test"], res["dataset_B"], res["gap"]
    print(f"Dataset A (test)  : Dice {sa['dice']:.4f}  IoU {sa['iou']:.4f}  (n={sa['n_images']})")
    print(f"Dataset B (unseen): Dice {sb['dice']:.4f}  IoU {sb['iou']:.4f}  (n={sb['n_images']})")
    print(f"Generalization gap: Dice {g['dice_gap']:.4f} ({g['dice_gap_pct']:.1f}%)")


if __name__ == "__main__":
    main()
