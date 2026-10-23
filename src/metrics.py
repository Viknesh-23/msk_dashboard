"""Evaluation metrics: Dice, IoU, Precision, Recall, Generalization Gap."""
import numpy as np
import torch
import torch.nn as nn

EPS = 1e-6


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        inter = (p * target).sum((1, 2, 3))
        dice = (2 * inter + EPS) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + EPS)
        return self.bce(logits, target) + (1 - dice).mean()


@torch.no_grad()
def batch_scores(logits, target, thr=0.5):
    """Per-sample scores for a batch (B,1,H,W). Returns dict of (B,) tensors."""
    pred = (torch.sigmoid(logits) > thr).float()
    tp = (pred * target).sum((1, 2, 3))
    fp = (pred * (1 - target)).sum((1, 2, 3))
    fn = ((1 - pred) * target).sum((1, 2, 3))
    return {
        "dice": (2 * tp + EPS) / (2 * tp + fp + fn + EPS),
        "iou": (tp + EPS) / (tp + fp + fn + EPS),
        "precision": (tp + EPS) / (tp + fp + EPS),
        "recall": (tp + EPS) / (tp + fn + EPS),
    }


def scores_np(pred_bin: np.ndarray, gt_bin: np.ndarray) -> dict:
    """Same metrics for two binary numpy masks (used by the dashboard)."""
    p, g = pred_bin.astype(bool), gt_bin.astype(bool)
    tp, fp, fn = (p & g).sum(), (p & ~g).sum(), (~p & g).sum()
    return {"dice": float((2 * tp + EPS) / (2 * tp + fp + fn + EPS)),
            "iou": float((tp + EPS) / (tp + fp + fn + EPS)),
            "precision": float((tp + EPS) / (tp + fp + EPS)),
            "recall": float((tp + EPS) / (tp + fn + EPS))}


def generalization_gap(src: dict, tgt: dict) -> dict:
    """Gap = score on Dataset A (internal test) - score on Dataset B (unseen)."""
    out = {}
    for k in ("dice", "iou"):
        gap = src[k] - tgt[k]
        out[f"{k}_gap"] = gap
        out[f"{k}_gap_pct"] = 100 * gap / max(src[k], EPS)
    return out
