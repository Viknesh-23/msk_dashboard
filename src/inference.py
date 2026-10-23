"""Module 4 - Muscle Segmentation (inference + overlay)."""
import cv2
import numpy as np
import torch

from .model import UNet
from .preprocess import normalize


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path, device=None):
    device = device or get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    model = UNet(base=ckpt.get("base", 32)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    meta = {k: ckpt.get(k) for k in ("img_size", "norm", "clahe", "base", "epoch", "val_dice",
                                     "method", "aug_level", "group_regex")}
    meta["method"] = meta["method"] or "baseline"
    return model, meta


@torch.no_grad()
def predict_prob(model, meta, img_gray: np.ndarray, device=None):
    """uint8 grayscale image -> probability map at the ORIGINAL image size."""
    device = device or next(model.parameters()).device
    h, w = img_gray.shape
    size = meta.get("img_size") or 256
    small = cv2.resize(img_gray, (size, size), interpolation=cv2.INTER_LINEAR)
    x = normalize(small, meta.get("norm") or "zscore", bool(meta.get("clahe")))
    x = torch.from_numpy(x)[None, None].to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


def overlay(img_gray, mask, color=(255, 60, 60), alpha=0.45, gt=None):
    """RGB overlay of predicted mask (red fill) and optional ground-truth contour (green)."""
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    fill = rgb.copy()
    fill[mask.astype(bool)] = color
    out = cv2.addWeighted(fill, alpha, rgb, 1 - alpha, 0)
    if gt is not None:
        cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (60, 255, 60), 1)
    return out
