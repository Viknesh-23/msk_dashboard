"""Module 1 - Dataset Management."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import augment, load_image, load_mask, normalize, resize_pair

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_pairs(root):
    """Return sorted [(image_path, mask_path)] matched by file stem."""
    root = Path(root)
    img_dir, msk_dir = root / "images", root / "masks"
    if not img_dir.exists() or not msk_dir.exists():
        return []
    masks = {p.stem: p for p in msk_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    return [(p, masks[p.stem]) for p in sorted(img_dir.iterdir())
            if p.suffix.lower() in IMG_EXT and p.stem in masks]


def split_pairs(pairs, split, val_frac=0.15, test_frac=0.15, seed=42):
    """Deterministic train/val/test split. split='all' returns everything (use for Dataset B)."""
    if split == "all":
        return list(pairs)
    idx = np.random.default_rng(seed).permutation(len(pairs))
    n_val, n_test = int(len(pairs) * val_frac), int(len(pairs) * test_frac)
    parts = {"test": idx[:n_test], "val": idx[n_test:n_test + n_val], "train": idx[n_test + n_val:]}
    return [pairs[i] for i in sorted(parts[split])]


class MuscleDataset(Dataset):
    def __init__(self, root, split="train", img_size=256, norm="zscore", clahe=False,
                 augment_data=None, seed=42):
        self.pairs = split_pairs(list_pairs(root), split, seed=seed)
        self.img_size, self.norm, self.clahe = img_size, norm, clahe
        self.augment = (split == "train") if augment_data is None else augment_data
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img, mask = resize_pair(load_image(ip), load_mask(mp), self.img_size)
        if self.augment:
            img, mask = augment(img, mask, self.rng)
        x = normalize(img, self.norm, self.clahe)
        return torch.from_numpy(x)[None], torch.from_numpy(mask.astype(np.float32))[None], ip.name


def dataset_summary(root):
    """Quick stats for the dashboard."""
    pairs = list_pairs(root)
    if not pairs:
        return None
    sample = pairs[:50]
    imgs = [load_image(p[0]) for p in sample]
    masks = [load_mask(p[1]) for p in sample]
    return {
        "n_pairs": len(pairs),
        "mean_intensity": float(np.mean([im.mean() for im in imgs])),
        "std_intensity": float(np.mean([im.std() for im in imgs])),
        "mean_mask_coverage_%": float(np.mean([m.mean() for m in masks]) * 100),
        "example_size": f"{imgs[0].shape[1]}x{imgs[0].shape[0]}",
    }
