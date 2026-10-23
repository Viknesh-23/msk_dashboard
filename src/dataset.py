"""Module 1 - Dataset Management."""
import re
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


def _group_key(stem, group_regex):
    if group_regex:
        m = re.search(group_regex, stem)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    return stem


def split_pairs(pairs, split, val_frac=0.15, test_frac=0.15, seed=42, group_regex=None):
    """Deterministic split.  split in {'train','val','test','heldout','all'}.

    * 'heldout' = val + test (used for Dataset B: train part is the unlabeled adaptation pool,
      heldout part is never used for adaptation -> fair before/after comparison).
    * group_regex (e.g. r'^(P\\d+)_'): images with the same captured group (patient / video)
      always land in the same split, preventing train/test leakage between near-identical frames.
    """
    if split == "all":
        return list(pairs)
    keys = [_group_key(p[0].stem, group_regex) for p in pairs]
    groups = sorted(set(keys))
    order = np.random.default_rng(seed).permutation(len(groups))
    shuffled = [groups[i] for i in order]
    n = len(shuffled)
    n_test = max(1, round(n * test_frac)) if n >= 3 else 0
    n_val = max(1, round(n * val_frac)) if n >= 3 else 0
    test_g = set(shuffled[:n_test])
    val_g = set(shuffled[n_test:n_test + n_val])

    def part(k):
        return "test" if k in test_g else "val" if k in val_g else "train"

    wanted = {"val", "test"} if split == "heldout" else {split}
    return [pr for pr, k in zip(pairs, keys) if part(k) in wanted]


class MuscleDataset(Dataset):
    """aug_level: 'none' | 'basic' | 'strong'. Evaluation splits should always use 'none'."""

    def __init__(self, root, split="train", img_size=256, norm="zscore", clahe=False,
                 aug_level="none", seed=42, group_regex=None):
        self.pairs = split_pairs(list_pairs(root), split, seed=seed, group_regex=group_regex)
        self.img_size, self.norm, self.clahe, self.aug_level = img_size, norm, clahe, aug_level
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img, mask = resize_pair(load_image(ip), load_mask(mp), self.img_size)
        img, mask = augment(img, mask, self.rng, self.aug_level)
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
