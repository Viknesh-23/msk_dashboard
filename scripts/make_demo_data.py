"""Create SYNTHETIC ultrasound-like data so you can test the whole pipeline before real data.
Dataset B is made darker, lower-contrast, noisier and blurrier than A -> a measurable domain shift.

    python scripts/make_demo_data.py --n_a 120 --n_b 40
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402


def make_sample(rng, size=256, domain="A"):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    top = rng.uniform(0.25, 0.4) * size + rng.uniform(5, 15) * np.sin(xx / size * rng.uniform(2, 5) + rng.uniform(0, 6))
    bot = top + rng.uniform(0.2, 0.32) * size + rng.uniform(3, 10) * np.sin(xx / size * rng.uniform(2, 5) + rng.uniform(0, 6))
    mask = ((yy > top) & (yy < bot)).astype(np.uint8)

    img = rng.normal(55, 18, (size, size)).astype(np.float32)                  # background tissue
    stri = 25 * np.sin(xx / rng.uniform(6, 14) + yy / rng.uniform(25, 60))      # fibre-like striations
    img += mask * (45 + stri + rng.normal(0, 14, (size, size)))
    for edge in (top, bot):                                                      # bright fascia lines
        img += 90 * np.exp(-((yy - edge) ** 2) / 8.0)
    img *= 1 + rng.normal(0, 0.1, img.shape)                                     # speckle

    if domain == "B":                                                            # shifted domain
        img = img * 0.65 + 12
        img = cv2.GaussianBlur(img, (0, 0), 1.3) + rng.normal(0, 9, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), mask * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_a", type=int, default=120)
    ap.add_argument("--n_b", type=int, default=40)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for name, root, n in (("A", C.DATASET_A, a.n_a), ("B", C.DATASET_B, a.n_b)):
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "masks").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, m = make_sample(rng, domain=name)
            cv2.imwrite(str(root / "images" / f"{name}_{i:04d}.png"), img)
            cv2.imwrite(str(root / "masks" / f"{name}_{i:04d}.png"), m)
        print(f"Dataset {name}: {n} pairs -> {root}")


if __name__ == "__main__":
    main()
