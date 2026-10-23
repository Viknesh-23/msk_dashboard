"""Run with:  pytest -q"""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.adapt import coral_loss                                    # noqa: E402
from src.dataset import split_pairs                                 # noqa: E402
from src.metrics import batch_scores, generalization_gap, scores_np  # noqa: E402
from src.model import UNet                                          # noqa: E402
from src.preprocess import augment, normalize                       # noqa: E402


def _pairs(names):
    return [(Path(f"{n}.png"), Path(f"{n}.png")) for n in names]


def test_unet_shapes_and_features():
    m = UNet(base=4).eval()
    x = torch.randn(2, 1, 64, 64)
    logits, feat = m(x, return_features=True)
    assert logits.shape == (2, 1, 64, 64) and feat.shape == (2, 64, 4, 4)


def test_metrics_perfect_and_disjoint():
    g = np.zeros((8, 8), np.uint8); g[2:6, 2:6] = 1
    assert scores_np(g, g)["dice"] == pytest.approx(1.0, abs=1e-4)
    assert scores_np(1 - g, g)["dice"] < 0.01
    logits = torch.where(torch.tensor(g).bool(), 10.0, -10.0)[None, None]
    assert batch_scores(logits, torch.tensor(g, dtype=torch.float32)[None, None])["dice"][0] > 0.999


def test_generalization_gap():
    gap = generalization_gap({"dice": 0.9, "iou": 0.8}, {"dice": 0.6, "iou": 0.5})
    assert gap["dice_gap"] == pytest.approx(0.3) and gap["dice_gap_pct"] == pytest.approx(33.33, abs=0.1)


def test_split_is_disjoint_complete_and_deterministic():
    pairs = _pairs([f"img{i:03d}" for i in range(40)])
    parts = {s: {p[0].name for p in split_pairs(pairs, s)} for s in ("train", "val", "test")}
    assert parts["train"].isdisjoint(parts["val"]) and parts["train"].isdisjoint(parts["test"])
    assert parts["val"].isdisjoint(parts["test"])
    assert sum(len(v) for v in parts.values()) == 40
    assert parts == {s: {p[0].name for p in split_pairs(pairs, s)} for s in ("train", "val", "test")}


def test_group_split_keeps_patients_together():
    pairs = _pairs([f"P{p:02d}_f{f}" for p in range(12) for f in range(5)])
    where = {}
    for s in ("train", "val", "test"):
        for ip, _ in split_pairs(pairs, s, group_regex=r"^(P\d+)_"):
            where.setdefault(ip.stem.split("_")[0], set()).add(s)
    assert all(len(v) == 1 for v in where.values()), "a patient leaked across splits"


def test_heldout_is_val_plus_test():
    pairs = _pairs([f"x{i:02d}" for i in range(30)])
    h = {p[0].name for p in split_pairs(pairs, "heldout")}
    assert h == {p[0].name for p in split_pairs(pairs, "val")} | {p[0].name for p in split_pairs(pairs, "test")}


def test_augmentation_levels_and_mask_stays_binary():
    rng = np.random.default_rng(0)
    img = (np.random.default_rng(1).random((64, 64)) * 255).astype(np.uint8)
    m = np.zeros((64, 64), np.uint8); m[20:40, 20:40] = 1
    i0, m0 = augment(img, m, rng, "none");  assert (i0 == img).all() and (m0 == m).all()
    for lvl in ("basic", "strong"):
        i, mm = augment(img, m, rng, lvl)
        assert i.dtype == np.uint8 and set(np.unique(mm)) <= {0, 1}


def test_normalize_zscore():
    x = normalize((np.random.default_rng(0).random((32, 32)) * 255).astype(np.uint8))
    assert abs(x.mean()) < 1e-4 and abs(x.std() - 1) < 1e-3


def test_coral_zero_for_identical_and_positive_otherwise():
    f = torch.randn(4, 16, 8, 8)
    assert coral_loss(f, f).item() == pytest.approx(0.0, abs=1e-9)
    assert coral_loss(f, f * 3 + 1).item() > 0
