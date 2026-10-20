# MSK Ultrasound Muscle Segmentation — Baseline Pipeline + Dashboard

Cross-Dataset Generalization and Domain Adaptation for Musculoskeletal Ultrasound Muscle Segmentation.

## What this code covers (≈ first 50% of the project)

| Module | File | Status |
|---|---|---|
| 1 Dataset Management | `src/dataset.py`, `src/config.py` | Done |
| 2 Preprocessing & Augmentation | `src/preprocess.py` | Done |
| 3 2D U-Net Training | `src/model.py`, `src/train.py` | Done |
| 4 Muscle Segmentation | `src/inference.py` | Done |
| 5 Cross-Dataset Eval + Generalization Gap | `src/evaluate.py`, `src/metrics.py` | Done |
| Dashboard | `app.py` | Done (shows baseline; adaptation slot ready) |
| 6 Domain Adaptation (CORAL, norm/aug) & Comparison | — | **Remaining 50%** |

## Setup
```bash
pip install -r requirements.txt
```

## Data layout
```
data/DatasetA/images/xxx.png   data/DatasetA/masks/xxx.png   (same file name)
data/DatasetB/images/yyy.png   data/DatasetB/masks/yyy.png
```
Masks can be 0/1 or 0/255. Dataset A is split 70/15/15 (train/val/test); all of Dataset B is the unseen test.

No data yet? Generate synthetic data (B is deliberately shifted) to test everything:
```bash
python scripts/make_demo_data.py
```

## Run
```bash
python -m src.train --epochs 30 --batch 8      # trains on Dataset A -> outputs/unet_baseline.pt
python -m src.evaluate                         # A-test vs B + generalization gap -> outputs/results_baseline.json
streamlit run app.py                           # dashboard
```
Options: `--norm zscore|minmax|none`, `--clahe`, `--size 256`, `--base 32` (use 16 on a weak CPU).

## Hooking in Module 6 later
- `UNet(..., return_features=True)` returns the bottleneck features for the CORAL loss.
- Write your adapted evaluation to `outputs/results_adapted.json` with the **same keys** as `results_baseline.json`
  (`dataset_B` → dice/iou/precision/recall). The dashboard's "Before vs after" table fills in automatically.
