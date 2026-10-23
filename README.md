# MSK Ultrasound Muscle Segmentation — Cross-Dataset Generalization & Domain Adaptation

2D U-Net trained on Dataset A, tested on an unseen Dataset B, then domain adaptation (Module 6) and a Streamlit dashboard.

## Project structure
| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (Overview, Dataset Explorer, Segmentation Demo with 2-model comparison, Training, Cross-Dataset Evaluation + before/after) |
| `src/config.py` | Paths and defaults |
| `src/dataset.py` | Module 1 - pairing, leak-free grouped split, `MuscleDataset` |
| `src/preprocess.py` | Module 2 - normalisation, CLAHE, augmentation levels `none/basic/strong` |
| `src/model.py` | Module 3 - 2D U-Net (exposes bottleneck features for CORAL) |
| `src/train.py` | Module 3 - baseline training (geometric augmentation only) |
| `src/inference.py` | Module 4 - load model, predict, overlay |
| `src/metrics.py` | Dice / IoU / Precision / Recall / Generalization Gap, loss |
| `src/evaluate.py` | Module 5 - A-test vs B evaluation |
| `src/adapt.py` | Module 6 - `adabn`, `clahe`, `aug`, `coral`, `coral_aug` |
| `src/compare.py` | Module 6 - baseline vs all methods |
| `scripts/make_demo_data.py` | Synthetic A/B data with a real domain shift, to test the pipeline |
| `tests/test_core.py` | 9 unit tests (`pytest -q`) |

## Setup
```bash
pip install -r requirements.txt
```

## Data layout
```
data/DatasetA/images/xxx.png   data/DatasetA/masks/xxx.png   (same file name)
data/DatasetB/images/yyy.png   data/DatasetB/masks/yyy.png
```
Masks 0/1 or 0/255. Dataset A: 70/15/15 train/val/test. Dataset B: 70% = unlabeled adaptation pool,
30% = held-out evaluation set. **If several frames come from one patient/video, pass `--group_regex`**
(e.g. `--group_regex "^(P\d+)_"`) so a patient never appears in both train and test.

## Run
```bash
python scripts/make_demo_data.py                # optional synthetic data
python -m src.train --epochs 30 --batch 8       # baseline (geometric aug only)
python -m src.evaluate                          # gap A vs B -> results_baseline.json

python -m src.adapt --method adabn
python -m src.adapt --method clahe
python -m src.adapt --method aug
python -m src.adapt --method coral      --lambda_coral 100
python -m src.adapt --method coral_aug  --lambda_coral 100
python -m src.compare                           # -> results_methods.json + results_adapted.json

streamlit run app.py
pytest -q
```
Weak CPU? `--base 16 --size 128`. Size must be divisible by 16.

## Experimental protocol (what to say in the report)
* Baseline uses **no** intensity augmentation, so strong augmentation is a genuine adaptation method.
* Dataset-B **masks are never used for training**; only B images (unlabeled) for `adabn` / `coral*`.
* Report B on the **held-out** split for before/after claims.
* Adapted checkpoints keep the final epoch (not best-on-A-val, which would favour the un-adapted epoch 1).
* Tune `--lambda_coral` (try 10, 100, 1000) — the raw CORAL term is small.
* Synthetic-data numbers are only for testing the code; do not report them as results.
