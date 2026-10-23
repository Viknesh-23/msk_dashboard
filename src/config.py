"""Central paths and defaults for the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Expected layout:  data/DatasetA/images/*.png   data/DatasetA/masks/*.png (same file names)
DATASET_A = DATA_DIR / "DatasetA"
DATASET_B = DATA_DIR / "DatasetB"

CKPT_BASELINE = OUT_DIR / "unet_baseline.pt"
HISTORY_BASELINE = OUT_DIR / "history_baseline.json"
RESULTS_BASELINE = OUT_DIR / "results_baseline.json"
RESULTS_METHODS = OUT_DIR / "results_methods.json"     # every method side by side (Module 6)
RESULTS_ADAPTED = OUT_DIR / "results_adapted.json"     # best adapted method, same keys as baseline

IMG_SIZE = 256
SEED = 42
THRESHOLD = 0.5
