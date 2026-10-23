"""Module 6 - Compare baseline vs every adapted model.

    python -m src.compare
Writes outputs/results_methods.json (all methods) and outputs/results_adapted.json (best adapted
method by Dice on the held-out part of Dataset B; same keys as results_baseline.json).
"""
import json

from . import config as C
from .evaluate import evaluate_ckpt


def main():
    ckpts = sorted(C.OUT_DIR.glob("unet_*.pt"))
    if not ckpts:
        raise SystemExit("No checkpoints in outputs/. Train the baseline first.")
    results = {}
    for ck in ckpts:
        res, _ = evaluate_ckpt(ck)
        results[res["method"]] = res
        print(f"{res['method']:10s} A {res['dataset_A_test']['dice']:.4f} | B {res['dataset_B']['dice']:.4f} | "
              f"B-heldout {res['dataset_B_heldout']['dice']:.4f}")
    C.RESULTS_METHODS.write_text(json.dumps(results, indent=2))

    adapted = {k: v for k, v in results.items() if k != "baseline"}
    if adapted:
        best = max(adapted, key=lambda k: adapted[k]["dataset_B_heldout"]["dice"])
        C.RESULTS_ADAPTED.write_text(json.dumps(adapted[best], indent=2))
        base_h = results.get("baseline", {}).get("dataset_B_heldout", {}).get("dice")
        if base_h is not None:
            print(f"best adapted: {best}  (B-heldout Dice {base_h:.4f} -> "
                  f"{adapted[best]['dataset_B_heldout']['dice']:.4f})")


if __name__ == "__main__":
    main()
