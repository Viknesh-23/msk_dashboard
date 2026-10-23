"""Streamlit dashboard - Cross-Dataset Generalization & Domain Adaptation for MSK Ultrasound Segmentation.

    streamlit run app.py
"""
import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src.dataset import dataset_summary, list_pairs
from src.metrics import scores_np
from src.preprocess import load_image, load_mask

st.set_page_config(page_title="MSK Ultrasound Segmentation", page_icon="🩻", layout="wide")


# ---------- helpers ----------
@st.cache_resource(show_spinner="Loading model...")
def get_model(path, mtime):
    from src.inference import load_model
    return load_model(path)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def model_ready():
    return C.CKPT_BASELINE.exists()


def hist_curve(paths, bins=32, n=40):
    acc = np.zeros(bins)
    for p in paths[:n]:
        acc += np.histogram(load_image(p), bins=bins, range=(0, 256))[0]
    return acc / max(acc.sum(), 1)


# ---------- sidebar ----------
st.sidebar.title("🩻 MSK-US Segmentation")
page = st.sidebar.radio("Module", ["Overview", "Dataset Explorer", "Segmentation Demo",
                                   "Training", "Cross-Dataset Evaluation"])
st.sidebar.caption("Project Phase-1 · Batch 20 · Dept. of IT")

# ---------- pages ----------
if page == "Overview":
    st.title("Cross-Dataset Generalization & Domain Adaptation")
    st.subheader("Musculoskeletal Ultrasound Muscle Segmentation using Deep Learning")
    st.markdown("**Pipeline:** Dataset A → Preprocessing → Train U-Net → Test A → Test B → "
                "Generalization Gap → Domain Adaptation → Test B again → Compare")

    res = read_json(C.RESULTS_BASELINE)
    c1, c2, c3, c4 = st.columns(4)
    sa, sb = dataset_summary(C.DATASET_A), dataset_summary(C.DATASET_B)
    c1.metric("Dataset A pairs", sa["n_pairs"] if sa else "—")
    c2.metric("Dataset B pairs", sb["n_pairs"] if sb else "—")
    c3.metric("Dice on A (test)", f"{res['dataset_A_test']['dice']:.3f}" if res else "—")
    c4.metric("Dice on B (unseen)", f"{res['dataset_B']['dice']:.3f}" if res else "—",
              delta=f"-{res['gap']['dice_gap']:.3f}" if res else None, delta_color="inverse")

    st.markdown("#### Module status")
    status = pd.DataFrame([
        ("1  Dataset Management", "Done" if sa else "Needs data"),
        ("2  Preprocessing & Augmentation", "Done"),
        ("3  2D U-Net Training", "Done" if model_ready() else "Run `python -m src.train`"),
        ("4  Muscle Segmentation", "Done" if model_ready() else "Needs trained model"),
        ("5  Cross-Dataset Evaluation", "Done" if res else "Run `python -m src.evaluate`"),
        ("6  Domain Adaptation & Comparison", "Done" if read_json(C.RESULTS_METHODS) else "Run `python -m src.adapt` then `python -m src.compare`"),
    ], columns=["Module", "Status"])
    st.dataframe(status, hide_index=True, width="stretch")

elif page == "Dataset Explorer":
    st.header("Dataset Explorer")
    pa, pb = list_pairs(C.DATASET_A), list_pairs(C.DATASET_B)
    if not pa and not pb:
        st.warning("No data found. Put images/masks under `data/DatasetA` and `data/DatasetB`, "
                   "or run `python scripts/make_demo_data.py`.")
        st.stop()

    cols = st.columns(2)
    for col, name, root in ((cols[0], "A", C.DATASET_A), (cols[1], "B", C.DATASET_B)):
        s = dataset_summary(root)
        with col:
            st.subheader(f"Dataset {name}")
            if s:
                st.json(s)
            else:
                st.info("not found")

    st.markdown("#### Intensity distribution (domain shift check)")
    chart = {}
    if pa: chart["Dataset A"] = hist_curve([p[0] for p in pa])
    if pb: chart["Dataset B"] = hist_curve([p[0] for p in pb])
    st.line_chart(pd.DataFrame(chart, index=np.arange(0, 256, 8)))
    st.caption("x: pixel intensity (0-255) · y: fraction of pixels")
    st.caption("Different curves between A and B indicate brightness/contrast differences — the source of the generalization gap.")

    st.markdown("#### Browse samples")
    ds_name = st.radio("Dataset", [n for n, p in (("A", pa), ("B", pb)) if p], horizontal=True)
    pairs = pa if ds_name == "A" else pb
    idx = st.slider("Sample", 0, len(pairs) - 1, 0)
    img, m = load_image(pairs[idx][0]), load_mask(pairs[idx][1])
    from src.inference import overlay
    c1, c2, c3 = st.columns(3)
    c1.image(img, caption=pairs[idx][0].name, clamp=True, width="stretch")
    c2.image(m * 255, caption="Ground-truth mask", clamp=True, width="stretch")
    c3.image(overlay(img, m), caption="Overlay", width="stretch")

elif page == "Segmentation Demo":
    st.header("Muscle Segmentation Demo")
    if not model_ready():
        st.warning("No trained model yet. Run `python -m src.train` first.")
        st.stop()
    from src.inference import overlay, predict_prob
    ckpts = sorted(C.OUT_DIR.glob("unet_*.pt"), key=lambda p: (p.stem != "unet_baseline", p.stem))
    names = [p.stem.replace("unet_", "") for p in ckpts]
    chosen = st.multiselect("Model(s) - pick two to compare before vs after adaptation", names,
                            default=names[:1], max_selections=2)
    models = {n: get_model(str(p), p.stat().st_mtime) for n, p in zip(names, ckpts) if n in chosen}

    src = st.radio("Input", ["Upload image", "Dataset A sample", "Dataset B sample"], horizontal=True)
    thr = st.slider("Threshold", 0.1, 0.9, C.THRESHOLD, 0.05)
    img, gt, label = None, None, ""
    if src == "Upload image":
        up = st.file_uploader("Ultrasound image", type=["png", "jpg", "jpeg", "bmp", "tif"])
        if up:
            img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            label = up.name
    else:
        pairs = list_pairs(C.DATASET_A if "A" in src else C.DATASET_B)
        if pairs:
            i = st.slider("Sample", 0, len(pairs) - 1, 0)
            img, gt, label = load_image(pairs[i][0]), load_mask(pairs[i][1]), pairs[i][0].name
        else:
            st.info("No samples found for that dataset.")

    if img is not None and not models:
        st.info("Select at least one model.")
    elif img is not None:
        cols = st.columns(len(models) + 1)
        cols[0].image(img, caption=f"Input: {label}", clamp=True, width="stretch")
        if gt is not None:
            cols[0].image(gt * 255, caption="Ground truth", clamp=True, width="stretch")
        for col, (name, (model, meta)) in zip(cols[1:], models.items()):
            pred = (predict_prob(model, meta, img) > thr).astype(np.uint8)
            col.image(overlay(img, pred, gt=gt), caption=f"{name} (green = ground truth)" if gt is not None else name,
                      width="stretch")
            col.metric("Muscle area (% of image)", f"{pred.mean() * 100:.1f}%")
            if gt is not None:
                sc = scores_np(pred, gt)
                col.metric("Dice", f"{sc['dice']:.3f}")
                col.caption(f"IoU {sc['iou']:.3f} · Precision {sc['precision']:.3f} · Recall {sc['recall']:.3f}")
            ok, buf = cv2.imencode(".png", pred * 255)
            col.download_button("Download mask", buf.tobytes(), file_name=f"mask_{name}_{label or 'upload'}.png",
                                mime="image/png", key=f"dl_{name}")

elif page == "Training":
    st.header("Training History")
    files = sorted(C.OUT_DIR.glob("history_*.json"), key=lambda p: (p.stem != "history_baseline", p.stem))
    if not files:
        st.info("No history yet. Run `python -m src.train`.")
        st.stop()
    pick = st.selectbox("Run", [f.stem.replace("history_", "") for f in files])
    hist = read_json(C.OUT_DIR / f"history_{pick}.json")
    df = pd.DataFrame(hist).set_index("epoch")
    c1, c2 = st.columns(2)
    c1.subheader("Loss"); c1.line_chart(df[["train_loss", "val_loss"]])
    c2.subheader("Dice"); c2.line_chart(df[["train_dice", "val_dice"]])
    if "coral" in df and df["coral"].abs().sum() > 0:
        st.subheader("CORAL loss"); st.line_chart(df[["coral"]])
    best = df["val_dice"].idxmax()
    st.success(f"Best validation Dice **{df['val_dice'].max():.4f}** at epoch **{best}**")
    st.dataframe(df.round(4), width="stretch")

elif page == "Cross-Dataset Evaluation":
    st.header("Cross-Dataset Evaluation")
    res = read_json(C.RESULTS_BASELINE)
    if not res:
        st.info("Run `python -m src.evaluate` to generate results.")
        st.stop()
    A, B, G = res["dataset_A_test"], res["dataset_B"], res["gap"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Dice — Dataset A", f"{A['dice']:.3f}")
    c2.metric("Dice — Dataset B", f"{B['dice']:.3f}", delta=f"-{G['dice_gap']:.3f}", delta_color="inverse")
    c3.metric("Generalization Gap (Dice)", f"{G['dice_gap']:.3f}", help=f"{G['dice_gap_pct']:.1f}% relative drop")

    table = pd.DataFrame({"Dataset A (internal test)": A, "Dataset B (unseen)": B}).loc[
        ["dice", "iou", "precision", "recall"]].round(4)
    st.dataframe(table, width="stretch")
    st.bar_chart(table)

    csv = C.OUT_DIR / "per_image_baseline.csv"
    if csv.exists():
        st.markdown("#### Per-image Dice")
        per = pd.read_csv(csv)
        wide = pd.DataFrame({d: g["dice"].describe() for d, g in per.groupby("dataset")}).round(3)
        st.dataframe(wide, width="stretch")
        worst = per[per.dataset == "B"].nsmallest(5, "dice")
        st.caption("Five hardest images in Dataset B:")
        st.dataframe(worst, hide_index=True, width="stretch")

    st.markdown("#### Before vs after domain adaptation")
    methods = read_json(C.RESULTS_METHODS)
    if not methods:
        st.info("Module 6 not run yet: `python -m src.adapt --method aug` (or adabn / clahe / coral / coral_aug), "
                "then `python -m src.compare`.")
    else:
        rows = []
        for name, r in methods.items():
            rows.append({"method": name,
                         "Dice A (test)": r["dataset_A_test"]["dice"],
                         "Dice B (all)": r["dataset_B"]["dice"],
                         "Dice B (held-out)": r["dataset_B_heldout"]["dice"],
                         "IoU B (held-out)": r["dataset_B_heldout"]["iou"],
                         "Gap (A - B held-out)": r["gap_heldout"]["dice_gap"]})
        comp = pd.DataFrame(rows).set_index("method")
        comp = comp.loc[["baseline"] + [m for m in comp.index if m != "baseline"]] if "baseline" in comp.index else comp
        st.dataframe(comp.round(4), width="stretch")
        st.bar_chart(comp[["Dice A (test)", "Dice B (held-out)"]])
        if "baseline" in comp.index and len(comp) > 1:
            best = comp.drop("baseline")["Dice B (held-out)"].idxmax()
            delta = comp.loc[best, "Dice B (held-out)"] - comp.loc["baseline", "Dice B (held-out)"]
            st.success(f"Best adapted method on held-out Dataset B: **{best}** (Dice +{delta:.3f} vs baseline)")
        st.caption("'Held-out' = part of Dataset B never used as the unlabeled adaptation pool. "
                   "Dataset-B masks are used only for evaluation, never for training.")
