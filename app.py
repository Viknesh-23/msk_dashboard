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
        ("6  Domain Adaptation & Comparison", "Done" if read_json(C.RESULTS_ADAPTED) else "Pending"),
    ], columns=["Module", "Status"])
    st.dataframe(status, hide_index=True, use_container_width=True)

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
    st.line_chart(pd.DataFrame(chart, index=np.arange(0, 256, 8)), x_label="Pixel intensity", y_label="Fraction")
    st.caption("Different curves between A and B indicate brightness/contrast differences — the source of the generalization gap.")

    st.markdown("#### Browse samples")
    ds_name = st.radio("Dataset", [n for n, p in (("A", pa), ("B", pb)) if p], horizontal=True)
    pairs = pa if ds_name == "A" else pb
    idx = st.slider("Sample", 0, len(pairs) - 1, 0)
    img, m = load_image(pairs[idx][0]), load_mask(pairs[idx][1])
    from src.inference import overlay
    c1, c2, c3 = st.columns(3)
    c1.image(img, caption=pairs[idx][0].name, clamp=True, use_container_width=True)
    c2.image(m * 255, caption="Ground-truth mask", clamp=True, use_container_width=True)
    c3.image(overlay(img, m), caption="Overlay", use_container_width=True)

elif page == "Segmentation Demo":
    st.header("Muscle Segmentation Demo")
    if not model_ready():
        st.warning("No trained model yet. Run `python -m src.train` first.")
        st.stop()
    from src.inference import overlay, predict_prob
    model, meta = get_model(str(C.CKPT_BASELINE), C.CKPT_BASELINE.stat().st_mtime)

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

    if img is not None:
        prob = predict_prob(model, meta, img)
        pred = (prob > thr).astype(np.uint8)
        c1, c2, c3 = st.columns(3)
        c1.image(img, caption=label, clamp=True, use_container_width=True)
        c2.image(pred * 255, caption="Predicted mask", clamp=True, use_container_width=True)
        c3.image(overlay(img, pred, gt=gt), caption="Overlay (green = ground truth)" if gt is not None else "Overlay",
                 use_container_width=True)
        st.metric("Muscle area (% of image)", f"{pred.mean() * 100:.1f}%")
        if gt is not None:
            s = scores_np(pred, gt)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Dice", f"{s['dice']:.3f}"); m2.metric("IoU", f"{s['iou']:.3f}")
            m3.metric("Precision", f"{s['precision']:.3f}"); m4.metric("Recall", f"{s['recall']:.3f}")
        ok, buf = cv2.imencode(".png", pred * 255)
        st.download_button("Download mask (PNG)", buf.tobytes(), file_name=f"mask_{label or 'upload'}.png", mime="image/png")

elif page == "Training":
    st.header("Training History")
    hist = read_json(C.HISTORY_BASELINE)
    if not hist:
        st.info("No history yet. Run `python -m src.train`.")
        st.stop()
    df = pd.DataFrame(hist).set_index("epoch")
    c1, c2 = st.columns(2)
    c1.subheader("Loss"); c1.line_chart(df[["train_loss", "val_loss"]])
    c2.subheader("Dice"); c2.line_chart(df[["train_dice", "val_dice"]])
    best = df["val_dice"].idxmax()
    st.success(f"Best validation Dice **{df['val_dice'].max():.4f}** at epoch **{best}**")
    st.dataframe(df.round(4), use_container_width=True)

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
    st.dataframe(table, use_container_width=True)
    st.bar_chart(table)

    csv = C.OUT_DIR / "per_image_baseline.csv"
    if csv.exists():
        st.markdown("#### Per-image Dice")
        per = pd.read_csv(csv)
        wide = pd.DataFrame({d: g["dice"].describe() for d, g in per.groupby("dataset")}).round(3)
        st.dataframe(wide, use_container_width=True)
        worst = per[per.dataset == "B"].nsmallest(5, "dice")
        st.caption("Five hardest images in Dataset B:")
        st.dataframe(worst, hide_index=True, use_container_width=True)

    st.markdown("#### Before vs after domain adaptation")
    adapted = read_json(C.RESULTS_ADAPTED)
    if not adapted:
        st.info("Module 6 (CORAL / normalization / augmentation) not run yet. When `outputs/results_adapted.json` "
                "exists (same format as `results_baseline.json`), the comparison appears here.")
    else:
        AB = adapted["dataset_B"]
        comp = pd.DataFrame({"Baseline on B": B, "Adapted on B": AB}).loc[["dice", "iou", "precision", "recall"]]
        comp["Improvement"] = comp["Adapted on B"] - comp["Baseline on B"]
        st.dataframe(comp.round(4), use_container_width=True)
        st.bar_chart(comp[["Baseline on B", "Adapted on B"]])
