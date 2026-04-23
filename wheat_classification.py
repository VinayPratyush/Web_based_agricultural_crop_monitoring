# =========================================================
# WHEAT CLASSIFICATION — CropHarvest global training
# =========================================================
# Reads CropHarvest .h5 files directly from disk — no
# cropharvest package needed (avoids its ancient dependency
# pins that break modern pandas/numpy).
#
# Prerequisites — download from https://zenodo.org/records/10251170:
#   1. features.tar.gz  (82.5 MB)   → extract into cropharvest_data/
#   2. labels.geojson   (85.7 MB)   → save into   cropharvest_data/
#
# Expected layout:
#   D:/agri_project/
#   ├── wheat_classification.py
#   └── cropharvest_data/
#       ├── labels.geojson
#       └── features/arrays/*.h5
#
# Output (unchanged — compatible with all_integrated.py):
#   wheat_model.pth
#   mean.npy   shape (1, 1, 11)
#   std.npy    shape (1, 1, 11)
#
# Input shape:  (batch, 12 months, 11 features)
#   features = [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI]
# =========================================================

import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, cohen_kappa_score, accuracy_score,
    roc_curve, precision_recall_curve, average_precision_score,
)

import json
import matplotlib
matplotlib.use("Agg")              # headless backend — no display needed
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CROPHARVEST_DIR = os.path.join(BASE_DIR, "cropharvest_data")
LABELS_PATH     = os.path.join(CROPHARVEST_DIR, "labels.geojson")
FEATURES_DIR    = os.path.join(CROPHARVEST_DIR, "features", "arrays")

MODEL_PATH = os.path.join(BASE_DIR, "wheat_model.pth")
MEAN_PATH  = os.path.join(BASE_DIR, "mean.npy")
STD_PATH   = os.path.join(BASE_DIR, "std.npy")

# Reporting folder — figures and metrics for the project report
REPORT_DIR = os.path.join(BASE_DIR, "wheat_classification_report")
os.makedirs(REPORT_DIR, exist_ok=True)

EPOCHS     = 25
BATCH_SIZE = 128
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WHEAT_KEYWORDS = ["wheat"]   # case-insensitive substring match

# CropHarvest 18-feature vector layout (from cropharvest/bands.py):
#   0..1   → Sentinel-1 [VV, VH]              (dropped — not in local pipeline)
#   2..12  → Sentinel-2 [B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12]
#   13..14 → ERA5 [temperature, precipitation] (dropped)
#   15..16 → DEM [elevation, slope]            (dropped)
#   17     → NDVI
#
# Our local pipeline in all_integrated.py downloads 10 S2 bands:
#   [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]
# We drop B9 from CropHarvest to match. Plus NDVI = 11 features total.
S2_INDICES      = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12]   # drop B9 (index 10)
NDVI_INDEX      = 17
FEATURE_INDICES = S2_INDICES + [NDVI_INDEX]          # 11 features
NUM_FEATURES    = len(FEATURE_INDICES)               # 11
NUM_TIMESTEPS   = 12                                 # monthly

# CropHarvest stores Sentinel-2 as raw DN values (0..10000). The local pipeline
# in all_integrated.py divides incoming S2 by 10000 to get reflectance (0..1).
# We do the same divide here so the model sees the same scale at train + infer.
S2_SCALE = 10_000.0

# =========================================================
# STEP 1 — Sanity check the data directory
# =========================================================
print("=" * 60)
print("Step 1/6 · Locating CropHarvest data")
print("=" * 60)

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(
        f"labels.geojson not found at {LABELS_PATH}\n"
        f"Download from: https://zenodo.org/records/10251170/files/labels.geojson"
    )
if not os.path.isdir(FEATURES_DIR):
    raise FileNotFoundError(
        f"Features folder not found at {FEATURES_DIR}\n"
        f"Download features.tar.gz from https://zenodo.org/records/10251170 "
        f"and extract into {CROPHARVEST_DIR}"
    )

print(f"  labels:   {LABELS_PATH}")
print(f"  features: {FEATURES_DIR}")

# =========================================================
# STEP 2 — Load labels and flag wheat parcels
# =========================================================
print("\n" + "=" * 60)
print("Step 2/6 · Reading labels.geojson")
print("=" * 60)

gdf = gpd.read_file(LABELS_PATH)
print(f"Total labeled parcels: {len(gdf)}")
print(f"Available columns: {gdf.columns.tolist()}")

# Label column name varies across CropHarvest releases. Check common options.
label_candidates = ["label", "classification_label", "crop_type", "original_label"]
present_cols = [c for c in label_candidates if c in gdf.columns]
print(f"Label-like columns found: {present_cols}")

def row_mentions_wheat(row):
    for col in present_cols:
        val = row.get(col)
        if isinstance(val, str) and any(k in val.lower() for k in WHEAT_KEYWORDS):
            return True
    return False

gdf["is_wheat"] = gdf.apply(row_mentions_wheat, axis=1)

n_wheat    = int(gdf["is_wheat"].sum())
n_nonwheat = len(gdf) - n_wheat
print(f"Wheat parcels:     {n_wheat}")
print(f"Non-wheat parcels: {n_nonwheat}")

if n_wheat == 0:
    print("\n[!] No 'wheat' labels found. Sample of unique values per column:")
    for col in present_cols:
        uniques = gdf[col].dropna().unique()[:30]
        print(f"  {col}: {list(uniques)}")
    raise ValueError("Could not locate wheat rows. Update WHEAT_KEYWORDS above.")

# =========================================================
# STEP 3 — Index .h5 files and match to labels
# =========================================================
print("\n" + "=" * 60)
print("Step 3/6 · Indexing .h5 files")
print("=" * 60)

h5_files = glob.glob(os.path.join(FEATURES_DIR, "**", "*.h5"), recursive=True)
print(f"Found {len(h5_files)} .h5 files on disk")

# CropHarvest uses <dataset>_<index>.h5 as the file naming convention.
# We key the index by the basename without extension.
h5_index = {os.path.splitext(os.path.basename(p))[0]: p for p in h5_files}

if "dataset" in gdf.columns and "index" in gdf.columns:
    # CropHarvest .h5 files are named <index>_<dataset>.h5
    # e.g. "0_ethiopia.h5"  (confirmed from inspection)
    gdf["h5_key"] = gdf["index"].astype(str) + "_" + gdf["dataset"].astype(str)
elif "index" in gdf.columns:
    gdf["h5_key"] = gdf["index"].astype(str)
else:
    gdf["h5_key"] = gdf.index.astype(str)

gdf["h5_path"] = gdf["h5_key"].map(h5_index)
matched = int(gdf["h5_path"].notna().sum())
print(f"Labels matched to .h5: {matched} / {len(gdf)}")

if matched == 0:
    # Show a few sample filenames vs. a few keys to help diagnose
    sample_keys  = gdf["h5_key"].head(5).tolist()
    sample_files = [os.path.basename(p) for p in h5_files[:5]]
    raise RuntimeError(
        f"Zero label-to-file matches.\n"
        f"Sample label keys:      {sample_keys}\n"
        f"Sample filenames on disk: {sample_files}\n"
        f"The key format differs — inspect the labels.geojson columns and "
        f"update the 'h5_key' construction above."
    )

gdf_matched = gdf[gdf["h5_path"].notna()].reset_index(drop=True)

# =========================================================
# STEP 4 — Load arrays into memory
# =========================================================
print("\n" + "=" * 60)
print("Step 4/6 · Loading per-parcel time series into RAM")
print("=" * 60)

def safe_load(path):
    try:
        with h5py.File(path, "r") as f:
            arr = np.asarray(f.get("array"))
        if arr.shape == (12, 18):
            return arr
    except Exception:
        pass
    return None

X_wheat, X_nonwheat = [], []
D_wheat, D_nonwheat = [], []        # dataset (country-ish) tag per sample
for row in tqdm(gdf_matched.itertuples(index=False),
                total=len(gdf_matched), desc="Loading arrays"):
    arr = safe_load(row.h5_path)
    if arr is None:
        continue
    if row.is_wheat:
        X_wheat.append(arr)
        D_wheat.append(str(row.dataset))
    else:
        X_nonwheat.append(arr)
        D_nonwheat.append(str(row.dataset))

print(f"Loaded wheat     arrays: {len(X_wheat)}")
print(f"Loaded non-wheat arrays: {len(X_nonwheat)}")

if len(X_wheat) == 0:
    raise RuntimeError("No wheat arrays were loaded — check label matching.")

# =========================================================
# STEP 5 — Balance, feature-select, normalize, split, train
# =========================================================
print("\n" + "=" * 60)
print("Step 5/6 · Preparing training data")
print("=" * 60)

min_size = min(len(X_wheat), len(X_nonwheat))
rng = np.random.default_rng(42)

wheat_idx    = rng.choice(len(X_wheat),    size=min_size, replace=False)
nonwheat_idx = rng.choice(len(X_nonwheat), size=min_size, replace=False)

X = np.stack(
    [X_wheat[i]    for i in wheat_idx] +
    [X_nonwheat[i] for i in nonwheat_idx]
)
y = np.array([1] * min_size + [0] * min_size, dtype=np.float32)
datasets = np.array(
    [D_wheat[i]    for i in wheat_idx] +
    [D_nonwheat[i] for i in nonwheat_idx]
)
print(f"Balanced samples: {len(X)}  (wheat={min_size}, non-wheat={min_size})")

# Keep only Sentinel-2 + NDVI  (correct CropHarvest indices)
X = X[:, :, FEATURE_INDICES].astype(np.float32)     # (N, 12, 11)

# Scale S2 bands (first 10 features) from 0..10000 DN → 0..1 reflectance.
# NDVI (last feature) is already a ratio in [-1, 1] — do NOT scale.
X[:, :, :10] = X[:, :, :10] / S2_SCALE
X[:, :, :10] = np.clip(X[:, :, :10], 0.0, 1.0)

print(f"Shape after feature selection: {X.shape}")
print(f"  S2 bands range: {X[:, :, :10].min():.3f} .. {X[:, :, :10].max():.3f}")
print(f"  NDVI    range: {X[:, :, 10].min():.3f} .. {X[:, :, 10].max():.3f}")

# Keep a raw (un-normalized) NDVI snapshot for the time-series figure before normalizing
X_raw_ndvi = X[:, :, 10].copy()   # NDVI is the 11th feature (index 10)

mean = X.mean(axis=(0, 1), keepdims=True)
std  = X.std(axis=(0, 1),  keepdims=True) + 1e-6
X    = (X - mean) / std

# Stratified split with dataset-label tracking
indices = np.arange(len(X))
tr_idx, te_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y
)
X_tr, X_te = X[tr_idx], X[te_idx]
y_tr, y_te = y[tr_idx], y[te_idx]
datasets_te = datasets[te_idx]        # dataset tag for each test sample

tr_loader = DataLoader(
    TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                  torch.tensor(y_tr, dtype=torch.float32)),
    batch_size=BATCH_SIZE, shuffle=True,
)
te_loader = DataLoader(
    TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                  torch.tensor(y_te, dtype=torch.float32)),
    batch_size=BATCH_SIZE,
)

# Simple LSTM-based classifier — should be enough to learn the seasonal NDVI signature of wheat.
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc1  = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.relu(self.fc1(out[:, -1, :]))
        return self.fc2(out)

model     = LSTMModel(input_size=NUM_FEATURES).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("\n" + "=" * 60)
print(f"Training LSTM on {DEVICE}")
print("=" * 60)

loss_history = []          # track for loss-curve plot

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for xb, yb in tr_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb).squeeze(-1)
        loss   = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss_history.append(total_loss)
    print(f"Epoch {epoch+1:>2}/{EPOCHS}  loss={total_loss:.4f}")

# =========================================================
# STEP 6 — Save and evaluate
# =========================================================
print("\n" + "=" * 60)
print("Step 6/6 · Saving and evaluating")
print("=" * 60)

torch.save(model.state_dict(), MODEL_PATH)
np.save(MEAN_PATH, mean)
np.save(STD_PATH,  std)
print(f"  -> {MODEL_PATH}")
print(f"  -> {MEAN_PATH}")
print(f"  -> {STD_PATH}")

model.eval()
all_probs, all_preds, all_true = [], [], []
with torch.no_grad():
    for xb, yb in te_loader:
        xb = xb.to(DEVICE)
        logits = model(xb).squeeze(-1)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend((probs > 0.5).astype(int))
        all_true.extend(yb.numpy().astype(int))

cm        = confusion_matrix(all_true, all_preds)
accuracy  = accuracy_score(all_true, all_preds)
precision = precision_score(all_true, all_preds)
recall    = recall_score(all_true, all_preds)
f1        = f1_score(all_true, all_preds)
roc_auc   = roc_auc_score(all_true, all_probs)
kappa     = cohen_kappa_score(all_true, all_preds)

print("\n===== GLOBAL RESULTS =====")
print("Confusion Matrix:\n", cm)
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"F1 Score:    {f1:.4f}")
print(f"ROC-AUC:     {roc_auc:.4f}")
print(f"Cohen Kappa: {kappa:.4f}")

# =========================================================
# REPORT GENERATION — figures and tables for wheat_classification_report/
# =========================================================
print("\n" + "=" * 60)
print(f"Generating report figures → {REPORT_DIR}")
print("=" * 60)

all_probs = np.asarray(all_probs)
all_preds = np.asarray(all_preds)
all_true  = np.asarray(all_true)

# Use a consistent, clean visual style for every figure
plt.rcParams.update({
    "figure.dpi":      120,
    "savefig.dpi":     200,
    "font.size":       11,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

WHEAT_C    = "#27AE60"     # emerald
NONWHEAT_C = "#E67E22"     # orange
ACCENT_C   = "#2980B9"     # blue

def save_fig(fig, name):
    path = os.path.join(REPORT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


# ---- 1. Confusion matrix ---------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm, cmap="Greens")
ax.set_title("Confusion Matrix · Wheat vs. Non-Wheat", pad=12, fontweight="bold")
ax.set_xticks([0, 1], ["Non-Wheat", "Wheat"])
ax.set_yticks([0, 1], ["Non-Wheat", "Wheat"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.grid(False)

# Annotate each cell with count + percentage
total = cm.sum()
for i in range(2):
    for j in range(2):
        count = cm[i, j]
        pct   = count / total * 100
        color = "white" if count > cm.max() / 2 else "black"
        ax.text(j, i, f"{count}\n({pct:.1f}%)",
                ha="center", va="center", color=color,
                fontsize=13, fontweight="bold")

fig.colorbar(im, ax=ax, fraction=0.045)
save_fig(fig, "1_confusion_matrix.png")


# ---- 2. ROC curve ----------------------------------------------------
fpr, tpr, _ = roc_curve(all_true, all_probs)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color=ACCENT_C, linewidth=2.5,
        label=f"LSTM  (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, label="Random baseline")
ax.fill_between(fpr, tpr, alpha=0.15, color=ACCENT_C)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve · Global Wheat Classification",
             pad=12, fontweight="bold")
ax.legend(loc="lower right", frameon=False)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
save_fig(fig, "2_roc_curve.png")


# ---- 3. Precision-Recall curve --------------------------------------
prec, rec, _ = precision_recall_curve(all_true, all_probs)
avg_prec     = average_precision_score(all_true, all_probs)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec, prec, color=WHEAT_C, linewidth=2.5,
        label=f"LSTM  (AP = {avg_prec:.3f})")
ax.fill_between(rec, prec, alpha=0.15, color=WHEAT_C)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision–Recall Curve", pad=12, fontweight="bold")
ax.legend(loc="lower left", frameon=False)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
save_fig(fig, "3_precision_recall_curve.png")


# ---- 4. Training loss curve -----------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(range(1, len(loss_history) + 1), loss_history,
        color=ACCENT_C, linewidth=2.2, marker="o", markersize=4)
ax.set_xlabel("Epoch")
ax.set_ylabel("Total Training Loss (BCE)")
ax.set_title("Training Loss Convergence", pad=12, fontweight="bold")
ax.set_xlim(1, len(loss_history))
save_fig(fig, "4_training_loss.png")


# ---- 5. Per-country F1 bar chart (the global-generalization story) ---
# Use only countries with >= 20 test samples — otherwise metrics are too noisy
country_f1 = {}
for c in np.unique(datasets_te):
    mask = datasets_te == c
    if mask.sum() < 20:
        continue
    if len(np.unique(all_true[mask])) < 2:      # single-class subset — can't F1
        continue
    country_f1[c] = {
        "n":  int(mask.sum()),
        "f1": f1_score(all_true[mask], all_preds[mask]),
        "acc": accuracy_score(all_true[mask], all_preds[mask]),
    }

if country_f1:
    # Sort descending by F1
    ordered = sorted(country_f1.items(), key=lambda kv: kv[1]["f1"], reverse=True)
    names  = [k.replace("-", " ").replace("_", " ").title() for k, _ in ordered]
    f1s    = [v["f1"] for _, v in ordered]
    ns     = [v["n"]  for _, v in ordered]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.35)))
    bars = ax.barh(names, f1s, color=WHEAT_C, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axvline(f1, linestyle="--", color="red", alpha=0.6,
               label=f"Global F1 = {f1:.3f}")
    for bar, score, n in zip(bars, f1s, ns):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}  (n={n})", va="center", fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Country Generalization · Wheat Classification",
                 pad=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    ax.invert_yaxis()
    save_fig(fig, "5_per_country_f1.png")
else:
    print("  (per-country plot skipped — no country has ≥20 test samples)")


# ---- 6. NDVI time-series signature (wheat vs. non-wheat) -------------
# The "money shot" figure — shows the model has learned real phenology.
wheat_mask_all    = y == 1
nonwheat_mask_all = y == 0

ndvi_wheat    = X_raw_ndvi[wheat_mask_all]          # (n_wheat, 12)
ndvi_nonwheat = X_raw_ndvi[nonwheat_mask_all]

months = np.arange(1, 13)
fig, ax = plt.subplots(figsize=(9, 4.8))

mean_w, std_w = ndvi_wheat.mean(axis=0),    ndvi_wheat.std(axis=0)
mean_n, std_n = ndvi_nonwheat.mean(axis=0), ndvi_nonwheat.std(axis=0)

ax.plot(months, mean_w, color=WHEAT_C, linewidth=2.5, marker="o",
        label=f"Wheat (n={wheat_mask_all.sum()})")
ax.fill_between(months, mean_w - std_w, mean_w + std_w,
                color=WHEAT_C, alpha=0.18)

ax.plot(months, mean_n, color=NONWHEAT_C, linewidth=2.5, marker="s",
        label=f"Non-wheat (n={nonwheat_mask_all.sum()})")
ax.fill_between(months, mean_n - std_n, mean_n + std_n,
                color=NONWHEAT_C, alpha=0.18)

ax.set_xlabel("Month of the agricultural year")
ax.set_ylabel("Mean NDVI  ±1 std")
ax.set_title("Seasonal NDVI Signature · Wheat vs. Non-Wheat",
             pad=12, fontweight="bold")
ax.set_xticks(months)
ax.legend(loc="upper right", frameon=False)
save_fig(fig, "6_ndvi_signature.png")


# ---- 7. Metrics summary table (as a PNG + JSON) ----------------------
metrics = {
    "accuracy":   float(accuracy),
    "precision":  float(precision),
    "recall":     float(recall),
    "f1_score":   float(f1),
    "roc_auc":    float(roc_auc),
    "avg_precision": float(avg_prec),
    "cohen_kappa":  float(kappa),
    "confusion_matrix": cm.tolist(),
    "n_train":    int(len(X_tr)),
    "n_test":     int(len(X_te)),
    "n_features": int(NUM_FEATURES),
    "n_timesteps": int(NUM_TIMESTEPS),
    "epochs":     int(EPOCHS),
    "device":     str(DEVICE),
    "per_country": country_f1,
}

with open(os.path.join(REPORT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("  ✓ metrics.json")

# Table figure (pasteable into the report)
rows = [
    ("Accuracy",    f"{accuracy:.4f}"),
    ("Precision",   f"{precision:.4f}"),
    ("Recall",      f"{recall:.4f}"),
    ("F1 Score",    f"{f1:.4f}"),
    ("ROC-AUC",     f"{roc_auc:.4f}"),
    ("Avg. Precision", f"{avg_prec:.4f}"),
    ("Cohen Kappa", f"{kappa:.4f}"),
    ("Train samples", f"{len(X_tr)}"),
    ("Test samples",  f"{len(X_te)}"),
    ("Input shape",   f"(12, {NUM_FEATURES})"),
    ("Epochs",        f"{EPOCHS}"),
    ("Device",        str(DEVICE).upper()),
]

fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(rows) + 1))
ax.axis("off")
tbl = ax.table(
    cellText=rows,
    colLabels=["Metric", "Value"],
    loc="center", cellLoc="left", colLoc="left",
    colWidths=[0.55, 0.35],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 1.5)
for (r, _), cell in tbl.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    if r == 0:
        cell.set_facecolor(WHEAT_C)
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#F7F7F7")
ax.set_title("Wheat Classifier · Summary Metrics",
             pad=18, fontweight="bold", fontsize=13)
save_fig(fig, "7_metrics_summary.png")

print(f"\n{'='*60}")
print(f"Report saved to: {REPORT_DIR}")
print(f"{'='*60}")
print("\nDone. Model ready for all_integrated.py inference.")
