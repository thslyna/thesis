#!/usr/bin/env python3
# ============================================================
# Figure 5 — Threshold Sensitivity (Error Trade-off)
# ============================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# ============================================================
# PATHS
# ============================================================

PRED_PATH = "/Users/lyna.thc/Desktop/thesis/results/rf_tuned/rf_v1_test_predictions.csv"

OUT_DIR = "/Users/lyna.thc/Desktop/thesis/results/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(PRED_PATH)

y_true = df["y_true"].values
y_prob = df["y_prob"].values

print("Dataset loaded:", df.shape)

# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

thresholds = np.arange(0.30, 0.71, 0.05)

results = []

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    results.append({
        "threshold": round(t, 2),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "false_positives": fp,
        "false_negatives": fn
    })

res_df = pd.DataFrame(results)

# Save metrics
res_df.to_csv(os.path.join(OUT_DIR, "threshold_metrics_v1.csv"), index=False)

print("\nThreshold analysis results:")
print(res_df)

# ============================================================
# FIGURE 5 — ERROR TRADE-OFF
# ============================================================

plt.figure(figsize=(8, 5))

# Curves
plt.plot(
    res_df["threshold"],
    res_df["false_negatives"],
    marker="o",
    label="False Negatives"
)

plt.plot(
    res_df["threshold"],
    res_df["false_positives"],
    marker="o",
    label="False Positives"
)

# ============================================================
# Highlight balanced region (KEY ADDITION)
# ============================================================

# Find closest intersection automatically (better than hardcoding)
diff = np.abs(res_df["false_negatives"] - res_df["false_positives"])
idx = diff.idxmin()

best_x = res_df.loc[idx, "threshold"]
best_y = res_df.loc[idx, "false_negatives"]

plt.scatter(best_x, best_y, zorder=5)

plt.text(
    best_x,
    best_y + 5,
    "Balanced region",
    ha="center",
    fontsize=10
)

# ============================================================
# Reference line (default threshold)
# ============================================================

plt.axvline(0.5, linestyle="--", linewidth=1)

# Labels
plt.xlabel("Classification Threshold")
plt.ylabel("Error Count")
plt.title("Threshold Sensitivity — Error Trade-off (Dataset V1)")

plt.legend()
plt.tight_layout()

# Save
plt.savefig(
    os.path.join(OUT_DIR, "fig5_threshold_sensitivity.png"),
    dpi=300
)

plt.close()

print("\n✅ Figure 5 saved → fig5_threshold_sensitivity.png")

# ============================================================
# OPTIONAL — PRECISION / RECALL CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    res_df["threshold"],
    res_df["precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    res_df["threshold"],
    res_df["recall"],
    marker="o",
    label="Recall"
)

plt.axvline(0.5, linestyle="--", linewidth=1)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Precision–Recall Trade-off vs Threshold (Dataset V1)")

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "precision_recall_vs_threshold.png"),
    dpi=300
)

plt.close()

print("📊 Extra plot saved → precision_recall_vs_threshold.png")
print("\nAll done ✅")