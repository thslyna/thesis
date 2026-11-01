#!/usr/bin/env python3
# ============================================================
# Figure 6 — Calibration Curve (Dataset V1)
# ============================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# ===============================
# PATHS
# ===============================
PRED_PATH = "/Users/lyna.thc/Desktop/thesis/results/rf_tuned/rf_v1_test_predictions.csv"

OUT_DIR = "/Users/lyna.thc/Desktop/thesis/results/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(PRED_PATH)

y_true = df["y_true"].values
y_prob = df["y_prob"].values

print("Dataset loaded:", df.shape)

# ===============================
# CALIBRATION
# ===============================
prob_true, prob_pred = calibration_curve(
    y_true,
    y_prob,
    n_bins=10,
    strategy="uniform"
)

# ===============================
# BRIER SCORE
# ===============================
brier = brier_score_loss(y_true, y_prob)

print(f"Brier Score: {brier:.4f}")

# ===============================
# PLOT
# ===============================
plt.figure(figsize=(6, 6))

# Model curve
plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    label=f"Model (Brier = {brier:.3f})"
)

# Perfect calibration
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Perfect calibration"
)

plt.xlabel("Predicted Probability")
plt.ylabel("Observed Frequency")
plt.title("Calibration Curve (Dataset V1)")

plt.legend()
plt.tight_layout()

# Save
plt.savefig(
    os.path.join(OUT_DIR, "fig6_calibration_curve.png"),
    dpi=300
)

plt.close()

print("\n✅ Figure 6 saved → fig6_calibration_curve.png")