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
OUT_DIR = "/Users/lyna.thc/Desktop/thesis/results/v1_calibration"
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# LOAD PREDICTIONS
# ===============================
df = pd.read_csv(PRED_PATH)

y_true = df["y_true"].values
y_prob = df["y_prob"].values

# ===============================
# CALIBRATION CURVE
# ===============================
# Split predictions into probability bins
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

# ===============================
# PLOT
# ===============================
plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker="o", label="Tuned Random Forest")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")

plt.xlabel("Predicted delay probability")
plt.ylabel("Observed delay frequency")
plt.title("Calibration Curve – Tuned Random Forest (V1)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, "calibration_curve_rf_v1.png"), dpi=150)
plt.close()

# ===============================
# SAVE METRICS
# ===============================
metrics = {
    "brier_score": round(brier, 4)
}

pd.DataFrame([metrics]).to_csv(
    os.path.join(OUT_DIR, "calibration_metrics.csv"),
    index=False
)

print("✅ Calibration analysis completed")
print("Brier score:", round(brier, 4))