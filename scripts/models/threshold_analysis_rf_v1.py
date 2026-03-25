import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# ===============================
# PATHS
# ===============================
PRED_PATH = "/Users/lyna.thc/Desktop/thesis/results/rf_tuned/rf_v1_test_predictions.csv"
OUT_DIR = "/Users/lyna.thc/Desktop/thesis/results/v1_threshold_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# LOAD PREDICTIONS
# ===============================
df = pd.read_csv(PRED_PATH)

y_true = df["y_true"].values
y_prob = df["y_prob"].values

# ===============================
# THRESHOLD ANALYSIS
# ===============================
thresholds = np.arange(0.30, 0.70, 0.05)
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
res_df.to_csv(os.path.join(OUT_DIR, "threshold_metrics.csv"), index=False)

# ===============================
# FP vs FN PLOT (CORE PLOT)
# ===============================
plt.figure(figsize=(8, 5))
plt.plot(res_df["threshold"], res_df["false_negatives"],
         marker="o", label="False Negatives")
plt.plot(res_df["threshold"], res_df["false_positives"],
         marker="o", label="False Positives")

plt.xlabel("Decision Threshold")
plt.ylabel("Number of Errors")
plt.title("Threshold Sensitivity Analysis – Tuned Random Forest (V1)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, "threshold_error_tradeoff.png"), dpi=150)
plt.close()

# ===============================
# PRECISION–RECALL vs THRESHOLD
# ===============================
plt.figure(figsize=(8, 5))
plt.plot(res_df["threshold"], res_df["precision"],
         marker="o", label="Precision")
plt.plot(res_df["threshold"], res_df["recall"],
         marker="o", label="Recall")

plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Precision–Recall Trade-off vs Threshold – Tuned RF (V1)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, "precision_recall_vs_threshold.png"), dpi=150)
plt.close()

print("\n✅ Threshold analysis completed")
print(res_df)