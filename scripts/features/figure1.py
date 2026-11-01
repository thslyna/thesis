import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# PATHS
# ===============================
RESULTS_PATH = "/Users/lyna.thc/Desktop/thesis/results/v1_full_evaluation/v1_model_comparison.csv"
PLOTS_DIR = "/Users/lyna.thc/Desktop/thesis/results/plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

# ===============================
# STYLE (THESIS-READY)
# ===============================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11
})

# ===============================
# LOAD RESULTS
# ===============================
df = pd.read_csv(RESULTS_PATH)

# Ensure correct order
model_order = [
    "Logistic Regression",
    "Random Forest (untuned)",
    "Random Forest (tuned)",
    "XGBoost (untuned)",
    "XGBoost (tuned)"
]

df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
df = df.sort_values("model")

# ===============================
# EXTRACT METRICS
# ===============================
models = df["model"].values
roc_auc = df["roc_auc"].values
precision = df["precision"].values
recall = df["recall"].values
f1 = df["f1_score"].values

# ===============================
# PLOT
# ===============================
x = np.arange(len(models))
width = 0.16

plt.figure(figsize=(8, 5))

plt.bar(x - 1.5*width, roc_auc, width, label="ROC-AUC")
plt.bar(x - 0.5*width, precision, width, label="Precision")
plt.bar(x + 0.5*width, recall, width, label="Recall")
plt.bar(x + 1.5*width, f1, width, label="F1-score")

# Axis & labels
plt.xticks(x, models, rotation=10)
plt.ylabel("Score")
plt.ylim(0.65, 0.9)
plt.title("Model Performance Comparison (Dataset V1)")

# Legend
plt.legend(frameon=False, ncol=2)

# Clean look
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Optional subtle reference line
plt.axhline(0.75, linestyle="--", linewidth=1)

plt.tight_layout()

# ===============================
# SAVE
# ===============================
save_path = os.path.join(PLOTS_DIR, "fig1_model_comparison.png")
plt.savefig(save_path, dpi=300)
plt.show()

print("Saved:", save_path)