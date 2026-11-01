import os
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# OUTPUT PATH
# ===============================
PLOT_DIR = "/Users/lyna.thc/Desktop/thesis/results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ===============================
# MODELS
# ===============================
models = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost (untuned)",
    "XGBoost (tuned)",
]

# ===============================
# DATA (FROM YOUR THESIS — VERIFIED)
# ===============================
phase1_auc = [0.748, 0.870, 0.858, 0.853]
phase2_auc = [0.782, 0.954, 0.885, 0.903]

# ===============================
# PLOT
# ===============================
x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(10, 6))

bars1 = plt.bar(x - width/2, phase1_auc, width,
                label="Phase 1 (Dataset V1)")
bars2 = plt.bar(x + width/2, phase2_auc, width,
                label="Phase 2 (V1 Imputed Fixed)")

# ===============================
# VALUE LABELS (IMPORTANT)
# ===============================
for i, v in enumerate(phase1_auc):
    plt.text(i - width/2, v + 0.005, f"{v:.3f}", ha='center', fontsize=9)

for i, v in enumerate(phase2_auc):
    plt.text(i + width/2, v + 0.005, f"{v:.3f}", ha='center', fontsize=9)

# ===============================
# AXES & TITLE
# ===============================
plt.xticks(x, models, rotation=20)
plt.ylabel("ROC-AUC")
plt.title("Impact of Dataset Expansion on Model Performance")

# ===============================
# BASELINE (LABELED — IMPORTANT)
# ===============================
plt.axhline(y=0.5, linestyle="--", color="gray",
            label="Random baseline (AUC = 0.5)")

# ===============================
# LEGEND & STYLE
# ===============================
plt.legend()
plt.ylim(0.5, 1.0)  # focus on meaningful range
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

# ===============================
# SAVE
# ===============================
out_path = os.path.join(PLOT_DIR, "fig7_phase1_vs_phase2_auc.png")
plt.savefig(out_path, dpi=150)
plt.close()

print(f"✅ Figure 7 saved → {out_path}")