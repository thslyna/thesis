# ===============================
# FIGURE 10 — MONTHLY DEGRADATION
# ===============================
 
import matplotlib.pyplot as plt
import numpy as np
import os
 
# ===============================
# OUTPUT
# ===============================
OUT_DIR = "/Users/lyna.thc/Desktop/thesis/results/plots"
os.makedirs(OUT_DIR, exist_ok=True)
 
# ===============================
# DATA (FROM TABLE 4.10 — VERIFIED)
# ===============================
months = ["Oct–Nov", "December", "January", "February"]
auc_values = [0.903, 0.514, 0.489, 0.467]
 
x = np.arange(len(months))
 
# ===============================
# PLOT
# ===============================
plt.figure(figsize=(9, 6))
 
# Line
plt.plot(x, auc_values, color="black", linewidth=2.5, marker="o")
 
# Points (color-coded)
colors = ["green", "orange", "red", "darkred"]
for i, (xi, yi) in enumerate(zip(x, auc_values)):
    plt.scatter(xi, yi, color=colors[i], s=130, zorder=3)
 
# Random baseline
plt.axhline(
    0.5,
    linestyle="--",
    color="red",
    linewidth=2,
    label="Random baseline (AUC = 0.5)"
)
 
# ===============================
# ANNOTATIONS (CLEAN & BALANCED)
# ===============================
for i, v in enumerate(auc_values):

    if v >= 0.5:
        # above baseline → label above
        plt.annotate(
            f"{v:.3f}",
            (i, v),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=12,
            fontweight='bold'
        )
    else:
        # below baseline → label below
        plt.annotate(
            f"{v:.3f}",
            (i, v),
            textcoords="offset points",
            xytext=(0, -15),
            ha='center',
            fontsize=12,
            fontweight='bold'
        )
    
 
# ===============================
# LABELS
# ===============================
plt.xticks(x, months, fontsize=12)
plt.ylabel("ROC-AUC", fontsize=13)
plt.xlabel("Month", fontsize=13)
 
plt.title("Model Performance Degradation Across Months (Temporal Validation)",
          fontsize=14)
 
plt.ylim(0.45, 1.0)
 
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
 
# ===============================
# SAVE
# ===============================
out_path = os.path.join(OUT_DIR, "fig10_monthly_auc.png")
plt.savefig(out_path, dpi=150)
plt.close()
 
print(f"📉 Figure 10 saved → {out_path}")