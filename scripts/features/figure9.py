import matplotlib.pyplot as plt
import numpy as np
import os

# ===============================
# OUTPUT
# ===============================
OUT_DIR = "/Users/lyna.thc/Desktop/thesis/results/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# DATA (FROM TABLE 4.9)
# ===============================
labels = [
    "Random Split\n(Oct–Nov)",
    "December\n(+1 month)",
    "Jan–Feb\n(+3–4 months)"
]

auc_values = [0.903, 0.514, 0.474]

x = np.arange(len(labels))

# ===============================
# PLOT
# ===============================
plt.figure(figsize=(9, 6))

# Line
plt.plot(x, auc_values, color="black", linewidth=2.5, zorder=2)

# Points with colors
colors = ["green", "orange", "red"]
for i, (xi, yi) in enumerate(zip(x, auc_values)):
    plt.scatter(xi, yi, color=colors[i], s=140, zorder=3)

# Fill area (performance drop)
plt.fill_between(x, auc_values, color="lightsteelblue", alpha=0.6)

# Random baseline
plt.axhline(0.5, linestyle="--", color="red", linewidth=2,
            label="Random baseline (AUC = 0.5)")

# ===============================
# ANNOTATIONS (offset diagonally away from the line so each label
# sits cleanly next to its point — no overlap with axis ticks or the curve)
# ===============================
annotation_configs = [
    # 0.903 — top-left point: push label right & slightly down so it sits
    # along the descending line without clipping the top
    {"xytext": (14, -4),  "ha": "left",   "va": "center"},
    # 0.514 — middle point: push straight up above the line
    {"xytext": (0,  14),  "ha": "center", "va": "bottom"},
    # 0.474 — bottom-right point: push label up-left so it clears the
    # x-axis "Jan–Feb" tick label entirely
    {"xytext": (-12, 14), "ha": "right",  "va": "bottom"},
]

for i, (v, cfg) in enumerate(zip(auc_values, annotation_configs)):
    plt.annotate(f"{v:.3f}",
                 (i, v),
                 textcoords="offset points",
                 xytext=cfg["xytext"],
                 ha=cfg["ha"],
                 va=cfg["va"],
                 fontsize=14,
                 fontweight='medium')

# ===============================
# LABELS
# ===============================
plt.xticks(x, labels, fontsize=12)
plt.ylabel("ROC-AUC", fontsize=13)
plt.xlabel("Evaluation Condition", fontsize=13)

plt.title("Temporal Validation Reveals Severe Performance Degradation Over Time",
          fontsize=14)

plt.ylim(0.45, 1.0)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# ===============================
# SAVE
# ===============================
out_path = os.path.join(OUT_DIR, "fig9_temporal_collapse.png")
plt.savefig(out_path, dpi=150)
plt.close()

print(f"✅ Figure 9 saved → {out_path}")
