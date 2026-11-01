import matplotlib.pyplot as plt
import os

# ===============================
# SETUP
# ===============================
os.makedirs("results/plots", exist_ok=True)

print("="*60)
print("📊 FIGURE 18 — PHASE 2 ABLATION (FINAL)")
print("="*60)

# ===============================
# DATA (FROM TABLE 4.20)
# ===============================
labels = ["No Weather", "No Historical", "No Traffic"]
auc_values = [0.762, 0.847, 0.901]
delta_auc = [-0.141, -0.056, -0.002]

# ===============================
# PLOT
# ===============================
plt.figure(figsize=(9, 5))

colors = ["#e74c3c", "#f39c12", "#2ecc71"]

bars = plt.barh(labels, delta_auc, color=colors, edgecolor="black")

# vertical line at 0
plt.axvline(0, color="black", linewidth=1)

# labels
plt.xlabel("Δ AUC vs Baseline", fontsize=12, fontweight="bold")
plt.title("Phase 2 Ablation Study (Impact on AUC)", fontsize=14, fontweight="bold")

# annotate bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(
        width - 0.004,
        bar.get_y() + bar.get_height()/2,
        f"{width:.3f}",
        va="center",
        ha="right",
        fontsize=11
    )

# axis limits for clarity
plt.xlim(-0.16, 0.02)

plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

# ===============================
# SAVE
# ===============================
output_path = "results/plots/fig18_ablation_final.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"✅ Saved → {output_path}")

print("\n" + "="*60)
print("✅ FIGURE 18 READY (CORRECT)")
print("="*60)