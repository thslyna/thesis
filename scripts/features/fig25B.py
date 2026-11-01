# ==========================================================
# FIGURE 25B — COST CONTRIBUTION (IMPROVED)
# ==========================================================
import matplotlib.pyplot as plt
import numpy as np

# Your values (from model)
fp_cost = 311
fn_cost = 273

labels = ["False Alarms (FP)", "Missed Delays (FN)"]
values = [fp_cost, fn_cost]

total = sum(values)
percentages = [v / total * 100 for v in values]

# ==========================================================
# PLOT
# ==========================================================
plt.figure(figsize=(7,5))

colors = ["#f39c12", "#e74c3c"]  # orange = FP, red = FN
bars = plt.bar(labels, values, color=colors, edgecolor="black")

plt.title("Cost Contribution by Error Type", fontsize=13, fontweight="bold")
plt.ylabel("Total Cost (Relative Units)")

# Add value + percentage labels
for i, (bar, v, p) in enumerate(zip(bars, values, percentages)):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        v + max(values)*0.03,
        f"{v} ({p:.1f}%)",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

# Add total annotation
plt.text(
    0.5, max(values)*1.15,
    f"Total Cost = {total}",
    ha="center",
    fontsize=11,
    transform=plt.gca().transAxes
)

plt.ylim(0, max(values)*1.25)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results/plots/fig25b_cost_breakdown.png", dpi=150)
plt.close()

print("✅ Improved plot saved")