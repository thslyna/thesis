import matplotlib.pyplot as plt
import os

os.makedirs("results/plots", exist_ok=True)

print("="*80)
print("📊 FIGURE 20 — SLIDING WINDOW VALIDATION")
print("="*80)

# ✅ REAL RESULTS
labels = [
    "Oct → Nov",
    "Nov → Dec",
    "Dec → Jan–Feb"
]

values = [0.5224, 0.4984, 0.5624]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(labels, values, marker='o', linewidth=3, markersize=10)

# Styling
ax.set_title("Sliding Window Validation (AUC)", fontsize=16, fontweight="bold")
ax.set_ylabel("AUC Score", fontsize=12)
ax.set_xlabel("Training → Testing Period", fontsize=12)

ax.set_ylim(0.45, 0.60)
ax.grid(True, alpha=0.3)

# Annotate values
for i, v in enumerate(values):
    ax.text(i, v + 0.004, f"{v:.3f}", ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("results/plots/fig20_sliding_window.png", dpi=150)
plt.close()

print("✅ Saved → results/plots/fig20_sliding_window.png")