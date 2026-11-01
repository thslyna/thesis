# ===============================
# FIGURE 14 — SEASONAL REGIME SHIFT (FINAL)
# ===============================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

sns.set_theme(style="whitegrid", context="talk")

os.makedirs('results/plots', exist_ok=True)

print("📊 FIGURE 14 — SEASONAL SHIFT")

# ===============================
# DATA (YOUR RESULTS)
# ===============================
season_data = pd.DataFrame({
    "Period": ["Oct–Nov", "December", "Jan–Feb"],
    "Effect": [7.07, 17.10, -7.78]
})

colors = ["#f4a261", "#d62828", "#2a9d8f"]

# ===============================
# PLOT
# ===============================
fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(
    season_data["Period"],
    season_data["Effect"],
    color=colors,
    edgecolor="black",
    linewidth=1.5
)

# Highlight December (peak)
bars[1].set_linewidth(2.5)

# Value labels
for bar, val in zip(bars, season_data["Effect"]):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        val + (0.8 if val > 0 else -0.8),
        f"{val:+.2f} pp",
        ha='center',
        va='bottom' if val > 0 else 'top',
        fontsize=12,
        fontweight='bold'
    )

# Zero line
ax.axhline(0, color='black', linewidth=1)

# ===============================
# STYLING (FIXED)
# ===============================
ax.set_title("Seasonal Regime Shifts in Delay Propagation Dynamics",
             fontweight='bold')

ax.set_ylabel("Propagation Effect (pp)")
ax.set_xlabel("Time Period")

# Balanced margins (prevents clipping)
ax.set_ylim(min(season_data["Effect"]) - 2,
            max(season_data["Effect"]) + 3)

sns.despine()
ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()

# ===============================
# SAVE
# ===============================
plt.savefig("results/plots/fig14_seasonal_shift.png", dpi=300)
plt.close()

print("✅ Saved → results/plots/fig14_seasonal_shift.png")