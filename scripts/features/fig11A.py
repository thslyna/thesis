import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results/plots", exist_ok=True)

# ===============================
# DATA (FROM TABLE 4.11)
# ===============================
features = [
    "Prev Hour Delay",
    "24h Avg Delay",
    "Airline 7d Delay"
]

oct_nov = np.array([0.647, 0.774, 0.781])
jan_feb = np.array([0.572, 0.689, 0.712])

x = np.arange(len(features))
width = 0.35

# ===============================
# PLOT
# ===============================
plt.figure(figsize=(8,5))

bars1 = plt.bar(x - width/2, oct_nov, width, label="Oct–Nov (Train)", color="#3498db")
bars2 = plt.bar(x + width/2, jan_feb, width, label="Jan–Feb (Test)", color="#e74c3c")

plt.ylabel("Delay Rate")
plt.title("Historical Feature Shift (Autumn vs Winter)")
plt.xticks(x, features, rotation=20)
plt.legend()

# annotate
for bars in [bars1, bars2]:
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha='center',
            fontsize=10
        )

plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("results/plots/fig11a_historical_shift.png", dpi=150)
plt.close()

print("✅ Saved: fig11a_historical_shift.png")