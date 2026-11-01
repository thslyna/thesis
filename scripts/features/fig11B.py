import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results/plots", exist_ok=True)

# ===============================
# DATA (FROM TABLE 4.12)
# ===============================
features = [
    "Temperature (°C)",
    "Humidity (%)",
    "Pressure (hPa)",
    "Wind Speed (m/s)"
]

oct_nov = np.array([18.2, 64.3, 1015.4, 4.2])
jan_feb = np.array([6.8, 71.2, 1018.7, 5.8])

x = np.arange(len(features))
width = 0.35

# ===============================
# PLOT
# ===============================
plt.figure(figsize=(9,5))

bars1 = plt.bar(x - width/2, oct_nov, width, label="Oct–Nov (Train)", color="#3498db")
bars2 = plt.bar(x + width/2, jan_feb, width, label="Jan–Feb (Test)", color="#e67e22")

plt.ylabel("Value")
plt.title("Weather Feature Shift (Autumn vs Winter)")
plt.xticks(x, features, rotation=20)
plt.legend()

# annotate
for bars in [bars1, bars2]:
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + (2 if bar.get_height() > 100 else 0.5),
            f"{bar.get_height():.1f}",
            ha='center',
            fontsize=10
        )

plt.tight_layout()
plt.savefig("results/plots/fig11b_weather_shift.png", dpi=150)
plt.close()

print("✅ Saved: fig11b_weather_shift.png")