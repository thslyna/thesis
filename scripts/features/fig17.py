import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results/plots", exist_ok=True)

print("📊 FIGURE 17 — TRAFFIC FEATURE CORRELATION")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("data/processed/flights_v1_IMPUTED_FIXED.csv")

# ===============================
# FIND TRAFFIC FEATURES
# ===============================
traffic_features = [
    col for col in df.columns
    if "traffic" in col.lower() or "road" in col.lower()
]

if len(traffic_features) == 0:
    raise ValueError("❌ No traffic features found!")

print(f"🚗 Traffic features found: {len(traffic_features)}")

# ===============================
# CORRELATION WITH TARGET
# ===============================
correlations = {}

for col in traffic_features:
    try:
        corr = df[col].corr(df["delayed_15"])
        correlations[col] = corr
    except:
        continue

corr_df = pd.DataFrame.from_dict(correlations, orient="index", columns=["correlation"])
corr_df = corr_df.sort_values("correlation")

# ===============================
# PLOT
# ===============================
fig, ax = plt.subplots(figsize=(10, 6))

colors = ["#e74c3c" if v > 0 else "#3498db" for v in corr_df["correlation"]]

bars = ax.barh(
    corr_df.index,
    corr_df["correlation"],
    color=colors,
    edgecolor="black"
)

ax.axvline(0, color="black", linewidth=1)
ax.set_title("Traffic Feature Correlation with Delay", fontsize=16, fontweight="bold")
ax.set_xlabel("Pearson Correlation with Delay (delayed_15)", fontsize=12, fontweight="bold")
ax.set_ylabel("Traffic Features", fontsize=12, fontweight="bold")

# value labels
for bar in bars:
    width = bar.get_width()
    ax.text(
        width,
        bar.get_y() + bar.get_height()/2,
        f"{width:.3f}",
        va="center",
        fontsize=9
    )

plt.tight_layout()
plt.savefig("results/plots/fig17_traffic_correlation.png", dpi=150)
plt.close()

print("✅ Saved → results/plots/fig17_traffic_correlation.png")