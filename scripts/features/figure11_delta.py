# ===============================
# DELTA PLOT (FEATURE SHIFT)
# ===============================
import matplotlib.pyplot as plt
import numpy as np

# difference
diff = test_vals - train_vals

plt.figure(figsize=(10,5))
bars = plt.bar(feature_labels, diff)

# zero line
plt.axhline(0, linestyle="--")

# labels on bars
for i, v in enumerate(diff):
    plt.text(i, v + (0.5 if v > 0 else -0.8),
             f"{v:.1f}",
             ha='center',
             fontweight='bold')

plt.title("Feature Shift (Jan–Feb vs Oct–Nov)")
plt.ylabel("Change in Value")
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig("results/fig11_feature_shift_delta.png", dpi=150)
plt.close()

print("✅ Delta figure saved")