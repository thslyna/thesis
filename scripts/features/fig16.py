# ===============================
# FIGURE 16 — SHAP GROUPED IMPORTANCE
# ===============================

import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", context="talk")

os.makedirs('results/plots', exist_ok=True)

print("📊 FIGURE 16 — SHAP GROUPED BY CATEGORY")

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load('results/rf_untuned/rf_untuned_model.joblib')

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv('data/processed/flights_v1_IMPUTED_FIXED.csv')
df['hour'] = pd.to_datetime(df['hour'])

DROP_COLS = ["flight_date", "hour", "flight_iata", "delayed_15"]
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# ===============================
# TRANSFORM
# ===============================
preprocessor = model.named_steps['prep']
model_core = model.named_steps['model']

X_transformed = preprocessor.transform(X)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

feature_names = list(preprocessor.get_feature_names_out())

# ===============================
# SAMPLE
# ===============================
np.random.seed(42)
idx = np.random.choice(len(X_transformed), size=min(500, len(X_transformed)), replace=False)
X_sample = X_transformed[idx]

# ===============================
# SHAP
# ===============================
print("⏳ Computing SHAP values...")

explainer = shap.TreeExplainer(model_core)
shap_values = explainer.shap_values(X_sample, check_additivity=False)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap_array = np.array(shap_values)

if shap_array.ndim == 3:
    shap_array = shap_array[:, :, 0]

print("✅ SHAP computed")

# ===============================
# IMPORTANCE
# ===============================
mean_abs_shap = np.abs(shap_array).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": mean_abs_shap
})

# ===============================
# GROUPING LOGIC (KEY PART)
# ===============================
def categorize_feature(f):
    f = f.lower()

    if "weather" in f or "temp" in f or "wind" in f or "pressure" in f or "visibility" in f:
        return "Weather"

    elif "hour" in f or "day" in f or "month" in f:
        return "Temporal"

    elif "traffic" in f or "congestion" in f or "lag" in f or "prev" in f:
        return "Traffic"

    elif "airline" in f:
        return "Airline"

    elif "iata" in f or "airport" in f:
        return "Airport"

    else:
        return "Other"

importance_df["category"] = importance_df["feature"].apply(categorize_feature)

# ===============================
# AGGREGATE BY CATEGORY
# ===============================
grouped = (
    importance_df
    .groupby("category")["importance"]
    .sum()
    .sort_values(ascending=False)
)

# ===============================
# PLOT — FIGURE 16
# ===============================
fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.barh(
    grouped.index[::-1],
    grouped.values[::-1],
    color="#6c5ce7",
    edgecolor="black"
)

# labels
for bar in bars:
    val = bar.get_width()
    ax.text(
        val + max(grouped.values)*0.01,
        bar.get_y() + bar.get_height()/2,
        f"{val:.2f}",
        va='center',
        fontsize=11
    )

ax.set_title("Feature Category Importance (SHAP)", fontweight='bold')
ax.set_xlabel("Total Mean |SHAP value|")
ax.set_ylabel("Feature Category")

sns.despine()
ax.grid(axis='x', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("results/plots/fig16_shap_grouped.png", dpi=300)
plt.close()

print("✅ Saved → results/plots/fig16_shap_grouped.png")