# ==========================================================
# 📊 FIGURE 23 — PERFORMANCE BY WEATHER (FINAL, CORRECT)
# ==========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

os.makedirs("results/plots", exist_ok=True)

print("="*80)
print("📊 FIGURE 23 — PERFORMANCE BY WEATHER (BALANCED ACCURACY)")
print("="*80)

# ==========================================================
# 1. LOAD DATA
# ==========================================================
df = pd.read_csv("data/processed/flights_v1_IMPUTED_FIXED.csv")
df["hour"] = pd.to_datetime(df["hour"])
df = df.sort_values("hour").reset_index(drop=True)

# ==========================================================
# 2. TIME-BASED SPLIT
# ==========================================================
train_size = int(len(df) * 0.75)
train = df.iloc[:train_size].copy()
test  = df.iloc[train_size:].copy()

print(f"Train: {train.shape} | Test: {test.shape}")

# ==========================================================
# 3. FEATURES
# ==========================================================
DROP_COLS = ["delayed_15", "flight_date", "hour", "flight_iata"]

X_train = train.drop(columns=[c for c in DROP_COLS if c in train.columns])
y_train = train["delayed_15"]

X_test = test.drop(columns=[c for c in DROP_COLS if c in test.columns])
y_test = test["delayed_15"]

# ==========================================================
# 4. PREPROCESSING
# ==========================================================
numeric_features     = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# ==========================================================
# 5. MODEL
# ==========================================================
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    tree_method="hist",
    n_jobs=-1
)

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

print("\n⏳ Training model...")
pipe.fit(X_train, y_train)

# ==========================================================
# 6. PREDICTIONS
# ==========================================================
y_pred = pipe.predict(X_test)

test = test.copy()
test["y_pred"] = y_pred
test["y_true"] = y_test.values

# ==========================================================
# 7. WEATHER CATEGORIES (SAME LOGIC, BUT CLEAN)
# ==========================================================
def weather_category(row):
    if row["wind_speed"] > 8 or row["humidity"] > 90:
        return "Severe"
    elif row["wind_speed"] > 4 or row["humidity"] > 75:
        return "Moderate"
    else:
        return "Clear"

test["weather_cat"] = test.apply(weather_category, axis=1)

# ==========================================================
# 8. PERFORMANCE BY WEATHER (FIXED)
# ==========================================================
MIN_SAMPLES = 100

weather_stats = []

for cat in ["Clear", "Moderate", "Severe"]:
    subset = test[test["weather_cat"] == cat]
    count = len(subset)

    if count >= MIN_SAMPLES:
        score = balanced_accuracy_score(subset["y_true"], subset["y_pred"])

        weather_stats.append({
            "weather": cat,
            "score": score,
            "count": count
        })

weather_df = pd.DataFrame(weather_stats)

print("\n📊 Weather performance:")
print(weather_df)

# ==========================================================
# 9. PLOT (CLEAN + DEFENSIBLE)
# ==========================================================
fig, ax = plt.subplots(figsize=(8,5))

colors = ["#4CAF50", "#FF9800", "#F44336"]

bars = ax.bar(weather_df["weather"], weather_df["score"], color=colors)

ax.set_ylabel("Balanced Accuracy", fontsize=12)
ax.set_ylim(0.4, 0.8)

# annotate
for bar, val in zip(bars, weather_df["score"]):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 0.02,
            f"{val:.2f}",
            ha="center",
            fontsize=10)

plt.title("Model Performance by Weather Conditions", fontweight="bold")

plt.tight_layout()

# ==========================================================
# 10. SAVE
# ==========================================================
output_path = "results/plots/fig23_accuracy_by_weather.png"
plt.savefig(output_path, dpi=150)
plt.close()

print(f"\n✅ Saved → {output_path}")

print("\n" + "="*80)
print("✅ FIGURE 23 COMPLETE (FINAL)")
print("="*80)