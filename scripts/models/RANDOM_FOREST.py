import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib  # ← ADD THIS for saving model

# ===============================
# PATHS
# ===============================
DATA_PATH = "/Users/lyna.thc/Desktop/thesis/data/processed/flights_with_weather_road_history.csv"
OUT_DIR = "/Users/lyna.thc/Desktop/thesis/results/rf_untuned"  # ← CHANGE NAME
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH)

print("📦 Dataset loaded")
print("Shape:", df.shape)

# ===============================
# TARGET
# ===============================
y = df["delayed_15"]

print("\n🎯 Target distribution:")
print(y.value_counts(normalize=True))

# ===============================
# FEATURES TO DROP
# ===============================
DROP_COLS = [
    "flight_date",
    "hour",
    "flight_iata",
    "delayed_15",
]

X = df.drop(columns=DROP_COLS)

# ===============================
# FEATURE TYPES
# ===============================
numeric_features = [
    "hour_of_day",
    "day_of_week",
    "temp",
    "humidity",
    "pressure",
    "wind_speed",
    "road_avg_speed",
    "road_vehicle_count",
    "road_congestion_index",
    "prev_hour_delay_rate",
    "avg_delay_last_24h",
    "airline_7d_delay_rate",
]

categorical_features = [
    "airline_name",
    "dep_iata",
    "arr_iata",
    "weather",
]

print("\n🧱 Numeric features:", numeric_features)
print("🧱 Categorical features:", categorical_features)

# ===============================
# PREPROCESSING
# ===============================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),  # ← Fixed order
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ===============================
# MODEL (UNTUNED - SKLEARN DEFAULTS ONLY!)
# ===============================
rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

# ===============================
# PIPELINE
# ===============================
pipeline = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("model", rf),
    ]
)

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\n📊 Train size:", X_train.shape)
print("📊 Test size:", X_test.shape)

# ===============================
# TRAIN
# ===============================
print("\n🚀 Training Random Forest (Untuned)...")
pipeline.fit(X_train, y_train)

# ===============================
# EVALUATION
# ===============================
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", round(roc_auc, 4))

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(pipeline, os.path.join(OUT_DIR, "rf_untuned_model.joblib"))
print(f"\n💾 Model saved")

# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – Random Forest (Untuned)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png")
plt.close()

# ===============================
# ROC CURVE
# ===============================
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Random Forest (Untuned)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve.png")
plt.close()

# ===============================
# FEATURE IMPORTANCE
# ===============================
feature_names = pipeline.named_steps["prep"].get_feature_names_out()
importances = pipeline.named_steps["model"].feature_importances_

fi = (
    pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    .sort_values("importance", ascending=False)
    .head(20)
)

plt.figure(figsize=(8, 6))
sns.barplot(x="importance", y="feature", data=fi)
plt.title("Top Random Forest (Untuned) Feature Importances")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png")
plt.close()

print("\n✅ Random Forest (Untuned) training finished")
print("📁 Results saved to:", OUT_DIR)