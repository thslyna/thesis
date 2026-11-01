# ==========================================================
# TRAIN MODEL FOR SHAP (CLEAN FEATURES)
# ==========================================================
import pandas as pd
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

os.makedirs("results/shap_model", exist_ok=True)

print("="*80)
print("🚀 TRAINING SHAP-CLEAN MODEL")
print("="*80)

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv("data/processed/flights_v1_IMPUTED_FIXED.csv")
df["hour"] = pd.to_datetime(df["hour"])

# ==========================================================
# DROP WRONG FEATURES (CRITICAL)
# ==========================================================
DROP_COLS = [
    "delayed_15",
    "flight_date",
    "hour",
    "flight_iata",

    # ❌ REMOVE THESE
    "arr_iata",
    "airline_name"
]

X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
y = df["delayed_15"]

print(f"📊 Features after cleaning: {X.shape}")

# ==========================================================
# SAME SPLIT AS BEFORE (CONSISTENT)
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ==========================================================
# PREPROCESS
# ==========================================================
num_features = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_features = X_train.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", "passthrough", num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
])

# ==========================================================
# MODEL
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

print("\n⏳ Training...")
pipe.fit(X_train, y_train)

# ==========================================================
# SAVE
# ==========================================================
joblib.dump(pipe, "results/shap_model/xgb_shap_clean.joblib")

print("\n✅ Saved → results/shap_model/xgb_shap_clean.joblib")