# ===============================
# XGB UNTUNED — PHASE 2 FINAL
# ===============================

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

DATA_PATH = "/Users/lyna.thc/Desktop/thesis/data/processed/flights_v1_IMPUTED_FIXED.csv"
OUT_DIR = "/Users/lyna.thc/Desktop/thesis/results/xgb_untuned_v2"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

y = df["delayed_15"]
X = df.drop(columns=["delayed_15", "flight_date", "hour", "flight_iata"])

numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])

model = Pipeline([
    ("prep", preprocessor),
    ("model", XGBClassifier(eval_metric="logloss", random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print("XGB Untuned AUC:", auc)

pd.DataFrame({
    "y_true": y_test.values,
    "y_prob": y_prob
}).to_csv(os.path.join(OUT_DIR, "xgb_untuned_predictions.csv"), index=False)