import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

os.makedirs('results/temporal_9k', exist_ok=True)

print("="*80)
print("⏳ TEMPORAL VALIDATION - 9.5K DATASET")
print("="*80)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv('data/processed/flights_v1_IMPUTED_FIXED.csv')
df['hour'] = pd.to_datetime(df['hour'])

print(f"📦 Dataset: {len(df):,} flights")

# ===============================
# CREATE TIME SPLIT
# ===============================
df['month'] = df['hour'].dt.month

# TRAIN: Oct-Nov (10, 11)
train = df[df['month'].isin([10, 11])]

# TEST: Jan-Feb (1, 2)
test = df[df['month'].isin([1, 2])]

print(f"\n📊 Train size (Oct-Nov): {len(train)}")
print(f"📊 Test size (Jan-Feb): {len(test)}")

# ===============================
# FEATURES
# ===============================
DROP_COLS = ["delayed_15", "flight_date", "hour", "flight_iata", "month"]

X_train = train.drop(columns=[c for c in DROP_COLS if c in train.columns])
y_train = train['delayed_15']

X_test = test.drop(columns=[c for c in DROP_COLS if c in test.columns])
y_test = test['delayed_15']

# ===============================
# FEATURE TYPES
# ===============================
numeric_features = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

# ===============================
# PREPROCESSING
# ===============================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ===============================
# MODEL (SAME AS YOUR FINAL)
# ===============================
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    tree_method="hist",
    n_jobs=-1
)

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

# ===============================
# TRAIN
# ===============================
print("\n🚀 Training on Oct-Nov...")
pipe.fit(X_train, y_train)

# ===============================
# TEST ON FUTURE (Jan-Feb)
# ===============================
print("🔮 Testing on Jan-Feb...")

y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:,1]

# ===============================
# METRICS
# ===============================
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 TEMPORAL VALIDATION RESULTS:")
print(f"ROC-AUC:   {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# ===============================
# SAVE RESULTS
# ===============================
results = {
    "roc_auc": float(auc),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "train_size": len(train),
    "test_size": len(test)
}

import json
with open('results/temporal_9k/metrics.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\n💾 Saved: results/temporal_9k/metrics.json")

# ===============================
# SIMPLE PLOT
# ===============================
metrics_names = ['ROC-AUC','Precision','Recall','F1']
values = [auc, precision, recall, f1]

plt.figure()
plt.bar(metrics_names, values)
plt.ylim(0,1)
plt.title("Temporal Validation (Train: Oct-Nov → Test: Jan-Feb)")
plt.savefig('results/temporal_9k/performance.png', dpi=150)

print("📊 Saved: performance.png")

print("\n" + "="*80)
print("✅ TEMPORAL VALIDATION COMPLETE!")
print("="*80)

