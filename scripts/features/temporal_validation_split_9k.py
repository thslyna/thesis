import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import os
import matplotlib.pyplot as plt
import json

os.makedirs('results/temporal_9k_split', exist_ok=True)
os.makedirs('results/final_plots', exist_ok=True)

print("="*80)
print("⏳ TEMPORAL VALIDATION (SPLIT VERSION)")
print("="*80)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv('data/processed/flights_v1_IMPUTED_FIXED.csv')
df['hour'] = pd.to_datetime(df['hour'])
df['month'] = df['hour'].dt.month

# ===============================
# SPLITS
# ===============================
train = df[df['month'].isin([10,11])]
test_dec = df[df['month'] == 12]
test_future = df[df['month'].isin([1,2])]

print(f"\n📊 Train (Oct-Nov): {len(train)}")
print(f"📊 Test Dec: {len(test_dec)}")
print(f"📊 Test Jan-Feb: {len(test_future)}")

# ===============================
# FEATURES
# ===============================
DROP_COLS = ["delayed_15", "flight_date", "hour", "flight_iata", "month"]

X_train = train.drop(columns=[c for c in DROP_COLS if c in train.columns])
y_train = train['delayed_15']

# ===============================
# PREPROCESS
# ===============================
numeric_features = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ===============================
# MODEL
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
print("\n🚀 Training model...")
pipe.fit(X_train, y_train)

# ===============================
# FUNCTION TO EVALUATE
# ===============================
def evaluate(name, data):
    X = data.drop(columns=[c for c in DROP_COLS if c in data.columns])
    y = data['delayed_15']

    y_pred = pipe.predict(X)
    y_prob = pipe.predict_proba(X)[:,1]

    auc = roc_auc_score(y, y_prob)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"\n📊 {name}:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")

    return {
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "size": len(data)
    }

# ===============================
# EVALUATE
# ===============================
results = {
    "december": evaluate("December (near future)", test_dec),
    "jan_feb": evaluate("Jan-Feb (far future)", test_future)
}

# ===============================
# SAVE JSON
# ===============================
with open('results/temporal_9k_split/results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\n💾 Saved: results/temporal_9k_split/results.json")

# ===============================
# 📊 FINAL PLOT: TEMPORAL PERFORMANCE
# ===============================
labels = ['Random Split', 'December', 'Jan–Feb']
values = [
    0.8847,  # baseline from random split
    results['december']['auc'],
    results['jan_feb']['auc']
]

plt.figure()
bars = plt.bar(labels, values)

plt.ylim(0,1)
plt.ylabel("ROC-AUC")
plt.title("Temporal Generalization Performance")

for bar, v in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.2f}", ha='center')

plt.tight_layout()
plt.savefig('results/final_plots/temporal_performance.png', dpi=150)

print("📊 Saved: results/final_plots/temporal_performance.png")

print("\n" + "="*80)
print("✅ TEMPORAL SPLIT ANALYSIS COMPLETE!")
print("="*80)