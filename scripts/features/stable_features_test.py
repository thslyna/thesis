import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

print("="*80)
print("🧪 STABLE FEATURES TEST")
print("="*80)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv('data/processed/flights_v1_IMPUTED_FIXED.csv')
df['hour'] = pd.to_datetime(df['hour'])
df['month'] = df['hour'].dt.month

# ===============================
# TEMPORAL SPLIT
# ===============================
train = df[df['month'].isin([10,11])]
test = df[df['month'].isin([1,2])]

print(f"\nTrain: {len(train)}")
print(f"Test: {len(test)}")

# ===============================
# SELECT ONLY STABLE FEATURES
# ===============================
FEATURES = [
    'hour_of_day',
    'day_of_week',
    'weather'
]

X_train = train[FEATURES]
y_train = train['delayed_15']

X_test = test[FEATURES]
y_test = test['delayed_15']

# ===============================
# PREPROCESS
# ===============================
numeric_features = ['hour_of_day', 'day_of_week']
categorical_features = ['weather']

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ===============================
# MODEL
# ===============================
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

# ===============================
# TRAIN
# ===============================
pipe.fit(X_train, y_train)

# ===============================
# EVALUATE
# ===============================
y_prob = pipe.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_prob)

print("\n📊 RESULT (Temporal Split):")
print(f"AUC: {auc:.4f}")

print("\n" + "="*80)
print("✅ DONE")
print("="*80)
