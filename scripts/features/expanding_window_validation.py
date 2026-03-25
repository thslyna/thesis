import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

print("="*80)
print("🔄 EXPANDING WINDOW VALIDATION")
print("="*80)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv('data/processed/flights_v1_IMPUTED_FIXED.csv')
df['hour'] = pd.to_datetime(df['hour'])
df['month'] = df['hour'].dt.month

# ===============================
# FUNCTION TO TRAIN + EVAL
# ===============================
def run_experiment(train_months, test_months, name):
    train = df[df['month'].isin(train_months)]
    test = df[df['month'].isin(test_months)]

    print(f"\n{name}")
    print(f"Train months: {train_months}")
    print(f"Test months: {test_months}")
    print(f"Train size: {len(train)} | Test size: {len(test)}")

    DROP_COLS = ["delayed_15", "flight_date", "hour", "flight_iata", "month"]

    X_train = train.drop(columns=[c for c in DROP_COLS if c in train.columns])
    y_train = train['delayed_15']

    X_test = test.drop(columns=[c for c in DROP_COLS if c in test.columns])
    y_test = test['delayed_15']

    numeric_features = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

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

    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"👉 AUC: {auc:.4f}")
    return auc

# ===============================
# RUN EXPERIMENTS
# ===============================
results = {}

results["Oct → Nov"] = run_experiment([10], [11], "1️⃣ OCT → NOV")
results["Oct+Nov → Dec"] = run_experiment([10,11], [12], "2️⃣ OCT+NOV → DEC")
results["Oct+Nov+Dec → Jan-Feb"] = run_experiment([10,11,12], [1,2], "3️⃣ OCT+NOV+DEC → JAN-FEB")

# ===============================
# SUMMARY
# ===============================
print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)

for k, v in results.items():
    print(f"{k:30s}: {v:.4f}")

print("\n" + "="*80)
print("✅ DONE")
print("="*80)