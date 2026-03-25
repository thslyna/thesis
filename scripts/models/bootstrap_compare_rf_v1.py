import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ===============================
# LOAD DATA
# ===============================
DATA_PATH = "/Users/lyna.thc/Desktop/thesis/data/processed/flights_with_weather_road_history.csv"

df = pd.read_csv(DATA_PATH)

y = df["delayed_15"]

DROP_COLS = ["delayed_15", "flight_date", "hour", "flight_iata"]
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ===============================
# TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# ===============================
# DEFINE MODELS
# ===============================

# Untuned RF
rf_untuned = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

# Tuned RF (your real params)
rf_tuned = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features="log2",
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# Build pipelines
pipe_untuned = Pipeline([("prep", preprocessor), ("model", rf_untuned)])
pipe_tuned = Pipeline([("prep", preprocessor), ("model", rf_tuned)])

pipe_untuned.fit(X_train, y_train)
pipe_tuned.fit(X_train, y_train)

y_prob_untuned = pipe_untuned.predict_proba(X_test)[:, 1]
y_prob_tuned = pipe_tuned.predict_proba(X_test)[:, 1]

# ===============================
# BOOTSTRAP
# ===============================
n_bootstrap = 1000
rng = np.random.default_rng(42)

auc_untuned = []
auc_tuned = []

for _ in range(n_bootstrap):
    indices = rng.choice(len(y_test), len(y_test), replace=True)
    
    auc_untuned.append(
        roc_auc_score(y_test.iloc[indices], y_prob_untuned[indices])
    )
    
    auc_tuned.append(
        roc_auc_score(y_test.iloc[indices], y_prob_tuned[indices])
    )

# Convert to arrays
auc_untuned = np.array(auc_untuned)
auc_tuned = np.array(auc_tuned)

# ===============================
# RESULTS
# ===============================
def summarize(name, values):
    print(f"\n{name}")
    print("Mean AUC:", round(values.mean(), 4))
    print("Std AUC:", round(values.std(), 4))
    print("95% CI:", np.percentile(values, [2.5, 97.5]))

summarize("Untuned RF", auc_untuned)
summarize("Tuned RF", auc_tuned)

# Difference distribution
diff = auc_untuned - auc_tuned

print("\nDifference (Untuned - Tuned)")
print("Mean difference:", round(diff.mean(), 4))
print("95% CI:", np.percentile(diff, [2.5, 97.5]))