import pandas as pd

df = pd.read_csv("~/Desktop/thesis/data/processed/flights_v1_REGIME_SHIFT.csv")

print(df["delayed_15"].value_counts())
print(df["delayed_15"].value_counts(normalize=True))