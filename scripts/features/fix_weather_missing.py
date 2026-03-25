#!/usr/bin/env python3
"""
fix_weather_missing.py

Fix missing weather, temp, humidity, wind, etc. by:
- Sorting by hour
- Forward-filling (use earlier weather)
- Backward-filling (use later weather)
- Ensuring no weather columns are left NaN

Saves: training_features_weather_fixed.csv
"""

import pandas as pd
from pathlib import Path

BASE = Path.home() / "thesis"
IN_FILE = BASE / "data" / "processed" / "training_data.csv"
OUT_FILE = BASE / "data" / "processed" / "training_data_weather_fixed.csv"

print(f"📦 Loading: {IN_FILE}")
df = pd.read_csv(IN_FILE, parse_dates=["hour"], low_memory=False)
print("Shape:", df.shape)

# Weather-related columns
weather_cols = ["weather", "temp", "humidity", "pressure", "wind_speed"]

print("\n🔎 Missing values BEFORE fix:")
print(df[weather_cols].isna().sum())

# Sort by hour so filling is chronological
df = df.sort_values("hour")

# Fill weather category first
df["weather"] = df["weather"].ffill().bfill()

# Fill numeric weather values
for col in ["temp", "humidity", "pressure", "wind_speed"]:
    df[col] = df[col].ffill().bfill()

print("\n🔧 Missing values AFTER fix:")
print(df[weather_cols].isna().sum())

# Save new file
df.to_csv(OUT_FILE, index=False)
print(f"\n✅ Saved FIXED features → {OUT_FILE}")