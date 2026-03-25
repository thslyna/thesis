#!/usr/bin/env python3
"""
merge_opensky_features.py

Merge hourly OpenSky congestion features into the baseline
training_features_with_history.csv without modifying the original file.

Output:
  training_features_with_history_opensky.csv
"""

from pathlib import Path
import pandas as pd

# -------------------------
# Paths
# -------------------------
BASE = Path.home() / "thesis"

FLIGHTS_FILE = BASE / "data" / "processed" / "training_features_with_history.csv"
OPENSKY_FILE = BASE / "data" / "processed" / "opensky_hourly_congestion.csv"

OUT_FILE = BASE / "data" / "processed" / "training_features_with_history_opensky.csv"

# -------------------------
# Load data
# -------------------------
print("📦 Loading baseline flight features...")
df = pd.read_csv(FLIGHTS_FILE, low_memory=False)

print("📦 Loading OpenSky hourly congestion...")
opensky = pd.read_csv(OPENSKY_FILE)

print(f"Flights shape: {df.shape}")
print(f"OpenSky shape: {opensky.shape}")

# -------------------------
# Parse time columns
# -------------------------
df["hour"] = pd.to_datetime(df["hour"], utc=True, errors="coerce").dt.floor("h")
opensky["hour"] = pd.to_datetime(opensky["hour"], utc=True, errors="coerce")

# -------------------------
# Merge (LEFT JOIN = safe)
# -------------------------
print("🔗 Merging OpenSky features by hour...")
merged = df.merge(
    opensky,
    on="hour",
    how="left"
)

# -------------------------
# Fill missing OpenSky values
# -------------------------
opensky_cols = [c for c in opensky.columns if c != "hour"]

for c in opensky_cols:
    merged[c] = merged[c].fillna(0)

# -------------------------
# Save
# -------------------------
merged.to_csv(OUT_FILE, index=False)

print("\n✅ Merge complete!")
print(f"Saved to: {OUT_FILE}")
print(f"Final shape: {merged.shape}")