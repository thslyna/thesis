#!/usr/bin/env python3
"""
aggregate_opensky_hourly.py

Aggregate OpenSky snapshot CSV files into hourly airspace congestion features.
"""

import pandas as pd
from pathlib import Path

# -------------------------
# Paths
# -------------------------
BASE = Path.home() / "Desktop" / "thesis"
OPENSKY_DIR = BASE / "data" / "raw" / "opensky"
OUT_FILE = BASE / "data" / "processed" / "opensky_hourly_congestion.csv"

# -------------------------
# Load all OpenSky snapshots
# -------------------------
files = sorted(OPENSKY_DIR.glob("opensky_ist_*.csv"))

if not files:
    raise SystemExit("❌ No OpenSky snapshot files found.")

print(f"📂 Found {len(files)} OpenSky snapshot files")

dfs = []

for f in files:
    df = pd.read_csv(f)

    # Ensure snapshot time is datetime
    df["snapshot_time_utc"] = pd.to_datetime(df["snapshot_time_utc"], utc=True)

    dfs.append(df)

# Combine all snapshots
data = pd.concat(dfs, ignore_index=True)
print("📦 Total OpenSky rows:", len(data))

# -------------------------
# Create hour column
# -------------------------
data["hour"] = data["snapshot_time_utc"].dt.floor("H")

# -------------------------
# Aggregate per hour
# -------------------------
hourly = (
    data.groupby("hour")
    .agg(
        opensky_avg_aircraft=("icao24", "count"),
        opensky_unique_aircraft=("icao24", "nunique"),
        opensky_airborne_ratio=("on_ground", lambda x: 1 - x.mean()),
        opensky_avg_speed=("velocity", "mean"),
        opensky_avg_altitude=("geo_altitude", "mean"),
        opensky_max_speed=("velocity", "max"),
    )
    .reset_index()
)

# Rename for clarity
hourly.rename(
    columns={
        "opensky_avg_aircraft": "opensky_aircraft_count",
    },
    inplace=True,
)

# -------------------------
# Save
# -------------------------
hourly.to_csv(OUT_FILE, index=False)

print("✅ Hourly OpenSky congestion features saved to:")
print(OUT_FILE)
print("Rows:", len(hourly))