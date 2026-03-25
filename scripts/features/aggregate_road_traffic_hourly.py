import pandas as pd
from pathlib import Path

BASE = Path.home() / "thesis"

IN_FILE = BASE / "data" / "raw" / "ibb_hourly_traffic_all.csv"
OUT_FILE = BASE / "data" / "processed" / "road_traffic_hourly_agg.csv"

print("📦 Loading road traffic data (this may take a bit)...")
df = pd.read_csv(IN_FILE, parse_dates=["DATE_TIME"])

# ---- OPTIONAL: restrict to Istanbul Airport area (rough box) ----
# IST approx: lat 41.26, lon 28.74
df = df[
    (df["LATITUDE"].between(41.15, 41.40)) &
    (df["LONGITUDE"].between(28.60, 28.90))
]

print("📍 After airport-area filter:", df.shape)

# ---- Aggregate hourly ----
hourly = (
    df
    .groupby(df["DATE_TIME"].dt.floor("H"))
    .agg(
        road_avg_speed=("AVERAGE_SPEED", "mean"),
        road_vehicle_count=("NUMBER_OF_VEHICLES", "sum"),
        road_min_speed=("MINIMUM_SPEED", "mean"),
        road_max_speed=("MAXIMUM_SPEED", "mean"),
    )
    .reset_index()
    .rename(columns={"DATE_TIME": "hour"})
)

# ---- Simple congestion proxy ----
hourly["road_congestion_index"] = (
    hourly["road_vehicle_count"] / (hourly["road_avg_speed"] + 1)
)

hourly.to_csv(OUT_FILE, index=False)

print("✅ Saved hourly road traffic features to:")
print(OUT_FILE)
print("Rows:", len(hourly))