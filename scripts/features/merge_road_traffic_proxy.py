import pandas as pd
from pathlib import Path

BASE = Path.home() / "thesis"

FLIGHTS_FILE = BASE / "data" / "processed" / "final_training_data.csv"
ROAD_FILE = BASE / "data" / "processed" / "road_traffic_hourly_agg.csv"
OUT_FILE = BASE / "data" / "processed" / "final_training_data_with_road_traffic_proxy.csv"

print("📦 Loading flight data...")
flights = pd.read_csv(FLIGHTS_FILE)
flights["hour"] = pd.to_datetime(flights["hour"])

print("📦 Loading road traffic data...")
road = pd.read_csv(ROAD_FILE)
road["hour"] = pd.to_datetime(road["hour"])

# --------------------------------------------------
# 1️⃣ Extract time patterns
# --------------------------------------------------
flights["hour_of_day"] = flights["hour"].dt.hour
flights["day_of_week"] = flights["hour"].dt.dayofweek

road["hour_of_day"] = road["hour"].dt.hour
road["day_of_week"] = road["hour"].dt.dayofweek

# --------------------------------------------------
# 2️⃣ Build historical traffic PROFILES
# --------------------------------------------------
print("📊 Computing historical road traffic patterns...")

road_profile = (
    road
    .groupby(["hour_of_day", "day_of_week"])
    .agg(
        road_avg_speed_proxy=("road_avg_speed", "mean"),
        road_vehicle_count_proxy=("road_vehicle_count", "mean"),
        road_congestion_index_proxy=("road_congestion_index", "mean"),
    )
    .reset_index()
)

print("Road profile shape:", road_profile.shape)

# --------------------------------------------------
# 3️⃣ Merge proxy features into flights
# --------------------------------------------------
print("🔗 Merging road traffic proxy features...")

final_df = flights.merge(
    road_profile,
    on=["hour_of_day", "day_of_week"],
    how="left"
)

# --------------------------------------------------
# 4️⃣ Save
# --------------------------------------------------
final_df.to_csv(OUT_FILE, index=False)

print("✅ Merge complete!")
print("Saved to:", OUT_FILE)
print("Final shape:", final_df.shape)

print("\n🧪 Missing values in proxy columns:")
print(
    final_df[
        [
            "road_avg_speed_proxy",
            "road_vehicle_count_proxy",
            "road_congestion_index_proxy"
        ]
    ].isna().mean() * 100
)