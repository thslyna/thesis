import pandas as pd
import numpy as np

print("🚀 CREATING V1 WITH SMART WEATHER IMPUTATION (LEAKAGE-FREE VERSION)\n")
print("="*70)

# ===============================
# LOAD FLIGHTS
# ===============================
df_flights = pd.read_csv("data/raw/flights_all.csv", low_memory=False)
df_flights['date'] = pd.to_datetime(df_flights['flight_date'])

# Filter Oct-Feb
df = df_flights[
    (df_flights['date'] >= '2025-10-01') & 
    (df_flights['date'] < '2026-03-01')
].copy()

print(f"📦 Raw Oct-Feb flights: {len(df):,}")

# ===============================
# CALCULATE DELAYS
# ===============================
print("\n⏱️  Calculating delays...")

df['delay_from_column'] = pd.to_numeric(df['dep_delay'], errors='coerce')

df['dep_scheduled_dt'] = pd.to_datetime(df['dep_scheduled'], utc=True, errors='coerce')
df['dep_actual_dt'] = pd.to_datetime(df['dep_actual'], utc=True, errors='coerce')

df['delay_from_timestamps'] = (
    df['dep_actual_dt'] - df['dep_scheduled_dt']
).dt.total_seconds() / 60

df['dep_delay'] = df['delay_from_column'].fillna(df['delay_from_timestamps'])

df_labeled = df[df['dep_delay'].notna()].copy()

print(f"✅ Flights with valid delay: {len(df_labeled):,}")

# ===============================
# TIME FEATURES
# ===============================
print("\n🕐 Adding time features...")

df_labeled['hour'] = pd.to_datetime(df_labeled['dep_scheduled'], utc=True).dt.floor('h')
df_labeled['hour_of_day'] = pd.to_datetime(df_labeled['dep_scheduled'], utc=True).dt.hour
df_labeled['day_of_week'] = pd.to_datetime(df_labeled['flight_date']).dt.dayofweek

df_labeled['delayed_15'] = (df_labeled['dep_delay'] >= 15).astype(int)

# Remove NULL hour ONLY (this is fine)
df_labeled = df_labeled[df_labeled['hour'].notna()].copy()

# ===============================
# LOAD WEATHER
# ===============================
print("\n🌤️  Loading weather data...")

df_weather = pd.read_csv("data/processed/weather_hourly_COMBINED.csv")
df_weather['hour'] = pd.to_datetime(df_weather['hour'], utc=True)

df_weather = df_weather.sort_values('hour').reset_index(drop=True)
df_labeled = df_labeled.sort_values('hour').reset_index(drop=True)

# ===============================
# MERGE WEATHER (NO DROPPING)
# ===============================
print("\n🧠 Matching weather (NO DATA LOSS)...")

df_merged = pd.merge_asof(
    df_labeled,
    df_weather,
    on='hour',
    direction='backward',
    tolerance=pd.Timedelta(hours=4)
)

print(f"✅ After merge: {len(df_merged):,} flights")

# ===============================
# FIX: IMPUTE WEATHER INSTEAD OF DROPPING
# ===============================
print("\n🧪 Imputing missing weather...")

df_merged['temp'] = df_merged['temp'].fillna(df_merged['temp'].median())
df_merged['humidity'] = df_merged['humidity'].fillna(df_merged['humidity'].median())
df_merged['pressure'] = df_merged['pressure'].fillna(df_merged['pressure'].median())
df_merged['wind_speed'] = df_merged['wind_speed'].fillna(df_merged['wind_speed'].median())
df_merged['weather'] = df_merged['weather'].fillna("Unknown")

df_with_weather = df_merged.copy()

# ===============================
# ADD ROAD TRAFFIC
# ===============================
print("\n🚗 Adding road traffic...")

df_road = pd.read_csv("data/processed/road_traffic_hour_profile.csv")

df_with_weather = df_with_weather.merge(
    df_road[['hour_of_day', 'road_avg_speed', 'road_vehicle_count', 'road_congestion_index']],
    on='hour_of_day',
    how='left'
)

# Fill road missing (safe)
df_with_weather['road_avg_speed'] = df_with_weather['road_avg_speed'].fillna(df_with_weather['road_avg_speed'].median())
df_with_weather['road_vehicle_count'] = df_with_weather['road_vehicle_count'].fillna(df_with_weather['road_vehicle_count'].median())
df_with_weather['road_congestion_index'] = df_with_weather['road_congestion_index'].fillna(df_with_weather['road_congestion_index'].median())

# ===============================
# HISTORICAL FEATURES (LEAKAGE-FREE)
# ===============================
print("\n📊 Adding historical features (STRICT PAST ONLY)...")

df_with_weather = df_with_weather.sort_values('hour').reset_index(drop=True)

df_with_weather['prev_hour_delay_rate'] = 0.0
df_with_weather['avg_delay_last_24h'] = 0.0
df_with_weather['airline_7d_delay_rate'] = 0.0

for idx, row in df_with_weather.iterrows():
    current_time = row['hour']
    
    past = df_with_weather[df_with_weather['hour'] < current_time]

    if len(past) > 0:

        # prev hour
        prev = past[
            (past['hour'] >= current_time - pd.Timedelta(hours=1)) &
            (past['hour'] < current_time)
        ]
        if len(prev) > 0:
            df_with_weather.at[idx, 'prev_hour_delay_rate'] = prev['delayed_15'].mean()

        # last 24h
        last24 = past[past['hour'] >= current_time - pd.Timedelta(hours=24)]
        if len(last24) > 0:
            df_with_weather.at[idx, 'avg_delay_last_24h'] = last24['delayed_15'].mean()

        # airline 7d
        airline = past[
            (past['airline_name'] == row['airline_name']) &
            (past['hour'] >= current_time - pd.Timedelta(days=7))
        ]
        if len(airline) > 0:
            df_with_weather.at[idx, 'airline_7d_delay_rate'] = airline['delayed_15'].mean()
        else:
            df_with_weather.at[idx, 'airline_7d_delay_rate'] = past['delayed_15'].mean()

    if idx % 1000 == 0:
        print(f"   {idx:,}/{len(df_with_weather):,}", end='\r')

print("\n✅ Historical features done")

# ===============================
# FINAL DATASET (NO DROPPING)
# ===============================
print("\n📋 Creating final dataset...")

final_cols = [
    'flight_date', 'airline_name', 'flight_iata', 'dep_iata', 'arr_iata',
    'hour', 'hour_of_day', 'day_of_week', 'delayed_15',
    'temp', 'humidity', 'pressure', 'wind_speed', 'weather',
    'road_avg_speed', 'road_vehicle_count', 'road_congestion_index',
    'prev_hour_delay_rate', 'avg_delay_last_24h', 'airline_7d_delay_rate'
]

df_final = df_with_weather[final_cols]

# ===============================
# SAVE
# ===============================
output_path = "data/processed/flights_v1_IMPUTED_FIXED.csv"
df_final.to_csv(output_path, index=False)

print("\n" + "="*70)
print("🎉 DATASET FIXED (NO BIAS + NO LEAKAGE)")
print("="*70)
print(f"📁 Saved to: {output_path}")
print(f"📦 Total flights: {len(df_final):,}")

# ===============================
# CHECK DISTRIBUTION
# ===============================
print("\n📊 FINAL DELAY DISTRIBUTION:")
print(df_final['delayed_15'].value_counts(normalize=True))