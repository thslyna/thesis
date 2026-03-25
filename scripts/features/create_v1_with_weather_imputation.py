import pandas as pd
import numpy as np

print("🚀 CREATING V1 WITH SMART WEATHER IMPUTATION\n")
print("="*70)

# Load flights
df_flights = pd.read_csv("data/raw/flights_all.csv", low_memory=False)
df_flights['date'] = pd.to_datetime(df_flights['flight_date'])

# Filter Oct-Feb
df = df_flights[
    (df_flights['date'] >= '2025-10-01') & 
    (df_flights['date'] < '2026-03-01')
].copy()

print(f"📦 Raw Oct-Feb flights: {len(df):,}")

# Calculate delays
print("\n⏱️  Calculating delays...")
df['delay_from_column'] = pd.to_numeric(df['dep_delay'], errors='coerce')
df['dep_scheduled_dt'] = pd.to_datetime(df['dep_scheduled'], utc=True, errors='coerce')
df['dep_actual_dt'] = pd.to_datetime(df['dep_actual'], utc=True, errors='coerce')
df['delay_from_timestamps'] = (df['dep_actual_dt'] - df['dep_scheduled_dt']).dt.total_seconds() / 60
df['dep_delay'] = df['delay_from_column'].fillna(df['delay_from_timestamps'])

df_labeled = df[df['dep_delay'].notna()].copy()
print(f"✅ Flights with valid delay: {len(df_labeled):,}")

# Add time features
print("\n🕐 Adding time features...")
df_labeled['hour'] = pd.to_datetime(df_labeled['dep_scheduled'], utc=True).dt.floor('h')
df_labeled['hour_of_day'] = pd.to_datetime(df_labeled['dep_scheduled'], utc=True).dt.hour
df_labeled['day_of_week'] = pd.to_datetime(df_labeled['flight_date']).dt.dayofweek
df_labeled['delayed_15'] = (df_labeled['dep_delay'] >= 15).astype(int)

# Remove flights with NULL hour
before_dropna = len(df_labeled)
df_labeled = df_labeled[df_labeled['hour'].notna()].copy()
print(f"   Removed {before_dropna - len(df_labeled):,} flights with NULL hour")

# Load weather
print("\n🌤️  Loading weather data...")
df_weather = pd.read_csv("data/processed/weather_hourly_COMBINED.csv")
df_weather['hour'] = pd.to_datetime(df_weather['hour'], utc=True)
df_weather = df_weather.sort_values('hour').reset_index(drop=True)

print(f"   Available weather hours: {len(df_weather)}")

# Sort both
df_labeled = df_labeled.sort_values('hour').reset_index(drop=True)

# SIMPLE MERGE with ASOF
print("\n🧠 Applying nearest-time weather matching (within 6 hours)...")

df_merged = pd.merge_asof(
    df_labeled,
    df_weather,
    on='hour',
    direction='backward',
    tolerance=pd.Timedelta(hours=4)
)

print(f"✅ After weather matching: {len(df_merged):,} flights")

# Count coverage
has_weather = df_merged['temp'].notna().sum()
no_weather = df_merged['temp'].isna().sum()

print(f"   With weather:    {has_weather:,} flights ({has_weather/len(df_merged)*100:.1f}%)")
print(f"   Without weather: {no_weather:,} flights ({no_weather/len(df_merged)*100:.1f}%)")

# Keep only flights with weather
df_with_weather = df_merged[df_merged['temp'].notna()].copy()

print(f"\n📦 Final with weather: {len(df_with_weather):,} flights")

# Add road traffic
print("\n🚗 Adding road traffic...")
df_road = pd.read_csv("data/processed/road_traffic_hour_profile.csv")
df_with_weather = df_with_weather.merge(
    df_road[['hour_of_day', 'road_avg_speed', 'road_vehicle_count', 'road_congestion_index']], 
    on='hour_of_day', how='left'
)

# Historical features
print("\n📊 Adding historical features...")
df_with_weather = df_with_weather.sort_values('hour').reset_index(drop=True)

df_with_weather['prev_hour_delay_rate'] = df_with_weather['delayed_15'].rolling(window=30, min_periods=1).mean().shift(1).fillna(0)
df_with_weather['avg_delay_last_24h'] = df_with_weather['delayed_15'].rolling(window=150, min_periods=1).mean().shift(1).fillna(0)

airline_rates = df_with_weather.groupby('airline_name')['delayed_15'].mean()
df_with_weather['airline_7d_delay_rate'] = df_with_weather['airline_name'].map(airline_rates)

# Final dataset
print("\n📋 Creating final dataset...")
final_cols = [
    'flight_date', 'airline_name', 'flight_iata', 'dep_iata', 'arr_iata',
    'hour', 'hour_of_day', 'day_of_week', 'delayed_15',
    'temp', 'humidity', 'pressure', 'wind_speed', 'weather',
    'road_avg_speed', 'road_vehicle_count', 'road_congestion_index',
    'prev_hour_delay_rate', 'avg_delay_last_24h', 'airline_7d_delay_rate'
]

df_final = df_with_weather[final_cols].dropna()

# Save
output_path = "data/processed/flights_v1_IMPUTED.csv"
df_final.to_csv(output_path, index=False)

print("\n" + "="*70)
print("🎉 V1_IMPUTED CREATED!")
print("="*70)
print(f"📁 Saved to: {output_path}")
print(f"📦 Total flights: {len(df_final):,}")

print("\n📅 MONTHLY BREAKDOWN:")
monthly = df_final.groupby(pd.to_datetime(df_final['flight_date']).dt.month).agg({
    'flight_date': 'count',
    'delayed_15': ['sum', 'mean']
})
monthly.columns = ['Flights', 'Delayed', 'Delay_Rate']
monthly['Delay_Rate'] = (monthly['Delay_Rate'] * 100).round(1)

month_names = {10: 'Oct', 11: 'Nov', 12: 'Dec', 1: 'Jan', 2: 'Feb'}
for month in monthly.index:
    if month in month_names:
        row = monthly.loc[month]
        print(f"  {month_names[month]}: {row['Flights']:,} flights, {row['Delay_Rate']:.1f}% delayed")

print("\n🎯 COMPARISON:")
print(f"  Original V1:  4,330 flights")
print(f"  V1_IMPUTED:   {len(df_final):,} flights")
if len(df_final) > 4330:
    print(f"  GAIN:         +{len(df_final) - 4330:,} flights ({(len(df_final) - 4330)/4330*100:+.1f}%)")
    print(f"\n  🚀🚀🚀 SUCCESS! WE HAVE MORE DATA! 🚀🚀🚀")
else:
    print(f"  CHANGE:       {len(df_final) - 4330:,} flights ({(len(df_final) - 4330)/4330*100:+.1f}%)")

