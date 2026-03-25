import os
import pandas as pd
import pytz

# === File paths ===
FLIGHTS = os.path.expanduser('~/thesis/data/raw/flights_all.csv')
WEATHER = os.path.expanduser('~/thesis/data/raw/weather_all.csv')
TRAFFIC = os.path.expanduser('~/thesis/data/raw/ibb_hourly_traffic_all.csv')
OUT = os.path.expanduser('~/thesis/data/processed/training_data.csv')

print("📦 Loading data...")
flights = pd.read_csv(FLIGHTS, low_memory=False)
weather = pd.read_csv(WEATHER)
traffic = pd.read_csv(TRAFFIC)

# === Clean and format FLIGHTS data ===
flights['dep_scheduled'] = pd.to_datetime(flights['dep_scheduled'], errors='coerce', utc=True)
flights['dep_actual'] = pd.to_datetime(flights['dep_actual'], errors='coerce', utc=True)

# Calculate delay in minutes (actual - scheduled)
flights['delay_min'] = (flights['dep_actual'] - flights['dep_scheduled']).dt.total_seconds() / 60
flights['delay_min'] = flights['delay_min'].fillna(flights['dep_delay'])
flights['delay_min'] = flights['delay_min'].fillna(0)

# Convert scheduled time to Istanbul local hour
istanbul = pytz.timezone('Europe/Istanbul')
flights['hour'] = flights['dep_scheduled'].dt.tz_convert(istanbul).dt.floor('h')

# === Clean and format WEATHER data ===
weather['timestamp'] = pd.to_datetime(weather['api_local_time'], errors='coerce')
weather['hour'] = weather['timestamp'].dt.floor('h')

# Make timezone-aware (Europe/Istanbul)
weather['hour'] = weather['hour'].dt.tz_localize('Europe/Istanbul', ambiguous='NaT', nonexistent='shift_forward')

# Keep only necessary columns
weather = weather[['hour', 'temp', 'humidity', 'pressure', 'wind_speed', 'weather']]

# === Clean and format TRAFFIC data ===
traffic['DATE_TIME'] = pd.to_datetime(traffic['DATE_TIME'], errors='coerce')
traffic['hour'] = traffic['DATE_TIME'].dt.floor('h')

# Make timezone-aware (Europe/Istanbul)
traffic['hour'] = traffic['hour'].dt.tz_localize('Europe/Istanbul', ambiguous='NaT', nonexistent='shift_forward')

# Aggregate traffic data by hour (average speed + total vehicles)
agg_traffic = traffic.groupby('hour').agg({
    'AVERAGE_SPEED': 'mean',
    'NUMBER_OF_VEHICLES': 'sum'
}).reset_index().rename(columns={
    'AVERAGE_SPEED': 'avg_speed',
    'NUMBER_OF_VEHICLES': 'total_vehicles'
})

# === Merge all datasets ===
merged = flights.merge(weather, how='left', on='hour')
merged = merged.merge(agg_traffic, how='left', on='hour')

# === Keep relevant columns only ===
keep = [
    'hour',
    'dep_scheduled',
    'dep_actual',
    'delay_min',
    'airline_name',
    'flight_number',
    'temp',
    'humidity',
    'pressure',
    'wind_speed',
    'weather',
    'avg_speed',
    'total_vehicles'
]
merged = merged[keep]

# === Save the merged dataset ===
os.makedirs(os.path.dirname(OUT), exist_ok=True)
merged.to_csv(OUT, index=False)

print(f"✅ Merged dataset saved: {OUT}")
print("Rows:", len(merged))
print("Columns:", len(merged.columns))
print("🎉 Merge completed successfully!")
