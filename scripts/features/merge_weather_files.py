import pandas as pd

print("🔧 MERGING OLD AND NEW WEATHER FILES\n")
print("="*70)

# Load both
weather_old = pd.read_csv("data/processed/weather_hourly.csv")
weather_new = pd.read_csv("data/processed/weather_hourly_FIXED.csv")

print(f"Old weather: {len(weather_old)} hours")
print(f"New weather: {len(weather_new)} hours")

# Combine them (keep all unique hours from both)
weather_combined = pd.concat([weather_old, weather_new], ignore_index=True)

# Remove duplicates (keep first occurrence)
weather_combined = weather_combined.drop_duplicates(subset=['hour'], keep='first')

# Sort by time
weather_combined['hour_dt'] = pd.to_datetime(weather_combined['hour'], utc=True)
weather_combined = weather_combined.sort_values('hour_dt').drop(columns=['hour_dt'])

# Save
weather_combined.to_csv("data/processed/weather_hourly_COMBINED.csv", index=False)

print(f"\n✅ COMBINED weather: {len(weather_combined)} hours")
print(f"   Gained from old: {len(weather_old)}")
print(f"   Gained from new: {len(weather_new)}")
print(f"   Total unique: {len(weather_combined)}")

# Check monthly coverage
weather_combined['hour_dt'] = pd.to_datetime(weather_combined['hour'], utc=True)
weather_combined['month'] = weather_combined['hour_dt'].dt.month

print(f"\n📊 MONTHLY BREAKDOWN:")
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
for month in sorted(weather_combined['month'].unique()):
    if month in month_names:
        count = len(weather_combined[weather_combined['month'] == month])
        print(f"  {month_names[month]}: {count} hours")

print(f"\n💾 Saved to: data/processed/weather_hourly_COMBINED.csv")

