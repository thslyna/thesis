import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# -------------------------
# Paths
# -------------------------
BASE = Path.home() / "Desktop" / "thesis"
OUT_DIR = BASE / "data" / "raw" / "opensky"
OUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp_now = datetime.utcnow()
out_file = OUT_DIR / f"opensky_ist_{timestamp_now.strftime('%Y%m%d_%H%M%S')}.csv"

# -------------------------
# OpenSky API
# -------------------------
URL = "https://opensky-network.org/api/states/all"

params = {
    "lamin": 40.7,
    "lamax": 41.5,
    "lomin": 28.5,
    "lomax": 29.5
}

print("📡 Requesting OpenSky data for IST...")
response = requests.get(URL, params=params, timeout=10)

if response.status_code != 200:
    print("❌ API error:", response.status_code)
    exit()

data = response.json()
states = data.get("states")

if states is None:
    print("⚠️ OpenSky returned no data (states=None). Skipping this run.")
    exit(0)

print(f"✈️ Planes detected near IST: {len(states)}")

# -------------------------
# Create DataFrame SAFELY
# -------------------------
df = pd.DataFrame(states)

df.columns = [
    "icao24", "callsign", "origin_country",
    "time_position", "last_contact",
    "longitude", "latitude",
    "baro_altitude", "on_ground",
    "velocity", "true_track", "vertical_rate",
    "sensors", "geo_altitude",
    "squawk", "spi",
    "position_source"
]

# Add snapshot time
df["snapshot_time_utc"] = timestamp_now

# -------------------------
# Save
# -------------------------
df.to_csv(out_file, index=False)

print("✅ Data saved to:")
print(out_file)