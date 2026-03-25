import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

API_KEY = os.environ.get("OPENWEATHER_KEY")
if not API_KEY:
    raise SystemExit("❌ OPENWEATHER_KEY not set")

LAT, LON = 41.275278, 28.751944
URL = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"

resp = requests.get(URL, timeout=20)
print("HTTP Status:", resp.status_code)

if resp.status_code != 200:
    raise SystemExit(resp.text)

data = resp.json()

dt_utc = datetime.fromtimestamp(data["dt"], tz=timezone.utc)
tz_offset = timedelta(seconds=data["timezone"])
dt_local = dt_utc + tz_offset

weather = {
    "lat": data["coord"]["lat"],
    "lon": data["coord"]["lon"],
    "temp": data["main"]["temp"],
    "humidity": data["main"]["humidity"],
    "pressure": data["main"]["pressure"],
    "wind_speed": data["wind"]["speed"],
    "weather": data["weather"][0]["main"],
    "api_utc_time": dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
    "api_local_time": dt_local.strftime("%Y-%m-%d %H:%M:%S"),
    "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

out_dir = os.path.expanduser("~/Desktop/thesis/data/raw")
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "weather_all.csv")
df = pd.DataFrame([weather])
df.to_csv(out_path, index=False, mode="a", header=not os.path.exists(out_path))

print("✅ Weather appended:", out_path)