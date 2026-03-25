#!/usr/bin/env python3
# collect_flights.py — robust aggregate-only version (IST)

import os
import json
from datetime import datetime, timezone
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ======================================================
# Paths
# ======================================================
BASE_DIR = os.path.expanduser("~/Desktop/thesis")
OUT_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_FILE = os.path.join(OUT_DIR, "flights_all.csv")

os.makedirs(OUT_DIR, exist_ok=True)

LOG_PREFIX = "[collect_flights]"

# ======================================================
# API key
# ======================================================
KEY = os.getenv("AVIATIONSTACK_KEY")
if not KEY:
    raise SystemExit(f"{LOG_PREFIX} AVIATIONSTACK_KEY not set. Export it in ~/.thesis_env")

# ======================================================
# API config
# ======================================================
BASE_URL = "https://api.aviationstack.com/v1/flights"
params = {
    "access_key": KEY,
    "dep_iata": "IST",
    "limit": 100
}

# ======================================================
# Requests session with retries
# ======================================================
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods={"GET"}
)
session.mount("https://", HTTPAdapter(max_retries=retries))

# ======================================================
# Helpers
# ======================================================
def nowstr():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# ======================================================
# Fetch flights
# ======================================================
def fetch():
    try:
        r = session.get(BASE_URL, params=params, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"{LOG_PREFIX} {nowstr()} Network error: {e}")
        return None, "network-error"

    print(f"{LOG_PREFIX} {nowstr()} HTTP Status: {r.status_code}")

    if r.status_code == 200:
        try:
            payload = r.json()
            return payload.get("data", []), None
        except json.JSONDecodeError:
            return None, "json-error"

    if r.status_code == 429:
        print(f"{LOG_PREFIX} ❌ API quota exceeded")
        return None, "quota-exceeded"

    print(f"{LOG_PREFIX} ❌ API error {r.status_code}: {r.text}")
    return None, f"api-error-{r.status_code}"

# ======================================================
# Save rows
# ======================================================
def save_rows(rows):
    if not rows:
        print(f"{LOG_PREFIX} No rows to save.")
        return

    df = pd.json_normalize(rows)

    flat = df.rename(columns={
        "departure.iata": "dep_iata",
        "departure.icao": "dep_icao",
        "departure.airport": "dep_airport",
        "departure.scheduled": "dep_scheduled",
        "departure.estimated": "dep_estimated",
        "departure.actual": "dep_actual",
        "departure.delay": "dep_delay",
        "arrival.iata": "arr_iata",
        "arrival.icao": "arr_icao",
        "arrival.airport": "arr_airport",
        "arrival.scheduled": "arr_scheduled",
        "arrival.estimated": "arr_estimated",
        "arrival.actual": "arr_actual",
        "arrival.delay": "arr_delay",
        "airline.name": "airline_name",
        "flight.number": "flight_number"
    })

    cols = [
        "flight_date", "flight_status",
        "dep_iata", "dep_icao", "dep_airport",
        "dep_scheduled", "dep_estimated", "dep_actual", "dep_delay",
        "arr_iata", "arr_icao", "arr_airport",
        "arr_scheduled", "arr_estimated", "arr_actual", "arr_delay",
        "airline_name", "flight_number"
    ]

    cols = [c for c in cols if c in flat.columns]
    flat_small = flat.loc[:, cols].copy()

    flat_small["collected_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    write_header = not os.path.exists(OUT_FILE)
    flat_small.to_csv(
        OUT_FILE,
        index=False,
        mode="a",
        header=write_header
    )

    print(f"{LOG_PREFIX} ✅ Flight data appended to: {OUT_FILE}")
    print(f"{LOG_PREFIX} 🛫 {len(flat_small)} flights saved.")

# ======================================================
# Main
# ======================================================
def main():
    rows, err = fetch()

    if err == "quota-exceeded":
        return 2
    if rows is None:
        return 1
    if not rows:
        print(f"{LOG_PREFIX} API returned no flights.")
        return 0

    try:
        save_rows(rows)
    except Exception as e:
        print(f"{LOG_PREFIX} Error saving rows: {e}")
        return 3

    return 0

if __name__ == "__main__":
    raise SystemExit(main())