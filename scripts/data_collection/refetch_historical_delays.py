import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os

# ===============================
# CONFIG
# ===============================
API_KEY = "3125e78554a5d0516ab235ae790444aa"
BASE_URL = "http://api.aviationstack.com/v1/flights"

AIRPORT_IATA = "IST"

INPUT_FILE = "data/raw/flights_all.csv"
OUTPUT_FILE = "data/raw/flights_all_UPDATED.csv"

START_DATE = "2026-01-01"
END_DATE = "2026-02-28"

# ===============================
# LOAD EXISTING DATA
# ===============================
print("📂 Loading existing flight data...")
df_existing = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Existing flights: {len(df_existing)}")

df_existing["flight_date"] = pd.to_datetime(
    df_existing["flight_date"], errors="coerce"
)

# ===============================
# FETCH FUNCTION
# ===============================
def fetch_flights_for_date(date_str):
    params = {
        "access_key": API_KEY,
        "dep_iata": AIRPORT_IATA,
        "flight_date": date_str,
        "limit": 100
    }

    all_flights = []
    offset = 0

    while True:
        params["offset"] = offset

        response = requests.get(BASE_URL, params=params, timeout=30)
        data = response.json()

        if "data" not in data or len(data["data"]) == 0:
            break

        flights = data["data"]
        all_flights.extend(flights)

        print(f"  {date_str}: +{len(flights)} (total: {len(all_flights)})")

        pagination = data.get("pagination", {})
        total = pagination.get("total", 0)

        if len(all_flights) >= total:
            break

        offset += 100
        time.sleep(1)

    return all_flights

# ===============================
# REFETCH LOOP
# ===============================
print(f"\n🔄 Re-fetching from {START_DATE} to {END_DATE}...")

start = datetime.strptime(START_DATE, "%Y-%m-%d")
end = datetime.strptime(END_DATE, "%Y-%m-%d")

new_rows = []
current = start

while current <= end:
    date_str = current.strftime("%Y-%m-%d")
    flights = fetch_flights_for_date(date_str)

    for flight in flights:
        new_rows.append({
            "flight_date": flight.get("flight_date"),
            "flight_status": flight.get("flight_status"),
            "dep_iata": flight.get("departure", {}).get("iata"),
            "dep_icao": flight.get("departure", {}).get("icao"),
            "dep_airport": flight.get("departure", {}).get("airport"),
            "dep_scheduled": flight.get("departure", {}).get("scheduled"),
            "dep_estimated": flight.get("departure", {}).get("estimated"),
            "dep_actual": flight.get("departure", {}).get("actual"),
            "dep_delay": flight.get("departure", {}).get("delay"),
            "arr_iata": flight.get("arrival", {}).get("iata"),
            "arr_icao": flight.get("arrival", {}).get("icao"),
            "arr_airport": flight.get("arrival", {}).get("airport"),
            "arr_scheduled": flight.get("arrival", {}).get("scheduled"),
            "arr_estimated": flight.get("arrival", {}).get("estimated"),
            "arr_actual": flight.get("arrival", {}).get("actual"),
            "arr_delay": flight.get("arrival", {}).get("delay"),
            "airline_name": flight.get("airline", {}).get("name"),
            "flight_number": flight.get("flight", {}).get("number"),
            "flight_iata": flight.get("flight", {}).get("iata"),
            "flight_icao": flight.get("flight", {}).get("icao"),
            "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    current += timedelta(days=1)
    time.sleep(2)

print(f"\n📊 Total re-fetched rows: {len(new_rows)}")

df_new = pd.DataFrame(new_rows)

if not df_new.empty:
    df_new["flight_date"] = pd.to_datetime(df_new["flight_date"], errors="coerce")

# ===============================
# REMOVE OLD JAN-FEB DATA
# ===============================
start_dt = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE)

df_existing_cleaned = df_existing[
    ~(
        (df_existing["flight_date"] >= start_dt) &
        (df_existing["flight_date"] <= end_dt)
    )
]

print(f"After removing old Jan–Feb data: {len(df_existing_cleaned)}")

# ===============================
# COMBINE + DEDUPE
# ===============================
df_combined = pd.concat(
    [df_existing_cleaned, df_new],
    ignore_index=True
)

df_combined = df_combined.sort_values("collected_at").drop_duplicates(
    subset=["flight_date", "flight_iata"],
    keep="last"
)

print(f"\n✅ Final dataset size: {len(df_combined)}")

# ===============================
# SAVE
# ===============================
df_combined.to_csv(OUTPUT_FILE, index=False)
print(f"\n💾 Saved to: {OUTPUT_FILE}")

# ===============================
# DELAY SUMMARY
# ===============================
print("\n📈 Delay availability by month:")

df_combined["flight_date"] = pd.to_datetime(
    df_combined["flight_date"], errors="coerce"
)

df_combined["month"] = df_combined["flight_date"].dt.to_period("M")

summary = df_combined.groupby("month").agg(
    total_flights=("flight_date", "count"),
    with_delay=("dep_delay", lambda x: x.notna().sum())
)

summary["% with delay"] = (
    summary["with_delay"] / summary["total_flights"] * 100
).round(2)

print(summary)