import os
import pandas as pd
import requests

# === Folder setup ===
raw_dir = os.path.expanduser("~/thesis/data/raw/ibb_traffic_monthly")
os.makedirs(raw_dir, exist_ok=True)

# Output merged file
out_path = os.path.expanduser("~/thesis/data/raw/ibb_hourly_traffic_all.csv")

# === Monthly datasets (Jan 2024 → Jan 2025) ===
datasets = {
    "Ocak_2024": "https://data.ibb.gov.tr/datastore/dump/7d9cbf11-f4b8-464d-bb3a-642b79e8b32b?bom=True",
    "Subat_2024": "https://data.ibb.gov.tr/datastore/dump/601cd734-9a62-44e0-89e5-bbfc2161d389?bom=True",
    "Mart_2024": "https://data.ibb.gov.tr/datastore/dump/b67e9415-0ba8-4319-8d36-359240a93808?bom=True",
    "Nisan_2024": "https://data.ibb.gov.tr/datastore/dump/0c7d60f3-8349-4836-a1c2-56ec93cbbd50?bom=True",
    "Mayis_2024": "https://data.ibb.gov.tr/datastore/dump/674604c8-8d08-42ff-a0b3-e2bde9f39455?bom=True",
    "Haziran_2024": "https://data.ibb.gov.tr/datastore/dump/674ba2c5-76b0-4f24-8e17-9aa2071d2572?bom=True",
    "Temmuz_2024": "https://data.ibb.gov.tr/datastore/dump/0019216e-48e0-4cec-9ab8-93d67f66dac3?bom=True",
    "Agustos_2024": "https://data.ibb.gov.tr/datastore/dump/168467fe-0495-4cdf-a93a-7c1e91179457?bom=True",
    "Eylul_2024": "https://data.ibb.gov.tr/datastore/dump/914cb0b9-d941-4408-98eb-f378519c26f4?bom=True",
    "Ekim_2024": "https://data.ibb.gov.tr/datastore/dump/d291989c-429d-4e61-9c70-1f76294b96b8?bom=True",
    "Kasim_2024": "https://data.ibb.gov.tr/datastore/dump/bedd5ab2-9a00-4966-9921-9672d4478a51?bom=True",
    "Aralik_2024": "https://data.ibb.gov.tr/datastore/dump/76671ebe-2fd2-426f-b85a-e3772263f483?bom=True",
    "Ocak_2025": "https://data.ibb.gov.tr/datastore/dump/57cb067b-1a0b-460b-8342-7884bd4537e8?bom=True",
}

# === Download and merge all ===
all_dfs = []

for month, url in datasets.items():
    print(f"⬇️  Downloading {month} ...")
    local_file = os.path.join(raw_dir, f"{month}.csv")

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(local_file, "wb") as f:
            f.write(r.content)
        print(f"✅ Saved: {local_file}")
    except Exception as e:
        print(f"⚠️ Failed to download {month}: {e}")
        continue

    try:
        df = pd.read_csv(local_file)
        df["source_month"] = month
        all_dfs.append(df)
    except Exception as e:
        print(f"⚠️ Error reading {local_file}: {e}")

# === Combine all ===
if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"\n✅ Combined dataset saved → {out_path}")
    print(f"📊 Total rows: {len(combined):,}")
else:
    print("⚠️ No data was merged.")
