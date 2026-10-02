import os
from datetime import datetime
import pandas as pd
import requests

# 10 Target Dams matching your imd_data directory
TARGET_DAMS = [
    {
        "filename": "almatti_dam_ready.csv",
        "name": "Almatti",
        "lat": 16.3308,
        "lon": 75.8881,
    },
    {
        "filename": "bhadra_dam_ready.csv",
        "name": "Bhadra",
        "lat": 13.7022,
        "lon": 75.6425,
    },
    {
        "filename": "hemavathy_dam_ready.csv",
        "name": "Hemavathy",
        "lat": 12.7844,
        "lon": 76.0506,
    },
    {
        "filename": "kabini_dam_ready.csv",
        "name": "Kabini",
        "lat": 11.9744,
        "lon": 76.3533,
    },
    {
        "filename": "krsagara_dam_ready.csv",
        "name": "KRSagara",
        "lat": 12.4258,
        "lon": 76.5711,
    },
    {
        "filename": "linganamakki_dam_ready.csv",
        "name": "Linganamakki",
        "lat": 14.1783,
        "lon": 74.8475,
    },
    {
        "filename": "malaprabha_dam_ready.csv",
        "name": "Malaprabha",
        "lat": 15.8242,
        "lon": 75.1092,
    },
    {
        "filename": "supa_dam_ready.csv",
        "name": "Supa",
        "lat": 15.2750,
        "lon": 74.5260,
    },
    {
        "filename": "tungabhadra_dam_ready.csv",
        "name": "Tungabhadra",
        "lat": 15.2589,
        "lon": 76.3353,
    },
    {
        "filename": "vanivilasa_sagar_dam_ready.csv",
        "name": "Vanivilasa Sagar",
        "lat": 13.8906,
        "lon": 76.4769,
    },
]

OUTPUT_DIR = "data/rainfall"
START_DATE = "1970-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")


def download_all_dams_rainfall():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lats = [dam["lat"] for dam in TARGET_DAMS]
    lons = [dam["lon"] for dam in TARGET_DAMS]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lats,
        "longitude": lons,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": "precipitation_sum",
        "timezone": "Asia/Kolkata",
    }

    print(f"Fetching 1970–present rainfall for 10 dams in a single query...")
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    results = response.json()

    # Open-Meteo returns a list of results corresponding to each lat/lon pair
    for dam, data in zip(TARGET_DAMS, results):
        df = pd.DataFrame(
            {
                "date": data["daily"]["time"],
                "rainfall_mm": data["daily"]["precipitation_sum"],
            }
        )

        df["dam_name"] = dam["name"]
        df["latitude"] = dam["lat"]
        df["longitude"] = dam["lon"]

        file_path = os.path.join(OUTPUT_DIR, dam["filename"])
        df.to_csv(file_path, index=False)
        print(f"Saved: {file_path} ({len(df):,} rows)")


if __name__ == "__main__":
    download_all_dams_rainfall()