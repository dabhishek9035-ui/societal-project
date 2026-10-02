import os
from datetime import datetime, timedelta
import pandas as pd
import requests

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


def update_daily_dams_rainfall():
    today_str = datetime.now().strftime("%Y-%m-%d")

    for dam in TARGET_DAMS:
        file_path = os.path.join(OUTPUT_DIR, dam["filename"])

        if not os.path.exists(file_path):
            print(f"Skipping {dam['filename']} (File does not exist).")
            continue

        existing_df = pd.read_csv(file_path)
        existing_df["date"] = pd.to_datetime(existing_df["date"])

        # Determine the start date for the fetch (day after the newest entry)
        max_date = existing_df["date"].max()
        start_date = (max_date + timedelta(days=1)).strftime("%Y-%m-%d")

        if start_date > today_str:
            print(
                f"{dam['name']:<18} | Up to date ({max_date.strftime('%Y-%m-%d')})"
            )
            continue

        # Open-Meteo Forecast endpoint handles recent observations up to today seamlessly
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": dam["lat"],
            "longitude": dam["lon"],
            "start_date": start_date,
            "end_date": today_str,
            "daily": "precipitation_sum",
            "timezone": "Asia/Kolkata",
        }

        try:
            res = requests.get(url, params=params, timeout=15)
            res.raise_for_status()
            data = res.json()

            if "daily" in data and data["daily"]["time"]:
                new_df = pd.DataFrame(
                    {
                        "date": data["daily"]["time"],
                        "rainfall_mm": data["daily"]["precipitation_sum"],
                        "dam_name": dam["name"],
                        "latitude": dam["lat"],
                        "longitude": dam["lon"],
                    }
                )
                new_df["date"] = pd.to_datetime(new_df["date"])

                # Merge, deduplicate by date, sort chronologically, and format date string
                combined_df = pd.concat([existing_df, new_df]).drop_duplicates(
                    subset=["date"], keep="last"
                )
                combined_df = combined_df.sort_values("date")
                combined_df["date"] = combined_df["date"].dt.strftime(
                    "%Y-%m-%d"
                )

                combined_df.to_csv(file_path, index=False)
                print(
                    f"{dam['name']:<18} | Appended {len(new_df)} new row(s) -> {file_path}"
                )
            else:
                print(f"{dam['name']:<18} | No new data available yet.")

        except requests.exceptions.RequestException as e:
            print(f"Error updating {dam['name']}: {e}")


if __name__ == "__main__":
    update_daily_dams_rainfall()