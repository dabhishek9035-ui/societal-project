import os
import pandas as pd
import requests

# 1. Load your master Karnataka NWDP CSV file
master_file = "data/nwdp_karnataka_master.csv"  # Update path if needed
df_master = pd.read_csv(master_file)

# 2. Define Approximate Coordinates (Lat, Lon) for major Karnataka reservoirs
# You can add more from your list as needed!
dam_coordinates = {
    "Almatti Dam": (16.34, 75.88),
    "Bhadra Dam": (13.71, 75.64),
    "Kabini Dam": (11.89, 76.33),
    "K.R.Sagara Dam": (12.42, 76.57),
    "Tungabhadra Dam": (15.25, 76.39),
    "Harangi Dam": (12.52, 75.92),
    "Hemavathy Dam": (12.76, 76.25),
    "Linganamakki Dam": (14.21, 74.84),
    "Malaprabha Dam": (15.82, 74.70),
    "Supa Dam": (15.28, 74.52),
    "Vanivilasa Sagar Dam": (13.89, 76.43),
}

os.makedirs("data/processed_dams", exist_ok=True)

# 3. Loop through each dam in your coordinates dictionary
for target_dam, (lat, lon) in dam_coordinates.items():
    print(f"\nProcessing {target_dam}...")

    # Filter data for this specific dam
    df_dam = df_master[df_master["Reservoir Name"] == target_dam].copy()
    if df_dam.empty:
        print(f"Skipping {target_dam}: No data found in master file.")
        continue

    # Clean dates and sort
    df_dam["Monitoring Date"] = pd.to_datetime(
        df_dam["Monitoring Date"], format="%d-%m-%Y %H:%M:%S"
    )
    df_dam = df_dam.sort_values("Monitoring Date").reset_index(drop=True)
    df_dam["Date"] = df_dam["Monitoring Date"].dt.date
    df_dam.set_index("Date", inplace=True)

    # Ensure continuous daily frequency and interpolate gaps
    df_dam = df_dam.asfreq("D")
    if "Reservoir Level (ft)" in df_dam.columns:
        df_dam["Reservoir Level (ft)"] = df_dam[
            "Reservoir Level (ft)"
        ].interpolate(method="linear")

    # Fetch matching rainfall from Open-Meteo API
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date=1970-01-01&end_date=2025-12-31&daily=precipitation_sum"
    )

    response = requests.get(url)
    res_json = response.json()

    if "daily" not in res_json:
        print(f"Could not fetch weather for {target_dam}. Skipping.")
        continue

    df_rain = pd.DataFrame(
        {
            "date": pd.to_datetime(res_json["daily"]["time"]),
            "RAINFALL": res_json["daily"]["precipitation_sum"],
        }
    )
    df_rain.set_index("date", inplace=True)

    # Merge reservoir levels and rainfall
    final_model_df = df_dam.join(df_rain["RAINFALL"], how="inner")
    if "Reservoir Level (ft)" in final_model_df.columns:
        final_model_df.dropna(
            subset=["Reservoir Level (ft)", "RAINFALL"], inplace=True
        )

    # Save individual processed file for this dam
    safe_name = target_dam.lower().replace(" ", "_").replace(".", "")
    output_path = f"data/processed_dams/{safe_name}_ready.csv"
    final_model_df.to_csv(output_path)
    print(f"Saved ready-to-train file to {output_path}")

print(
    "\nAll specified Karnataka reservoirs processed and saved inside"
    " data/processed_dams/!"
)