import os
import pandas as pd

# Load your master NWDP CSV file
df_master = pd.read_csv("data/nwdp_karnataka_master.csv")

# 1. Check what reservoirs are inside your file
print("Available Reservoirs:")
print(df_master["Reservoir Name"].unique())

# 2. Filter for just ONE specific reservoir (e.g., 'Almatti Dam')
target_dam = "Almatti Dam"
df_dam = df_master[df_master["Reservoir Name"] == target_dam].copy()

# 3. Sort chronologically (very important for time-series!)
df_dam["Monitoring Date"] = pd.to_datetime(
    df_dam["Monitoring Date"], format="%d-%m-%Y %H:%M:%S"
)
df_dam = df_dam.sort_values("Monitoring Date").reset_index(drop=True)

print(f"Filtered down to {len(df_dam)} rows for {target_dam}.")

# Set date as index and ensure daily frequency
df_dam["Date"] = df_dam["Monitoring Date"].dt.date
df_dam.set_index("Date", inplace=True)

# If there are gaps in dates, reindex to a complete daily timeline and interpolate small gaps
df_dam = df_dam.asfreq("D")
df_dam["Reservoir Level (ft)"] = df_dam["Reservoir Level (ft)"].interpolate(
    method="linear"
)

# --- SAVE TO /data/ DIRECTORY ---
os.makedirs("data", exist_ok=True)
output_path = "data/almatti_dam_processed.csv"
df_dam.to_csv(output_path)

print(f"DataFrame successfully saved to {output_path}!")