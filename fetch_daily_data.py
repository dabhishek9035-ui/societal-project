import os
import requests
import pandas as pd

BASE_URL = "https://nwdp.nwic.gov.in/api/3/action/datastore_search"
RESOURCE_ID = "26800982-045c-41de-821d-b1ca080a79a8"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMD_DIR = os.path.join(BASE_DIR, "data", "imd_data")

DAM_QUERIES = {
    "almatti_dam_ready.csv": "Almatti",
    "bhadra_dam_ready.csv": "Bhadra",
    "hemavathy_dam_ready.csv": "Hemavathy",
    "kabini_dam_ready.csv": "Kabini",
    "krsagara_dam_ready.csv": "Krishna Raja Sagara",
    "linganamakki_dam_ready.csv": "Linganamakki",
    "malaprabha_dam_ready.csv": "Malaprabha",
    "supa_dam_ready.csv": "Supa",
    "tungabhadra_dam_ready.csv": "Tungabhadra",
    "vanivilasa_sagar_dam_ready.csv": "Vanivilasa"
}

# NWDP search is substring based: "Bhadra" also matches "Tungabhadra",
# and "Krishna" returns every reservoir in the Krishna basin. Keep only rows
# whose Reservoir Name identifies the requested reservoir.
DAM_SOURCE_NAMES = {
    "almatti_dam_ready.csv": {"almatti dam"},
    "bhadra_dam_ready.csv": {"bhadra dam"},
    "hemavathy_dam_ready.csv": {"hemavathy dam", "hemavathi dam"},
    "kabini_dam_ready.csv": {"kabini dam"},
    "krsagara_dam_ready.csv": {
        "krishna raja sagara dam", "krishna raja sagar dam", "krishna raja sagara",
        "krishna raja sagar", "krs dam", "krs reservoir",
    },
    "linganamakki_dam_ready.csv": {"linganamakki dam"},
    "malaprabha_dam_ready.csv": {"malaprabha dam"},
    "supa_dam_ready.csv": {"supa dam"},
    "tungabhadra_dam_ready.csv": {"tungabhadra dam"},
    "vanivilasa_sagar_dam_ready.csv": {"vanivilasa sagar dam", "vani vilasa sagar dam", "vani vilas sagar dam"},
}


def reservoir_key(value):
    return " ".join(str(value).casefold().replace(".", " ").split())


def filter_reservoir_rows(df, csv_file, keep_unlabeled):
    """Prevent substring search results for neighboring dams entering a dataset."""
    if "Reservoir Name" not in df or df.empty:
        return df
    accepted = DAM_SOURCE_NAMES[csv_file]
    names = df["Reservoir Name"].map(reservoir_key)
    valid = names.isin(accepted)
    if keep_unlabeled:
        valid |= names.eq("") | names.eq("nan")
    rejected = int((~valid).sum())
    if rejected:
        print(f"  -> Removed {rejected} row(s) for a different reservoir from {csv_file}")
        if not keep_unlabeled:
            rejected_names = df.loc[~valid, "Reservoir Name"].dropna().value_counts().head(8).to_dict()
            if rejected_names:
                print(f"  -> Unmatched source names: {rejected_names}")
    return df.loc[valid].copy()

# Standard 22 columns expected across all dam CSVs
ALL_COLUMNS = [
    "Date", "_id", "Reservoir Name", "Basin", "Sub Basin", "River",
    "Monitoring Date", "Percentage Full", "Reservoir Level (ft)",
    "Design Gross Capacity (TMC)", "Gross Capacity (TMC)", "Live Capacity (TMC)",
    "Live Above Cill (TMC)", "Inflow (Cusecs)", "Outflow to River (Cusecs)",
    "Canal Withdrawal (Cusecs)", "Evaporation (Cusecs)", "Cumulative Inflow (TMC)",
    "Cumulative Outflow (TMC)", "Cumulative Withdrawal (TMC)",
    "Cumulative Evaporation (TMC)", "RAINFALL"
]

def sanitize_cols(df):
    """Clean column headers by stripping whitespace and newlines."""
    df.columns = [str(c).strip().replace("\n", "").replace("\r", "") for c in df.columns]
    return df

def clean_local_csv(file_path):
    """Loads existing local CSV and aligns columns properly."""
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=ALL_COLUMNS)
    
    try:
        df = pd.read_csv(file_path, low_memory=False)
        df = sanitize_cols(df)

        # Standardize 'Sub Basin' and 'Live Capacity' key names if they had newlines
        rename_dict = {}
        for c in df.columns:
            if "Sub" in c and "Basin" in c:
                rename_dict[c] = "Sub Basin"
            elif "Live Capacity" in c:
                rename_dict[c] = "Live Capacity (TMC)"
            elif "Cumulative Evaporation" in c:
                rename_dict[c] = "Cumulative Evaporation (TMC)"
        
        if rename_dict:
            df = df.rename(columns=rename_dict)

        # Ensure Date column is filled correctly
        if "Date" not in df.columns or df["Date"].isna().all():
            if "Monitoring Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Monitoring Date"], errors="coerce").dt.strftime("%Y-%m-%d")

        # Ensure all standard columns exist in local dataframe
        for col in ALL_COLUMNS:
            if col not in df.columns:
                df[col] = None

        return df[ALL_COLUMNS]

    except Exception as e:
        print(f"Warning reading {file_path}: {e}")
        return pd.DataFrame(columns=ALL_COLUMNS)

def fetch_and_update_dam(csv_file, search_term):
    file_path = os.path.join(IMD_DIR, csv_file)
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    
    params = {
        "resource_id": RESOURCE_ID,
        "q": search_term,
        "limit": 10000,
        "sort": "_id desc"
    }

    try:
        # 1. Read existing local data with all columns intact
        local_df = filter_reservoir_rows(clean_local_csv(file_path), csv_file, keep_unlabeled=True)

        # 2. Fetch new API records
        res = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
        res.raise_for_status()
        data = res.json()

        if data.get("success") and data["result"]["records"]:
            df_api = pd.DataFrame(data["result"]["records"])
            df_api = sanitize_cols(df_api)

            # Map API column names if needed
            rename_dict = {}
            for c in df_api.columns:
                if "Sub" in c and "Basin" in c:
                    rename_dict[c] = "Sub Basin"
                elif "Live Capacity" in c:
                    rename_dict[c] = "Live Capacity (TMC)"
                elif "Cumulative Evaporation" in c:
                    rename_dict[c] = "Cumulative Evaporation (TMC)"
            
            if rename_dict:
                df_api = df_api.rename(columns=rename_dict)

            # Format Date from Monitoring Date
            if "Monitoring Date" in df_api.columns:
                df_api["Date"] = pd.to_datetime(
                    df_api["Monitoring Date"], 
                    format="%d-%m-%Y %H:%M:%S", 
                    errors="coerce"
                )
                mask = df_api["Date"].isna()
                if mask.any():
                    df_api.loc[mask, "Date"] = pd.to_datetime(df_api.loc[mask, "Monitoring Date"], errors="coerce")
                
                df_api["Date"] = df_api["Date"].dt.strftime("%Y-%m-%d")

            # Fill missing standard columns
            for col in ALL_COLUMNS:
                if col not in df_api.columns:
                    df_api[col] = None

            api_clean = filter_reservoir_rows(df_api[ALL_COLUMNS], csv_file, keep_unlabeled=False)
        else:
            api_clean = pd.DataFrame(columns=ALL_COLUMNS)

        # 3. Concatenate local and API data
        combined_df = pd.concat([local_df, api_clean], ignore_index=True)

        # 4. Clean dates and deduplicate
        combined_df["Date_Parsed"] = pd.to_datetime(combined_df["Date"], errors="coerce")
        combined_df = combined_df.dropna(subset=["Date_Parsed"])
        
        combined_df["Date_Str"] = combined_df["Date_Parsed"].dt.strftime("%Y-%m-%d")
        combined_df = combined_df.drop_duplicates(subset=["Date_Str"], keep="last")

        # Sort chronologically
        combined_df = combined_df.sort_values("Date_Parsed").reset_index(drop=True)
        combined_df["Date"] = combined_df["Date_Str"]

        # Drop temporary parsing columns
        final_df = combined_df[ALL_COLUMNS]

        # 5. Overwrite the CSV cleanly
        final_df.to_csv(file_path, index=False)

        print(f"Updated {csv_file} | Total Columns: {len(final_df.columns)} | Latest Date: {final_df['Date'].iloc[-1]} | Total Rows: {len(final_df)}")

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

if __name__ == "__main__":
    for csv_file, query_term in DAM_QUERIES.items():
        fetch_and_update_dam(csv_file, query_term)
