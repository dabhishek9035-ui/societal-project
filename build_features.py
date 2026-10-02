import os
import glob
import numpy as np
import pandas as pd

DAM_CAPACITIES = {
    "almatti": 123.08,
    "bhadra": 71.50,
    "hemavathy": 37.10,
    "kabini": 19.50,
    "krsagara": 105.79,
    "linganamakki": 156.61,
    "malaprabha": 37.73,
    "supa": 147.53,
    "tungabhadra": 100.80,
    "vanivilasa_sagar": 30.40
}

DAM_SOURCE_NAMES = {
    "almatti": {"almatti dam"},
    "bhadra": {"bhadra dam"},
    "hemavathy": {"hemavathy dam", "hemavathi dam"},
    "kabini": {"kabini dam"},
    "krsagara": {"krishna raja sagara dam", "krishna raja sagar dam", "krs dam"},
    "linganamakki": {"linganamakki dam"},
    "malaprabha": {"malaprabha dam"},
    "supa": {"supa dam"},
    "tungabhadra": {"tungabhadra dam"},
    "vanivilasa_sagar": {"vanivilasa sagar dam", "vani vilasa sagar dam", "vani vilas sagar dam"},
}

# Daily series written by fetch_rainfall.py in data/rainfall. These are point
# observations at the dam coordinates, not catchment averages.
RAINFALL_FILES = {
    "almatti": "almatti_dam_ready.csv",
    "bhadra": "bhadra_dam_ready.csv",
    "hemavathy": "hemavathy_dam_ready.csv",
    "kabini": "kabini_dam_ready.csv",
    "krsagara": "krsagara_dam_ready.csv",
    "linganamakki": "linganamakki_dam_ready.csv",
    "malaprabha": "malaprabha_dam_ready.csv",
    "supa": "supa_dam_ready.csv",
    "tungabhadra": "tungabhadra_dam_ready.csv",
    "vanivilasa_sagar": "vanivilasa_sagar_dam_ready.csv",
}


def reservoir_key(value):
    return " ".join(str(value).casefold().replace(".", " ").split())


def merge_daily_rainfall(df, date_col, dam_name):
    """Date-join a dam's rainfall series without treating missing data as rain-free."""
    rainfall_path = os.path.join("data", "rainfall", RAINFALL_FILES.get(dam_name, ""))
    if not RAINFALL_FILES.get(dam_name) or not os.path.exists(rainfall_path):
        # Keep the source's rainfall only when it actually has observations.
        source_rain = pd.to_numeric(df.get("RAINFALL", pd.Series(np.nan, index=df.index)), errors="coerce")
        df["rainfall_available"] = source_rain.notna().astype(int)
        df["RAINFALL"] = source_rain
        print(f"  -> No dedicated rainfall file; source rainfall available on {df['rainfall_available'].mean():.1%} of rows")
        return df

    rain = pd.read_csv(rainfall_path, usecols=["date", "rainfall_mm"])
    rain["date"] = pd.to_datetime(rain["date"], errors="coerce").dt.normalize()
    rain["rainfall_mm"] = pd.to_numeric(rain["rainfall_mm"], errors="coerce")
    rain = rain.dropna(subset=["date"]).groupby("date", as_index=False)["rainfall_mm"].mean()
    rain = rain.rename(columns={"date": "_rain_date", "rainfall_mm": "_rainfall_mm"})
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    result = df.copy()
    result["_rain_date"] = dates
    result = result.merge(rain, how="left", on="_rain_date", sort=False, validate="many_to_one")
    result["rainfall_available"] = result["_rainfall_mm"].notna().astype(int)
    # Dedicated precipitation is in mm/day and takes precedence over the
    # source's often sparse RAINFALL field wherever the rain series covers.
    source_rain = pd.to_numeric(result.get("RAINFALL", pd.Series(np.nan, index=result.index)), errors="coerce")
    result["RAINFALL"] = result["_rainfall_mm"].combine_first(source_rain)
    coverage = result["rainfall_available"].mean()
    print(f"  -> Joined daily rainfall: {coverage:.1%} exact-date coverage from {os.path.basename(rainfall_path)}")
    return result.drop(columns=["_rain_date", "_rainfall_mm"])


def build_level_only_features(df, date_col, dam_name, output_dir):
    """Build usable level history when a source has no storage-volume labels."""
    level_col = "Reservoir Level (ft)"
    df[level_col] = pd.to_numeric(df[level_col], errors="coerce")
    df = df.loc[df[date_col].notna() & df[level_col].notna()].copy()
    if df.empty:
        print(f"Error: neither storage nor usable reservoir-level observations exist for {dam_name}")
        return None

    day_of_year = df[date_col].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)
    df["is_monsoon"] = df[date_col].dt.month.isin([6, 7, 8, 9]).astype(int)
    for lag in [1, 2, 7, 14, 30]:
        df[f"level_lag_{lag}"] = df[level_col].shift(lag)
    if "RAINFALL" in df.columns:
        for window in [3, 7, 14, 30]:
            df[f"rain_roll_sum_{window}d"] = df["RAINFALL"].rolling(window).sum()
            df[f"rain_roll_mean_{window}d"] = df["RAINFALL"].rolling(window).mean()
            df[f"rain_observed_fraction_{window}d"] = df["rainfall_available"].rolling(window, min_periods=1).mean()
    for horizon in [1, 7, 14, 30]:
        df[f"target_level_ft_t_plus_{horizon}"] = df[level_col].shift(-horizon)

    rainfall_cols = [c for c in df if c == "RAINFALL" or c == "rainfall_available" or c.startswith("rain_")]
    non_rain_features = [
        c for c in df
        if c not in rainfall_cols and c not in [date_col, level_col]
        and not c.startswith("target_") and not c.startswith("_target_")
    ]
    df[non_rain_features] = df[non_rain_features].ffill().fillna(0)
    df[rainfall_cols] = df[rainfall_cols].fillna(0)
    out_file = os.path.join(output_dir, f"{dam_name}_features.csv")
    df.to_csv(out_file, index=False)
    print(f" -> Saved {len(df)} level-only feature rows to {out_file}; no TMC storage target is available")
    return df


def clean_source_rows(df, date_col, dam_name, max_cap):
    """Remove physically impossible source values before deriving any features.

    Invalid storage observations are interpolated only across short gaps. The
    original source file is left untouched; a compact audit is printed during building.
    """
    numeric = [
        'Percentage Full', 'Reservoir Level (ft)', 'Design Gross Capacity (TMC)',
        'Gross Capacity (TMC)', 'Live Capacity (TMC)', 'Live Above Cill (TMC)',
        'Current_Storage_TMC',
    ]
    for col in numeric:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    invalid = pd.Series(False, index=df.index)
    storage_sources = [c for c in ('Current_Storage_TMC', 'Live Capacity (TMC)') if c in df]
    # Design capacity is the most reliable physical ceiling when present.
    design = df['Design Gross Capacity (TMC)'] if 'Design Gross Capacity (TMC)' in df else pd.Series(np.nan, index=df.index)
    cap = pd.to_numeric(design, errors='coerce')
    cap = cap.where(cap > 0, max_cap).fillna(max_cap)
    cap = cap.clip(upper=max_cap * 1.25)
    for col in ('Design Gross Capacity (TMC)', 'Gross Capacity (TMC)'):
        if col in df:
            bad_capacity = df[col].notna() & ((df[col] <= 0) | (df[col] > max_cap * 1.25))
            df.loc[bad_capacity, col] = np.nan
    if storage_sources:
        source = storage_sources[0]
        vals = df[source]
        invalid |= vals.notna() & ((vals < 0) | (vals > cap * 1.02))

    if 'Percentage Full' in df:
        pct = df['Percentage Full']
        bad_pct = pct.notna() & ((pct < 0) | (pct > 100))
        # An invalid percentage should not discard a physically valid measured
        # volume; it is only fatal when percentage is the sole storage source.
        if not storage_sources:
            invalid |= bad_pct
        df.loc[bad_pct, 'Percentage Full'] = np.nan

    # A zero live-storage reading is treated as missing only where other
    # measurements show the reservoir is materially full.
    if 'Live Capacity (TMC)' in df:
        live = df['Live Capacity (TMC)']
        pct_full = df.get('Percentage Full', pd.Series(np.nan, index=df.index))
        invalid |= live.notna() & (live <= 0.05) & (pct_full > 5)

    if storage_sources:
        for col in storage_sources:
            df.loc[invalid, col] = np.nan

    # Reservoir level is normally smooth day to day. Detect extreme isolated
    # spikes against a centered 15-day median, which catches transcription
    # errors such as 29,161 ft without clipping real seasonal changes.
    level_col = 'Reservoir Level (ft)'
    if level_col in df:
        level = df[level_col]
        local = level.rolling(15, center=True, min_periods=5).median()
        deviation = (level - local).abs()
        local_mad = deviation.rolling(15, center=True, min_periods=5).median()
        level_bad = level.notna() & local.notna() & (deviation > np.maximum(100.0, 10 * local_mad.fillna(0)))
        df.loc[level_bad, level_col] = np.nan

    n_invalid = int(invalid.sum())
    if n_invalid:
        print(f"  -> Flagged {n_invalid} impossible storage/percentage row(s)")

    # Fill storage gaps using adjacent observations only. No backward fill is
    # used, so leading missing values remain missing and are dropped below.
    for col in storage_sources:
        df[col] = df[col].interpolate(method='linear', limit=3,
                                      limit_direction='both', limit_area='inside')
    if level_col in df:
        df[level_col] = df[level_col].interpolate(method='linear', limit=3,
                                                  limit_direction='both', limit_area='inside')
    return df

def process_dam_features(dam_file_path, output_dir="data/features"):
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(dam_file_path)
    
    date_col = 'Date' if 'Date' in df.columns else 'Monitoring Date'
    if date_col not in df.columns:
        print(f"Skipping {dam_file_path}: Missing Date column.")
        return None
        
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    dam_name = os.path.basename(dam_file_path).replace('_dam_ready.csv', '').replace('_ready.csv', '').replace('.csv', '')
    print(f"\nProcessing {dam_name} (Initial rows: {len(df)})...")

    # Reject records pulled in by substring searches (e.g. Bhadra matching
    # Tungabhadra). Unlabeled historical rows are retained for continuity.
    if 'Reservoir Name' in df.columns and dam_name in DAM_SOURCE_NAMES:
        names = df['Reservoir Name'].map(reservoir_key)
        valid = names.isin(DAM_SOURCE_NAMES[dam_name]) | names.isin(['', 'nan'])
        rejected = int((~valid).sum())
        if rejected:
            print(f"  -> Dropped {rejected} row(s) belonging to another reservoir")
        df = df.loc[valid].copy().reset_index(drop=True)
        labeled = names.loc[valid].isin(DAM_SOURCE_NAMES[dam_name]).sum()
        if labeled == 0 and (~names.isin(['', 'nan'])).any():
            print(f"  -> Skipping {dam_name}: named rows identify a different reservoir")
            return None
        if labeled == 0:
            print(f"  -> Reservoir Name is empty; trusting the dam-specific source filename")

    df = merge_daily_rainfall(df, date_col, dam_name)

    max_cap = DAM_CAPACITIES.get(dam_name, 100.0)
    df = clean_source_rows(df, date_col, dam_name, max_cap)
    
    storage_col = None
    for col in ['Current_Storage_TMC', 'Live Capacity (TMC)', 'Percentage Full', 'Gross Capacity (TMC)']:
        if col in df.columns and df[col].notna().sum() > 0:
            storage_col = col
            break
            
    if storage_col is None:
        if "Reservoir Level (ft)" in df.columns and df["Reservoir Level (ft)"].notna().any():
            return build_level_only_features(df, date_col, dam_name, output_dir)
        print(f"Error: storage volume and reservoir level are missing in {dam_file_path}")
        return None

    if storage_col == 'Percentage Full' or (df[storage_col].std() < 0.1 and 'Percentage Full' in df.columns):
        df['Current_Storage_TMC'] = (df['Percentage Full'].clip(0, 100) / 100.0) * max_cap
        storage_col = 'Current_Storage_TMC'
    else:
        df['Current_Storage_TMC'] = df[storage_col]
        storage_col = 'Current_Storage_TMC'

    # Cyclical Calendar Features
    day_of_year = df[date_col].dt.dayofyear
    df['sin_doy'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * day_of_year / 365.25)
    df['is_monsoon'] = df[date_col].dt.month.isin([6, 7, 8, 9]).astype(int)
    
    # Lags
    for lag in [1, 2, 7, 14, 30]:
        df[f'storage_lag_{lag}'] = df[storage_col].shift(lag)
        if 'Inflow (Cusecs)' in df.columns:
            df[f'inflow_lag_{lag}'] = df['Inflow (Cusecs)'].shift(lag)
        if 'Outflow to River (Cusecs)' in df.columns:
            df[f'outflow_lag_{lag}'] = df['Outflow to River (Cusecs)'].shift(lag)
            
    # Rolling Features
    if 'RAINFALL' in df.columns and df['RAINFALL'].notna().sum() > 0:
        for w in [3, 7, 14, 30]:
            df[f'rain_roll_sum_{w}d'] = df['RAINFALL'].rolling(window=w).sum()
            df[f'rain_roll_mean_{w}d'] = df['RAINFALL'].rolling(window=w).mean()
            df[f'rain_observed_fraction_{w}d'] = df['rainfall_available'].rolling(window=w, min_periods=1).mean()

    if 'Inflow (Cusecs)' in df.columns:
        for w in [3, 7, 14]:
            df[f'inflow_roll_mean_{w}d'] = df['Inflow (Cusecs)'].rolling(window=w).mean()

    # Differential Target Formulation: Delta S = S_{t+h} - S_t
    target_cols = []
    for horizon in [1, 7, 14, 30]:
        df[f'target_storage_t_plus_{horizon}'] = df[storage_col].shift(-horizon)
        t_delta_col = f'target_delta_t_plus_{horizon}'
        df[t_delta_col] = df[storage_col].shift(-horizon) - df[storage_col]
        target_cols.append(t_delta_col)

    feature_cols = [c for c in df.columns if c not in [date_col, 'Reservoir Name', 'Basin', 'River', 'Monitoring Date'] and not c.startswith('target_')]
    # Forward fill only; backfilling would leak future source values into past
    # feature rows. Remaining leading missing values are neutral-filled.
    rainfall_cols = [c for c in feature_cols if c == 'RAINFALL' or c == 'rainfall_available' or c.startswith('rain_')]
    fill_cols = [c for c in feature_cols if c != storage_col and c not in rainfall_cols]
    df[fill_cols] = df[fill_cols].ffill().fillna(0)
    # A missing daily gauge reading is not zero rainfall and must never be
    # forward-filled from a previous day. Zero is the model's numeric fill;
    # rainfall_available and rolling observed fractions retain missingness.
    df[rainfall_cols] = df[rainfall_cols].fillna(0)

    # Keep the newest observations even though they do not yet have future
    # labels. Training rebuilds dated targets and drops unlabeled origins; the
    # API needs these final rows to anchor a forecast on the actual latest day.
    df_clean = df.dropna(subset=[storage_col]).reset_index(drop=True)
    
    out_file = os.path.join(output_dir, f"{dam_name}_features.csv")
    df_clean.to_csv(out_file, index=False)
    print(f" -> Saved {len(df_clean)} rows to: {out_file} (Shape: {df_clean.shape})")
    return df_clean

if __name__ == "__main__":
    dam_files = glob.glob("data/imd_data/*_ready.csv") + glob.glob("data/imd_data/*.csv")
    dam_files = list(set([f for f in dam_files if "master" not in f.lower()]))
    
    if not dam_files:
        print("No dam CSV files found in 'data/imd_data/'.")
    else:
        for file in sorted(dam_files):
            process_dam_features(file)
