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

def clean_target_series(series: pd.Series, max_cap: float) -> pd.Series:
    s = series.copy()
    s[s > max_cap] = np.nan
    is_zero = s <= 0.05
    surrounding_avg = (s.shift(1) + s.shift(-1)) / 2.0
    s[is_zero & (surrounding_avg > 1.0)] = np.nan
    s = s.interpolate(method="linear").ffill().bfill()
    return s

def patch_feature_file(file_path: str):
    dam_name = os.path.basename(file_path).replace('_features.csv', '').replace('_dam_ready', '').replace('_ready', '')
    print(f"Patching dataset: {file_path}")
    
    df = pd.read_csv(file_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
    max_cap = DAM_CAPACITIES.get(dam_name, 160.0)
    
    storage_col = None
    for col in ['Current_Storage_TMC', 'Live Capacity (TMC)', 'Percentage Full', 'Gross Capacity (TMC)']:
        if col in df.columns and df[col].notna().sum() > 0:
            storage_col = col
            break
            
    if storage_col is None:
        print(f"  -> Warning: No valid storage column found in {file_path}. Skipping.")
        return

    if 'Percentage Full' in df.columns and (df[storage_col].std() < 0.1 and df['Percentage Full'].std() > 1.0):
        print(f"  -> Re-deriving storage from 'Percentage Full' * {max_cap} TMC...")
        df['Current_Storage_TMC'] = (df['Percentage Full'].clip(0, 100) / 100.0) * max_cap
        target_col = 'Current_Storage_TMC'
    else:
        df['Current_Storage_TMC'] = df[storage_col]
        target_col = 'Current_Storage_TMC'

    df[target_col] = clean_target_series(df[target_col], max_cap)

    for lag in [1, 2, 7, 14, 30]:
        df[f'storage_lag_{lag}'] = df[target_col].shift(lag)

    target_cols = []
    for h in [1, 7, 14, 30]:
        t_abs = f'target_storage_t_plus_{h}'
        t_delta = f'target_delta_t_plus_{h}'
        df[t_abs] = df[target_col].shift(-h)
        df[t_delta] = df[t_abs] - df[target_col]
        target_cols.append(t_delta)

    feature_cols = [c for c in df.columns if c not in ['Date', 'Reservoir Name', 'Basin'] and not c.startswith('target_')]
    df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0)
    df_clean = df.dropna(subset=target_cols).reset_index(drop=True)

    df_clean.to_csv(file_path, index=False)
    print(f"  -> Successfully updated {file_path} (Shape: {df_clean.shape})")

if __name__ == "__main__":
    feature_files = glob.glob("data/features/*_features.csv")
    if not feature_files:
        print("No feature files found in data/features/")
    else:
        for filepath in sorted(feature_files):
            patch_feature_file(filepath)