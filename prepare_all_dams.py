import glob
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMD_DIR = os.path.join(BASE_DIR, "data", "imd_data")
FEATURE_DIR = os.path.join(BASE_DIR, "data", "features")
SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits")

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

MIN_REQUIRED_SAMPLES = 15  # Minimum rows required for processing


def process_dam(file_path):
    filename = os.path.basename(file_path)
    dam_name = filename.replace("_dam_ready.csv", "").lower()
    print(f"\n--- Processing Dam: {dam_name.upper()} ---")

    df = pd.read_csv(file_path)

    # 1. Identify Date Column (case-insensitive)
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if not date_col:
        print(f"⚠️ Skipping {dam_name}: Missing 'Date' column.")
        return

    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    if date_col != "Date":
        df = df.drop(columns=[date_col])
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # 2. Clean comma strings and coerce non-date columns to numeric floats
    non_date_cols = [c for c in df.columns if c != "Date"]
    for col in non_date_cols:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", "").str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop columns that became entirely NaN (e.g., text/metadata columns)
    df = df.dropna(how="all", axis=1)

    # 3. Identify Target Column
    target_col = next(
        (c for c in df.columns if "level" in c.lower() or "reservoir" in c.lower()),
        None,
    )
    if not target_col:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            print(f"⚠️ Skipping {dam_name}: No valid numeric columns found.")
            return
        target_col = num_cols[0]

    # MSL to Gauge Height conversion for KRSagara
    if dam_name == "krsagara" and df[target_col].mean() > 2000:
        df[target_col] = df[target_col] - 2362.7

    # Physical Outlier Cleaning (Filters corrupted zero values & unrealistic single-day jumps > 15 ft)
    level_diff = df[target_col].diff().abs()
    df.loc[(level_diff > 15.0) | (df[target_col] <= 0), target_col] = np.nan

    # Smoothly interpolate missing target values across time
    df[target_col] = df[target_col].interpolate(method="linear").ffill().bfill()

    # Rename target column to standardized name
    if target_col != "Reservoir Level (ft)":
        df = df.rename(columns={target_col: "Reservoir Level (ft)"})
        target_col = "Reservoir Level (ft)"

    # 4. Feature Engineering (Lags & Rolling Metrics)
    df["level_lag_1"] = df[target_col].shift(1)
    df["level_lag_3"] = df[target_col].shift(3)
    df["level_lag_7"] = df[target_col].shift(7)

    df["level_roll_mean_7"] = (
        df[target_col].shift(1).rolling(window=7, min_periods=1).mean()
    )
    df["level_roll_std_7"] = (
        df[target_col].shift(1).rolling(window=7, min_periods=1).std().fillna(0)
    )

    # Calendar Cyclical Features
    df["day_of_year"] = df["Date"].dt.dayofyear
    df["month"] = df["Date"].dt.month
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # Forward & Backward fill initial lag NaNs instead of dropping all rows
    df = df.ffill().bfill().reset_index(drop=True)

    if len(df) < MIN_REQUIRED_SAMPLES:
        print(
            f"⚠️ Skipping {dam_name}: Insufficient samples after cleaning ({len(df)} rows, minimum {MIN_REQUIRED_SAMPLES} required)."
        )
        return

    # Save feature dataset
    feat_path = os.path.join(FEATURE_DIR, f"{dam_name}_features.csv")
    df.to_csv(feat_path, index=False)

    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Temporal Train/Val/Test Split (80/10/10)
    n = len(df)
    train_end = max(1, int(n * 0.8))
    val_end = max(train_end + 1, int(n * 0.9))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # Fit scaler strictly on the training set to avoid data leakage
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    val_scaled = (
        scaler.transform(val_df[feature_cols])
        if len(val_df) > 0
        else train_scaled[:1]
    )
    test_scaled = (
        scaler.transform(test_df[feature_cols])
        if len(test_df) > 0
        else train_scaled[:1]
    )

    dam_split_dir = os.path.join(SPLIT_DIR, dam_name)
    os.makedirs(dam_split_dir, exist_ok=True)

    # Save scaler and scaled train/val/test datasets
    joblib.dump(scaler, os.path.join(dam_split_dir, "scaler.joblib"))

    pd.DataFrame(train_scaled, columns=feature_cols).assign(
        Date=train_df["Date"].values
    ).to_csv(os.path.join(dam_split_dir, "train_scaled.csv"), index=False)

    pd.DataFrame(val_scaled, columns=feature_cols).assign(
        Date=val_df["Date"].values
        if len(val_df) > 0
        else train_df["Date"].values[:1]
    ).to_csv(os.path.join(dam_split_dir, "val_scaled.csv"), index=False)

    pd.DataFrame(test_scaled, columns=feature_cols).assign(
        Date=test_df["Date"].values
        if len(test_df) > 0
        else train_df["Date"].values[:1]
    ).to_csv(os.path.join(dam_split_dir, "test_scaled.csv"), index=False)

    print(f"✅ Created features and splits for {dam_name} ({n} samples).")


if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(IMD_DIR, "*_dam_ready.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_dam_ready.csv files found in {IMD_DIR}")

    for f in csv_files:
        process_dam(f)