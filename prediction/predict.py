import argparse
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DAM_THRESHOLDS = {
    "almatti": {"full": 1705.0, "warning": 1700.0, "min": 1660.0},
    "bhadra": {"full": 2158.0, "warning": 2150.0, "min": 2100.0},
    "hemavathy": {"full": 2922.0, "warning": 2915.0, "min": 2850.0},
    "kabini": {"full": 2284.0, "warning": 2280.0, "min": 2240.0},
    "krsagara": {"full": 124.8, "warning": 120.0, "min": 80.0},
    "linganamakki": {"full": 1819.0, "warning": 1812.0, "min": 1750.0},
    "malaprabha": {"full": 2079.5, "warning": 2072.0, "min": 2020.0},
    "supa": {"full": 1850.0, "warning": 1840.0, "min": 1770.0},
    "tungabhadra": {"full": 1633.0, "warning": 1628.0, "min": 1580.0},
    "vanivilasa_sagar": {"full": 130.0, "warning": 125.0, "min": 80.0},
}

PRED_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PRED_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SingleStepLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

class MultiStepLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.2, horizon=7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, horizon))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def predict_for_dam(dam_name="krsagara"):
    dam_name = dam_name.lower().replace(" ", "_")
    
    feat_path = os.path.join(BASE_DIR, "data", "features", f"{dam_name}_features.csv")
    scaler_path = os.path.join(BASE_DIR, "data", "splits", dam_name, "scaler.joblib")
    single_model_path = os.path.join(BASE_DIR, "train", "models", f"{dam_name}_lstm_best.pth")
    multi_model_path = os.path.join(BASE_DIR, "train", "models", f"{dam_name}_multistep_lstm.pth")

    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature dataset not found for '{dam_name}' at {feat_path}")

    thresholds = DAM_THRESHOLDS.get(dam_name, {"full": 100.0, "warning": 90.0, "min": 20.0})

    df = pd.read_csv(feat_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    scaler = joblib.load(scaler_path)

    SEQ_LEN = min(60, max(5, len(df) - 1))
    latest_window = df.tail(SEQ_LEN).copy()
    last_date = latest_window["Date"].iloc[-1]
    current_level = latest_window["Reservoir Level (ft)"].iloc[-1]

    scaled_data = scaler.transform(latest_window[feature_cols])
    x_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)

    single_model = SingleStepLSTM(len(feature_cols)).to(device)
    single_model.load_state_dict(torch.load(single_model_path, map_location=device))
    single_model.eval()

    multi_model = MultiStepLSTM(len(feature_cols), horizon=7).to(device)
    multi_model.load_state_dict(torch.load(multi_model_path, map_location=device))
    multi_model.eval()

    with torch.no_grad():
        delta_1day = single_model(x_tensor).cpu().numpy().item()
        deltas_7day = multi_model(x_tensor).cpu().numpy().flatten()

    # Model directly predicts deltas in feet
    pred_level_1day = current_level + delta_1day
    preds_levels_7day = current_level + deltas_7day

    max_7day = np.max(preds_levels_7day)
    min_7day = np.min(preds_levels_7day)

    if max_7day >= thresholds["full"] or pred_level_1day >= thresholds["full"]:
        risk_status = "CRITICAL: OVERFLOW RISK"
        alert_color = "RED"
    elif max_7day >= thresholds["warning"] or pred_level_1day >= thresholds["warning"]:
        risk_status = "WARNING: HIGH RESERVOIR LEVEL"
        alert_color = "YELLOW"
    elif min_7day <= thresholds["min"]:
        risk_status = "ALERT: LOW STORAGE LEVEL"
        alert_color = "ORANGE"
    else:
        risk_status = "NORMAL OPERATIONAL STATUS"
        alert_color = "GREEN"

    print("=" * 65)
    print(f"       {dam_name.upper()} DAM WATER LEVEL FORECAST & RISK MONITOR")
    print("=" * 65)
    print(f" As of Historical Date : {last_date.strftime('%Y-%m-%d')}")
    print(f" Current Reservoir Level: {current_level:.2f} ft")
    print(f" Full Reservoir Capacity: {thresholds['full']:.2f} ft")
    print("-" * 65)
    print(f" Day +1 Forecast       : {pred_level_1day:.2f} ft ({delta_1day:+.2f} ft)")
    print(f" Peak 7-Day Level      : {max_7day:.2f} ft")
    print(f" 7-Day Net Delta       : {deltas_7day[-1]:+.2f} ft")
    print("-" * 65)
    print(f" RISK STATUS: [{alert_color}] {risk_status}")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dam", type=str, default="krsagara", help="Dam identifier name")
    args = parser.parse_args()
    predict_for_dam(args.dam)   