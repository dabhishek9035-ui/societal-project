import os
import glob
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "train", "models")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")

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

# 1. Model Architecture Matching Trained Checkpoints
class ReservoirLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=4):
        super(ReservoirLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# 2. Prediction Pipeline Function
def predict_reservoir_storage(dam_name: str, sequence_length: int = 14):
    """
    Loads latest sequence data, scales features, runs inference on the trained
    PyTorch Delta-LSTM model, and returns predicted storage for t+1, t+7, t+14, and t+30 days.
    """
    model_path = os.path.join(MODELS_DIR, f"{dam_name}_lstm.pth")
    scaler_x_path = os.path.join(MODELS_DIR, f"{dam_name}_scaler_x.pkl")
    scaler_y_path = os.path.join(MODELS_DIR, f"{dam_name}_scaler_y.pkl")
    feature_path = os.path.join(FEATURES_DIR, f"{dam_name}_features.csv")

    for p in [model_path, scaler_x_path, scaler_y_path, feature_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    df = pd.read_csv(feature_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    if hasattr(scaler_x, 'feature_names_in_'):
        feature_cols = list(scaler_x.feature_names_in_)
    else:
        target_cols = [c for c in df.columns if c.startswith('target_')]
        ignore_cols = ['Date', 'Reservoir Name', 'Basin', 'Monitoring Date', 'River', 'Sub Basin', '_id'] + target_cols
        feature_cols = [c for c in df.columns if c not in ignore_cols and np.issubdtype(df[c].dtype, np.number)]

    if len(df) < sequence_length:
        raise ValueError(f"Insufficient historical rows ({len(df)}) for sequence length {sequence_length}.")

    latest_df = df.iloc[-sequence_length:].copy()
    latest_date = latest_df['Date'].iloc[-1]
    current_storage = float(latest_df['Current_Storage_TMC'].iloc[-1])
    max_cap = DAM_CAPACITIES.get(dam_name, 120.0)

    # Scale Features
    X_scaled = scaler_x.transform(latest_df[feature_cols])
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReservoirLSTM(input_dim=len(feature_cols), output_dim=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        delta_scaled = model(X_tensor).cpu().numpy()

    delta_raw = scaler_y.inverse_transform(delta_scaled)[0]

    forecast_horizons = [1, 7, 14, 30]
    forecasts = []
    for h, val in zip(forecast_horizons, delta_raw):
        forecast_date = latest_date + timedelta(days=h)
        pred_storage = float(np.clip(current_storage + val, 0.0, max_cap))
        forecasts.append({
            "horizon_days": h,
            "target_date": forecast_date.strftime("%Y-%m-%d"),
            "predicted_delta_tmc": round(float(val), 4),
            "predicted_storage_tmc": round(pred_storage, 4),
            "percentage_full": round((pred_storage / max_cap * 100.0), 1)
        })

    return {
        "dam_name": dam_name,
        "as_of_date": latest_date.strftime("%Y-%m-%d"),
        "latest_observed_storage_tmc": round(current_storage, 4),
        "gross_capacity_tmc": max_cap,
        "forecasts": forecasts
    }

if __name__ == "__main__":
    test_dams = [f.replace("_lstm.pth", "") for f in os.listdir(MODELS_DIR) if f.endswith("_lstm.pth")]
    if test_dams:
        dam = "almatti"
        print(f"Running test prediction for: {dam}")
        result = predict_reservoir_storage(dam)
        import json
        print(json.dumps(result, indent=2))