import glob
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits")
MODEL_DIR = os.path.join(BASE_DIR, "train", "models")

os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
BATCH_SIZE = 16
EPOCHS = 30
LR = 0.001

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

def create_single_step_sequences(df, feature_cols, target_col, seq_len):
    X, y = [], []
    data = df[feature_cols].values
    target_idx = feature_cols.index(target_col)

    for i in range(len(df) - seq_len):
        X.append(data[i : i + seq_len])
        # Predict delta relative to current sequence end
        current_val = data[i + seq_len - 1, target_idx]
        next_val = data[i + seq_len, target_idx]
        y.append(next_val - current_val)

    return np.array(X), np.array(y)

def create_multi_step_sequences(df, feature_cols, target_col, seq_len, horizon=7):
    X, y = [], []
    data = df[feature_cols].values
    target_idx = feature_cols.index(target_col)

    for i in range(len(df) - seq_len - horizon + 1):
        X.append(data[i : i + seq_len])
        current_val = data[i + seq_len - 1, target_idx]
        future_vals = data[i + seq_len : i + seq_len + horizon, target_idx]
        y.append(future_vals - current_val)

    return np.array(X), np.array(y)

def train_dam_models(dam_dir):
    dam_name = os.path.basename(dam_dir)
    print(f"\n==========================================")
    print(f"   TRAINING MODELS FOR: {dam_name.upper()}")
    print(f"==========================================")

    tr_path = os.path.join(dam_dir, "train_scaled.csv")
    va_path = os.path.join(dam_dir, "val_scaled.csv")

    if not os.path.exists(tr_path):
        print(f"⚠️ Split files not found for {dam_name}. Skipping.")
        return

    tr_df = pd.read_csv(tr_path)
    va_df = pd.read_csv(va_path)

    feature_cols = [c for c in tr_df.columns if c != "Date"]
    target_col = "Reservoir Level (ft)" if "Reservoir Level (ft)" in feature_cols else feature_cols[0]

    # Dynamically select sequence length based on training sample size
    seq_len = min(60, max(5, len(tr_df) - 8))
    
    X_tr_1, y_tr_1 = create_single_step_sequences(tr_df, feature_cols, target_col, seq_len)
    X_va_1, y_va_1 = create_single_step_sequences(va_df, feature_cols, target_col, seq_len)

    if len(X_tr_1) == 0:
        print(f"⚠️ Not enough samples in {dam_name} to form sequence length {seq_len}. Skipping.")
        return

    # --- 1. Train Single-Step LSTM ---
    print(f"1. Training Single-Step LSTM (Seq Len: {seq_len})...")
    tr_ds = TensorDataset(torch.tensor(X_tr_1, dtype=torch.float32), torch.tensor(y_tr_1, dtype=torch.float32))
    tr_loader = DataLoader(tr_ds, batch_size=min(BATCH_SIZE, len(tr_ds)), shuffle=True)

    single_model = SingleStepLSTM(len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(single_model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    single_model.train()
    for epoch in range(EPOCHS):
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = single_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    torch.save(single_model.state_dict(), os.path.join(MODEL_DIR, f"{dam_name}_lstm_best.pth"))

    # --- 2. Train Multi-Step LSTM ---
    horizon = min(7, max(1, len(tr_df) - seq_len))
    X_tr_m, y_tr_m = create_multi_step_sequences(tr_df, feature_cols, target_col, seq_len, horizon=horizon)

    if len(X_tr_m) > 0:
        print(f"2. Training Multi-Step LSTM (Horizon: {horizon})...")
        tr_ds_m = TensorDataset(torch.tensor(X_tr_m, dtype=torch.float32), torch.tensor(y_tr_m, dtype=torch.float32))
        tr_loader_m = DataLoader(tr_ds_m, batch_size=min(BATCH_SIZE, len(tr_ds_m)), shuffle=True)

        multi_model = MultiStepLSTM(len(feature_cols), horizon=horizon).to(device)
        optimizer_m = torch.optim.Adam(multi_model.parameters(), lr=LR)

        multi_model.train()
        for epoch in range(EPOCHS):
            for bx, by in tr_loader_m:
                bx, by = bx.to(device), by.to(device)
                optimizer_m.zero_grad()
                out = multi_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer_m.step()

        torch.save(multi_model.state_dict(), os.path.join(MODEL_DIR, f"{dam_name}_multistep_lstm.pth"))
        print(f"✅ Models trained and saved for {dam_name}.")

if __name__ == "__main__":
    dam_dirs = [os.path.join(SPLIT_DIR, d) for d in os.listdir(SPLIT_DIR) if os.path.isdir(os.path.join(SPLIT_DIR, d))]
    for d in dam_dirs:
        train_dam_models(d)