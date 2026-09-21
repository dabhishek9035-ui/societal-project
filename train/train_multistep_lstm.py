import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

# 1. Setup Environment & Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits")
MODEL_DIR = os.path.join(BASE_DIR, "train", "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Hyperparameters
SEQ_LEN = 60       # 60 days input history
HORIZON = 7        # Predict 7 days into the future
BATCH_SIZE = 32
HIDDEN_DIM = 32
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 12

TARGET_COL_IDX = 0  # Reservoir Level (ft)

# 3. Build Multi-Horizon Sequences
train_scaled_df = pd.read_csv(os.path.join(SPLIT_DIR, "train_scaled.csv"))
val_scaled_df = pd.read_csv(os.path.join(SPLIT_DIR, "val_scaled.csv"))
test_scaled_df = pd.read_csv(os.path.join(SPLIT_DIR, "test_scaled.csv"))

feature_cols = [c for c in train_scaled_df.columns if c != "Date"]

def create_multistep_sequences(df, seq_len=60, horizon=7):
    data = df[feature_cols].values
    X, Y, Y_abs = [], [], []
    for i in range(len(data) - seq_len - horizon + 1):
        x_window = data[i : i + seq_len, :]
        current_level = data[i + seq_len - 1, TARGET_COL_IDX]
        future_levels = data[i + seq_len : i + seq_len + horizon, TARGET_COL_IDX]
        
        # Delta sequence relative to last historical day (t)
        delta_targets = future_levels - current_level
        
        X.append(x_window)
        Y.append(delta_targets)
        Y_abs.append(future_levels)
        
    return np.array(X), np.array(Y), np.array(Y_abs)

X_train, y_train_delta, y_train_abs = create_multistep_sequences(train_scaled_df, SEQ_LEN, HORIZON)
X_val, y_val_delta, y_val_abs = create_multistep_sequences(val_scaled_df, SEQ_LEN, HORIZON)
X_test, y_test_delta, y_test_abs = create_multistep_sequences(test_scaled_df, SEQ_LEN, HORIZON)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_delta, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val_delta, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test_delta, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. Multi-Horizon LSTM Architecture
class MultiStepLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.2, horizon=7):
        super(MultiStepLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take final hidden state
        return out

INPUT_DIM = X_train.shape[2]
model = MultiStepLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, HORIZON).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# 5. Model Training Loop
best_val_loss = float("inf")
best_model_path = os.path.join(MODEL_DIR, "almatti_multistep_lstm.pth")
patience_counter = 0

print(f"\n--- Training 7-Day Multi-Horizon LSTM ({INPUT_DIM} features) ---")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        preds = model(X_b)
        loss = criterion(preds, y_b)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(y_b)
    
    train_loss /= len(train_dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            preds = model(X_b)
            loss = criterion(preds, y_b)
            val_loss += loss.item() * len(y_b)
            
    val_loss /= len(val_dataset)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        patience_counter = 0
    else:
        patience_counter += 1
        
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")
        
    if patience_counter >= PATIENCE:
        print(f"\n[Early Stopping] Triggered at epoch {epoch}.")
        break

# 6. Unscale & Evaluate Performance by Horizon
model.load_state_dict(torch.load(best_model_path))
model.eval()

pred_deltas_list = []
with torch.no_grad():
    for X_b, _ in test_loader:
        X_b = X_b.to(device)
        preds = model(X_b)
        pred_deltas_list.append(preds.cpu().numpy())

pred_deltas = np.vstack(pred_deltas_list)

# Reconstruct scaled predictions: y_hat_{t+k} = y_t + Delta_y_{t+k}
last_hist_levels = X_test[:, -1, TARGET_COL_IDX, None]  # shape (N, 1)
pred_scaled_abs = last_hist_levels + pred_deltas

# Unscale to real feet
scaler = joblib.load(os.path.join(SPLIT_DIR, "scaler.joblib"))
min_val = scaler.min_[TARGET_COL_IDX]
scale_val = scaler.scale_[TARGET_COL_IDX]

y_test_real = (y_test_abs - min_val) / scale_val
preds_real = (pred_scaled_abs - min_val) / scale_val

print("\n--- Horizon-by-Horizon Performance (Test Set) ---")
for h in range(HORIZON):
    mae_h = mean_absolute_error(y_test_real[:, h], preds_real[:, h])
    rmse_h = np.sqrt(mean_squared_error(y_test_real[:, h], preds_real[:, h]))
    r2_h = r2_score(y_test_real[:, h], preds_real[:, h])
    print(f"Day +{h+1}: MAE = {mae_h:.3f} ft | RMSE = {rmse_h:.3f} ft | R² = {r2_h:.4f}")

# 7. Visualization: Sample 7-Day Forecast Cones
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(14, 6))

test_dates = pd.to_datetime(test_scaled_df["Date"].values[SEQ_LEN : SEQ_LEN + len(y_test_real)])

# Plot actual trajectory
ax.plot(test_dates, y_test_real[:, 0], label="Actual Level (Day +1)", color="#1f77b4", linewidth=1.5, alpha=0.7)

# Overlay 7-day forecast trajectories every 60 days
for sample_idx in range(0, len(y_test_real) - HORIZON, 60):
    start_date = test_dates[sample_idx]
    forecast_dates = [start_date + pd.Timedelta(days=k) for k in range(HORIZON)]
    ax.plot(
        forecast_dates,
        preds_real[sample_idx, :],
        color="#e377c2",
        linestyle="--",
        marker="o",
        markersize=4,
        alpha=0.85
    )

ax.set_title("Almatti Dam: 7-Day Horizon Multi-Step Forecast Curves", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Reservoir Level (ft)", fontsize=11)
ax.legend(["Actual Level", "7-Day Forecast Cones"], loc="upper left")
plt.tight_layout()

plot_path = os.path.join(PLOT_DIR, "multistep_forecast.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"\nSaved multi-step forecast plot to: {plot_path}")
print(f"Saved multi-step model to: {best_model_path}")