import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

# 1. Setup Plot Style & Directories
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
os.makedirs("plots", exist_ok=True)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Paths Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEQ_DIR = os.path.join(BASE_DIR, "data", "sequences")
SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits")
MODEL_PATH = os.path.join(BASE_DIR, "train", "models", "almatti_lstm_best.pth")

# 3. Model Definition
class DeltaLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.2, output_dim=1):
        super(DeltaLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)

# 4. Load Data & Tensors
X_test = np.load(os.path.join(SEQ_DIR, "X_test.npy"))
y_test = np.load(os.path.join(SEQ_DIR, "y_test.npy"))
test_scaled_df = pd.read_csv(os.path.join(SPLIT_DIR, "test_scaled.csv"))

# Map dates to sequence targets (offset by 60 days history window)
SEQ_LEN = 60
test_dates = pd.to_datetime(test_scaled_df["Date"].values[SEQ_LEN:])

# Load Model
INPUT_DIM = X_test.shape[2]
TARGET_COL_IDX = 0

model = DeltaLSTM(INPUT_DIM, hidden_dim=32, num_layers=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 5. Generate Predictions
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
with torch.no_grad():
    predicted_deltas = model(X_test_tensor).cpu().numpy()

# Reconstruct scaled water levels
preds_scaled = X_test[:, -1, TARGET_COL_IDX] + predicted_deltas

# Unscale back to actual feet
scaler = joblib.load(os.path.join(SPLIT_DIR, "scaler.joblib"))
min_val = scaler.min_[TARGET_COL_IDX]
scale_val = scaler.scale_[TARGET_COL_IDX]

y_real = (y_test - min_val) / scale_val
preds_real = (preds_scaled - min_val) / scale_val
residuals = y_real - preds_real

# 6. Plot 1: Actual vs. Predicted Water Levels
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(test_dates, y_real, label="Actual Level (ft)", color="#1f77b4", linewidth=1.8)
ax.plot(test_dates, preds_real, label="Predicted Level (ft)", color="#ff7f0e", linestyle="--", linewidth=1.5, alpha=0.9)
ax.set_title("Almatti Dam: Actual vs. Predicted Water Levels (Test Set 2023–2025)", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Reservoir Level (ft)", fontsize=12)
ax.legend(fontsize=11, loc="upper left")
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png", dpi=300)
plt.close()

# 7. Plot 2: Residual Analysis (Errors Over Time & Distribution)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# Plot 2a: Residuals over time
ax1.plot(test_dates, residuals, color="#d62728", alpha=0.7, linewidth=1)
ax1.axhline(0, color="black", linestyle="--", linewidth=1)
ax1.set_title("Residual Error Over Time ($y_{actual} - \hat{y}_{pred}$)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Date", fontsize=11)
ax1.set_ylabel("Error (ft)", fontsize=11)
ax1.grid(True, linestyle=":", alpha=0.6)

# Plot 2b: Residual Distribution Histogram
sns.histplot(residuals, kde=True, ax=ax2, color="#2ca02c", bins=40)
ax2.axvline(0, color="black", linestyle="--", linewidth=1)
ax2.set_title("Residual Error Distribution", fontsize=12, fontweight="bold")
ax2.set_xlabel("Error (ft)", fontsize=11)
ax2.set_ylabel("Frequency", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("plots/residual_analysis.png", dpi=300)
plt.close()

print("Plots successfully saved to 'plots/' directory:")
print(" - plots/actual_vs_predicted.png")
print(" - plots/residual_analysis.png")
print(f"\nResidual Stats:")
print(f" Mean Error (Bias): {np.mean(residuals):.4f} ft")
print(f" Std Deviation:    {np.std(residuals):.4f} ft")
print(f" Max Positive Error: {np.max(residuals):.4f} ft")
print(f" Max Negative Error: {np.min(residuals):.4f} ft")