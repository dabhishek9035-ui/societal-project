import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

# 1. Dynamic Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) == "train":
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
    TRAIN_DIR = SCRIPT_DIR
else:
    BASE_DIR = SCRIPT_DIR
    TRAIN_DIR = os.path.join(BASE_DIR, "train")

MODEL_DIR = os.path.join(TRAIN_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 3. Load sequence tensors from data/sequences/
SEQ_DIR = os.path.join(BASE_DIR, "data", "sequences")
SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits")

X_train = np.load(os.path.join(SEQ_DIR, "X_train.npy"))
y_train = np.load(os.path.join(SEQ_DIR, "y_train.npy"))
X_val = np.load(os.path.join(SEQ_DIR, "X_val.npy"))
y_val = np.load(os.path.join(SEQ_DIR, "y_val.npy"))
X_test = np.load(os.path.join(SEQ_DIR, "X_test.npy"))
y_test = np.load(os.path.join(SEQ_DIR, "y_test.npy"))

TARGET_COL_IDX = 0  # Reservoir Level (ft)

# Target Differencing: predict change in level (Delta_y = y_{t+1} - y_t)
delta_train = y_train - X_train[:, -1, TARGET_COL_IDX]
delta_val = y_val - X_val[:, -1, TARGET_COL_IDX]
delta_test = y_test - X_test[:, -1, TARGET_COL_IDX]

# PyTorch DataLoaders
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(delta_train, dtype=torch.float32),
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(delta_val, dtype=torch.float32),
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(delta_test, dtype=torch.float32),
)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 4. Regularized LSTM Architecture
class DeltaLSTM(nn.Module):

    def __init__(
        self, input_dim, hidden_dim=32, num_layers=2, dropout=0.2, output_dim=1
    ):
        super(DeltaLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Extract last hidden state
        return out.squeeze(-1)


# Hyperparameters
INPUT_DIM = X_train.shape[2]  # 17 features
HIDDEN_DIM = 32
NUM_LAYERS = 2
EPOCHS = 100
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4  # L2 Regularization
PATIENCE = 10  # Early stopping patience

model = DeltaLSTM(
    INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=0.2
).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# 5. Training Loop with Early Stopping & Checkpointing
best_val_loss = float("inf")
best_model_path = os.path.join(MODEL_DIR, "almatti_lstm_best.pth")
patience_counter = 0

print("\n--- Starting Training (Target Differencing + Regularization) ---")
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

    # Validation Phase
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
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | Train Loss (MSE):"
            f" {train_loss:.6f} | Val Loss (MSE): {val_loss:.6f}"
        )

    if patience_counter >= PATIENCE:
        print(
            f"\n[Early Stopping] Triggered after {epoch} epochs (Val loss"
            " stagnant)."
        )
        break

# 6. Evaluate Best Model on Unseen Test Data
print("\n--- Evaluating Best Model on Test Data ---")
model.load_state_dict(torch.load(best_model_path))
model.eval()

predicted_deltas = []
with torch.no_grad():
    for X_b, _ in test_loader:
        X_b = X_b.to(device)
        preds = model(X_b)
        predicted_deltas.extend(preds.cpu().numpy())

predicted_deltas = np.array(predicted_deltas)

# Reconstruct predicted level in scaled space: y_hat_{t+1} = y_t + Delta_y_hat
preds_scaled = X_test[:, -1, TARGET_COL_IDX] + predicted_deltas

# Unscale back to real feet
scaler_path = os.path.join(SPLIT_DIR, "scaler.joblib")
scaler = joblib.load(scaler_path)

min_val = scaler.min_[TARGET_COL_IDX]
scale_val = scaler.scale_[TARGET_COL_IDX]

y_test_real = (y_test - min_val) / scale_val
preds_real = (preds_scaled - min_val) / scale_val

rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))
mae = mean_absolute_error(y_test_real, preds_real)
r2 = r2_score(y_test_real, preds_real)

print(f"\nTest Metrics (Real Units - Feet):")
print(f"  RMSE: {rmse:.3f} ft")
print(f"  MAE:  {mae:.3f} ft")
print(f"  R² Score: {r2:.4f}")
print(f"\nBest model saved to: {best_model_path}")