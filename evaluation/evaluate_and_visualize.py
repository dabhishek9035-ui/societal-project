import argparse
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Directory Setup
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EVAL_DIR)
FEATURE_DIR = os.path.join(BASE_DIR, "data", "features")
SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits")
MODEL_DIR = os.path.join(BASE_DIR, "train", "models")
PLOT_DIR = os.path.join(EVAL_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sns.set_theme(style="whitegrid")


# Model Architectures
class SingleStepLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class MultiStepLSTM(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=32, num_layers=2, dropout=0.2, horizon=7
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, horizon)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def calculate_metrics(y_true, y_pred):
    """Calculates RMSE, MAE, R2, and MAPE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Avoid division by zero in MAPE calculation
    non_zero_mask = y_true != 0
    mape = (
        np.mean(
            np.abs(
                (y_true[non_zero_mask] - y_pred[non_zero_mask])
                / y_true[non_zero_mask]
            )
        )
        * 100
    )

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def evaluate_dam(dam_name):
    dam_name = dam_name.lower().replace(" ", "_")

    feat_path = os.path.join(FEATURE_DIR, f"{dam_name}_features.csv")
    scaler_path = os.path.join(SPLIT_DIR, dam_name, "scaler.joblib")
    single_model_path = os.path.join(MODEL_DIR, f"{dam_name}_lstm_best.pth")
    multi_model_path = os.path.join(
        MODEL_DIR, f"{dam_name}_multistep_lstm.pth"
    )

    if not all(
        os.path.exists(p)
        for p in [feat_path, scaler_path, single_model_path, multi_model_path]
    ):
        print(f"⚠️ Skipping {dam_name}: Missing model or feature/scaler files.")
        return None

    # 1. Load Data
    df = pd.read_csv(feat_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = (
        "Reservoir Level (ft)"
        if "Reservoir Level (ft)" in feature_cols
        else feature_cols[0]
    )

    scaler = joblib.load(scaler_path)

    # 2. Extract Test Split (matching prepare_all_dams.py 80/10/10 temporal split)
    n = len(df)
    train_end = max(1, int(n * 0.8))
    val_end = max(train_end + 1, int(n * 0.9))

    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    if len(test_df) < 10:
        print(
            f"⚠️ Skipping {dam_name}: Test set too small ({len(test_df)} samples)."
        )
        return None

    # Scale full dataset to maintain rolling sequence context
    scaled_features = scaler.transform(df[feature_cols])

    seq_len = min(60, max(5, train_end - 8))
    test_start_idx = val_end

    X_test, y_true_1day, actual_levels_base = [], [], []
    y_true_7day = []

    horizon = 7

    for i in range(test_start_idx, n - horizon):
        if i - seq_len < 0:
            continue
        X_test.append(scaled_features[i - seq_len : i])
        actual_curr = df[target_col].iloc[i - 1]
        actual_next_1 = df[target_col].iloc[i]
        actual_next_7 = df[target_col].iloc[i : i + horizon].values

        actual_levels_base.append(actual_curr)
        y_true_1day.append(actual_next_1)
        y_true_7day.append(actual_next_7[-1])  # Compare against day +7 level

    if len(X_test) == 0:
        print(f"⚠️ Insufficient sequences generated for {dam_name}.")
        return None

    X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32).to(
        device
    )
    actual_levels_base = np.array(actual_levels_base)
    y_true_1day = np.array(y_true_1day)
    y_true_7day = np.array(y_true_7day)

    # 3. Load Models & Inference
    single_model = SingleStepLSTM(len(feature_cols)).to(device)
    single_model.load_state_dict(
        torch.load(single_model_path, map_location=device)
    )
    single_model.eval()

    multi_model = MultiStepLSTM(len(feature_cols), horizon=horizon).to(device)
    multi_model.load_state_dict(
        torch.load(multi_model_path, map_location=device)
    )
    multi_model.eval()

    with torch.no_grad():
        pred_deltas_1day = single_model(X_test_tensor).cpu().numpy().flatten()
        pred_deltas_7day = multi_model(X_test_tensor).cpu().numpy()

    y_pred_1day = actual_levels_base + pred_deltas_1day
    y_pred_7day = actual_levels_base + pred_deltas_7day[:, -1]

    # 4. Metrics
    m_1day = calculate_metrics(y_true_1day, y_pred_1day)
    m_7day = calculate_metrics(y_true_7day, y_pred_7day)

    print(
        f"\n📊 Evaluation Metrics for {dam_name.upper()} (Test Samples: {len(y_true_1day)})"
    )
    print(
        f"   1-Day Ahead -> RMSE: {m_1day['RMSE']:.3f} ft | MAE: {m_1day['MAE']:.3f} ft | R²: {m_1day['R2']:.4f} | MAPE: {m_1day['MAPE']:.2f}%"
    )
    print(
        f"   7-Day Ahead -> RMSE: {m_7day['RMSE']:.3f} ft | MAE: {m_7day['MAE']:.3f} ft | R²: {m_7day['R2']:.4f} | MAPE: {m_7day['MAPE']:.2f}%"
    )

    # 5. Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Model Evaluation Dashboard: {dam_name.upper()} Dam",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Time-series Forecast vs Actual (Day +1)
    test_dates = df["Date"].iloc[-len(y_true_1day) :].values
    axes[0, 0].plot(
        test_dates,
        y_true_1day,
        label="Actual Level",
        color="#1f77b4",
        linewidth=2,
    )
    axes[0, 0].plot(
        test_dates,
        y_pred_1day,
        label="1-Day Prediction",
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.8,
    )
    axes[0, 0].set_title("1-Day Ahead Forecast vs Ground Truth")
    axes[0, 0].set_ylabel("Reservoir Level (ft)")
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis="x", rotation=30)

    # Plot 2: Time-series Forecast vs Actual (Day +7)
    axes[0, 1].plot(
        test_dates,
        y_true_7day,
        label="Actual Level (t+7)",
        color="#1f77b4",
        linewidth=2,
    )
    axes[0, 1].plot(
        test_dates,
        y_pred_7day,
        label="7-Day Prediction",
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
    )
    axes[0, 1].set_title("7-Day Ahead Forecast vs Ground Truth")
    axes[0, 1].set_ylabel("Reservoir Level (ft)")
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis="x", rotation=30)

    # Plot 3: Scatter Plot (1-Day)
    sns.regplot(
        x=y_true_1day,
        y=y_pred_1day,
        ax=axes[1, 0],
        color="#d62728",
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "black", "linestyle": "--"},
    )
    axes[1, 0].set_title(
        f"1-Day Scatter Alignment ($R^2 = {m_1day['R2']:.3f}$)"
    )
    axes[1, 0].set_xlabel("Actual Level (ft)")
    axes[1, 0].set_ylabel("Predicted Level (ft)")

    # Plot 4: Residual Distribution
    residuals_1day = y_pred_1day - y_true_1day
    residuals_7day = y_pred_7day - y_true_7day
    sns.histplot(
        residuals_1day,
        ax=axes[1, 1],
        kde=True,
        color="#9467bd",
        label="1-Day Residuals",
        stat="density",
    )
    sns.histplot(
        residuals_7day,
        ax=axes[1, 1],
        kde=True,
        color="#8c564b",
        label="7-Day Residuals",
        stat="density",
        alpha=0.4,
    )
    axes[1, 1].set_title("Residual Error Distribution (Feet)")
    axes[1, 1].set_xlabel("Error (Predicted - Actual in ft)")
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = os.path.join(PLOT_DIR, f"{dam_name}_evaluation.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"   🖼️ Plot saved to: {plot_filename}")

    return {
        "Dam": dam_name.upper(),
        "1Day_RMSE": m_1day["RMSE"],
        "1Day_MAE": m_1day["MAE"],
        "1Day_R2": m_1day["R2"],
        "1Day_MAPE": m_1day["MAPE"],
        "7Day_RMSE": m_7day["RMSE"],
        "7Day_MAE": m_7day["MAE"],
        "7Day_R2": m_7day["R2"],
        "7Day_MAPE": m_7day["MAPE"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dam",
        type=str,
        default="all",
        help="Target dam name or 'all' to evaluate all dams",
    )
    args = parser.parse_args()

    results = []

    if args.dam.lower() == "all":
        dams = [
            d
            for d in os.listdir(SPLIT_DIR)
            if os.path.isdir(os.path.join(SPLIT_DIR, d))
        ]
        for d in dams:
            res = evaluate_dam(d)
            if res:
                results.append(res)
    else:
        res = evaluate_dam(args.dam)
        if res:
            results.append(res)

    if results:
        summary_df = pd.DataFrame(results)
        summary_path = os.path.join(EVAL_DIR, "metrics_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print("\n" + "=" * 80)
        print("                  SUMMARY EVALUATION METRICS ACROSS DAMS")
        print("=" * 80)
        print(
            summary_df.to_string(
                index=False,
                columns=[
                    "Dam",
                    "1Day_RMSE",
                    "1Day_MAE",
                    "1Day_R2",
                    "7Day_RMSE",
                    "7Day_MAE",
                    "7Day_R2",
                ],
            )
        )
        print("=" * 80)
        print(f"📄 Summary report saved to: {summary_path}")