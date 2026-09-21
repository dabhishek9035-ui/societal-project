import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn

st.set_page_config(page_title="Multi-Dam Hydrological Forecast Platform", page_icon="🌊", layout="wide")

DAM_CONFIGS = {
    "Almatti": {"key": "almatti", "full": 1705.0, "warning": 1700.0, "min": 1660.0},
    "Bhadra": {"key": "bhadra", "full": 2158.0, "warning": 2150.0, "min": 2100.0},
    "Hemavathy": {"key": "hemavathy", "full": 2922.0, "warning": 2915.0, "min": 2850.0},
    "Kabini": {"key": "kabini", "full": 2284.0, "warning": 2280.0, "min": 2240.0},
    "KRSagara (KRS)": {"key": "krsagara", "full": 124.8, "warning": 120.0, "min": 80.0},
    "Linganamakki": {"key": "linganamakki", "full": 1819.0, "warning": 1812.0, "min": 1750.0},
    "Malaprabha": {"key": "malaprabha", "full": 2079.5, "warning": 2072.0, "min": 2020.0},
    "Supa": {"key": "supa", "full": 1850.0, "warning": 1840.0, "min": 1770.0},
    "Tungabhadra": {"key": "tungabhadra", "full": 1633.0, "warning": 1628.0, "min": 1580.0},
    "Vanivilasa Sagar": {"key": "vanivilasa_sagar", "full": 130.0, "warning": 125.0, "min": 80.0},
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

st.sidebar.title("🌊 Dam Selector")
selected_dam_label = st.sidebar.selectbox("Select Target Reservoir:", list(DAM_CONFIGS.keys()))
config = DAM_CONFIGS[selected_dam_label]
dam_key = config["key"]

@st.cache_data
def load_dam_data(key):
    feat_path = os.path.join(BASE_DIR, "data", "features", f"{key}_features.csv")
    if not os.path.exists(feat_path):
        return None
    df = pd.read_csv(feat_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)

df = load_dam_data(dam_key)

if df is None:
    st.error(f"No feature dataset found for {selected_dam_label}. Run `python prepare_all_dams.py` first.")
    st.stop()

feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

scaler_path = os.path.join(BASE_DIR, "data", "splits", dam_key, "scaler.joblib")
single_model_path = os.path.join(BASE_DIR, "train", "models", f"{dam_key}_lstm_best.pth")
multi_model_path = os.path.join(BASE_DIR, "train", "models", f"{dam_key}_multistep_lstm.pth")

if not os.path.exists(single_model_path):
    st.error(f"Models not trained for {selected_dam_label}. Run `python train/train_all_dams.py` first.")
    st.stop()

scaler = joblib.load(scaler_path)

single_model = SingleStepLSTM(len(feature_cols)).to(device)
single_model.load_state_dict(torch.load(single_model_path, map_location=device))
single_model.eval()

multi_model = MultiStepLSTM(len(feature_cols), horizon=7).to(device)
multi_model.load_state_dict(torch.load(multi_model_path, map_location=device))
multi_model.eval()

history_window = st.sidebar.slider("Historical Window (Days)", 30, 365, 120)
seq_len = min(60, max(5, len(df) - 1))
selected_idx = st.sidebar.number_input("Simulation Index", seq_len, len(df) - 1, len(df) - 1)

sequence_df = df.iloc[selected_idx - seq_len + 1 : selected_idx + 1].copy()
current_date = sequence_df["Date"].iloc[-1]
current_level = sequence_df["Reservoir Level (ft)"].iloc[-1]

scaled_input = scaler.transform(sequence_df[feature_cols])
x_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    delta_1day = single_model(x_tensor).cpu().numpy().item()
    deltas_7day = multi_model(x_tensor).cpu().numpy().flatten()

pred_1day = current_level + delta_1day
preds_7day = current_level + deltas_7day

forecast_dates = [current_date + pd.Timedelta(days=i) for i in range(1, 8)]
max_7day = float(np.max(preds_7day))

st.title(f"🌊 {selected_dam_label} Dam Forecast & Risk Platform")
st.markdown("Automated Multi-Horizon Hydrological Forecasting System")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Level", f"{current_level:.2f} ft", f"{current_date.strftime('%Y-%m-%d')}")
col2.metric("Day +1 Forecast", f"{pred_1day:.2f} ft", f"{delta_1day:+.2f} ft")
col3.metric("7-Day Peak Level", f"{max_7day:.2f} ft", f"{max_7day - current_level:+.2f} ft")
col4.metric("7-Day Net Delta", f"{preds_7day[-1]:.2f} ft", f"{deltas_7day[-1]:+.2f} ft")
col5.metric("Storage %", f"{(current_level / config['full']) * 100:.1f} %")

hist_df = df.iloc[max(0, selected_idx - history_window) : selected_idx + 1]
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Reservoir Level (ft)"], mode="lines", name="Historical Level", line=dict(color="#1f77b4", width=2.5)))
fig.add_trace(go.Scatter(x=[current_date] + forecast_dates, y=[current_level] + list(preds_7day), mode="lines+markers", name="7-Day Forecast", line=dict(color="#ff7f0e", width=3, dash="dash")))

fig.add_hline(y=config["full"], line_dash="dot", line_color="red", annotation_text=f"Full Capacity ({config['full']} ft)")
fig.add_hline(y=config["warning"], line_dash="dot", line_color="orange", annotation_text=f"Warning ({config['warning']} ft)")

fig.update_layout(xaxis_title="Date", yaxis_title="Reservoir Level (ft)", height=500, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)