import os
import glob
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "train", "models")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
IMD_DIR = os.path.join(PROJECT_ROOT, "data", "imd_data")
EVAL_RESULTS_PATH = os.path.join(PROJECT_ROOT, "evaluation", "evaluation_results.csv")

# Reservoir structural metadata
DAM_METADATA: Dict[str, Dict[str, Any]] = {
    "almatti": {
        "id": "almatti",
        "name": "Almatti Dam (Lal Bahadur Shastri Sagar)",
        "short_name": "Almatti",
        "basin": "Krishna",
        "sub_basin": "Middle Krishna",
        "river": "Krishna River",
        "district": "Vijayapura / Bagalkote",
        "full_reservoir_level_ft": 1708.0,
        "gross_capacity_tmc": 123.08,
        "live_capacity_tmc": 123.08,
        "purpose": "Irrigation, Hydropower (290 MW), Drinking Water",
        "coordinates": {"lat": 16.331, "lon": 75.888}
    },
    "bhadra": {
        "id": "bhadra",
        "name": "Bhadra Dam (Lakkavalli)",
        "short_name": "Bhadra",
        "basin": "Krishna",
        "sub_basin": "Tungabhadra",
        "river": "Bhadra River",
        "district": "Chikkamagaluru",
        "full_reservoir_level_ft": 2158.0,
        "gross_capacity_tmc": 71.50,
        "live_capacity_tmc": 71.50,
        "purpose": "Irrigation & Hydropower",
        "coordinates": {"lat": 13.702, "lon": 75.642}
    },
    "hemavathy": {
        "id": "hemavathy",
        "name": "Hemavathy Reservoir (Gorur)",
        "short_name": "Hemavathy",
        "basin": "Cauvery",
        "sub_basin": "Upper Cauvery",
        "river": "Hemavathi River",
        "district": "Hassan",
        "full_reservoir_level_ft": 2922.0,
        "gross_capacity_tmc": 37.10,
        "live_capacity_tmc": 37.10,
        "purpose": "Irrigation & Drinking Water",
        "coordinates": {"lat": 12.909, "lon": 76.053}
    },
    "kabini": {
        "id": "kabini",
        "name": "Kabini Reservoir (Beechanahalli)",
        "short_name": "Kabini",
        "basin": "Cauvery",
        "sub_basin": "Kabini Sub-basin",
        "river": "Kabini River",
        "district": "Mysuru",
        "full_reservoir_level_ft": 2284.0,
        "gross_capacity_tmc": 19.50,
        "live_capacity_tmc": 19.50,
        "purpose": "Cauvery Basin Irrigation & Bengaluru Water",
        "coordinates": {"lat": 11.973, "lon": 76.353}
    },
    "krsagara": {
        "id": "krsagara",
        "name": "Krishna Raja Sagara Dam (KRS)",
        "short_name": "KRS Dam",
        "basin": "Cauvery",
        "sub_basin": "Upper Cauvery",
        "river": "Cauvery River",
        "district": "Mandya",
        "full_reservoir_level_ft": 124.8, # Gauge height in ft above bed level (RL 752m)
        "gross_capacity_tmc": 105.79,
        "live_capacity_tmc": 105.79,
        "purpose": "Bengaluru Drinking Water & Cauvery Delta Irrigation",
        "coordinates": {"lat": 12.424, "lon": 76.572}
    },
    "linganamakki": {
        "id": "linganamakki",
        "name": "Linganamakki Dam",
        "short_name": "Linganamakki",
        "basin": "Sharavathi (West Flowing)",
        "sub_basin": "Sharavathi Basin",
        "river": "Sharavathi River",
        "district": "Shivamogga",
        "full_reservoir_level_ft": 1819.0,
        "gross_capacity_tmc": 156.61,
        "live_capacity_tmc": 156.61,
        "purpose": "Sharavathi Hydroelectric Project (Jog Falls)",
        "coordinates": {"lat": 14.197, "lon": 74.843}
    },
    "malaprabha": {
        "id": "malaprabha",
        "name": "Malaprabha Reservoir (Renuka Sagar / Saundatti)",
        "short_name": "Malaprabha",
        "basin": "Krishna",
        "sub_basin": "Malaprabha Sub-basin",
        "river": "Malaprabha River",
        "district": "Belagavi",
        "full_reservoir_level_ft": 2079.5,
        "gross_capacity_tmc": 37.73,
        "live_capacity_tmc": 37.73,
        "purpose": "Irrigation for North Karnataka",
        "coordinates": {"lat": 15.823, "lon": 75.118}
    },
    "supa": {
        "id": "supa",
        "name": "Supa Dam (Kalinadi Hydro)",
        "short_name": "Supa",
        "basin": "Kali (West Flowing)",
        "sub_basin": "Kalinadi",
        "river": "Kali River",
        "district": "Uttara Kannada",
        "full_reservoir_level_ft": 1850.0,
        "gross_capacity_tmc": 147.53,
        "live_capacity_tmc": 147.53,
        "purpose": "Kalinadi Hydroelectric Project (100 MW)",
        "coordinates": {"lat": 15.275, "lon": 74.526}
    },
    "tungabhadra": {
        "id": "tungabhadra",
        "name": "Tungabhadra Dam (Pampa Sagar)",
        "short_name": "Tungabhadra",
        "basin": "Krishna",
        "sub_basin": "Tungabhadra",
        "river": "Tungabhadra River",
        "district": "Vijayanagara / Hosapete",
        "full_reservoir_level_ft": 1633.0,
        "gross_capacity_tmc": 100.80,
        "live_capacity_tmc": 100.80,
        "purpose": "Inter-state Irrigation (Karnataka & Andhra)",
        "coordinates": {"lat": 15.263, "lon": 76.335}
    },
    "vanivilasa_sagar": {
        "id": "vanivilasa_sagar",
        "name": "Vani Vilasa Sagara (Mari Kanive)",
        "short_name": "Vani Vilasa",
        "basin": "Krishna (Vedavathi)",
        "sub_basin": "Vedavathi Basin",
        "river": "Vedavathi River",
        "district": "Chitradurga",
        "full_reservoir_level_ft": 130.0, # Gauge height in ft
        "gross_capacity_tmc": 30.40,
        "live_capacity_tmc": 30.40,
        "purpose": "Drought-prone Chitradurga Irrigation & Heritage Dam",
        "coordinates": {"lat": 13.987, "lon": 76.489}
    }
}

class ReservoirLSTM(nn.Module):
    """Deep PyTorch Delta-LSTM Architecture matching checkpoints."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 4):
        super(ReservoirLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
