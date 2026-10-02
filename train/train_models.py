"""Train one deployable, per-reservoir multi-horizon storage forecaster.

Run from the repository root with ``python train/train_models.py``.  The
checkpoint and scaler filenames intentionally match the existing prediction
services.  Splits are chronological and the model selection score is measured
on held-out forecast origins, never on shuffled rows.
"""

from __future__ import annotations

import copy
import csv
import glob
import json
import os
import random
from datetime import datetime, timezone
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(ROOT, "data", "features")
IMD_DIR = os.path.join(ROOT, "data", "imd_data")
MODELS_DIR = os.path.join(ROOT, "train", "models")
HORIZONS = (1, 7, 14, 30)
SEQUENCE_LENGTH = 14  # Must match the current inference services' default.
SEED = 42
MIN_VALIDATION_SKILL = 0.05  # Promote a learned head only with >=5% lower validation MAE.
MIN_VALIDATION_BLOCK_DAYS = 365  # Require a full seasonal cycle in each validation block.
MIN_TREND_VALIDATION_BLOCK_DAYS = 60  # Require two months per block for the fixed, discounted trend baseline.
MIN_TREND_VALIDATION_SKILL = 0.005  # Require at least 0.5% MAE improvement in both blocks and overall.
TREND_LOOKBACK_DAYS = 30
TREND_MAX_OBSERVATION_GAP_DAYS = 90
TREND_SHRINK = 0.25  # Discount short-term storage momentum for longer horizons.


class ReservoirLSTM(nn.Module):
    """Architecture kept compatible with backend.models.ReservoirLSTM."""

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, output_dim: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


def _number_columns(frame: pd.DataFrame) -> list[str]:
    ignored = {"Date", "Monitoring Date", "Reservoir Name", "Basin", "Sub Basin",
               "River", "_id"}
    return [c for c in frame.columns if c not in ignored and not c.startswith("target_")
            and pd.api.types.is_numeric_dtype(frame[c])]


def _prepare_frame(path: str) -> tuple[str, pd.DataFrame, list[str], str]:
    dam = os.path.basename(path).replace("_features.csv", "")
    frame = pd.read_csv(path)
    if "Date" not in frame:
        raise ValueError(f"{path}: required Date column is missing")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    frame = frame.reset_index(drop=True)
    source_path = os.path.join(IMD_DIR, f"{dam}_dam_ready.csv")
    if os.path.exists(source_path):
        source_dates = pd.read_csv(source_path, usecols=["Date"])["Date"]
        source_latest = pd.to_datetime(source_dates, errors="coerce").max()
        if pd.isna(source_latest) or frame["Date"].iloc[-1] > source_latest:
            raise ValueError(f"{dam}: feature data is newer than its cleaned source; rebuild features before training")
    storage_col = "Current_Storage_TMC" if "Current_Storage_TMC" in frame else "Live Capacity (TMC)"
    if storage_col not in frame:
        raise ValueError(f"{path}: no storage column (Current_Storage_TMC or Live Capacity (TMC))")
    frame[storage_col] = pd.to_numeric(frame[storage_col], errors="coerce")
    # Always rebuild labels from dated observations. This makes horizon semantics
    # calendar-day based even when a reservoir has gaps or irregular reporting.
    observed = frame[["Date", storage_col]].dropna().copy()
    observed = observed.rename(columns={storage_col: "_future_storage"})
    origin_dates = frame["Date"]
    max_gap = pd.Timedelta(days=3)
    for horizon in HORIZONS:
        query = pd.DataFrame({"_wanted": origin_dates + pd.Timedelta(days=horizon), "_row": frame.index})
        matched = pd.merge_asof(query.sort_values("_wanted"),
                                observed.sort_values("Date"),
                                left_on="_wanted", right_on="Date", direction="forward",
                                tolerance=max_gap)
        future = matched.set_index("_row")["_future_storage"].reindex(frame.index)
        frame[f"target_storage_t_plus_{horizon}"] = future
        frame[f"target_delta_t_plus_{horizon}"] = future - frame[storage_col]

    labels = [f"target_delta_t_plus_{h}" for h in HORIZONS]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    features = _number_columns(frame)
    if not features:
        raise ValueError(f"{path}: no numeric predictor columns found")
    # Forward fill only: backward fill would let a historical row use a future
    # observation. Missing leading values remain unavailable and are median-filled
    # from the training period after the chronological split.
    frame[[c for c in features if c != storage_col]] = frame[
        [c for c in features if c != storage_col]
    ].ffill()
    frame = frame.dropna(subset=[storage_col] + labels).reset_index(drop=True)
    return dam, frame, features, storage_col


def _sequence_arrays(x: np.ndarray, y: np.ndarray, storage: np.ndarray,
                     origins: Iterable[int], sequence_length: int):
    valid = [int(i) for i in origins if i >= sequence_length - 1]
    xs = np.stack([x[i - sequence_length + 1:i + 1] for i in valid])
    return xs, y[valid], storage[valid], np.asarray(valid, dtype=int)


def _recent_trend_deltas(frame: pd.DataFrame, origins: np.ndarray,
                         storage_col: str) -> np.ndarray:
    """Return conservative horizon deltas from the preceding 30-day slope."""
    dates = frame["Date"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    storage = frame[storage_col].to_numpy(dtype=float)
    previous = np.searchsorted(dates, dates[origins] - TREND_LOOKBACK_DAYS, side="right") - 1
    valid = previous >= 0
    elapsed = np.zeros(len(origins), dtype=float)
    elapsed[valid] = dates[origins[valid]] - dates[previous[valid]]
    valid &= (elapsed >= 14) & (elapsed <= TREND_MAX_OBSERVATION_GAP_DAYS)
    slope = np.zeros(len(origins), dtype=float)
    slope[valid] = (storage[origins[valid]] - storage[previous[valid]]) / elapsed[valid]
    return np.column_stack([
        slope * horizon * TREND_SHRINK for horizon in HORIZONS
    ])


def _scores(actual: np.ndarray, predicted: np.ndarray) -> dict:
    output = {}
    for j, horizon in enumerate(HORIZONS):
        truth, estimate = actual[:, j], predicted[:, j]
        error = estimate - truth
        denom = np.sum((truth - truth.mean()) ** 2)
        r2 = float(1.0 - np.sum(error ** 2) / denom) if denom > 1e-12 else None
        output[f"t+{horizon}"] = {
            "r2": round(r2, 5) if r2 is not None else None,
            "mae_tmc": round(float(np.mean(np.abs(error))), 5),
            "rmse_tmc": round(float(np.sqrt(np.mean(error ** 2))), 5),
        }
    return output


def _skill_scores(actual: np.ndarray, predicted: np.ndarray,
                  baseline: np.ndarray) -> dict:
    output = {}
    for j, horizon in enumerate(HORIZONS):
        model_error = predicted[:, j] - actual[:, j]
        baseline_error = baseline[:, j] - actual[:, j]
        model_mae = float(np.mean(np.abs(model_error)))
        baseline_mae = float(np.mean(np.abs(baseline_error)))
        model_rmse = float(np.sqrt(np.mean(model_error ** 2)))
        baseline_rmse = float(np.sqrt(np.mean(baseline_error ** 2)))
        output[f"t+{horizon}"] = {
            "model_mae_tmc": round(model_mae, 5),
            "persistence_mae_tmc": round(baseline_mae, 5),
            "mae_skill_pct": round((1.0 - model_mae / baseline_mae) * 100.0, 3) if baseline_mae > 1e-12 else None,
            "model_rmse_tmc": round(model_rmse, 5),
            "persistence_rmse_tmc": round(baseline_rmse, 5),
            "rmse_skill_pct": round((1.0 - model_rmse / baseline_rmse) * 100.0, 3) if baseline_rmse > 1e-12 else None,
        }
    return output


def write_evaluation_csv(summary: dict, output_path: str | None = None) -> str:
    """Publish the same chronological holdout results consumed by the API."""
    if output_path is None:
        output_path = os.path.join(ROOT, "evaluation", "evaluation_results.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fields = ["dam_name", "horizon_days", "rmse_tmc", "mae_tmc", "nse", "r2",
              "persistence_rmse_tmc", "persistence_mae_tmc", "selected_strategy",
              "mae_skill_pct", "rmse_skill_pct", "n_test_origins", "validation_coverage_days",
              "validation_start", "validation_end", "test_coverage_days", "test_start", "test_end"]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in summary.get("results", []):
            for horizon in HORIZONS:
                key = f"t+{horizon}"
                model = result["test_storage_metrics"][key]
                persistence = result["persistence_baseline"][key]
                skill = result.get("test_skill_vs_persistence", {}).get(key)
                if skill is None:
                    model_mae, baseline_mae = model["mae_tmc"], persistence["mae_tmc"]
                    model_rmse, baseline_rmse = model["rmse_tmc"], persistence["rmse_tmc"]
                    skill = {
                        "mae_skill_pct": round((1 - model_mae / baseline_mae) * 100, 3) if baseline_mae > 1e-12 else None,
                        "rmse_skill_pct": round((1 - model_rmse / baseline_rmse) * 100, 3) if baseline_rmse > 1e-12 else None,
                    }
                writer.writerow({
                    "dam_name": result["dam_name"], "horizon_days": key,
                    "rmse_tmc": model["rmse_tmc"], "mae_tmc": model["mae_tmc"],
                    "nse": model["r2"], "r2": model["r2"],
                    "persistence_rmse_tmc": persistence["rmse_tmc"],
                    "persistence_mae_tmc": persistence["mae_tmc"],
                    "selected_strategy": result["horizon_strategy"][key]["selected"],
                    "mae_skill_pct": skill["mae_skill_pct"],
                    "rmse_skill_pct": skill["rmse_skill_pct"],
                    "n_test_origins": result["test_origins"],
                    "validation_coverage_days": result.get("validation_coverage_days", ""),
                    "validation_start": result.get("validation_start", ""),
                    "validation_end": result.get("validation_end", ""),
                    "test_coverage_days": result.get("test_coverage_days", ""),
                    "test_start": result.get("test_start", ""), "test_end": result.get("test_end", ""),
                })
    return output_path


def _predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()


def train_dam_model(feature_file_path: str, sequence_length: int = SEQUENCE_LENGTH,
                    epochs: int = 120, batch_size: int = 64, patience: int = 14,
                    learning_rate: float = 5e-4) -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    dam, frame, feature_cols, storage_col = _prepare_frame(feature_file_path)
    n = len(frame)
    if n < max(120, sequence_length + 60):
        raise ValueError(f"{dam}: only {n} labeled rows; need at least 120 for temporal holdout")

    # Reserve a final chronological test segment; keep validation separate for
    # early stopping. Give each segment the preceding rows as sequence context.
    train_end = int(n * 0.60)
    val_end = int(n * 0.85)
    max_horizon = max(HORIZONS)
    train_origins = np.arange(sequence_length - 1, max(sequence_length - 1, train_end - max_horizon))
    val_origins = np.arange(max(sequence_length - 1, train_end), max(train_end, val_end - max_horizon))
    test_origins = np.arange(max(sequence_length - 1, val_end), n)
    if min(len(train_origins), len(val_origins), len(test_origins)) < 10:
        raise ValueError(f"{dam}: insufficient rows in train/validation/test after horizon and sequence windows")

    # Keep the actual future storage horizon rows available for scoring. Each
    # target was built from a real observation within 3 calendar days of target.
    x_raw = frame[feature_cols].to_numpy(dtype=np.float64)
    y_raw = frame[[f"target_delta_t_plus_{h}" for h in HORIZONS]].to_numpy(dtype=np.float64)
    storage = frame[storage_col].to_numpy(dtype=np.float64)
    train_features = x_raw[:train_end]
    medians = np.nanmedian(train_features, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    x_raw = np.where(np.isfinite(x_raw), x_raw, medians)

    scaler_x = StandardScaler()
    scaler_x.fit(pd.DataFrame(x_raw[:train_end], columns=feature_cols))
    scaler_y = StandardScaler().fit(y_raw[train_origins])
    x_scaled = scaler_x.transform(pd.DataFrame(x_raw, columns=feature_cols)).astype(np.float32)
    y_scaled = scaler_y.transform(y_raw).astype(np.float32)
    train_x, train_y, _, _ = _sequence_arrays(x_scaled, y_scaled, storage, train_origins, sequence_length)
    val_x, val_y, _, val_idx = _sequence_arrays(x_scaled, y_scaled, storage, val_origins, sequence_length)
    test_x, _, _, test_idx = _sequence_arrays(x_scaled, y_scaled, storage, test_origins, sequence_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReservoirLSTM(len(feature_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                           patience=4, min_lr=1e-5)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                              batch_size=min(batch_size, len(train_x)), shuffle=True)
    val_tensor = torch.as_tensor(val_x, dtype=torch.float32, device=device)
    val_target = torch.as_tensor(val_y, dtype=torch.float32, device=device)
    loss_fn = nn.SmoothL1Loss()
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_tensor), val_target).item())
        scheduler.step(val_loss)
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, best_epoch, stale = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"{dam}: training did not produce a valid checkpoint")
    model.load_state_dict(best_state)

    capacity = float(pd.to_numeric(frame.get("Design Gross Capacity (TMC)", pd.Series(dtype=float)),
                                   errors="coerce").dropna().median()) if "Design Gross Capacity (TMC)" in frame else np.nan
    if not np.isfinite(capacity) or capacity <= 0:
        capacity = float(np.nanmax(storage))

    # Select each head from validation only. The final test segment is never
    # used to decide which forecast is deployed.
    val_delta = scaler_y.inverse_transform(_predict(model, val_x, device))
    val_true = np.clip(storage[val_idx, None] + y_raw[val_idx], 0, capacity)
    val_model = np.clip(storage[val_idx, None] + val_delta, 0, capacity)
    val_persistence = np.repeat(storage[val_idx, None], len(HORIZONS), axis=1)
    val_trend_delta = _recent_trend_deltas(frame, val_idx, storage_col)
    val_trend = np.clip(storage[val_idx, None] + val_trend_delta, 0, capacity)
    validation_start = frame.loc[val_idx[0], "Date"]
    validation_end = frame.loc[val_idx[-1], "Date"]
    validation_coverage_days = (validation_end - validation_start).days + 1
    validation_midpoint = len(val_idx) // 2
    validation_blocks = (np.arange(validation_midpoint), np.arange(validation_midpoint, len(val_idx)))
    validation_block_reports = []
    for block_idx in validation_blocks:
        block_origins = val_idx[block_idx]
        block_start = frame.loc[block_origins[0], "Date"]
        block_end = frame.loc[block_origins[-1], "Date"]
        validation_block_reports.append({
            "start": block_start.strftime("%Y-%m-%d"),
            "end": block_end.strftime("%Y-%m-%d"),
            "coverage_days": (block_end - block_start).days + 1,
        })
    horizon_strategy = {}
    with torch.no_grad():
        for j, horizon in enumerate(HORIZONS):
            model_mae = float(np.mean(np.abs(val_model[:, j] - val_true[:, j])))
            persistence_mae = float(np.mean(np.abs(val_persistence[:, j] - val_true[:, j])))
            trend_mae = float(np.mean(np.abs(val_trend[:, j] - val_true[:, j])))
            mae_skill = 1.0 - model_mae / persistence_mae if persistence_mae > 1e-12 else 0.0
            trend_skill = 1.0 - trend_mae / persistence_mae if persistence_mae > 1e-12 else 0.0
            block_scores = []
            for block_idx in validation_blocks:
                block_model_mae = float(np.mean(np.abs(val_model[block_idx, j] - val_true[block_idx, j])))
                block_persistence_mae = float(np.mean(np.abs(val_persistence[block_idx, j] - val_true[block_idx, j])))
                block_trend_mae = float(np.mean(np.abs(val_trend[block_idx, j] - val_true[block_idx, j])))
                block_scores.append({
                    "model_mae_tmc": round(block_model_mae, 5),
                    "persistence_mae_tmc": round(block_persistence_mae, 5),
                    "mae_skill_pct": round((1.0 - block_model_mae / block_persistence_mae) * 100.0, 3) if block_persistence_mae > 1e-12 else None,
                    "recent_trend_mae_tmc": round(block_trend_mae, 5),
                    "recent_trend_skill_pct": round((1.0 - block_trend_mae / block_persistence_mae) * 100.0, 3) if block_persistence_mae > 1e-12 else None,
                })
            has_full_year_blocks = all(block["coverage_days"] >= MIN_VALIDATION_BLOCK_DAYS for block in validation_block_reports)
            stable_across_blocks = all(
                score["mae_skill_pct"] is not None
                and score["mae_skill_pct"] >= MIN_VALIDATION_SKILL * 100.0
                for score in block_scores
            )
            trend_stable = all(
                score["recent_trend_skill_pct"] is not None
                and score["recent_trend_skill_pct"] >= MIN_TREND_VALIDATION_SKILL * 100.0
                for score in block_scores
            )
            model_qualified = (has_full_year_blocks and stable_across_blocks
                               and mae_skill >= MIN_VALIDATION_SKILL)
            trend_has_sufficient_blocks = all(
                block["coverage_days"] >= MIN_TREND_VALIDATION_BLOCK_DAYS
                for block in validation_block_reports
            )
            trend_qualified = (trend_has_sufficient_blocks and trend_stable
                               and trend_skill >= MIN_TREND_VALIDATION_SKILL)
            # Compare the LSTM and a discounted 30-day trend baseline on the
            # same validation windows. Both need repeatable improvement over
            # persistence before being selected.
            selected = "persistence"
            if model_qualified and (not trend_qualified or model_mae <= trend_mae):
                selected = "model"
            elif trend_qualified:
                selected = "recent_trend"
            horizon_strategy[f"t+{horizon}"] = {
                "selected": selected,
                "validation_model_mae_tmc": round(model_mae, 5),
                "validation_persistence_mae_tmc": round(persistence_mae, 5),
                "validation_recent_trend_mae_tmc": round(trend_mae, 5),
                "validation_mae_skill_pct": round(mae_skill * 100.0, 3),
                "validation_recent_trend_skill_pct": round(trend_skill * 100.0, 3),
                "minimum_skill_required_pct": MIN_VALIDATION_SKILL * 100.0,
                "validation_coverage_days": validation_coverage_days,
                "validation_blocks": validation_block_reports,
                "block_scores": block_scores,
                "minimum_validation_block_days": MIN_VALIDATION_BLOCK_DAYS,
                "minimum_validation_skill_each_block_pct": MIN_VALIDATION_SKILL * 100.0,
                "beats_persistence_in_both_validation_blocks": stable_across_blocks,
                "recent_trend_beats_persistence_in_both_validation_blocks": trend_stable,
                "recent_trend_validation_blocks_sufficient": trend_has_sufficient_blocks,
                "minimum_recent_trend_validation_block_days": MIN_TREND_VALIDATION_BLOCK_DAYS,
                "minimum_recent_trend_validation_skill_pct": MIN_TREND_VALIDATION_SKILL * 100.0,
                "trend_lookback_days": TREND_LOOKBACK_DAYS,
                "trend_max_observation_gap_days": TREND_MAX_OBSERVATION_GAP_DAYS,
                "trend_shrink": TREND_SHRINK,
            }
            if selected != "model":
                model.fc.weight[j].zero_()
                model.fc.bias[j] = float(-scaler_y.mean_[j] / scaler_y.scale_[j])

    # Report comparable storage-space forecast performance, plus persistence.
    delta_scaled = _predict(model, test_x, device)
    delta_pred = scaler_y.inverse_transform(delta_scaled)
    test_trend_delta = _recent_trend_deltas(frame, test_idx, storage_col)
    for j, horizon in enumerate(HORIZONS):
        if horizon_strategy[f"t+{horizon}"]["selected"] == "recent_trend":
            delta_pred[:, j] = test_trend_delta[:, j]
    y_true_delta = y_raw[test_idx]
    true_storage = storage[test_idx, None] + y_true_delta
    pred_storage = storage[test_idx, None] + delta_pred
    true_storage = np.clip(true_storage, 0, capacity)
    pred_storage = np.clip(pred_storage, 0, capacity)
    persistence = np.repeat(storage[test_idx, None], len(HORIZONS), axis=1)
    report = {
        "dam_name": dam,
        "rows_labeled": n,
        "feature_count": len(feature_cols),
        "sequence_length": sequence_length,
        "best_epoch": best_epoch,
        "test_origins": len(test_idx),
        "validation_start": validation_start.strftime("%Y-%m-%d"),
        "validation_end": validation_end.strftime("%Y-%m-%d"),
        "validation_coverage_days": validation_coverage_days,
        "test_start": frame.loc[test_idx[0], "Date"].strftime("%Y-%m-%d"),
        "test_end": frame.loc[test_idx[-1], "Date"].strftime("%Y-%m-%d"),
        "test_coverage_days": (frame.loc[test_idx[-1], "Date"] - frame.loc[test_idx[0], "Date"]).days + 1,
        "horizon_strategy": horizon_strategy,
        "test_storage_metrics": _scores(true_storage, pred_storage),
        "persistence_baseline": _scores(true_storage, persistence),
        "test_skill_vs_persistence": _skill_scores(true_storage, pred_storage, persistence),
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    # Preserve all legacy filenames used by backend and command-line inference.
    model_path = os.path.join(MODELS_DIR, f"{dam}_lstm.pth")
    scaler_x_path = os.path.join(MODELS_DIR, f"{dam}_scaler_x.pkl")
    scaler_y_path = os.path.join(MODELS_DIR, f"{dam}_scaler_y.pkl")
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler_x, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)
    # Manifest carries schema and metrics for retraining/promotion decisions.
    manifest = {
        **report,
        "feature_columns": feature_cols,
        "horizons_days": list(HORIZONS),
        "missing_feature_fill_values": dict(zip(feature_cols, medians.tolist())),
        "capacity_tmc": capacity,
        "horizon_strategy": horizon_strategy,
        "model_file": os.path.basename(model_path),
    }
    with open(os.path.join(MODELS_DIR, f"{dam}_manifest.json"), "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"{dam}: {len(train_x)} train, {len(val_x)} validation, {len(test_x)} test windows; "
          f"best epoch {best_epoch}")
    for horizon in HORIZONS:
        key = f"t+{horizon}"
        model_mae = report["test_storage_metrics"][key]["mae_tmc"]
        baseline_mae = report["persistence_baseline"][key]["mae_tmc"]
        skill = report["test_skill_vs_persistence"][key]["mae_skill_pct"]
        print(f"  {key}: selected={horizon_strategy[key]['selected']}; test MAE={model_mae:.3f} TMC "
              f"vs persistence={baseline_mae:.3f} TMC; skill={skill if skill is not None else 'n/a'}%")
    print(f"  Saved compatible checkpoint/scalers and {dam}_manifest.json")
    return report


def main() -> None:
    files = sorted(glob.glob(os.path.join(FEATURES_DIR, "*_features.csv")))
    if not files:
        raise SystemExit(f"No feature CSV files found under {FEATURES_DIR}. Run build_features.py first.")
    results, failures = [], []
    for path in files:
        try:
            results.append(train_dam_model(path))
        except Exception as exc:
            failures.append({"file": os.path.basename(path), "error": str(exc)})
            print(f"SKIP {os.path.basename(path)}: {exc}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    summary = {
        "run_id": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "selection_metric": "validation MAE; choose the LSTM only with >=5% pooled and per-block skill in both 365-day blocks, or choose a discounted 30-day trend only with >=0.5% pooled and per-block skill in both 60-day blocks; otherwise use persistence",
        "test_policy": "final chronological holdout; used for reporting only, never model selection",
        "results": results,
        "failures": failures,
    }
    with open(os.path.join(MODELS_DIR, "training_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    csv_path = write_evaluation_csv(summary)
    print(f"Wrote API-compatible holdout metrics to {csv_path}")
    if not results:
        raise SystemExit("No reservoir model was trained successfully; see the errors above.")


if __name__ == "__main__":
    main()
