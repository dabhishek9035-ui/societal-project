"""Train chronological, API-compatible reservoir storage forecasts (v3).

Run from the repository root: ``python train/train_models.py``.

Artifact contract with the backend is unchanged: IndependentHorizonLSTM state
dict, scaler_x / scaler_y pickles, manifest.json, training_summary.json,
evaluation_results.csv, and the strategy names persistence / model /
recent_trend / seasonal_naive.

What changed vs v2 (why the learned model was never getting picked):
  1. Loss was SmoothL1 on standardized deltas, i.e. ~MSE. Monsoon surges are
     heavy-tailed, so the net chased the mean while we score MAE (median
     territory). Now plain L1, which matches the metric.
  2. One 20% validation window + "must beat persistence by 5% in all three
     365-day blocks" is basically a coin flip per dam, and made the three
     short-history dams ineligible by construction. Now: rolling-origin CV
     (3 expanding folds) -> pooled out-of-fold predictions -> gate on pooled
     skill + majority-of-folds + block-bootstrap lower bound.
  3. Epoch count was found under ReduceLROnPlateau, then refit with a constant
     LR. Now constant LR everywhere, so the epoch count actually transfers.
  4. Raw LSTM deltas are noisy. A per-horizon shrink factor (fit on OOF) pulls
     them toward persistence. It is baked into scaler_y, so the backend's
     ``inverse_transform`` returns the shrunk delta with no code change.
  5. Training windows that span big date gaps (14 *rows* != 14 days) are
     dropped from training. Val/test origins are untouched, so scoring stays
     honest.
  6. Learned model is only a candidate when OOF history spans >= 1 year.
  7. Input noise regularises the long-horizon heads; deploy epochs floor is 1.
The final 15% is reporting-only and never touches selection or training.
"""

from __future__ import annotations

import copy
import csv
import glob
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.models import DAM_METADATA, FEATURES_DIR, IMD_DIR, MODELS_DIR, IndependentHorizonLSTM


HORIZONS = (1, 7, 14, 30)
SEQUENCE_LENGTH = 14  # Must match ForecastService and prediction/predict.py.

# --- Splits -----------------------------------------------------------------
# Expanding-window CV over the first 85% of rows. Fold k trains on rows
# [0, edge_k) and predicts [edge_k, edge_k+1). Pooled out-of-fold predictions
# drive strategy selection. The last 15% is the untouched test set.
FOLD_EDGES = (0.55, 0.65, 0.75, 0.85)
DEV_FRACTION = FOLD_EDGES[-1]
EARLY_STOP_TAIL = 0.15  # tail of each fold's train rows used only for early stopping
MAX_LABEL_DATE_TOLERANCE_DAYS = 1
MAX_WINDOW_SPAN_DAYS = 45  # training windows whose 14 rows span more than this are dropped

MIN_FIT_ORIGINS = 40
MIN_STOP_ORIGINS = 10
MIN_VAL_ORIGINS = 10
MIN_DEPLOY_ORIGINS = 30
MIN_TEST_ORIGINS = 12
DEFAULT_EPOCHS_NO_CV = 30  # only used when no CV fold is usable (very short history)

# --- Baselines ---------------------------------------------------------------
TREND_LOOKBACK_DAYS = 30
TREND_MAX_OBSERVATION_GAP_DAYS = 90
TREND_SHRINK = 0.25
SEASONAL_LAGS_YEARS = (1, 2, 3)
SEASONAL_TOLERANCE_DAYS = 14

# --- Promotion gate (same for model / trend / seasonal) ------------------------
# A candidate replaces persistence only if ALL hold on pooled OOF origins:
#   * pooled MAE skill >= MIN_SKILL_PCT
#   * beats persistence in at least 2/3 of the folds
#   * 5th percentile of a moving-block bootstrap of the skill is > 0
MIN_SKILL_PCT = 2.0
FOLD_WIN_FRACTION = 2 / 3
N_BOOT = 1000
MIN_BOOT_BLOCKS = 8
SHRINK_GRID = (1.0, 0.75, 0.5, 0.25)
# Fewer than a full seasonal cycle of OOF data means a "model win" is mostly luck
# (short-history dams), so the learned model is not allowed to compete.
MIN_MODEL_OOF_DAYS = 365

# --- Optimisation -----------------------------------------------------------
EPOCHS = 400
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 30
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
# Gaussian noise on standardized inputs, training only (no backend change). Long-horizon
# heads were peaking at 1-3 epochs, i.e. memorising monsoon years. Try 0.1 if that persists.
INPUT_NOISE_STD = 0.05
SEED = int(os.environ.get("SEED", 42))  # e.g. SEED=7 python train/train_models.py

EVALUATION_FIELDS = (
    "dam_name", "horizon_days", "rmse_tmc", "mae_tmc", "nse", "r2",
    "persistence_rmse_tmc", "persistence_mae_tmc", "selected_strategy",
    "mae_skill_pct", "rmse_skill_pct", "n_test_origins",
    "validation_coverage_days", "validation_start", "validation_end",
    "test_coverage_days", "test_start", "test_end", "test_diagnostics",
)

IGNORED_COLUMNS = {
    "Date", "Monitoring Date", "Reservoir Name", "Basin", "Sub Basin",
    "River", "_id",
}


# =============================================================================
# Data prep (unchanged logic)
# =============================================================================
def _seed_for_dam(dam_id: str) -> int:
    """Stable per-dam seed without relying on Python's salted hash."""
    return SEED + sum((index + 1) * ord(char) for index, char in enumerate(dam_id))


def _numeric_feature_columns(frame: pd.DataFrame, warmup_rows: int) -> list[str]:
    """Numeric inputs the unmodified backend can scale without imputation.

    ForecastService calls StandardScaler on the latest 14 rows and does not
    fill NaNs, so only keep columns that are finite across the usable history.
    """
    selected: list[str] = []
    for column in frame.columns:
        if column in IGNORED_COLUMNS or column.startswith("target_") or column.startswith("_target_"):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        usable_values = values.iloc[warmup_rows:].to_numpy(dtype=float)
        if values.notna().sum() == 0 or not np.isfinite(usable_values).all():
            continue
        frame[column] = values
        selected.append(column)
    if "Current_Storage_TMC" not in selected:
        raise ValueError("required numeric input Current_Storage_TMC is missing or incomplete")
    if not selected or warmup_rows >= len(frame):
        raise ValueError("no complete numeric feature columns are available")
    return selected


def _match_future_observations(frame: pd.DataFrame, storage_column: str) -> pd.DataFrame:
    """Add future storage labels matched within one day after the calendar target."""
    observations = frame.loc[frame[storage_column].notna(), ["Date", storage_column]].copy()
    observations = observations.rename(columns={
        "Date": "_observed_date",
        storage_column: "_observed_storage",
    }).sort_values("_observed_date")

    labeled = frame.copy()
    origin_rows = np.arange(len(frame), dtype=int)
    tolerance = pd.Timedelta(days=MAX_LABEL_DATE_TOLERANCE_DAYS)
    for horizon in HORIZONS:
        query = pd.DataFrame({
            "_origin_row": origin_rows,
            "_wanted_date": frame["Date"] + pd.Timedelta(days=horizon),
        }).sort_values("_wanted_date")
        matched = pd.merge_asof(
            query,
            observations,
            left_on="_wanted_date",
            right_on="_observed_date",
            direction="forward",
            tolerance=tolerance,
        ).set_index("_origin_row").reindex(origin_rows)
        target_storage = matched["_observed_storage"].to_numpy(dtype=float)
        target_date = pd.to_datetime(matched["_observed_date"]).to_numpy()
        labeled[f"_target_storage_t_plus_{horizon}"] = target_storage
        labeled[f"_target_date_t_plus_{horizon}"] = target_date
        labeled[f"target_delta_t_plus_{horizon}"] = target_storage - labeled[storage_column].to_numpy(dtype=float)
    return labeled


def _prepare_dam(feature_path: str | Path, sequence_length: int) -> tuple[str, pd.DataFrame, list[str]]:
    dam_id = Path(feature_path).name.removesuffix("_features.csv")
    if dam_id not in DAM_METADATA:
        raise ValueError(f"{dam_id}: no matching reservoir entry in backend.models.DAM_METADATA")

    frame = pd.read_csv(feature_path)
    if "Date" not in frame.columns:
        raise ValueError("feature CSV must contain a Date column")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    if "Current_Storage_TMC" in frame:
        frame["Current_Storage_TMC"] = pd.to_numeric(frame["Current_Storage_TMC"], errors="coerce")
    if ("Current_Storage_TMC" not in frame or not frame["Current_Storage_TMC"].notna().any()) and "Reservoir Level (ft)" in frame:
        raise ValueError(
            "level-only feature data; training a TMC forecast needs measured storage values "
            "or a verified KRS level-to-storage curve"
        )
    finite_storage = np.isfinite(frame["Current_Storage_TMC"].to_numpy(dtype=float))
    frame = (
        frame.loc[frame["Date"].notna() & finite_storage]
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise ValueError("feature CSV has no dated storage observations")

    source_path = Path(IMD_DIR) / f"{dam_id}_dam_ready.csv"
    if source_path.exists():
        source_dates = pd.to_datetime(pd.read_csv(source_path, usecols=["Date"])["Date"], errors="coerce")
        source_latest = source_dates.max()
        if pd.isna(source_latest) or frame["Date"].iloc[-1] > source_latest:
            raise ValueError("feature data is newer than the cleaned source observations; rebuild features first")

    labeled = _match_future_observations(frame, "Current_Storage_TMC")
    # Lag/rolling features are empty at the start of a series; skip those
    # warm-up rows instead of filling them from the future.
    warmup_rows = min(max(30, sequence_length), max(0, len(labeled) - sequence_length))
    features = _numeric_feature_columns(labeled, warmup_rows)
    if len(labeled) < max(120, sequence_length + max(HORIZONS)):
        raise ValueError(f"only {len(labeled)} dated storage rows; at least 120 are required")
    return dam_id, labeled, features


# =============================================================================
# Origins / segments
# =============================================================================
def _label_complete(frame: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    """True where every horizon has a matched label dated on/before ``cutoff``.

    This is the purge step: a label that lands after ``cutoff`` would leak
    information from the next segment into this one.
    """
    ok = np.ones(len(frame), dtype=bool)
    cut = cutoff.to_datetime64()
    for horizon in HORIZONS:
        value = frame[f"_target_storage_t_plus_{horizon}"].to_numpy(dtype=float)
        date = frame[f"_target_date_t_plus_{horizon}"].to_numpy(dtype="datetime64[ns]")
        ok &= np.isfinite(value) & ~np.isnat(date) & (date <= cut)
    return ok


def _window_ok(frame: pd.DataFrame, sequence_length: int) -> np.ndarray:
    """False where the last ``sequence_length`` rows span an unusually long date range."""
    days = frame["Date"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    span = np.full(len(days), 10 ** 9, dtype=np.int64)
    span[sequence_length - 1:] = days[sequence_length - 1:] - days[:len(days) - sequence_length + 1]
    return span <= MAX_WINDOW_SPAN_DAYS


def _build_segments(frame: pd.DataFrame, sequence_length: int, warmup_rows: int) -> dict[str, Any]:
    """CV folds + deployment + test origins, all purged against their own cutoffs."""
    n = len(frame)
    dates = frame["Date"]
    rows = np.arange(n)
    first_row = warmup_rows + sequence_length - 1
    dev_end = int(n * DEV_FRACTION)
    if not 0 < first_row < dev_end < n:
        raise ValueError("not enough dated rows for development + test segments")
    window_ok = _window_ok(frame, sequence_length)

    def pick(lo: int, hi: int, cutoff: pd.Timestamp, train: bool = False) -> np.ndarray:
        mask = _label_complete(frame, cutoff) & (rows >= max(lo, first_row)) & (rows < hi)
        origins = np.flatnonzero(mask)
        if train:
            # Drop gap-spanning windows, but never throw away more than half the data.
            filtered = origins[window_ok[origins]]
            if len(filtered) >= 0.5 * len(origins):
                origins = filtered
        return origins

    dev_cutoff = dates.iloc[dev_end - 1]
    folds: list[dict[str, Any]] = []
    for k in range(len(FOLD_EDGES) - 1):
        fit_end = int(n * FOLD_EDGES[k])
        val_end = int(n * FOLD_EDGES[k + 1])
        stop_start = int(fit_end * (1.0 - EARLY_STOP_TAIL))
        fold = {
            "stop_start": stop_start,
            "fit_end": fit_end,
            "fit": pick(0, stop_start, dates.iloc[stop_start - 1], train=True),
            "stop": pick(stop_start, fit_end, dates.iloc[fit_end - 1]),
            # Val labels are purged at the end of the dev region, not the fold,
            # so test-period storage never scores a strategy.
            "val": pick(fit_end, val_end, dev_cutoff),
        }
        if (len(fold["fit"]) >= MIN_FIT_ORIGINS and len(fold["stop"]) >= MIN_STOP_ORIGINS
                and len(fold["val"]) >= MIN_VAL_ORIGINS):
            folds.append(fold)

    deployment = pick(0, dev_end, dev_cutoff, train=True)
    test = pick(dev_end, n, dates.iloc[-1])
    if len(deployment) < MIN_DEPLOY_ORIGINS:
        raise ValueError(f"deployment_train={len(deployment)} origins (<{MIN_DEPLOY_ORIGINS})")
    if len(test) < MIN_TEST_ORIGINS:
        raise ValueError(f"test={len(test)} origins (<{MIN_TEST_ORIGINS})")
    return {"folds": folds, "deployment": deployment, "test": test, "dev_end": dev_end}


def _coverage(frame: pd.DataFrame, origins: np.ndarray) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    start, end = frame.loc[origins[0], "Date"], frame.loc[origins[-1], "Date"]
    return start, end, int((end - start).days + 1)


# =============================================================================
# Features / targets / baselines
# =============================================================================
def _make_sequences(values: np.ndarray, origins: np.ndarray, sequence_length: int) -> np.ndarray:
    return np.stack([values[i - sequence_length + 1:i + 1] for i in origins]).astype(np.float32)


def _target_matrix(frame: pd.DataFrame, origins: np.ndarray, target_kind: str) -> np.ndarray:
    prefix = "_target_storage_t_plus_" if target_kind == "storage" else "target_delta_t_plus_"
    return frame.loc[origins, [f"{prefix}{h}" for h in HORIZONS]].to_numpy(dtype=float)


def _fit_preprocessing(raw: np.ndarray, columns: list[str], warmup_rows: int, end_row: int):
    """Median-impute + StandardScaler fitted ONLY on rows [warmup, end_row)."""
    sample = raw[warmup_rows:end_row]
    fills = np.array([
        float(np.median(col[np.isfinite(col)])) if np.isfinite(col).any() else 0.0
        for col in sample.T
    ])
    filled = np.where(np.isfinite(raw), raw, fills)
    scaler = StandardScaler().fit(pd.DataFrame(filled[warmup_rows:end_row], columns=columns))
    scaled = scaler.transform(pd.DataFrame(filled, columns=columns)).astype(np.float32)
    return fills, scaler, scaled


def _recent_trend_deltas(frame: pd.DataFrame, origins: np.ndarray) -> np.ndarray:
    """Mirror ForecastService's conservative 30-day storage trend baseline."""
    dates = frame["Date"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    storage = frame["Current_Storage_TMC"].to_numpy(dtype=float)
    previous = np.searchsorted(dates, dates[origins] - TREND_LOOKBACK_DAYS, side="right") - 1
    elapsed = np.zeros(len(origins), dtype=float)
    valid = previous >= 0
    elapsed[valid] = dates[origins[valid]] - dates[previous[valid]]
    valid &= (elapsed >= 14) & (elapsed <= TREND_MAX_OBSERVATION_GAP_DAYS)
    valid &= np.isfinite(storage[np.maximum(previous, 0)])
    slope = np.zeros(len(origins), dtype=float)
    slope[valid] = (storage[origins[valid]] - storage[previous[valid]]) / elapsed[valid]
    return np.column_stack([slope * horizon * TREND_SHRINK for horizon in HORIZONS])


def _seasonal_storage(frame: pd.DataFrame, origins: np.ndarray) -> np.ndarray:
    """Median storage near the same calendar date in prior years, past data only."""
    dates = pd.DatetimeIndex(frame["Date"])
    storage = frame["Current_Storage_TMC"].to_numpy(dtype=float)
    result = np.empty((len(origins), len(HORIZONS)), dtype=float)
    for row, origin in enumerate(origins):
        as_of = dates[origin]
        for col, horizon in enumerate(HORIZONS):
            target = as_of + pd.Timedelta(days=horizon)
            matches = []
            for years in SEASONAL_LAGS_YEARS:
                anchor = target - pd.DateOffset(years=years)
                left = dates.searchsorted(anchor - pd.Timedelta(days=SEASONAL_TOLERANCE_DAYS), side="left")
                right = dates.searchsorted(anchor + pd.Timedelta(days=SEASONAL_TOLERANCE_DAYS), side="right")
                candidates = np.arange(left, min(right, origin + 1), dtype=int)  # never look past the origin
                if len(candidates):
                    best = candidates[np.argmin(np.abs((dates[candidates] - anchor).days))]
                    if np.isfinite(storage[best]):
                        matches.append(storage[best])
            result[row, col] = float(np.median(matches)) if matches else storage[origin]
    return result


# =============================================================================
# Model fitting
# =============================================================================
def _predict_deltas(model: nn.Module, sequences: np.ndarray, scaler_y: StandardScaler,
                    device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        scaled = model(torch.as_tensor(sequences, dtype=torch.float32, device=device)).cpu().numpy()
    return scaler_y.inverse_transform(scaled)


def _predict_horizon_delta(head: nn.Module, sequences: np.ndarray, scaler_y: StandardScaler,
                           horizon_index: int, device: torch.device) -> np.ndarray:
    """One independently fitted horizon head, in TMC delta units."""
    head.eval()
    with torch.no_grad():
        scaled = head(torch.as_tensor(sequences, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
    return scaled * scaler_y.scale_[horizon_index] + scaler_y.mean_[horizon_index]


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _fit_model(
    x_fit: np.ndarray,
    y_fit_scaled: np.ndarray,
    scaler_y: StandardScaler,
    seed: int,
    device: torch.device,
    capacity: float,
    stop: dict[str, np.ndarray] | None = None,
    fixed_epochs: dict[str, int] | None = None,
) -> tuple[nn.Module, dict[str, int]]:
    """Fit all horizon heads. Early-stop on ``stop`` (MAE in TMC) or run ``fixed_epochs``.

    Constant LR in both modes on purpose: the epoch count found with early
    stopping is only meaningful for the refit if the schedule is identical.
    The best checkpoint (not the last epoch) is restored after early stopping.
    """
    torch.manual_seed(seed)
    model = IndependentHorizonLSTM(input_dim=x_fit.shape[-1]).to(device)
    loss_function = nn.L1Loss()  # matches the MAE metric; robust to monsoon-surge outliers
    epochs_used: dict[str, int] = {}

    for index, horizon in enumerate(HORIZONS):
        key = f"t+{horizon}"
        head = model.models[f"h{horizon}"]
        optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit_scaled[:, index:index + 1])),
            batch_size=min(BATCH_SIZE, len(x_fit)),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + index),
        )
        n_epochs = EPOCHS if stop is not None else fixed_epochs[key]
        best_score, best_epoch, best_state, stale = float("inf"), 0, None, 0

        for epoch in range(1, n_epochs + 1):
            head.train()
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_x = batch_x + INPUT_NOISE_STD * torch.randn_like(batch_x)
                optimizer.zero_grad(set_to_none=True)
                loss_function(head(batch_x), batch_y).backward()
                nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
                optimizer.step()

            if stop is None:
                continue
            delta = _predict_horizon_delta(head, stop["x"], scaler_y, index, device)
            pred = np.clip(stop["anchor"] + delta, 0.0, capacity)
            score = _mae(stop["actual"][:, index], pred)
            if score < best_score - 1e-5:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = copy.deepcopy(head.state_dict())
            else:
                stale += 1
                if stale >= EARLY_STOPPING_PATIENCE:
                    break

        if stop is not None:
            head.load_state_dict(best_state)
            epochs_used[key] = best_epoch
        else:
            epochs_used[key] = n_epochs
    return model, epochs_used


# =============================================================================
# Strategy selection on pooled out-of-fold predictions
# =============================================================================
def _skill_pct(err_base: np.ndarray, err_cand: np.ndarray) -> float | None:
    base = float(np.mean(err_base))
    if base <= 1e-12:
        return None
    return (1.0 - float(np.mean(err_cand)) / base) * 100.0


def _bootstrap_skill_lower(err_base: np.ndarray, err_cand: np.ndarray, block_len: int,
                           rng: np.random.Generator) -> float | None:
    """5th percentile of MAE skill under a moving-block bootstrap.

    Errors at neighbouring origins are heavily autocorrelated (overlapping
    horizons), so resample contiguous blocks, not individual origins.
    """
    ids = np.arange(len(err_base)) // block_len
    n_blocks = int(ids[-1]) + 1
    if n_blocks < MIN_BOOT_BLOCKS:
        return None
    base_sum = np.bincount(ids, weights=err_base, minlength=n_blocks)
    cand_sum = np.bincount(ids, weights=err_cand, minlength=n_blocks)
    draw = rng.integers(0, n_blocks, size=(N_BOOT, n_blocks))
    denom = base_sum[draw].sum(axis=1)
    keep = denom > 1e-12
    if not keep.any():
        return None
    skills = (1.0 - cand_sum[draw].sum(axis=1)[keep] / denom[keep]) * 100.0
    return float(np.percentile(skills, 5))


def _select_strategy(
    horizon: int,
    actual: np.ndarray,
    preds: dict[str, np.ndarray | None],
    fold_ids: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Pick the lowest-MAE candidate among persistence + those passing the gate."""
    err_base = np.abs(actual - preds["persistence"])
    block_len = max(30, 2 * horizon)
    fold_masks = [fold_ids == f for f in np.unique(fold_ids)]
    need_wins = int(np.ceil(len(fold_masks) * FOLD_WIN_FRACTION))

    scores = {"persistence": round(float(err_base.mean()), 5)}
    details: dict[str, Any] = {}
    qualified: list[str] = []
    for name in ("model", "recent_trend", "seasonal_naive"):
        if preds.get(name) is None:
            continue
        err = np.abs(actual - preds[name])
        pooled = _skill_pct(err_base, err)
        fold_skills = [_skill_pct(err_base[m], err[m]) for m in fold_masks]
        wins = sum(1 for s in fold_skills if s is not None and s > 0)
        lower = _bootstrap_skill_lower(err_base, err, block_len, rng)
        ok = (pooled is not None and pooled >= MIN_SKILL_PCT
              and wins >= need_wins and lower is not None and lower > 0)
        scores[name] = round(float(err.mean()), 5)
        details[name] = {
            "pooled_mae_skill_pct": round(pooled, 3) if pooled is not None else None,
            "fold_mae_skill_pct": [round(s, 3) if s is not None else None for s in fold_skills],
            "folds_won": wins,
            "bootstrap_p5_skill_pct": round(lower, 3) if lower is not None else None,
            "qualified": bool(ok),
        }
        if ok:
            qualified.append(name)

    pool = {name: scores[name] for name in ["persistence", *qualified]}
    return {
        "selected": min(pool, key=pool.get),
        "validation_mae_tmc": scores,
        "candidates": details,
        "folds_required_to_win": need_wins,
        "minimum_skill_required_pct": MIN_SKILL_PCT,
        # Kept so the backend / manifest readers see the same trend settings as before.
        "trend_lookback_days": TREND_LOOKBACK_DAYS,
        "trend_max_observation_gap_days": TREND_MAX_OBSERVATION_GAP_DAYS,
        "trend_shrink": TREND_SHRINK,
    }


# =============================================================================
# Metrics / reporting
# =============================================================================
def _storage_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, dict[str, float | None]]:
    metrics: dict[str, dict[str, float | None]] = {}
    for index, horizon in enumerate(HORIZONS):
        truth, estimate = actual[:, index], predicted[:, index]
        error = estimate - truth
        denominator = float(np.sum((truth - truth.mean()) ** 2))
        r2 = 1.0 - float(np.sum(error ** 2)) / denominator if denominator > 1e-12 else None
        metrics[f"t+{horizon}"] = {
            "r2": round(r2, 5) if r2 is not None else None,
            "mae_tmc": round(_mae(truth, estimate), 5),
            "rmse_tmc": round(float(np.sqrt(np.mean(error ** 2))), 5),
        }
    return metrics


def _skill_metrics(actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for index, horizon in enumerate(HORIZONS):
        model_error = predicted[:, index] - actual[:, index]
        baseline_error = baseline[:, index] - actual[:, index]
        model_mae, baseline_mae = float(np.mean(np.abs(model_error))), float(np.mean(np.abs(baseline_error)))
        model_rmse = float(np.sqrt(np.mean(model_error ** 2)))
        baseline_rmse = float(np.sqrt(np.mean(baseline_error ** 2)))
        result[f"t+{horizon}"] = {
            "model_mae_tmc": round(model_mae, 5),
            "persistence_mae_tmc": round(baseline_mae, 5),
            "mae_skill_pct": round((1.0 - model_mae / baseline_mae) * 100.0, 3) if baseline_mae > 1e-12 else None,
            "model_rmse_tmc": round(model_rmse, 5),
            "persistence_rmse_tmc": round(baseline_rmse, 5),
            "rmse_skill_pct": round((1.0 - model_rmse / baseline_rmse) * 100.0, 3) if baseline_rmse > 1e-12 else None,
        }
    return result


def write_evaluation_csv(summary: dict[str, Any]) -> Path:
    output_path = ROOT / "evaluation" / "evaluation_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=EVALUATION_FIELDS)
        writer.writeheader()
        for result in summary.get("results", []):
            for horizon in HORIZONS:
                key = f"t+{horizon}"
                model = result["test_storage_metrics"][key]
                persistence = result["persistence_baseline"][key]
                skill = result["test_skill_vs_persistence"][key]
                writer.writerow({
                    "dam_name": result["dam_name"],
                    "horizon_days": key,
                    "rmse_tmc": model["rmse_tmc"],
                    "mae_tmc": model["mae_tmc"],
                    "nse": model["r2"],
                    "r2": model["r2"],
                    "persistence_rmse_tmc": persistence["rmse_tmc"],
                    "persistence_mae_tmc": persistence["mae_tmc"],
                    "selected_strategy": result["horizon_strategy"][key]["selected"],
                    "mae_skill_pct": skill["mae_skill_pct"],
                    "rmse_skill_pct": skill["rmse_skill_pct"],
                    "n_test_origins": result["test_origins"],
                    "validation_coverage_days": result["validation_coverage_days"],
                    "validation_start": result["validation_start"],
                    "validation_end": result["validation_end"],
                    "test_coverage_days": result["test_coverage_days"],
                    "test_start": result["test_start"],
                    "test_end": result["test_end"],
                    "test_diagnostics": "; ".join(result.get("test_diagnostics", [])),
                })
    return output_path


# =============================================================================
# Per-dam pipeline
# =============================================================================
def train_dam_model(feature_path: str | Path, sequence_length: int = SEQUENCE_LENGTH) -> dict[str, Any]:
    feature_path = Path(feature_path)
    dam_id, frame, feature_columns = _prepare_dam(feature_path, sequence_length)
    seed = _seed_for_dam(dam_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    row_count = len(frame)
    warmup_rows = min(max(30, sequence_length), max(0, row_count - sequence_length))
    seg = _build_segments(frame, sequence_length, warmup_rows)
    folds, deploy_origins, test_origins = seg["folds"], seg["deployment"], seg["test"]
    dev_end = seg["dev_end"]

    capacity = float(DAM_METADATA[dam_id]["live_capacity_tmc"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = frame[feature_columns].to_numpy(dtype=float)
    storage = frame["Current_Storage_TMC"].to_numpy(dtype=float)

    # ---- 1) Rolling-origin CV: out-of-fold predictions for every fold --------
    oof_origins: list[np.ndarray] = []
    oof_delta: list[np.ndarray] = []
    oof_fold: list[np.ndarray] = []
    fold_epochs: list[dict[str, int]] = []
    fold_reports: list[dict[str, Any]] = []
    for fold_number, fold in enumerate(folds):
        # Scalers see only the fit rows; early-stop/val rows are transformed with them.
        _, scaler_x, scaled = _fit_preprocessing(raw, feature_columns, warmup_rows, fold["stop_start"])
        delta_fit = _target_matrix(frame, fold["fit"], "delta")
        scaler_y = StandardScaler().fit(delta_fit)
        stop = {
            "x": _make_sequences(scaled, fold["stop"], sequence_length),
            "anchor": storage[fold["stop"]],
            "actual": np.clip(_target_matrix(frame, fold["stop"], "storage"), 0.0, capacity),
        }
        model, epochs = _fit_model(
            _make_sequences(scaled, fold["fit"], sequence_length),
            scaler_y.transform(delta_fit).astype(np.float32),
            scaler_y, seed + 100 * (fold_number + 1), device, capacity, stop=stop,
        )
        val_x = _make_sequences(scaled, fold["val"], sequence_length)
        oof_origins.append(fold["val"])
        oof_delta.append(_predict_deltas(model, val_x, scaler_y, device))
        oof_fold.append(np.full(len(fold["val"]), fold_number))
        fold_epochs.append(epochs)
        start, end, days = _coverage(frame, fold["val"])
        fold_reports.append({
            "fit_origins": len(fold["fit"]), "early_stop_origins": len(fold["stop"]),
            "val_origins": len(fold["val"]), "val_start": start.strftime("%Y-%m-%d"),
            "val_end": end.strftime("%Y-%m-%d"), "val_coverage_days": days, "best_epochs": epochs,
        })

    # ---- 2) Pool OOF, fit per-horizon shrink, pick strategy per horizon --------
    horizon_strategy: dict[str, dict[str, Any]] = {}
    shrink = np.ones(len(HORIZONS))
    if folds:
        origins = np.concatenate(oof_origins)
        fold_ids = np.concatenate(oof_fold)
        model_delta = np.vstack(oof_delta)
        actual = np.clip(_target_matrix(frame, origins, "storage"), 0.0, capacity)
        anchor = storage[origins]
        persistence = np.clip(np.repeat(anchor[:, None], len(HORIZONS), axis=1), 0.0, capacity)
        trend = np.clip(anchor[:, None] + _recent_trend_deltas(frame, origins), 0.0, capacity)
        seasonal = np.clip(_seasonal_storage(frame, origins), 0.0, capacity)
        val_start, val_end, val_days = _coverage(frame, origins)
        for index, horizon in enumerate(HORIZONS):
            # Shrink toward persistence by whichever factor minimises OOF MAE.
            best_mae = float("inf")
            for alpha in SHRINK_GRID:
                mae = _mae(actual[:, index], np.clip(anchor + alpha * model_delta[:, index], 0.0, capacity))
                if mae < best_mae - 1e-9:
                    best_mae, shrink[index] = mae, alpha
            model_pred = np.clip(anchor + shrink[index] * model_delta[:, index], 0.0, capacity)
            horizon_strategy[f"t+{horizon}"] = _select_strategy(
                horizon, actual[:, index],
                {"persistence": persistence[:, index],
                 # None = not a candidate (_select_strategy skips it)
                 "model": model_pred if val_days >= MIN_MODEL_OOF_DAYS else None,
                 "recent_trend": trend[:, index], "seasonal_naive": seasonal[:, index]},
                fold_ids, rng,
            )
            horizon_strategy[f"t+{horizon}"]["model_shrink"] = float(shrink[index])
        val_origin_count = len(origins)
    else:
        # History too short for CV: no learned candidate, baselines are untested -> persistence.
        for horizon in HORIZONS:
            horizon_strategy[f"t+{horizon}"] = {
                "selected": "persistence", "note": "no usable CV fold; persistence retained",
                "model_shrink": 1.0, "trend_lookback_days": TREND_LOOKBACK_DAYS,
                "trend_max_observation_gap_days": TREND_MAX_OBSERVATION_GAP_DAYS,
                "trend_shrink": TREND_SHRINK,
            }
        val_start = val_end = frame["Date"].iloc[0]
        val_days, val_origin_count = 0, 0

    # ---- 3) Deployment refit on all dev rows with the CV epoch counts -------
    if fold_epochs:
        deploy_epochs = {
            f"t+{h}": max(1, int(np.median([e[f"t+{h}"] for e in fold_epochs]))) for h in HORIZONS
        }
    else:
        deploy_epochs = {f"t+{h}": DEFAULT_EPOCHS_NO_CV for h in HORIZONS}
    fill_values, scaler_x, scaled = _fit_preprocessing(raw, feature_columns, warmup_rows, dev_end)
    delta_deploy = _target_matrix(frame, deploy_origins, "delta")
    scaler_y = StandardScaler().fit(delta_deploy)
    model, _ = _fit_model(
        _make_sequences(scaled, deploy_origins, sequence_length),
        scaler_y.transform(delta_deploy).astype(np.float32),
        scaler_y, seed + 1000, device, capacity, fixed_epochs=deploy_epochs,
    )
    # Bake the shrink into scaler_y: inverse_transform(s) = s*scale + mean, so
    # scaling both by alpha returns alpha * delta. Backend needs no change.
    scaler_y.scale_ = scaler_y.scale_ * shrink
    scaler_y.mean_ = scaler_y.mean_ * shrink

    # ---- 4) Untouched test: report only -----------------------------------
    test_actual = np.clip(_target_matrix(frame, test_origins, "storage"), 0.0, capacity)
    test_anchor = storage[test_origins]
    test_persistence = np.clip(np.repeat(test_anchor[:, None], len(HORIZONS), axis=1), 0.0, capacity)
    test_model_delta = _predict_deltas(model, _make_sequences(scaled, test_origins, sequence_length), scaler_y, device)
    test_trend_delta = _recent_trend_deltas(frame, test_origins)
    test_seasonal = np.clip(_seasonal_storage(frame, test_origins), 0.0, capacity)
    candidate_preds = {
        "persistence": test_persistence,
        "model": np.clip(test_anchor[:, None] + test_model_delta, 0.0, capacity),
        "recent_trend": np.clip(test_anchor[:, None] + test_trend_delta, 0.0, capacity),
        "seasonal_naive": test_seasonal,
    }
    test_predicted = np.empty_like(test_persistence)
    for index, horizon in enumerate(HORIZONS):
        test_predicted[:, index] = candidate_preds[horizon_strategy[f"t+{horizon}"]["selected"]][:, index]
    test_storage_metrics = _storage_metrics(test_actual, test_predicted)
    persistence_metrics = _storage_metrics(test_actual, test_persistence)
    test_skill = _skill_metrics(test_actual, test_predicted, test_persistence)
    # Test MAE of every candidate, for diagnosing the gate. Never used to select.
    test_candidate_mae = {
        f"t+{h}": {name: round(_mae(test_actual[:, i], pred[:, i]), 5) for name, pred in candidate_preds.items()}
        for i, h in enumerate(HORIZONS)
    }

    test_start, test_end, test_coverage_days = _coverage(frame, test_origins)
    test_diagnostics: list[str] = []
    if test_coverage_days < 365:
        test_diagnostics.append("test coverage is under one year; R2 may be unstable for low-variation series")
    if not folds:
        test_diagnostics.append("history too short for CV; all horizons fell back to persistence")
    for horizon in HORIZONS:
        key = f"t+{horizon}"
        r2 = test_storage_metrics[key]["r2"]
        mae_skill, rmse_skill = test_skill[key]["mae_skill_pct"], test_skill[key]["rmse_skill_pct"]
        if r2 is not None and r2 < 0:
            test_diagnostics.append(f"{key}: negative test R2")
        if mae_skill is not None and rmse_skill is not None and mae_skill > 0 >= rmse_skill:
            test_diagnostics.append(f"{key}: MAE improves while RMSE does not; inspect large misses")

    report = {
        "dam_name": dam_id,
        "rows_labeled": int(frame[[f"_target_storage_t_plus_{h}" for h in HORIZONS]].notna().all(axis=1).sum()),
        "feature_count": len(feature_columns),
        "sequence_length": sequence_length,
        "best_epoch_by_horizon": deploy_epochs,
        "final_fit_epochs_by_horizon": deploy_epochs,
        "cv_folds": fold_reports,
        "deployment_train_origins": len(deploy_origins),
        "model_shrink_by_horizon": {f"t+{h}": float(shrink[i]) for i, h in enumerate(HORIZONS)},
        "test_origins": len(test_origins),
        "validation_origins": val_origin_count,
        "validation_start": val_start.strftime("%Y-%m-%d"),
        "validation_end": val_end.strftime("%Y-%m-%d"),
        "validation_coverage_days": val_days,
        "test_start": test_start.strftime("%Y-%m-%d"),
        "test_end": test_end.strftime("%Y-%m-%d"),
        "test_coverage_days": test_coverage_days,
        "test_diagnostics": test_diagnostics,
        "horizon_strategy": horizon_strategy,
        "model_architecture": "independent_horizon_lstm_v1",
        "seasonal_naive": {"lags_years": list(SEASONAL_LAGS_YEARS), "tolerance_days": SEASONAL_TOLERANCE_DAYS},
        "test_storage_metrics": test_storage_metrics,
        "persistence_baseline": persistence_metrics,
        "test_skill_vs_persistence": test_skill,
        "test_candidate_mae_tmc": test_candidate_mae,
    }

    # ---- 5) Artifacts (names/paths are the backend contract) -----------------
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{dam_id}_lstm.pth"
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, model_path)
    joblib.dump(scaler_x, models_dir / f"{dam_id}_scaler_x.pkl")
    joblib.dump(scaler_y, models_dir / f"{dam_id}_scaler_y.pkl")
    manifest = {
        **report,
        "feature_columns": feature_columns,
        "horizons_days": list(HORIZONS),
        "missing_feature_fill_values": dict(zip(feature_columns, fill_values.tolist())),
        "capacity_tmc": capacity,
        "model_file": model_path.name,
    }
    (models_dir / f"{dam_id}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"{dam_id}: {len(deploy_origins)} deploy-train, {len(folds)} CV folds "
        f"({val_origin_count} OOF origins), {len(test_origins)} test origins; epochs {deploy_epochs}"
    )
    for index, horizon in enumerate(HORIZONS):
        key = f"t+{horizon}"
        cands = test_candidate_mae[key]
        print(
            f"  {key}: {horizon_strategy[key]['selected']} (shrink {shrink[index]}); "
            f"test MAE {test_storage_metrics[key]['mae_tmc']:.3f} TMC vs persistence "
            f"{persistence_metrics[key]['mae_tmc']:.3f}; skill {test_skill[key]['mae_skill_pct']}% "
            f"| shrunk model {cands['model']:.3f}"
        )
    for diagnostic in test_diagnostics:
        print(f"  REVIEW: {diagnostic}")
    return report


# =============================================================================
# Driver
# =============================================================================
def _missing_feature_reason(dam_id: str) -> str:
    source_path = Path(IMD_DIR) / f"{dam_id}_dam_ready.csv"
    if not source_path.exists():
        return "missing feature file; build or restore this reservoir's feature data"
    storage_columns = ("Current_Storage_TMC", "Live Capacity (TMC)", "Percentage Full", "Gross Capacity (TMC)")
    header = pd.read_csv(source_path, nrows=0).columns
    available = [c for c in storage_columns if c in header]
    populated = []
    if available:
        values = pd.read_csv(source_path, usecols=available)
        populated = [c for c in available if pd.to_numeric(values[c], errors="coerce").notna().any()]
    if not populated:
        return (f"no feature file; {source_path.name} has no populated storage field "
                "(a reservoir level-to-storage curve or storage observations are required)")
    return "missing feature file; build or restore this reservoir's feature data"


def main() -> None:
    feature_files = sorted(glob.glob(os.path.join(FEATURES_DIR, "*_features.csv")))
    if not feature_files:
        raise SystemExit(f"No feature CSVs found under {FEATURES_DIR}; run build_features.py first.")

    feature_by_dam = {Path(p).name.removesuffix("_features.csv"): p for p in feature_files}
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for dam_id in sorted(DAM_METADATA):
        feature_file = feature_by_dam.get(dam_id)
        if feature_file is None:
            reason = _missing_feature_reason(dam_id)
            failures.append({"file": f"{dam_id}_features.csv", "error": reason})
            print(f"SKIP {dam_id}: {reason}")
            continue
        try:
            results.append(train_dam_model(feature_file))
        except Exception as error:
            failures.append({"file": os.path.basename(feature_file), "error": str(error)})
            print(f"SKIP {os.path.basename(feature_file)}: {error}")

    for dam_id, feature_file in feature_by_dam.items():
        if dam_id not in DAM_METADATA:
            failures.append({
                "file": os.path.basename(feature_file),
                "error": "feature file has no matching reservoir in backend.models.DAM_METADATA",
            })

    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "run_status": "complete" if not failures else "partial",
        "selection_metric": (
            "rolling-origin CV (3 expanding folds over the first 85% of rows) -> pooled out-of-fold "
            "predictions; per horizon compare persistence, shrunk LSTM, recent trend, seasonal naive; "
            f"a candidate replaces persistence only with >={MIN_SKILL_PCT}% pooled MAE skill, wins in "
            "at least 2/3 of folds, and a block-bootstrap 5th-percentile skill > 0. "
            "The final 15% is reporting-only."
        ),
        "label_policy": (
            f"Calendar-day targets use the first observed storage on or after t+h, "
            f"within {MAX_LABEL_DATE_TOLERANCE_DAYS} day(s); origins without a match are omitted"
        ),
        "test_policy": "final chronological 15% holdout; reporting only, never model or strategy selection",
        "expected_dams": sorted(DAM_METADATA),
        "trained_dams": sorted(r["dam_name"] for r in results),
        "results": results,
        "failures": failures,
    }
    summary_path = models_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = write_evaluation_csv(summary)
    print(f"Wrote benchmark results to {csv_path}")
    if not results:
        raise SystemExit("No reservoir model was trained successfully; see the errors above.")
    if failures:
        raise SystemExit(
            f"Partial training run: trained {len(results)} of {len(DAM_METADATA)} reservoirs; "
            f"{len(failures)} missing or failed. See {summary_path} before using the CSV."
        )


if __name__ == "__main__":
    main()