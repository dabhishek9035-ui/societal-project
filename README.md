# Jaladrishti

Jaladrishti is a reservoir monitoring and short-range forecasting platform for Karnataka. It combines daily reservoir observations, rainfall data, a FastAPI service, per-horizon PyTorch models, and a Next.js dashboard.

The platform shows 1-, 7-, 14-, and 30-day storage forecasts when a trained artifact is available. For each horizon, the training pipeline compares its own LSTM with persistence, a recent-trend forecast, and a seasonal-naive forecast. Persistence remains the selected forecast whenever the other candidates do not meet the validation rules. The forecasts are for situational awareness; they are not instructions for reservoir releases or dam operations.

## What the application includes

- **Home (`/`)** — statewide introduction and links into the reservoir and forecast views.
- **Reservoirs (`/dams`)** — searchable reservoir register with current readings and status.
- **Predictions (`/predictions`)** — reservoir history, horizon selection, date navigation, forecast comparison, and scenario exploration.
- **FastAPI backend** — reservoir summaries, time series, trained forecasts, scenario simulations, and evaluation metrics.
- **Training and evaluation pipeline** — chronological splits, per-horizon candidate selection, persisted manifests, and a CSV holdout report.

The dashboard can use generated demo records when `NEXT_PUBLIC_USE_MOCK=true`. Demo data is illustrative and is not live telemetry. Backdated predictions in the interface are labeled synthetic estimates; the backend model is only used for the latest available observation.

## Technology

- Next.js 14, React, TypeScript, Tailwind CSS
- Three.js / React Three Fiber for the shared visual scene
- TanStack Query, Zustand, Zod, Recharts
- FastAPI, Pandas, scikit-learn, PyTorch

## Requirements

- Python 3.10 or newer
- Node.js 18.17 or newer and npm
- Network access only when fetching or updating upstream data

There is no checked-in Python requirements file. Install the runtime packages in a virtual environment as shown below. PyTorch CPU is sufficient; CUDA is used by training when available.

## Local development

Run the backend and frontend in separate terminals from the repository root.

### 1. Set up Python

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install fastapi uvicorn pandas numpy scikit-learn joblib torch requests
```

If PowerShell blocks environment activation, run the virtual environment's Python directly with `\.venv\Scripts\python.exe`.

### 2. Set up the frontend

```powershell
npm install
Copy-Item .env.example .env.local
npm run dev
```

Open <http://localhost:3000>.

For a production frontend build, run `npm run build` and then `npm start`. `npm run lint` runs the Next.js lint command. Keep the FastAPI process available separately when using real reservoir data.

### 3. Start the backend

In a second terminal:

```powershell
.\.venv\Scripts\python.exe start_platform.py
```

The API is available at <http://127.0.0.1:8000>. Health check: <http://127.0.0.1:8000/api/health>. Interactive API documentation: <http://127.0.0.1:8000/docs>.

The project also has a local `venv` directory in some development environments. The commands above create and use `.venv`; either environment is fine as long as it has the required packages installed.

## Configuration

Copy `.env.example` to `.env.local` and configure these frontend variables:

| Variable | Default | Description |
| --- | --- | --- |
| `NEXT_PUBLIC_API_BASE_URL` | `http://127.0.0.1:8000` | FastAPI base URL used by the browser. |
| `NEXT_PUBLIC_USE_MOCK` | `false` | Set to `true` to use generated demo records instead of the backend. |

Restart the Next.js process after changing environment variables. `src/config.ts` also contains the displayed repository link and project copy.

## API routes

The FastAPI app in `backend/main.py` currently exposes:

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Service health and available reservoir count. |
| `GET` | `/api/summary` | Aggregate storage, capacity, inflow/outflow, and basin totals. |
| `GET` | `/api/reservoirs` | Reservoir list; optional `basin` filter. |
| `GET` | `/api/reservoirs/{dam_id}?history_days=90` | Reservoir metadata and recent time series. `history_days` is 7–365. |
| `GET` | `/api/reservoirs/{dam_id}/forecast` | Latest-date 1/7/14/30-day forecasts and selected strategies. |
| `POST` | `/api/reservoirs/{dam_id}/simulate` | Scenario trajectory using `rain_factor` and `outflow_factor`. |
| `GET` | `/api/evaluation` | Saved evaluation metrics. |

The frontend adapter in `src/data/api.ts` normalizes backend responses for the UI and validates response shapes. The browser history adapter exposes at most the most recent year. The backend forecast route does not accept a historical `as_of` date; historical UI estimates therefore must not be interpreted as model backtests.

## Data files

### Reservoir observations

Place cleaned daily reservoir records under `data/imd_data/`, normally named `<dam_id>_dam_ready.csv`. Each file should contain a `Date` column and reservoir observations. The feature builder can derive TMC storage from populated storage/capacity fields or percentage-full observations. It also uses available level, inflow, outflow, and other numeric telemetry fields.

`fetch_daily_data.py` updates local reservoir CSVs from the configured NWDP datastore API. It performs reservoir-name filtering because substring searches can return records for similarly named dams. Review source data before replacing local files; fetching requires network access.

### Rainfall

`fetch_rainfall.py` downloads a daily rainfall series for each of the ten configured dam coordinates into:

```text
data/rainfall/<dam_id>_dam_ready.csv
```

The expected columns are `date` and `rainfall_mm`, with optional dam and coordinate metadata. The current source is Open-Meteo's historical archive at point locations near each dam. It is **point rainfall, not a catchment-average rainfall measurement**. `build_features.py` joins it to reservoir records on normalized calendar dates, adds a rainfall availability flag, and derives rolling rainfall sums, means, and observation coverage. Missing rainfall is not forward-filled as if it were observed rain.

The raw and generated data directories are ignored by Git. Keep backups of any locally maintained input CSVs.

## Build features and train

From the repository root, run:

```powershell
# Optional: refresh the reservoir observations from NWDP
python fetch_daily_data.py

# Optional: refresh the daily point rainfall files
python fetch_rainfall.py

# Join the source observations and construct dated features
python build_features.py

# Train each reservoir model and write the holdout report
python train/train_models.py
```

The feature builder writes `data/features/<dam_id>_features.csv`. Features include calendar terms, storage and operations lags, rolling inflow and rainfall features when available, rainfall coverage indicators, and forward targets. Targets are created for 1, 7, 14, and 30 days. The trainer matches each target to the first observed storage on or after its calendar target date, allowing at most one day of delay; unmatched targets are omitted. Inputs and splits are chronological, and labels crossing a split boundary are purged.

### Model selection

The training split is chronological:

| Portion | Use |
| --- | --- |
| First 55% | Preliminary model fitting. |
| 55–65% | Per-horizon early stopping. |
| 65–85% | Strategy comparison across three chronological validation windows. |
| Final 15% | Untouched reporting-only test set. |

Each horizon is fitted as an independent LSTM with its own early-stopping epoch count. Candidates are persistence (no storage change), a discounted recent trend, and a seasonal-naive forecast based on prior-year storage near the target calendar date. A learned or seasonal strategy must beat persistence by at least 5% validation MAE overall and in every validation window covering at least one year. Recent trend requires at least 5% skill overall and in each validation window covering at least 180 days. The qualified candidate with the lowest validation MAE is selected; persistence remains the fallback and is retained whenever no qualified candidate wins. The final test set is never used for model or strategy selection.

The trainer writes model checkpoints, scalers, and per-dam manifests under `train/models/`, plus:

- `train/models/training_summary.json` — run status, validation details, selected strategies, test metrics, and failures.
- `evaluation/evaluation_results.csv` — four test rows per successfully trained dam.

Model weights and scalers are ignored by Git. A fresh checkout must train models before the backend can serve trained forecasts. A run with missing or failed reservoirs is marked `partial`, writes its successful metrics, and exits non-zero. Check `training_summary.json` before treating a partial CSV as complete.

KRSagara currently has a dated reservoir-level series but no populated storage/TMC field. The feature builder can create level-only features for it, but the TMC trainer correctly skips it until measured storage data or a verified level-to-storage curve is available. Rainfall alone cannot provide the storage target.

### Republish an existing evaluation report

To recreate the CSV from the saved training summary without retraining:

```powershell
python evaluate.py
```

This republishes the stored chronological holdout metrics; it does not recalculate splits or scores.

## Latest forecast from the command line

After model artifacts exist:

```powershell
python prediction/predict.py --dam almatti
```

Available reservoir IDs are defined in `backend/models.py` (for example `almatti`, `bhadra`, `kabini`, and `tungabhadra`). The command uses the same `ForecastService` and saved strategy as the API. It forecasts from the latest observation; it does not perform a historical backtest.

## Evaluation and interpretation

`evaluation/evaluation_results.csv` reports holdout RMSE, MAE, NSE, R², persistence MAE/RMSE, selected strategy, skill versus persistence, origin counts, date coverage, and diagnostics for each horizon. Positive MAE skill means the selected forecast beat persistence on that test window; negative skill means it did worse. Small test coverage and low variation can make R² unstable. Always inspect the validation policy and `test_diagnostics` alongside headline scores.

The user interface displays reservoir levels in feet, while the trained model predicts storage in TMC. The adapter estimates level changes from storage changes using the observed level and storage as an anchor. Without reservoir-specific elevation/storage curves, this conversion is approximate and is not suitable for operational decisions. The UI's normalized dead-pool boundary is also an estimate where authoritative MDDL data is unavailable.

Scenario simulation applies user-selected rainfall and outflow multipliers to the baseline trajectory. It is a simplified what-if visualization, not a hydrological routing model or release recommendation.

## Troubleshooting

- **No reservoirs appear:** check that `start_platform.py` is running and visit `/api/health`; confirm `.env.local` points to the right API URL.
- **Forecast endpoint returns a missing-model error:** run feature building and training; confirm the dam's `.pth`, scaler, and manifest files exist under `train/models/`.
- **Training reports a partial run:** inspect `failures` in `train/models/training_summary.json`. Missing storage observations, inadequate chronological coverage, or incomplete source features can prevent training.
- **KRSagara has features but no TMC forecast:** the local source contains level but no storage series. Obtain a verified elevation/storage curve or observed storage values before converting level to a TMC target.
- **Rainfall features are absent:** check that `data/rainfall/<dam_id>_dam_ready.csv` exists with `date` and `rainfall_mm`, then rerun `python build_features.py` before training.
- **Frontend shows demo values:** set `NEXT_PUBLIC_USE_MOCK=false` and restart Next.js.

## Repository layout

```text
backend/
  main.py                      FastAPI routes
  models.py                    Reservoir metadata and model architectures
  services/                    Reservoir data and forecast services
data/
  imd_data/                    Source reservoir observations (local input)
  rainfall/                    Daily dam-coordinate rainfall (local input)
  features/                    Generated model features (local output)
evaluation/
  evaluation_results.csv       Chronological holdout report
prediction/
  predict.py                   Latest forecast CLI
src/
  app/                         Next.js routes and global providers/styles
  components/                  Shared navigation, UI, and scene
  data/                        API adapter, schemas, and demo data
  state/                       Client-side app and scene state
train/
  train_models.py              Chronological training and strategy selection
  models/                      Generated checkpoints, scalers, and manifests
build_features.py              Merge source data and create model features
fetch_daily_data.py            Update reservoir source records
fetch_rainfall.py              Download daily rainfall by dam coordinates
evaluate.py                    Republish saved holdout metrics
start_platform.py              Start the FastAPI backend
```
