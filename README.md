# Jaladrishti — Karnataka Reservoir Intelligence

Jaladrishti is an immersive, three-page frontend for exploring reservoir levels and short-range water-level outlooks. The existing repository is a Next.js 14 project, so the implementation uses its App Router for `/`, `/dams`, and `/predictions`; the React, Three.js, chart, and Tailwind stack in the project brief is retained where compatible.

## Run locally

Start the FastAPI service and Next.js app in separate terminals from the project root:

```powershell
# Terminal 1: backend
.\venv\Scripts\python.exe start_platform.py
```

```powershell
# Terminal 2: frontend
npm install
Copy-Item .env.example .env.local
npm run dev
```

Open `http://localhost:3000`. The frontend uses the backend by default; check `http://127.0.0.1:8000/api/health` if reservoir data does not load. Set `NEXT_PUBLIC_USE_MOCK=true` in `.env.local` only when you want standalone demo data. Restart Next.js after changing environment variables.

## Data and configuration

`src/data/types.ts` defines the normalized dam, observation, and forecast models. `src/data/api.ts` is the adapter boundary. `src/data/mock.ts` supplies ten Karnataka reservoirs and deterministic seasonal daily series for roughly four years, including uncertainty ranges and backtest actuals. Mock values are illustrative and should not be treated as live telemetry.

Set these variables in `.env.local`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `NEXT_PUBLIC_USE_MOCK` | `false` | `true` selects generated demo data. |
| `NEXT_PUBLIC_API_BASE_URL` | `http://127.0.0.1:8000` | Backend origin. |

The frontend first tries the contract in the brief:

- `GET /api/dams`
- `GET /api/dams/:id/levels?from=YYYY-MM-DD&to=YYYY-MM-DD`
- `GET /api/predict?dam_id=:id&as_of=YYYY-MM-DD&horizon=1|7|14|30`

It also adapts this repository’s current FastAPI routes:

- `GET /api/reservoirs`
- `GET /api/reservoirs/:id?history_days=120..365`
- `GET /api/reservoirs/:id/forecast`

The current backend forecast endpoint only evaluates its latest observation and does not accept an `as_of` date or return per-horizon persistence scores. For a backdated selection, the adapter builds an in-browser seasonal/trend estimate from the loaded history and requests actual observations for comparison; the persistence baseline is the last observed level. If an API response provides a `persistence` object, that value takes precedence. Latest-date forecasts return storage in TMC, so the adapter converts storage changes to level changes using the observed level and storage as the anchor; an unchanged-storage forecast therefore does not invent a sudden level drop. This linear change estimate is only approximate; a reservoir-specific elevation/storage curve is needed for operational level estimates. Add the canonical endpoints and elevation predictions to the service to use the trained model for historical backtests. Response objects are checked with Zod before entering the UI.

`src/config.ts` holds the project copy and GitHub link. Change `GITHUB_URL` there to point to the canonical repository.

## Retrain and evaluate

From the project root, run `python build_features.py` after updating the source reservoir CSVs, then `python train/train_models.py`. Training uses chronological 60/25/15 train/validation/test segments and selects each forecast horizon using validation MAE only; the final test segment is reporting-only. An LSTM head is promoted only when it improves on persistence by at least 5% overall and in each of two full-year validation blocks. If it does not, a fixed trend baseline projects 25% of the preceding 30-day storage slope (using the nearest observation 14–90 days back when readings are sparse); it must improve on persistence by at least 0.5% overall and in each of two validation blocks covering at least 60 days. Otherwise, the horizon uses persistence. Training writes per-dam manifests, a canonical `train/models/training_summary.json`, and the API-compatible `evaluation/evaluation_results.csv`. Run `python evaluate.py` to republish the CSV from the saved training summary without recomputing different splits or metrics.

## Interaction notes

- The prediction URL stores `dam`, `horizon`, `window`, and `asOf`; the page restores them on load. Arrow keys step dates unless focus is in an input.
- The active range fraction shown in the tank is `(level − minLevel) / (maxLevel − minLevel)`. `minLevel` represents the chosen minimum/dead-pool boundary. The available backend metadata does not currently provide MDDL/dead-pool elevations, so the normalized adapter estimates this boundary at 68% of FRL; replace that mapping when authoritative values are available. The mock records have per-dam minimum values in their data layer.
- History controls show 120–365 days; date travel allows up to four years of generated mock observations. The existing backend adapter exposes at most the most recent year, so its date picker is limited to that returned range.
- Session MAE, RMSE, and beats-persistence rate accumulate only on dates visited with the step or play controls. Reset clears the session.
- The canvas is mounted in the root layout and shared by all routes. High quality adds bloom and a light vignette; the FPS probe selects high, medium, or low particle counts, and `Alt + Shift + Q` opens a hidden scene control to cycle tiers. Rendering pauses while the document is hidden. DPR is capped at 1.75.
- Reduced-motion mode uses slower shader motion, removes scroll-driven bubble travel, and disables decorative CSS movement.

## Structure

```text
src/
  app/                 App Router pages, global styles, query provider
  components/          Persistent scene, shell, and shared UI
  data/                Typed models, API/mock adapters, reservoir data
  state/               Zustand app and scene state
```

The frontend keeps data and content in the adapter/config layer rather than embedding reservoir metadata in page components. `start_platform.py` runs the FastAPI backend on port 8000; Next.js runs separately on port 3000.
