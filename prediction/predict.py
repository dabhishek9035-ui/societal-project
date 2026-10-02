"""Run the same manifest-aware latest forecast used by the backend API.

Run from the repository root with ``python prediction/predict.py --dam almatti``.
The backend service applies each reservoir's selected persistence, trend, or
Delta-LSTM strategy so this CLI cannot silently disagree with evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.forecast_service import ForecastService


def predict_reservoir_storage(dam_name: str, sequence_length: int = 14) -> dict[str, Any]:
    """Forecast storage at the latest observed date using the saved strategy."""
    result = ForecastService().predict_forecast(dam_name, sequence_length=sequence_length)
    strategies = result.get("strategies", {})
    return {
        "dam_name": dam_name,
        "as_of_date": result["as_of_date"],
        "latest_observed_storage_tmc": result["current_storage_tmc"],
        "gross_capacity_tmc": result["live_capacity_tmc"],
        "forecasts": [
            {
                "horizon_days": point["horizon_days"],
                "target_date": point["target_date"],
                "strategy": strategies.get(str(point["horizon_days"]), "persistence"),
                "predicted_delta_tmc": point["predicted_delta_tmc"],
                "predicted_storage_tmc": point["predicted_storage_tmc"],
                "percentage_full": point["percentage_full"],
            }
            for point in result["forecast_horizons"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dam", default="almatti", help="reservoir id (default: almatti)")
    parser.add_argument("--sequence-length", type=int, default=14)
    args = parser.parse_args()
    print(json.dumps(
        predict_reservoir_storage(args.dam, sequence_length=args.sequence_length),
        indent=2,
    ))


if __name__ == "__main__":
    main()
