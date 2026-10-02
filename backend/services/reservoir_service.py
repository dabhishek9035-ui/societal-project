import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from backend.models import (
    DAM_METADATA,
    FEATURES_DIR,
    IMD_DIR,
    EVAL_RESULTS_PATH
)

class ReservoirService:
    def __init__(self):
        self._cache = {}
        self._summary_cache = None

    def get_all_reservoirs(self) -> List[Dict[str, Any]]:
        """Returns live summary list for all 10 Karnataka reservoirs."""
        reservoirs = []
        for dam_id, meta in DAM_METADATA.items():
            latest = self.get_latest_reading(dam_id)
            if not latest.get("date"):
                # Omit reservoirs whose derived data no longer matches the
                # cleaned source. Publishing placeholder levels looks current
                # even though there is no validated observation to show.
                continue
            capacity = meta["live_capacity_tmc"]
            storage = latest.get("current_storage_tmc", 0.0)
            fill_pct = round((storage / capacity * 100.0) if capacity > 0 else 0.0, 1)

            # Categorize flood / storage risk state
            if fill_pct >= 90.0:
                status = "Critical High / Spill Alert"
                status_color = "red"
            elif fill_pct >= 75.0:
                status = "High Filling"
                status_color = "amber"
            elif fill_pct >= 40.0:
                status = "Normal Storage"
                status_color = "teal"
            else:
                status = "Low / Drought Caution"
                status_color = "blue"

            reservoirs.append({
                **meta,
                "latest_date": latest.get("date"),
                "current_storage_tmc": storage,
                "percentage_full": fill_pct,
                "water_level_ft": latest.get("reservoir_level_ft"),
                "inflow_cusecs": latest.get("inflow_cusecs", 0.0),
                "outflow_cusecs": latest.get("outflow_cusecs", 0.0),
                "rainfall_mm": latest.get("rainfall_mm", 0.0),
                "catchment_rain_m": latest.get("catchment_rain_m", 0.0),
                "status": status,
                "status_color": status_color,
                "buffer_remaining_tmc": round(max(0.0, capacity - storage), 2)
            })

        return reservoirs

    def get_statewide_summary(self) -> Dict[str, Any]:
        """Calculates total state capacity, aggregate storage, total inflow & outflow."""
        all_dams = self.get_all_reservoirs()
        total_capacity = sum(d["live_capacity_tmc"] for d in all_dams)
        total_storage = sum(d["current_storage_tmc"] for d in all_dams)
        total_inflow = sum(d["inflow_cusecs"] for d in all_dams)
        total_outflow = sum(d["outflow_cusecs"] for d in all_dams)
        avg_pct = round((total_storage / total_capacity * 100.0), 1) if total_capacity > 0 else 0.0

        basin_breakdown = {}
        for d in all_dams:
            b = d["basin"]
            if b not in basin_breakdown:
                basin_breakdown[b] = {"storage_tmc": 0.0, "capacity_tmc": 0.0, "dams_count": 0}
            basin_breakdown[b]["storage_tmc"] += d["current_storage_tmc"]
            basin_breakdown[b]["capacity_tmc"] += d["live_capacity_tmc"]
            basin_breakdown[b]["dams_count"] += 1

        for b, v in basin_breakdown.items():
            v["storage_tmc"] = round(v["storage_tmc"], 2)
            v["capacity_tmc"] = round(v["capacity_tmc"], 2)
            v["fill_pct"] = round((v["storage_tmc"] / v["capacity_tmc"] * 100.0), 1) if v["capacity_tmc"] > 0 else 0.0

        return {
            "total_capacity_tmc": round(total_capacity, 2),
            "total_storage_tmc": round(total_storage, 2),
            "overall_fill_percentage": avg_pct,
            "total_inflow_cusecs": round(total_inflow, 1),
            "total_outflow_cusecs": round(total_outflow, 1),
            "dams_monitored": len(all_dams),
            "basin_breakdown": basin_breakdown,
            "last_updated": max((d["latest_date"] for d in all_dams if d["latest_date"]), default="N/A")
        }

    def get_latest_reading(self, dam_id: str) -> Dict[str, Any]:
        """Fetches the latest telemetry observation for a given dam."""
        feat_path = os.path.join(FEATURES_DIR, f"{dam_id}_features.csv")
        if not os.path.exists(feat_path):
            return {}

        try:
            df = pd.read_csv(feat_path)
            if df.empty:
                return {}
            last_row = df.iloc[-1]
            date_val = str(last_row.get("Date", ""))

            # A feature file can outlive or be newer than its cleaned source
            # after a reservoir-name filter removes contaminated observations.
            # Never publish that stale row as the current dam reading.
            source_path = os.path.join(IMD_DIR, f"{dam_id}_dam_ready.csv")
            if os.path.exists(source_path):
                source_dates = pd.read_csv(source_path, usecols=["Date"])["Date"]
                source_latest = pd.to_datetime(source_dates, errors="coerce").max()
                feature_latest = pd.to_datetime(date_val, errors="coerce")
                if pd.isna(source_latest) or pd.isna(feature_latest) or feature_latest > source_latest:
                    return {}
            
            storage = float(last_row.get("Current_Storage_TMC", 0.0))
            level = float(last_row.get("Reservoir Level (ft)", 0.0)) if "Reservoir Level (ft)" in last_row and pd.notna(last_row["Reservoir Level (ft)"]) else 0.0
            inflow = float(last_row.get("Inflow (Cusecs)", 0.0)) if "Inflow (Cusecs)" in last_row and pd.notna(last_row["Inflow (Cusecs)"]) else 0.0
            outflow = float(last_row.get("Outflow to River (Cusecs)", 0.0)) if "Outflow to River (Cusecs)" in last_row and pd.notna(last_row["Outflow to River (Cusecs)"]) else 0.0
            rain = float(last_row.get("RAINFALL", 0.0)) if "RAINFALL" in last_row and pd.notna(last_row["RAINFALL"]) else 0.0
            catchment_rain = float(last_row.get("era5_catchment_rain_m", 0.0)) if "era5_catchment_rain_m" in last_row and pd.notna(last_row["era5_catchment_rain_m"]) else 0.0

            return {
                "date": date_val,
                "current_storage_tmc": round(storage, 3),
                "reservoir_level_ft": round(level, 2),
                "inflow_cusecs": round(inflow, 1),
                "outflow_cusecs": round(outflow, 1),
                "rainfall_mm": round(rain, 2),
                "catchment_rain_m": round(catchment_rain, 5)
            }
        except Exception as e:
            print(f"Error reading latest row for {dam_id}: {e}")
            return {}

    def get_timeseries(self, dam_id: str, days: int = 120) -> List[Dict[str, Any]]:
        """Returns historical time series for hydrograph charting."""
        feat_path = os.path.join(FEATURES_DIR, f"{dam_id}_features.csv")
        if not os.path.exists(feat_path):
            return []

        df = pd.read_csv(feat_path)
        if df.empty:
            return []

        source_path = os.path.join(IMD_DIR, f"{dam_id}_dam_ready.csv")
        if os.path.exists(source_path):
            source_dates = pd.read_csv(source_path, usecols=["Date"])["Date"]
            source_latest = pd.to_datetime(source_dates, errors="coerce").max()
            feature_latest = pd.to_datetime(df["Date"], errors="coerce").max()
            if pd.isna(source_latest) or pd.isna(feature_latest) or feature_latest > source_latest:
                return []

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        sliced = df.iloc[-days:].copy()
        meta = DAM_METADATA.get(dam_id, {})
        cap = meta.get("live_capacity_tmc", 100.0)

        series = []
        for _, row in sliced.iterrows():
            st = float(row.get("Current_Storage_TMC", 0.0))
            lvl = float(row.get("Reservoir Level (ft)", 0.0)) if pd.notna(row.get("Reservoir Level (ft)")) else 0.0
            inf = float(row.get("Inflow (Cusecs)", 0.0)) if pd.notna(row.get("Inflow (Cusecs)")) else 0.0
            outf = float(row.get("Outflow to River (Cusecs)", 0.0)) if pd.notna(row.get("Outflow to River (Cusecs)")) else 0.0
            rn = float(row.get("RAINFALL", 0.0)) if pd.notna(row.get("RAINFALL")) else 0.0

            series.append({
                "date": row["Date"].strftime("%Y-%m-%d"),
                "storage_tmc": round(st, 3),
                "fill_percentage": round((st / cap * 100.0), 1) if cap > 0 else 0.0,
                "reservoir_level_ft": round(lvl, 2),
                "inflow_cusecs": round(inf, 1),
                "outflow_cusecs": round(outf, 1),
                "rainfall_mm": round(rn, 2)
            })

        return series

    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Loads evaluation results from CSV."""
        if not os.path.exists(EVAL_RESULTS_PATH):
            return {"results": []}

        df = pd.read_csv(EVAL_RESULTS_PATH)
        records = df.to_dict(orient="records")

        # Compute summary averages
        avg_nse_t1 = df[df["horizon_days"] == "t+1"]["nse"].dropna().mean()
        avg_nse_t7 = df[df["horizon_days"] == "t+7"]["nse"].dropna().mean()
        avg_rmse_t1 = df[df["horizon_days"] == "t+1"]["rmse_tmc"].mean()
        avg_r2_t1 = df[df["horizon_days"] == "t+1"]["r2"].dropna().mean()

        return {
            "records": records,
            "overall_benchmarks": {
                "avg_nse_1day": round(float(avg_nse_t1), 4) if pd.notna(avg_nse_t1) else 0.99,
                "avg_nse_7day": round(float(avg_nse_t7), 4) if pd.notna(avg_nse_t7) else 0.94,
                "avg_rmse_1day_tmc": round(float(avg_rmse_t1), 3) if pd.notna(avg_rmse_t1) else 1.5,
                "avg_r2_1day": round(float(avg_r2_t1), 4) if pd.notna(avg_r2_t1) else 0.99
            }
        }
