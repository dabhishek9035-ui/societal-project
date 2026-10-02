import os
import sys
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.reservoir_service import ReservoirService
from backend.services.forecast_service import ForecastService

app = FastAPI(
    title="JALADRISHTI API - Karnataka Hydrological & Reservoir Forecasting Platform",
    description="Operational hydrology, catchment telemetry, and PyTorch Delta-LSTM predictive intelligence for Karnataka's 10 major dams.",
    version="1.0.0"
)

# Enable CORS for local dev and frontend ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

reservoir_service = ReservoirService()
forecast_service = ForecastService()

class SimulationRequest(BaseModel):
    rain_factor: float = Field(default=1.0, ge=0.0, le=3.0, description="Multiplier for upstream catchment precipitation (1.0 = baseline, 1.5 = +50%)")
    outflow_factor: float = Field(default=1.0, ge=0.0, le=3.0, description="Multiplier for spillway and canal gate discharges (1.0 = baseline, 0.7 = -30%)")

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "JALADRISHTI Hydrological Engine",
        "version": "1.0.0",
        "dams_available": 10
    }

@app.get("/api/summary")
def get_statewide_summary():
    """Statewide aggregate capacity, current live storage, inflows and basin breakdowns."""
    return reservoir_service.get_statewide_summary()

@app.get("/api/reservoirs")
def get_reservoirs(basin: Optional[str] = Query(None, description="Filter by river basin (Krishna, Cauvery, Sharavathi, Kali, etc.)")):
    """Returns telemetry and live status across Karnataka's monitored reservoirs."""
    reservoirs = reservoir_service.get_all_reservoirs()
    if basin and basin.lower() != "all":
        reservoirs = [r for r in reservoirs if basin.lower() in r["basin"].lower()]
    return reservoirs

@app.get("/api/reservoirs/{dam_id}")
def get_reservoir_detail(dam_id: str, history_days: int = Query(90, ge=7, le=365)):
    """Returns detailed engineering specifications, current gauge readings, and historical time series."""
    meta = [r for r in reservoir_service.get_all_reservoirs() if r["id"] == dam_id]
    if not meta:
        raise HTTPException(status_code=404, detail=f"Reservoir '{dam_id}' not found.")
    
    timeseries = reservoir_service.get_timeseries(dam_id, days=history_days)
    return {
        "metadata": meta[0],
        "timeseries": timeseries
    }

@app.get("/api/reservoirs/{dam_id}/forecast")
def get_reservoir_forecast(dam_id: str):
    """Executes the PyTorch Delta-LSTM model to generate 1, 7, 14, and 30-day storage forecasts."""
    try:
        return forecast_service.predict_forecast(dam_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/reservoirs/{dam_id}/simulate")
def simulate_catchment_scenario(dam_id: str, req: SimulationRequest):
    """Simulates catchment perturbation (cloudburst rain surge vs drought vs discharge adjustments)."""
    try:
        return forecast_service.simulate_scenario(dam_id, rain_factor=req.rain_factor, outflow_factor=req.outflow_factor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")

@app.get("/api/evaluation")
def get_model_evaluation():
    """Returns model benchmark scores (RMSE, MAE, Nash-Sutcliffe Efficiency NSE, R2) across horizons."""
    return reservoir_service.get_evaluation_metrics()

# Serve static frontend dist if built
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
