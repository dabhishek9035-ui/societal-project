"""
JALADRISHTI Platform Launcher
Starts the FastAPI backend used by the Next.js frontend.
"""
import os
import sys
import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("  JALADRISHTI (ಜಲದೃಷ್ಟಿ) — Karnataka Hydrological Telemetry Console")
    print("  PyTorch Delta-LSTM Forecasting & Catchment Simulation Platform")
    print("=" * 70)
    print("  -> API health: http://127.0.0.1:8000/api/health")
    print("  -> Interactive API docs: http://127.0.0.1:8000/docs")
    print("  -> Frontend: start Next.js separately at http://localhost:3000")
    print("=" * 70)
    
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
