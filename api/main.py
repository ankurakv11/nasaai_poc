import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests

from app.routes import measurement, recommendation, body_measurement
from app.middleware.memory_cleanup import MemoryCleanupMiddleware

WORKER_URL = os.getenv("WORKER_URL", "http://worker:8001")

app = FastAPI(
    title="Standalone Sizing API",
    description="API Gateway for clothing size recommendation",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(MemoryCleanupMiddleware)

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Register your routes
app.include_router(measurement.router, prefix="", tags=["measurements"])
app.include_router(recommendation.router, prefix="", tags=["recommendations"])
app.include_router(body_measurement.router, prefix="", tags=["body-measurements"])

@app.get("/")
def root():
    return {"message": "API Gateway running"}

@app.get("/health")
def health():
    # Check local
    api_status = {"api": "ok"}

    # Check worker
    try:
        worker_status = requests.get(f"{WORKER_URL}/health", timeout=3).json()
    except Exception:
        worker_status = {"worker": "unreachable"}

    return {"api_status": api_status, "worker_status": worker_status}

