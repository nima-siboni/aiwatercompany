"""
PR #248: Serve the CO2-uptake surrogate, with MC-dropout uncertainty

Puts a FastAPI service in front of the uptake surrogate so the screening dashboard and the
agents can call it. The model (train_surrogate.py) predicts log10(uptake in mmol/g) from the
Zeo++ descriptors plus log10(P / Pa) and 1000/T, all standardised with the scaler saved at
training time. Each prediction comes with an uncertainty from 30 MC-dropout passes, and every
prediction is logged to Postgres for monitoring. `/screen` runs large screens in the
background. Deployed with `uvicorn service:app --workers 4`; Kubernetes probes `/health`.
"""
import uuid
from contextlib import asynccontextmanager

import joblib
import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from surrogate import PredictionLog, UptakeMLP  # MLP blocks: Linear -> BatchNorm1d -> ReLU -> Dropout

MODEL_VERSION = "v3"
engine = create_engine("postgresql+psycopg://screening@db/screening")
session = Session(engine)
JOBS = {}


@asynccontextmanager
async def lifespan(app):
    torch.set_grad_enabled(False)
    model = UptakeMLP()
    model.load_state_dict(torch.load(f"models/uptake_mlp_{MODEL_VERSION}.pt", weights_only=True))
    app.state.model = model.eval()
    app.state.scaler = joblib.load(f"models/uptake_scaler_{MODEL_VERSION}.joblib")
    yield


app = FastAPI(lifespan=lifespan)


class Query(BaseModel):
    mof_id: str
    descriptors: dict[str, float]  # {"lcd": ..., "pld": ..., "density": ..., "void_fraction": ...}
    pressure_bar: float
    temperature_k: float = 298.0


class Prediction(BaseModel):
    mof_id: str
    uptake_mmol_g: float
    std: float


def featurise(queries):
    rows = []
    for q in queries:
        x = list(q.descriptors.values())
        x += [np.log(q.pressure_bar), 1000.0 / q.temperature_k]
        rows.append(x)
    return torch.tensor(app.state.scaler.transform(np.array(rows)), dtype=torch.float32)


def mc_dropout(model, x, n_samples=30):
    model.train()  # switch dropout on
    with torch.no_grad():
        samples = torch.stack([model(x).squeeze(-1) for _ in range(n_samples)])
    model.eval()
    return samples.mean(0), samples.std(0)


def predict_queries(queries):
    # the dashboard sends the same (MOF, P, T) many times: predict each one once
    unique = {(q.mof_id, q.pressure_bar, q.temperature_k): q for q in queries}
    mean, std = mc_dropout(app.state.model, featurise(list(unique.values())))
    preds = [Prediction(mof_id=q.mof_id, uptake_mmol_g=m, std=s)
             for q, m, s in zip(queries, mean.tolist(), std.tolist())]
    for p in preds:
        session.add(PredictionLog(mof_id=p.mof_id, uptake=p.uptake_mmol_g, model=MODEL_VERSION))
    session.commit()
    return preds


@app.post("/predict", response_model=list[Prediction])
def predict(queries: list[Query]):
    return predict_queries(queries)


def run_screen(job_id, queries):
    try:
        JOBS[job_id] = {"status": "done", "predictions": predict_queries(queries)}
    except Exception as exc:
        JOBS[job_id] = {"status": "failed", "error": str(exc)}


@app.post("/screen")
def screen(queries: list[Query], background: BackgroundTasks):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "running"}
    background.add_task(run_screen, job_id, queries)
    return {"job_id": job_id}


@app.get("/screen/{job_id}")
def screen_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="unknown job")
    return JOBS[job_id]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_VERSION}
