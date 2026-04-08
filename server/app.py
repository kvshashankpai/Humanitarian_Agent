"""
app/main.py — FastAPI server exposing OpenEnv HTTP API

Endpoints:
  POST /reset          → Observation
  POST /step           → StepResult
  GET  /state          → current state dict
  POST /grade          → GraderResult
  GET  /tasks          → list available tasks
  GET  /health         → 200 OK
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Humanitarian_Agent.humanitarian_env import (
    Action,
    GraderResult,
    HumanitarianAidEnv,
    Observation,
    StepResult,
)

app = FastAPI(
    title="Humanitarian Aid Allocation — OpenEnv",
    description="OpenEnv environment for humanitarian supply distribution under scarcity.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per task (stateful, single-user demo)
_envs: dict[str, HumanitarianAidEnv] = {}
_active_task: str = "easy"


def _get_env(task: Optional[str] = None) -> HumanitarianAidEnv:
    t = task or _active_task
    if t not in _envs:
        _envs[t] = HumanitarianAidEnv(task=t, seed=42)
    return _envs[t]


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: Optional[str] = None
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    zone_id: int = 0
    quantity: int = 0
    priority: str = "med"
    task: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "humanitarian-aid-allocation-openenv"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "easy",   "name": "Small Zone Coverage",       "horizon": 12, "n_zones": 3},
            {"id": "medium", "name": "Multi-Zone with Blockages",  "horizon": 16, "n_zones": 5},
            {"id": "hard",   "name": "Supply Shock Crisis",        "horizon": 20, "n_zones": 7},
        ]
    }


@app.post("/reset", response_model=Observation)
def reset(body: ResetRequest = ResetRequest()):
    global _active_task
    task = body.task or "easy"
    _active_task = task
    seed = body.seed if body.seed is not None else 42
    _envs[task] = HumanitarianAidEnv(task=task, seed=seed)
    obs = _envs[task].reset()
    return obs


@app.post("/step", response_model=StepResult)
def step(body: StepRequest):
    env = _get_env(body.task)
    action = Action(zone_id=body.zone_id, quantity=body.quantity, priority=body.priority)
    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/state")
def state(task: Optional[str] = None):
    env = _get_env(task)
    return env.state()


@app.post("/grade", response_model=GraderResult)
def grade(task: Optional[str] = None):
    env = _get_env(task)
    return env.grade()
