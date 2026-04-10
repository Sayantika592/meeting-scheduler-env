"""
FastAPI application for the Meeting Scheduler Environment.

Uses openenv's create_app() for OpenEnv spec compliance (schema, health,
metadata, ws, docs endpoints), but OVERRIDES /reset and /step to persist
environment state across HTTP calls.

WHY: openenv's default HTTP handlers create a NEW environment instance
per request and destroy it after. This breaks stateful environments where
/reset sets up state that /step needs to access. The competition runner
calls /reset then /step via HTTP, so state must persist.
"""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from e

import sys
import os

# Ensure project root is on path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import SchedulerAction, SchedulerObservation
from env.environment import SchedulerEnvironment
from env.graders import grade_episode

from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Create the base app via openenv (gives us /health, /schema, /metadata, /ws, /docs)
# ---------------------------------------------------------------------------
app = create_app(
    SchedulerEnvironment,
    SchedulerAction,
    SchedulerObservation,
    env_name="meeting-scheduler",
    max_concurrent_envs=1,
)

# ---------------------------------------------------------------------------
# Persistent environment instance for HTTP /reset and /step
# ---------------------------------------------------------------------------
_env: Optional[SchedulerEnvironment] = None


class ResetRequest(BaseModel):
    task: str = "easy"
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class StepRequestBody(BaseModel):
    action: Dict[str, Any]
    timeout_s: Optional[float] = None


class EnvResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


def _serialize_obs(obs: SchedulerObservation) -> dict:
    """Convert observation to API response format."""
    obs_dict = obs.model_dump(exclude={"reward", "done", "metadata"})
    return {
        "observation": obs_dict,
        "reward": obs.reward,
        "done": obs.done,
    }


# ---------------------------------------------------------------------------
# OVERRIDE /reset — persists the environment
# ---------------------------------------------------------------------------
# Remove the default routes first, then re-add
for route in list(app.routes):
    if hasattr(route, 'path') and route.path in ('/reset', '/step', '/state'):
        app.routes.remove(route)


@app.post("/reset", response_model=EnvResponse)
async def reset_endpoint(request: ResetRequest = Body(default_factory=ResetRequest)):
    global _env
    _env = SchedulerEnvironment()
    obs = _env.reset(
        seed=request.seed,
        episode_id=request.episode_id,
        task=request.task,
    )
    return EnvResponse(**_serialize_obs(obs))


# ---------------------------------------------------------------------------
# OVERRIDE /step — uses the persisted environment
# ---------------------------------------------------------------------------
@app.post("/step", response_model=EnvResponse)
async def step_endpoint(request: StepRequestBody):
    global _env
    if _env is None:
        # Auto-reset if step called without reset
        _env = SchedulerEnvironment()
        _env.reset(task="easy")

    action = SchedulerAction.model_validate(request.action)
    obs = _env.step(action)
    return EnvResponse(**_serialize_obs(obs))


# ---------------------------------------------------------------------------
# OVERRIDE /state — uses the persisted environment
# ---------------------------------------------------------------------------
@app.get("/state")
async def state_endpoint():
    global _env
    if _env is None:
        return {"episode_id": None, "step_count": 0}
    state = _env.state
    return state.model_dump()


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
