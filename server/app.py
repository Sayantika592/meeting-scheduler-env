"""
FastAPI application for the Meeting Scheduler Environment.

Uses openenv's create_app() which auto-generates ALL required endpoints:
  POST /reset, POST /step, GET /state, GET /schema, GET /health,
  WS /ws, GET /docs, GET /openapi.json

This is required for `openenv validate` to pass.
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


app = create_app(
    SchedulerEnvironment,
    SchedulerAction,
    SchedulerObservation,
    env_name="meeting-scheduler",
    max_concurrent_envs=1,
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
