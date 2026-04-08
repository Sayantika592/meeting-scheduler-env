"""
FastAPI server for the Meeting Scheduler RL Environment.

Three endpoints:
  POST /reset?task=easy    Start a new episode (easy/medium/hard)
  POST /step               Send an action, get observation + reward
  GET  /state              Get episode metadata

Also:
  GET  /health             Health check (judges ping this to verify deployment)

HOW IT WORKS:
  - One global SchedulerEnv instance (single-session, fine for competition)
  - /reset creates a fresh environment and returns the first observation
  - /step receives a SchedulerAction, calls env.step(), returns the result
  - /state returns lightweight metadata (step count, assignments made, etc.)

ERROR HANDLING:
  - If /step is called before /reset → returns error JSON, doesn't crash
  - If the agent sends malformed JSON → Pydantic rejects it with a clear message
  - If the action is invalid (conflict, wrong room) → returns negative reward + reason
    (this is handled by the environment, not the server)
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# We import from env package (one level up)
# When running with uvicorn from project root: uvicorn server.server:app
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.environment import SchedulerEnv
from env.models import SchedulerAction, SchedulerObservation, SchedulerState
from env.graders import grade_episode

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Meeting Scheduler RL Environment",
    description="RL environment for scheduling meetings with constraints.",
    version="1.0.0",
)

# Global environment instance (single-session)
# Reset to None so we can detect "not initialized" state
env: Optional[SchedulerEnv] = None

@app.get("/")
def root():
    return {"status": "ok", "message": "Meeting Scheduler Environment API is running"}



# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class StepRequest(BaseModel):
    """Request body for POST /step."""
    action: SchedulerAction


class ResetRequest(BaseModel):
    """Optional request body for POST /reset."""
    task: str = "easy"


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ResetResponse(BaseModel):
    """Response from POST /reset."""
    observation: SchedulerObservation


class StepResponse(BaseModel):
    """Response from POST /step."""
    observation: SchedulerObservation
    reward: float
    done: bool
    info: dict = {}
    final_score: Optional[float] = None  # Only set when done=True


class ErrorResponse(BaseModel):
    """Error response."""
    error: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Health check endpoint.

    Judges ping this to verify the Space is alive.
    Returns 200 with a simple status message.
    """
    return {"status": "healthy", "environment": "meeting-scheduler"}


@app.post("/reset")
def reset(task: str = "easy"):
    """
    Start a new episode.

    Query parameter:
      task: "easy", "medium", or "hard" (default: "easy")

    Returns:
      The first observation (first meeting to schedule + empty calendar).

    Example:
      POST /reset?task=medium
    """
    global env

    # Validate task name before creating environment
    valid_tasks = ["easy", "medium", "hard"]
    if task not in valid_tasks:
        return {"error": f"Unknown task: '{task}'. Choose from: {valid_tasks}"}

    # Create fresh environment and reset
    env = SchedulerEnv()
    obs = env.reset(task)

    return {"observation": obs.model_dump()}


@app.post("/step")
def step(req: StepRequest):
    """
    Take one scheduling action.

    Request body:
      {"action": {"timeslot": "10:00", "room": "Room A"}}
      or
      {"action": {"timeslot": "skip"}}

    Returns:
      observation: updated calendar state + next meeting
      reward: float (positive for good scheduling, negative for conflicts)
      done: bool (true when all meetings have been processed)
      final_score: float 0.0-1.0 (only present when done=true)

    Error cases:
      - Environment not initialized → returns error message
      - Malformed action → Pydantic validation error (422)
      - Invalid scheduling (conflict, wrong room) → negative reward + reason
        (this is NOT an HTTP error — it's a valid game response)
    """
    global env

    # Check environment is initialized
    if env is None:
        return {
            "error": "Environment not initialized. Call POST /reset first.",
            "observation": {},
            "reward": 0.0,
            "done": False,
        }

    # Process the action
    obs, reward, done, info = env.step(req.action)

    response = {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }

    # When episode ends, run the grader and include the final score
    if done:
        response["final_score"] = grade_episode(env)

    return response


@app.get("/state")
def get_state():
    """
    Get episode metadata.

    Returns:
      task_name, step_count, episode_id, assignments_made, total_meetings, done

    Does NOT return the full calendar — use the observation from /step for that.
    """
    global env

    if env is None:
        return {"error": "Environment not initialized. Call POST /reset first."}

    return env.state().model_dump()

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()


