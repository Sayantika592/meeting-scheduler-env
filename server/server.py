from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

from env.environment import SchedulerEnv
from env.tasks import get_task
from env.graders import grade_episode

app = FastAPI()

env = None


# -------- REQUEST SCHEMAS --------
class StepRequest(BaseModel):
    action: Dict


# -------- RESET --------
@app.post("/reset")
def reset(task: str = "easy"):
    global env
    task_data = get_task(task)
    env = SchedulerEnv(task_data)
    obs = env.reset()
    return {"observation": obs}


# -------- STEP --------
@app.post("/step")
def step(req: StepRequest):
    global env

    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}

    obs, reward, done, info = env.step(req.action)

    response = {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }

    if done:
        score = grade_episode(env)
        response["final_score"] = score

    return response


# -------- STATE --------
@app.get("/state")
def state():
    global env

    if env is None:
        return {"error": "Environment not initialized."}

    return env.state()