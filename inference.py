"""
Inference Script for Meeting Scheduler RL Environment
=====================================================

This script runs an LLM agent against the Meeting Scheduler environment.
It plays all 3 tasks (easy, medium, hard) and logs results in the exact
format required by the competition.

REQUIRED ENVIRONMENT VARIABLES:
  API_BASE_URL   The API endpoint for the LLM (default: HF router)
  MODEL_NAME     The model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       Your Hugging Face / API key

STDOUT FORMAT (must be EXACT — judges parse this programmatically):
  [START] task=<task> env=meeting-scheduler model=<model>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

USAGE:
  # Set environment variables first
  export API_BASE_URL="https://router.huggingface.co/v1"
  export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
  export HF_TOKEN="hf_..."

  # Start the server (in another terminal)
  uvicorn server.server:app --host 0.0.0.0 --port 8000

  # Run inference
  python inference.py
"""

import json
import os
import re
import requests
from openai import OpenAI


# ---------------------------------------------------------------------------
# Configuration — reads from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

# Where the Meeting Scheduler server is running
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8002")

# Tasks to run (all 3 required by competition)
TASKS = ["easy", "medium", "hard"]

# Safety limits
MAX_STEPS = 20          # Hard cap to prevent infinite loops
TEMPERATURE = 0.3       # Low temperature = more deterministic/reliable output
MAX_TOKENS = 200        # Short — we only need a small JSON response


# ---------------------------------------------------------------------------
# System prompt — teaches the LLM how to play
# ---------------------------------------------------------------------------
# Design choices:
#   - Starts with the role and goal (sets context)
#   - Shows EXACT action format with examples (LLMs copy examples well)
#   - Lists constraints (so the LLM avoids invalid actions)
#   - Gives strategic advice (prioritize high-priority, save scarce rooms)
#   - Ends with a strong format reminder (LLMs tend to forget by mid-conversation)

SYSTEM_PROMPT = """You are a meeting scheduling assistant. Your job is to assign meetings to time slots and rooms.

## HOW EACH STEP WORKS
1. You will see a calendar with current bookings and one meeting to schedule.
2. You must respond with a JSON action to schedule or skip it.

## ACTION FORMAT
To schedule a meeting, respond with ONLY this JSON (no other text):
{"timeslot": "HH:MM", "room": "Room Name"}

To skip a meeting you cannot schedule, respond with ONLY:
{"timeslot": "skip"}

## CONSTRAINTS (violating any of these gives -1.0 penalty)
- Participants cannot be in two meetings at the same time
- The room must have enough capacity for all participants
- The room must have any required equipment (projector, video_conferencing, whiteboard)
- Multi-slot meetings (60 or 90 min) block consecutive time slots
- One meeting per room per time slot

## STRATEGY
- HIGH priority meetings are most important — schedule them first and protect their slots
- Check the PARTICIPANT SCHEDULES section to avoid conflicts
- If a meeting requires special equipment, only certain rooms have it
- If a meeting genuinely cannot fit anywhere (all valid slots are taken), skip it
- Prefer meeting preferred time windows when possible (bonus reward)
- On hard tasks, you may NEED to skip some low-priority meetings — that is okay

## CRITICAL: Response format
Respond with ONLY the JSON object. No explanation, no markdown, no extra text.
Example: {"timeslot": "10:00", "room": "Room A"}
"""


# ---------------------------------------------------------------------------
# LLM client setup
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)


# ---------------------------------------------------------------------------
# Helper: Parse LLM response into an action
# ---------------------------------------------------------------------------

def parse_llm_response(response_text: str) -> dict:
    """
    Extract a scheduling action from LLM output.

    LLMs are unreliable with output format. This function tries multiple
    strategies to extract {"timeslot": "...", "room": "..."} from whatever
    the LLM returns.

    Strategy (in order):
      1. Try parsing the entire response as JSON
      2. Look for {...} pattern anywhere in the response
      3. Look for timeslot and room keywords in plain text
      4. Fall back to skip (safe default — never crashes the server)

    WHY the fallback to "skip":
      A crash here would end the entire inference run. A skip just loses
      one meeting (small penalty) and the game continues. Better to skip
      one meeting than to fail the entire evaluation.
    """
    text = response_text.strip()

    # Strategy 1: Direct JSON parse
    try:
        parsed = json.loads(text)
        if "timeslot" in parsed:
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: Find JSON object {...} anywhere in the text
    # LLMs often wrap JSON in markdown backticks or add explanation text
    json_match = re.search(r'\{[^{}]*"timeslot"[^{}]*\}', text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if "timeslot" in parsed:
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 3: Look for slot and room mentions in plain text
    # Handles cases like: "I suggest scheduling at 10:00 in Room A"
    slot_match = re.search(r'(\d{1,2}:\d{2})', text)
    room_match = re.search(r'(Room\s+[A-Z])', text, re.IGNORECASE)
    if slot_match:
        action = {"timeslot": slot_match.group(1)}
        if room_match:
            action["room"] = room_match.group(1)
        return action

    # Strategy 4: Check if it's just saying "skip"
    if "skip" in text.lower():
        return {"timeslot": "skip"}

    # Strategy 5: Give up — default to skip (safe fallback)
    return {"timeslot": "skip"}


# ---------------------------------------------------------------------------
# Helper: Call the LLM
# ---------------------------------------------------------------------------

def ask_llm(observation_text: str, message_history: list = None) -> str:
    """
    Send the calendar observation to the LLM and get a scheduling action.

    We send:
      - System prompt (explains the game and action format)
      - The observation text (current calendar state + meeting to schedule)

    Returns the raw LLM response text.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": observation_text},
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # If LLM call fails (network error, rate limit, etc.), skip this meeting
        # Better to skip one meeting than crash the entire evaluation
        print(f"  [WARNING] LLM call failed: {e}")
        return '{"timeslot": "skip"}'


# ---------------------------------------------------------------------------
# Helper: Call the environment server
# ---------------------------------------------------------------------------

def env_reset(task: str) -> dict:
    """Call POST /reset and return the response."""
    r = requests.post(f"{ENV_BASE_URL}/reset", params={"task": task})
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    """Call POST /step and return the response."""
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action},
        headers={"Content-Type": "application/json"},
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Main: Run one task
# ---------------------------------------------------------------------------

def run_task(task_name: str):
    """
    Play one complete episode on the given task.

    Logs in the exact [START]/[STEP]/[END] format required by the competition.

    Returns the list of per-step rewards (for summary).
    """
    # --- [START] ---
    print(f"[START] task={task_name} env=meeting-scheduler model={MODEL_NAME}")

    # Reset environment
    reset_data = env_reset(task_name)
    obs = reset_data["observation"]

    rewards = []
    step_num = 0
    done = False

    while not done and step_num < MAX_STEPS:
        step_num += 1

        # Get the text summary (what the LLM reads)
        text_summary = obs.get("text_summary", "")

        # Ask the LLM what to do
        llm_response = ask_llm(text_summary)

        # Parse LLM response into an action dict
        action = parse_llm_response(llm_response)

        # Ensure action has required fields
        if "timeslot" not in action:
            action = {"timeslot": "skip"}

        # Send action to environment
        step_data = env_step(action)

        # Extract results
        reward = step_data.get("reward", 0.0)
        done = step_data.get("done", False)
        obs = step_data.get("observation", {})
        error = step_data.get("error", None)

        rewards.append(reward)

        # Format action as a compact string for logging
        action_str = json.dumps(action, separators=(",", ":"))

        # Format error field
        error_str = "null" if error is None else str(error)

        # --- [STEP] --- (exact format required by competition)
        print(
            f"[STEP] step={step_num} "
            f"action={action_str} "
            f"reward={reward:.2f} "
            f"done={'true' if done else 'false'} "
            f"error={error_str}"
        )

    # --- [END] ---
    # Determine success: final_score > 0.5 counts as success
    final_score = step_data.get("final_score", 0.0) if done else 0.0
    success = final_score > 0.5 if done else False

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step_num} "
        f"rewards={rewards_str}"
    )

    return rewards, final_score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run all 3 tasks in sequence."""
    print(f"=== Meeting Scheduler Inference ===")
    print(f"LLM:    {MODEL_NAME}")
    print(f"Server: {ENV_BASE_URL}")
    print(f"Tasks:  {TASKS}")
    print()

    all_results = {}

    for task in TASKS:
        rewards, score = run_task(task)
        all_results[task] = {
            "rewards": rewards,
            "final_score": score,
            "total_reward": sum(rewards),
        }
        print()  # Blank line between tasks

    # Summary
    print("=== SUMMARY ===")
    for task, result in all_results.items():
        print(
            f"  {task:8s}: score={result['final_score']:.2f}, "
            f"total_reward={result['total_reward']:.1f}, "
            f"steps={len(result['rewards'])}"
        )


if __name__ == "__main__":
    main()
