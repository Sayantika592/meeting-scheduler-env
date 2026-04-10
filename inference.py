"""
Inference Script for Meeting Scheduler RL Environment
=====================================================

MANDATORY ENVIRONMENT VARIABLES:
  API_BASE_URL   The API endpoint for the LLM (default: HF router)
  MODEL_NAME     The model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       Your Hugging Face / API key

STDOUT FORMAT (must be EXACT — judges parse this programmatically):
  [START] task=<task> env=meeting-scheduler model=<model>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

USAGE:
  export HF_TOKEN="hf_..."
  python inference.py
"""

import json
import os
import re
import sys

from openai import OpenAI

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SchedulerAction
from env.environment import SchedulerEnvironment
from env.graders import grade_episode


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 20
TEMPERATURE = 0.3
MAX_TOKENS = 200


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a meeting scheduling assistant. Your job is to assign meetings to time slots and rooms.

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
# LLM client
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)


# ---------------------------------------------------------------------------
# Parse LLM response
# ---------------------------------------------------------------------------

def parse_llm_response(response_text: str) -> dict:
    """
    Extract a scheduling action from LLM output.

    Tries multiple strategies to handle unreliable LLM output:
      1. Direct JSON parse
      2. Find {...} with "timeslot" anywhere in text
      3. Regex for time + room in plain text
      4. "skip" keyword detection
      5. Default to skip (never crashes)
    """
    text = response_text.strip()

    # Strategy 1: Direct JSON parse
    try:
        parsed = json.loads(text)
        if "timeslot" in parsed:
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: Find JSON object anywhere
    json_match = re.search(r'\{[^{}]*"timeslot"[^{}]*\}', text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if "timeslot" in parsed:
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 3: Regex for time + room
    slot_match = re.search(r'(\d{1,2}:\d{2})', text)
    room_match = re.search(r'(Room\s+[A-Z])', text, re.IGNORECASE)
    if slot_match:
        action = {"timeslot": slot_match.group(1)}
        if room_match:
            action["room"] = room_match.group(1)
        return action

    # Strategy 4: "skip" keyword
    if "skip" in text.lower():
        return {"timeslot": "skip"}

    # Strategy 5: Safe fallback
    return {"timeslot": "skip"}


# ---------------------------------------------------------------------------
# Ask the LLM
# ---------------------------------------------------------------------------

def ask_llm(observation_text: str) -> str:
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
        print(f"  [WARNING] LLM call failed: {e}", file=sys.stderr)
        return '{"timeslot": "skip"}'


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task(task_name: str):
    """Play one complete episode. Logs in exact [START]/[STEP]/[END] format."""

    print(f"[START] task={task_name} env=meeting-scheduler model={MODEL_NAME}")

    # Instantiate environment directly (no server needed)
    env = SchedulerEnvironment()
    obs = env.reset(task=task_name)

    rewards = []
    step_num = 0
    done = False
    last_error = None

    while not done and step_num < MAX_STEPS:
        step_num += 1

        # Get the text summary (what the LLM reads)
        text_summary = obs.text_summary

        # Ask the LLM
        llm_response = ask_llm(text_summary)

        # Parse into action
        action_dict = parse_llm_response(llm_response)
        if "timeslot" not in action_dict:
            action_dict = {"timeslot": "skip"}

        # Create typed action
        action = SchedulerAction(
            timeslot=action_dict.get("timeslot", "skip"),
            room=action_dict.get("room"),
        )

        # Step the environment
        obs = env.step(action)

        reward = obs.reward if obs.reward is not None else 0.01
        done = obs.done
        rewards.append(reward)

        # Format for logging
        action_str = json.dumps(action_dict, separators=(",", ":"))
        error_str = "null"  # Our env never returns errors via the action

        print(
            f"[STEP] step={step_num} "
            f"action={action_str} "
            f"reward={reward:.2f} "
            f"done={'true' if done else 'false'} "
            f"error={error_str}"
        )

    # Compute final score
    final_score = grade_episode(env) if done else 0.01
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
    print(f"=== Meeting Scheduler Inference ===")
    print(f"LLM:   {MODEL_NAME}")
    print(f"Tasks: {TASKS}")
    print()

    all_results = {}

    for task in TASKS:
        rewards, score = run_task(task)
        all_results[task] = {
            "rewards": rewards,
            "final_score": score,
            "total_reward": sum(rewards),
        }
        print()

    print("=== SUMMARY ===")
    for task, result in all_results.items():
        print(
            f"  {task:8s}: score={result['final_score']:.2f}, "
            f"total_reward={result['total_reward']:.1f}, "
            f"steps={len(result['rewards'])}"
        )


if __name__ == "__main__":
    main()
