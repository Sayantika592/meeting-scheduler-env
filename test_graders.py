#!/usr/bin/env python3
"""
Quick local test — runs all 3 tasks directly (no server needed).
Verifies grader scores and prints a pass/fail summary.

Usage:
    python test_graders.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import SchedulerEnv
from env.graders import grade_episode
from env.models import SchedulerAction

TASKS = ["easy", "medium", "hard"]


def optimal_easy(env):
    """Hand-coded optimal play for easy task — should score ~1.0."""
    actions = [
        SchedulerAction(timeslot="9:00", room="Room A"),   # Morning Standup (Alice, Bob)
        SchedulerAction(timeslot="9:00", room="Room B"),   # Sprint Planning (Carol, Dave)
        SchedulerAction(timeslot="9:30", room="Room A"),   # Design Review (Eve)
        SchedulerAction(timeslot="9:30", room="Room B"),   # Client Sync (Alice→free at 9:30, Carol→free at 9:30)
        SchedulerAction(timeslot="10:00", room="Room A"),  # Team Retro (Bob, Dave, Eve)
    ]
    return actions


def optimal_medium(env):
    """Good play for medium task — should score ~0.75+."""
    actions = [
        SchedulerAction(timeslot="9:00", room="Room B"),    # Eng Standup (Alice,Bob,Carol) before 10
        SchedulerAction(timeslot="9:30", room="Room A"),    # Q3 Budget 60min (Alice@9:30,Dave) needs projector
        SchedulerAction(timeslot="9:00", room="Room C"),    # Client Demo 60min (Bob@9:30 busy!) — try 9:00
        SchedulerAction(timeslot="10:30", room="Room B"),   # UX Feedback (Carol, Frank)
        SchedulerAction(timeslot="10:30", room="Room C"),   # Vendor Negotiation (Dave,Eve) needs video
        SchedulerAction(timeslot="10:30", room="Room A"),   # Architecture (Alice,Frank) needs whiteboard
        SchedulerAction(timeslot="11:00", room="Room B"),   # Onboarding (Carol) after 11
        SchedulerAction(timeslot="11:00", room="Room A"),   # Security Audit (Bob, Dave)
    ]
    return actions


def skip_all(env):
    """Skip every meeting — baseline worst-case for grader."""
    return [SchedulerAction(timeslot="skip")] * len(env.meetings)


def run_with_actions(task_name, actions, label=""):
    """Run an episode with given actions and return the grader score."""
    env = SchedulerEnv()
    env.reset(task_name)

    total_reward = 0.0
    for i, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    score = grade_episode(env)
    return score, total_reward, env.step_idx


def main():
    print("=" * 60)
    print("  Meeting Scheduler — Grader Score Verification")
    print("=" * 60)
    print()

    all_pass = True

    # 1. Test that graders return scores in [0.0, 1.0] for all tasks
    print("--- Skip-all baseline (verifies grader range) ---")
    for task in TASKS:
        env = SchedulerEnv()
        env.reset(task)
        actions = skip_all(env)
        score, reward, steps = run_with_actions(task, actions)
        in_range = 0.0 <= score <= 1.0
        status = "✓" if in_range else "✗ OUT OF RANGE"
        print(f"  {task:8s}: score={score:.4f}  reward={reward:.1f}  steps={steps}  {status}")
        if not in_range:
            all_pass = False
    print()

    # 2. Test optimal play on easy task
    print("--- Optimal play (easy task — should score ~1.0) ---")
    env = SchedulerEnv()
    env.reset("easy")
    actions = optimal_easy(env)
    score, reward, steps = run_with_actions("easy", actions)
    status = "✓" if score > 0.9 else "⚠ LOW"
    print(f"  easy:     score={score:.4f}  reward={reward:.1f}  steps={steps}  {status}")
    if score <= 0.5:
        all_pass = False
    print()

    # 3. Test medium task with a reasonable strategy
    print("--- Good play (medium task — should score ~0.7+) ---")
    env = SchedulerEnv()
    env.reset("medium")
    actions = optimal_medium(env)
    score, reward, steps = run_with_actions("medium", actions)
    status = "✓" if score > 0.6 else "⚠ LOW"
    print(f"  medium:   score={score:.4f}  reward={reward:.1f}  steps={steps}  {status}")
    print()

    # 4. Verify grader never returns negative or >1
    print("--- Edge case: empty episode ---")
    env = SchedulerEnv()
    env.reset("easy")
    # Don't step at all — call grader with 0 assignments
    score = grade_episode(env)
    status = "✓" if score == 0.0 else "✗"
    print(f"  empty:    score={score:.4f}  {status}")
    print()

    # 5. Verify reset produces clean state
    print("--- State reset verification ---")
    for task in TASKS:
        env = SchedulerEnv()
        obs = env.reset(task)
        state = env.state()
        ok = (state.step_count == 0
              and state.assignments_made == 0
              and not state.done
              and obs.current_meeting is not None
              and obs.remaining_count > 0
              and len(obs.calendar_grid) > 0)
        status = "✓" if ok else "✗"
        print(f"  {task:8s}: meetings={state.total_meetings}  rooms={len(obs.available_rooms)}  {status}")
    print()

    # Summary
    print("=" * 60)
    if all_pass:
        print("  ALL CHECKS PASSED ✓")
    else:
        print("  SOME CHECKS FAILED ✗ — Review output above")
    print("=" * 60)


if __name__ == "__main__":
    main()
