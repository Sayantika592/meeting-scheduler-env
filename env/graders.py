"""
Episode grader for the Meeting Scheduler Environment.

Called ONCE at episode end to produce a final score in [0.0, 1.0].

The grader examines env.assignments (the full history of scheduling decisions)
and scores across 4 dimensions:

  FORMULA:
    score = (0.30 × completion_rate)
          + (0.35 × priority_weighted_completion)
          + (0.15 × preference_satisfaction)
          + (0.20 × skip_quality)

  Each component is in [0.0, 1.0], so the final score is too.

  WHY THESE WEIGHTS:
  - Priority completion (35%): The most important thing. Scheduling a high-priority
    investor call matters more than scheduling a low-priority social event.
  - Completion rate (30%): Raw throughput. Did the agent schedule most meetings?
  - Skip quality (20%): On the hard task, skipping is NECESSARY. But skipping
    schedulable meetings is bad. This rewards smart triage.
  - Preference satisfaction (15%): Nice-to-have. Respecting "before 10:00" shows
    attention to detail but isn't as critical as getting meetings scheduled at all.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .environment import SchedulerEnvironment


def grade_episode(env: "SchedulerEnvironment") -> float:
    """
    Grade a completed episode. Returns a float in [0.0, 1.0].

    Must be called AFTER the episode is done (all meetings processed).
    Reads env.assignments for the full decision history.
    """
    assignments = env.assignments
    total = len(assignments)

    if total == 0:
        return 0.0

    # ------------------------------------------------------------------
    # Component 1: COMPLETION RATE (0.0 – 1.0)
    # What fraction of meetings were successfully scheduled?
    #
    # Formula: valid_count / total_meetings
    #
    # Example: 4 out of 5 scheduled → 0.8
    # ------------------------------------------------------------------
    valid_count = sum(1 for a in assignments if a.valid)
    completion_rate = valid_count / total

    # ------------------------------------------------------------------
    # Component 2: PRIORITY-WEIGHTED COMPLETION (0.0 – 1.0)
    # Did the agent schedule the important meetings?
    #
    # Each meeting has a priority weight: high=3, medium=2, low=1
    # Score = sum(weights of scheduled meetings) / sum(weights of ALL meetings)
    #
    # Example: Schedule 1 high (3) + 1 low (1) out of 1 high + 2 medium + 1 low
    #          = 4 / (3 + 4 + 1) = 4/8 = 0.5
    #
    # WHY: This makes skipping a high-priority meeting 3× worse than
    # skipping a low-priority one. Forces the agent to protect important meetings.
    # ------------------------------------------------------------------
    priority_weights = {"high": 3, "medium": 2, "low": 1}

    achieved_priority = 0
    max_priority = 0
    for a in assignments:
        weight = priority_weights.get(a.meeting.priority, 1)
        max_priority += weight
        if a.valid:
            achieved_priority += weight

    priority_completion = achieved_priority / max_priority if max_priority > 0 else 0.0

    # ------------------------------------------------------------------
    # Component 3: PREFERENCE SATISFACTION (0.0 – 1.0)
    # Among meetings that HAVE a preferred time window AND were scheduled,
    # what fraction had their preference met?
    #
    # Meetings without preferences are excluded from this calculation
    # (they can't contribute positively or negatively).
    #
    # If NO meetings have preferences (easy task), this defaults to 1.0
    # so it doesn't penalize the agent for something outside its control.
    #
    # Example: 2 meetings with preferences, 1 met → 0.5
    # ------------------------------------------------------------------
    meetings_with_prefs = [a for a in assignments if a.meeting.preferred_time_window is not None]

    if len(meetings_with_prefs) == 0:
        # No meetings have preferences → full marks (nothing to satisfy)
        preference_satisfaction = 1.0
    else:
        # Only count scheduled meetings with preferences
        scheduled_with_prefs = [a for a in meetings_with_prefs if a.valid]
        if len(scheduled_with_prefs) == 0:
            # Had preferences but none were scheduled → 0
            preference_satisfaction = 0.0
        else:
            prefs_met = sum(1 for a in scheduled_with_prefs if a.preferred_time_met)
            preference_satisfaction = prefs_met / len(scheduled_with_prefs)

    # ------------------------------------------------------------------
    # Component 4: SKIP QUALITY (0.0 – 1.0)
    # When the agent skipped meetings, was it smart or lazy?
    #
    # - Skipping an unschedulable meeting = GOOD (smart triage)
    # - Skipping a schedulable meeting = BAD (lazy)
    # - Not skipping at all = PERFECT (if nothing needed skipping)
    #
    # Formula:
    #   If no skips occurred → 1.0 (no mistakes possible)
    #   Otherwise → smart_skips / total_skips
    #
    # Example: Skipped 3 meetings, 2 were genuinely unschedulable → 2/3 = 0.67
    # ------------------------------------------------------------------
    skipped = [a for a in assignments if a.skipped]

    if len(skipped) == 0:
        skip_quality = 1.0  # Never skipped → perfect (no bad skips)
    else:
        smart_skips = sum(1 for a in skipped if not a.was_schedulable)
        skip_quality = smart_skips / len(skipped)

    # ------------------------------------------------------------------
    # FINAL SCORE
    # ------------------------------------------------------------------
    score = (
        0.30 * completion_rate
        + 0.35 * priority_completion
        + 0.15 * preference_satisfaction
        + 0.20 * skip_quality
    )

    # Clamp to [0.0, 1.0] (should already be in range, but safety first)
    return max(0.0, min(1.0, score))
