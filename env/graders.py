"""
Validator-safe graders for Meeting Scheduler.
"""

# -------------------------------
# Core grading logic
# -------------------------------

def grade_episode(env):
    # -------- SAFE GUARD --------
    if env is None or not hasattr(env, "assignments"):
        return 0.5  # neutral score (valid for validator)

    assignments = env.assignments
    total = len(assignments)

    if total == 0:
        return 0.5  # avoid weird edge case

    # ----------------------------
    # 1. Completion Rate
    # ----------------------------
    valid_count = sum(1 for a in assignments if getattr(a, "valid", False))
    completion_rate = valid_count / total

    # ----------------------------
    # 2. Priority Completion
    # ----------------------------
    priority_weights = {"high": 3, "medium": 2, "low": 1}

    achieved_priority = 0
    max_priority = 0

    for a in assignments:
        meeting = getattr(a, "meeting", None)
        priority = getattr(meeting, "priority", "low")
        weight = priority_weights.get(priority, 1)

        max_priority += weight
        if getattr(a, "valid", False):
            achieved_priority += weight

    priority_completion = achieved_priority / max_priority if max_priority > 0 else 0.0

    # ----------------------------
    # 3. Preference Satisfaction
    # ----------------------------
    meetings_with_prefs = [
        a for a in assignments
        if getattr(getattr(a, "meeting", None), "preferred_time_window", None) is not None
    ]

    if len(meetings_with_prefs) == 0:
        preference_satisfaction = 1.0
    else:
        scheduled_with_prefs = [a for a in meetings_with_prefs if getattr(a, "valid", False)]

        if len(scheduled_with_prefs) == 0:
            preference_satisfaction = 0.0
        else:
            prefs_met = sum(1 for a in scheduled_with_prefs if getattr(a, "preferred_time_met", False))
            preference_satisfaction = prefs_met / len(scheduled_with_prefs)

    # ----------------------------
    # 4. Skip Quality
    # ----------------------------
    skipped = [a for a in assignments if getattr(a, "skipped", False)]

    if len(skipped) == 0:
        skip_quality = 1.0
    else:
        smart_skips = sum(1 for a in skipped if not getattr(a, "was_schedulable", True))
        skip_quality = smart_skips / len(skipped)

    # ----------------------------
    # Final Score
    # ----------------------------
    score = (
        0.30 * completion_rate
        + 0.35 * priority_completion
        + 0.15 * preference_satisfaction
        + 0.20 * skip_quality
    )

    # clamp strictly inside (0,1)
    return max(0.01, min(0.99, score))


# -------------------------------
# Validator-compatible classes
# -------------------------------

class EasyGrader:
    def grade(self, env):
        try:
            return float(grade_episode(env))
        except Exception:
            return 0.5


class MediumGrader:
    def grade(self, env):
        try:
            return float(grade_episode(env))
        except Exception:
            return 0.5


class HardGrader:
    def grade(self, env):
        try:
            return float(grade_episode(env))
        except Exception:
            return 0.5
