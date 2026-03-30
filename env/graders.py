def grade_episode(env):
    assignments = env.assignments
    total = len(assignments)

    weights = {"low": 1, "medium": 2, "high": 3}

    scheduled = 0
    priority_score = 0
    max_priority_score = 0
    slot_usage = {}

    for item in assignments:
        meeting = item["meeting"]
        valid = item["valid"]
        slot = item["timeslot"]

        max_priority_score += weights[meeting["priority"]]

        if valid:
            scheduled += 1
            priority_score += weights[meeting["priority"]]

            if slot:
                slot_usage.setdefault(slot, 0)
                slot_usage[slot] += 1

    # ---------------- CORE SCORES ----------------
    completion = scheduled / total
    priority = priority_score / max_priority_score if max_priority_score else 0

    # 🔥 CRITICAL FIX: distribution penalty
    if slot_usage:
        max_slot = max(slot_usage.values())
        concentration = max_slot / total   # how concentrated scheduling is
    else:
        concentration = 1

    # ---------------- FINAL SCORE ----------------
    score = (
        0.6 * completion
        + 0.4 * priority
        - 0.6 * concentration   # 🔥 THIS fixes worst case
    )

    return max(0.0, min(1.0, score))