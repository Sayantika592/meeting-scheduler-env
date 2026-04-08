"""
Task definitions for the Meeting Scheduler Environment.

Each task returns a dict with:
  - meetings: list of meeting dicts (converted to Meeting models by the environment)
  - rooms: list of room dicts (converted to Room models)
  - time_slots: ordered list of 30-min slot strings

Difficulty progression:
  - easy:   Few meetings, plenty of space, no equipment/preference constraints, all 30-min
  - medium: More meetings, shared participants, equipment requirements, 30+60 min durations
  - hard:   Over-constrained — more meetings than can fit, agent must triage by priority

Why these specific numbers?
  Easy:  5 meetings × 1 slot each = 5 slots needed, 6 slots × 2 rooms = 12 available → always fits
  Medium: 8 meetings, some 60-min (2 slots) ≈ 12 slots needed, 6 × 3 = 18 available → tight but possible
  Hard: 15 meetings, mixed durations ≈ 25 slots needed, 8 × 4 = 32 available BUT participant
        conflicts make it impossible to schedule all → must skip ~3-5 low-priority ones
"""


def get_task(task_name: str) -> dict:
    """Return task data by name. Raises ValueError for unknown tasks."""
    tasks = {
        "easy": easy_task,
        "medium": medium_task,
        "hard": hard_task,
    }
    if task_name not in tasks:
        raise ValueError(f"Unknown task: {task_name!r}. Choose from: {list(tasks.keys())}")
    return tasks[task_name]()


# ---------------------------------------------------------------------------
# EASY — 5 meetings, 6 slots, 2 rooms, all 30 min, no equipment, no prefs
# ---------------------------------------------------------------------------
# Design: Every meeting can fit somewhere. Zero conflict pressure.
# A random agent could score decently. Tests basic slot+room assignment.

def easy_task() -> dict:
    return {
        "meetings": [
            {
                "id": 1,
                "name": "Morning Standup",
                "participants": ["Alice", "Bob"],
                "priority": "medium",
                "duration_minutes": 30,
                "required_equipment": [],
                "preferred_time_window": None,
            },
            {
                "id": 2,
                "name": "Sprint Planning",
                "participants": ["Carol", "Dave"],
                "priority": "high",
                "duration_minutes": 30,
                "required_equipment": [],
                "preferred_time_window": None,
            },
            {
                "id": 3,
                "name": "Design Review",
                "participants": ["Eve"],
                "priority": "low",
                "duration_minutes": 30,
                "required_equipment": [],
                "preferred_time_window": None,
            },
            {
                "id": 4,
                "name": "Client Sync",
                "participants": ["Alice", "Carol"],
                "priority": "high",
                "duration_minutes": 30,
                "required_equipment": [],
                "preferred_time_window": None,
            },
            {
                "id": 5,
                "name": "Team Retrospective",
                "participants": ["Bob", "Dave", "Eve"],
                "priority": "medium",
                "duration_minutes": 30,
                "required_equipment": [],
                "preferred_time_window": None,
            },
        ],
        "rooms": [
            {"name": "Room A", "capacity": 6, "equipment": []},
            {"name": "Room B", "capacity": 4, "equipment": []},
        ],
        # 6 slots × 2 rooms = 12 room-slots, 5 meetings × 1 slot = 5 needed → easy
        "time_slots": ["9:00", "9:30", "10:00", "10:30", "11:00", "11:30"],
    }


# ---------------------------------------------------------------------------
# MEDIUM — 8 meetings, 6 slots, 3 rooms, 30+60 min, equipment, some prefs
# ---------------------------------------------------------------------------
# Design: Participant overlaps create conflicts. Equipment narrows room choices.
# 60-min meetings block 2 consecutive slots. Preferred windows add soft constraints.
# A naive agent will hit conflicts; a thoughtful agent can schedule all 8.
#
# Key tensions:
#   - Alice is in 4 meetings → her schedule is the bottleneck
#   - Only Room C has video_conferencing → 2 meetings compete for it
#   - Two 60-min meetings eat 4 room-slots each, tightening the grid

def medium_task() -> dict:
    return {
        "meetings": [
            {
                "id": 1,
                "name": "Engineering Standup",
                "participants": ["Alice", "Bob", "Carol"],
                "priority": "medium",
                "duration_minutes": 30,
                "required_equipment": [],
                "preferred_time_window": "before 10:00",
            },
            {
                "id": 2,
                "name": "Q3 Budget Review",
                "participants": ["Alice", "Dave"],
                "priority": "high",
                "duration_minutes": 60,
                "required_equipment": ["projector"],
                "preferred_time_window": None,
            },
            {
                "id": 3,
                "name": "Client Demo",
                "participants": ["Bob", "Eve"],
                "priority": "high",
                "duration_minutes": 60,
                "required_equipment": ["video_conferencing"],
                "preferred_time_window": "before 11:00",
            },
            {
                "id": 4,
                "name": "UX Feedback Session",
                "participants": ["Carol", "Frank"],
                "priority": "medium",
                "duration_minutes": 30,
                "required_equipment": [],
                "preferred_time_window": None,
            },
            {
                "id": 5,
                "name": "Vendor Negotiation",
                "participants": ["Dave", "Eve"],
                "priority": "high",
                "duration_minutes": 30,
                "required_equipment": ["video_conferencing"],
                "preferred_time_window": None,
            },
            {
                "id": 6,
                "name": "Architecture Discussion",
                "participants": ["Alice", "Frank"],
                "priority": "medium",
                "duration_minutes": 30,
                "required_equipment": ["whiteboard"],
                "preferred_time_window": None,
            },
            {
                "id": 7,
                "name": "Onboarding Check-in",
                "participants": ["Carol"],
                "priority": "low",
                "duration_minutes": 30,
                "required_equipment": [],
                "preferred_time_window": "after 11:00",
            },
            {
                "id": 8,
                "name": "Security Audit Prep",
                "participants": ["Bob", "Dave"],
                "priority": "medium",
                "duration_minutes": 30,
                "required_equipment": [],
                "preferred_time_window": None,
            },
        ],
        "rooms": [
            # Room A: big, has projector + whiteboard (good for reviews, planning)
            {"name": "Room A", "capacity": 8, "equipment": ["projector", "whiteboard"]},
            # Room B: medium, no special equipment (general purpose)
            {"name": "Room B", "capacity": 5, "equipment": []},
            # Room C: small-ish, has video conferencing (the only one!)
            {"name": "Room C", "capacity": 4, "equipment": ["video_conferencing"]},
        ],
        # 6 slots × 3 rooms = 18 room-slots
        # 6 meetings × 1 slot + 2 meetings × 2 slots = 10 room-slots needed → tight but doable
        "time_slots": ["9:00", "9:30", "10:00", "10:30", "11:00", "11:30"],
    }


# ---------------------------------------------------------------------------
# HARD — 15 meetings, 8 slots, 4 rooms, 30/60/90 min, equipment, prefs,
#         OVER-CONSTRAINED (not all meetings can be scheduled)
# ---------------------------------------------------------------------------
# Design: This is intentionally impossible to schedule everything.
# The agent must make priority-based triage decisions.
#
# Why it's over-constrained:
#   - 15 meetings with mixed durations ≈ 22-25 slot-blocks needed
#   - 8 slots × 4 rooms = 32 room-slots available (seems enough...)
#   - BUT participant overlaps are severe: Alice(5), Bob(4), Carol(4), Dave(4)
#     These people can't be in two rooms at once, so the REAL capacity is much lower
#   - Equipment bottlenecks: only 1 room with video_conferencing, 2 with projectors
#   - Expected optimal: ~10-12 meetings scheduled, 3-5 skipped
#   - A good agent skips LOW priority meetings and protects HIGH ones
#
# Key bottleneck people:
#   Alice: meetings 1, 3, 6, 10, 14 (5 meetings, she's the CEO archetype)
#   Bob:   meetings 1, 5, 8, 12      (4 meetings, senior engineer)
#   Dave:  meetings 2, 7, 9, 13      (4 meetings, product lead)

def hard_task() -> dict:
    return {
        "meetings": [
            # --- HIGH PRIORITY (5 meetings) — agent should protect these ---
            {
                "id": 1,
                "name": "Executive Strategy Session",
                "participants": ["Alice", "Bob", "Carol"],
                "priority": "high",
                "duration_minutes": 90,
                "required_equipment": ["projector"],
                "preferred_time_window": "before 10:00",
            },
            {
                "id": 2,
                "name": "Product Launch Review",
                "participants": ["Dave", "Eve", "Frank"],
                "priority": "high",
                "duration_minutes": 60,
                "required_equipment": ["projector"],
                "preferred_time_window": None,
            },
            {
                "id": 3,
                "name": "Investor Update Call",
                "participants": ["Alice", "Grace"],
                "priority": "high",
                "duration_minutes": 30,
                "required_equipment": ["video_conferencing"],
                "preferred_time_window": "before 11:00",
            },
            {
                "id": 4,
                "name": "Critical Bug Triage",
                "participants": ["Hank", "Ivy"],
                "priority": "high",
                "duration_minutes": 30,
                "required_equipment": [],
                "preferred_time_window": None,
            },
            {
                "id": 5,
                "name": "Partnership Agreement Review",
                "participants": ["Bob", "Eve"],
                "priority": "high",
                "duration_minutes": 60,
                "required_equipment": ["video_conferencing"],
                "preferred_time_window": None,
            },
            # --- MEDIUM PRIORITY (5 meetings) — schedule if possible ---
            {
                "id": 6,
                "name": "Engineering Standup",
                "participants": ["Alice", "Hank", "Ivy"],
                "priority": "medium",
                "duration_minutes": 30,
                "required_equipment": [],
                "preferred_time_window": "before 10:00",
            },
            {
                "id": 7,
                "name": "Sprint Retrospective",
                "participants": ["Dave", "Carol", "Frank"],
                "priority": "medium",
                "duration_minutes": 60,
                "required_equipment": ["whiteboard"],
                "preferred_time_window": None,
            },
            {
                "id": 8,
                "name": "Security Audit Planning",
                "participants": ["Bob", "Grace"],
                "priority": "medium",
                "duration_minutes": 30,
                "required_equipment": [],
                "preferred_time_window": None,
            },
            {
                "id": 9,
                "name": "Customer Feedback Debrief",
                "participants": ["Dave", "Eve"],
                "priority": "medium",
                "duration_minutes": 30,
                "required_equipment": [],
                "preferred_time_window": "after 11:00",
            },
            {
                "id": 10,
                "name": "Quarterly OKR Check-in",
                "participants": ["Alice", "Frank"],
                "priority": "medium",
                "duration_minutes": 30,
                "required_equipment": [],
                "preferred_time_window": None,
            },
            # --- LOW PRIORITY (5 meetings) — skip these first under pressure ---
            {
                "id": 11,
                "name": "Lunch & Learn: AI Trends",
                "participants": ["Ivy", "Grace"],
                "priority": "low",
                "duration_minutes": 60,
                "required_equipment": ["projector"],
                "preferred_time_window": "after 11:00",
            },
            {
                "id": 12,
                "name": "Office Space Rearrangement",
                "participants": ["Bob", "Carol"],
                "priority": "low",
                "duration_minutes": 30,
                "required_equipment": [],
                "preferred_time_window": None,
            },
            {
                "id": 13,
                "name": "Intern Project Check-in",
                "participants": ["Dave", "Hank"],
                "priority": "low",
                "duration_minutes": 30,
                "required_equipment": [],
                "preferred_time_window": None,
            },
            {
                "id": 14,
                "name": "Team Social Planning",
                "participants": ["Alice", "Ivy"],
                "priority": "low",
                "duration_minutes": 30,
                "required_equipment": [],
                "preferred_time_window": None,
            },
            {
                "id": 15,
                "name": "Documentation Cleanup",
                "participants": ["Frank", "Grace"],
                "priority": "low",
                "duration_minutes": 30,
                "required_equipment": [],
                "preferred_time_window": None,
            },
        ],
        "rooms": [
            # Room A: large conference room — projector + whiteboard
            {"name": "Room A", "capacity": 10, "equipment": ["projector", "whiteboard"]},
            # Room B: medium room — projector only
            {"name": "Room B", "capacity": 6, "equipment": ["projector"]},
            # Room C: small huddle room — video conferencing (THE bottleneck room)
            {"name": "Room C", "capacity": 4, "equipment": ["video_conferencing"]},
            # Room D: general purpose, no special equipment
            {"name": "Room D", "capacity": 5, "equipment": []},
        ],
        # 8 slots × 4 rooms = 32 room-slots
        # But participant conflicts and equipment bottlenecks make it ~20 effective
        "time_slots": [
            "9:00", "9:30", "10:00", "10:30",
            "11:00", "11:30", "12:00", "12:30",
        ],
    }
