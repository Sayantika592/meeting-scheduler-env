"""
Pydantic models for the Meeting Scheduler RL Environment.

Three top-level models required by OpenEnv spec:
  - SchedulerAction: what the agent sends each step
  - SchedulerObservation: what the agent sees each step
  - SchedulerState: episode metadata (GET /state)

Plus supporting models for structured data:
  - Meeting: a meeting to be scheduled
  - Room: a room with capacity and equipment
  - Assignment: record of a scheduling decision
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Supporting models (nested inside observations / tasks)
# ---------------------------------------------------------------------------

class Meeting(BaseModel):
    """A meeting that needs to be scheduled."""
    id: int
    name: str
    participants: List[str]
    priority: str = Field(..., pattern=r"^(low|medium|high)$")
    duration_minutes: int = Field(..., description="30, 60, or 90")
    required_equipment: List[str] = Field(default_factory=list)
    preferred_time_window: Optional[str] = Field(
        default=None,
        description='e.g. "before 11:00" or "after 10:00"'
    )


class Room(BaseModel):
    """A room available for booking."""
    name: str
    capacity: int
    equipment: List[str] = Field(default_factory=list)


class Assignment(BaseModel):
    """Record of one scheduling decision (used internally + in grader)."""
    meeting: Meeting
    timeslot: Optional[str] = None  # None if skipped
    room: Optional[str] = None      # None if skipped
    valid: bool = False
    skipped: bool = False
    was_schedulable: bool = True     # Could it have been scheduled?
    preferred_time_met: bool = False


# ---------------------------------------------------------------------------
# Top-level API models (required by OpenEnv spec)
# ---------------------------------------------------------------------------

class SchedulerAction(BaseModel):
    """
    What the agent sends each step.

    To schedule: {"timeslot": "10:00", "room": "Room A"}
    To skip:     {"timeslot": "skip"}
    """
    timeslot: str = Field(..., description='A time like "9:00" or "skip"')
    room: Optional[str] = Field(
        default=None,
        description="Room name. Required when timeslot is not 'skip'."
    )


class SchedulerObservation(BaseModel):
    """
    What the agent sees each step.

    Includes both structured fields (for programmatic use) and a
    text_summary string (the human-readable view fed to the LLM).
    """
    current_meeting: Optional[Meeting] = Field(
        default=None,
        description="The meeting to schedule this step. None if episode is done."
    )
    available_rooms: List[Room] = Field(default_factory=list)
    calendar_grid: Dict[str, Dict[str, Optional[str]]] = Field(
        default_factory=dict,
        description='timeslot -> room_name -> meeting_name or None'
    )
    remaining_count: int = 0
    scheduled_so_far: List[str] = Field(
        default_factory=list,
        description="Names of successfully scheduled meetings"
    )
    text_summary: str = Field(
        default="",
        description="Human-readable calendar state for the LLM prompt"
    )
    step_reward: Optional[float] = Field(
        default=None,
        description="Reward received for the last action"
    )
    message: str = Field(
        default="",
        description="Feedback message about the last action"
    )
    done: bool = False


class SchedulerState(BaseModel):
    """
    Episode metadata returned by GET /state.
    Lightweight — doesn't contain the full calendar.
    """
    task_name: str = ""
    step_count: int = 0
    episode_id: str = ""
    assignments_made: int = 0
    total_meetings: int = 0
    done: bool = False
