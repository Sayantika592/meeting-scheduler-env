"""
OpenEnv-compliant Pydantic models for the Meeting Scheduler Environment.

These models inherit from openenv.core base classes (Action, Observation)
which is REQUIRED for `openenv validate` to pass.

Action/Observation base classes provide:
  - Action: metadata field, extra="forbid"
  - Observation: done, reward, metadata fields, extra="forbid"
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# Supporting models (nested inside observations — NOT top-level OpenEnv types)
# These are plain BaseModel because they're nested data, not API contracts.
# ---------------------------------------------------------------------------

from pydantic import BaseModel


class MeetingInfo(BaseModel):
    """A meeting to be scheduled (nested in observation, not an OpenEnv type)."""
    id: int
    name: str
    participants: List[str]
    priority: str
    duration_minutes: int
    required_equipment: List[str] = Field(default_factory=list)
    preferred_time_window: Optional[str] = None


class RoomInfo(BaseModel):
    """A room available for booking (nested in observation)."""
    name: str
    capacity: int
    equipment: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# OpenEnv API models — these inherit from the framework base classes
# ---------------------------------------------------------------------------

class SchedulerAction(Action):
    """
    What the agent sends each step.

    To schedule: {"timeslot": "10:00", "room": "Room A"}
    To skip:     {"timeslot": "skip"}
    """
    timeslot: str = Field(..., description='A time like "9:00" or "skip"')
    room: Optional[str] = Field(
        default=None,
        description="Room name. Required when timeslot is not 'skip'.",
    )


class SchedulerObservation(Observation):
    """
    What the agent sees each step.

    Inherits from Observation which provides:
      - done: bool (whether episode is complete)
      - reward: Optional[float] (reward from last action)
      - metadata: Dict[str, Any]
    """
    current_meeting: Optional[MeetingInfo] = Field(
        default=None,
        description="The meeting to schedule this step. None if episode is done.",
    )
    available_rooms: List[RoomInfo] = Field(default_factory=list)
    calendar_grid: Dict[str, Dict[str, Optional[str]]] = Field(
        default_factory=dict,
        description="timeslot -> room_name -> meeting_name or None",
    )
    remaining_count: int = 0
    scheduled_so_far: List[str] = Field(
        default_factory=list,
        description="Names of successfully scheduled meetings",
    )
    text_summary: str = Field(
        default="",
        description="Human-readable calendar state for the LLM prompt",
    )
    message: str = Field(
        default="",
        description="Feedback message about the last action",
    )
