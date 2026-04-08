"""
Internal data models for environment logic.

These are NOT exposed via the OpenEnv API — they're used internally by
the environment to track state and by the grader to score episodes.

The root-level models.py has the OpenEnv-compliant Action/Observation models.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


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
        description='e.g. "before 11:00" or "after 10:00"',
    )


class Room(BaseModel):
    """A room available for booking."""
    name: str
    capacity: int
    equipment: List[str] = Field(default_factory=list)


class Assignment(BaseModel):
    """Record of one scheduling decision (used internally + in grader)."""
    meeting: Meeting
    timeslot: Optional[str] = None
    room: Optional[str] = None
    valid: bool = False
    skipped: bool = False
    was_schedulable: bool = True
    preferred_time_met: bool = False
