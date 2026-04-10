"""
Meeting Scheduler RL Environment — OpenEnv compliant.

Subclasses openenv.core.env_server.interfaces.Environment so that
create_app() can wire up /reset, /step, /state, /schema, /ws automatically.

KEY DIFFERENCES FROM RAW FASTAPI VERSION:
  - reset() returns SchedulerObservation (not a dict)
  - step() returns SchedulerObservation (not a tuple)
    → reward and done are FIELDS on the Observation object
  - state is a @property returning State
  - The class is generic: Environment[SchedulerAction, SchedulerObservation, State]
"""

import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Root-level OpenEnv models (Action/Observation subclasses)
try:
    from ..models import (
        MeetingInfo,
        RoomInfo,
        SchedulerAction,
        SchedulerObservation,
    )
except ImportError:
    from models import (
        MeetingInfo,
        RoomInfo,
        SchedulerAction,
        SchedulerObservation,
    )

# Internal models (not exposed via OpenEnv API)
from .internal_models import Assignment, Meeting, Room
from .tasks import get_task
from .graders import grade_episode


class SchedulerEnvironment(Environment):
    """
    Meeting Scheduler RL Environment.

    The agent receives meetings one at a time and must assign each to a
    (timeslot, room) pair or skip it. Rewards are given per-step based on
    validity, priority, and preference satisfaction. At episode end, a
    grader produces a final score in [0.0, 1.0].
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self.meetings: List[Meeting] = []
        self.rooms: List[Room] = []
        self.time_slots: List[str] = []

        self.calendar_grid: Dict[str, Dict[str, Optional[str]]] = {}
        self.participant_slots: Dict[str, Set[str]] = {}
        self.assignments: List[Assignment] = []
        self.step_idx: int = 0
        self._done: bool = False
        self.task_name: str = ""

        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: str = "easy",
        **kwargs: Any,
    ) -> SchedulerObservation:
        """
        Start a new episode.

        Args:
            seed: ignored (deterministic tasks)
            episode_id: optional custom episode id
            task: "easy", "medium", or "hard"

        Returns:
            SchedulerObservation with the first meeting to schedule.
        """
        self.task_name = task
        eid = episode_id or str(uuid.uuid4())
        self._state = State(episode_id=eid, step_count=0)

        task_data = get_task(task)
        self.meetings = [Meeting(**m) for m in task_data["meetings"]]
        self.rooms = [Room(**r) for r in task_data["rooms"]]
        self.time_slots = task_data["time_slots"]

        self.calendar_grid = {}
        for slot in self.time_slots:
            self.calendar_grid[slot] = {}
            for room in self.rooms:
                self.calendar_grid[slot][room.name] = None

        self.participant_slots = {}
        self.assignments = []
        self.step_idx = 0
        self._done = False

        return self._build_observation(
            reward=0.01,
            message="New episode started. Schedule the first meeting.",
        )

    # ------------------------------------------------------------------
    # STEP — returns Observation (reward & done are fields on it)
    # ------------------------------------------------------------------
    def step(
        self,
        action: SchedulerAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SchedulerObservation:
        """
        Process one scheduling decision.

        Returns a SchedulerObservation with reward and done set on the object.
        """
        if self._done:
            return self._build_observation(
                reward=0.01,
                message="Episode already finished.",
            )

        self._state.step_count += 1
        current_meeting = self.meetings[self.step_idx]
        reward = 0.01

        # ----- SKIP -----
        if action.timeslot.lower().strip() == "skip":
            was_schedulable = self._is_meeting_schedulable(current_meeting)
            if was_schedulable:
                reward = -0.3
                message = (
                    f"Skipped '{current_meeting.name}', but it COULD have been "
                    f"scheduled. Penalty: -0.3"
                )
            else:
                reward = 0.01
                message = (
                    f"Skipped '{current_meeting.name}'. Correct — no valid slot existed."
                )

            self.assignments.append(Assignment(
                meeting=current_meeting,
                timeslot=None,
                room=None,
                valid=False,
                skipped=True,
                was_schedulable=was_schedulable,
                preferred_time_met=False,
            ))

        # ----- SCHEDULE -----
        else:
            is_valid, reason = self._validate_action(action, current_meeting)

            if is_valid:
                reward = 1.0
                priority_bonus = {"high": 0.5, "medium": 0.3, "low": 0.1}
                reward += priority_bonus.get(current_meeting.priority, 0.0)

                preferred_met = False
                if current_meeting.preferred_time_window:
                    preferred_met = self._check_preferred_time(
                        action.timeslot, current_meeting.preferred_time_window
                    )
                    if preferred_met:
                        reward += 0.2

                required_slots = self._get_required_slots(
                    action.timeslot, current_meeting.duration_minutes
                )
                for slot in required_slots:
                    self.calendar_grid[slot][action.room] = current_meeting.name
                    for person in current_meeting.participants:
                        if person not in self.participant_slots:
                            self.participant_slots[person] = set()
                        self.participant_slots[person].add(slot)

                message = (
                    f"Scheduled '{current_meeting.name}' at {action.timeslot} "
                    f"in {action.room}. Reward: +{reward:.1f}"
                )

                self.assignments.append(Assignment(
                    meeting=current_meeting,
                    timeslot=action.timeslot,
                    room=action.room,
                    valid=True,
                    skipped=False,
                    was_schedulable=True,
                    preferred_time_met=preferred_met,
                ))

            else:
                reward = -1.0
                message = f"Invalid: {reason}. Penalty: -1.0"

                self.assignments.append(Assignment(
                    meeting=current_meeting,
                    timeslot=action.timeslot,
                    room=action.room,
                    valid=False,
                    skipped=False,
                    was_schedulable=True,
                    preferred_time_met=False,
                ))

        # Advance
        self.step_idx += 1
        if self.step_idx >= len(self.meetings):
            self._done = True
            message += " All meetings processed — episode complete."

        reward = max(0.01, min(0.99, (reward + 1.0) / 2.7))
        return self._build_observation(reward=reward, message=message)

    # ------------------------------------------------------------------
    # STATE — property (not a method) per OpenEnv spec
    # ------------------------------------------------------------------
    @property
    def state(self) -> State:
        return self._state

    # ==================================================================
    # INTERNAL METHODS (unchanged from original)
    # ==================================================================

    def _validate_action(
        self, action: SchedulerAction, meeting: Meeting
    ) -> Tuple[bool, str]:
        timeslot = action.timeslot.strip()
        room_name = action.room.strip() if action.room else ""

        if timeslot not in self.time_slots:
            return False, (
                f"'{timeslot}' is not a valid time slot. Valid: {self.time_slots}"
            )

        room = self._find_room(room_name)
        if room is None:
            valid_names = [r.name for r in self.rooms]
            return False, (
                f"'{room_name}' is not a valid room. Valid: {valid_names}"
            )

        required_slots = self._get_required_slots(timeslot, meeting.duration_minutes)
        if required_slots is None:
            return False, (
                f"Not enough consecutive time slots for {meeting.duration_minutes} min "
                f"starting at {timeslot}. Need {meeting.duration_minutes // 30} slots."
            )

        for equip in meeting.required_equipment:
            if equip not in room.equipment:
                return False, (
                    f"Room '{room_name}' lacks required equipment: '{equip}'. "
                    f"Room has: {room.equipment}"
                )

        if len(meeting.participants) > room.capacity:
            return False, (
                f"Room '{room_name}' has capacity {room.capacity}, "
                f"but meeting has {len(meeting.participants)} participants."
            )

        for slot in required_slots:
            occupant = self.calendar_grid[slot][room_name]
            if occupant is not None:
                return False, (
                    f"Room '{room_name}' is already booked at {slot} "
                    f"for '{occupant}'."
                )

        for slot in required_slots:
            for person in meeting.participants:
                if (
                    person in self.participant_slots
                    and slot in self.participant_slots[person]
                ):
                    blocking = self._find_blocking_meeting(person, slot)
                    return False, (
                        f"Participant '{person}' is already busy at {slot}"
                        f"{f' ({blocking})' if blocking else ''}."
                    )

        return True, "OK"

    def _get_required_slots(
        self, start_slot: str, duration_minutes: int
    ) -> Optional[List[str]]:
        num_slots = duration_minutes // 30
        if start_slot not in self.time_slots:
            return None
        start_idx = self.time_slots.index(start_slot)
        if start_idx + num_slots > len(self.time_slots):
            return None
        return self.time_slots[start_idx : start_idx + num_slots]

    def _find_room(self, room_name: str) -> Optional[Room]:
        for room in self.rooms:
            if room.name == room_name:
                return room
        return None

    def _find_blocking_meeting(self, person: str, slot: str) -> Optional[str]:
        for room_name, occupant in self.calendar_grid.get(slot, {}).items():
            if occupant is not None:
                for assignment in self.assignments:
                    if (
                        assignment.valid
                        and assignment.meeting.name == occupant
                        and person in assignment.meeting.participants
                    ):
                        return occupant
        return None

    def _is_meeting_schedulable(self, meeting: Meeting) -> bool:
        for slot in self.time_slots:
            for room in self.rooms:
                test_action = SchedulerAction(timeslot=slot, room=room.name)
                is_valid, _ = self._validate_action(test_action, meeting)
                if is_valid:
                    return True
        return False

    def _check_preferred_time(
        self, timeslot: str, preference: Optional[str]
    ) -> bool:
        if not preference:
            return False
        preference = preference.strip().lower()
        try:
            if preference.startswith("before "):
                threshold = preference.replace("before ", "")
                return self._time_to_minutes(timeslot) < self._time_to_minutes(
                    threshold
                )
            elif preference.startswith("after "):
                threshold = preference.replace("after ", "")
                return self._time_to_minutes(timeslot) >= self._time_to_minutes(
                    threshold
                )
        except (ValueError, IndexError):
            return False
        return False

    @staticmethod
    def _time_to_minutes(time_str: str) -> int:
        parts = time_str.strip().split(":")
        return int(parts[0]) * 60 + int(parts[1])

    def _build_observation(
        self, reward: float, message: str
    ) -> SchedulerObservation:
        current_meeting = None
        current_meeting_info = None
        if not self._done and self.step_idx < len(self.meetings):
            current_meeting = self.meetings[self.step_idx]
            current_meeting_info = MeetingInfo(
                id=current_meeting.id,
                name=current_meeting.name,
                participants=current_meeting.participants,
                priority=current_meeting.priority,
                duration_minutes=current_meeting.duration_minutes,
                required_equipment=current_meeting.required_equipment,
                preferred_time_window=current_meeting.preferred_time_window,
            )

        room_infos = [
            RoomInfo(name=r.name, capacity=r.capacity, equipment=r.equipment)
            for r in self.rooms
        ]

        grid = {}
        for slot in self.time_slots:
            grid[slot] = {}
            for room in self.rooms:
                grid[slot][room.name] = self.calendar_grid.get(slot, {}).get(
                    room.name
                )

        scheduled_names = [a.meeting.name for a in self.assignments if a.valid]
        remaining = len(self.meetings) - self.step_idx if not self._done else 0

        return SchedulerObservation(
            current_meeting=current_meeting_info,
            available_rooms=room_infos,
            calendar_grid=grid,
            remaining_count=remaining,
            scheduled_so_far=scheduled_names,
            text_summary=self._build_text_summary(current_meeting),
            message=message,
            # OpenEnv Observation base fields:
            done=self._done,
            reward=reward,
            metadata={
                "task": self.task_name,
                "step": self._state.step_count,
                "final_score": grade_episode(self) if self._done else None,
            },
        )

    def _build_text_summary(self, current_meeting: Optional[Meeting]) -> str:
        lines = []

        if current_meeting:
            lines.append("=== CURRENT MEETING TO SCHEDULE ===")
            lines.append(f"  Name: {current_meeting.name}")
            lines.append(f"  Priority: {current_meeting.priority.upper()}")
            slots_needed = current_meeting.duration_minutes // 30
            lines.append(
                f"  Duration: {current_meeting.duration_minutes} min "
                f"({slots_needed} slot{'s' if slots_needed > 1 else ''})"
            )
            lines.append(
                f"  Participants: {', '.join(current_meeting.participants)}"
            )
            if current_meeting.required_equipment:
                lines.append(
                    f"  Required equipment: "
                    f"{', '.join(current_meeting.required_equipment)}"
                )
            if current_meeting.preferred_time_window:
                lines.append(
                    f"  Preferred time: {current_meeting.preferred_time_window}"
                )
            lines.append("")
        else:
            lines.append("=== ALL MEETINGS PROCESSED ===")
            lines.append("")

        lines.append("=== AVAILABLE ROOMS ===")
        for room in self.rooms:
            equip = ", ".join(room.equipment) if room.equipment else "none"
            lines.append(
                f"  - {room.name} (capacity: {room.capacity}, equipment: {equip})"
            )
        lines.append("")

        lines.append("=== CALENDAR ===")
        room_names = [r.name for r in self.rooms]
        col_width = 20
        header = "  Time".ljust(12)
        for rn in room_names:
            header += f"| {rn}".ljust(col_width)
        lines.append(header)
        lines.append("  " + "-" * (10 + col_width * len(room_names)))

        for slot in self.time_slots:
            row = f"  {slot}".ljust(12)
            for rn in room_names:
                occupant = self.calendar_grid.get(slot, {}).get(rn)
                if occupant:
                    cell = f"[{occupant}]"
                else:
                    cell = "---"
                if len(cell) > col_width - 4:
                    cell = cell[: col_width - 7] + "...]"
                row += f"| {cell}".ljust(col_width)
            lines.append(row)
        lines.append("")

        if self.participant_slots:
            lines.append("=== PARTICIPANT SCHEDULES ===")
            for person in sorted(self.participant_slots.keys()):
                busy = sorted(
                    self.participant_slots[person],
                    key=lambda s: self._time_to_minutes(s),
                )
                lines.append(f"  {person}: busy at {', '.join(busy)}")
            lines.append("")

        valid_count = sum(1 for a in self.assignments if a.valid)
        total = len(self.meetings)
        remaining = total - self.step_idx if not self._done else 0
        lines.append("=== STATUS ===")
        lines.append(
            f"  Scheduled: {valid_count}/{total} meetings | Remaining: {remaining}"
        )

        return "\n".join(lines)
