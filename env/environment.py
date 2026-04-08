"""
Meeting Scheduler RL Environment.

Core class: SchedulerEnv
  - reset(task_name)  → SchedulerObservation  (start a new episode)
  - step(action)      → (SchedulerObservation, float, bool, dict)  (process one scheduling decision)
  - state()           → SchedulerState  (episode metadata)

Internal data structures:
  - calendar_grid: {timeslot: {room_name: meeting_name|None}}
    Tracks which rooms are booked at which times.

  - participant_slots: {person_name: set(timeslots)}
    Tracks when each person is busy. Used for conflict detection.
    A 60-min meeting at 10:00 adds BOTH "10:00" and "10:30" to each participant's set.

  - assignments: list[Assignment]
    Full history of every decision (schedule or skip) for the grader.
"""

import uuid
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    Assignment,
    Meeting,
    Room,
    SchedulerAction,
    SchedulerObservation,
    SchedulerState,
)
from .tasks import get_task


class SchedulerEnv:
    def __init__(self):
        # Task data (loaded on reset)
        self.meetings: List[Meeting] = []
        self.rooms: List[Room] = []
        self.time_slots: List[str] = []

        # Episode state
        self.calendar_grid: Dict[str, Dict[str, Optional[str]]] = {}
        self.participant_slots: Dict[str, Set[str]] = {}
        self.assignments: List[Assignment] = []
        self.step_idx: int = 0
        self.done: bool = False
        self.task_name: str = ""
        self.episode_id: str = ""

    # ------------------------------------------------------------------
    # RESET — Start a new episode
    # ------------------------------------------------------------------
    def reset(self, task_name: str = "easy") -> SchedulerObservation:
        """
        Load a task and initialize a fresh calendar.

        Returns the first observation (the first meeting to schedule).
        """
        self.task_name = task_name
        self.episode_id = str(uuid.uuid4())

        # Load task data and convert dicts to Pydantic models
        task_data = get_task(task_name)
        self.meetings = [Meeting(**m) for m in task_data["meetings"]]
        self.rooms = [Room(**r) for r in task_data["rooms"]]
        self.time_slots = task_data["time_slots"]

        # Initialize empty calendar: every slot × every room = None
        self.calendar_grid = {}
        for slot in self.time_slots:
            self.calendar_grid[slot] = {}
            for room in self.rooms:
                self.calendar_grid[slot][room.name] = None

        # No one is busy yet
        self.participant_slots = {}

        # Reset episode tracking
        self.assignments = []
        self.step_idx = 0
        self.done = False

        return self._build_observation(step_reward=None, message="New episode started. Schedule the first meeting.")

    # ------------------------------------------------------------------
    # STEP — Process one scheduling decision
    # ------------------------------------------------------------------
    def step(self, action: SchedulerAction) -> Tuple[SchedulerObservation, float, bool, dict]:
        """
        Process the agent's action for the current meeting.

        Returns: (observation, reward, done, info)

        The reward logic:
          Valid assignment:   +1.0 base + priority bonus + preference bonus
          Invalid assignment: -1.0 (conflict, room too small, etc.)
          Smart skip:          0.0 (meeting genuinely couldn't fit)
          Lazy skip:          -0.3 (meeting could have been scheduled)
        """
        if self.done:
            obs = self._build_observation(step_reward=0.0, message="Episode already finished.")
            return obs, 0.0, True, {"error": "Episode already finished"}

        current_meeting = self.meetings[self.step_idx]
        reward = 0.0
        info = {}

        # ----- SKIP -----
        if action.timeslot.lower().strip() == "skip":
            was_schedulable = self._is_meeting_schedulable(current_meeting)
            if was_schedulable:
                reward = -0.3
                message = f"Skipped '{current_meeting.name}', but it COULD have been scheduled. Penalty: -0.3"
            else:
                reward = 0.0
                message = f"Skipped '{current_meeting.name}'. Correct — no valid slot existed."

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
            # Validate the proposed assignment
            is_valid, reason = self._validate_action(action, current_meeting)

            if is_valid:
                # Calculate reward
                reward = 1.0  # base reward for valid assignment

                # Priority bonus
                priority_bonus = {"high": 0.5, "medium": 0.3, "low": 0.1}
                reward += priority_bonus.get(current_meeting.priority, 0.0)

                # Preferred time bonus (only if the meeting HAS a preference)
                preferred_met = False
                if current_meeting.preferred_time_window:
                    preferred_met = self._check_preferred_time(
                        action.timeslot, current_meeting.preferred_time_window
                    )
                    if preferred_met:
                        reward += 0.2

                # Book it: mark ALL required slots in the calendar and participant schedules
                required_slots = self._get_required_slots(action.timeslot, current_meeting.duration_minutes)
                for slot in required_slots:
                    self.calendar_grid[slot][action.room] = current_meeting.name
                    for person in current_meeting.participants:
                        if person not in self.participant_slots:
                            self.participant_slots[person] = set()
                        self.participant_slots[person].add(slot)

                message = (
                    f"Scheduled '{current_meeting.name}' at {action.timeslot} in {action.room}. "
                    f"Reward: +{reward:.1f}"
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
                # Invalid assignment — conflict, room too small, etc.
                reward = -1.0
                message = f"Invalid: {reason}. Penalty: -1.0"

                self.assignments.append(Assignment(
                    meeting=current_meeting,
                    timeslot=action.timeslot,
                    room=action.room,
                    valid=False,
                    skipped=False,
                    was_schedulable=True,  # We don't know — they tried and failed
                    preferred_time_met=False,
                ))

        # Advance to next meeting
        self.step_idx += 1
        if self.step_idx >= len(self.meetings):
            self.done = True
            message += " All meetings processed — episode complete."

        obs = self._build_observation(step_reward=reward, message=message)
        return obs, reward, self.done, info

    # ------------------------------------------------------------------
    # STATE — Episode metadata
    # ------------------------------------------------------------------
    def state(self) -> SchedulerState:
        valid_count = sum(1 for a in self.assignments if a.valid)
        return SchedulerState(
            task_name=self.task_name,
            step_count=self.step_idx,
            episode_id=self.episode_id,
            assignments_made=valid_count,
            total_meetings=len(self.meetings),
            done=self.done,
        )

    # ==================================================================
    # INTERNAL METHODS
    # ==================================================================

    def _validate_action(self, action: SchedulerAction, meeting: Meeting) -> Tuple[bool, str]:
        """
        Check ALL constraints for a proposed assignment.

        Returns (is_valid, reason_string).
        Checks in order of cheapest-to-compute first:
          1. Is the timeslot valid?
          2. Is the room real?
          3. Are there enough consecutive slots for the meeting duration?
          4. Does the room have required equipment?
          5. Does the room have enough capacity?
          6. Are all required time slots free in this room?
          7. Are all participants free during all required time slots?
        """
        timeslot = action.timeslot.strip()
        room_name = action.room.strip() if action.room else ""

        # 1. Valid timeslot?
        if timeslot not in self.time_slots:
            return False, f"'{timeslot}' is not a valid time slot. Valid: {self.time_slots}"

        # 2. Room exists?
        room = self._find_room(room_name)
        if room is None:
            valid_names = [r.name for r in self.rooms]
            return False, f"'{room_name}' is not a valid room. Valid: {valid_names}"

        # 3. Enough consecutive slots?
        required_slots = self._get_required_slots(timeslot, meeting.duration_minutes)
        if required_slots is None:
            return False, (
                f"Not enough consecutive time slots for {meeting.duration_minutes} min "
                f"starting at {timeslot}. Need {meeting.duration_minutes // 30} slots."
            )

        # 4. Equipment?
        for equip in meeting.required_equipment:
            if equip not in room.equipment:
                return False, (
                    f"Room '{room_name}' lacks required equipment: '{equip}'. "
                    f"Room has: {room.equipment}"
                )

        # 5. Capacity?
        if len(meeting.participants) > room.capacity:
            return False, (
                f"Room '{room_name}' has capacity {room.capacity}, "
                f"but meeting has {len(meeting.participants)} participants."
            )

        # 6. Room availability across ALL required slots?
        for slot in required_slots:
            occupant = self.calendar_grid[slot][room_name]
            if occupant is not None:
                return False, (
                    f"Room '{room_name}' is already booked at {slot} "
                    f"for '{occupant}'."
                )

        # 7. Participant conflicts across ALL required slots?
        for slot in required_slots:
            for person in meeting.participants:
                if person in self.participant_slots and slot in self.participant_slots[person]:
                    # Find what meeting is blocking them (for a helpful error message)
                    blocking = self._find_blocking_meeting(person, slot)
                    return False, (
                        f"Participant '{person}' is already busy at {slot}"
                        f"{f' ({blocking})' if blocking else ''}."
                    )

        return True, "OK"

    def _get_required_slots(self, start_slot: str, duration_minutes: int) -> Optional[List[str]]:
        """
        Given a start time and duration, return the list of consecutive slots needed.

        A 60-min meeting at "10:00" → ["10:00", "10:30"]
        A 90-min meeting at "9:00" → ["9:00", "9:30", "10:00"]

        Returns None if there aren't enough consecutive slots remaining.
        """
        num_slots = duration_minutes // 30

        if start_slot not in self.time_slots:
            return None

        start_idx = self.time_slots.index(start_slot)

        # Check: are there enough slots left after start_idx?
        if start_idx + num_slots > len(self.time_slots):
            return None

        return self.time_slots[start_idx : start_idx + num_slots]

    def _find_room(self, room_name: str) -> Optional[Room]:
        """Look up a room by name."""
        for room in self.rooms:
            if room.name == room_name:
                return room
        return None

    def _find_blocking_meeting(self, person: str, slot: str) -> Optional[str]:
        """Find which meeting is blocking a person at a given slot (for error messages)."""
        for room_name, occupant in self.calendar_grid.get(slot, {}).items():
            if occupant is not None:
                # Check if this person is in that meeting
                for assignment in self.assignments:
                    if (assignment.valid
                            and assignment.meeting.name == occupant
                            and person in assignment.meeting.participants):
                        return occupant
        return None

    def _is_meeting_schedulable(self, meeting: Meeting) -> bool:
        """
        Brute-force check: can this meeting fit in ANY (timeslot, room) combination?

        Used when the agent skips — to determine if skipping was smart or lazy.
        With max 8 slots × 4 rooms = 32 combinations, this is trivially fast.
        """
        for slot in self.time_slots:
            for room in self.rooms:
                test_action = SchedulerAction(timeslot=slot, room=room.name)
                is_valid, _ = self._validate_action(test_action, meeting)
                if is_valid:
                    return True
        return False

    def _check_preferred_time(self, timeslot: str, preference: Optional[str]) -> bool:
        """
        Check if a timeslot satisfies a preferred_time_window.

        Supports: "before HH:MM" and "after HH:MM".
        Returns True if no preference is set (vacuously satisfied).
        Note: caller should only award bonus when preference is not None.
        """
        if not preference:
            return False  # No preference set → caller won't award bonus

        preference = preference.strip().lower()

        # Parse "before HH:MM" or "after HH:MM"
        try:
            if preference.startswith("before "):
                threshold = preference.replace("before ", "")
                return self._time_to_minutes(timeslot) < self._time_to_minutes(threshold)
            elif preference.startswith("after "):
                threshold = preference.replace("after ", "")
                return self._time_to_minutes(timeslot) >= self._time_to_minutes(threshold)
        except (ValueError, IndexError):
            return False

        return False

    @staticmethod
    def _time_to_minutes(time_str: str) -> int:
        """Convert "HH:MM" to minutes since midnight for comparison."""
        parts = time_str.strip().split(":")
        return int(parts[0]) * 60 + int(parts[1])

    def _build_observation(self, step_reward: Optional[float], message: str) -> SchedulerObservation:
        """Build the full observation including the human-readable text summary."""
        current_meeting = None
        if not self.done and self.step_idx < len(self.meetings):
            current_meeting = self.meetings[self.step_idx]

        # Build serializable calendar grid (meeting name or None)
        grid = {}
        for slot in self.time_slots:
            grid[slot] = {}
            for room in self.rooms:
                grid[slot][room.name] = self.calendar_grid.get(slot, {}).get(room.name)

        # List of successfully scheduled meeting names
        scheduled_names = [a.meeting.name for a in self.assignments if a.valid]

        remaining = len(self.meetings) - self.step_idx
        if self.done:
            remaining = 0

        return SchedulerObservation(
            current_meeting=current_meeting,
            available_rooms=self.rooms,
            calendar_grid=grid,
            remaining_count=remaining,
            scheduled_so_far=scheduled_names,
            text_summary=self._build_text_summary(current_meeting),
            step_reward=step_reward,
            message=message,
            done=self.done,
        )

    def _build_text_summary(self, current_meeting: Optional[Meeting]) -> str:
        """
        Build the human-readable calendar view that gets fed to the LLM.

        This is the most important output for inference.py — the LLM reads THIS,
        not the JSON fields. Format:

        === CURRENT MEETING TO SCHEDULE ===
        Name: Q3 Budget Review
        Priority: HIGH
        Duration: 60 min (2 slots)
        Participants: Alice, Dave
        Required equipment: projector
        Preferred time: before 11:00

        === AVAILABLE ROOMS ===
        - Room A (capacity: 8, equipment: projector, whiteboard)
        - Room B (capacity: 5, equipment: none)

        === CALENDAR ===
                    | Room A         | Room B         |
        9:00        | [Standup]      | ---            |
        9:30        | ---            | ---            |
        ...

        === STATUS ===
        Scheduled: 3/8 meetings | Remaining: 5
        """
        lines = []

        # --- Current meeting ---
        if current_meeting:
            lines.append("=== CURRENT MEETING TO SCHEDULE ===")
            lines.append(f"  Name: {current_meeting.name}")
            lines.append(f"  Priority: {current_meeting.priority.upper()}")
            slots_needed = current_meeting.duration_minutes // 30
            lines.append(f"  Duration: {current_meeting.duration_minutes} min ({slots_needed} slot{'s' if slots_needed > 1 else ''})")
            lines.append(f"  Participants: {', '.join(current_meeting.participants)}")
            if current_meeting.required_equipment:
                lines.append(f"  Required equipment: {', '.join(current_meeting.required_equipment)}")
            if current_meeting.preferred_time_window:
                lines.append(f"  Preferred time: {current_meeting.preferred_time_window}")
            lines.append("")
        else:
            lines.append("=== ALL MEETINGS PROCESSED ===")
            lines.append("")

        # --- Rooms ---
        lines.append("=== AVAILABLE ROOMS ===")
        for room in self.rooms:
            equip = ", ".join(room.equipment) if room.equipment else "none"
            lines.append(f"  - {room.name} (capacity: {room.capacity}, equipment: {equip})")
        lines.append("")

        # --- Calendar grid ---
        lines.append("=== CALENDAR ===")

        # Header row
        room_names = [r.name for r in self.rooms]
        col_width = 20
        header = "  Time".ljust(12)
        for rn in room_names:
            header += f"| {rn}".ljust(col_width)
        lines.append(header)
        lines.append("  " + "-" * (10 + col_width * len(room_names)))

        # Each timeslot row
        for slot in self.time_slots:
            row = f"  {slot}".ljust(12)
            for rn in room_names:
                occupant = self.calendar_grid.get(slot, {}).get(rn)
                if occupant:
                    cell = f"[{occupant}]"
                else:
                    cell = "---"
                # Truncate long meeting names to fit
                if len(cell) > col_width - 4:
                    cell = cell[: col_width - 7] + "...]"
                row += f"| {cell}".ljust(col_width)
            lines.append(row)
        lines.append("")

        # --- Participant busy times (helps the LLM spot conflicts) ---
        if self.participant_slots:
            lines.append("=== PARTICIPANT SCHEDULES ===")
            for person in sorted(self.participant_slots.keys()):
                busy = sorted(self.participant_slots[person],
                              key=lambda s: self._time_to_minutes(s))
                lines.append(f"  {person}: busy at {', '.join(busy)}")
            lines.append("")

        # --- Status ---
        valid_count = sum(1 for a in self.assignments if a.valid)
        total = len(self.meetings)
        remaining = total - self.step_idx if not self.done else 0
        lines.append("=== STATUS ===")
        lines.append(f"  Scheduled: {valid_count}/{total} meetings | Remaining: {remaining}")

        return "\n".join(lines)
