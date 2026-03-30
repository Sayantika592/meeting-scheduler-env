import random
from copy import deepcopy


class SchedulerEnv:
    def __init__(self, task_data):
        self.task_data = task_data
        self.meetings = []
        self.calendar = {}
        self.step_idx = 0
        self.done = False

    # ---------------- RESET ----------------
    def reset(self):
        self.assignments = []
        self.meetings = deepcopy(self.task_data["meetings"])
        random.shuffle(self.meetings)

        self.calendar = deepcopy(self.task_data["calendar"])
        self.step_idx = 0
        self.done = False

        return self._get_obs()

    # ---------------- STEP ----------------
    def step(self, action):
        if self.done:
            raise Exception("Episode already finished")

        meeting = self.meetings[self.step_idx]
        timeslot = action.get("timeslot")

        reward = 0

        # --- Skip action ---
        if timeslot == "skip":
            reward -= 0.5
            self.assignments.append({
                "meeting": meeting,
                "timeslot": None,
                "valid": False
            })

        else:
            participants = meeting["participants"]

            # check conflict
            conflict = any(
                p in self.calendar[timeslot] for p in participants
            )

            slot_size = len(self.calendar[timeslot])

            if conflict:
                reward -= 2
                valid = False

            elif slot_size >= 2:
                # overcrowded slot → still allowed but BAD
                self.calendar[timeslot].extend(participants)
                reward -= 1
                valid = False   # 🔥 KEY CHANGE
            else:
                self.calendar[timeslot].extend(participants)
                reward += 1

                if meeting["priority"] == "high":
                    reward += 1
                elif meeting["priority"] == "medium":
                    reward += 0.5

                if timeslot == list(self.calendar.keys())[0]:
                    reward += 0.3

                self.assignments.append({
                    "meeting": meeting,
                    "timeslot": timeslot,
                    "valid": True
                })

        # move to next meeting
        self.step_idx += 1
        if self.step_idx >= len(self.meetings):
            self.done = True

        return self._get_obs(), reward, self.done, {}

    # ---------------- OBS ----------------
    def _get_obs(self):
        if self.done:
            return {}

        return {
            "current_meeting": self.meetings[self.step_idx],
            "calendar": self.calendar,
            "remaining_meetings": len(self.meetings) - self.step_idx,
            "step": self.step_idx,
        }

    # ---------------- STATE ----------------
    def state(self):
        return {
            "calendar": self.calendar,
            "meetings": self.meetings,
            "step": self.step_idx,
        }