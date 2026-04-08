"""Meeting Scheduler Environment Client.

Connects to the environment server via WebSocket for persistent sessions.
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SchedulerAction, SchedulerObservation


class SchedulerEnv(
    EnvClient[SchedulerAction, SchedulerObservation, State]
):
    """
    Client for the Meeting Scheduler Environment.

    Example:
        >>> with SchedulerEnv(base_url="http://localhost:8000").sync() as client:
        ...     result = client.reset()
        ...     result = client.step(SchedulerAction(timeslot="9:00", room="Room A"))
        ...     print(result.observation.message)
    """

    def _step_payload(self, action: SchedulerAction) -> Dict:
        payload = {"timeslot": action.timeslot}
        if action.room is not None:
            payload["room"] = action.room
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[SchedulerObservation]:
        obs_data = payload.get("observation", {})
        observation = SchedulerObservation(
            current_meeting=obs_data.get("current_meeting"),
            available_rooms=obs_data.get("available_rooms", []),
            calendar_grid=obs_data.get("calendar_grid", {}),
            remaining_count=obs_data.get("remaining_count", 0),
            scheduled_so_far=obs_data.get("scheduled_so_far", []),
            text_summary=obs_data.get("text_summary", ""),
            message=obs_data.get("message", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
