from env.environment import SchedulerEnv
from env.tasks import get_task
from env.graders import grade_episode

task = get_task("easy")
env = SchedulerEnv(task)

obs = env.reset()

while True:
    meeting = obs["current_meeting"]
    calendar = obs["calendar"]

    # assign to least occupied slot (balanced)
    best_slot = min(calendar, key=lambda s: len(calendar[s]))

    action = {"timeslot": best_slot}

    obs, reward, done, _ = env.step(action)

    if done:
        break

score = grade_episode(env)
print("Perfect Score:", score)