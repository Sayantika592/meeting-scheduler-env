import random
from env.environment import SchedulerEnv
from env.tasks import get_task
from env.graders import grade_episode

task = get_task("easy")
env = SchedulerEnv(task)

obs = env.reset()

slots = ["9:00", "9:30", "10:00", "skip"]

while True:
    action = {"timeslot": random.choice(slots)}
    obs, reward, done, _ = env.step(action)

    if done:
        break

score = grade_episode(env)
print("Random Score:", score)