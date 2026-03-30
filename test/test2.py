from env.environment import SchedulerEnv
from env.tasks import get_task
from env.graders import grade_episode

task = get_task("easy")
env = SchedulerEnv(task)

obs = env.reset()

while True:
    # force everything into same slot → worst packing
    action = {"timeslot": "9:00"}

    obs, reward, done, _ = env.step(action)

    if done:
        break
    
score = grade_episode(env)
print("Worst Score:", score)