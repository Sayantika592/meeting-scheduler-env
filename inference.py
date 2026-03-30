import os
import requests

# ---------------- CONFIG ----------------
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")
TASK = "hard"   # change to easy / medium / hard

# ---------------- HELPER ----------------
def choose_action(obs):
    meeting = obs["current_meeting"]
    calendar = obs["calendar"]

    priority = meeting["priority"]
    participants = meeting["participants"]

    # sort timeslots (earliest first)
    timeslots = list(calendar.keys())

    # try to assign safely
    for slot in timeslots:
        conflict = any(p in calendar[slot] for p in participants)
        if not conflict:
            return {"timeslot": slot}

    # if no safe slot
    if priority == "high":
        # force into first slot (take risk)
        return {"timeslot": timeslots[0]}
    else:
        return {"timeslot": "skip"}


# ---------------- MAIN LOOP ----------------
def main():
    # reset environment
    res = requests.post(f"{BASE_URL}/reset", params={"task": TASK})
    data = res.json()

    obs = data["observation"]
    done = False
    total_reward = 0

    step = 0

    while not done:
        action = choose_action(obs)

        res = requests.post(
            f"{BASE_URL}/step",
            json={"action": action}
        )
        data = res.json()

        obs = data["observation"]
        reward = data["reward"]
        done = data["done"]

        total_reward += reward
        step += 1

        print(f"Step {step}: Action={action}, Reward={reward}")

    print("\nEpisode finished!")
    print("Total reward:", total_reward)

    if "final_score" in data:
        print("Final Score:", data["final_score"])


if __name__ == "__main__":
    main()