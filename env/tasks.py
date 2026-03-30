def get_task(task_name="easy"):
    if task_name == "easy":
        return easy_task()
    elif task_name == "medium":
        return medium_task()
    elif task_name == "hard":
        return hard_task()
    else:
        raise ValueError("Unknown task")


# ---------------- EASY ----------------
def easy_task():
    return {
        "meetings": [
            {
                "id": 1,
                "participants": ["A"],
                "priority": "low",
                "duration": 30,
            },
            {
                "id": 2,
                "participants": ["B"],
                "priority": "high",
                "duration": 30,
            },
            {
                "id": 3,
                "participants": ["C"],
                "priority": "medium",
                "duration": 30,
            },
        ],
        "calendar": {
            "9:00": [],
            "9:30": [],
            "10:00": [],
        },
    }


# ---------------- MEDIUM ----------------
def medium_task():
    return {
        "meetings": [
            {
                "id": 1,
                "participants": ["A", "B"],
                "priority": "high",
                "duration": 30,
            },
            {
                "id": 2,
                "participants": ["A"],
                "priority": "medium",
                "duration": 30,
            },
            {
                "id": 3,
                "participants": ["B"],
                "priority": "low",
                "duration": 30,
            },
            {
                "id": 4,
                "participants": ["C"],
                "priority": "medium",
                "duration": 30,
            },
            {
                "id": 5,
                "participants": ["A", "C"],
                "priority": "high",
                "duration": 30,
            },
        ],
        "calendar": {
            "9:00": [],
            "9:30": [],
            "10:00": [],
        },
    }


# ---------------- HARD ----------------
def hard_task():
    return {
        "meetings": [
            {
                "id": 1,
                "participants": ["A", "B"],
                "priority": "high",
                "duration": 30,
            },
            {
                "id": 2,
                "participants": ["A"],
                "priority": "medium",
                "duration": 30,
            },
            {
                "id": 3,
                "participants": ["B"],
                "priority": "low",
                "duration": 30,
            },
            {
                "id": 4,
                "participants": ["C"],
                "priority": "medium",
                "duration": 30,
            },
            {
                "id": 5,
                "participants": ["A", "C"],
                "priority": "high",
                "duration": 30,
            },
            {
                "id": 6,
                "participants": ["D"],
                "priority": "low",
                "duration": 30,
            },
            {
                "id": 7,
                "participants": ["B", "D"],
                "priority": "high",
                "duration": 30,
            },
            {
                "id": 8,
                "participants": ["C"],
                "priority": "low",
                "duration": 30,
            },
        ],
        "calendar": {
            "9:00": [],
            "9:30": [],
            "10:00": [],
        },
    }