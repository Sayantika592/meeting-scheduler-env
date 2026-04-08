# Meeting Scheduler RL Environment

An RL environment that simulates real-world meeting scheduling. An AI agent assigns meetings to time slots and rooms while respecting participant availability, room capacity, equipment requirements, and time preferences.

**Deployed at:** [https://ME22B191-meeting-scheduler-env.hf.space](https://ME22B191-meeting-scheduler-env.hf.space)

## Why Meeting Scheduling?

Meeting scheduling is a task every organization faces daily. It involves constraint satisfaction (no double-booking), prioritization (executive meetings before socials), and resource allocation (projector rooms are scarce). The hard task is intentionally over-constrained — not all meetings can fit — forcing the agent to make priority-based triage decisions. This tests reasoning capabilities that go beyond simple slot-filling.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset?task=easy` | POST | Start a new episode. Returns first meeting to schedule. |
| `/step` | POST | Send a scheduling action. Returns observation + reward. |
| `/state` | GET | Get episode metadata (step count, assignments). |
| `/health` | GET | Health check. Returns `{"status": "healthy"}`. |

## Action Space

To schedule a meeting:
```json
{"action": {"timeslot": "10:00", "room": "Room A"}}
```

To skip a meeting:
```json
{"action": {"timeslot": "skip"}}
```

## Observation Space

Each step returns:

| Field | Type | Description |
|-------|------|-------------|
| `current_meeting` | Meeting | The meeting to schedule (name, participants, priority, duration, equipment, preferred time) |
| `available_rooms` | List[Room] | All rooms with capacity and equipment |
| `calendar_grid` | Dict | Timeslot → Room → meeting name or null |
| `remaining_count` | int | Meetings left to schedule |
| `scheduled_so_far` | List[str] | Successfully scheduled meeting names |
| `text_summary` | str | Human-readable calendar for LLM prompts |
| `step_reward` | float | Reward from the last action |
| `message` | str | Feedback about the last action |
| `done` | bool | Whether all meetings have been processed |

## Reward Design

**Per-step rewards (immediate feedback):**

| Outcome | Reward | Conditions |
|---------|--------|------------|
| Valid assignment | +1.0 base | No conflicts, room fits, equipment available |
| Priority bonus | +0.5 / +0.3 / +0.1 | High / medium / low priority |
| Preference bonus | +0.2 | Preferred time window respected |
| Invalid assignment | -1.0 | Conflict, room too small, missing equipment |
| Lazy skip | -0.3 | Meeting could have been scheduled |
| Smart skip | 0.0 | Meeting genuinely couldn't fit |

Maximum reward per step: +1.7 (high priority + preference met). Minimum: -1.0 (invalid action).

**End-of-episode grader (0.0 – 1.0):**

```
score = 0.30 × completion_rate
      + 0.35 × priority_weighted_completion
      + 0.15 × preference_satisfaction
      + 0.20 × skip_quality
```

## Tasks

### Easy (5 meetings, 6 slots, 2 rooms)
All 30-minute meetings, no equipment requirements, no time preferences. Every meeting can fit — tests basic scheduling competence. A random agent scores ~0.5, a competent agent scores 1.0.

### Medium (8 meetings, 6 slots, 3 rooms)
Introduces 60-minute meetings, equipment requirements (projector, video conferencing), and preferred time windows. Shared participants create conflicts. Alice appears in 4 meetings — she's the scheduling bottleneck. Only Room C has video conferencing. A thoughtful agent can schedule all 8.

### Hard (15 meetings, 8 slots, 4 rooms)
**Over-constrained by design.** 15 meetings with mixed durations (30/60/90 min) cannot all fit due to participant overlaps and equipment bottlenecks. The agent must triage: schedule all 5 high-priority meetings, fit medium-priority ones where possible, and skip low-priority meetings when necessary. Expected optimal: 10–12 scheduled, 3–5 skipped. This tests strategic reasoning and priority-based trade-offs.

## Constraints Enforced

- **No participant double-booking:** A person cannot be in two meetings at the same time
- **Room capacity:** Meeting participants must fit within room capacity
- **Equipment requirements:** Room must have required equipment (projector, video_conferencing, whiteboard)
- **Multi-slot blocking:** 60-min meetings block 2 consecutive slots, 90-min blocks 3
- **One meeting per room per slot:** No room overlap

## Setup

### Run Locally

```bash
git clone https://huggingface.co/spaces/ME22B191/meeting-scheduler-env
cd meeting-scheduler-env
pip install -r requirements.txt
uvicorn server.server:app --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
docker build -t meeting-scheduler .
docker run -p 8000:8000 meeting-scheduler
```

### Run Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"
python inference.py
```

## Project Structure

```
meeting-scheduler-env/
├── env/
│   ├── models.py          ← Pydantic Action, Observation, State models
│   ├── environment.py     ← Core scheduling logic (reset, step, state)
│   ├── tasks.py           ← Easy, medium, hard task definitions
│   └── graders.py         ← End-of-episode scoring (0.0–1.0)
├── server/
│   └── server.py          ← FastAPI server with /reset, /step, /state
├── inference.py           ← LLM agent script (OpenAI client)
├── openenv.yaml           ← Environment manifest
├── Dockerfile             ← Container definition
├── requirements.txt       ← Python dependencies
└── README.md              ← This file
```

## Baseline Scores

| Task | Score | Total Reward | Steps |
|------|-------|-------------|-------|
| Easy | ~1.00 | ~6.7 | 5 |
| Medium | ~0.75 | ~8.5 | 8 |
| Hard | ~0.65 | ~12.0 | 15 |

*Scores measured with Qwen2.5-72B-Instruct. Results vary by model and temperature.*
