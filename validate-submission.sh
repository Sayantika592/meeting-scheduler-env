#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator
#
# Checks that your Docker image builds, server starts, all 3 tasks
# run correctly, openenv.yaml is valid, and (optionally) your HF Space is live.
#
# Prerequisites:
#   - Docker:  https://docs.docker.com/get-docker/
#   - curl     (usually pre-installed)
#   - python3  (for YAML validation)
#
# Usage:
#   chmod +x validate-submission.sh
#   ./validate-submission.sh [ping_url] [repo_dir]
#
# Arguments:
#   ping_url   (optional) Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#   repo_dir   (optional) Path to your repo (default: current directory)
#
# Examples:
#   ./validate-submission.sh
#   ./validate-submission.sh https://ME22B191-meeting-scheduler-env.hf.space
#   ./validate-submission.sh https://ME22B191-meeting-scheduler-env.hf.space ./my-repo
#

set -uo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOCKER_BUILD_TIMEOUT=600
DOCKER_IMAGE_NAME="openenv-validate-meeting-scheduler"
DOCKER_CONTAINER_NAME="openenv-validate-container"
SERVER_PORT=18742          # Use a high port to avoid conflicts
SERVER_STARTUP_TIMEOUT=60  # Seconds to wait for the server to start
EPISODE_STEP_TIMEOUT=10    # Seconds per API call

# ---------------------------------------------------------------------------
# Colors (only when running interactively)
# ---------------------------------------------------------------------------

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

pass()   { echo -e "  ${GREEN}✓ PASS${NC}: $1"; }
fail()   { echo -e "  ${RED}✗ FAIL${NC}: $1"; FAILURES=$((FAILURES + 1)); }
warn()   { echo -e "  ${YELLOW}⚠ WARN${NC}: $1"; }
header() { echo -e "\n${BOLD}[$1]${NC} $2"; }

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

cleanup() {
  echo ""
  echo "Cleaning up..."
  docker rm -f "$DOCKER_CONTAINER_NAME" 2>/dev/null || true
}

trap cleanup EXIT

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ ! -d "$REPO_DIR" ]; then
  echo "Error: '$REPO_DIR' is not a directory."
  exit 1
fi

# Resolve to absolute path
REPO_DIR="$(cd "$REPO_DIR" && pwd)"

FAILURES=0

echo -e "${BOLD}=============================================${NC}"
echo -e "${BOLD}  OpenEnv Submission Validator${NC}"
echo -e "${BOLD}  Meeting Scheduler Environment${NC}"
echo -e "${BOLD}=============================================${NC}"
echo "  Repo: $REPO_DIR"
[ -n "$PING_URL" ] && echo "  Space URL: $PING_URL"
echo ""

# ===================================================================
# CHECK 1: Required files exist
# ===================================================================
header "1/7" "Required files"

# openenv.yaml
if [ -f "$REPO_DIR/openenv.yaml" ]; then
  pass "openenv.yaml exists"
else
  fail "openenv.yaml not found at project root"
fi

# Dockerfile
if [ -f "$REPO_DIR/Dockerfile" ]; then
  pass "Dockerfile exists"
else
  fail "Dockerfile not found at project root"
fi

# inference.py
if [ -f "$REPO_DIR/inference.py" ]; then
  pass "inference.py exists"
else
  fail "inference.py not found at project root"
fi

# requirements.txt
if [ -f "$REPO_DIR/requirements.txt" ]; then
  pass "requirements.txt exists"
else
  fail "requirements.txt not found at project root"
fi

# ===================================================================
# CHECK 2: openenv.yaml content validation
# ===================================================================
header "2/7" "openenv.yaml validation"

if [ -f "$REPO_DIR/openenv.yaml" ]; then
  # Check for required fields using grep (no YAML parser dependency)
  YAML_OK=true

  if grep -q "^name:" "$REPO_DIR/openenv.yaml"; then
    pass "openenv.yaml has 'name' field"
  else
    fail "openenv.yaml missing 'name' field"
    YAML_OK=false
  fi

  if grep -q "^tasks:" "$REPO_DIR/openenv.yaml"; then
    pass "openenv.yaml has 'tasks' field"
  else
    fail "openenv.yaml missing 'tasks' field"
    YAML_OK=false
  fi

  if grep -q "^observation_space:" "$REPO_DIR/openenv.yaml"; then
    pass "openenv.yaml has 'observation_space' field"
  else
    fail "openenv.yaml missing 'observation_space' field"
    YAML_OK=false
  fi

  if grep -q "^action_space:" "$REPO_DIR/openenv.yaml"; then
    pass "openenv.yaml has 'action_space' field"
  else
    fail "openenv.yaml missing 'action_space' field"
    YAML_OK=false
  fi

  # Check that all 3 task difficulty levels are listed
  TASK_COUNT=0
  for task in easy medium hard; do
    if grep -q "  - $task" "$REPO_DIR/openenv.yaml"; then
      TASK_COUNT=$((TASK_COUNT + 1))
    fi
  done

  if [ "$TASK_COUNT" -ge 3 ]; then
    pass "openenv.yaml lists all 3 tasks (easy, medium, hard)"
  else
    fail "openenv.yaml should list at least 3 tasks (easy, medium, hard). Found: $TASK_COUNT"
  fi
else
  fail "Skipping openenv.yaml validation — file not found"
fi

# ===================================================================
# CHECK 3: Docker build
# ===================================================================
header "3/7" "Docker build"

if command -v docker &>/dev/null; then
  echo "  Building Docker image (timeout: ${DOCKER_BUILD_TIMEOUT}s)..."
  if run_with_timeout "$DOCKER_BUILD_TIMEOUT" \
       docker build -t "$DOCKER_IMAGE_NAME" "$REPO_DIR" > /tmp/docker-build.log 2>&1; then
    pass "Docker image built successfully"
  else
    fail "Docker build failed (see /tmp/docker-build.log)"
    echo "  Last 10 lines of build log:"
    tail -10 /tmp/docker-build.log | sed 's/^/    /'
  fi
else
  fail "Docker is not installed. Install from https://docs.docker.com/get-docker/"
fi

# ===================================================================
# CHECK 4: Server starts and responds to /health
# ===================================================================
header "4/7" "Server health check"

CONTAINER_OK=false
if docker image inspect "$DOCKER_IMAGE_NAME" &>/dev/null 2>&1; then
  # Stop any previous container with the same name
  docker rm -f "$DOCKER_CONTAINER_NAME" 2>/dev/null || true

  # Start the container
  echo "  Starting container on port $SERVER_PORT..."
  docker run -d \
    --name "$DOCKER_CONTAINER_NAME" \
    -p "$SERVER_PORT:8000" \
    "$DOCKER_IMAGE_NAME" > /dev/null 2>&1

  # Wait for the server to start
  echo "  Waiting for server to become ready (timeout: ${SERVER_STARTUP_TIMEOUT}s)..."
  ELAPSED=0
  while [ "$ELAPSED" -lt "$SERVER_STARTUP_TIMEOUT" ]; do
    if curl -sf "http://localhost:$SERVER_PORT/health" > /dev/null 2>&1; then
      CONTAINER_OK=true
      break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
  done

  if [ "$CONTAINER_OK" = true ]; then
    HEALTH_RESPONSE=$(curl -sf "http://localhost:$SERVER_PORT/health")
    pass "Server /health responds: $HEALTH_RESPONSE"
  else
    fail "Server did not respond to /health within ${SERVER_STARTUP_TIMEOUT}s"
    echo "  Container logs:"
    docker logs "$DOCKER_CONTAINER_NAME" 2>&1 | tail -20 | sed 's/^/    /'
  fi
else
  fail "Docker image not found — skipping server health check"
fi

# ===================================================================
# CHECK 5: /reset endpoint works
# ===================================================================
header "5/7" "Reset endpoint"

if [ "$CONTAINER_OK" = true ]; then
  RESET_OK=true
  for task in easy medium hard; do
    RESPONSE=$(curl -sf -X POST "http://localhost:$SERVER_PORT/reset?task=$task" 2>/dev/null)
    if [ $? -eq 0 ] && echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
obs = data.get('observation', {})
assert 'current_meeting' in obs, 'Missing current_meeting'
assert 'available_rooms' in obs, 'Missing available_rooms'
assert 'calendar_grid' in obs, 'Missing calendar_grid'
assert 'text_summary' in obs, 'Missing text_summary'
" 2>/dev/null; then
      pass "POST /reset?task=$task returns valid observation"
    else
      fail "POST /reset?task=$task did not return expected observation"
      RESET_OK=false
    fi
  done
else
  fail "Skipping /reset check — server not running"
  RESET_OK=false
fi

# ===================================================================
# CHECK 6: Run a full episode on each task and verify graders
# ===================================================================
header "6/7" "Task episodes & grader scores"

if [ "$CONTAINER_OK" = true ]; then
  for task in easy medium hard; do

    # Reset for this task
    RESET_RESP=$(curl -sf -X POST "http://localhost:$SERVER_PORT/reset?task=$task" 2>/dev/null)
    if [ $? -ne 0 ]; then
      fail "Could not reset task=$task"
      continue
    fi

    DONE=false
    STEP=0
    MAX_STEPS=25  # Safety cap — hard task has 15 meetings
    FINAL_SCORE=""

    while [ "$DONE" = false ] && [ "$STEP" -lt "$MAX_STEPS" ]; do
      STEP=$((STEP + 1))

      # Send "skip" for every meeting — ensures the episode completes
      # This is NOT testing agent quality, just that the environment runs end-to-end
      STEP_RESP=$(curl -sf -X POST "http://localhost:$SERVER_PORT/step" \
        -H "Content-Type: application/json" \
        -d '{"action": {"timeslot": "skip"}}' 2>/dev/null)

      if [ $? -ne 0 ]; then
        fail "POST /step failed at step $STEP for task=$task"
        break
      fi

      # Check if done
      DONE=$(echo "$STEP_RESP" | python3 -c "import sys,json; print(str(json.load(sys.stdin).get('done', False)).lower())" 2>/dev/null)
      if [ "$DONE" = "True" ] || [ "$DONE" = "true" ]; then
        DONE=true
        FINAL_SCORE=$(echo "$STEP_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('final_score',''))" 2>/dev/null)
      fi
    done

    if [ "$DONE" = true ]; then
      pass "Task '$task': episode completed in $STEP steps"

      # Validate final_score is in [0.0, 1.0]
      if [ -n "$FINAL_SCORE" ]; then
        SCORE_VALID=$(python3 -c "
s = float('$FINAL_SCORE')
print('yes' if 0.0 <= s <= 1.0 else 'no')
" 2>/dev/null)
        if [ "$SCORE_VALID" = "yes" ]; then
          pass "Task '$task': final_score=$FINAL_SCORE (in [0.0, 1.0])"
        else
          fail "Task '$task': final_score=$FINAL_SCORE is outside [0.0, 1.0]"
        fi
      else
        fail "Task '$task': no final_score returned when done=true"
      fi
    else
      fail "Task '$task': episode did not complete within $MAX_STEPS steps"
    fi
  done
else
  fail "Skipping episode tests — server not running"
fi

# ===================================================================
# CHECK 7: HF Space liveness (optional)
# ===================================================================
header "7/7" "HF Space ping"

if [ -n "$PING_URL" ]; then
  echo "  Pinging $PING_URL ..."
  HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "$PING_URL/health" 2>/dev/null)
  if [ "$HTTP_CODE" = "200" ]; then
    pass "HF Space /health returned HTTP 200"
  else
    fail "HF Space /health returned HTTP $HTTP_CODE (expected 200)"
  fi

  # Also try /reset
  RESET_CODE=$(curl -sf -o /dev/null -w "%{http_code}" -X POST "$PING_URL/reset?task=easy" 2>/dev/null)
  if [ "$RESET_CODE" = "200" ]; then
    pass "HF Space POST /reset?task=easy returned HTTP 200"
  else
    fail "HF Space POST /reset?task=easy returned HTTP $RESET_CODE (expected 200)"
  fi
else
  warn "No HF Space URL provided — skipping liveness check"
  echo "  To test your Space, run: ./validate-submission.sh https://your-space.hf.space"
fi

# ===================================================================
# Summary
# ===================================================================

echo ""
echo -e "${BOLD}=============================================${NC}"
if [ "$FAILURES" -eq 0 ]; then
  echo -e "${GREEN}${BOLD}  ALL CHECKS PASSED ✓${NC}"
  echo -e "  Your submission is ready for upload."
else
  echo -e "${RED}${BOLD}  $FAILURES CHECK(S) FAILED ✗${NC}"
  echo -e "  Fix the issues above before submitting."
fi
echo -e "${BOLD}=============================================${NC}"

exit "$FAILURES"
