#!/usr/bin/env bash

BASE_URL=${1:-http://localhost:8000}

echo "🔍 Testing environment at $BASE_URL"

echo "---- RESET ----"
curl -s -X POST "$BASE_URL/reset"
echo -e "\n"

echo "---- STEP LOOP ----"

DONE=false
COUNT=0

while [ "$DONE" = false ] && [ $COUNT -lt 20 ]; do
  RESPONSE=$(curl -s -X POST "$BASE_URL/step" \
    -H "Content-Type: application/json" \
    -d '{"action": {"timeslot": "9:00"}}')

  echo "$RESPONSE"

  DONE=$(echo "$RESPONSE" | grep -o '"done":[^,]*' | cut -d: -f2)

  COUNT=$((COUNT + 1))
done

echo -e "\n---- STATE ----"
curl -s "$BASE_URL/state"

echo -e "\n\n✅ Validation finished"