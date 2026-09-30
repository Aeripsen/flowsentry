#!/usr/bin/env bash
# The load test against the real serving image on Linux: the image's Python
# 3.12 and its pip-resolved dependencies. Run by
# .github/workflows/loadtest-linux.yml on a GitHub-hosted runner; runs anywhere
# with docker and `pip install -e ".[loadtest]"` plus a trained model.
#
# Arms, alternating ABBA per round, at 1 and 4 workers (WEB_CONCURRENCY, which
# the image's uvicorn reads; nothing else changes):
#   lock    the image exactly as built: its CMD, the scoring lock on
#   nolock  the same image and command with FLOWSENTRY_SCORE_LOCK=0
# No --cpus limit, --network host (docker-proxy would add its own CPU cost to
# every request). The client runs on the runner, outside the container, and
# the container's PID is handed to it so server CPU is measured.
set -euo pipefail
IMAGE=${IMAGE:-flowsentry:loadtest}
ROUNDS=${ROUNDS:-3}
WORKERS=${WORKERS:-"1 4"}
OUT=${OUT:-artifacts/loadtest_linux}
LEVELS=${LEVELS:-1,2,4,8,16,32,64,128}
mkdir -p "$OUT"
IMAGE_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE")

run_arm() {
  local arm=$1 workers=$2 round=$3 lockenv=1
  [ "$arm" = nolock ] && lockenv=0
  local cid pid
  cid=$(docker run -d --network host -e WEB_CONCURRENCY="$workers" \
    -e FLOWSENTRY_SCORE_LOCK="$lockenv" "$IMAGE")
  pid=$(docker inspect -f '{{.State.Pid}}' "$cid")
  python scripts/loadtest.py --url http://127.0.0.1:8000 --server-pid "$pid" \
    --arm "$arm" --workers "$workers" --levels "$LEVELS" \
    --label "${arm}_w${workers}_r${round}" --out "$OUT/${arm}_w${workers}_r${round}.json" \
    --meta "where=docker container on a GitHub-hosted ubuntu runner" \
    --meta "image_id=$IMAGE_ID" \
    --meta "container_cmd=the image CMD, unchanged" \
    --meta "docker_run=--network host -e WEB_CONCURRENCY=$workers -e FLOWSENTRY_SCORE_LOCK=$lockenv (no --cpus)" \
    --meta "ab_round=$round"
  docker logs "$cid" 2>&1 | tail -n 3
  docker rm -f "$cid" >/dev/null
}

for round in $(seq 1 "$ROUNDS"); do
  for workers in $WORKERS; do
    if [ $((round % 2)) -eq 1 ]; then arms="lock nolock"; else arms="nolock lock"; fi
    for arm in $arms; do run_arm "$arm" "$workers" "$round"; done
  done
done
python scripts/loadtest.py --ab-summary "$OUT"
