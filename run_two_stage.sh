#!/usr/bin/env bash
set -uo pipefail

SCRIPT_STARTED_AT="$(date +%s)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

format_seconds() {
  local total="$1"
  if [[ "$total" -lt 0 ]]; then
    total=0
  fi
  local hours=$((total / 3600))
  local minutes=$(((total % 3600) / 60))
  local seconds=$((total % 60))
  if [[ "$hours" -gt 0 ]]; then
    printf "%dh%02dm%02ds" "$hours" "$minutes" "$seconds"
  else
    printf "%dm%02ds" "$minutes" "$seconds"
  fi
}

log() {
  local now elapsed
  now="$(date '+%Y-%m-%d %H:%M:%S %z')"
  elapsed="$(format_seconds "$(($(date +%s) - SCRIPT_STARTED_AT))")"
  printf '[%s elapsed=%s] %s\n' "$now" "$elapsed" "$*"
}

if [[ "$#" -lt 1 ]]; then
  cat <<'EOF'
Usage: ./run_two_stage.sh INPUT [two-stage-subtitle options]

Example:
  ./run_two_stage.sh input.mp3 --device cuda:0 --diarization-device cuda --num-speakers 5 --hotword person_a
EOF
  exit 2
fi

step_started_at="$(date +%s)"
log "Starting run_two_stage.sh."
log "Command: uv run --no-sync python ${ROOT_DIR}/two_stage_subtitle.py $*"

cd "$ROOT_DIR" || exit 1
uv run --no-sync python "${ROOT_DIR}/two_stage_subtitle.py" "$@"
status="$?"
step_elapsed="$(format_seconds "$(($(date +%s) - step_started_at))")"

if [[ "$status" -ne 0 ]]; then
  log "two-stage pipeline failed after ${step_elapsed}, exit_code=${status}"
  exit "$status"
fi

log "two-stage pipeline completed in ${step_elapsed}"
