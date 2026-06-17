#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-}"
SKIP_FFMPEG=0
SKIP_TORCH=0
INSTALL_DIARIZATION=0
SCRIPT_STARTED_AT="$(date +%s)"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

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

run_step() {
  local label="$1"
  shift
  local started_at status elapsed
  started_at="$(date +%s)"
  log "Start: ${label}"
  set +e
  (
    set -euo pipefail
    "$@"
  )
  status="$?"
  set -e
  elapsed="$(format_seconds "$(($(date +%s) - started_at))")"
  if [[ "$status" -ne 0 ]]; then
    log "Failed: ${label}, elapsed=${elapsed}, exit_code=${status}"
    return "$status"
  fi
  log "Done: ${label}, elapsed=${elapsed}"
}

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Options:
  --python VERSION       Python version for uv, default: 3.11
  --pytorch-index URL    PyTorch wheel index URL, e.g. https://download.pytorch.org/whl/cu121
  --cpu                  Install CPU PyTorch wheels
  --diarization          Install pyannote.audio dependencies for two-stage speaker diarization
  --skip-ffmpeg          Do not install ffmpeg
  --skip-torch           Do not install torch/torchaudio
  -h, --help             Show this help

Environment:
  PYTHON_VERSION         Same as --python
  PYTORCH_INDEX_URL      Same as --pytorch-index

Examples:
  ./install.sh
  ./install.sh --pytorch-index https://download.pytorch.org/whl/cu121
  PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 ./install.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --pytorch-index)
      PYTORCH_INDEX_URL="$2"
      shift 2
      ;;
    --cpu)
      PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
      shift
      ;;
    --diarization)
      INSTALL_DIARIZATION=1
      shift
      ;;
    --skip-ffmpeg)
      SKIP_FFMPEG=1
      shift
      ;;
    --skip-torch)
      SKIP_TORCH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

install_uv() {
  if command_exists uv; then
    log "uv already installed."
    return
  fi

  log "Installing uv..."
  if ! command_exists curl; then
    log "curl is required to install uv. Install curl first, then rerun this script."
    exit 1
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
}

install_ffmpeg() {
  if [[ "$SKIP_FFMPEG" -eq 1 ]] || command_exists ffmpeg; then
    log "ffmpeg install skipped or already available."
    return
  fi

  log "Installing ffmpeg..."
  if command_exists apt-get; then
    sudo apt-get update
    sudo apt-get install -y ffmpeg curl ca-certificates
  elif command_exists dnf; then
    sudo dnf install -y ffmpeg curl ca-certificates
  elif command_exists yum; then
    sudo yum install -y ffmpeg curl ca-certificates
  elif command_exists pacman; then
    sudo pacman -Sy --noconfirm ffmpeg curl ca-certificates
  elif command_exists apk; then
    sudo apk add --no-cache ffmpeg curl ca-certificates
  elif command_exists brew; then
    brew install ffmpeg
  else
    log "Could not detect a supported package manager. Install ffmpeg manually, then rerun with --skip-ffmpeg."
    exit 1
  fi
}

install_project() {
  log "Creating uv environment with Python ${PYTHON_VERSION}..."
  uv python install "$PYTHON_VERSION"
  uv venv --python "$PYTHON_VERSION" .venv

  if [[ "$SKIP_TORCH" -eq 0 ]]; then
    log "Installing torch and torchaudio..."
    if [[ -n "$PYTORCH_INDEX_URL" ]]; then
      uv pip install --python .venv/bin/python torch torchaudio --index-url "$PYTORCH_INDEX_URL"
    else
      uv pip install --python .venv/bin/python torch torchaudio
    fi
  else
    log "torch install skipped."
  fi

  if [[ "$INSTALL_DIARIZATION" -eq 1 ]]; then
    log "Installing project dependencies with diarization extra..."
    uv pip install --python .venv/bin/python -e '.[diarization]'
  else
    log "Installing project dependencies..."
    uv pip install --python .venv/bin/python -e .
  fi
}

main() {
  run_step "Install ffmpeg" install_ffmpeg
  run_step "Install uv" install_uv
  run_step "Install Python environment and project dependencies" install_project

  echo
  log "Install complete."
  echo "Run a smoke check with:"
  echo "  uv run --no-sync funasr-subtitle --help"
  echo "  uv run --no-sync two-stage-subtitle --help"
  echo
  echo "Example transcription:"
  echo "  uv run --no-sync funasr-subtitle input.mp4 --device cuda:0 --hotword person_a --hotword person_b"
  echo
  echo "Example two-stage speaker workflow:"
  echo "  ./run_two_stage.sh input.mp3 --device cuda:0 --diarization-device cuda --num-speakers 5 --hotword person_a"
}

main "$@"
