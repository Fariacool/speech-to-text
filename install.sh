#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-}"
SKIP_FFMPEG=0
SKIP_TORCH=0

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Options:
  --python VERSION       Python version for uv, default: 3.11
  --pytorch-index URL    PyTorch wheel index URL, e.g. https://download.pytorch.org/whl/cu121
  --cpu                  Install CPU PyTorch wheels
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
    return
  fi

  echo "Installing uv..."
  if ! command_exists curl; then
    echo "curl is required to install uv. Install curl first, then rerun this script." >&2
    exit 1
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
}

install_ffmpeg() {
  if [[ "$SKIP_FFMPEG" -eq 1 ]] || command_exists ffmpeg; then
    return
  fi

  echo "Installing ffmpeg..."
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
    echo "Could not detect a supported package manager. Install ffmpeg manually, then rerun with --skip-ffmpeg." >&2
    exit 1
  fi
}

install_project() {
  echo "Creating uv environment with Python ${PYTHON_VERSION}..."
  uv python install "$PYTHON_VERSION"
  uv venv --python "$PYTHON_VERSION" .venv

  if [[ "$SKIP_TORCH" -eq 0 ]]; then
    echo "Installing torch and torchaudio..."
    if [[ -n "$PYTORCH_INDEX_URL" ]]; then
      uv pip install --python .venv/bin/python torch torchaudio --index-url "$PYTORCH_INDEX_URL"
    else
      uv pip install --python .venv/bin/python torch torchaudio
    fi
  fi

  echo "Installing project dependencies..."
  uv pip install --python .venv/bin/python -e .
}

main() {
  install_ffmpeg
  install_uv
  install_project

  echo
  echo "Install complete."
  echo "Run a smoke check with:"
  echo "  uv run --no-sync funasr-subtitle --help"
  echo
  echo "Example transcription:"
  echo "  uv run --no-sync funasr-subtitle input.mp4 --device cuda:0 --hotword person_a --hotword person_b"
}

main "$@"
