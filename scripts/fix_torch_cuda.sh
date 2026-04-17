#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run ./scripts/bootstrap_env.sh first." >&2
  exit 1
fi

source .venv/bin/activate

python -m pip uninstall -y torch torchvision torchaudio || true
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m vision_rlm.cli doctor
