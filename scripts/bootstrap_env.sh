#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[core,dev]'
python -m vision_rlm.cli bootstrap-dirs
python -m vision_rlm.cli doctor
python -m vision_rlm.cli show-paths
