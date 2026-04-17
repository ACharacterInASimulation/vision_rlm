#!/usr/bin/env bash
#SBATCH --job-name=visionrlm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/l/users/badrinath.chandana/vision_rlm/logs/slurm/%j.out
#SBATCH --error=/l/users/badrinath.chandana/vision_rlm/logs/slurm/%j.err

set -euo pipefail

REPO_ROOT="/home/badrinath.chandana/git/ACharacterInASimulation/vision_rlm"
LOG_DIR="/l/users/badrinath.chandana/vision_rlm/logs/slurm"
COMMAND="${COMMAND:-./scripts/bootstrap_env.sh}"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "Host: $(hostname)"
echo "Date: $(date -Is)"
echo "Command: $COMMAND"

bash -lc "$COMMAND"
