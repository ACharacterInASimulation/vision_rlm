#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-visionrlm}"
shift || true

if [[ "$#" -eq 0 ]]; then
  exec tmux new-session -A -s "$SESSION_NAME"
fi

COMMAND="$*"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux send-keys -t "$SESSION_NAME" "$COMMAND" C-m
else
  tmux new-session -d -s "$SESSION_NAME" "bash -lc '$COMMAND'"
fi

echo "Session: $SESSION_NAME"
echo "Command queued: $COMMAND"
echo "Attach with: tmux attach -t $SESSION_NAME"
