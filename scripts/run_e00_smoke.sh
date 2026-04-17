#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input-pdf-or-dir> <output-name> [limit-pages]" >&2
  exit 1
fi

INPUT_PATH="$1"
OUTPUT_NAME="$2"
LIMIT_PAGES="${3:-5}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

python -m vision_rlm.preprocess.render_pages \
  --input-path "$INPUT_PATH" \
  --output-name "$OUTPUT_NAME" \
  --limit-docs 1

python -m vision_rlm.preprocess.parse_layout \
  --render-manifest "/l/users/badrinath.chandana/vision_rlm/data/processed/rendered/${OUTPUT_NAME}/page_manifest.jsonl" \
  --output-name "$OUTPUT_NAME" \
  --parser pdf_text \
  --limit-pages "$LIMIT_PAGES"

python -m vision_rlm.preprocess.build_regions \
  --layout-manifest "/l/users/badrinath.chandana/vision_rlm/data/processed/layout/${OUTPUT_NAME}/layout_manifest.jsonl" \
  --output-name "$OUTPUT_NAME"
