#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <layout_manifest> <region_manifest> <question_manifest> <output_name> <run_name> [artifact_subdir]" >&2
  exit 1
fi

LAYOUT_MANIFEST="$1"
REGION_MANIFEST="$2"
QUESTION_MANIFEST="$3"
OUTPUT_NAME="$4"
RUN_NAME="$5"
ARTIFACT_SUBDIR="${6:-debug}"

REPO_ROOT="/home/badrinath.chandana/git/ACharacterInASimulation/vision_rlm"
cd "$REPO_ROOT"
source .venv/bin/activate

echo "[run_e10_bm25] building indices for $OUTPUT_NAME"
python -m vision_rlm.preprocess.build_indices \
  --layout-manifest "$LAYOUT_MANIFEST" \
  --region-manifest "$REGION_MANIFEST" \
  --output-name "$OUTPUT_NAME" \
  --verbose

PAGE_INDEX="/l/users/badrinath.chandana/vision_rlm/data/processed/indices/${OUTPUT_NAME}/page_index.jsonl"

echo "[run_e10_bm25] evaluating retrieval for $RUN_NAME"
python -m vision_rlm.eval.eval_slidevqa \
  --mode page_retrieval_bm25 \
  --question-manifest "$QUESTION_MANIFEST" \
  --page-index "$PAGE_INDEX" \
  --run-name "$RUN_NAME" \
  --budgets small medium \
  --artifact-subdir "$ARTIFACT_SUBDIR" \
  --verbose

echo "[run_e10_bm25] done metrics=/l/users/badrinath.chandana/vision_rlm/artifacts/E10_page_retrieval_bm25/${ARTIFACT_SUBDIR}/${RUN_NAME}/metrics.json"
