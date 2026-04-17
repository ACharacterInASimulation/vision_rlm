#!/usr/bin/env bash
set -euo pipefail

# Colab bootstrap for teacher-rollout work.
#
# Usage in Colab:
#   !bash scripts/colab_teacher_setup.sh
#
# Optional environment variables:
#   HF_TOKEN=hf_...
#   VISION_RLM_REPO_URL=https://github.com/<you>/<repo>.git
#   VISION_RLM_REPO_DIR=/content/vision_rlm
#   VISION_RLM_COLAB_CACHE=/content/vision_rlm_cache
#   VISION_RLM_TEACHER_MODEL=Qwen/Qwen2.5-VL-72B-Instruct-AWQ
#   VISION_RLM_DOWNLOAD_SLIDEVQA=1
#   VISION_RLM_RUN_SMOKE=1

REPO_URL="${VISION_RLM_REPO_URL:-}"
REPO_DIR="${VISION_RLM_REPO_DIR:-/content/vision_rlm}"
CACHE_ROOT="${VISION_RLM_COLAB_CACHE:-/content/vision_rlm_cache}"
HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
MODEL_ID="${VISION_RLM_TEACHER_MODEL:-Qwen/Qwen2.5-VL-72B-Instruct-AWQ}"
DOWNLOAD_SLIDEVQA="${VISION_RLM_DOWNLOAD_SLIDEVQA:-1}"
RUN_SMOKE="${VISION_RLM_RUN_SMOKE:-0}"

echo "[vision_rlm] repo dir: $REPO_DIR"
echo "[vision_rlm] cache root: $CACHE_ROOT"
echo "[vision_rlm] model: $MODEL_ID"

mkdir -p "$CACHE_ROOT" "$HF_HOME"
export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

python -m pip install -U pip setuptools wheel

# Ensure Colab uses a CUDA-enabled PyTorch build instead of a CPU-only wheel.
python -m pip uninstall -y torch torchvision torchaudio || true
python -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Qwen's official model card recommends latest transformers from source.
python -m pip install -U \
  "git+https://github.com/huggingface/transformers" \
  accelerate \
  huggingface_hub \
  datasets \
  optimum \
  pillow \
  pymupdf \
  rank_bm25 \
  sentencepiece \
  qwen-vl-utils[decord]==0.0.8

python -m pip install gptqmodel --no-build-isolation

if [[ -n "${HF_TOKEN:-}" ]]; then
  hf auth login --token "$HF_TOKEN" --add-to-git-credential
else
  echo "[vision_rlm] HF_TOKEN not set; assuming Colab already has Hugging Face auth."
fi

if [[ -n "$REPO_URL" && ! -d "$REPO_DIR/.git" ]]; then
  echo "[vision_rlm] cloning repo from $REPO_URL"
  git clone "$REPO_URL" "$REPO_DIR"
fi

if [[ -d "$REPO_DIR" ]]; then
  echo "[vision_rlm] installing repo from $REPO_DIR"
  python -m pip install -e "$REPO_DIR"
fi

MODEL_DIR="$CACHE_ROOT/models/${MODEL_ID//\//__}"
mkdir -p "$MODEL_DIR"
echo "[vision_rlm] downloading teacher model to $MODEL_DIR"
hf download "$MODEL_ID" \
  --local-dir "$MODEL_DIR"

if [[ "$DOWNLOAD_SLIDEVQA" == "1" ]]; then
  DATASET_DIR="$CACHE_ROOT/datasets/slidevqa"
  mkdir -p "$DATASET_DIR"
  echo "[vision_rlm] downloading SlideVQA dataset to $DATASET_DIR"
  hf download NTT-hil-insight/SlideVQA \
    --repo-type dataset \
    --local-dir "$DATASET_DIR"
fi

if [[ "$RUN_SMOKE" == "1" ]]; then
  python "$REPO_DIR/scripts/colab_teacher_smoke.py" \
    --model-dir "$MODEL_DIR"
fi

cat <<EOF
[vision_rlm] setup complete
[vision_rlm] model dir: $MODEL_DIR
[vision_rlm] dataset dir: ${DATASET_DIR:-skipped}

Next:
  cd "$REPO_DIR"
  python scripts/colab_teacher_smoke.py --model-dir "$MODEL_DIR"
EOF
