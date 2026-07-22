#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${ENV_FILE:-/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing env file: $ENV_FILE" >&2
  exit 2
fi

set -a
source "$ENV_FILE"
set +a

: "${NVIDIA_API_KEY:?set NVIDIA_API_KEY in $ENV_FILE}"

RUN_DIR="${RUN_DIR:-runs/introspection/super-teacher-r27-50k-thinking-delta-local-$(date -u +%Y%m%dT%H%M%SZ)}"
INPUT="${INPUT:-runs/introspection/super-teacher-r27-30k-thinking-20260522T203317Z/base_R27.parquet}"
OUTPUT="${OUTPUT:-$RUN_DIR/base_R27_super_thinking_rows30000_49950_explained.parquet}"
ROW_OFFSET="${ROW_OFFSET:-30000}"
ROW_LIMIT="${ROW_LIMIT:-19950}"
CHUNK_SIZE="${CHUNK_SIZE:-96}"
CONCURRENCY="${CONCURRENCY:-48}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
REASONING_BUDGET="${REASONING_BUDGET:-4096}"

mkdir -p "$RUN_DIR"
export PYTHONPATH="$ROOT/scripts:$ROOT/external/natural_language_autoencoders:${PYTHONPATH:-}"

python scripts/nano_stage2_super_teacher.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --prompt-file prompts/super_teacher_predictive_features.txt \
  --row-offset "$ROW_OFFSET" \
  --row-limit "$ROW_LIMIT" \
  --chunk-size "$CHUNK_SIZE" \
  --model nvidia/nvidia/nemotron-3-super-v3 \
  --endpoint https://inference-api.nvidia.com/v1/chat/completions \
  --api-key-env NVIDIA_API_KEY \
  --enable-thinking \
  --reasoning-effort high \
  --reasoning-budget "$REASONING_BUDGET" \
  --max-tokens "$MAX_TOKENS" \
  --temperature 0.2 \
  --concurrency "$CONCURRENCY" \
  --max-retries 8 \
  --timeout 300
