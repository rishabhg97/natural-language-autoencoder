#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

API_KEY="${API_KEY:-${ANTHROPIC_API_KEY:-}}"
export API_KEY
: "${API_KEY:?set API_KEY or ANTHROPIC_API_KEY for https://inference-api.nvidia.com}"

CLUSTER_HOST="${CLUSTER_HOST:-cs-oci-ord-login-03.nvidia.com}"
CLUSTER_ROOT="${CLUSTER_ROOT:-/lustre/fsw/portfolios/llmservice/users/rigarg/mech_interp/research-projects-stage3/nano30b-nla-pilot}"
CLUSTER_RUN_DIR="${CLUSTER_RUN_DIR:-runs/introspection/stage0-r27-r20-r34-50k-pos10-20260522T200648Z}"
CLUSTER_JOB_ID="${CLUSTER_JOB_ID:-28145997_0}"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-runs/introspection/super-teacher-r27-30k-thinking-$(date -u +%Y%m%dT%H%M%SZ)}"
ROW_LIMIT="${ROW_LIMIT:-30000}"
CONCURRENCY="${CONCURRENCY:-32}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
REASONING_BUDGET="${REASONING_BUDGET:-4096}"

mkdir -p "$LOCAL_RUN_DIR"

echo "LOCAL_RUN_DIR=$LOCAL_RUN_DIR"
echo "CLUSTER_RUN_DIR=$CLUSTER_RUN_DIR"
echo "CLUSTER_JOB_ID=$CLUSTER_JOB_ID"
echo "waiting for cluster R_27 extraction to finish..."

while true; do
  STATE="$(
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$CLUSTER_HOST" \
      "STATE=\$(squeue -h -j '$CLUSTER_JOB_ID' -o %T 2>/dev/null | head -1 | tr -d ' '); if [ -z \"\$STATE\" ]; then STATE=\$(sacct -j '$CLUSTER_JOB_ID' --format=State -n -P 2>/dev/null | head -1 | tr -d ' '); fi; printf '%s\n' \"\$STATE\" || true"
  )"
  ROWS="$(
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$CLUSTER_HOST" \
      "cd '$CLUSTER_ROOT' && python3 - <<'PY' 2>/dev/null || true
import json
from pathlib import Path
p=Path('$CLUSTER_RUN_DIR/base_R27.metadata.json')
print(json.load(open(p)).get('row_count', 0) if p.exists() else 0)
PY"
  )"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) state=${STATE:-unknown} rows=${ROWS:-0}"
  if [[ "$STATE" == "COMPLETED" ]]; then
    break
  fi
  if [[ "$STATE" == FAILED* || "$STATE" == CANCELLED* || "$STATE" == TIMEOUT* ]]; then
    echo "cluster extraction ended in non-success state: $STATE" >&2
    exit 1
  fi
  sleep 300
done

echo "copying R_27 base parquet locally..."
rsync -avP \
  "$CLUSTER_HOST:$CLUSTER_ROOT/$CLUSTER_RUN_DIR/base_R27.parquet" \
  "$CLUSTER_HOST:$CLUSTER_ROOT/$CLUSTER_RUN_DIR/base_R27.parquet.nla_meta.yaml" \
  "$CLUSTER_HOST:$CLUSTER_ROOT/$CLUSTER_RUN_DIR/base_R27.metadata.json" \
  "$LOCAL_RUN_DIR/"

export PYTHONPATH="$ROOT/scripts:$ROOT/external/natural_language_autoencoders:${PYTHONPATH:-}"

python scripts/nano_stage2_super_teacher.py \
  --input "$LOCAL_RUN_DIR/base_R27.parquet" \
  --output "$LOCAL_RUN_DIR/base_R27_super_thinking_30k_explained.parquet" \
  --prompt-file "$ROOT/prompts/super_teacher_predictive_features.txt" \
  --row-limit "$ROW_LIMIT" \
  --chunk-size "$CHUNK_SIZE" \
  --model nvidia/nvidia/nemotron-3-super-v3 \
  --endpoint https://inference-api.nvidia.com/v1/chat/completions \
  --enable-thinking \
  --reasoning-effort high \
  --reasoning-budget "$REASONING_BUDGET" \
  --max-tokens "$MAX_TOKENS" \
  --temperature 0.2 \
  --concurrency "$CONCURRENCY" \
  --max-retries 8 \
  --timeout 300

echo "uploading explained parquet back to cluster..."
ssh -o BatchMode=yes -o ConnectTimeout=10 "$CLUSTER_HOST" \
  "mkdir -p '$CLUSTER_ROOT/$LOCAL_RUN_DIR'"
rsync -avP \
  "$LOCAL_RUN_DIR/base_R27_super_thinking_30k_explained.parquet" \
  "$LOCAL_RUN_DIR/base_R27_super_thinking_30k_explained.parquet.nla_meta.yaml" \
  "$LOCAL_RUN_DIR/base_R27_super_thinking_30k_explained.parquet.report.json" \
  "$LOCAL_RUN_DIR/base_R27_super_thinking_30k_explained.parquet.chunks" \
  "$CLUSTER_HOST:$CLUSTER_ROOT/$LOCAL_RUN_DIR/"

echo "DONE_LOCAL_RUN_DIR=$LOCAL_RUN_DIR"
