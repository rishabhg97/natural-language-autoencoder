#!/usr/bin/env bash
set -euo pipefail

# Qwen-faithful AV-SFT path: Miles FSDP2 + NLADataSource + NLAFSDPActor.
# This launcher expects an AV-SFT parquet with prompt/response/activation_vector.

CODE_ROOT=${CODE_ROOT:-/workspace/interp/code/nano30b-nla-pilot-current}
MILES_ROOT=${MILES_ROOT:-/workspace/interp/code/miles-051cd15}
INPUT_PARQUET=${INPUT_PARQUET:?set INPUT_PARQUET to an AV-SFT parquet}
OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft}
MODEL_ID=${MODEL_ID:-/workspace/interp/models/nano-30b-a3b-bf16-hf}
PYTHON=${PYTHON:-python}

EXPERIMENT_CLASS=${EXPERIMENT_CLASS:-small-smoke}
ROW_LIMIT=${ROW_LIMIT:-96}
TRAIN_FRACTION=${TRAIN_FRACTION:-0.8}
VALIDATION_FRACTION=${VALIDATION_FRACTION:-0.1}
TEST_FRACTION=${TEST_FRACTION:-0.1}

NUM_GPUS=${NUM_GPUS:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-$GLOBAL_BATCH_SIZE}
MAX_STEPS=${MAX_STEPS:-}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-1}
LR=${LR:-1e-5}
SAVE_INTERVAL=${SAVE_INTERVAL:-1}
EVAL_INTERVAL=${EVAL_INTERVAL:-}
INJECTION_SCALE=${INJECTION_SCALE:-75}
NO_SAVE_OPTIM=${NO_SAVE_OPTIM:-0}
ADAM_FOREACH=${ADAM_FOREACH:-0}
DISABLE_CHECKPOINT_SAVE=${DISABLE_CHECKPOINT_SAVE:-0}
NLA_TIMING_DEBUG=${NLA_TIMING_DEBUG:-0}
NLA_SKIP_GRAD_NORM=${NLA_SKIP_GRAD_NORM:-0}
NLA_LOCAL_GRAD_NORM=${NLA_LOCAL_GRAD_NORM:-1}
NLA_ROUTER_METRICS=${NLA_ROUTER_METRICS:-0}
NLA_PATCH_NEMOTRON_REMOTE_CODE=${NLA_PATCH_NEMOTRON_REMOTE_CODE:-0}
LOAD_CHECKPOINT=${LOAD_CHECKPOINT:-}
LOSS_MASK_TYPE=${LOSS_MASK_TYPE:-qwen}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-eager}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

WANDB_PROJECT=${WANDB_PROJECT:-nano30b-nla-pilot}
WANDB_GROUP=${WANDB_GROUP:-nano-av-miles-fsdp2-sft}
WANDB_MODE=${WANDB_MODE:-offline}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}

case "$EXPERIMENT_CLASS" in
  small-smoke)
    if (( ROW_LIMIT > 96 )); then
      echo "small-smoke requires ROW_LIMIT <= 96 (got $ROW_LIMIT)" >&2
      exit 2
    fi
    ;;
  medium-small)
    if (( ROW_LIMIT > 960 )); then
      echo "medium-small requires ROW_LIMIT <= 960 (got $ROW_LIMIT)" >&2
      exit 2
    fi
    ;;
  complete-performance)
    if (( ROW_LIMIT < 90000 )); then
      echo "complete-performance requires ROW_LIMIT >= 90000 (got $ROW_LIMIT)" >&2
      exit 2
    fi
    if [[ "$TRAIN_FRACTION,$VALIDATION_FRACTION,$TEST_FRACTION" != "0.9,0.05,0.05" ]]; then
      echo "complete-performance requires TRAIN/VALIDATION/TEST_FRACTION=0.9/0.05/0.05" >&2
      exit 2
    fi
    if [[ "$DISABLE_CHECKPOINT_SAVE" == "1" ]]; then
      echo "complete-performance cannot disable checkpoint saves" >&2
      exit 2
    fi
    ;;
  *)
    echo "unknown EXPERIMENT_CLASS=$EXPERIMENT_CLASS" >&2
    exit 2
    ;;
esac

if [[ -n "$LOAD_CHECKPOINT" ]]; then
  load_leaf=$(basename "$LOAD_CHECKPOINT")
  if [[ "$load_leaf" =~ ^iter_[0-9]+$ ]]; then
    echo "LOAD_CHECKPOINT must point to the checkpoint root containing latest_checkpointed_iteration.txt, not an iter_XXXXXXX leaf. Use $(dirname "$LOAD_CHECKPOINT") instead." >&2
    exit 2
  fi
fi

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
RUN_NAME=${RUN_NAME:-nano-av-miles-fsdp2-${EXPERIMENT_CLASS}-rows${ROW_LIMIT}-gb${GLOBAL_BATCH_SIZE}-mb${MICRO_BATCH_SIZE}-${timestamp}}
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SAVE_DIR="${RUN_DIR}/checkpoints"
WANDB_DIR="${RUN_DIR}/wandb"
mkdir -p "$RUN_DIR" "$SAVE_DIR" "$WANDB_DIR"

export EXPERIMENT_CLASS ROW_LIMIT TRAIN_FRACTION VALIDATION_FRACTION TEST_FRACTION
export MODEL_ID GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE ROLLOUT_BATCH_SIZE MAX_STEPS TRAIN_EPOCHS
export LR SAVE_INTERVAL EVAL_INTERVAL INJECTION_SCALE NO_SAVE_OPTIM ADAM_FOREACH DISABLE_CHECKPOINT_SAVE NLA_TIMING_DEBUG NLA_SKIP_GRAD_NORM NLA_LOCAL_GRAD_NORM NLA_ROUTER_METRICS NLA_PATCH_NEMOTRON_REMOTE_CODE LOAD_CHECKPOINT WANDB_PROJECT WANDB_GROUP RUN_NAME
export PYTHONPATH="${CODE_ROOT}/external/natural_language_autoencoders:${MILES_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF
export WANDB_MODE=offline
export WANDB_DIR
export NLA_KEEP_LOCAL=${NLA_KEEP_LOCAL:-1}

PROMPT_DATA="$INPUT_PARQUET"
if (( ROW_LIMIT > 0 )); then
  total_rows=$("$PYTHON" - <<'PY' "$INPUT_PARQUET"
import pyarrow.parquet as pq
import sys
print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)
PY
)
  if (( ROW_LIMIT < total_rows )); then
    PROMPT_DATA="${RUN_DIR}/av_sft_rows${ROW_LIMIT}.parquet"
    "$PYTHON" - <<'PY' "$INPUT_PARQUET" "$PROMPT_DATA" "$ROW_LIMIT"
from pathlib import Path
import sys
import pyarrow.parquet as pq
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
limit = int(sys.argv[3])
table = pq.read_table(src).slice(0, limit)
pq.write_table(table, dst)
sidecar_src = src.with_name(src.name + ".nla_meta.yaml")
sidecar_dst = dst.with_name(dst.name + ".nla_meta.yaml")
meta = yaml.safe_load(sidecar_src.read_text())
meta["row_count"] = limit
sidecar_dst.write_text(yaml.safe_dump(meta, sort_keys=False))
PY
  fi
fi

"$PYTHON" - <<'PY' "$RUN_DIR/run_config.json" "$PROMPT_DATA"
import json
import os
import sys

keys = [
    "EXPERIMENT_CLASS", "ROW_LIMIT", "TRAIN_FRACTION", "VALIDATION_FRACTION",
    "TEST_FRACTION", "MODEL_ID", "GLOBAL_BATCH_SIZE", "MICRO_BATCH_SIZE",
    "ROLLOUT_BATCH_SIZE", "MAX_STEPS", "TRAIN_EPOCHS", "LR", "SAVE_INTERVAL",
    "EVAL_INTERVAL", "INJECTION_SCALE", "NO_SAVE_OPTIM", "ADAM_FOREACH",
    "DISABLE_CHECKPOINT_SAVE", "NLA_TIMING_DEBUG", "NLA_SKIP_GRAD_NORM",
    "NLA_LOCAL_GRAD_NORM", "NLA_ROUTER_METRICS", "NLA_PATCH_NEMOTRON_REMOTE_CODE", "LOAD_CHECKPOINT",
    "PYTORCH_CUDA_ALLOC_CONF", "WANDB_MODE", "WANDB_PROJECT", "WANDB_GROUP",
    "RUN_NAME", "NLA_KEEP_LOCAL",
]
config = {key: os.environ.get(key) for key in keys}
config["prompt_data"] = sys.argv[2]
with open(sys.argv[1], "w") as f:
    json.dump(config, f, indent=2, sort_keys=True)
    f.write("\n")
PY

cmd=(
  "$PYTHON" "${MILES_ROOT}/train.py"
  --train-backend fsdp
  --custom-actor-cls-path nla.train_actor.NLAFSDPActor
  --loss-type sft_loss
  --debug-train-only
  --disable-compute-advantages-and-returns
  --rollout-function-path nla.rollout.sft_actor.generate_rollout
  --data-source-path nla.data_source.NLADataSource
  --prompt-data "$PROMPT_DATA"
  --hf-checkpoint "$MODEL_ID"
  --input-key prompt
  --nla-injection-scale "$INJECTION_SCALE"
  --actor-num-nodes 1
  --actor-num-gpus-per-node "$NUM_GPUS"
  --num-gpus-per-node "$NUM_GPUS"
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --lr "$LR"
  --loss-mask-type "$LOSS_MASK_TYPE"
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --use-wandb
  --wandb-mode offline
  --wandb-dir "$WANDB_DIR"
  --wandb-project "$WANDB_PROJECT"
  --wandb-group "$WANDB_GROUP"
  --disable-wandb-random-suffix
  --rollout-shuffle
)

if [[ "$DISABLE_CHECKPOINT_SAVE" != "1" ]]; then
  cmd+=(--save "$SAVE_DIR" --save-interval "$SAVE_INTERVAL")
fi

if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  cmd+=(--gradient-checkpointing)
fi

if [[ "$NO_SAVE_OPTIM" == "1" ]]; then
  cmd+=(--no-save-optim)
fi

if [[ "$ADAM_FOREACH" == "1" ]]; then
  cmd+=(--adam-foreach)
fi

if [[ "$NLA_TIMING_DEBUG" == "1" ]]; then
  cmd+=(--nla-timing-debug)
fi

if [[ "$NLA_SKIP_GRAD_NORM" == "1" ]]; then
  cmd+=(--nla-skip-grad-norm)
fi

if [[ "$NLA_LOCAL_GRAD_NORM" == "0" ]]; then
  cmd+=(--no-nla-local-grad-norm)
fi

if [[ -n "$LOAD_CHECKPOINT" ]]; then
  cmd+=(--load "$LOAD_CHECKPOINT")
fi

if [[ -n "$WANDB_RUN_NAME" ]]; then
  cmd+=(--wandb-run-name "$WANDB_RUN_NAME")
fi

if [[ -n "$MAX_STEPS" ]]; then
  if (( MAX_STEPS <= 0 )); then
    echo "MAX_STEPS must be positive for Miles FSDP2 SFT; zero-step load-only trips the LR scheduler. For a resume smoke, set MAX_STEPS to latest_checkpoint + resume_steps and disable checkpoint save." >&2
    exit 2
  fi
  cmd+=(--num-rollout "$MAX_STEPS")
else
  cmd+=(--num-epoch "$TRAIN_EPOCHS")
fi

if [[ -n "$EVAL_INTERVAL" ]]; then
  echo "EVAL_INTERVAL is recorded in run_config but not passed to Miles yet; use scripts/eval_nano_av_miles_checkpoint.py until a Miles eval rollout is wired." >&2
fi

printf '%q ' "${cmd[@]}" | tee "${RUN_DIR}/command.txt"
printf '\n' | tee -a "${RUN_DIR}/command.txt"
exec "${cmd[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
