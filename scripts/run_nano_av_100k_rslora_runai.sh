#!/usr/bin/env bash
set -euo pipefail

# Reproducible RunAI launcher for the Nano30B AV 100k rsLoRA warm-start run.
# Intended to run inside the train-dev workspace/container.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_root="$(cd "${script_dir}/.." && pwd)"

VENV="${VENV:-/workspace/interp/.venv}"
INPUT_EXPLAINED="${INPUT_EXPLAINED:-/workspace/interp/artifacts/nano30b-nla-pilot/super-teacher-r27-100k-thinking-merged-20260525T2150Z/base_R27_super_thinking_99570_explained.parquet}"
MODEL_ID="${MODEL_ID:-/workspace/interp/models/nano-30b-a3b-bf16-hf}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/interp/outputs/nano30b-nla-pilot/av-r27-100k-rslora-repro}"
LOG_DIR="${LOG_DIR:-/workspace/interp/tmp/nano_av_peft_logs}"

ROW_LIMIT="${ROW_LIMIT:-99570}"
ROW_OFFSET="${ROW_OFFSET:-0}"
EXPERIMENT_CLASS="${EXPERIMENT_CLASS:-complete-performance}"
TRAIN_LR="${TRAIN_LR:-1e-5}"
TRAIN_STEPS="${TRAIN_STEPS:-800}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-0}"
TRAIN_SAMPLING="${TRAIN_SAMPLING:-random}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-25}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.9}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.05}"
TEST_FRACTION="${TEST_FRACTION:-0.05}"
SPLIT_MODE="${SPLIT_MODE:-doc}"
EVAL_TRAIN_LIMIT="${EVAL_TRAIN_LIMIT:-64}"
EVAL_HELDOUT_LIMIT="${EVAL_HELDOUT_LIMIT:-128}"
EVAL_VALIDATION_LIMIT="${EVAL_VALIDATION_LIMIT:-${EVAL_HELDOUT_LIMIT}}"
EVAL_TEST_LIMIT="${EVAL_TEST_LIMIT:-${EVAL_HELDOUT_LIMIT}}"
SEED="${SEED:-0}"
INJECTION_SCALE="${INJECTION_SCALE:-75}"
MAX_TARGET_TOKENS="${MAX_TARGET_TOKENS:-512}"
GENERATE_EXAMPLES="${GENERATE_EXAMPLES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

LORA_R="${LORA_R:-192}"
LORA_ALPHA="${LORA_ALPHA:-384}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,in_proj,out_proj,up_proj,down_proj}"
LORA_USE_RSLORA="${LORA_USE_RSLORA:-1}"
LORA_USE_DORA="${LORA_USE_DORA:-0}"

TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-nano30b-nla-pilot}"
WANDB_GROUP="${WANDB_GROUP:-av-r27-100k-rslora-repro}"
WANDB_TAGS="${WANDB_TAGS:-av,r27,100k,rslora,repro}"
NANO_WANDB="${NANO_WANDB:-1}"
SAVE_TRAINABLE_STATE="${SAVE_TRAINABLE_STATE:-1}"
BACKGROUND="${BACKGROUND:-1}"
DRY_RUN="${DRY_RUN:-0}"

timestamp="$(date -u +%Y%m%dT%H%MZ)"
train_label="s${TRAIN_STEPS}"
if [[ "${TRAIN_EPOCHS}" != "0" && "${TRAIN_EPOCHS}" != "0.0" ]]; then
  train_label="e${TRAIN_EPOCHS}-${TRAIN_SAMPLING}"
fi
RUN_NAME="${RUN_NAME:-av-r27-99570-rslora-r${LORA_R}-broad-scale${INJECTION_SCALE}-lr${TRAIN_LR}-b${TRAIN_BATCH_SIZE}-${train_label}-save-gen${GENERATE_EXAMPLES}-2gpu-offline-${timestamp}}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

cd "${code_root}"
if [[ -f "${VENV}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV}/bin/activate"
fi

export CUDA_VISIBLE_DEVICES
export WANDB_MODE
export NANO_WANDB
export PYTHONPATH="${code_root}/scripts:${code_root}/external/natural_language_autoencoders:${PYTHONPATH:-}"

cmd=(
  python scripts/nano_av_warmstart_smoke.py
  --input-explained "${INPUT_EXPLAINED}"
  --output-root "${OUTPUT_ROOT}"
  --timestamp "${RUN_NAME}"
  --row-limit "${ROW_LIMIT}"
  --row-offset "${ROW_OFFSET}"
  --experiment-class "${EXPERIMENT_CLASS}"
  --train-fraction "${TRAIN_FRACTION}"
  --validation-fraction "${VALIDATION_FRACTION}"
  --test-fraction "${TEST_FRACTION}"
  --split-mode "${SPLIT_MODE}"
  --eval-train-limit "${EVAL_TRAIN_LIMIT}"
  --eval-heldout-limit "${EVAL_HELDOUT_LIMIT}"
  --eval-validation-limit "${EVAL_VALIDATION_LIMIT}"
  --eval-test-limit "${EVAL_TEST_LIMIT}"
  --seed "${SEED}"
  --injection-scale "${INJECTION_SCALE}"
  --max-target-tokens "${MAX_TARGET_TOKENS}"
  --generate-examples "${GENERATE_EXAMPLES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --train-steps "${TRAIN_STEPS}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --train-epochs "${TRAIN_EPOCHS}"
  --train-sampling "${TRAIN_SAMPLING}"
  --train-lr "${TRAIN_LR}"
  --trainable-subset none
  --peft-method lora
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --lora-target-modules "${LORA_TARGET_MODULES}"
  --train-log-every "${TRAIN_LOG_EVERY}"
  --model-id "${MODEL_ID}"
  --device-map "${DEVICE_MAP}"
  --torch-dtype "${TORCH_DTYPE}"
  --wandb
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-name "${RUN_NAME}"
  --wandb-tags "${WANDB_TAGS}"
  --wandb-mode "${WANDB_MODE}"
)

if [[ "${LORA_USE_RSLORA}" != "0" ]]; then
  cmd+=(--lora-use-rslora)
fi
if [[ "${LORA_USE_DORA}" != "0" ]]; then
  cmd+=(--lora-use-dora)
fi
if [[ "${SAVE_TRAINABLE_STATE}" != "0" ]]; then
  cmd+=(--save-trainable-state)
fi

log_path="${LOG_DIR}/${RUN_NAME}.log"
pid_path="${LOG_DIR}/${RUN_NAME}.pid"
run_dir="${OUTPUT_ROOT}/${RUN_NAME}"

printf 'run_name=%s\n' "${RUN_NAME}"
printf 'run_dir=%s\n' "${run_dir}"
printf 'log=%s\n' "${log_path}"

if [[ "${DRY_RUN}" != "0" ]]; then
  printf 'command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

if [[ "${BACKGROUND}" != "0" ]]; then
  nohup "${cmd[@]}" >"${log_path}" 2>&1 &
  echo "$!" >"${pid_path}"
  printf 'pid=%s\n' "$(cat "${pid_path}")"
  printf 'pid_file=%s\n' "${pid_path}"
else
  "${cmd[@]}"
fi
