#!/bin/bash
# NLA RL: simultaneous AV (GRPO) + AR (supervised MSE).
#
# The AR is trained alongside the AV because it IS the reward model: reward =
# -MSE(AR_fwd(explanation), gold_activation), computed via Ray remote on the
# live AR-trainer GPUs (nla/reward.py). A frozen AR would give stale rewards
# the AV would game — the AR must learn alongside the AV what semantic content
# each activation direction encodes.
#
# Two RayTrainGroups: actor (GRPO + injection) + critic (MSE). Both consume the
# same rollout_data_ref each step.
#
# Prerequisite: ACTOR_SFT_CKPT and CRITIC_SL_CKPT must both have nla_meta.yaml
# with matching token IDs / prompt templates.

: "${RL_PARQUET:?set RL_PARQUET to the Stage 3c parquet path}"
: "${INSTRUCT_MODEL:?HF base instruct model (e.g. Qwen/Qwen2.5-7B-Instruct) — supplies tokenizer/config}"
: "${ACTOR_SFT_CKPT:?DCP iter dir from actor_sft.sh (e.g. .../iter_0002000) — supplies weights + nla_meta.yaml}"
: "${CRITIC_SL_CKPT:?HF dir from critic_sft.sh (e.g. .../iter_0002000/hf) — already truncated, K is in its config.json}"
: "${RUN_DIR:?}"

PYTHON="${PYTHON:-python}"
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-train.py}"
ACTOR_LOAD_CKPT="${ACTOR_LOAD_CKPT:-$ACTOR_SFT_CKPT}"
ACTOR_REF_CKPT="${ACTOR_REF_CKPT:-$ACTOR_LOAD_CKPT}"
ACTOR_SIDECAR_SOURCE="${ACTOR_SIDECAR_SOURCE:-$ACTOR_SFT_CKPT}"

# --kl-coef is a NO-OP for GRPO (get_grpo_returns discards the kl tensor).
# --use-kl-loss is the correct path (adds KL to policy loss, logs train/kl_loss)
# but it's action="store_true" — once passed, callers can't un-pass it. Gate on
# env var so KL_LOSS_COEF=0 drops the flags entirely (small-scale test runs uses this to
# skip the --ref-load / DCP→HF conversion step).
KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-k1}"
case "$KL_LOSS_TYPE" in
    k1 | k2 | k3 | low_var_kl) ;;
    *)
        echo "unsupported KL_LOSS_TYPE=$KL_LOSS_TYPE" >&2
        exit 2
        ;;
esac
if "$PYTHON" -c "import sys; sys.exit(0 if float('$KL_LOSS_COEF') != 0 else 1)"; then
    KL_FLAGS=(--use-kl-loss --kl-loss-coef "$KL_LOSS_COEF" --kl-loss-type "$KL_LOSS_TYPE")
else
    KL_FLAGS=()
fi

# Per-step 1.1GB embedding dump for nla_generate — /tmp is disk (overlay fs),
# /dev/shm is tmpfs (RAM). ~1.5s → ~0.1s per step. Zero code change.
export NLA_EMBED_DUMP_DIR="${NLA_EMBED_DUMP_DIR:-/dev/shm/nla}"
mkdir -p "$NLA_EMBED_DUMP_DIR"

# actor_dp may differ from critic_dp since _repartition_for_critic —
# critic rank i pulls actor partitions [i, i+critic_dp, ...]. Defaults keep
# them symmetric for standalone test runs.
ACTOR_NODES=${ACTOR_NODES:-1}
ACTOR_GPUS=${ACTOR_GPUS:-8}
CRITIC_NODES=${CRITIC_NODES:-$ACTOR_NODES}
CRITIC_GPUS=${CRITIC_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
NLA_ACTOR_GPUS="${NLA_ACTOR_GPUS:-$ACTOR_GPUS}"
NLA_CRITIC_GPUS="${NLA_CRITIC_GPUS:-$CRITIC_GPUS}"
NLA_ROLLOUT_GPUS="${NLA_ROLLOUT_GPUS:-$ROLLOUT_GPUS}"
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-150}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-300}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-128}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-constant}
NLA_CUSTOM_ROLLOUT_LOG_FUNCTION=${NLA_CUSTOM_ROLLOUT_LOG_FUNCTION:-nla.rollout.rl_metrics.log_rollout_data}
NLA_CUSTOM_TRAIN_GUARD_FUNCTION=${NLA_CUSTOM_TRAIN_GUARD_FUNCTION:-}
NLA_ROLLOUT_PROMPT_BATCH="${NLA_ROLLOUT_PROMPT_BATCH:-$ROLLOUT_BATCH_SIZE}"
NLA_ROLLOUT_SAMPLES_PER_PROMPT="${NLA_ROLLOUT_SAMPLES_PER_PROMPT:-$N_SAMPLES_PER_PROMPT}"
NLA_ROLLOUT_GENERATED_SAMPLES="${NLA_ROLLOUT_GENERATED_SAMPLES:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
NLA_ROLLOUT_GLOBAL_BATCH="${NLA_ROLLOUT_GLOBAL_BATCH:-$GLOBAL_BATCH_SIZE}"
if [[ "${NLA_ROLLOUT_GLOBAL_MATCH:-}" == "" ]]; then
    if [[ "$GLOBAL_BATCH_SIZE" == "$NLA_ROLLOUT_GENERATED_SAMPLES" ]]; then
        NLA_ROLLOUT_GLOBAL_MATCH=1
    else
        NLA_ROLLOUT_GLOBAL_MATCH=0
    fi
fi
export NLA_ROLLOUT_PROMPT_BATCH
export NLA_ROLLOUT_SAMPLES_PER_PROMPT
export NLA_ROLLOUT_GENERATED_SAMPLES
export NLA_ROLLOUT_GLOBAL_BATCH
export NLA_ROLLOUT_GLOBAL_MATCH
export NLA_ACTOR_GPUS
export NLA_CRITIC_GPUS
export NLA_ROLLOUT_GPUS

echo "[NLA RL CONFIG] prompts=${NLA_ROLLOUT_PROMPT_BATCH} samples_per_prompt=${NLA_ROLLOUT_SAMPLES_PER_PROMPT} generated_samples=${NLA_ROLLOUT_GENERATED_SAMPLES} global_batch_size=${NLA_ROLLOUT_GLOBAL_BATCH} global_matches_rollout=${NLA_ROLLOUT_GLOBAL_MATCH}" >&2
echo "[NLA RL CONFIG] topology workspace_gpus=${NLA_WORKSPACE_GPUS:-unset} actor_gpus=${ACTOR_GPUS} critic_gpus=${CRITIC_GPUS} rollout_gpus=${ROLLOUT_GPUS} rollout_gpus_per_engine=${NLA_ROLLOUT_GPUS_PER_ENGINE:-unset} sglang_tp_size=${NLA_SGLANG_TP_SIZE:-unset} sglang_base_gpu_id=${NLA_SGLANG_BASE_GPU_ID:-unset}" >&2

ROLLOUT_LOG_ARGS=()
if [[ -n "$NLA_CUSTOM_ROLLOUT_LOG_FUNCTION" ]]; then
    ROLLOUT_LOG_ARGS=(--custom-rollout-log-function-path "$NLA_CUSTOM_ROLLOUT_LOG_FUNCTION")
fi

TRAIN_GUARD_ARGS=()
if [[ -n "$NLA_CUSTOM_TRAIN_GUARD_FUNCTION" ]]; then
    TRAIN_GUARD_ARGS=(--custom-train-guard-function-path "$NLA_CUSTOM_TRAIN_GUARD_FUNCTION")
fi

SAVE_INTERVAL_ARGS=()
SAVE_INTERVAL_VALUE="${SAVE_INTERVAL-100}"
case "$SAVE_INTERVAL_VALUE" in
    "" | "none" | "None" | "NONE" | "null" | "Null" | "NULL")
        ;;
    *)
        SAVE_INTERVAL_ARGS=(--save-interval "$SAVE_INTERVAL_VALUE")
        ;;
esac

if [[ -n "${NLA_SAVE_ITERATIONS:-}" ]]; then
    echo "[NLA SAVE SCHEDULE] explicit_iterations=${NLA_SAVE_ITERATIONS} miles_save_interval=disabled"
elif [[ ${#SAVE_INTERVAL_ARGS[@]} -gt 0 ]]; then
    echo "[NLA SAVE SCHEDULE] interval=${SAVE_INTERVAL_VALUE}"
else
    echo "[NLA SAVE SCHEDULE] disabled"
fi

is_truthy() {
    case "${1:-}" in
        1 | true | TRUE | True | yes | YES | Yes | on | ON | On)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

FINETUNE_ARGS=()
if is_truthy "${FINETUNE:-0}"; then
    FINETUNE_ARGS=(--finetune)
fi

NO_LOAD_OPTIM_ARGS=()
if is_truthy "${NO_LOAD_OPTIM:-0}"; then
    NO_LOAD_OPTIM_ARGS=(--no-load-optim)
fi

ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-grpo}
QKV_FORMAT=${QKV_FORMAT:-thd}
if [[ "$QKV_FORMAT" != "thd" && "$QKV_FORMAT" != "bshd" ]]; then
    echo "QKV_FORMAT must be thd or bshd, got: $QKV_FORMAT" >&2
    exit 2
fi

NORMALIZE_ADVANTAGES_ARGS=()
if is_truthy "${NORMALIZE_ADVANTAGES:-0}"; then
    NORMALIZE_ADVANTAGES_ARGS=(--normalize-advantages)
fi

REWARDS_NORMALIZATION_ARGS=()
if ! is_truthy "${REWARDS_NORMALIZATION:-1}"; then
    REWARDS_NORMALIZATION_ARGS=(--disable-rewards-normalization)
fi

GRPO_STD_NORMALIZATION_ARGS=()
if ! is_truthy "${GRPO_STD_NORMALIZATION:-1}"; then
    GRPO_STD_NORMALIZATION_ARGS=(--disable-grpo-std-normalization)
fi

export NLA_PHASE_METRICS="${NLA_PHASE_METRICS:-${NLA_SYSTEM_METRICS:-0}}"
export NLA_PHASE_METRICS_ALL_GPUS="${NLA_PHASE_METRICS_ALL_GPUS:-1}"
export NLA_PHASE_METRICS_WANDB="${NLA_PHASE_METRICS_WANDB:-1}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/dev/shm/nano30b-nla-pilot/hf_modules_cache}"
mkdir -p "$HF_MODULES_CACHE"

if is_truthy "${NLA_PREWARM_HF_MODULES:-0}"; then
    "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

from transformers.dynamic_module_utils import get_class_from_dynamic_module

for name in ("INSTRUCT_MODEL", "ACTOR_REF_CKPT", "CRITIC_SL_CKPT"):
    path_text = os.environ.get(name)
    if not path_text:
        continue
    path = Path(path_text)
    config_path = path / "config.json"
    if not config_path.is_file():
        continue
    auto_map = json.loads(config_path.read_text()).get("auto_map") or {}
    class_ref = auto_map.get("AutoModelForCausalLM") or auto_map.get("AutoModel")
    if not class_ref:
        continue
    cls = get_class_from_dynamic_module(class_ref, path, trust_remote_code=True)
    print(f"[NLA RL CONFIG] prewarmed HF remote code {name}={path} class={cls.__module__}.{cls.__name__}", flush=True)
PY
fi

json_escape() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//$'\n'/\\n}"
    value="${value//$'\r'/\\r}"
    value="${value//$'\t'/\\t}"
    printf '%s' "$value"
}

NLA_TRAIN_ENV_KEYS=${NLA_TRAIN_ENV_KEYS:-"PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF HF_MODULES_CACHE NLA_PREWARM_HF_MODULES NLA_EMBED_DUMP_DIR NLA_BYPASS_ROUTER NLA_BF16_B64_EMBEDS NLA_WEIGHT_UPDATE_LOG_EVERY NLA_FREEZE_CRITIC_TRAIN NLA_ASSERT_PACKED_EQUIV NLA_ASSERT_ACTOR_PACKED_EQUIV NLA_ACTOR_PACKED_EQUIV_RTOL NLA_ACTOR_PACKED_EQUIV_ATOL NLA_TRAIN_MAMBA_KERNEL_MODE NLA_ACTOR_TRAIN_MAMBA_KERNEL_MODE NLA_CRITIC_TRAIN_MAMBA_KERNEL_MODE NLA_CRITIC_FWD_DISABLE_MAMBA_FAST_PATH NLA_CRITIC_REWARD_LAYOUT_MSE_RATIO_TOL NLA_CRITIC_TRAIN_MODE_MSE_RATIO_TOL NLA_MIN_CRITIC_RETAINED_FRACTION NLA_ROLLOUT_TEXT_DUMP NLA_ROLLOUT_PROMPT_BATCH NLA_ROLLOUT_SAMPLES_PER_PROMPT NLA_ROLLOUT_GENERATED_SAMPLES NLA_ROLLOUT_GLOBAL_BATCH NLA_ROLLOUT_GLOBAL_MATCH NLA_WORKSPACE_GPUS NLA_ACTOR_GPUS NLA_CRITIC_GPUS NLA_ROLLOUT_GPUS NLA_ROLLOUT_GPUS_PER_ENGINE NLA_SGLANG_TP_SIZE NLA_SGLANG_BASE_GPU_ID NLA_REF_LOG_PROBS_PLACEMENT KL_LOSS_TYPE QKV_FORMAT NLA_TRAIN_GUARD_MAX_LOGPROB_ABS_DIFF NLA_TRAIN_GUARD_CONSECUTIVE_STEPS NLA_TRAIN_GUARD_METRIC NLA_TRAIN_GUARD_RULES_JSON NLA_ROLLOUT_GUARD_RULES_JSON GRADIENT_CHECKPOINTING OFFLOAD_TRAIN OFFLOAD_ROLLOUT OFFLOAD_ROLLOUT_LEVEL FSDP_CPU_OFFLOAD FSDP_CPU_BACKEND COLOCATE NLA_SYSTEM_METRICS NLA_SYSTEM_METRICS_INTERVAL_STEPS NLA_SYSTEM_METRICS_NVSMI_INTERVAL_STEPS NLA_PHASE_METRICS NLA_PHASE_METRICS_ALL_GPUS NLA_PHASE_METRICS_WANDB NLA_ROUTER_METRICS NLA_BACKUP_REMOTE NLA_BACKUP_STORAGE_CLS"}
TRAIN_ENV_JSON="{"
TRAIN_ENV_SEP=""
for name in $NLA_TRAIN_ENV_KEYS; do
    if [[ -n "${!name+x}" ]]; then
        TRAIN_ENV_JSON+="${TRAIN_ENV_SEP}\"$(json_escape "$name")\":\"$(json_escape "${!name}")\""
        TRAIN_ENV_SEP=","
    fi
done
TRAIN_ENV_JSON+="}"
TRAIN_ENV_ARGS=()
if [[ "$TRAIN_ENV_JSON" != "{}" ]]; then
    TRAIN_ENV_ARGS=(--train-env-vars "$TRAIN_ENV_JSON")
fi

WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ARGS=()
if [[ "$WANDB_MODE" != "disabled" ]]; then
    WANDB_DIR="${WANDB_DIR:-$RUN_DIR/wandb}"
    WANDB_PROJECT="${WANDB_PROJECT:-nano30b-nla-pilot}"
    WANDB_GROUP="${WANDB_GROUP:-nano-rl}"
    mkdir -p "$WANDB_DIR"
    WANDB_ARGS=(
        --use-wandb
        --wandb-mode "$WANDB_MODE"
        --wandb-dir "$WANDB_DIR"
        --wandb-project "$WANDB_PROJECT"
        --wandb-group "$WANDB_GROUP"
        --disable-wandb-random-suffix
        --wandb-always-use-train-step
    )
    if [[ -n "${WANDB_RUN_ID:-}" ]]; then
        WANDB_ARGS+=(--wandb-run-id "$WANDB_RUN_ID")
    fi
fi

FSDP_ARGS=()
if [[ -n "${FSDP_REDUCE_DTYPE:-}" ]]; then
    FSDP_ARGS+=(--fsdp-reduce-dtype "$FSDP_REDUCE_DTYPE")
fi
if is_truthy "${FSDP_DISABLE_BACKWARD_PREFETCH:-0}"; then
    FSDP_ARGS+=(--fsdp-disable-backward-prefetch)
fi
if is_truthy "${FSDP_CPU_OFFLOAD:-0}"; then
    FSDP_ARGS+=(--fsdp-cpu-offload)
fi
if [[ -n "${FSDP_CPU_BACKEND:-}" ]]; then
    FSDP_ARGS+=(--fsdp-cpu-backend "$FSDP_CPU_BACKEND")
fi

THROUGHPUT_ARGS=()
if is_truthy "${GRADIENT_CHECKPOINTING:-0}"; then
    THROUGHPUT_ARGS+=(--gradient-checkpointing)
fi
if is_truthy "${OFFLOAD_TRAIN:-0}"; then
    THROUGHPUT_ARGS+=(--offload-train)
fi
if is_truthy "${OFFLOAD_ROLLOUT:-0}"; then
    THROUGHPUT_ARGS+=(--offload-rollout)
fi
if [[ -n "${OFFLOAD_ROLLOUT_LEVEL:-}" ]]; then
    read -r -a OFFLOAD_ROLLOUT_LEVEL_PARTS <<< "$OFFLOAD_ROLLOUT_LEVEL"
    if [[ "${#OFFLOAD_ROLLOUT_LEVEL_PARTS[@]}" -gt 0 ]]; then
        THROUGHPUT_ARGS+=(--offload-rollout-level "${OFFLOAD_ROLLOUT_LEVEL_PARTS[@]}")
    fi
fi
if is_truthy "${COLOCATE:-0}"; then
    THROUGHPUT_ARGS+=(--colocate)
fi

# Dynamic batching is OFF by default (--use-dynamic-batch-size not set).
# With micro-batch-size=4: critic gets 4 samples/microbatch regardless of length.
# To enable (packs by token budget instead — critic tokens are ~300 each):
#   --use-dynamic-batch-size --max-tokens-per-gpu 4096
# Safe for the critic: _swap_rollout_to_critic_tokens sets total_lengths to
# critic token lengths, which is what get_data_iterator reads for packing.

"$PYTHON" "$TRAIN_ENTRYPOINT" \
    --train-backend "${TRAIN_BACKEND:-fsdp}" \
    --custom-actor-cls-path "${ACTOR_CLS:-nla.train_actor.NLAFSDPActor}" \
    --loss-type policy_loss \
    --advantage-estimator "$ADVANTAGE_ESTIMATOR" \
    --force-use-critic \
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT" \
    --rollout-function-path miles.rollout.sglang_rollout.generate_rollout \
    --custom-generate-function-path nla.rollout.nla_generate.generate \
    --custom-rm-path nla.reward.nla_rm \
    --data-source-path nla.data_source.NLADataSource \
    --prompt-data "$RL_PARQUET" \
    --input-key prompt \
    --hf-checkpoint "$INSTRUCT_MODEL" \
    --ref-load "$ACTOR_REF_CKPT" \
    --load "$ACTOR_LOAD_CKPT" \
    --nla-sidecar-source "$ACTOR_SIDECAR_SOURCE" \
    --save "$RUN_DIR/actor" \
    --critic-load "$CRITIC_SL_CKPT" \
    --critic-save "$RUN_DIR/critic" \
    --critic-lr "${CRITIC_LR:-1e-5}" \
    --actor-num-nodes "$ACTOR_NODES" \
    --actor-num-gpus-per-node "$ACTOR_GPUS" \
    --critic-num-nodes "$CRITIC_NODES" \
    --critic-num-gpus-per-node "$CRITIC_GPUS" \
    --rollout-num-gpus "$ROLLOUT_GPUS" \
    --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN" \
    --rollout-max-context-len "$ROLLOUT_MAX_CONTEXT_LEN" \
    `# REQUIRED for NLA — radix cache keys on token IDs, but we inject raw activation` \
    `# vectors at the marker token. Cache would hit across DIFFERENT activations that` \
    `# share the same marker token → silent wrong output. DO NOT REMOVE to "optimize".` \
    --sglang-disable-radix-cache \
    --router-history-backend none \
    `# cache_aware (default) builds prefix tree storing request bodies — with NLA` \
    `# input_embeds (~6-12MB each) that IS the leak. round_robin: no tree.` \
    `# CB-disable + short retry: large bodies drop connections, CB false-positive` \
    `# locks engines 60s. NLA workload is uniform — no routing smarts needed.` \
    --router-policy round_robin \
    --router-disable-circuit-breaker \
    --router-retry-max-backoff-ms 500 --router-retry-max-retries 2 \
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --micro-batch-size "${ACTOR_MICRO:-4}" \
    --qkv-format "$QKV_FORMAT" \
    --clip-grad "${CLIP_GRAD:-1.0}" \
    --attn-implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
    --lr "${ACTOR_LR:-1e-6}" --lr-decay-style "$LR_DECAY_STYLE" \
    "${FINETUNE_ARGS[@]}" \
    "${NO_LOAD_OPTIM_ARGS[@]}" \
    "${KL_FLAGS[@]}" \
    "${NORMALIZE_ADVANTAGES_ARGS[@]}" \
    "${REWARDS_NORMALIZATION_ARGS[@]}" \
    "${GRPO_STD_NORMALIZATION_ARGS[@]}" \
    "${TRAIN_ENV_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${ROLLOUT_LOG_ARGS[@]}" \
    "${TRAIN_GUARD_ARGS[@]}" \
    "${SAVE_INTERVAL_ARGS[@]}" \
    "${FSDP_ARGS[@]}" \
    "${THROUGHPUT_ARGS[@]}" \
    --loss-mask-type "${LOSS_MASK_TYPE:-qwen}" \
    "$@"
