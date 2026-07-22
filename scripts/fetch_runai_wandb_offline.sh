#!/usr/bin/env bash
set -euo pipefail

# Fetch a RunAI offline W&B directory to the local Mac and optionally sync it.
# This script is intentionally small: the only secret it reads is WANDB_API_KEY
# from the environment or a local dotenv file, and it never prints the value.

RUNAI_BIN="${RUNAI_BIN:-/Users/rigarg/.runai/bin/2.116.2/runai}"
PROJECT="${RUNAI_PROJECT:-trustworthy-ai-inference}"
WORKSPACE="${RUNAI_WORKSPACE:-train-dev}"
LOCAL_ROOT="${LOCAL_ROOT:-runs/wandb_offline}"
ENV_FILE="${ENV_FILE:-.env}"
SYNC="${SYNC:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-nano30b-nla-pilot}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/fetch_runai_wandb_offline.sh RUN_DIR [OFFLINE_RUN_NAME]

Examples:
  scripts/fetch_runai_wandb_offline.sh \
    /workspace/interp/outputs/.../av-r27-run-name

  SYNC=1 scripts/fetch_runai_wandb_offline.sh \
    /workspace/interp/outputs/.../av-r27-run-name \
    offline-run-20260526_164318-ggrd168y
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

run_dir="$1"
offline_name="${2:-}"
run_name="$(basename "${run_dir}")"

if [[ -z "${offline_name}" ]]; then
  offline_name="$("${RUNAI_BIN}" workspace exec "${WORKSPACE}" -p "${PROJECT}" -- \
    bash -lc "find '${run_dir}/wandb' -maxdepth 1 -type d -name 'offline-run-*' -printf '%f\n' 2>/dev/null | sort | tail -1")"
fi

if [[ -z "${offline_name}" ]]; then
  echo "No offline-run-* directory found under ${run_dir}/wandb" >&2
  exit 1
fi

local_dir="${LOCAL_ROOT}/${run_name}"
archive="${local_dir}/wandb_offline.tgz"
mkdir -p "${local_dir}"

"${RUNAI_BIN}" workspace exec "${WORKSPACE}" -p "${PROJECT}" -- \
  tar -C "${run_dir}/wandb" -czf - "${offline_name}" >"${archive}"
tar -C "${local_dir}" -xzf "${archive}"

echo "fetched=${local_dir}/${offline_name}"
echo "archive=${archive}"

if [[ "${SYNC}" == "0" ]]; then
  exit 0
fi

set -a
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
fi
set +a
if [[ -z "${WANDB_API_KEY:-}" && -n "${wandb_api_key:-}" ]]; then
  export WANDB_API_KEY="${wandb_api_key}"
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "SYNC=1 requested, but WANDB_API_KEY/wandb_api_key is not available." >&2
  exit 1
fi

export WANDB_MODE=online
"${PYTHON_BIN}" -m wandb sync --project "${WANDB_PROJECT}" "${local_dir}/${offline_name}"
