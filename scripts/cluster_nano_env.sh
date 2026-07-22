#!/usr/bin/env bash
# Source this file on the SLURM cluster before running the Nano30B NLA harness.
# It pins the remote-code and model caches to Lustre paths that avoid the
# quota-limited default Hugging Face cache.

export NANO_PROJECT_ROOT="${NANO_PROJECT_ROOT:-/lustre/fsw/portfolios/llmservice/users/rigarg/mech_interp/research-projects/nano30b-nla-pilot}"
export NANO_PYTHON="${NANO_PYTHON:-/lustre/fs11/portfolios/llmservice/projects/llmservice_nemo_mlops/users/rigarg/conda_env/nla/bin/python}"

export NANO_MODEL_ID="${NANO_MODEL_ID:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
export NANO_MODEL_REVISION="${NANO_MODEL_REVISION:-cbd3fa9f933d55ef16a84236559f4ee2a0526848}"
export NANO_TOKENIZER_REVISION="${NANO_TOKENIZER_REVISION:-${NANO_MODEL_REVISION}}"
export NANO_CONDA_PREFIX="${NANO_CONDA_PREFIX:-$(cd "$(dirname "${NANO_PYTHON}")/.." && pwd)}"
export NANO_LIBSTDCXX="${NANO_LIBSTDCXX:-${NANO_CONDA_PREFIX}/lib/libstdc++.so.6}"

export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/lustre/fsw/portfolios/llmservice/users/rigarg/mech_interp/research-projects/.hf_nla_cache/modules}"
export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/llmservice/users/rigarg/.cache-models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
# Cluster login shells put GCC's libstdc++ first; selective_scan_cuda needs the
# newer CXXABI symbols shipped with the conda env. Preload only libstdc++ so
# system shells do not pick up unrelated conda libraries such as libtinfo.
case ":${LD_PRELOAD:-}:" in
  *":${NANO_LIBSTDCXX}:"*) ;;
  *) export LD_PRELOAD="${NANO_LIBSTDCXX}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-1}"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  cat <<EOF
Source this file before running the harness:

  source scripts/cluster_nano_env.sh
  cd "\$NANO_PROJECT_ROOT"
  "\$NANO_PYTHON" scripts/nano_introspection.py --load-mode meta --local-files-only

Current key settings:
  NANO_PROJECT_ROOT=$NANO_PROJECT_ROOT
  NANO_PYTHON=$NANO_PYTHON
  NANO_MODEL_REVISION=$NANO_MODEL_REVISION
  HF_MODULES_CACHE=$HF_MODULES_CACHE
  HF_HOME=$HF_HOME
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
EOF
fi
