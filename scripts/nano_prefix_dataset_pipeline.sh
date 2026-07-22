#!/usr/bin/env bash
set -euo pipefail

# Build a teacher-backed Nano activation dataset from exact token prefixes.
# This is the hero-prep path for R33 and is reusable for other residual layers.
# It does not launch AR/AV training.

CODE_ROOT="${CODE_ROOT:-/workspace/interp/code/nano30b-nla-pilot-current}"
cd "$CODE_ROOT"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="$CODE_ROOT/external/natural_language_autoencoders:$CODE_ROOT/external/natural_language_autoencoders/Miles:$CODE_ROOT:${PYTHONPATH:-}"

PY="${PY:-/workspace/interp/.venv/bin/python}"
MODEL="${MODEL:-/workspace/interp/models/nano-30b-a3b-bf16-hf}"
SOURCE_PARQUET="${SOURCE_PARQUET:-/workspace/interp/outputs/nano30b-nla-pilot/ar-r27-r30-fullscan-20260528T234403Z/R_27/ar_sft.parquet}"
CONTRACT="${CONTRACT:-/workspace/interp/outputs/nano30b-nla-pilot/ar-r27-r30-fullscan-20260528T234403Z/R_27/ar_sft.parquet.nla_meta.yaml}"
LAYER="${LAYER:-33}"
LAYERS="${LAYERS:-R${LAYER}}"
ROW_START="${ROW_START:-0}"
ROW_LIMIT="${ROW_LIMIT:-}"
SLUG="${SLUG:-r${LAYER}_prefix_fullscan275396}"
OUT="${OUT:-/workspace/interp/outputs/nano30b-nla-pilot/${SLUG}}"
EXTRACT_ROOT="${EXTRACT_ROOT:-$OUT/extract_prefix_${SLUG}}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SOURCE_BATCH_SIZE="${SOURCE_BATCH_SIZE:-4096}"
EXTRACT_DEVICES="${EXTRACT_DEVICES:-}"
EXTRACT_SHARD_ALIGNMENT="${EXTRACT_SHARD_ALIGNMENT:-document_batch}"
EXTRACT_DETERMINISTIC_ALGORITHMS="${EXTRACT_DETERMINISTIC_ALGORITHMS:-0}"
EXTRACT_ALLOW_TF32="${EXTRACT_ALLOW_TF32:-0}"
EXTRACT_CUDNN_BENCHMARK="${EXTRACT_CUDNN_BENCHMARK:-0}"
EXTRACT_FLOAT32_MATMUL_PRECISION="${EXTRACT_FLOAT32_MATMUL_PRECISION:-highest}"
EXTRACT_CUBLAS_WORKSPACE_CONFIG="${EXTRACT_CUBLAS_WORKSPACE_CONFIG:-}"
EXTRACT_SEED="${EXTRACT_SEED:-}"
EXPECTED_D_MODEL="${EXPECTED_D_MODEL:-2688}"
EXPECTED_SOURCE_PARQUET_SHA256="${EXPECTED_SOURCE_PARQUET_SHA256:-}"
BUILD_AR="${BUILD_AR:-1}"
BUILD_AV="${BUILD_AV:-1}"
PREP_CRITIC="${PREP_CRITIC:-1}"
RUN_VERIFY="${RUN_VERIFY:-1}"
PUBLICATION_MODE="${PUBLICATION_MODE:-0}"
MODEL_FINGERPRINT_JSON="${MODEL_FINGERPRINT_JSON:-}"
RUNTIME_PROVENANCE_JSON="${RUNTIME_PROVENANCE_JSON:-}"
CONTENT_FAMILY_MANIFEST="${CONTENT_FAMILY_MANIFEST:-}"
CONTENT_FAMILY_MANIFEST_SHA256="${CONTENT_FAMILY_MANIFEST_SHA256:-}"
CRITIC_VALUE_HEAD_INIT="${CRITIC_VALUE_HEAD_INIT:-identity}"
CRITIC_INITIALIZATION_SEED="${CRITIC_INITIALIZATION_SEED:-0}"
CRITIC_VALUE_HEAD_ROTATION_RADIANS="${CRITIC_VALUE_HEAD_ROTATION_RADIANS:-0.2}"
CRITIC_ROUTER_INIT="${CRITIC_ROUTER_INIT:-pretrained}"
CRITIC_ROUTER_RELATIVE_STD="${CRITIC_ROUTER_RELATIVE_STD:-0.01}"
CRITIC="${CRITIC:-/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-critic-init/nano-ar-r${LAYER}-critic-init}"

mkdir -p "$OUT"

if [ ! -f "$SOURCE_PARQUET" ]; then
  echo "source parquet not found: $SOURCE_PARQUET" >&2
  exit 1
fi
if [ ! -f "$CONTRACT" ]; then
  echo "AR sidecar contract not found: $CONTRACT" >&2
  exit 1
fi
if [ -n "$EXPECTED_SOURCE_PARQUET_SHA256" ]; then
  "$PY" - "$SOURCE_PARQUET" "$EXPECTED_SOURCE_PARQUET_SHA256" <<'PY'
import hashlib
from pathlib import Path
import sys

source = Path(sys.argv[1])
expected = sys.argv[2]
digest = hashlib.sha256()
with source.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
actual = digest.hexdigest()
if actual != expected:
    raise SystemExit(
        f"source parquet SHA-256 mismatch: expected {expected}, got {actual}"
    )
print({"source_parquet": str(source), "sha256": actual})
PY
fi
if [ "$PUBLICATION_MODE" = "1" ]; then
  if [ ! -f "$MODEL_FINGERPRINT_JSON" ]; then
    echo "publication model fingerprint not found: $MODEL_FINGERPRINT_JSON" >&2
    exit 1
  fi
  if [ ! -f "$RUNTIME_PROVENANCE_JSON" ]; then
    echo "publication runtime provenance not found: $RUNTIME_PROVENANCE_JSON" >&2
    exit 1
  fi
  if [ "$RUN_VERIFY" = "1" ] && [ ! -f "$CONTENT_FAMILY_MANIFEST" ]; then
    echo "publication content family manifest not found: $CONTENT_FAMILY_MANIFEST" >&2
    exit 1
  fi
  if [ "$RUN_VERIFY" = "1" ] && [ -z "$CONTENT_FAMILY_MANIFEST_SHA256" ]; then
    echo "publication content family manifest SHA-256 is required" >&2
    exit 1
  fi
  if [ "$EXTRACT_DETERMINISTIC_ALGORITHMS" != "1" ] || \
     [ "$EXTRACT_ALLOW_TF32" != "0" ] || \
     [ "$EXTRACT_CUDNN_BENCHMARK" != "0" ] || \
     [ "$EXTRACT_FLOAT32_MATMUL_PRECISION" != "highest" ] || \
     [ -z "$EXTRACT_CUBLAS_WORKSPACE_CONFIG" ] || \
     [ -z "$EXTRACT_SEED" ]; then
    echo "publication extraction requires the deterministic execution profile" >&2
    exit 1
  fi
fi

VERIFY_SPLIT_ARGS=()
if [ -n "$CONTENT_FAMILY_MANIFEST" ]; then
  VERIFY_SPLIT_ARGS+=(
    --content-family-manifest "$CONTENT_FAMILY_MANIFEST"
    --content-family-manifest-sha256 "$CONTENT_FAMILY_MANIFEST_SHA256"
  )
fi

EXPECTED_ROWS="${EXPECTED_ROWS:-$("$PY" - "$SOURCE_PARQUET" "$ROW_START" "$ROW_LIMIT" <<'PY'
from pathlib import Path
import sys
import pyarrow.parquet as pq

source = Path(sys.argv[1])
row_start = int(sys.argv[2])
row_limit = sys.argv[3]
total = pq.ParquetFile(source).metadata.num_rows
remaining = max(0, total - row_start)
if row_limit:
    remaining = min(remaining, int(row_limit))
print(remaining)
PY
)}"

BASE="$OUT/base_R${LAYER}_${SLUG}.parquet"
AR_SFT="$OUT/ar_sft_R${LAYER}_${SLUG}.parquet"
AV_SFT="$OUT/av_sft_R${LAYER}_${SLUG}.parquet"
AR_VERIFY="$OUT/verify_ar_R${LAYER}_${SLUG}.json"
AV_VERIFY="$OUT/verify_av_R${LAYER}_${SLUG}.json"

if [ ! -f "$BASE" ]; then
  extractor="scripts/nano_prefix_activation_extract.py"
  if [ -n "$EXTRACT_DEVICES" ]; then
    extractor="scripts/nano_prefix_sharded_extract.py"
  fi
  extract_args=(
    "$extractor"
    --source-parquet "$SOURCE_PARQUET"
    --output-root "$EXTRACT_ROOT"
    --layers "$LAYERS"
    --row-start "$ROW_START"
    --source-batch-size "$SOURCE_BATCH_SIZE"
    --batch-size "$BATCH_SIZE"
    --model-id "$MODEL"
    --local-files-only
    --overwrite
  )
  if [ "$EXTRACT_DETERMINISTIC_ALGORITHMS" = "1" ]; then
    extract_args+=(--deterministic-algorithms)
  else
    extract_args+=(--no-deterministic-algorithms)
  fi
  if [ "$EXTRACT_ALLOW_TF32" = "1" ]; then
    extract_args+=(--allow-tf32)
  else
    extract_args+=(--no-allow-tf32)
  fi
  if [ "$EXTRACT_CUDNN_BENCHMARK" = "1" ]; then
    extract_args+=(--cudnn-benchmark)
  else
    extract_args+=(--no-cudnn-benchmark)
  fi
  extract_args+=(
    --float32-matmul-precision "$EXTRACT_FLOAT32_MATMUL_PRECISION"
  )
  if [ -n "$EXTRACT_CUBLAS_WORKSPACE_CONFIG" ]; then
    extract_args+=(
      --cublas-workspace-config "$EXTRACT_CUBLAS_WORKSPACE_CONFIG"
    )
  fi
  if [ -n "$EXTRACT_SEED" ]; then
    export PYTHONHASHSEED="$EXTRACT_SEED"
    extract_args+=(--seed "$EXTRACT_SEED")
  fi
  if [ -n "$EXTRACT_DEVICES" ]; then
    extract_args+=(
      --devices "$EXTRACT_DEVICES"
      --shard-alignment "$EXTRACT_SHARD_ALIGNMENT"
    )
  fi
  if [ -n "$ROW_LIMIT" ]; then
    extract_args+=(--row-limit "$ROW_LIMIT")
  fi
  if [ "$PUBLICATION_MODE" = "1" ]; then
    extract_args+=(
      --publication-mode
      --model-fingerprint-json "$MODEL_FINGERPRINT_JSON"
      --runtime-provenance-json "$RUNTIME_PROVENANCE_JSON"
    )
  fi
  "$PY" "${extract_args[@]}"
  cp "$EXTRACT_ROOT/R_${LAYER}/base.parquet" "$BASE"
  cp "$EXTRACT_ROOT/R_${LAYER}/base.parquet.nla_meta.yaml" "$BASE.nla_meta.yaml"
  cp "$EXTRACT_ROOT/R_${LAYER}/base.parquet.metadata.json" "$BASE.metadata.json"
fi

"$PY" - \
  "$BASE" \
  "$EXPECTED_ROWS" \
  "$EXPECTED_D_MODEL" \
  "$PUBLICATION_MODE" \
  "$MODEL_FINGERPRINT_JSON" \
  "$RUNTIME_PROVENANCE_JSON" \
  "$EXTRACT_DETERMINISTIC_ALGORITHMS" \
  "$EXTRACT_ALLOW_TF32" \
  "$EXTRACT_CUDNN_BENCHMARK" \
  "$EXTRACT_FLOAT32_MATMUL_PRECISION" \
  "$EXTRACT_CUBLAS_WORKSPACE_CONFIG" \
  "$EXTRACT_SEED" <<'PY'
from pathlib import Path
import json
import sys
import pyarrow.parquet as pq
import pyarrow.compute as pc
import yaml

base = Path(sys.argv[1])
expected_rows = int(sys.argv[2])
expected_d_model = int(sys.argv[3])
publication_mode = sys.argv[4] == "1"
model_fingerprint_json = Path(sys.argv[5]) if sys.argv[5] else None
runtime_provenance_json = Path(sys.argv[6]) if sys.argv[6] else None
expected_deterministic = sys.argv[7] == "1"
expected_allow_tf32 = sys.argv[8] == "1"
expected_cudnn_benchmark = sys.argv[9] == "1"
expected_matmul_precision = sys.argv[10]
expected_cublas_workspace = sys.argv[11]
expected_seed = int(sys.argv[12]) if sys.argv[12] else None
pf = pq.ParquetFile(base)
if pf.metadata.num_rows != expected_rows:
    raise SystemExit(f"base rows {pf.metadata.num_rows} != expected {expected_rows}")
table = pq.read_table(base, columns=["activation_vector", "api_explanation"])
first = table.column("activation_vector")[0].as_py()
if len(first) != expected_d_model:
    raise SystemExit(f"d_model {len(first)} != expected {expected_d_model}")
empty = pc.sum(pc.equal(pc.utf8_trim_whitespace(table.column("api_explanation")), "")).as_py()
if empty:
    raise SystemExit(f"empty api_explanation rows: {empty}")
sidecar = yaml.safe_load(Path(str(base) + ".nla_meta.yaml").read_text())
if int(sidecar.get("row_count", -1)) != expected_rows:
    raise SystemExit("base sidecar row_count mismatch")
if publication_mode:
    provenance = sidecar.get("publication_provenance") or {}
    model_report = json.loads(model_fingerprint_json.read_text())
    runtime_payload = json.loads(runtime_provenance_json.read_text())
    runtime = runtime_payload.get("runtime", runtime_payload)
    if (provenance.get("model") or {}).get("sha256") != model_report.get("sha256"):
        raise SystemExit("base sidecar model fingerprint mismatch")
    if (provenance.get("runtime") or {}).get("sha256") != runtime.get("sha256"):
        raise SystemExit("base sidecar runtime fingerprint mismatch")
    execution = provenance.get("execution") or {}
    if execution.get("deterministic_algorithms") is not True:
        raise SystemExit("base sidecar is missing deterministic algorithms")
    if execution.get("allow_tf32") is not False:
        raise SystemExit("base sidecar permits TF32")
    if execution.get("cudnn_benchmark") is not False:
        raise SystemExit("base sidecar permits cuDNN benchmarking")
    expected_execution = {
        "deterministic_algorithms": expected_deterministic,
        "allow_tf32": expected_allow_tf32,
        "cudnn_benchmark": expected_cudnn_benchmark,
        "float32_matmul_precision": expected_matmul_precision,
        "cublas_workspace_config": expected_cublas_workspace,
        "seed": expected_seed,
    }
    if execution != expected_execution:
        raise SystemExit(
            f"base sidecar execution profile mismatch: {execution} != "
            f"{expected_execution}"
        )
print({"base": str(base), "rows": expected_rows, "d_model": expected_d_model, "empty_api_explanation": int(empty)})
PY

if [ "$BUILD_AR" = "1" ]; then
  if [ ! -f "$AR_SFT" ]; then
    "$PY" scripts/nano_realdata_ar_build.py \
      --local-files-only \
      --model-id "$MODEL" \
      --input "$BASE" \
      --output "$AR_SFT"
  fi

  "$PY" scripts/nano_prefix_dataset_sidecar.py ar \
    --base "$BASE" \
    --ar "$AR_SFT" \
    --contract "$CONTRACT" \
    --layer "$LAYER" \
    --slug "$SLUG"
fi

if [ "$BUILD_AV" = "1" ]; then
  if [ ! -f "$AV_SFT" ]; then
    "$PY" scripts/nano_av_from_layer_probe.py \
      --input "$BASE" \
      --source-sidecar "$BASE.nla_meta.yaml" \
      --output "$AV_SFT" \
      --layer "$LAYER"
  fi

  "$PY" scripts/nano_prefix_dataset_sidecar.py av \
    --base "$BASE" \
    --av "$AV_SFT" \
    --contract "$CONTRACT" \
    --layer "$LAYER" \
    --slug "$SLUG"
fi

if [ "$PREP_CRITIC" = "1" ] && [ "$BUILD_AR" = "1" ] && [ ! -f "$CRITIC/nla_meta.yaml" ]; then
  "$PY" -m nla.scripts.prepare_critic_checkpoint \
    --base-model "$MODEL" \
    --num-layers "$LAYER" \
    --dataset-sidecar "$AR_SFT" \
    --output "$CRITIC" \
    --torch-dtype bfloat16 \
    --value-head-init "$CRITIC_VALUE_HEAD_INIT" \
    --initialization-seed "$CRITIC_INITIALIZATION_SEED" \
    --value-head-rotation-radians "$CRITIC_VALUE_HEAD_ROTATION_RADIANS" \
    --router-init "$CRITIC_ROUTER_INIT" \
    --router-relative-std "$CRITIC_ROUTER_RELATIVE_STD"
fi

if [ "$RUN_VERIFY" = "1" ] && [ "$BUILD_AR" = "1" ]; then
  "$PY" scripts/verify_nano_miles_ar_dataset.py "$AR_SFT" \
    --expected-d-model "$EXPECTED_D_MODEL" \
    --expected-rows "$EXPECTED_ROWS" \
    --tokenizer-model "$CRITIC" \
    --prompt-check-limit 1024 \
    --report-json "$AR_VERIFY" \
    "${VERIFY_SPLIT_ARGS[@]}"
fi

if [ "$RUN_VERIFY" = "1" ] && [ "$BUILD_AV" = "1" ]; then
  "$PY" scripts/verify_nano_miles_av_dataset.py "$AV_SFT" \
    --expected-d-model "$EXPECTED_D_MODEL" \
    --expected-rows "$EXPECTED_ROWS" \
    --tokenizer-model "$MODEL" \
    --report-json "$AV_VERIFY" \
    "${VERIFY_SPLIT_ARGS[@]}"
fi

echo "prefix dataset pipeline complete"
echo "base=$BASE"
echo "ar_sft=$AR_SFT"
echo "av_sft=$AV_SFT"
