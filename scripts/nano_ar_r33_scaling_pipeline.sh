#!/usr/bin/env bash
set -euo pipefail

# RunAI-side preparation for the R33 100k scaling slice.
# It builds the larger R33 activation/explanation tables, AR-SFT parquet,
# AV-SFT parquet, critic init, and verifier reports. It does not launch training.

CODE_ROOT="${CODE_ROOT:-/workspace/interp/code/nano30b-nla-pilot-current}"
cd "$CODE_ROOT"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="$CODE_ROOT/external/natural_language_autoencoders:$CODE_ROOT/external/natural_language_autoencoders/Miles:$CODE_ROOT:${PYTHONPATH:-}"

PY="${PY:-/workspace/interp/.venv/bin/python}"
MODEL="${MODEL:-/workspace/interp/models/nano-30b-a3b-bf16-hf}"
OUT="${OUT:-/workspace/interp/outputs/nano30b-nla-pilot/r33_scaling_100k}"
TEACHER="${TEACHER:-/workspace/interp/artifacts/nano30b-nla-pilot/super-teacher-r27-100k-thinking-merged-20260525T2150Z/base_R27_super_thinking_99570_explained.parquet}"
TEACHER_FALLBACK="${TEACHER_FALLBACK:-/workspace/interp/code/nano30b-nla-pilot-current/runs/introspection/ar-r27-r30-fullscan-20260528T234403Z/handoff/R_27/teacher_keys_api_explanation.parquet}"
CONTRACT="${CONTRACT:-/workspace/interp/outputs/nano30b-nla-pilot/ar-r27-r30-fullscan-20260528T234403Z/R_27/ar_sft.parquet.nla_meta.yaml}"
CONTRACT_FALLBACK="${CONTRACT_FALLBACK:-/workspace/interp/code/nano30b-nla-pilot-current/runs/introspection/ar-r27-r30-fullscan-20260528T234403Z/handoff/R_27/ar_sft.parquet.nla_meta.yaml}"
EXPECTED_ROWS="${EXPECTED_ROWS:-99570}"
CORPUS_START="${CORPUS_START:-500}"
CORPUS_LENGTH="${CORPUS_LENGTH:-10000}"
POSITIONS_PER_DOC="${POSITIONS_PER_DOC:-10}"
EXTRACT_ROOT="${EXTRACT_ROOT:-$OUT/extract_start${CORPUS_START}_len${CORPUS_LENGTH}}"
EXTRACT_SHARD_DOCS="${EXTRACT_SHARD_DOCS:-2048}"

BASE="$OUT/base_R33_start${CORPUS_START}_len${CORPUS_LENGTH}.parquet"
MERGED="$OUT/base_R33_start${CORPUS_START}_len${CORPUS_LENGTH}_explained.parquet"
AR_SFT="$OUT/ar_sft_r33_start${CORPUS_START}_len${CORPUS_LENGTH}.parquet"
AV_SFT="$OUT/av_sft_r33_start${CORPUS_START}_len${CORPUS_LENGTH}.parquet"
CRITIC="${CRITIC:-/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-critic-init/nano-ar-r33-critic-init}"

mkdir -p "$OUT"

if [ ! -f "$TEACHER" ] && [ -f "$TEACHER_FALLBACK" ]; then
  TEACHER="$TEACHER_FALLBACK"
fi
if [ ! -f "$CONTRACT" ] && [ -f "$CONTRACT_FALLBACK" ]; then
  CONTRACT="$CONTRACT_FALLBACK"
fi

if [ ! -f "$TEACHER" ]; then
  echo "teacher table not found: $TEACHER" >&2
  echo "set TEACHER=/path/to R27 100k or fullscan teacher_keys parquet" >&2
  exit 1
fi
if [ ! -f "$CONTRACT" ]; then
  echo "AR sidecar contract not found: $CONTRACT" >&2
  echo "set CONTRACT=/path/to known-good R27 AR sidecar" >&2
  exit 1
fi

"$PY" - "$TEACHER" "$CORPUS_START" "$CORPUS_LENGTH" "$EXPECTED_ROWS" "$OUT/preflight_teacher_overlap_r33_start${CORPUS_START}_len${CORPUS_LENGTH}.json" <<'PY'
from pathlib import Path
import json
import sys
import time

import pyarrow.parquet as pq

teacher_path = Path(sys.argv[1])
corpus_start = int(sys.argv[2])
corpus_length = int(sys.argv[3])
expected_rows = int(sys.argv[4])
report_path = Path(sys.argv[5])


def doc_suffix(value):
    try:
        return int(str(value).rsplit(":", 1)[-1])
    except ValueError as exc:
        raise SystemExit(f"could not parse numeric doc suffix from {value!r}") from exc


teacher = pq.read_table(teacher_path, columns=["doc_id"])
expected_doc_ids = set(range(corpus_start, corpus_start + corpus_length))
teacher_doc_suffixes = [doc_suffix(value) for value in teacher["doc_id"].to_pylist()]
teacher_overlap_rows = sum(1 for value in teacher_doc_suffixes if value in expected_doc_ids)
teacher_overlap_docs = len(set(teacher_doc_suffixes) & expected_doc_ids)
report = {
    "schema_version": "preflight_teacher_overlap.v1",
    "teacher": str(teacher_path),
    "corpus_start": corpus_start,
    "corpus_length": corpus_length,
    "expected_rows": expected_rows,
    "teacher_rows": teacher.num_rows,
    "teacher_min_doc_suffix": min(teacher_doc_suffixes) if teacher_doc_suffixes else None,
    "teacher_max_doc_suffix": max(teacher_doc_suffixes) if teacher_doc_suffixes else None,
    "teacher_overlap_docs": teacher_overlap_docs,
    "teacher_overlap_rows": teacher_overlap_rows,
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
if teacher_overlap_rows < expected_rows:
    raise SystemExit(
        "teacher overlap too small for requested corpus slice: "
        f"{teacher_overlap_rows} < {expected_rows}"
    )
PY

if [ ! -f "$BASE" ]; then
  shard_paths=()
  shard_start="$CORPUS_START"
  shard_end=$((CORPUS_START + CORPUS_LENGTH))
  while [ "$shard_start" -lt "$shard_end" ]; do
    shard_len="$EXTRACT_SHARD_DOCS"
    remaining=$((shard_end - shard_start))
    if [ "$remaining" -lt "$shard_len" ]; then
      shard_len="$remaining"
    fi
    shard_root="$EXTRACT_ROOT/shards/start${shard_start}_len${shard_len}"
    shard_base="$shard_root/R_33/base.parquet"
    if ! "$PY" - "$shard_base" <<'PY'
from pathlib import Path
import sys
import pyarrow.parquet as pq

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
try:
    metadata = pq.ParquetFile(path).metadata
except Exception:
    raise SystemExit(1)
if metadata.num_rows <= 0:
    raise SystemExit(1)
print(f"valid_shard={path} rows={metadata.num_rows}")
PY
    then
      rm -rf "$shard_root"
      "$PY" scripts/nano_ar_layer_sweep.py extract \
        --layers R33 \
        --output-root "$shard_root" \
        --model-id "$MODEL" \
        --corpus-start "$shard_start" \
        --corpus-length "$shard_len" \
        --positions-per-doc "$POSITIONS_PER_DOC" \
        --chunk-size 8 \
        --batch-size 2 \
        --max-length 1024 \
        --seed 42 \
        --local-files-only \
        --overwrite
    fi
    shard_paths+=("$shard_base")
    shard_start=$((shard_start + shard_len))
  done

  "$PY" - "$BASE" "$BASE.metadata.json" "$BASE.nla_meta.yaml" "$CORPUS_START" "$CORPUS_LENGTH" "$EXPECTED_ROWS" "${shard_paths[@]}" <<'PY'
from pathlib import Path
import json
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

base_path = Path(sys.argv[1])
metadata_path = Path(sys.argv[2])
sidecar_path = Path(sys.argv[3])
corpus_start = int(sys.argv[4])
corpus_length = int(sys.argv[5])
expected_rows = int(sys.argv[6])
shard_paths = [Path(value) for value in sys.argv[7:]]
if not shard_paths:
    raise SystemExit("no shard paths passed")

tables = []
row_counts = []
for path in shard_paths:
    table = pq.read_table(path)
    tables.append(table)
    row_counts.append(table.num_rows)

combined = pa.concat_tables(tables, promote_options="default")
base_path.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(combined, base_path)
rows = combined.num_rows
metadata = {
    "schema_version": "nano_ar_layer_sweep_extract.v1",
    "layer": 33,
    "row_count": rows,
    "output": str(base_path),
    "shards": [
        {"path": str(path), "rows": count}
        for path, count in zip(shard_paths, row_counts, strict=True)
    ],
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

first_sidecar = Path(str(shard_paths[0]) + ".nla_meta.yaml")
meta = yaml.safe_load(first_sidecar.read_text())
meta["row_count"] = rows
meta["dataset_id"] = f"base_R33_start{corpus_start}_len{corpus_length}_sharded"
meta["created_by"] = "scripts.nano_ar_r33_scaling_pipeline"
meta.setdefault("extraction", {})["corpus_slice"] = {"start": corpus_start, "length": corpus_length}
meta.setdefault("extraction", {})["layer_index"] = 33
meta["parent_datasets"] = [str(path) for path in shard_paths]
sidecar_path.write_text(yaml.safe_dump(meta, sort_keys=False))
print(json.dumps(metadata, indent=2, sort_keys=True))
if rows < int(expected_rows * 0.95):
    raise SystemExit(f"combined too few rows: {rows} < 95% of {expected_rows}")
PY
fi

if [ ! -f "$MERGED" ]; then
  "$PY" - "$BASE" "$TEACHER" "$MERGED" "$OUT/r33_100k_merge_report.json" "$EXPECTED_ROWS" <<'PY'
from pathlib import Path
import json
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

base_path = Path(sys.argv[1])
teacher_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
report_path = Path(sys.argv[4])
expected_rows = int(sys.argv[5])

key_candidates = ["doc_id", "token_position", "token_id", "n_raw_tokens"]
text_candidates = ["api_explanation", "explanation", "teacher_explanation"]

base = pq.read_table(base_path)
teacher = pq.read_table(teacher_path)
base_names = set(base.column_names)
teacher_names = set(teacher.column_names)
keys = [name for name in key_candidates if name in base_names and name in teacher_names]
if "doc_id" not in keys or not ({"n_raw_tokens", "token_position"} & set(keys)):
    raise SystemExit(
        "unsafe teacher merge keys; need doc_id plus n_raw_tokens or token_position, "
        f"got {keys}"
    )
text_column = next((name for name in text_candidates if name in teacher_names), None)
if text_column is None:
    raise SystemExit(f"teacher table has none of {text_candidates}")

teacher_cols = {name: teacher[name].to_pylist() for name in keys + [text_column]}
teacher_by_key = {}
duplicates = 0
for row_idx, explanation in enumerate(teacher_cols[text_column]):
    key = tuple(teacher_cols[name][row_idx] for name in keys)
    if key in teacher_by_key:
        duplicates += 1
    teacher_by_key[key] = explanation

base_cols = {name: base[name].to_pylist() for name in keys}
keep = []
explanations = []
missing = 0
for row_idx in range(base.num_rows):
    key = tuple(base_cols[name][row_idx] for name in keys)
    explanation = teacher_by_key.get(key)
    if explanation is None or not str(explanation).strip():
        keep.append(False)
        missing += 1
        continue
    keep.append(True)
    explanations.append(str(explanation))

joined = base.filter(pa.array(keep, type=pa.bool_())).append_column(
    "api_explanation",
    pa.array(explanations, type=pa.string()),
)
out_path.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(joined, out_path)
report = {
    "schema_version": "r33_100k_teacher_merge.v1",
    "base": str(base_path),
    "teacher": str(teacher_path),
    "output": str(out_path),
    "join_keys": keys,
    "text_column": text_column,
    "base_rows": base.num_rows,
    "teacher_rows": teacher.num_rows,
    "merged_rows": joined.num_rows,
    "activation_rows_without_teacher": missing,
    "teacher_duplicate_keys": duplicates,
    "expected_rows": expected_rows,
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
if duplicates:
    raise SystemExit("teacher merge produced duplicate keys")
if joined.num_rows < int(expected_rows * 0.95):
    raise SystemExit(f"merged too few rows: {joined.num_rows} < 95% of {expected_rows}")
PY
fi

if [ ! -f "$AR_SFT" ]; then
  "$PY" scripts/nano_realdata_ar_build.py \
    --local-files-only \
    --model-id "$MODEL" \
    --input "$MERGED" \
    --output "$AR_SFT"
fi

"$PY" - "$BASE" "$AR_SFT" "$CONTRACT" "$CORPUS_START" "$CORPUS_LENGTH" <<'PY'
from pathlib import Path
import sys

import pyarrow.parquet as pq
import yaml

base_path = Path(sys.argv[1])
ar_path = Path(sys.argv[2])
contract_path = Path(sys.argv[3])
corpus_start = int(sys.argv[4])
corpus_length = int(sys.argv[5])

meta = yaml.safe_load(Path(str(base_path) + ".nla_meta.yaml").read_text())
contract = yaml.safe_load(contract_path.read_text())
rows = pq.read_table(ar_path, columns=["doc_id"]).num_rows
old_dataset = meta.get("dataset_id", f"base_R33_start{corpus_start}_len{corpus_length}")
meta["stage"] = "ar_sft"
meta["row_count"] = rows
meta["dataset_id"] = f"nano30b_r33_ar_sft_start{corpus_start}_len{corpus_length}_teacher_reuse"
meta["created_by"] = "scripts.nano_ar_r33_scaling_pipeline"
meta["parent_datasets"] = [old_dataset, "r27_100k_or_fullscan_teacher_keys"]
meta.setdefault("critic", {})["extraction_layer_index"] = 33
for key in ("tokens", "prompt_templates"):
    meta[key] = contract[key]
for key in ("sequence", "normalization"):
    if key in contract:
        meta[key] = contract[key]
for key in ("mse_scale", "injection_scale"):
    value = contract.get("extraction", {}).get(key)
    if value is not None:
        meta.setdefault("extraction", {})[key] = value
Path(str(ar_path) + ".nla_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
print(ar_path, rows)
PY

if [ ! -f "$AV_SFT" ]; then
  "$PY" scripts/nano_av_from_layer_probe.py \
    --input "$MERGED" \
    --source-sidecar "$BASE.nla_meta.yaml" \
    --output "$AV_SFT" \
    --layer 33
fi

if [ ! -f "$CRITIC/config.json" ]; then
  "$PY" -m nla.scripts.prepare_critic_checkpoint \
    --base-model "$MODEL" \
    --num-layers 33 \
    --dataset-sidecar "$AR_SFT" \
    --output "$CRITIC" \
    --torch-dtype bfloat16
fi

"$PY" scripts/verify_nano_miles_ar_dataset.py "$AR_SFT" \
  --expected-d-model 2688 \
  --expected-rows "$EXPECTED_ROWS" \
  --tokenizer-model "$CRITIC" \
  --report-json "$OUT/ar_dataset_verify_r33_start${CORPUS_START}_len${CORPUS_LENGTH}.json"

"$PY" scripts/verify_nano_miles_av_dataset.py "$AV_SFT" \
  --expected-d-model 2688 \
  --expected-rows "$EXPECTED_ROWS" \
  --tokenizer-model "$MODEL" \
  --report-json "$OUT/av_dataset_verify_r33_start${CORPUS_START}_len${CORPUS_LENGTH}.json"

echo "r33_scaling_100k_ready=$OUT"
