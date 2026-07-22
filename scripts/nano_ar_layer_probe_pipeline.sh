#!/usr/bin/env bash
set -euo pipefail

cd /workspace/interp/code/nano30b-nla-pilot-current

export WANDB_MODE=offline
export PYTHONPATH="/workspace/interp/code/nano30b-nla-pilot-current/external/natural_language_autoencoders:/workspace/interp/code/nano30b-nla-pilot-current/external/natural_language_autoencoders/Miles:/workspace/interp/code/nano30b-nla-pilot-current:${PYTHONPATH:-}"

PY=/workspace/interp/.venv/bin/python
MODEL=/workspace/interp/models/nano-30b-a3b-bf16-hf
SWEEP=/workspace/interp/outputs/nano30b-nla-pilot/layer_sweeps/r25_r51_20k_start10500_len2048
TEACHER=/workspace/interp/outputs/nano30b-nla-pilot/r34_probe/teacher_keys_api_explanation_start10500_len2048.parquet
CONTRACT=/workspace/interp/outputs/nano30b-nla-pilot/ar-r27-r30-fullscan-20260528T234403Z/R_27/ar_sft.parquet.nla_meta.yaml
JOB_ROOT=/workspace/interp/outputs/nano30b-nla-pilot/layer_probe_jobs/r33_r51_20260605
QUEUE="$JOB_ROOT/r33_r51_probe_queue.yaml"

mkdir -p "$JOB_ROOT"

for L in 33 51; do
  OUT=/workspace/interp/outputs/nano30b-nla-pilot/layer_probe_r${L}
  mkdir -p "$OUT"
  BASE_SRC=$SWEEP/R_${L}/base.parquet
  BASE=$OUT/base_R${L}_start10500_len2048.parquet
  MERGED=$OUT/base_R${L}_start10500_len2048_explained.parquet
  AR_SFT=$OUT/ar_sft_r${L}_start10500_len2048.parquet
  CRITIC=/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-critic-init/nano-ar-r${L}-critic-init

  if [ ! -f "$BASE" ]; then
    cp -f "$BASE_SRC" "$BASE"
    cp -f "$BASE_SRC.metadata.json" "$BASE.metadata.json"
    cp -f "$BASE_SRC.nla_meta.yaml" "$BASE.nla_meta.yaml"
  fi

  if [ ! -f "$MERGED" ]; then
    echo "merging teacher explanations for R${L}"
    "$PY" - "$L" "$BASE" "$TEACHER" "$MERGED" "$OUT/r${L}_merge_report.json" <<'PY'
from pathlib import Path
import json
import sys
import time
import pyarrow as pa
import pyarrow.parquet as pq

layer = int(sys.argv[1])
base_path = Path(sys.argv[2])
teacher_path = Path(sys.argv[3])
out_path = Path(sys.argv[4])
report_path = Path(sys.argv[5])
key_cols = ["doc_id", "token_position", "token_id", "n_raw_tokens"]
base = pq.read_table(base_path)
teacher = pq.read_table(teacher_path)
teacher_cols = {name: teacher[name].to_pylist() for name in key_cols + ["api_explanation"]}
teacher_by_key = {}
duplicates = 0
for row_idx, explanation in enumerate(teacher_cols["api_explanation"]):
    key = tuple(teacher_cols[name][row_idx] for name in key_cols)
    if key in teacher_by_key:
        duplicates += 1
    teacher_by_key[key] = explanation
base_cols = {name: base[name].to_pylist() for name in key_cols}
keep = []
explanations = []
missing = 0
for row_idx in range(base.num_rows):
    key = tuple(base_cols[name][row_idx] for name in key_cols)
    explanation = teacher_by_key.get(key)
    if explanation is None:
        keep.append(False)
        missing += 1
    else:
        keep.append(True)
        explanations.append(explanation)
joined = base.filter(pa.array(keep, type=pa.bool_())).append_column(
    "api_explanation",
    pa.array(explanations, type=pa.string()),
)
out_path.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(joined, out_path)
report = {
    "schema_version": "layer_probe_teacher_merge.v1",
    "layer": layer,
    "base": str(base_path),
    "teacher": str(teacher_path),
    "output": str(out_path),
    "base_rows": base.num_rows,
    "teacher_rows": teacher.num_rows,
    "merged_rows": joined.num_rows,
    "activation_rows_without_teacher": missing,
    "teacher_duplicate_keys": duplicates,
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
if joined.num_rows < 20000 or duplicates:
    raise SystemExit(1)
PY
  fi

  if [ ! -f "$AR_SFT" ]; then
    echo "building R${L} AR-SFT parquet"
    "$PY" scripts/nano_realdata_ar_build.py \
      --local-files-only \
      --model-id "$MODEL" \
      --input "$MERGED" \
      --output "$AR_SFT"
  fi

  "$PY" - "$L" "$BASE" "$AR_SFT" "$CONTRACT" <<'PY'
from pathlib import Path
import sys
import pyarrow.parquet as pq
import yaml

layer = int(sys.argv[1])
base_path = Path(sys.argv[2])
ar_path = Path(sys.argv[3])
contract_path = Path(sys.argv[4])
meta = yaml.safe_load(Path(str(base_path) + ".nla_meta.yaml").read_text())
contract = yaml.safe_load(contract_path.read_text())
rows = pq.read_table(ar_path, columns=["doc_id"]).num_rows
old_dataset = meta.get("dataset_id", f"base_R{layer}")
meta["stage"] = "ar_sft"
meta["row_count"] = rows
meta["dataset_id"] = f"nano30b_r{layer}_ar_sft_start10500_len2048_teacher_reuse"
meta["created_by"] = "nano_ar_layer_probe_pipeline"
meta["parent_datasets"] = [old_dataset, "r27_teacher_keys_start10500_len2048"]
meta.setdefault("critic", {})["extraction_layer_index"] = layer
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

  if [ ! -f "$CRITIC/config.json" ]; then
    echo "preparing R${L} critic init"
    "$PY" -m nla.scripts.prepare_critic_checkpoint \
      --base-model "$MODEL" \
      --num-layers "$L" \
      --dataset-sidecar "$AR_SFT" \
      --output "$CRITIC" \
      --torch-dtype bfloat16
  fi

  "$PY" scripts/verify_nano_miles_ar_dataset.py "$AR_SFT" \
    --expected-d-model 2688 \
    --tokenizer-model "$CRITIC" \
    --report-json "$OUT/ar_dataset_verify_r${L}_start10500_len2048.json"
done

cat > "$QUEUE" <<'YAML'
schema_version: nano_ar_hpo_queue.v1
defaults:
  code_root: /workspace/interp/code/nano30b-nla-pilot-current
  python: /workspace/interp/.venv/bin/python
  validation_limit: 512
  test_limit: 512
  batch_size: 4
  controls: [teacher, teacher_shuffled, blank, generic, mean, source_context, source_raw]
  study_jsonl: /workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/r33_r51_trials.jsonl
items:
  - name: r33-mini-20k-lr2e5-cos128
    config: configs/nano_ar/hpo/r33_mini_probe_20k_lr2e5_cosine_128steps.yaml
    run_id: nano-ar-r33-mini-probe-20k-lr2e5-cosine-128steps
    status: pending
    notes: Bounded R33 K+1 critic mini-probe using reused teacher text from the R27/R30 fullscan handoff.
  - name: r51-mini-20k-lr2e5-cos128
    config: configs/nano_ar/hpo/r51_mini_probe_20k_lr2e5_cosine_128steps.yaml
    run_id: nano-ar-r51-mini-probe-20k-lr2e5-cosine-128steps
    status: pending
    notes: Bounded R51 K+1 critic mini-probe using reused teacher text from the R27/R30 fullscan handoff.
YAML

"$PY" scripts/nano_ar_hpo_queue.py "$QUEUE" --run-until-empty
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
