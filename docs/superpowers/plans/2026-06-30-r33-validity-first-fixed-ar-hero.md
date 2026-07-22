# R33 Validity-First Fixed-AR Hero Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Independently validate the existing R33 RL gain, correct KL and batching mechanics, run bounded fixed-AR probes, and leave the full-data hero queue gated on trustworthy evidence.

**Architecture:** Add small reusable evaluation modules for paired statistics, semantic transformations, exact source-row resolution, and Nano boundary patching. Compose them through config-driven CLIs and the existing queue, then expose non-negative KL and strict batch semantics through the launcher. Keep target-model functional evaluation separate from the frozen reward AR so the promotion decision is not circular.

**Tech Stack:** Python 3.12, NumPy, PyTorch, PyArrow, Hugging Face Transformers remote code, Miles FSDP2 patches, SGLang, YAML, unittest/pytest, RunAI, W&B offline, S3.

---

## File Map

- `scripts/nano_eval_core.py`: dependency-light paired statistics and logit metrics.
- `scripts/nano_roundtrip_transforms.py`: deterministic explanation transforms and versioned transform records.
- `scripts/eval_nano_roundtrip_invariance.py`: transform generated records, reuse AR scoring, and report FVE retention.
- `scripts/nano_r33_source_rows.py`: stable provenance keys and streaming source-parquet lookup.
- `scripts/nano_r33_functional_core.py`: activation rescaling, final-token boundary replacement, and functional metrics.
- `scripts/eval_nano_r33_functional_recovery.py`: end-to-end R33 target-model functional evaluator.
- `scripts/eval_nano_r33_validity_gate.py`: merges round-trip, invariance, functional, and health evidence into one hard gate.
- `scripts/eval_nano_av_ar_roundtrip_gate.py`: add stable provenance to generated records.
- `scripts/nano_roundtrip_eval_config.py`: render transform and functional-eval config fields.
- `external/natural_language_autoencoders/configs/rl.sh`: render KL estimator and train guard hook.
- `external/natural_language_autoencoders/nla/train_guard.py`: stateful actor drift guard.
- `external/natural_language_autoencoders/nla/miles_patches/0017_train_guard_hook.patch`: generic Miles train-step guard hook.
- `scripts/nano_rl_queue.py`: validate exact generated/global/trained batch semantics and render new settings.
- `configs/nano_rl/r33_component_validity_eval.yaml`: checkpoint-neutral validity evaluation contract.
- `configs/nano_rl/r33_component_corrected_k3_hpo_queue_8h100.yaml`: two corrected eight-update probes and blocked confirmation.
- `configs/nano_rl/r33_component_fixed_ar_hero_queue_8h100.yaml`: blocked full-data topology canaries and 256-update run.
- `scripts/build_nano_r33_rl_dataset.py`: build a train-only RL parquet with source lineage.
- `scripts/verify_nano_r33_rl_dataset.py`: enforce dimensions, finiteness, and split/content isolation.
- `scripts/nano_checkpoint_retention.py`: manifest-driven checkpoint selection and guarded cleanup.

### Task 1: Add Paired Statistics And Functional Logit Metrics

**Files:**
- Create: `scripts/nano_eval_core.py`
- Create: `tests/test_nano_eval_core.py`

- [ ] **Step 1: Write failing tests for paired bootstrap direction and logit metrics**

```python
import numpy as np

from nano_eval_core import functional_logit_metrics, paired_bootstrap_improvement


def test_paired_bootstrap_positive_when_candidate_loss_is_lower():
    baseline = np.array([4.0, 3.0, 2.0, 1.0])
    candidate = np.array([3.0, 2.0, 1.0, 0.5])
    result = paired_bootstrap_improvement(baseline, candidate, seed=7, resamples=2000)
    assert result["mean_improvement"] > 0
    assert result["ci95_low"] > 0


def test_functional_metrics_identical_logits_are_perfect():
    logits = np.array([3.0, 2.0, 1.0, -1.0])
    result = functional_logit_metrics(logits, logits, top_ks=(2, 3))
    assert result["kl_original_to_patched"] == 0.0
    assert result["js_divergence"] == 0.0
    assert result["top_2_overlap"] == 1.0
    assert result["original_top1_rank"] == 1
```

- [ ] **Step 2: Run the focused tests and confirm the module is missing**

Run: `python3 -m pytest tests/test_nano_eval_core.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'nano_eval_core'`.

- [ ] **Step 3: Implement strict paired statistics and numerically stable logit metrics**

```python
def paired_bootstrap_improvement(baseline, candidate, *, seed=0, resamples=10_000):
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if baseline.shape != candidate.shape or baseline.ndim != 1 or baseline.size == 0:
        raise ValueError("baseline and candidate must be non-empty paired 1D arrays")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("paired arrays must be finite")
    delta = baseline - candidate
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(resamples, delta.size))
    draws = delta[indices].mean(axis=1)
    return {
        "count": int(delta.size),
        "mean_improvement": float(delta.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "candidate_better_fraction": float(np.mean(delta > 0)),
    }
```

Implement `functional_logit_metrics` with log-sum-exp softmax, finite checks,
`KL(original || patched)`, Jensen-Shannon divergence, Pearson correlation,
top-k set overlap, and one-indexed rank of the original top-1 token.

- [ ] **Step 4: Run the new unit tests**

Run: `python3 -m pytest tests/test_nano_eval_core.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the isolated new files**

```bash
git add scripts/nano_eval_core.py tests/test_nano_eval_core.py
git commit -m "feat: add paired NLA evaluation metrics"
```

### Task 2: Add Versioned Semantic Transformations

**Files:**
- Create: `scripts/nano_roundtrip_transforms.py`
- Create: `scripts/eval_nano_roundtrip_invariance.py`
- Create: `tests/test_nano_roundtrip_transforms.py`
- Create: `tests/test_eval_nano_roundtrip_invariance.py`

- [ ] **Step 1: Write failing tests for deterministic transforms and provenance**

```python
from nano_roundtrip_transforms import build_transform_record, normalize_formatting, reorder_units


def test_reorder_units_is_seeded_and_preserves_units():
    text = "- first point\n- second point\n- third point"
    one = reorder_units(text, seed=19)
    two = reorder_units(text, seed=19)
    assert one == two
    assert sorted(one.splitlines()) == sorted(text.splitlines())


def test_transform_record_hashes_source_and_never_falls_back():
    record = build_transform_record(
        row_key="doc-1:20",
        source="A sentence. Another sentence.",
        transform="format_normalized",
        transformed=normalize_formatting("A sentence.  Another sentence."),
        seed=0,
    )
    assert record["schema_version"] == "nano_roundtrip_transform.v1"
    assert len(record["source_sha256"]) == 64
    assert record["transformed_text"] != ""
```

- [ ] **Step 2: Run the tests and confirm the module is missing**

Run: `python3 -m pytest tests/test_nano_roundtrip_transforms.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement focused transformation functions**

Implement:

```python
def normalize_formatting(text: str) -> str:
    lines = [" ".join(line.split()) for line in str(text).strip().splitlines()]
    return "\n".join(line for line in lines if line)


def split_semantic_units(text: str) -> list[str]:
    normalized = normalize_formatting(text)
    lines = [line for line in normalized.splitlines() if line]
    if len(lines) > 1:
        return lines
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def reorder_units(text: str, *, seed: int) -> str:
    units = split_semantic_units(text)
    shuffled = list(units)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) > 1 and shuffled == units:
        shuffled = shuffled[1:] + shuffled[:1]
    separator = "\n" if "\n" in str(text) else " "
    return separator.join(shuffled)


def build_transform_record(*, row_key: str, source: str, transform: str,
                           transformed: str, seed: int,
                           model: str | None = None,
                           prompt_sha256: str | None = None) -> dict:
    if not str(transformed).strip():
        raise TransformError("transformed text is empty")
    return {
        "schema_version": "nano_roundtrip_transform.v1",
        "row_key": row_key,
        "transform": transform,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "transformed_text": transformed,
        "seed": seed,
        "model": model,
        "prompt_sha256": prompt_sha256,
    }


def apply_transform_records(generated_records: list[dict],
                            transforms_by_key: dict[tuple[str, str], dict],
                            *, transform: str) -> list[dict]:
    output = copy.deepcopy(generated_records)
    for record in output:
        row_key = stable_row_key(record)
        item = transforms_by_key.get((row_key, transform))
        if item is None:
            raise TransformError(f"missing {transform} transform for {row_key}")
        source = str(record["controls"]["real"]["generated"])
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        if item["source_sha256"] != source_hash or not str(item["transformed_text"]).strip():
            raise TransformError(f"invalid {transform} transform for {row_key}")
        record["controls"]["real"]["generated"] = item["transformed_text"]
    return output
```

`apply_transform_records` must raise on missing or empty requested transforms.
It must update only `controls.real.generated` while retaining row identity and
the unmodified record metadata.

- [ ] **Step 4: Implement invariance scoring by reusing the existing AR scorer**

The CLI loads the original generated JSONL, creates `format_normalized` and
`unit_reordered` record sets, optionally loads versioned light-paraphrase
records, and calls `score_generated_records` from
`eval_nano_av_ar_roundtrip_gate.py` for each variant. Its report contains raw
and transformed FVE plus `transformed_fve / raw_fve` for validation and test.
It rejects row-identity mismatch and any transform fallback.

- [ ] **Step 5: Verify transforms, invariance scoring, and existing round-trip tests**

Run: `python3 -m pytest tests/test_nano_roundtrip_transforms.py tests/test_eval_nano_roundtrip_invariance.py tests/test_nano_av_ar_roundtrip_gate.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the transformation and invariance modules**

```bash
git add scripts/nano_roundtrip_transforms.py scripts/eval_nano_roundtrip_invariance.py tests/test_nano_roundtrip_transforms.py tests/test_eval_nano_roundtrip_invariance.py
git commit -m "feat: add semantic invariance transforms"
```

### Task 3: Preserve Stable Provenance And Resolve Exact Source Rows

**Files:**
- Modify: `scripts/eval_nano_av_ar_roundtrip_gate.py`
- Create: `scripts/nano_r33_source_rows.py`
- Modify: `tests/test_nano_av_ar_roundtrip_gate.py`
- Create: `tests/test_nano_r33_source_rows.py`

- [ ] **Step 1: Add failing tests for generated-record provenance**

Extend the generation-record test fixture with `doc_id`, `n_raw_tokens`,
`token_position`, `token_id`, and `sample_uuid`, then assert all present fields
are copied to the generated JSONL record.

```python
for key in ("doc_id", "n_raw_tokens", "token_position", "token_id", "sample_uuid"):
    assert generated_record[key] == source_row[key]
```

- [ ] **Step 2: Add failing tests for key priority and streaming lookup**

```python
from nano_r33_source_rows import provenance_key, resolve_source_rows


def test_provenance_key_prefers_uuid_then_position_then_raw_tokens():
    assert provenance_key({"sample_uuid": "u", "doc_id": "d"}) == ("uuid", "u")
    assert provenance_key({"doc_id": "d", "token_position": 8}) == ("position", "d", 8)
    assert provenance_key({"doc_id": "d", "n_raw_tokens": 9}) == ("raw_tokens", "d", 9)


def test_duplicate_source_key_fails(tmp_path):
    path = tmp_path / "base.parquet"
    table = pa.table({
        "doc_id": ["d", "d"],
        "token_position": [8, 8],
        "n_raw_tokens": [9, 9],
        "token_id": [11, 11],
        "token_ids_prefix": [[1, 11], [1, 11]],
        "activation_vector": [[0.0, 1.0], [0.0, 1.0]],
    })
    pq.write_table(table, path)
    with pytest.raises(SourceRowError, match="duplicate source provenance key"):
        resolve_source_rows(path, [{"doc_id": "d", "token_position": 8}])
```

- [ ] **Step 3: Run focused tests and confirm missing behavior**

Run: `python3 -m pytest tests/test_nano_av_ar_roundtrip_gate.py tests/test_nano_r33_source_rows.py -q`

Expected: provenance assertion failure and missing module failure.

- [ ] **Step 4: Implement provenance copying and streaming PyArrow lookup**

In `eval_nano_av_ar_roundtrip_gate.py`, copy only fields that exist:

```python
for key in ("n_raw_tokens", "token_position", "token_id", "sample_uuid"):
    if row.get(key) is not None:
        item[key] = row[key]
```

In `nano_r33_source_rows.py`, implement `provenance_key(record)` and:

```python
def resolve_source_rows(parquet_path: Path, requested: list[dict]) -> dict[tuple, dict]:
    wanted = {provenance_key(record) for record in requested}
    found = {}
    parquet = pq.ParquetFile(parquet_path)
    columns = [name for name in (
        "sample_uuid", "doc_id", "token_position", "n_raw_tokens",
        "token_id", "token_ids_prefix", "activation_vector",
    ) if name in parquet.schema_arrow.names]
    for batch in parquet.iter_batches(batch_size=4096, columns=columns):
        for row in batch.to_pylist():
            key = provenance_key(row)
            if key not in wanted:
                continue
            if key in found:
                raise SourceRowError(f"duplicate source provenance key: {key}")
            found[key] = row
        if len(found) == len(wanted):
            break
    missing = sorted(wanted - found.keys(), key=repr)
    if missing:
        raise SourceRowError(f"missing source rows: {missing[:10]}")
    return found
```

- [ ] **Step 5: Run provenance tests**

Run: `python3 -m pytest tests/test_nano_av_ar_roundtrip_gate.py tests/test_nano_r33_source_rows.py -q`

Expected: PASS.

- [ ] **Step 6: Commit only the reviewed provenance changes**

```bash
git add scripts/eval_nano_av_ar_roundtrip_gate.py scripts/nano_r33_source_rows.py tests/test_nano_av_ar_roundtrip_gate.py tests/test_nano_r33_source_rows.py
git commit -m "feat: preserve R33 roundtrip provenance"
```

### Task 4: Add Nano Boundary Replacement Primitives

**Files:**
- Create: `scripts/nano_r33_functional_core.py`
- Create: `tests/test_nano_r33_functional_core.py`

- [ ] **Step 1: Write failing toy-model tests for direction scaling and hook replacement**

```python
import pytest
import torch

from nano_r33_functional_core import make_boundary_replacement_hook, rescale_direction


def test_rescale_direction_matches_gold_norm():
    prediction = torch.tensor([[3.0, 4.0]])
    gold = torch.tensor([[0.0, 10.0]])
    scaled = rescale_direction(prediction, gold)
    assert torch.allclose(scaled.norm(dim=-1), gold.norm(dim=-1))


def test_hook_replaces_only_requested_final_positions():
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    replacement = torch.tensor([[100.0] * 4, [200.0] * 4)
    hook = make_boundary_replacement_hook(replacement, positions=torch.tensor([2, 1]))
    output = hook(None, None, (hidden, "cache"))
    assert torch.equal(output[0][0, :2], hidden[0, :2])
    assert torch.equal(output[0][0, 2], replacement[0])
    assert torch.equal(output[0][1, 1], replacement[1])
    assert output[1] == "cache"
```

- [ ] **Step 2: Run the test and confirm the module is missing**

Run: `python3 -m pytest tests/test_nano_r33_functional_core.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement safe tensor and tuple output replacement**

```python
def rescale_direction(prediction, gold, eps=1e-12):
    pred_norm = prediction.float().norm(dim=-1, keepdim=True).clamp_min(eps)
    gold_norm = gold.float().norm(dim=-1, keepdim=True)
    return prediction.float() / pred_norm * gold_norm


def make_boundary_replacement_hook(replacement, *, positions):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[0] != replacement.shape[0]:
            raise FunctionalRecoveryError("replacement batch does not match hidden batch")
        updated = hidden.clone()
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        updated[rows, positions.to(hidden.device)] = replacement.to(hidden.device, hidden.dtype)
        return (updated, *output[1:]) if isinstance(output, tuple) else updated
    return hook
```

Also implement `gather_last_logits(logits, positions)` and finite/shape checks.

- [ ] **Step 4: Run toy-model tests in local and RunAI Python**

Run locally: `python3 -m pytest tests/test_nano_r33_functional_core.py -q`

Run on RunAI: `/workspace/interp/.venv/bin/python -m pytest tests/test_nano_r33_functional_core.py -q`

Expected: PASS in both environments; local may skip only when PyTorch is absent.

- [ ] **Step 5: Commit the boundary primitives**

```bash
git add scripts/nano_r33_functional_core.py tests/test_nano_r33_functional_core.py
git commit -m "feat: add Nano functional patch primitives"
```

### Task 5: Build The End-To-End Functional Recovery Evaluator

**Files:**
- Create: `scripts/eval_nano_r33_functional_recovery.py`
- Create: `scripts/nano_functional_eval_config.py`
- Create: `tests/test_eval_nano_r33_functional_recovery.py`
- Create: `tests/test_nano_functional_eval_config.py`

- [ ] **Step 1: Write failing tests for config rendering and identity blocking**

```python
def test_config_renders_required_paths_and_limits(tmp_path):
    config = write_functional_config(tmp_path, validation_limit=8, test_limit=8)
    command = module.build_command(module.load_config(config), config_path=config)
    assert "scripts/eval_nano_r33_functional_recovery.py" in command
    assert command[command.index("--source-base-parquet") + 1] == "/data/base.parquet"
    assert command[command.index("--boundary") + 1] == "33"


def test_identity_canary_failure_blocks_candidate_metrics():
    report = build_report_from_fixture(identity_relative_l2=0.02)
    assert report["gate"]["identity_passed"] is False
    assert report["splits"] == {}
```

- [ ] **Step 2: Run focused tests and confirm both modules are missing**

Run: `python3 -m pytest tests/test_eval_nano_r33_functional_recovery.py tests/test_nano_functional_eval_config.py -q`

Expected: FAIL with missing module errors.

- [ ] **Step 3: Implement a strict `nano_functional_eval.v1` config renderer**

Required path keys are `generated_jsonl`, `ar_checkpoint_dir`,
`source_base_parquet`, `target_model`, and `report_json`. Required evaluation
keys are `boundary`, `validation_limit`, `test_limit`, `batch_size`, and the
three identity tolerances. Reject missing keys and non-positive limits.

- [ ] **Step 4: Implement functional evaluation data flow**

The evaluator must:

```python
records = read_generated_jsonl(args.generated_jsonl)
selected = select_exact_split_rows(records, args.validation_limit, args.test_limit)
sources = resolve_source_rows(args.source_base_parquet, selected)
prompts = build_ar_prompts(selected, args.control, critic_template)
predicted = predict_prompts(ar_model, ar_tokenizer, prompts, batch_size=args.ar_batch_size)
replacements = rescale_direction(torch.from_numpy(predicted), gold_activations)
```

For each row, run one unmodified forward and one batched forward containing the
identity replacement, candidate replacement, teacher replacement, SFT
replacement when supplied, mean activation, zero activation, and shuffled
activation. Hook block `boundary - 1`, gather logits at the true final token,
and compute metrics with `functional_logit_metrics`.

- [ ] **Step 5: Enforce identity before candidate aggregation**

Use the existing extraction tolerances and write row-level identity metrics.
If any row exceeds tolerance, write a partial report with `identity_passed:
false`, list failing provenance keys, and exit 2 without candidate summaries.

- [ ] **Step 6: Add paired summaries and a lightweight manifest**

For each split and variant, store row-level metrics plus means. Compare every
candidate with SFT using `paired_bootstrap_improvement` where lower KL/JSD is
better and higher overlap/correlation is better. Record source path, source
schema, model path, boundary, config hash, code revision, and row keys.

- [ ] **Step 7: Run dependency-light tests**

Run: `python3 -m pytest tests/test_nano_eval_core.py tests/test_nano_r33_source_rows.py tests/test_nano_r33_functional_core.py tests/test_eval_nano_r33_functional_recovery.py tests/test_nano_functional_eval_config.py -q`

Expected: PASS.

- [ ] **Step 8: Commit the evaluator**

```bash
git add scripts/eval_nano_r33_functional_recovery.py scripts/nano_functional_eval_config.py tests/test_eval_nano_r33_functional_recovery.py tests/test_nano_functional_eval_config.py
git commit -m "feat: evaluate R33 functional recovery"
```

### Task 6: Add The Composite Validity Gate

**Files:**
- Create: `scripts/eval_nano_r33_validity_gate.py`
- Create: `tests/test_eval_nano_r33_validity_gate.py`
- Create: `configs/nano_rl/r33_component_validity_eval.yaml`

- [ ] **Step 1: Write failing tests for every hard gate**

Create a passing fixture, then independently mutate each condition:

```python
assert evaluate_gate(passing_fixture())["passed"] is True

for mutation in (
    regress_roundtrip_nmse,
    cross_zero_nmse_ci,
    regress_functional_kl,
    regress_topk_overlap,
    lose_transform_fve,
    lower_control_win_rate,
    lower_parse_health,
    add_cjk_leak,
    exceed_qualitative_flag_rate,
):
    assert evaluate_gate(mutation(passing_fixture()))["passed"] is False
```

- [ ] **Step 2: Run the test and confirm the gate module is missing**

Run: `python3 -m pytest tests/test_eval_nano_r33_validity_gate.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement report loading, row-identity checks, and explicit reasons**

`evaluate_gate` must first require identical split and row keys across SFT,
candidate, transform, and functional reports. It returns:

```python
{
    "schema_version": "nano_r33_validity_gate.v1",
    "passed": all(check["passed"] for check in checks),
    "checks": checks,
    "blockers": [check["name"] for check in checks if not check["passed"]],
}
```

Every check records observed value, threshold, split, and evidence path.

- [ ] **Step 4: Add the checked-in validity configuration**

The YAML names SFT, update-16, and update-32 generated JSONLs and reports,
requires 512/512 rows, points temporary HF conversion roots to `/dev/shm`, and
sets the exact thresholds from the approved design. It does not launch
training.

- [ ] **Step 5: Run gate and config tests**

Run: `python3 -m pytest tests/test_eval_nano_r33_validity_gate.py -q`

Expected: PASS.

- [ ] **Step 6: Commit validity orchestration**

```bash
git add scripts/eval_nano_r33_validity_gate.py tests/test_eval_nano_r33_validity_gate.py configs/nano_rl/r33_component_validity_eval.yaml
git commit -m "feat: gate R33 RL on independent validity"
```

### Task 7: Expose Non-Negative KL And Reject Truncated Batches

**Files:**
- Modify: `external/natural_language_autoencoders/configs/rl.sh`
- Modify: `scripts/nano_rl_queue.py`
- Modify: `tests/test_nano_miles_launcher.py`
- Modify: `tests/test_nano_rl_queue.py`

- [ ] **Step 1: Write failing launcher tests for `k3` rendering**

```python
def test_rl_launcher_renders_configurable_kl_type():
    text = RL_SCRIPT.read_text()
    assert 'KL_LOSS_TYPE="${KL_LOSS_TYPE:-k1}"' in text
    assert '--kl-loss-type "$KL_LOSS_TYPE"' in text
```

- [ ] **Step 2: Write failing queue tests for exact batch divisibility**

```python
def test_rejects_global_batch_not_divisible_by_actor_dp_times_microbatch():
    queue = queue_fixture(actor_gpus=6, actor_micro_batch=32,
                          rollout_batch_size=30, n_samples_per_prompt=16,
                          global_batch_size=480, require_exact_actor_batch=True)
    with pytest.raises(QueueError, match="actor_gpus.*actor_micro_batch"):
        build_run_spec(queue)


def test_accepts_exact_384_sample_batch():
    queue = queue_fixture(actor_gpus=6, actor_micro_batch=32,
                          rollout_batch_size=48, n_samples_per_prompt=8,
                          global_batch_size=384, require_exact_actor_batch=True)
    spec = build_run_spec(queue)
    assert spec["rollout_batch_plan"]["effective_trained_samples"] == 384
```

- [ ] **Step 3: Run focused tests and confirm failures**

Run: `python3 -m pytest tests/test_nano_miles_launcher.py tests/test_nano_rl_queue.py -q`

Expected: FAIL because KL type and actor-batch divisibility are not rendered.

- [ ] **Step 4: Add validated KL type to `rl.sh`**

```bash
KL_LOSS_TYPE="${KL_LOSS_TYPE:-k1}"
case "$KL_LOSS_TYPE" in
    k1 | k2 | k3 | low_var_kl) ;;
    *) echo "unsupported KL_LOSS_TYPE=$KL_LOSS_TYPE" >&2; exit 2 ;;
esac

if "$PYTHON" -c "import sys; sys.exit(0 if float('$KL_LOSS_COEF') != 0 else 1)"; then
    KL_FLAGS=(--use-kl-loss --kl-loss-coef "$KL_LOSS_COEF" --kl-loss-type "$KL_LOSS_TYPE")
else
    KL_FLAGS=()
fi
```

Add `KL_LOSS_TYPE` to `NLA_TRAIN_ENV_KEYS` so Ray workers receive it.

- [ ] **Step 5: Add queue rendering and strict divisibility validation**

```python
def _validate_actor_batch_plan(*, generated_samples, global_batch_size,
                               actor_gpus, actor_micro_batch, required):
    if generated_samples != global_batch_size:
        raise QueueError("generated_samples must equal global_batch_size")
    divisor = actor_gpus * actor_micro_batch
    if global_batch_size % divisor:
        message = (
            f"global_batch_size={global_batch_size} must be divisible by "
            f"actor_gpus={actor_gpus} * actor_micro_batch={actor_micro_batch}"
        )
        if required:
            raise QueueError(message)
        return {"effective_trained_samples": global_batch_size - global_batch_size % divisor,
                "samples_per_actor": global_batch_size // actor_gpus,
                "warning": message}
    return {"effective_trained_samples": global_batch_size, "samples_per_actor": global_batch_size // actor_gpus}
```

Set `env["KL_LOSS_TYPE"]` from `training.kl_loss_type`, defaulting to `k1` for
backward compatibility. Corrected configs must pin `k3` explicitly. Invoke
strict divisibility when `training.require_exact_actor_batch` is true; retain a
warning-only legacy path for historical configs that intentionally documented
truncation.

- [ ] **Step 6: Run focused and shell-syntax tests**

Run: `bash -n external/natural_language_autoencoders/configs/rl.sh`

Run: `python3 -m pytest tests/test_nano_miles_launcher.py tests/test_nano_rl_queue.py -q`

Expected: PASS.

- [ ] **Step 7: Commit only reviewed launcher and queue hunks**

Because these files were already dirty before this plan, inspect the staged
diff and ensure the commit retains all pre-existing work without reverting it.

```bash
git add external/natural_language_autoencoders/configs/rl.sh scripts/nano_rl_queue.py tests/test_nano_miles_launcher.py tests/test_nano_rl_queue.py
git diff --cached --check
git commit -m "fix: use exact batches and configurable KL"
```

### Task 8: Add A Generic Miles Train-Guard Hook And NLA Drift Guard

**Files:**
- Create: `external/natural_language_autoencoders/nla/train_guard.py`
- Create: `external/natural_language_autoencoders/nla/miles_patches/0017_train_guard_hook.patch`
- Modify: `external/natural_language_autoencoders/configs/rl.sh`
- Modify: `scripts/nano_rl_queue.py`
- Create: `tests/test_nla_train_guard.py`
- Modify: `tests/test_nano_miles_launcher.py`
- Modify: `tests/test_nano_rl_queue.py`

- [ ] **Step 1: Write failing state-machine tests**

```python
from nla.train_guard import DriftGuard, DriftGuardTriggered


def test_guard_requires_two_consecutive_exceedances():
    guard = DriftGuard(max_logprob_abs_diff=0.75, consecutive_steps=2)
    guard.check({"train/train_rollout_logprob_abs_diff": 0.8}, role="actor", step=3)
    with pytest.raises(DriftGuardTriggered):
        guard.check({"train/train_rollout_logprob_abs_diff": 0.9}, role="actor", step=4)


def test_guard_resets_after_healthy_step():
    guard = DriftGuard(max_logprob_abs_diff=0.75, consecutive_steps=2)
    guard.check({"train/train_rollout_logprob_abs_diff": 0.8}, role="actor", step=3)
    guard.check({"train/train_rollout_logprob_abs_diff": 0.2}, role="actor", step=4)
    guard.check({"train/train_rollout_logprob_abs_diff": 0.8}, role="actor", step=5)
```

- [ ] **Step 2: Run the guard test and confirm the module is missing**

Run: `PYTHONPATH=external/natural_language_autoencoders python3 -m pytest tests/test_nla_train_guard.py -q`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement the role-aware guard**

`DriftGuard.check` ignores critic metrics, validates every numeric actor metric
for finiteness, increments a consecutive counter only when the configured
log-probability metric exceeds the threshold, and raises a structured error on
the configured count. `check_train_metrics(args, metrics, role, step)` lazily
constructs one process-local guard from environment variables.

- [ ] **Step 4: Add a generic optional hook to Miles**

Patch Miles to add `--custom-train-guard-function-path`. In
`miles/backends/training_utils/log_utils.py::log_train_step`, after building
`log_dict_out` and before returning, load and call:

```python
if args.custom_train_guard_function_path:
    guard = load_function(args.custom_train_guard_function_path)
    guard(args=args, metrics=log_dict_out, role=role, step=accumulated_step_id)
```

The call executes on every actor rank because aggregated metrics are identical;
all ranks therefore fail together instead of deadlocking a collective.

- [ ] **Step 5: Render guard configuration through shell and YAML**

Add these YAML-backed environment values:

```text
NLA_TRAIN_GUARD_MAX_LOGPROB_ABS_DIFF=0.75
NLA_TRAIN_GUARD_CONSECUTIVE_STEPS=2
```

Render `--custom-train-guard-function-path nla.train_guard.check_train_metrics`
when `training.drift_guard.enabled` is true.

- [ ] **Step 6: Verify patch syntax and guard tests**

Run: `python3 scripts/check_miles_patches.py`

Run: `PYTHONPATH=external/natural_language_autoencoders python3 -m pytest tests/test_nla_train_guard.py tests/test_nano_miles_launcher.py tests/test_nano_rl_queue.py -q`

Expected: patch check and tests PASS.

- [ ] **Step 7: Commit the guard as one coherent change**

```bash
git add external/natural_language_autoencoders/nla/train_guard.py external/natural_language_autoencoders/nla/miles_patches/0017_train_guard_hook.patch external/natural_language_autoencoders/configs/rl.sh scripts/nano_rl_queue.py tests/test_nla_train_guard.py tests/test_nano_miles_launcher.py tests/test_nano_rl_queue.py
git commit -m "feat: stop unstable NLA RL updates"
```

### Task 9: Add Corrected Probe, Dataset, And Blocked Hero Configurations

**Files:**
- Create: `configs/nano_rl/r33_component_corrected_k3_hpo_queue_8h100.yaml`
- Create: `configs/nano_rl/r33_component_fixed_ar_hero_queue_8h100.yaml`
- Create: `scripts/build_nano_r33_rl_dataset.py`
- Create: `scripts/verify_nano_r33_rl_dataset.py`
- Create: `scripts/nano_checkpoint_retention.py`
- Create: `tests/test_nano_r33_rl_dataset.py`
- Create: `tests/test_nano_checkpoint_retention.py`
- Modify: `tests/test_nano_rl_queue.py`

- [ ] **Step 1: Write failing checked-in config tests**

```python
def test_corrected_k3_queue_has_two_probes_and_blocked_confirmation():
    queue = load_yaml(CONFIGS / "r33_component_corrected_k3_hpo_queue_8h100.yaml")
    assert [item["training"]["actor_lr"] for item in queue["items"][:2]] == ["1e-5", "2e-5"]
    assert all(item["training"]["kl_loss_type"] == "k3" for item in queue["items"])
    assert all(item["rollout"]["rollout_batch_size"] == 48 for item in queue["items"])
    assert all(item["rollout"]["n_samples_per_prompt"] == 8 for item in queue["items"])
    assert all(item["rollout"]["global_batch_size"] == 384 for item in queue["items"])
    assert queue["items"][2]["status"] == "blocked"


def test_hero_queue_is_blocked_on_validity_gate():
    queue = load_yaml(CONFIGS / "r33_component_fixed_ar_hero_queue_8h100.yaml")
    assert all(item["status"] == "blocked" for item in queue["items"])
    assert queue["items"][-1]["rollout"]["num_rollout"] == 256
```

- [ ] **Step 2: Write failing dataset builder/verifier tests**

Use tiny PyArrow fixtures with train, validation, and test documents. Assert the
builder includes only train rows, preserves `token_ids_prefix`, and writes
`nano_r33_rl_dataset.v1` lineage. Assert the verifier rejects non-finite
vectors, `d_model != 2688`, duplicated provenance keys, and any held-out doc.

- [ ] **Step 3: Write failing manifest-driven retention tests**

```python
def test_retention_never_selects_protected_or_last_known_good(tmp_path):
    policy = RetentionPolicy(
        output_root=tmp_path,
        protected={tmp_path / "sft", tmp_path / "ar", tmp_path / "best"},
        keep_challenger=tmp_path / "challenger",
    )
    plan = build_cleanup_plan(policy, candidates=[
        tmp_path / "sft", tmp_path / "best", tmp_path / "challenger", tmp_path / "loser"
    ])
    assert plan.delete == [tmp_path / "loser"]
    assert tmp_path / "best" in plan.keep
```

Also assert `--apply` refuses symlinks and paths outside `output_root`, and that
the JSON cleanup manifest is written before the first deletion.

- [ ] **Step 4: Run config, dataset, and retention tests and confirm missing files**

Run: `python3 -m pytest tests/test_nano_r33_rl_dataset.py tests/test_nano_checkpoint_retention.py tests/test_nano_rl_queue.py -q`

Expected: FAIL because scripts and configs do not exist.

- [ ] **Step 5: Implement the train-only dataset builder**

The builder accepts `--base-parquet`, `--split-manifest`, `--output`, and
`--report-json`. It streams batches, filters by train document/component IDs,
and writes `prompt`, `activation_vector`, stable provenance,
`token_ids_prefix`, and source lineage. It never requires or generates teacher
text.

- [ ] **Step 6: Implement strict verification**

The verifier reports rows, unique documents, key uniqueness, dimension counts,
non-finite rows, split overlap, content-component overlap, and source hashes.
It exits nonzero on any blocker and writes the report before exit.

- [ ] **Step 7: Implement guarded retention planning and apply mode**

The CLI is dry-run by default. It accepts an explicit JSON policy containing
`output_root`, `protected`, `current_best`, `challenger`, and `candidates`.
Resolve every path, reject symlinks and out-of-root paths, write the full keep
and delete manifest atomically, and only then remove listed candidate
directories when `--apply` is present.

- [ ] **Step 8: Add the corrected probe queue**

Defaults:

```yaml
resources: {actor_gpus: 6, critic_gpus: 1, rollout_gpus: 1}
training:
  actor_micro_batch: 32
  kl_loss_type: k3
  kl_loss_coef: 0.001
  drift_guard:
    enabled: true
    max_logprob_abs_diff: 0.75
    consecutive_steps: 2
rollout:
  rollout_batch_size: 48
  n_samples_per_prompt: 8
  global_batch_size: 384
  num_rollout: 8
```

Items 1 and 2 use actor LR `1e-5` and `2e-5`. Item 3 is a blocked 32-update
confirmation dependent on a named winning probe and validity-gate pass. Keep
W&B offline and model-only checkpoints at 8, 16, 24, and 32.

- [ ] **Step 9: Add the blocked hero queue**

Include one-update `6+1+1` and `5+1+2` canaries plus a blocked 256-update item.
The hero saves model-only checkpoints at 64, 128, and 256 and uses temporary
HF roots under `/dev/shm`. Every item depends on the corrected Stage 2 validity
report; nothing is pending at commit time.

- [ ] **Step 10: Run tests and dry-run both queues**

Run: `python3 -m pytest tests/test_nano_r33_rl_dataset.py tests/test_nano_checkpoint_retention.py tests/test_nano_rl_queue.py -q`

Run: `python3 scripts/nano_rl_queue.py configs/nano_rl/r33_component_corrected_k3_hpo_queue_8h100.yaml --dry-run`

Run: `python3 scripts/nano_rl_queue.py configs/nano_rl/r33_component_fixed_ar_hero_queue_8h100.yaml --dry-run`

Expected: tests PASS; corrected probe 1 is the only initial pending training
item; every hero item remains blocked.

- [ ] **Step 11: Commit configs, dataset tooling, and retention guard**

```bash
git add configs/nano_rl/r33_component_corrected_k3_hpo_queue_8h100.yaml configs/nano_rl/r33_component_fixed_ar_hero_queue_8h100.yaml scripts/build_nano_r33_rl_dataset.py scripts/verify_nano_r33_rl_dataset.py scripts/nano_checkpoint_retention.py tests/test_nano_r33_rl_dataset.py tests/test_nano_checkpoint_retention.py tests/test_nano_rl_queue.py
git commit -m "feat: queue corrected R33 fixed-AR scaling"
```

### Task 10: Verify, Sync, Run Validity Evaluation, Then Submit Corrected Probes

**Files:**
- Modify: `docs/rl_logbook.md`
- Modify: `docs/experiment_logbook.md`
- Modify: `docs/nano_av_job_tracker.md`

- [ ] **Step 1: Run the complete local dependency-light suite**

Run:

```bash
python3 -m pytest \
  tests/test_nano_eval_core.py \
  tests/test_nano_roundtrip_transforms.py \
  tests/test_eval_nano_roundtrip_invariance.py \
  tests/test_nano_r33_source_rows.py \
  tests/test_nano_r33_functional_core.py \
  tests/test_eval_nano_r33_functional_recovery.py \
  tests/test_nano_functional_eval_config.py \
  tests/test_eval_nano_r33_validity_gate.py \
  tests/test_nla_train_guard.py \
  tests/test_nano_miles_launcher.py \
  tests/test_nano_rl_queue.py \
  tests/test_nano_r33_rl_dataset.py \
  tests/test_nano_checkpoint_retention.py -q
```

Expected: PASS or explicit dependency skips only; `git diff --check` PASS.

- [ ] **Step 2: Build a source-only S3 sync bundle**

Follow `docs/runbooks/runai_s3_sync.md`. Exclude checkpoints, parquets,
artifacts, W&B payloads, caches, `.git`, and secrets. Upload the archive and
SHA256 manifest to the project sync prefix, then fetch and unpack it into
`/workspace/interp/code/nano30b-nla-pilot-current` without deleting unrelated
remote artifacts.

- [ ] **Step 3: Apply the new Miles patch and run RunAI tests**

Run inside `train`:

```bash
/workspace/interp/.venv/bin/python scripts/check_miles_patches.py
/workspace/interp/.venv/bin/python -m pytest \
  tests/test_nano_eval_core.py \
  tests/test_nano_roundtrip_transforms.py \
  tests/test_eval_nano_roundtrip_invariance.py \
  tests/test_nano_r33_source_rows.py \
  tests/test_nano_r33_functional_core.py \
  tests/test_eval_nano_r33_functional_recovery.py \
  tests/test_eval_nano_r33_validity_gate.py \
  tests/test_nla_train_guard.py \
  tests/test_nano_miles_launcher.py \
  tests/test_nano_rl_queue.py -q
```

Expected: patch check and tests PASS.

- [ ] **Step 4: Run a four-row R33 functional identity smoke**

Use the clean SFT generated JSONL first. Require all four source rows to resolve,
all identity canaries to pass, finite functional metrics, and temporary model
cleanup. Stop without patching code if identity fails; preserve the report and
last 100 log lines.

- [ ] **Step 5: Run Stage 1 on SFT, update 16, and update 32**

Convert one candidate at a time under `/dev/shm`, run 512/512 round-trip reuse,
deterministic transforms, functional recovery, and composite validity scoring,
then remove only that temporary HF directory. Preserve all generated JSONLs and
reports.

- [ ] **Step 6: Make the Stage 1 decision before training**

If the update-32 checkpoint passes, retain it as the scouting best and proceed.
If it fails only semantic or functional gates, record the exact blocker and run
the corrected probes, but keep Stage 3 blocked. If source identity fails, stop
all training and treat extraction provenance as the blocker.

- [ ] **Step 7: Audit response closure position and select cap**

Tokenize existing valid update-32 responses with the AV tokenizer. Select the
smallest of 150, 192, 224, and 256 for which at least 95% close on both
held-out splits. Write the audit JSON and update the corrected queue before
launch.

- [ ] **Step 8: Launch corrected probes through the queue**

Run the `1e-5` probe first, then `2e-5`. Evaluate and clean each sequentially.
Promote at most one to the 32-update confirmation. Do not unblock or launch the
hero queue until the composite validity gate passes.

- [ ] **Step 9: Build and verify the full train-only RL parquet while probes run**

This CPU/storage task may run concurrently because it does not use GPUs. Require
the verifier report to pass before any hero item can become pending.

- [ ] **Step 10: Update all experiment records after each completed milestone**

Record exact config, code revision, rows, checkpoint, round-trip and functional
metrics, transform retention, text health, policy dynamics, topology, elapsed
time, peak memory, storage before/after, decision, and cleanup manifest. Label
all historical signed-`k1` runs as scouting evidence.

- [ ] **Step 11: Run final verification before unblocking hero configs**

Run the full focused tests locally and on RunAI, validate all report hashes,
confirm at least 300 GB free on `/workspace/interp`, and confirm only the SFT,
selected AR, current best RL, and one challenger checkpoint remain.

- [ ] **Step 12: Commit lightweight evidence and documentation**

```bash
git add docs/rl_logbook.md docs/experiment_logbook.md docs/nano_av_job_tracker.md configs/nano_rl scripts tests external/natural_language_autoencoders
git diff --cached --check
git commit -m "docs: record R33 validity-first RL results"
```
