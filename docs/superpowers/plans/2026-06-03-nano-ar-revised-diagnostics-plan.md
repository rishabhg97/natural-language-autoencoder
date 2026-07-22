# Nano30B AR Revised Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the diagnostic-first AR improvement workflow from the revised design, starting with correctness audits and information-ceiling baselines before launching new AR training probes.

**Architecture:** Add small, focused diagnostic scripts that read existing AR parquets, checkpoint sidecars, eval reports, and optional model files. Keep training code unchanged until diagnostics identify a concrete lever. Extend eval/reporting only where needed to make row-level predictions, bootstrap CIs, and promotion decisions reproducible.

**Tech Stack:** Python, PyArrow, NumPy, PyTorch where needed, safetensors where available, existing NLA sidecar/config helpers, pytest, YAML experiment configs, RunAI queue scripts.

---

## File Structure

- Create `scripts/nano_ar_correctness_audit.py`
  - Reads AR split parquets, sidecars, and checkpoint metadata.
  - Emits JSON/Markdown checks for R27 boundary, critic hidden layers, suffix/last-token contract, identity value-head distance, split overlap, and value-head/norm metadata.

- Create `scripts/nano_ar_information_ceiling.py`
  - Computes duplicate-explanation floors, near-neighbor text floors, token/position/local-window baselines, hard-negative summaries, and specificity buckets from existing AR split parquets.

- Modify `scripts/eval_nano_ar_miles_checkpoint.py`
  - Add optional row-level prediction dump.
  - Add paired bootstrap confidence intervals for teacher/control NMSE deltas.
  - Preserve current default report shape unless flags are passed.

- Extend `scripts/nano_ar_frozen_baseline.py`
  - Reuse existing value-head utilities for closed-form/frozen-feature baselines when possible.
  - Add ridge/procrustes helpers if not already present.

- Create or extend tests:
  - `tests/test_nano_ar_correctness_audit.py`
  - `tests/test_nano_ar_information_ceiling.py`
  - `tests/test_eval_nano_ar_report_extensions.py`
  - `tests/test_nano_ar_frozen_baseline_math.py`

- Update docs/logs:
  - `docs/experiment_logbook.md` after diagnostics are run.
  - `docs/nano_av_job_tracker.md` only when a new run/probe is launched or completed.

## Task 1: Correctness Audit Script

**Files:**
- Create: `scripts/nano_ar_correctness_audit.py`
- Test: `tests/test_nano_ar_correctness_audit.py`

- [x] **Step 1: Write failing tests for R27 boundary logic**

Test cases:

```python
def test_expected_hidden_layers_for_r27():
    assert expected_hidden_layers("R27") == 27
    assert expected_zero_based_layer("R27") == 26
```

- [x] **Step 2: Write failing tests for split overlap detection**

Use small synthetic rows with `doc_id` overlap and assert the audit marks
overlap as failing.

- [x] **Step 3: Write failing tests for identity value-head distance**

Create a tiny safetensors or torch tensor weight shaped like identity and a
non-identity matrix. Assert normalized Frobenius distance is `0` for identity
and positive for the perturbed weight.

- [x] **Step 4: Implement boundary, overlap, and identity helpers**

Keep functions pure and separately testable:

- `parse_boundary_name`
- `expected_zero_based_layer`
- `expected_hidden_layers`
- `doc_overlap_summary`
- `identity_distance`

- [x] **Step 5: Implement checkpoint/sidecar audit assembly**

Read `config.json`, `nla_meta.yaml`, tokenizer metadata if available, and
`value_head.safetensors` if present. Missing optional files should be recorded
as warnings, not hidden.

- [x] **Step 6: Add CLI JSON/Markdown output**

CLI should accept:

```text
--checkpoint-dir
--train-parquet
--validation-parquet
--test-parquet
--boundary-name R27
--report-json
--report-md
```

- [x] **Step 7: Run tests and commit**

Run:

```bash
pytest tests/test_nano_ar_correctness_audit.py
```

Commit message:

```text
nano30b: add AR correctness audit
```

## Task 2: Information-Ceiling Diagnostics

**Files:**
- Create: `scripts/nano_ar_information_ceiling.py`
- Test: `tests/test_nano_ar_information_ceiling.py`

- [x] **Step 1: Write tests for canonical explanation grouping**

Verify XML tag stripping, whitespace normalization, lowercase normalization, and
boilerplate-insensitive grouping.

- [x] **Step 2: Write tests for group constant floor**

Use known unit vectors where the normalized group mean has an exact expected
NMSE. Assert group floor math matches.

- [x] **Step 3: Write tests for nearest-neighbor floor**

Use a synthetic embedding matrix and known nearest neighbors. Assert the
predicted activation is the normalized mean of retrieved train activations.

- [x] **Step 4: Write tests for token/position/local-window baseline extraction**

Use synthetic rows with token fields and target metadata. Assert baseline keys
are generated deterministically.

- [x] **Step 5: Implement diagnostic helpers**

Pure helpers:

- `canonicalize_explanation`
- `normalized_mean_vector`
- `normalized_mse_rows`
- `duplicate_group_floor`
- `knn_floor`
- `bucket_summary`
- `hard_negative_summary`

- [x] **Step 6: Implement CLI report**

CLI should read split parquets and emit:

- duplicate exact floor
- near-neighbor floor
- token/position/local-window baseline summaries
- specificity buckets
- hard-negative summary

- [x] **Step 7: Run tests and commit**

Run:

```bash
pytest tests/test_nano_ar_information_ceiling.py
```

Commit message:

```text
nano30b: add AR information ceiling diagnostics
```

## Task 3: Eval Report Extensions

**Files:**
- Modify: `scripts/eval_nano_ar_miles_checkpoint.py`
- Test: `tests/test_eval_nano_ar_report_extensions.py`

- [x] **Step 1: Write tests for bootstrap CI math**

Use paired row losses with a deterministic seed and assert output keys:

- `mean`
- `ci_low`
- `ci_high`
- `n`

- [x] **Step 2: Write tests for row-level prediction dump shape**

Mock predictions/targets and assert rows include:

- row index
- doc ID
- split
- control
- normalized MSE
- cosine
- pred norm
- gold norm

- [x] **Step 3: Add optional CLI flags**

Add:

```text
--prediction-dump-jsonl
--bootstrap-samples
--bootstrap-seed
```

Default behavior must remain unchanged when flags are omitted.

- [x] **Step 4: Implement CI and dump generation**

Use existing per-control rowwise losses. Keep output bounded and stream JSONL
for dumps.

- [x] **Step 5: Run tests and commit**

Run:

```bash
pytest tests/test_eval_nano_ar_report_extensions.py
```

Commit message:

```text
nano30b: extend AR eval diagnostics
```

## Task 4: Frozen-Feature Closed-Form Baselines

**Files:**
- Modify: `scripts/nano_ar_frozen_baseline.py`
- Test: `tests/test_nano_ar_frozen_baseline_math.py`

- [x] **Step 1: Inspect existing baseline utilities**

Confirm whether ridge, linear+bias, and head evaluation already exist. Reuse
them instead of adding duplicate helpers.

- [x] **Step 2: Write tests for ridge fit**

Use a small synthetic linear system and assert the recovered prediction beats the
mean baseline.

- [x] **Step 3: Write tests for Procrustes fit**

Use an orthogonal transform synthetic system and assert the fitted map recovers
targets within tolerance.

- [x] **Step 4: Add readout mode metadata**

Support report rows for:

- final token
- mean pool
- last-k pool

Actual GPU feature extraction can remain a later RunAI execution step; local
tests cover math and report structure.

- [x] **Step 5: Run tests and commit**

Run:

```bash
pytest tests/test_nano_ar_frozen_baseline_math.py
```

Commit message:

```text
nano30b: add AR frozen-feature baseline math
```

## Task 5: Run Diagnostics On Current Artifacts

**Files:**
- Update: `docs/experiment_logbook.md`

- [ ] **Step 1: Verify RunAI train status**

Use Visor and do not print secrets.

- [ ] **Step 2: Run correctness audit on best AR checkpoint**

Target checkpoint:

```text
/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-hpo/nano-ar-r27-best1547-continue-lr2e5-cosine-256steps-20260602T0710Z
```

- [ ] **Step 3: Run information-ceiling diagnostics on current splits**

Use the same train/validation/test parquets as the current best AR eval.

- [ ] **Step 4: Run eval extension on current best checkpoint**

Generate row-level prediction dump and bootstrap CIs with `512/512` limits.

- [ ] **Step 5: Summarize diagnostic read**

Append a concise logbook section with:

- audit pass/fail
- information floor values
- bootstrap CI around current best
- whether next step should be contract repair, readout/head probe, or data/text
  enrichment design

- [ ] **Step 6: Commit diagnostics summary**

Commit message:

```text
nano30b: record AR diagnostic results
```

## Task 6: Select First Training Probe Only After Diagnostics

**Files:**
- Create or modify: `configs/nano_ar/hpo/*.yaml`
- Update: `docs/nano_av_job_tracker.md`

- [ ] **Step 1: Choose probe family from diagnostics**

Decision rules:

- correctness audit fails -> repair contract, no training
- information floor near `0.40-0.45` -> design text/hint enrichment, no broad HPO
- frozen-feature/readout beats `0.40` -> queue matching readout/head probe
- source-raw curriculum appears promising -> queue curriculum probe

- [ ] **Step 2: Create only one or two bounded configs**

Use `128-256` steps and `512/512` eval first.

- [ ] **Step 3: Queue with existing watcher**

Use current YAML queue process. Keep W&B offline and checkpoint retention
minimal.

- [ ] **Step 4: Commit queued config and tracker update**

Commit message:

```text
nano30b: queue AR diagnostic probe
```

## Verification Rules

- Every task must run its focused tests before commit.
- No GPU training launches before Tasks 1-5 complete.
- Do not overwrite unrelated dirty files.
- Do not delete checkpoints in this plan unless explicitly requested.
- Do not print secrets.
