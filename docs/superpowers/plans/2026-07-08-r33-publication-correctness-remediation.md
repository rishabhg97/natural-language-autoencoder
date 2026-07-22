# R33 Publication Correctness Remediation Implementation Plan

> **Historical plan, not an execution guide.** The July 11 remediation
> supersedes its queue paths, metric names, fidelity assumptions, and proposed
> publication-specific runtime branch. Use `docs/current_state.md`,
> `docs/methods/measurement_contract.md`, and the ordinary config-driven queues.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a mathematically defensible, reproducible R33 NLA result with protocol-matched baselines, family-disjoint evaluation, independent-critic transfer, correctly named metrics, and a predeclared confirmatory RL replication.

**Architecture:** Split the work into three gates. First harden the measurement and provenance system so invalid comparisons fail closed. Second use that system to rescore the retained hero and determine whether its signal survives without retraining. Third, only if the corrected signal survives, run a frozen-config independent replication and a one-shot publication test.

**Tech Stack:** Python 3.12, NumPy, PyArrow, PyYAML, Hugging Face/Miles FSDP2, SGLang generation, offline W&B, pytest, RunAI, S3.

---

## Publication Claim Policy

The project must distinguish two claim tiers:

1. **Directional reconstruction:** cosine similarity improves under a metric that L2-normalizes both hidden states. This must never be called raw hidden-state reconstruction.
2. **Magnitude-aware activation reconstruction:** raw MSE beats both SFT and the train-mean predictor, with positive centered R2 on heldout families.

The current checkpoint may qualify for tier 1 after corrected evaluation. It does not currently qualify for tier 2.

No public headline may use the existing `30.97% / 32.34%` result. It remains archived as an invalidated exploratory comparison because half of its SFT baseline used a mismatched generation prefix.

## File Map

| Responsibility | Files |
|---|---|
| Core paired statistics and metric definitions | `scripts/nano_eval_core.py`, `tests/test_nano_eval_core.py` |
| Round-trip generation, scoring, reports, and gates | `scripts/eval_nano_av_ar_roundtrip_gate.py`, `tests/test_nano_av_ar_roundtrip_gate.py` |
| Config-to-command contract | `scripts/nano_roundtrip_eval_config.py`, `tests/test_nano_roundtrip_eval_config.py` |
| Content-family construction and heldout selection | `scripts/nano_functional_eval_data.py`, `tests/test_nano_functional_eval_data.py` |
| RL dataset fail-closed build and verification | `scripts/build_nano_r33_rl_dataset.py`, `scripts/verify_nano_r33_rl_dataset.py`, `tests/test_nano_r33_rl_dataset.py` |
| Source/data/runtime attestation | `scripts/nano_source_provenance.py`, `scripts/reconcile_miles_runtime.py`, `tests/test_nano_source_provenance.py`, `tests/test_reconcile_miles_runtime.py` |
| Queue launch and immutable run contract | `scripts/nano_rl_queue.py`, `tests/test_nano_rl_queue.py` |
| Cross-critic promotion | `scripts/eval_nano_cross_critic_gate.py`, `tests/test_nano_cross_critic_gate.py` |
| Stale-claim/document consistency check | New `scripts/verify_docs_consistency.py`, new `tests/test_verify_docs_consistency.py` |
| Publication configurations | New files under `configs/nano_roundtrip/publication/` and `configs/nano_rl/publication/` |
| Claim corrections and final report | `docs/current_state.md`, `docs/rl_logbook.md`, `docs/experiment_logbook.md`, `docs/runs/r33_rl_hero_20260708.md`, new publication report under `docs/runs/` |

### Task 1: Preserve The Audit And Invalidate The Old Headline

**Files:**
- Create: `docs/reviews/2026-07-08-r33-rl-hero-publication-audit.md`
- Create: `scripts/verify_docs_consistency.py`
- Create: `tests/test_verify_docs_consistency.py`
- Modify: `docs/current_state.md`
- Modify: `docs/runs/r33_rl_hero_20260708.md`
- Modify: `runs/registry/experiments.yaml`

- [x] **Step 1: Preserve the complete audit report in the repository**

Copy the supplied audit verbatim, then prepend provenance containing the audit date and evidence root:

```markdown
Evidence root: artifacts/runai_evidence/20260708T172617Z_complete_lightweight_evidence/
Status: external audit received; critical findings pending remediation
```

- [x] **Step 2: Mark the existing headline invalid for publication**

Add a prominent note to the current-state and hero documents:

```markdown
The archived 30.97%/32.34% comparison is arithmetically reproducible but not
publication-valid because the 512-row SFT baseline mixed two generation
protocols. Do not cite it as the final effect size.
```

- [x] **Step 3: Add the failed attempts and audit disposition to the registry**

Record the original hero, guard retries, resume lineage, baseline defect, and disposition instead of retaining only the winner.

- [x] **Step 4: Add a stale-claim consistency checker**

Write a focused checker that scans active claim documents, permits the old
numbers only in explicitly invalidated historical sections, and fails if an
active headline presents them as publication-valid. Test both accepted and
rejected fixtures before running it on the repository.

- [x] **Step 5: Verify documentation consistency**

Run:

```bash
/Users/rigarg/.local/bin/visor run python scripts/verify_docs_consistency.py
```

Expected: no active document presents `30.97%/32.34%` as publication-valid.

- [x] **Step 6: Commit**

```bash
git add docs/reviews docs/current_state.md docs/runs/r33_rl_hero_20260708.md runs/registry/experiments.yaml scripts/verify_docs_consistency.py tests/test_verify_docs_consistency.py
git commit -m "docs: invalidate mismatched R33 hero baseline"
```

### Task 2: Make Generation Protocol Parity Fail Closed

**Files:**
- Modify: `scripts/nano_roundtrip_eval_config.py`
- Modify: `scripts/eval_nano_av_ar_roundtrip_gate.py`
- Modify: `scripts/nano_av_probe_queue.py`
- Modify: `tests/test_nano_roundtrip_eval_config.py`
- Modify: `tests/test_nano_av_ar_roundtrip_gate.py`
- Modify: `tests/test_nano_av_probe_queue.py`

- [x] **Step 1: Write failing tests for protocol mismatch**

Add tests that construct candidate and baseline reports differing in one generation-mechanics field at a time: prefix bytes, backend, tokenizer, stop text, maximum new tokens, sampling parameters, and injection scale. Each mismatch must fail the publication gate. Model/checkpoint identity is recorded and hash-pinned as provenance, but is intentionally excluded from protocol equality because the SFT baseline and RL candidate must be different model checkpoints.

The report contract must contain:

```python
generation_protocol = {
    "backend": "legacy_batch",
    "prefix": "",
    "prefix_sha256": sha256(b"").hexdigest(),
    "stop_text": "</explanation>",
    "max_new_tokens": 256,
    "do_sample": False,
    "temperature": 0.0,
    "tokenizer_fingerprint": "...",
}
```

The adjacent `generation_provenance` object records the model fingerprint,
checkpoint, model revision, and tokenizer revision.

- [x] **Step 2: Run the focused tests and confirm failure**

```bash
/Users/rigarg/.local/bin/visor run .venv/bin/python -m pytest \
  tests/test_nano_roundtrip_eval_config.py \
  tests/test_nano_av_ar_roundtrip_gate.py -q
```

Expected: the new parity assertions fail before implementation.

- [x] **Step 3: Implement protocol capture and parity checks**

Make baseline/candidate protocol equality a required gate condition. Record the actual generation protocol when text is generated, not when cached text is later scored. Cached generation must carry a sidecar hash that binds every generated row to the protocol.

- [x] **Step 4: Reject ambiguous prefixes**

Publication configs must use no forced prefix for both SFT and RL. Preserve the old prefix configs for historical reproducibility, but do not permit them in publication gates.

- [x] **Step 5: Run focused and queue tests**

```bash
/Users/rigarg/.local/bin/visor run .venv/bin/python -m pytest \
  tests/test_nano_roundtrip_eval_config.py \
  tests/test_nano_av_ar_roundtrip_gate.py \
  tests/test_nano_rl_queue.py -q
```

Verification on RunAI: the changed protocol/config/AV queue surface passes
`44` tests plus `13` subtests. `tests/test_nano_rl_queue.py` remains at its
pre-existing baseline of `49 passed, 4 failed`; those four failures assert
`status: pending` against historical queues that are now `failed` or
`complete` and are addressed under Task 9's immutable queue work.

- [x] **Step 6: Commit**

```bash
git add scripts/nano_roundtrip_eval_config.py scripts/eval_nano_av_ar_roundtrip_gate.py tests
git commit -m "fix: bind round-trip gates to generation protocol"
```

### Task 3: Replace The Ambiguous NMSE With A Complete Metric Suite

**Files:**
- Modify: `scripts/nano_eval_core.py`
- Modify: `scripts/eval_nano_av_ar_roundtrip_gate.py`
- Modify: `tests/test_nano_eval_core.py`
- Modify: `tests/test_nano_av_ar_roundtrip_gate.py`

- [x] **Step 1: Write synthetic metric tests**

Use a prediction that is an exact scalar multiple of the target. Assert that cosine/directional MSE is perfect while raw MSE and centered R2 are not:

```python
target = np.array([[1.0, 2.0]], dtype=np.float64)
prediction = 10.0 * target
metrics = activation_reconstruction_metrics(prediction, target, train_mean=np.zeros(2))
assert metrics["cosine_mean"] == pytest.approx(1.0)
assert metrics["directional_mse"] == pytest.approx(0.0)
assert metrics["raw_mse"] > 0.0
assert metrics["centered_r2"] < 0.0
```

- [x] **Step 2: Implement one canonical metric function**

Add `activation_reconstruction_metrics(prediction, target, train_mean)` returning:

```python
{
    "directional_mse": float,
    "cosine_mean": float,
    "raw_mse": float,
    "mean_predictor_raw_mse": float,
    "centered_r2": float,
    "prediction_norm_mean": float,
    "target_norm_mean": float,
    "norm_ratio_mean": float,
}
```

Keep `normalized_mse` only as a deprecated alias for `directional_mse` in historical readers.

- [x] **Step 3: Emit rowwise directional and raw losses**

The gate report must support paired family-level uncertainty for both metric families.

- [x] **Step 4: Run tests**

```bash
/Users/rigarg/.local/bin/visor run .venv/bin/python -m pytest \
  tests/test_nano_eval_core.py tests/test_nano_av_ar_roundtrip_gate.py -q
```

Verification on Mac and in the RunAI venv: `55 passed` across the metric core,
round-trip gate, AR readout diagnostic, config renderer, and AV queue tests.

- [x] **Step 5: Commit**

```bash
git add scripts/nano_eval_core.py scripts/eval_nano_av_ar_roundtrip_gate.py tests
git commit -m "feat: report directional and raw activation recovery"
```

### Task 4: Build Shared Content Families And A Truly Heldout Evaluation Set

**Files:**
- Create: `scripts/build_nano_content_family_manifest.py`
- Modify: `scripts/nano_functional_eval_data.py`
- Modify: `scripts/eval_nano_av_ar_roundtrip_gate.py`
- Modify: `tests/test_nano_functional_eval_data.py`
- Modify: `tests/test_nano_av_ar_roundtrip_gate.py`
- Create: `configs/nano_roundtrip/publication/r33_family_eval_manifest.yaml`

- [x] **Step 1: Write tests for exact and shifted near-duplicates**

Fixtures must include identical text under different doc IDs and position-shifted copies. Assert that deterministic normalized-text shingles and union-find assign them to one `content_family_id`.

- [x] **Step 2: Implement deterministic family construction**

Use normalized source text plus token shingles. Store the normalization version, shingle width, similarity threshold, algorithm version, and family assignments in the manifest. Do not infer families from `doc_id` alone.

- [x] **Step 3: Create one shared AR/AV/RL split manifest**

Assign whole families, never rows or docs, to train, validation, and test. Require zero family overlap across all three datasets and both AV/AR source tables.

- [x] **Step 4: Replace first-N evaluation with seeded family-stratified sampling**

Sampling must be deterministic from a committed seed and spread rows across families. The report must state row count and independent family count separately.

- [x] **Step 5: Bootstrap at family level**

The publication gate must reject reports whose `independent_unit` is not
`content_family_id`. Report both a paired family bootstrap interval and an
exact/Monte-Carlo paired sign-flip permutation p-value. Require at least 100
independent families per publication split or explicitly downgrade the work
to a small-sample pilot.

- [x] **Step 6: Audit whether a fresh untouched holdout exists**

Generate a coverage report against every AR, AV, and RL training manifest. Decision:

```text
If at least 512 validation and 512 test rows from disjoint, never-trained
families exist: retain the current SFT checkpoints for corrected evaluation.
Otherwise: Task 10 clean AR/AV retraining becomes mandatory.
```

Result: a fresh holdout exists. After unioning AV-SFT, AR-SFT, and RL train
exposure and conservatively dropping two families shared by the candidate
holdouts, validation retains `11,007` rows / `214` families and test retains
`10,915` rows / `218` families with zero overlap. Existing clean SFT
checkpoints can be retained for corrected evaluation.

- [x] **Step 7: Run tests**

```bash
/Users/rigarg/.local/bin/visor run .venv/bin/python -m pytest \
  tests/test_nano_functional_eval_data.py \
  tests/test_nano_av_ar_roundtrip_gate.py -q
```

Verification in the RunAI venv: `72 passed` across family construction,
metric/gate, config, queue, and AR diagnostic tests. Production selection
preflight selected `512/512` rows spanning `214/218` families with zero
cross-split family overlap.

- [x] **Step 8: Commit**

```bash
git add scripts/nano_functional_eval_data.py scripts/eval_nano_av_ar_roundtrip_gate.py tests configs/nano_roundtrip/publication
git commit -m "feat: evaluate NLA on disjoint content families"
```

### Task 5: Make Dataset And Runtime Provenance Complete

**Files:**
- Modify: `scripts/build_nano_r33_rl_dataset.py`
- Modify: `scripts/verify_nano_r33_rl_dataset.py`
- Modify: `scripts/nano_source_provenance.py`
- Modify: `scripts/reconcile_miles_runtime.py`
- Modify: `scripts/nano_rl_queue.py`
- Modify: `tests/test_nano_r33_rl_dataset.py`
- Modify: `tests/test_nano_source_provenance.py`
- Modify: `tests/test_reconcile_miles_runtime.py`
- Modify: `tests/test_nano_rl_queue.py`

- [x] **Step 1: Add failing tests for every silent downgrade**

The builder and launcher must fail when component filtering is absent, a dataset hash differs, Miles or `miles_patches` is outside the source fingerprint, or a queue/config changes after launch.

- [x] **Step 2: Remove doc-ID-only fallback for publication runs**

Require `component_filter_applied=true`, exact family-manifest binding, expected row count, and zero heldout-family overlap.

- [x] **Step 3: Expand the source fingerprint boundary**

Hash all launch-critical project files, the complete Miles tree used by training, all `miles_patches`, resolved queue YAML, dataset manifest and parquet, Python package lock, container image digest, and checkpoint metadata.

- [x] **Step 4: Write an immutable launch contract**

Copy resolved configs into the run directory before launch and reject in-place mutation. Store hashes in `runtime_contracts.json` and W&B config.

- [x] **Step 5: Quantify stored-activation fidelity**

Recompute a seeded sample of R33 activations live and compare them with the
stored bf16 vectors. Report absolute/relative L2 error, cosine agreement, norm
ratio, and the effect on every primary metric. Bind the extraction code,
checkpoint, layer, dtype, and sample identities into the data manifest.

- [x] **Step 6: Run tests**

```bash
/Users/rigarg/.local/bin/visor run .venv/bin/python -m pytest \
  tests/test_nano_r33_rl_dataset.py \
  tests/test_nano_source_provenance.py \
  tests/test_reconcile_miles_runtime.py \
  tests/test_nano_rl_queue.py -q
```

- [x] **Step 7: Commit**

```bash
git add scripts tests
git commit -m "fix: fail closed on RL data and runtime provenance"
```

Task 5 empirical result (`2026-07-08`): the seeded `8/8` R33 activation
fidelity diagnostic is deterministic and fully fingerprinted, but all `16`
rows fail the existing identity tolerances. Current-runtime full-forward versus
stored vectors has mean/max relative L2 `0.023913 / 0.053206`; exact original
batch geometry has max relative L2 `0.073748`. AR-SFT vectors are byte-identical
to the original extraction parquet, so postprocessing is not responsible. The
report therefore records `publication_ready=false`; clean activation
re-extraction under a frozen runtime is required before publication retraining.

Task 5 remediation result (`2026-07-09`): the first frozen re-extraction was
also rejected because its execution profile did not bind deterministic
algorithms. Commit `0dabaad` added one shared fail-closed numerical profile to
the direct extractor, sharded extractor, dataset config, pipeline, and
diagnostic. The replacement `r33_deterministic_full275396` dataset passed both
all-row AR/AV verifiers and an independent full-stream replay. The two merged
275,396-row Parquets are byte-identical with SHA-256
`e3008a150831b8e894eac0de9f360a46823ffbfbd7cc73a9673f7e61e84521ac`.
The earlier sparse diagnostic is retained as a superseded method because it
did not replay stateful shard history from each worker's initial call.

### Task 6: Regenerate The Correct SFT Baseline And Rescore The Existing Hero

**Files:**
- Create: `configs/nano_roundtrip/publication/r33_sft_primary_generate.yaml`
- Create: `configs/nano_roundtrip/publication/r33_sft_primary_score.yaml`
- Create: `configs/nano_roundtrip/publication/r33_hero_primary_generate.yaml`
- Create: `configs/nano_roundtrip/publication/r33_hero_primary_score.yaml`
- Create: `configs/nano_roundtrip/publication/r33_existing_hero_corrected_queue.yaml`

- [x] **Step 1: Freeze the publication protocol before generation**

Use identical no-prefix greedy generation, tokenizer, stop text, maximum tokens, backend, family manifest, and row order for SFT and RL.

- [x] **Step 2: Dry-run and validate all resolved commands**

```bash
/Users/rigarg/.local/bin/visor run .venv/bin/python scripts/nano_roundtrip_queue.py \
  --queue configs/nano_roundtrip/publication/r33_existing_hero_corrected_queue.yaml \
  --dry-run
```

Expected: candidate and baseline protocol hashes are identical.

- [x] **Step 3: Generate and score SFT once**

Cache generated rows with row identity, family ID, protocol hash, and model hash. Never append rows generated under a different protocol.

- [ ] **Step 4: Generate and score the retained hero**

Run primary-critic scoring for the same rows and report family-clustered directional and raw metrics.

- [ ] **Step 5: Apply the corrected gate**

Do not promote if protocol parity, family independence, raw metric reporting, or baseline binding fails.

- [ ] **Step 6: Treat controls and the short gate correctly**

Controls are sanity checks, not evidence of RL improvement, because a working
SFT model also beats them. Label 64/64 as a smoke subset only; do not count it
as an independent second gate or include it in multiplicity-free evidence.

- [ ] **Step 7: Preserve reports and update the logbook**

Label this as a salvage evaluation of an exploratory checkpoint, not an independent replication.

### Task 7: Add Independent-Critic And Length-Controlled Evaluation

**Files:**
- Modify: `scripts/eval_nano_cross_critic_gate.py`
- Modify: `tests/test_nano_cross_critic_gate.py`
- Modify: `scripts/eval_nano_av_ar_roundtrip_gate.py`
- Create: `configs/nano_roundtrip/publication/r33_hero_independent_score.yaml`

- [x] **Step 1: Rescore cached SFT and hero text with the retained independent critic**

No AV generation is required. Require the same row and family identities as the primary reports.

- [ ] **Step 2: Train a genuinely independent critic if the retained reseed passes**

The retained critic changed data order but reused the same initialization and
recipe. For publication, train a second critic with fresh head/router
initialization, independent seed, immutable family-disjoint training manifest,
and no access to publication test families. Record both critics rather than
renaming the existing reseed as fully independent.

- [x] **Step 3: Strengthen the cross-critic gate**

Predeclare:

```yaml
min_primary_relative_improvement: 0.10
min_independent_relative_improvement: 0.05
min_independent_to_primary_gain_ratio: 0.75
require_family_clustered_ci_positive: true
```

- [x] **Step 4: Add per-row length controls**

Score RL explanations truncated to matched SFT token length and matched teacher token length. Also report gain per generated token and correlations between length delta and reconstruction delta.

- [x] **Step 5: Add parse-quality rules**

Repetition loops and fallback-only malformed outputs are not usable. Record repetition rate, factual-number density, close-tag rate, and true usable fraction.

- [x] **Step 6: Run tests**

```bash
/Users/rigarg/.local/bin/visor run .venv/bin/python -m pytest \
  tests/test_nano_cross_critic_gate.py \
  tests/test_nano_av_ar_roundtrip_gate.py -q
```

- [x] **Step 7: Run corrected cross-critic and length-control scoring**

The retained hero survives only if its family-clustered paired gain remains positive under the independent critic and at least one length-matched comparison.

### Task 8: Evaluate Checkpoint Dose Response And Decide Whether To Retrain

**Files:**
- Create: `configs/nano_roundtrip/publication/r33_checkpoint_dose_response_queue.yaml`
- Create: `docs/runs/r33_corrected_existing_hero_eval_202607.md`

- [x] **Step 1: Inventory retained update-16, update-228, and update-342 checkpoints**

Do not infer availability from docs; verify model shards and metadata.

- [ ] **Step 2: Evaluate every retained checkpoint under the corrected protocol**

Use validation only for checkpoint comparison. Do not open the new publication test split.

- [ ] **Step 3: Analyze learning saturation**

Compare directional gain, raw R2, independent-critic transfer, length-controlled gain, KL, entropy, and explanation length at each update.

- [ ] **Step 4: Apply the retraining decision**

```text
GO to confirmatory replication only if a retained checkpoint passes protocol,
family, independent-critic, and length-controlled validation gates.
NO-GO otherwise; repair SFT/data or RL objective before more scale.
```

- [ ] **Step 5: Commit corrected evaluation evidence and decision**

```bash
git add configs/nano_roundtrip/publication docs/runs docs/experiment_logbook.md docs/rl_logbook.md
git commit -m "docs: record corrected R33 hero evaluation"
```

### Task 9: Freeze Confirmatory HPO, Guards, And Test Isolation

**Files:**
- Create: `configs/nano_rl/publication/r33_confirmatory_replication.yaml`
- Create: `docs/runs/r33_publication_preregistration.md`
- Modify: `scripts/nano_rl_queue.py`
- Modify: `tests/test_nano_rl_queue.py`
- Modify: `scripts/nano_ar_hpo_study.py`
- Create: `scripts/nano_av_hpo_study.py`
- Create: `tests/test_nano_av_hpo_study.py`

- [x] **Step 1: Remove test metrics from every HPO objective**

Select AR, AV, RL hyperparameters and checkpoints using validation only. Add tests that fail if `test` contributes to an objective.

- [ ] **Step 2: Predeclare one immutable guard policy**

Specify KL, length, gradient, truncation, and parse guards before launch. A guard failure ends the confirmatory run. Any changed policy is a new registered experiment, not a resume of the same confirmatory run.

Progress (`2026-07-10`): the draft policy is written and the launcher now
hash-binds it through `nano_rl_publication_contract.v1`. This remains unchecked
until the validation-only stability grid selects a recipe and the draft status
changes to `registered` without editing thresholds in response to confirmatory
results.

- [ ] **Step 3: Calibrate stability on validation-only bounded probes**

Before the confirmatory run, run a finite, committed grid over KL coefficient,
KL estimator/target, and optional length regularization. Use training and
validation telemetry only. Select a setting that prevents recurrent extreme
KL/gradient excursions rather than merely raising abort thresholds. Record all
probes and freeze the selected recipe.

Progress: the four-item, 16-update grid and deterministic tie-break are frozen
in the draft preregistration. No clean probe has launched.

- [ ] **Step 4: Write and commit the preregistration**

State hypotheses, primary/secondary endpoints, family-level sample size,
power target of at least `0.80` for a predeclared 10% relative directional
gain, seeds, checkpoint-selection rule, guard/stopping policy,
multiplicity handling, and the exact one-shot test command before launch.

Progress: `docs/runs/r33_publication_preregistration.md` is committed as a
draft with test sealed. Final registration still requires clean AV validation,
the protocol-matched SFT round trip, independent critic/AR evidence, the
stability-grid decision, and the power calculation.

- [ ] **Step 5: Predeclare checkpoint saves**

Save model-only checkpoints at updates 16, 64, 110, 228, and 342 so dose response is measurable.

Progress: the exact schedule is declared in the draft preregistration and
implemented through `training.save_iterations` / `NLA_SAVE_ITERATIONS` with
strict ordering and final-update validation. It remains unchecked until the
final confirmatory queue is materialized and hash-bound to that schedule.

- [ ] **Step 6: Predeclare primary endpoint and multiplicity handling**

Primary endpoint: family-clustered paired directional reconstruction gain over protocol-matched SFT under the independent critic. Raw R2 and length-matched gain are mandatory secondary endpoints. Test is evaluated once after validation selection is locked. Apply Holm correction across confirmatory secondary hypotheses and show unadjusted effect sizes and intervals as descriptive results.

Progress: these rules are present in the draft preregistration and launcher
contract. Final lock is pending the same upstream clean-SFT and stability gates.

- [ ] **Step 7: Require finalized role-aware W&B logs**

Actor, rollout, critic, and system runs must log reward distribution, KL
quantiles, entropy, policy-gradient and KL-loss components, actor update norm,
gradient norms, the actual local/global clipping implementation selected at
runtime, response lengths, truncation/repetition, throughput, and checkpoint
eval pointers.

Progress: role-aware offline W&B and system telemetry exist in the RL runtime;
the final confirmatory config and evidence-field completeness check remain
pending.

- [x] **Step 8: Run tests and commit**

```bash
/Users/rigarg/.local/bin/visor run .venv/bin/python -m pytest \
  tests/test_nano_rl_queue.py \
  tests/test_nano_ar_hpo_study.py \
  tests/test_nano_av_hpo_study.py -q
git add configs/nano_rl/publication scripts tests
git commit -m "feat: freeze publication RL contract"
```

Completed in commit `0c95712`; focused RunAI verification passed `98` tests.
This verifies the contract machinery, not readiness to launch confirmatory RL.

### Task 10: Retrain Clean AR/AV SFT After Holdout Or Activation-Fidelity Failure

**Files:**
- Create: `configs/nano_ar/publication/r33_family_clean_sft.yaml`
- Create: `configs/nano_ar/publication/r33_family_clean_sft_queue.yaml`
- Create: `configs/nano_av/publication/r33_family_clean_sft.yaml`
- Create: `configs/nano_av/publication/r33_family_clean_sft_queue.yaml`
- Update: `docs/runs/r33_ar_hpo_202606.md`
- Update: `docs/runs/r33_av_hpo_202606.md`

- [x] **Step 1: Apply the Task 4 holdout and Task 5 activation-fidelity triggers**

If no sufficiently large never-trained family holdout exists, rebuild shared family-disjoint AR/AV splits and retrain both SFT components. Do not reuse the contaminated test for selection.

Task 5 overrides the initially favorable holdout decision. Clean deterministic
extraction is complete and exactly replicated; clean AR/AV retraining remains
mandatory even though enough disjoint evaluation families exist.

- [x] **Step 2: Reuse val-only hyperparameters**

Start from the prior val-only winners, then run only minimal confirmation probes. Do not repeat broad test-aware HPO.

The predeclared clean configs use the prior validation-selected AR
`5e-5/warmup25/gb192` family and AV
`1e-4/warmup25/gb192/mb2/dynamic-4096` recipe. The clean AR run used
`mb48` on four H100-NVL GPUs after the padded-path equivalence proof; AV used
eight H100-NVL GPUs only after the packed-vs-padded `dyn4096` gate passed. AR,
AV, and the separately
initialized transfer critic consume the same frozen content-family split
manifest and expose validation only during checkpoint selection.

The confirmatory split is distinct from the exploratory salvage manifest.
Exact-prefix refinement merged residual cross-family duplicates into `5,009`
families. Train/validation/test contain `247,865 / 13,766 / 13,765` rows and
`4,504 / 250 / 255` families. Its manifest hash is
`479cbab5d21cd031cb72a770eebb3428e0d5419ebf8cce38c2ca6025e49741b6`;
clean AR, independent AR, and AV configs bind to that path.

- [x] **Step 3: Require verifier and protocol reports**

Rows, hidden dimension, finite activations, empty explanations, family overlap, dataset hashes, and source fingerprints must all pass before RL.

Both deterministic AR/AV verifier reports pass on all `275,396` rows with
`d_model=2688`, zero nonfinite activations, zero empty or malformed text, exact
manifest coverage, and zero document/family/content overlap. The immutable
runtime and full-replication reports bind the source, model, execution profile,
shard plan, and exact merged dataset hash.

Queue-readiness result (`2026-07-09`): the seed-`314159` independent critic
initialization passes its explicit primary-versus-independent verifier. Primary
AR, independent AR, and AV queue dry-runs all exited `0` from source commit
`294cc1e`; RunAI passed `65/65` focused tests. Since that readiness snapshot,
primary clean AR completed and passed its validation gate, and clean AV
completed all optimizer updates after its live equivalence gate. Its bounded
validation eval remains unreported. The independent critic init was removed
only after reproducibility evidence was frozen; source `3676b93` now stages a
manifest-hashed rebuild, while independent AR correctly remains
`blocked_missing_critic_init`. This does not satisfy independent critic
training or the protocol-matched SFT baseline step below.

- [ ] **Step 4: Establish protocol-matched SFT round-trip baseline**

The new baseline becomes immutable before confirmatory RL starts.

### Task 11: Run One Independent Confirmatory RL Replication

**Files:**
- Use: `configs/nano_rl/publication/r33_confirmatory_replication.yaml`
- Create: `docs/runs/r33_confirmatory_replication.md`

- [ ] **Step 1: Select a fresh seed and register it before launch**

Record seed, source/data/config hashes, topology, guard policy, endpoints, and stopping rule.

- [ ] **Step 2: Launch through the queue without script edits**

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.6/runai workspace exec train \
  -p trustworthy-ai-inference -- bash -lc '
cd /workspace/interp/code/nano30b-nla-pilot-current
WANDB_MODE=offline /workspace/interp/.venv/bin/python scripts/nano_rl_queue.py \
  --queue configs/nano_rl/publication/r33_confirmatory_replication.yaml --watch
'
```

- [ ] **Step 3: Select checkpoint using validation only**

Evaluate saved checkpoints on the publication validation families. Lock the selected checkpoint and analysis before opening test.

- [ ] **Step 4: Run one-shot test evaluation**

Run primary critic, independent critic, raw/directional metrics, family bootstrap, length controls, parse quality, and factuality checks once.

- [ ] **Step 5: Require replication consistency**

The independent seed must show the same direction with a positive family-clustered confidence interval. Report both seeds; do not average away a failed replication.

### Task 12: Build The Publication Evidence Package

**Files:**
- Create: `scripts/analyze_r33_publication_result.py`
- Create: `tests/test_analyze_r33_publication_result.py`
- Create: `docs/runs/r33_publication_result.md`
- Modify: `README.md`
- Modify: `docs/current_state.md`
- Modify: `runs/registry/experiments.yaml`

- [ ] **Step 1: Recompute all headline numbers from rowwise files**

Use `scripts/analyze_r33_publication_result.py` to read generated JSONL and
reports, compute every table, and emit checksums. Unit tests must reconstruct a
known synthetic result before the script is used on hero evidence. No number
may be copied manually from chat or logs.

- [ ] **Step 2: Add factual-quality evaluation**

Report names/dates/numbers hallucination rates and review at least 200
family-stratified rows for SFT, RL, and teacher explanations with two blinded
raters, a fixed rubric, inter-rater agreement, and adjudication of conflicts.

- [ ] **Step 3: Add R27 only if claiming R33 superiority**

Use the same family holdout, generation protocol, critics appropriate to each layer, and metric suite. Otherwise state that the paper makes no R27 superiority claim.

- [ ] **Step 4: Publish complete claim boundaries**

Include directional versus raw reconstruction, critic dependence, effective family count, all seeds, all stopped runs, HPO/test separation, length controls, and provenance hashes.

- [ ] **Step 5: Archive lightweight evidence and selected model checkpoints**

Keep logs, reports, generated rows, W&B payloads, source/data manifests, and selected model-only checkpoints. Exclude optimizer shards unless needed for continuation.

- [ ] **Step 6: Run final verification**

```bash
/Users/rigarg/.local/bin/visor run .venv/bin/python -m pytest tests -q
/Users/rigarg/.local/bin/visor run python scripts/verify_docs_consistency.py
/Users/rigarg/.local/bin/visor run git status --short
```

Expected: all tests pass, documentation contains no stale headline, and only intended files are changed.

## Final Publication Gate

Publication requires all of the following:

- Candidate and SFT generation protocol hashes match exactly.
- AR, AV, RL train, validation, and test content families have zero overlap.
- The test set was not used for HPO, checkpoint selection, guard changes, or debugging.
- Primary directional gain has a positive family-clustered interval on validation and one-shot test.
- Independent-critic gain is positive with transfer ratio at least `0.75`.
- At least one length-matched comparison has positive family-clustered gain.
- Raw MSE, centered R2, cosine, vector norms, and mean-predictor results are all disclosed.
- A second RL seed reproduces the direction of effect.
- Full Miles, patches, project source, data, environment, configs, and checkpoint lineage are hash-pinned.
- Failed runs and adaptive decisions remain visible in the registry.
- Any R27/Qwen superiority statement is supported by a protocol-matched fresh comparison.

If raw centered R2 remains non-positive, publication must explicitly claim directional representation recovery rather than activation reconstruction.

## Audit Coverage Matrix

| Audit finding | Remediation tasks |
|---|---|
| Corrupted mixed-protocol SFT baseline | Tasks 1, 2, 6 |
| Same-critic hero headline | Task 7 |
| Direction-only metric mislabeled as NMSE | Tasks 3, 12 |
| Adaptive guard relaxation and post-hoc promotion | Tasks 1, 9, 11 |
| Near-duplicate families, leakage, anti-conservative CI | Tasks 4, 10 |
| Length/channel-capacity confound | Task 7 |
| Flat late training and unknown best checkpoint | Task 8 |
| KL/gradient instability and ineffective anchoring | Tasks 8, 9 |
| Non-discriminative controls and nested 64/512 evidence | Task 6 |
| Unpinned Miles, patches, configs, data, and activation drift | Task 5 |
| Test-aware HPO | Tasks 9, 10, 11 |
| Silent component-filter fallback and permissive parse usability | Tasks 5, 7 |
| Winner-only registry and incomplete failure history | Tasks 1, 12 |
| Explanation hallucination and missing task-level quality evidence | Task 12 |
| Missing fresh R27 comparison | Task 12, only for an R33 superiority claim |
