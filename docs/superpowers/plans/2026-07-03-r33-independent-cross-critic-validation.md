# R33 Independent Cross-Critic Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train an independently ordered R33 AR critic and require statistically positive round-trip improvement under both the RL reward critic and that independent critic before a hero run can launch.

**Architecture:** Add one config-level training shuffle seed to the existing Miles SFT runner, then train a model-only independent critic on the unchanged component split. Reuse candidate and SFT generated text, score both with the independent critic, and combine primary plus independent reports in a focused fail-closed cross-critic gate. Existing Stage-2 invariance, functional, qualitative, and validity tools remain separate and are fed by a corrected analysis queue.

**Tech Stack:** Python 3.12, PyTorch/Miles FSDP2, PyArrow, YAML queues, pytest, offline W&B, RunAI 8xH100.

---

### Task 1: Configurable SFT Shuffle Seed

**Files:**
- Modify: `scripts/nano_av_runner.py`
- Modify: `tests/test_nano_av_runner_spec.py`

- [ ] **Step 1: Write the failing renderer test**

Add `training.rollout_seed: 314159` to the schedule-rendering fixture and assert:

```python
self.assertEqual(command[command.index("--rollout-seed") + 1], "314159")
```

- [ ] **Step 2: Verify RED**

Run:

```bash
/Users/rigarg/miniconda3/bin/python3 -m pytest tests/test_nano_av_runner_spec.py::NanoAVRunnerSpecTests::test_training_schedule_knobs_render_miles_lr_flags -q
```

Expected: fail because `--rollout-seed` is absent.

- [ ] **Step 3: Render the seed through the existing optional training flags**

Add `("rollout_seed", "--rollout-seed")` to `optional_training_flags` in
`render_miles_command`. Do not change `dataset.seed`; split membership must
remain identical to the selected critic.

- [ ] **Step 4: Verify GREEN**

Run the targeted test and `tests/test_nano_av_runner_spec.py -q`.

### Task 2: Independent R33 AR Config and Queue

**Files:**
- Create: `configs/nano_ar/hpo/r33_component_full_independent_critic_seed314159.yaml`
- Create: `configs/nano_ar/hpo/r33_component_full_independent_critic_queue.yaml`
- Modify: `tests/test_nano_ar_hpo_queue.py`

- [ ] **Step 1: Write a failing config contract test**

Load the independent config and assert: component-full verified source,
`dataset.seed == 42`, `training.rollout_seed == 314159`, `lr == 5e-5`,
`gb192/mb96`, one epoch, 512/512 eval, `no_save_optim == true`, and one final
checkpoint.

- [ ] **Step 2: Verify RED**

Run the new test; expect missing config failure.

- [ ] **Step 3: Add the config and one-item queue**

Use a distinct output root under
`miles-fsdp2-ar-sft-r33-independent-critic`. Preserve the exact seed-42
content-component split and selected critic initialization. Set
`rollout_seed: 314159`, final-only save interval 1289, model-only checkpoint,
offline W&B, and all teacher/control evals.

- [ ] **Step 4: Verify GREEN and dry-render the command**

Run the targeted queue/config test and use `nano_ar_hpo_queue.py --status` plus
the runner's dry-run mode. Confirm `--rollout-seed 314159` appears exactly
once.

### Task 3: Cross-Critic Promotion Gate

**Files:**
- Create: `scripts/eval_nano_cross_critic_gate.py`
- Create: `tests/test_nano_cross_critic_gate.py`

- [ ] **Step 1: Write failing pure-function tests**

Cover pass, insufficient independent gain, non-positive clustered CI, row-win
failure, primary/independent gain-ratio failure, row/hash mismatch, and missing
split. Use tiny report dictionaries; no model mocks.

- [ ] **Step 2: Verify RED**

Run `tests/test_nano_cross_critic_gate.py -q`; expect import failure.

- [ ] **Step 3: Implement a focused report combiner**

The CLI accepts primary candidate, independent candidate, primary SFT, and
independent SFT reports. It verifies matching dataset hashes and row keys,
reads each candidate report's paired baseline summary, and writes:

```json
{
  "schema_version": "nano_cross_critic_gate.v1",
  "passed": true,
  "splits": {},
  "thresholds": {},
  "sources": {}
}
```

Defaults: independent relative improvement `0.05`, row wins strictly above
`0.50`, clustered CI lower bound above zero, and independent/primary relative
gain ratio at least `0.25` on validation and test. Any missing/nonfinite value
fails closed.

- [ ] **Step 4: Verify GREEN**

Run the focused tests and the canonical test suite.

### Task 4: Corrected Stage-2 Analysis Queue

**Files:**
- Create: `configs/nano_rl/r33_component_corrected_stage2_analysis_queue_8h100.yaml`
- Create: `configs/nano_rl/r33_component_corrected_validity_eval.yaml`
- Create: `tests/test_r33_corrected_stage2_analysis_queue.py`

- [ ] **Step 1: Write the failing queue contract test**

Require sequential items for provenance enrichment, independent candidate/SFT
scoring with `--reuse-generated`, invariance, functional recovery, qualitative
panel creation, response-closure audit, composite validity, and cross-critic
gate. Assert no training command exists.

- [ ] **Step 2: Verify RED**

Run the focused test; expect missing config failure.

- [ ] **Step 3: Add configs using the fixed confirmation paths**

Use update-32's trusted 512/512 report and generated JSONL from
`r33_corrected_k3_confirm_lr1e5_update32_runtimefix_retry1`, the hardened SFT report/text, and
the independent critic final checkpoint. Write the final reports to:

```text
validity/r33-corrected-stage2-gate.json
validity/r33-corrected-cross-critic-gate.json
```

Generate the qualitative panel before its reviewed decision item. The queue
must stop rather than synthesize human decisions when the review file is
absent.

- [ ] **Step 4: Verify GREEN and preflight all existing input paths**

Run the focused and canonical suites. On RunAI, validate every path before
starting the analysis queue.

### Task 5: RunAI Execution and Promotion

- [ ] Complete the guarded 32-update confirmation and parse every staged gate.
- [ ] If any confirmation gate fails, preserve evidence, document the failure,
  and stop before independent critic or hero compute.
- [ ] Sync the implementation through S3 and rerun the RunAI test suite.
- [ ] Train/evaluate the independent critic and require acceptable 512/512
  teacher/control quality.
- [ ] Run corrected Stage-2 analyses, review both 50-row qualitative panels,
  and emit the two required `passed: true` reports.
- [ ] Regenerate a 342-update hero queue from the passing confirmation recipe,
  preflight storage and source provenance, then launch it only when both
  required reports pass.
- [ ] Preserve the selected model-only checkpoint and all reports/W&B logs in
  S3, remove losing checkpoint payloads, and update the experiment and RL
  logbooks.
