# R33 RL Medium Scale-Up Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make offline RL observability role-safe and queue the validated R33 16-update confirmation followed by a gate-dependent 32-update medium run.

**Architecture:** Miles currently starts W&B once in the driver and once each in rollout, actor, and critic Ray processes. Online shared mode supports that pattern, but offline mode creates multiple files with one run ID. Patch the Miles tracking boundary to label secondary processes by role and derive deterministic offline run IDs. Extend the YAML queue with a minimal `depends_on` condition so later experiments unlock only after an earlier item completes with a passing post-evaluation gate.

**Tech Stack:** Python 3.12, PyYAML queue runner, Bash RL launcher, Miles FSDP patch series, W&B offline, pytest, RunAI 8x H100.

---

### Task 1: Add Role-Safe Offline W&B Identity

**Files:**
- Create: `external/natural_language_autoencoders/nla/miles_patches/0016_wandb_offline_role_runs.patch`
- Modify: `tests/test_nano_miles_launcher.py`

- [ ] **Step 1: Write a failing patch-content test**

Assert that the new patch defines role-aware secondary tracking, uses `f"{wandb_run_id}-{role}"` for offline secondary IDs, and tags actor, critic, and rollout call sites.

- [ ] **Step 2: Run the focused test and confirm it fails because patch `0016` is absent**

```bash
python3 -m pytest tests/test_nano_miles_launcher.py::NanoMilesLauncherTests::test_miles_patch_gives_offline_secondary_runs_distinct_role_ids -q
```

- [ ] **Step 3: Add the minimal Miles patch**

Change `tracking_utils.init_tracking` to accept a `role` keyword, pass that role to `init_wandb_secondary`, call it as `role="rollout"` from `miles/ray/rollout.py`, and pass the existing FSDP actor `role` argument from `miles/backends/fsdp_utils/actor.py`.

In `wandb_utils`, keep online shared mode unchanged. For offline secondary writers, use a deterministic `id` and `name` suffixed with the role, retaining the configured W&B group. Set the configured primary `wandb_run_id` explicitly for the driver so the role identities share a stable root.

- [ ] **Step 4: Verify patch syntax and focused tests**

```bash
python3 scripts/check_miles_patches.py
python3 -m pytest tests/test_nano_miles_launcher.py -q
```

### Task 2: Add Gate-Dependent Queue Promotion

**Files:**
- Modify: `scripts/nano_rl_queue.py`
- Modify: `tests/test_nano_rl_queue.py`

- [ ] **Step 1: Write failing tests for a blocked dependent item**

Create a two-item queue fixture where the second item is `blocked` with:

```yaml
depends_on:
  item: confirmation
  require_gate_pass: true
```

Verify it remains blocked before the prerequisite completes, becomes pending after the prerequisite is `complete` with `gate_passed: true`, and remains blocked after a failed gate.

- [ ] **Step 2: Run the focused test and confirm it fails because promotion does not exist**

```bash
python3 -m pytest tests/test_nano_rl_queue.py::NanoRLQueueTests::test_promotes_blocked_dependency_only_after_passing_gate -q
```

- [ ] **Step 3: Implement minimal dependency validation and promotion**

Validate `depends_on.item` against known queue-item names. Before selecting the next pending item, promote eligible blocked items to pending, recording the prerequisite and promotion timestamp. Do not promote a failed or gate-failed prerequisite.

- [ ] **Step 4: Verify queue behavior**

```bash
python3 -m pytest tests/test_nano_rl_queue.py -q
```

### Task 3: Define The R33 Scale-Up Queue

**Files:**
- Create: `configs/nano_rl/r33_component_medium_scaleup_queue_8h100.yaml`
- Modify: `tests/test_nano_rl_queue.py`
- Modify: `docs/rl_logbook.md`

- [ ] **Step 1: Write a failing checked-in configuration test**

Require two items: a pending 16-update confirmation and a blocked 32-update dependent medium run. Require the measured T4 topology: `actor_gpus=6`, `critic_gpus=1`, `rollout_gpus=1`, `rollout_batch_size=30`, `n_samples_per_prompt=16`, `global_batch_size=480`, `actor_micro_batch=32`, `actor_lr=2e-5`, `kl_loss_coef=0.001`, and `normalize_advantages=false`.

- [ ] **Step 2: Run the focused test and confirm it fails because the scale-up queue is absent**

```bash
python3 -m pytest tests/test_nano_rl_queue.py::NanoRLQueueTests::test_checked_in_r33_medium_scaleup_queue_uses_selected_hpo_config -q
```

- [ ] **Step 3: Add the YAML queue**

The confirmation runs 16 updates / 7,680 generated samples, saves only the final actor checkpoint, and runs a `256/256` real-vs-control, matched-SFT round-trip gate. The medium run uses the same configuration for 32 updates / 15,360 samples, is locked behind the confirmation gate, and uses a `512/512` final gate. Both preserve W&B offline logs and clean temporary HF after evaluation.

- [ ] **Step 4: Document the scale criteria**

Record that promotion requires healthy parse/control results, two-split SFT improvement, a reward-to-gate correlation diagnostic, and no late policy-drift signature comparable to the rejected normalized-advantage candidate.

### Task 4: Sync, Verify, And Launch The Confirmation

**Files:**
- Modify: `docs/rl_logbook.md`

- [ ] **Step 1: Run local focused verification**

```bash
python3 -m pytest tests/test_nano_miles_launcher.py tests/test_nano_rl_queue.py tests/test_nla_rl_metrics.py tests/test_nla_system_metrics.py -q
python3 scripts/check_miles_patches.py
```

- [ ] **Step 2: Sync the source/config superset to RunAI through S3**

Upload a source-only bundle, retrieve it in `/workspace/interp/code/nano30b-nla-pilot-current`, and apply the new Miles patch to `/workspace/interp/code/miles-051cd15` after checking it applies cleanly.

- [ ] **Step 3: Run the same focused tests in the RunAI venv and dry-run the queue**

Confirm the 16-update item renders as the next pending item and the 32-update item remains blocked.

- [ ] **Step 4: Launch the queue watcher**

Use `--run-until-empty`; it will execute the 16-update confirmation plus its 256/256 post-eval, then automatically unlock and execute the 32-update run only if `gate_passed` is true.

## Self-Review

- The W&B patch solves offline run-ID collisions without changing online shared semantics.
- The dependency feature is generic and does not encode R33-specific names in queue code.
- The 32-update experiment cannot start from a failed or inconclusive 16-update confirmation.
- The queue uses the measured `2e-5 / 1e-3 / mb32 / 6+1+1` configuration rather than the obsolete `5e-6 / mb1` tier template.
