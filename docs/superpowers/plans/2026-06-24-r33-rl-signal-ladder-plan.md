# R33 RL Signal Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make R33 RL experiments queueable with larger rollout/effective batches, explicit reward/advantage/KL controls, and automatic round-trip delta evaluation against the SFT baseline.

**Architecture:** Keep `scripts/nano_rl_queue.py` as the queue orchestrator and `external/natural_language_autoencoders/configs/rl.sh` as the Miles launcher boundary. Training knobs become YAML-driven environment variables; optional post-run evaluation converts the saved actor DCP checkpoint to a temporary HF directory, runs `scripts/eval_nano_av_ar_roundtrip_gate.py` with `--baseline-report-json`, records gate/report paths in the queue item, and removes temporary HF output.

**Tech Stack:** Python queue runner, YAML configs, Miles RL shell launcher, FSDP DCP-to-HF converter, round-trip eval script, pytest.

---

### Task 1: Configurable RL Signal Knobs

**Files:**
- Modify: `scripts/nano_rl_queue.py`
- Modify: `external/natural_language_autoencoders/configs/rl.sh`
- Test: `tests/test_nano_rl_queue.py`

- [ ] **Step 1: Add queue tests**

Add a test that a YAML `training` block with `advantage_estimator`, `normalize_advantages`, `rewards_normalization`, and `grpo_std_normalization` maps to launcher env vars.

- [ ] **Step 2: Add launcher tests**

Add a shell-capture test proving `ADVANTAGE_ESTIMATOR=grpo`, `NORMALIZE_ADVANTAGES=1`, `REWARDS_NORMALIZATION=0`, `GRPO_STD_NORMALIZATION=0`, and nonzero `KL_LOSS_COEF` produce `--advantage-estimator grpo`, `--normalize-advantages`, `--disable-rewards-normalization`, `--disable-grpo-std-normalization`, and `--use-kl-loss --kl-loss-coef`.

- [ ] **Step 3: Implement queue env mapping**

Set `ADVANTAGE_ESTIMATOR`, `NORMALIZE_ADVANTAGES`, `REWARDS_NORMALIZATION`, and `GRPO_STD_NORMALIZATION` in `build_run_spec()`.

- [ ] **Step 4: Implement launcher flags**

Replace the hard-coded `--advantage-estimator grpo` with `$ADVANTAGE_ESTIMATOR` and add optional arrays for `--normalize-advantages`, `--disable-rewards-normalization`, and `--disable-grpo-std-normalization`.

- [ ] **Step 5: Run focused tests**

Run:

```bash
/Users/rigarg/.local/bin/visor run python3 -m pytest tests/test_nano_rl_queue.py -q
```

Expected: all tests in `tests/test_nano_rl_queue.py` pass.

### Task 2: Post-Run Round-Trip Gate Hook

**Files:**
- Modify: `scripts/nano_rl_queue.py`
- Test: `tests/test_nano_rl_queue.py`

- [ ] **Step 1: Add post-eval spec builder tests**

Add a test that an item-level `post_eval.roundtrip` block creates a converter command, an eval command with `--baseline-report-json`, and expected report/generated/temp-HF paths for `iter_0000008`.

- [ ] **Step 2: Add post-eval runner**

Add a helper that derives the checkpoint iteration from `post_eval.roundtrip.iteration` or `rollout.num_rollout`, converts `run_dir/actor/iter_<N>` to a temporary HF directory, runs the round-trip gate, parses `gate.passed`, updates the queue item with report/generated paths and `gate_passed`, and deletes temp HF when `cleanup_hf: true`.

- [ ] **Step 3: Preserve failure evidence**

If conversion or eval fails, mark the item failed with the failing command stage and leave run logs/checkpoints intact.

- [ ] **Step 4: Run focused tests**

Run:

```bash
/Users/rigarg/.local/bin/visor run python3 -m pytest tests/test_nano_rl_queue.py -q
```

Expected: all tests in `tests/test_nano_rl_queue.py` pass.

### Task 3: R33 RL Signal Ladder Queue

**Files:**
- Create: `configs/nano_rl/r33_component_full_signal_ladder_queue_4h200_len512.yaml`
- Modify: `docs/rl_logbook.md`

- [ ] **Step 1: Create queue config**

Create a 4-H200 queue with the same SFT actor, actor load/ref roots, AR critic, SGLang service, W&B offline, and system metrics as Medium-A.

- [ ] **Step 2: Add four items**

Queue:

1. `rb4_n4_kl1e4`: larger effective group, KL anchor.
2. `rb4_n4_kl3e4`: same with stronger KL.
3. `rb4_n4_no_std_kl1e4`: disable GRPO std normalization, keep group centering and KL.
4. `rb4_n4_raw_reward_kl1e4`: disable reward normalization, keep KL.

Each item saves only its final checkpoint and runs 64/64 round-trip eval against the SFT baseline report.

- [ ] **Step 3: Document intent**

Add a logbook section explaining that this ladder tests batch/rollout scale, KL anchoring, and reward normalization before any RL hero.

- [ ] **Step 4: Dry-run queue**

Run:

```bash
/Users/rigarg/.local/bin/visor run python3 scripts/nano_rl_queue.py configs/nano_rl/r33_component_full_signal_ladder_queue_4h200_len512.yaml --dry-run --once
```

Expected: dry-run prints the first item command and does not launch training.
