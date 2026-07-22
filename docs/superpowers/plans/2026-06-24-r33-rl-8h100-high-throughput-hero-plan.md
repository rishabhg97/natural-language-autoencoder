# R33 RL 8x H100 High-Throughput Hero Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move R33 Nano30B NLA RL from fragile smoke attempts to a documented, high-throughput 8x H100 hero run with validated round-trip improvements over the clean SFT baseline.

**Architecture:** Use the `train` RunAI workspace on one 8x H100 NVL node with a config-driven topology: actor FSDP on GPUs 0-3, TP2 external SGLang rollout on GPUs 4-5, and AR critic/reward on GPUs 6-7. Progress through a fit canary, a throughput-calibrated medium run, then the hero run, with W&B offline/system metrics and logbook updates at every milestone.

**Tech Stack:** RunAI, Visor, S3 sync helper, Python queue runner, YAML queue configs, Miles RL launcher, external SGLang, Ray, PyTorch FSDP, W&B offline, round-trip eval gate, pytest.

---

## Current 8x H100 State

- Local repo root: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot`
- Remote code root: `/workspace/interp/code/nano30b-nla-pilot-current`
- RunAI workspace: `train`
- RunAI project: `trustworthy-ai-inference`
- Node: `4u8g-gen-0176.ipp3a2.colossus.nvidia.com`
- GPU topology: `8x NVIDIA H100 NVL`, about `95.8 GiB` each
- Intended RL placement:
  - GPUs `0-3`: actor FSDP
  - GPUs `4-5`: one TP2 external SGLang rollout server
  - GPUs `6-7`: critic/reward AR model
- Latest blocker:
  - The H100 canary launched SGLang with `--tp-size 2`.
  - Miles expected `tp_size=1` because `--rollout-num-gpus-per-engine 2` was not passed to the training command.
  - Failure line: `AssertionError: name='tp_size' expect_value=1 actual_value=2`.

## Documentation Contract

Every milestone must update at least one of:

- `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/rl_logbook.md`
- `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/current_state.md`
- `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/experiment_logbook.md`

Each log entry must include:

- Absolute local and remote config paths.
- RunAI topology and GPU count.
- Queue item name and run directory.
- Exact command or queue invocation.
- Status: `planned`, `running`, `failed`, `passed`, `promoted`, or `rejected`.
- Evidence: failure traceback, GPU peak/current memory, rollout throughput, reward stats, KL, parse/close rate, round-trip NMSE, and checkpoint path when available.
- Cleanup actions: checkpoint retention/deletion and temp HF/SGLang staging cleanup.

---

### Task 1: Make the H100 TP2 Topology Explicit

**Files:**
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/configs/nano_rl/r33_component_full_signal_ladder_queue_8h100_len512.yaml`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/tests/test_nano_rl_queue.py`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/rl_logbook.md`

- [x] **Step 1: Add or update the queue test for TP2 propagation**

Add a focused test proving that the checked-in H100 queue renders the Miles command with a single TP2 rollout endpoint and the matching Miles topology flag:

```python
def test_checked_in_r33_8h100_ladder_uses_tp2_rollout_engine(self):
    queue_path = REPO_ROOT / "configs/nano_rl/r33_component_full_signal_ladder_queue_8h100_len512.yaml"
    doc = nano_rl_queue.load_queue(queue_path)
    spec = nano_rl_queue.build_run_spec(doc, doc["items"][0], queue_path=queue_path)

    command = spec["command"]
    self.assertEqual(spec["resource_total_gpus"], 8)
    self.assertIn("--rollout-external-engine-addrs", command)
    addr_index = command.index("--rollout-external-engine-addrs")
    self.assertEqual(command[addr_index + 1], "127.0.0.1:31000")
    self.assertNotEqual(command[addr_index + 2] if len(command) > addr_index + 2 else None, "127.0.0.1:31000")
    self.assertIn("--rollout-num-gpus-per-engine", command)
    self.assertEqual(command[command.index("--rollout-num-gpus-per-engine") + 1], "2")

    service = spec["sglang_service"]
    self.assertEqual(service["engine_addrs"], ["127.0.0.1:31000"])
    start = service["start_commands"][0]
    self.assertIn("--tp-size", start)
    self.assertEqual(start[start.index("--tp-size") + 1], "2")
    self.assertIn("--base-gpu-id", start)
    self.assertEqual(start[start.index("--base-gpu-id") + 1], "4")
```

- [x] **Step 2: Run the failing test before changing the YAML**

Run:

```bash
/Users/rigarg/.local/bin/visor run python3 -m pytest tests/test_nano_rl_queue.py::NanoRLQueueTests::test_checked_in_r33_8h100_ladder_uses_tp2_rollout_engine -q
```

Expected before the YAML fix: FAIL because the queue lacks `--rollout-num-gpus-per-engine 2` and/or has duplicate `127.0.0.1:31000` addresses.

- [x] **Step 3: Fix the H100 YAML topology**

Set the H100 queue defaults to one TP2 endpoint and pass the matching Miles argument:

```yaml
defaults:
  resources:
    actor_nodes: 1
    actor_gpus: 4
    critic_nodes: 1
    critic_gpus: 2
    rollout_gpus: 2
    min_actor_gpus: 4
  sglang:
    mode: external
    managed: true
    engine_addrs:
      - 127.0.0.1:31000
    start_commands:
      - - /workspace/interp/.venvs/sglang-cu130/bin/python
        - -m
        - sglang.launch_server
        - --model-path
        - /workspace/interp/outputs/nano30b-nla-pilot/rl_smoke/r33_component_full_sft_init_512row_3h200/actor_sft_hf_iter_0001291
        - --host
        - 127.0.0.1
        - --port
        - "31000"
        - --tp-size
        - "2"
        - --base-gpu-id
        - "4"
        - --trust-remote-code
        - --skip-server-warmup
        - --disable-radix-cache
        - --disable-custom-all-reduce
        - --attention-backend
        - fa3
        - --schedule-conservativeness
        - "1.5"
        - --enable-draft-weights-cpu-backup
  extra_args:
    - --gradient-checkpointing
    - --nla-skip-grad-norm
    - --no-save-optim
    - --rollout-num-gpus-per-engine
    - "2"
    - --sglang-attention-backend
    - fa3
```

- [x] **Step 4: Reset failed canary state for a clean retry**

In the same YAML, rename the first item so it is not confused with failed history:

```yaml
items:
  - name: r33-component-full-rl-8h100-fit-rb4-n4-kl3e4-tp2fix
    status: pending
    training:
      actor_lr: 3e-6
      save_interval: 2
    wandb:
      run_id: r33-component-full-rl-8h100-fit-rb4-n4-kl3e4-tp2fix-20260624
    run_dir: /workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/r33_component_full_rl_8h100_fit_rb4_n4_kl3e4_tp2fix
```

Keep the existing medium item pending, but update its notes to say it is gated on the `tp2fix` canary.

- [x] **Step 5: Verify focused tests pass**

Run:

```bash
/Users/rigarg/.local/bin/visor run python3 -m pytest tests/test_nano_rl_queue.py::NanoRLQueueTests::test_checked_in_r33_8h100_ladder_uses_tp2_rollout_engine -q
```

Expected: PASS.

- [x] **Step 6: Document Milestone M1**

Append this section to `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/rl_logbook.md`:

```markdown
## 2026-06-24 - M1 8x H100 TP2 Topology Fix

- Status: use exactly one of `planned`, `passed`, or `failed`
- Workspace: train, project trustworthy-ai-inference
- Topology: 8x H100 NVL on 4u8g-gen-0176
- Config: /workspace/interp/code/nano30b-nla-pilot-current/configs/nano_rl/r33_component_full_signal_ladder_queue_8h100_len512.yaml
- Previous blocker: external SGLang was TP2 but Miles expected TP1, failing with `AssertionError: name='tp_size' expect_value=1 actual_value=2`.
- Fix: one external endpoint `127.0.0.1:31000`, SGLang `--tp-size 2`, Miles `--rollout-num-gpus-per-engine 2`.
- Verification: pytest command and result.
```

### Task 2: Sync H100 Code and Clean the Remote Runtime State

**Files:**
- Read: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/scripts/nano_s3.py`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/rl_logbook.md`

- [x] **Step 1: Run local preflight tests**

Run:

```bash
/Users/rigarg/.local/bin/visor run python3 -m pytest tests/test_nano_rl_queue.py tests/test_nla_system_metrics.py tests/test_nla_rl_metrics.py -q
```

Expected: PASS. If unrelated pre-existing test failures appear, log the failing test names and stop before syncing.

- [x] **Step 2: Sync the local superset to S3**

Use the existing S3 helper path already used for Mac-to-RunAI sync. The command form must preserve code/config/tests/docs and exclude heavy artifacts, checkpoints, `.cache`, and W&B payloads:

```bash
/Users/rigarg/.local/bin/visor run python3 scripts/nano_s3.py cp-up \
  --source /Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot \
  --dest s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/code-sync/nano30b-nla-pilot-current-h100-rl-tp2fix.tar.zst \
  --exclude-heavy
```

If this helper invocation differs from the current helper CLI, run:

```bash
/Users/rigarg/.local/bin/visor run python3 scripts/nano_s3.py --help
```

and document the exact working replacement in `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/rl_logbook.md`.

- [x] **Step 3: Pull the bundle on RunAI**

Run on `train`:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
set -euo pipefail
cd /workspace/interp/code/nano30b-nla-pilot-current
AWS_SHARED_CREDENTIALS_FILE=/workspace/interp/secrets/aws/credentials \
AWS_CONFIG_FILE=/workspace/interp/secrets/aws/config \
/workspace/interp/.venv/bin/aws s3api get-object \
  --bucket team-ipp-trustworthy-ai \
  --key nano30b-nla-pilot/code-sync/nano30b-nla-pilot-current-h100-rl-tp2fix.tar.zst \
  /workspace/interp/tmp/nano30b-nla-pilot-current-h100-rl-tp2fix.tar.zst
'
```

Expected: S3 object downloads without printing credentials.

- [x] **Step 4: Clean stale Ray/SGLang processes only if no RL job is active**

Run:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
set -euo pipefail
ps -eo pid,etime,stat,cmd | grep -E "nano_rl_queue|sglang.launch_server|train_actor|ray::|raylet|gcs_server" | grep -v grep || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
'
```

If only defunct Ray actors remain and all GPUs show idle, run:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
ray stop --force || true
pkill -9 -f "sglang.launch_server" || true
pkill -9 -f "nano_rl_queue.py" || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
'
```

Expected: no active training/rollout process and GPUs return to near-empty memory.

- [x] **Step 5: Document Milestone M2**

Append a logbook section with:

```markdown
## 2026-06-24 - M2 8x H100 Sync and Runtime Cleanup

- Status: use exactly one of `passed`, `failed`, or `blocked`
- S3 object:
- Remote code root:
- Remote pytest result:
- Processes before cleanup:
- Processes after cleanup:
- GPU memory after cleanup:
- Disk:
```

### Task 3: Run the TP2 Fit Canary

**Files:**
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/rl_logbook.md`
- Read: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/configs/nano_rl/r33_component_full_signal_ladder_queue_8h100_len512.yaml`

- [x] **Step 1: Dry-run the H100 queue on RunAI**

Run:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
cd /workspace/interp/code/nano30b-nla-pilot-current
/workspace/interp/.venv/bin/python scripts/nano_rl_queue.py configs/nano_rl/r33_component_full_signal_ladder_queue_8h100_len512.yaml --dry-run --once
'
```

Expected command contains:

```text
--rollout-external
--rollout-external-engine-addrs 127.0.0.1:31000
--rollout-num-gpus-per-engine 2
--rollout-batch-size 4
--n-samples-per-prompt 4
--global-batch-size 16
```

- [x] **Step 2: Launch one fit canary**

Run:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
cd /workspace/interp/code/nano30b-nla-pilot-current
nohup /workspace/interp/.venv/bin/python scripts/nano_rl_queue.py \
  configs/nano_rl/r33_component_full_signal_ladder_queue_8h100_len512.yaml \
  --once \
  > /workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/queue_driver_tp2fix_$(date -u +%Y%m%dT%H%MZ).log 2>&1 &
'
```

Expected: command returns immediately; the run continues on the cluster.

- [x] **Step 3: Poll until it reaches real rollout/training**

Run every 5-10 minutes:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
RUN=/workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/r33_component_full_rl_8h100_fit_rb4_n4_kl3e4_tp2fix
echo ===gpu===
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
echo ===queue===
/workspace/interp/.venv/bin/python scripts/nano_queue_status.py configs/nano_rl/r33_component_full_signal_ladder_queue_8h100_len512.yaml || true
echo ===log===
tail -n 120 "$RUN/train.log" 2>/dev/null || true
'
```

Expected to pass the previous TP assertion. If it fails, capture the first traceback and stop before attempting another fix.

- [x] **Step 4: Acceptance criteria for canary**

Mark M3 passed only if all are true:

- Queue item status is `completed` or reaches post-eval.
- No `tp_size` sanity-check failure.
- No actor OOM.
- W&B offline run exists.
- GPU usage shows all intended placements loaded:
  - Actor GPUs 0-3 nontrivial memory.
  - SGLang GPUs 4-5 high memory.
  - Critic GPUs 6-7 nontrivial memory.
- Round-trip post-eval report is produced or, if post-eval fails, the training checkpoint and failure stage are preserved.

- [x] **Step 5: Document Milestone M3**

Append:

```markdown
## 2026-06-24 - M3 TP2 Fit Canary

- Status: use exactly one of `passed`, `failed`, or `blocked`
- Queue item:
- Run dir:
- Command:
- Start/finish:
- Peak GPU memory by GPU:
- Approx rollout throughput:
- Reward mean/std/min/max:
- KL:
- Parse/close/usable rate:
- Round-trip report:
- Failure traceback if failed:
- Decision: promote/retry/reject
```

### Task 4: Throughput Profile and Medium-Run Parameter Choice

**Files:**
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/configs/nano_rl/r33_component_full_signal_ladder_queue_8h100_len512.yaml`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/rl_logbook.md`

- [x] **Step 1: Read canary metrics before changing batch size**

Extract from W&B offline logs and train log:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
RUN=/workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/r33_component_full_rl_8h100_fit_rb4_n4_kl3e4_tp2fix
grep -inE "reward|kl|advantage|rollout|tokens/s|samples/s|global|loss|step|iter|oom|out of memory|traceback|exception" "$RUN/train.log" | tail -200 || true
find "$RUN/wandb" -maxdepth 4 -type f | sort | tail -50 || true
'
```

- [x] **Step 2: Choose one medium run shape**

Use this decision table:

| Canary evidence | Medium run |
| --- | --- |
| No OOM, rollout GPU util below 70%, parse healthy | `rb8_n8_global64_num16` |
| No OOM, rollout GPU util high, actor/critic lower | keep `rb4_n4_global16`, increase `num_rollout` to `16` |
| Actor or critic memory near limit | keep canary batch, tune KL/reward normalization only |
| Rollout server bottlenecked but stable | keep TP2 and test `rb8_n4_global32` before `rb8_n8` |

- [x] **Step 3: Update the medium queue item**

Preferred medium item if canary is healthy:

```yaml
- name: r33-component-full-rl-8h100-medium-rb8-n8-kl3e4-tp2
  status: pending
  rollout:
    rollout_batch_size: 8
    global_batch_size: 64
    n_samples_per_prompt: 8
    num_rollout: 16
  training:
    actor_lr: 3e-6
    save_interval: 16
  wandb:
    run_id: r33-component-full-rl-8h100-medium-rb8-n8-kl3e4-tp2-20260624
  run_dir: /workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/r33_component_full_rl_8h100_medium_rb8_n8_kl3e4_tp2
```

- [x] **Step 4: Document Milestone M4**

Append:

```markdown
## 2026-06-24 - M4 Medium Run Parameter Selection

- Status: use exactly one of `selected`, `rejected`, or `blocked`
- Canary evidence:
- Chosen medium shape:
- Rejected alternatives and reason:
- Expected GPU placement:
- Expected ETA:
```

### Task 5: Run and Gate the Medium RL Candidate

**Files:**
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/rl_logbook.md`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/current_state.md`

- [x] **Step 1: Launch medium run only after M3 passes**

Run:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
cd /workspace/interp/code/nano30b-nla-pilot-current
nohup /workspace/interp/.venv/bin/python scripts/nano_rl_queue.py \
  configs/nano_rl/r33_component_full_signal_ladder_queue_8h100_len512.yaml \
  --once \
  > /workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/queue_driver_medium_tp2_$(date -u +%Y%m%dT%H%MZ).log 2>&1 &
'
```

- [x] **Step 2: Monitor throughput and quality**

Poll:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
RUN=/workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/r33_component_full_rl_8h100_medium_rb8_n8_kl3e4_tp2
echo ===gpu===
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
echo ===recent metrics===
grep -inE "reward|kl|advantage|rollout|tokens/s|samples/s|loss|step|iter|roundtrip|nmse|parse|closed|usable" "$RUN/train.log" | tail -160 || true
echo ===disk===
df -h /workspace/interp /workspace/models /dev/shm
'
```

- [x] **Step 3: Medium pass criteria**

Mark medium as passed only if:

- It completes `num_rollout=16`.
- No OOM or Ray actor death.
- Reward variance is not collapsed to zero.
- KL is finite and not exploding.
- Round-trip eval parses enough generated explanations:
  - closed fraction at least `0.8`
  - usable fraction at least `0.95`
- Round-trip NMSE improves over or is competitive with the clean SFT baseline in `baseline_report_json`.

- Result: the original 16-rollout medium and follow-up 32-rollout variant both completed without OOM and with healthy parse/control behavior, but neither passed the clean SFT baseline gate on both validation and test. Treat Task 5 as completed/rejected, not passed for hero promotion.

- [x] **Step 4: Document Milestone M5**

Append:

```markdown
## 2026-06-24 - M5 Medium RL Candidate

- Status: use exactly one of `passed`, `failed`, or `blocked`
- Queue item:
- Run dir:
- Runtime:
- Peak GPU memory/utilization:
- Throughput:
- Reward stats:
- KL stats:
- Round-trip report:
- Comparison to clean SFT baseline:
- Decision: hero-promote / run one more medium variant / stop
```

### Task 6: Launch the High-Throughput Hero Run

Task 6 is intentionally not launched as of `2026-06-25`: the best medium evidence so far (`r33-component-full-rl-8h100-medium-rb8-n8-kl3e4-tp2-rollout32`) completed but failed the round-trip gate on test versus the clean SFT baseline. Keep the hero item gated until a medium candidate improves or matches the clean SFT baseline on both validation and test.

**Files:**
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/configs/nano_rl/r33_component_full_signal_ladder_queue_8h100_len512.yaml`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/rl_logbook.md`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/current_state.md`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/experiment_logbook.md`

- [ ] **Step 1: Choose hero params from medium evidence**

Default hero if M5 passes:

```yaml
- name: r33-component-full-rl-8h100-hero-rb8-n8-kl3e4-tp2
  status: pending
  rollout:
    rollout_batch_size: 8
    global_batch_size: 64
    n_samples_per_prompt: 8
    num_rollout: 128
  training:
    actor_lr: 3e-6
    save_interval: 32
  wandb:
    run_id: r33-component-full-rl-8h100-hero-rb8-n8-kl3e4-tp2-20260624
  run_dir: /workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/r33_component_full_rl_8h100_hero_rb8_n8_kl3e4_tp2
```

Use `num_rollout=64` instead of `128` if medium runtime implies the 128-rollout run would exceed the available unattended window or create too much checkpoint/log pressure.

- [ ] **Step 2: Pre-hero cleanup**

Before launch, preserve selected medium checkpoint and remove only rejected RL canary checkpoint directories:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
set -euo pipefail
df -h /workspace/interp /workspace/models /dev/shm
find /workspace/interp/outputs/nano30b-nla-pilot/rl_8h100 -maxdepth 3 -type d -name "iter_*" | sort
'
```

Do not delete:

- clean R33 AV SFT hero checkpoint `iter_0001291`
- clean R33 AR SFT hero checkpoint `iter_0001289`
- the selected medium RL checkpoint
- W&B offline logs
- round-trip reports

- [ ] **Step 3: Launch hero**

Run:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
cd /workspace/interp/code/nano30b-nla-pilot-current
nohup /workspace/interp/.venv/bin/python scripts/nano_rl_queue.py \
  configs/nano_rl/r33_component_full_signal_ladder_queue_8h100_len512.yaml \
  --once \
  > /workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/queue_driver_hero_tp2_$(date -u +%Y%m%dT%H%MZ).log 2>&1 &
'
```

- [ ] **Step 4: Hero monitoring cadence**

Poll every 30 minutes:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
RUN=/workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/r33_component_full_rl_8h100_hero_rb8_n8_kl3e4_tp2
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
/workspace/interp/.venv/bin/python scripts/nano_queue_status.py configs/nano_rl/r33_component_full_signal_ladder_queue_8h100_len512.yaml || true
grep -inE "reward|kl|advantage|rollout|tokens/s|samples/s|loss|step|iter|roundtrip|nmse|parse|closed|usable|traceback|exception|oom|out of memory" "$RUN/train.log" | tail -220 || true
df -h /workspace/interp /workspace/models /dev/shm
'
```

- [ ] **Step 5: Hero pass criteria**

Mark hero as passed only if:

- Training completes the configured rollout count.
- Final checkpoint exists and is preserved.
- W&B offline logs exist.
- Final round-trip gate exists.
- Final generated-text parse/close rates are healthy.
- Final round-trip NMSE beats the clean SFT baseline materially or shows a clearly better reward/round-trip tradeoff without control collapse.
- Controls remain meaningful; no generated-text shortcut or parser artifact dominates the score.

- [ ] **Step 6: Document Milestone M6**

Append to all three docs:

```markdown
## 2026-06-24 - M6 R33 8x H100 RL Hero

- Status: use exactly one of `passed`, `failed`, or `blocked`
- Queue item:
- Run dir:
- Final checkpoint:
- Runtime:
- Peak GPU memory/utilization:
- Throughput:
- Reward stats:
- KL stats:
- Round-trip gate:
- SFT baseline comparison:
- Retained artifacts:
- Deleted artifacts:
- Next decision:
```

### Task 7: Post-Hero Artifact Hygiene and Sync

**Files:**
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/rl_logbook.md`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/current_state.md`

- [ ] **Step 1: Inventory retained artifacts**

Run:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
find /workspace/interp/outputs/nano30b-nla-pilot/rl_8h100 -maxdepth 4 \
  \\( -name "train.log" -o -name "*report.json" -o -name "queue_status.json" -o -name "iter_*" \\) \
  -print | sort
du -sh /workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/* 2>/dev/null | sort -h
df -h /workspace/interp /workspace/models
'
```

- [ ] **Step 2: Upload selected hero artifacts to S3**

Upload:

- final actor checkpoint or HF-converted model if worth preserving
- final round-trip report JSON
- generated text parquet/jsonl if produced
- W&B offline run directory or compressed snapshot
- queue status and train log

Use:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
set -euo pipefail
AWS_SHARED_CREDENTIALS_FILE=/workspace/interp/secrets/aws/credentials \
AWS_CONFIG_FILE=/workspace/interp/secrets/aws/config \
/workspace/interp/.venv/bin/aws s3 cp --recursive \
  /workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/r33_component_full_rl_8h100_hero_rb8_n8_kl3e4_tp2 \
  s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/rl_8h100/r33_component_full_rl_8h100_hero_rb8_n8_kl3e4_tp2 \
  --exclude "checkpoints/*/optimizer*" \
  --exclude "**/optimizer*" \
  --exclude "**/tmp_hf/*"
'
```

- [ ] **Step 3: Sync lightweight reports back to local**

Use S3 or `nano_s3.py` to pull reports and logs into:

```text
/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/artifacts/runai_rl/
```

- [ ] **Step 4: Document Milestone M7**

Append:

```markdown
## 2026-06-24 - M7 Post-Hero Artifact Sync

- Status: use exactly one of `passed`, `failed`, or `blocked`
- S3 prefix:
- Local artifact path:
- Preserved checkpoint:
- Deleted checkpoint shards:
- W&B offline path:
- Disk before/after cleanup:
```

---

## Self-Review

- Spec coverage: The plan covers H100 topology correction, S3 sync, runtime cleanup, canary, throughput profiling, medium run, hero run, post-hero sync, and milestone docs.
- Placeholder scan: No `TBD`/`TODO` placeholders remain. Commands and expected evidence are explicit.
- Type consistency: The plan consistently uses `rollout_num_gpus_per_engine` as Miles CLI `--rollout-num-gpus-per-engine`, SGLang `--tp-size 2`, and one external endpoint.
- Main risk: If Miles rejects one TP2 external endpoint even with `--rollout-num-gpus-per-engine 2`, stop after the canary traceback and switch to a two-TP1-engine design only after documenting why TP2 is incompatible.
