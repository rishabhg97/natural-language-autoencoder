# R33 RL Qwen-Comparable Two-Tier Scaling Plan

**Goal:** Scale R33 Nano30B NLA RL from the completed 2,048-generation medium gate to sequential Qwen-comparable RL targets without launching a larger run until the prior tier passes round-trip gates.

**Architecture:** Keep the proven 8x H100 TP2 topology first: actor FSDP on GPUs 0-3, one TP2 SGLang rollout engine on GPUs 4-5, and frozen AR reward/critic on GPUs 6-7. Increase generated samples per update from 64 to 512 using `rollout_batch_size=64`, `n_samples_per_prompt=8`, and `global_batch_size=512`, while keeping actor microbatch at `1` because actor GPUs already peak near 90GB. Only consider a 16-GPU topology after measuring the 512-generation/update fit canary.

**Tech Stack:** RunAI, Visor, S3 sync, `scripts/nano_rl_queue.py`, Miles FSDP RL, external SGLang TP2, frozen R33 AR reward, W&B offline, `scripts/eval_nano_av_ar_roundtrip_gate.py`, pytest.

---

## Scale Targets

Completed baseline:

- Run: `r33-component-full-rl-8h100-medium-rb8-n8-kl3e4-tp2-rollout32`.
- Size: `32 updates * 8 prompts/update * 8 samples/prompt = 2,048` generated rollouts.
- Result: systems pass; quality gate false because test NMSE regressed against clean SFT.

Tier 1: high-batch Nano RL medium.

- Target shape: `rollout_batch_size=64`, `n_samples_per_prompt=8`, `global_batch_size=512`.
- Fit canary: `2 updates * 512 = 1,024` generated rollouts.
- Medium A: `32 updates * 512 = 16,384` generated rollouts.
- Medium B: `64 updates * 512 = 32,768` generated rollouts.
- Purpose: establish whether Qwen-style batch size and actor LR `5e-6` produce a stable positive round-trip delta before spending days on Tier 2.

Tier 2: Qwen-comparable Nano RL hero.

- Minimum Qwen-useful comparison: `150 updates * 512 = 76,800` generated rollouts.
- Preferred Nano hero if Tier 1 is positive: `256 updates * 512 = 131,072` generated rollouts.
- Purpose: become comparable to the Qwen useful-saturation scale, not just the previous Nano systems hero.

Do not launch Tier 2 unless Tier 1 Medium A or B beats the clean SFT baseline on both validation and test.

Clean SFT baseline for gates:

- Validation primary NMSE: `0.0001096800115192309`.
- Test primary NMSE: `0.00012166367378085852`.
- Baseline report: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/roundtrip_iter_0001291_v256_t256_report.json`.

---

### Task 1: Create A Qwen-Comparable 8x H100 Queue

**Files:**
- Create: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/tests/test_nano_rl_queue.py`

- [ ] **Step 1: Add a queue test for 512-generation/update items**

Add this test to `tests/test_nano_rl_queue.py`:

```python
def test_checked_in_r33_qwen_comparable_queue_uses_512_generation_updates(self):
    queue_path = REPO_ROOT / "configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml"
    doc = nano_rl_queue.load_queue(queue_path)

    expected = {
        "r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-tp2": 2,
        "r33-component-rl-8h100-tier1-medium-rb64-n8-gb512-lr5e6-rollout32": 32,
        "r33-component-rl-8h100-tier1-confirm-rb64-n8-gb512-lr5e6-rollout64": 64,
        "r33-component-rl-8h100-tier2-minhero-rb64-n8-gb512-lr5e6-rollout150": 150,
        "r33-component-rl-8h100-tier2-hero-rb64-n8-gb512-lr5e6-rollout256": 256,
    }
    items = {item["name"]: item for item in doc["items"]}
    self.assertEqual(set(items), set(expected))

    for name, num_rollout in expected.items():
        item = items[name]
        rollout = item["rollout"]
        training = item["training"]
        self.assertEqual(rollout["rollout_batch_size"], 64)
        self.assertEqual(rollout["n_samples_per_prompt"], 8)
        self.assertEqual(rollout["global_batch_size"], 512)
        self.assertEqual(rollout["num_rollout"], num_rollout)
        self.assertEqual(rollout["rollout_batch_size"] * rollout["n_samples_per_prompt"], 512)
        self.assertEqual(training["actor_lr"], "5e-6")
        self.assertEqual(training["actor_micro_batch"], 1)
        self.assertEqual(training["kl_loss_coef"], 0.0003)

    self.assertEqual(items["r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-tp2"]["status"], "pending")
    for name in expected:
        if name != "r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-tp2":
            self.assertEqual(items[name]["status"], "blocked")

    spec = nano_rl_queue.build_run_spec(doc, items["r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-tp2"], queue_path=queue_path)
    command = spec["command"]
    self.assertIn("--rollout-num-gpus-per-engine", command)
    self.assertEqual(command[command.index("--rollout-num-gpus-per-engine") + 1], "2")
    addr_index = command.index("--rollout-external-engine-addrs")
    self.assertEqual(command[addr_index + 1], "127.0.0.1:31000")
    self.assertEqual(spec["sglang_service"]["engine_addrs"], ["127.0.0.1:31000"])
```

- [ ] **Step 2: Run the test and verify it fails before the queue exists**

Run:

```bash
/Users/rigarg/.local/bin/visor run python3 -m pytest tests/test_nano_rl_queue.py::NanoRLQueueTests::test_checked_in_r33_qwen_comparable_queue_uses_512_generation_updates -q
```

Expected: FAIL because `configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml` does not exist.

- [ ] **Step 3: Create the queue YAML**

Create `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml` by copying the `defaults` block from `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/configs/nano_rl/r33_component_full_signal_ladder_queue_8h100_len512.yaml`, then replace `items:` with this exact item list:

```yaml
items:
  - name: r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-tp2
    status: pending
    rollout:
      rollout_batch_size: 64
      global_batch_size: 512
      n_samples_per_prompt: 8
      num_rollout: 2
    training:
      actor_lr: "5e-6"
      actor_micro_batch: 1
      kl_loss_coef: 0.0003
      save_interval: 2
    wandb:
      run_id: r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-tp2-20260625
    rl_parquet: /workspace/interp/outputs/nano30b-nla-pilot/posthero/20260621T163000Z/rl_smoke/rl_R33_fullscan275396_smoke512.parquet
    instruct_model: /workspace/interp/models/nano-30b-a3b-bf16-hf
    actor_sft_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291
    actor_load_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints
    actor_ref_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/rl_smoke/r33_component_full_sft_init_512row_3h200/actor_sft_hf_iter_0001291
    actor_sidecar_source: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291
    critic_sl_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001289/hf
    run_dir: /workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/r33_component_rl_8h100_tier1_fit_rb64_n8_gb512_lr5e6_tp2
    notes: "Tier 1 fit canary: first 512-generation/update R33 RL run. Two updates only. Do not judge quality; measure memory, throughput, KL/reward diagnostics, SGLang/reward pressure, and cleanup."

  - name: r33-component-rl-8h100-tier1-medium-rb64-n8-gb512-lr5e6-rollout32
    status: blocked
    rollout:
      rollout_batch_size: 64
      global_batch_size: 512
      n_samples_per_prompt: 8
      num_rollout: 32
    training:
      actor_lr: "5e-6"
      actor_micro_batch: 1
      kl_loss_coef: 0.0003
      save_interval: 32
    wandb:
      run_id: r33-component-rl-8h100-tier1-medium-rb64-n8-gb512-lr5e6-rollout32-20260625
    rl_parquet: /workspace/interp/outputs/nano30b-nla-pilot/posthero/20260621T163000Z/rl_smoke/rl_R33_fullscan275396_smoke512.parquet
    instruct_model: /workspace/interp/models/nano-30b-a3b-bf16-hf
    actor_sft_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291
    actor_load_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints
    actor_ref_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/rl_smoke/r33_component_full_sft_init_512row_3h200/actor_sft_hf_iter_0001291
    actor_sidecar_source: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291
    critic_sl_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001289/hf
    run_dir: /workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/r33_component_rl_8h100_tier1_medium_rb64_n8_gb512_lr5e6_rollout32
    notes: "Tier 1 medium A: 16,384 generated rollouts. Unblock only after the fit canary completes without OOM and has finite KL/reward diagnostics."

  - name: r33-component-rl-8h100-tier1-confirm-rb64-n8-gb512-lr5e6-rollout64
    status: blocked
    rollout:
      rollout_batch_size: 64
      global_batch_size: 512
      n_samples_per_prompt: 8
      num_rollout: 64
    training:
      actor_lr: "5e-6"
      actor_micro_batch: 1
      kl_loss_coef: 0.0003
      save_interval: 64
    wandb:
      run_id: r33-component-rl-8h100-tier1-confirm-rb64-n8-gb512-lr5e6-rollout64-20260625
    rl_parquet: /workspace/interp/outputs/nano30b-nla-pilot/posthero/20260621T163000Z/rl_smoke/rl_R33_fullscan275396_smoke512.parquet
    instruct_model: /workspace/interp/models/nano-30b-a3b-bf16-hf
    actor_sft_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291
    actor_load_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints
    actor_ref_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/rl_smoke/r33_component_full_sft_init_512row_3h200/actor_sft_hf_iter_0001291
    actor_sidecar_source: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291
    critic_sl_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001289/hf
    run_dir: /workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/r33_component_rl_8h100_tier1_confirm_rb64_n8_gb512_lr5e6_rollout64
    notes: "Tier 1 medium B: 32,768 generated rollouts. Unblock only if Tier 1 medium A has no OOM and is at least neutral versus clean SFT on both validation and test."

  - name: r33-component-rl-8h100-tier2-minhero-rb64-n8-gb512-lr5e6-rollout150
    status: blocked
    rollout:
      rollout_batch_size: 64
      global_batch_size: 512
      n_samples_per_prompt: 8
      num_rollout: 150
    training:
      actor_lr: "5e-6"
      actor_micro_batch: 1
      kl_loss_coef: 0.0003
      save_interval: 150
    wandb:
      run_id: r33-component-rl-8h100-tier2-minhero-rb64-n8-gb512-lr5e6-rollout150-20260625
    rl_parquet: /workspace/interp/outputs/nano30b-nla-pilot/posthero/20260621T163000Z/rl_smoke/rl_R33_fullscan275396_smoke512.parquet
    instruct_model: /workspace/interp/models/nano-30b-a3b-bf16-hf
    actor_sft_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291
    actor_load_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints
    actor_ref_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/rl_smoke/r33_component_full_sft_init_512row_3h200/actor_sft_hf_iter_0001291
    actor_sidecar_source: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291
    critic_sl_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001289/hf
    run_dir: /workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/r33_component_rl_8h100_tier2_minhero_rb64_n8_gb512_lr5e6_rollout150
    notes: "Tier 2 minimum Qwen-useful hero: 76,800 generated rollouts. Unblock only if Tier 1 confirm passes both-split clean SFT gate."

  - name: r33-component-rl-8h100-tier2-hero-rb64-n8-gb512-lr5e6-rollout256
    status: blocked
    rollout:
      rollout_batch_size: 64
      global_batch_size: 512
      n_samples_per_prompt: 8
      num_rollout: 256
    training:
      actor_lr: "5e-6"
      actor_micro_batch: 1
      kl_loss_coef: 0.0003
      save_interval: 256
    wandb:
      run_id: r33-component-rl-8h100-tier2-hero-rb64-n8-gb512-lr5e6-rollout256-20260625
    rl_parquet: /workspace/interp/outputs/nano30b-nla-pilot/posthero/20260621T163000Z/rl_smoke/rl_R33_fullscan275396_smoke512.parquet
    instruct_model: /workspace/interp/models/nano-30b-a3b-bf16-hf
    actor_sft_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291
    actor_load_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints
    actor_ref_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/rl_smoke/r33_component_full_sft_init_512row_3h200/actor_sft_hf_iter_0001291
    actor_sidecar_source: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291
    critic_sl_ckpt: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001289/hf
    run_dir: /workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/r33_component_rl_8h100_tier2_hero_rb64_n8_gb512_lr5e6_rollout256
    notes: "Tier 2 preferred Nano hero: 131,072 generated rollouts. Unblock only after the 150-rollout minimum hero passes and storage/time budget is acceptable."
```

- [ ] **Step 4: Run the queue test and full focused tests**

Run:

```bash
/Users/rigarg/.local/bin/visor run python3 -m pytest tests/test_nano_rl_queue.py tests/test_nla_system_metrics.py tests/test_nla_rl_metrics.py -q
```

Expected: `34 passed` plus the new queue test count included.

### Task 2: Sync And Dry-Run The Tier 1 Fit Canary

**Files:**
- Read: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/scripts/nano_s3.py`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/rl_logbook.md`

- [ ] **Step 1: Sync code/config to RunAI through S3**

Package a lightweight source bundle:

```bash
/Users/rigarg/.local/bin/visor run bash -lc '
set -euo pipefail
TS=$(date -u +%Y%m%dT%H%MZ)
BUNDLE=.cache/sync/nano30b-nla-pilot-current-qwen-rl-tier-plan-${TS}.tgz
mkdir -p .cache/sync
COPYFILE_DISABLE=1 tar \
  --exclude=.git \
  --exclude=.cache \
  --exclude=.claude \
  --exclude=artifacts \
  --exclude=runs \
  --exclude=outputs \
  --exclude=wandb \
  --exclude="*.pt" \
  --exclude="*.pth" \
  --exclude="*.safetensors" \
  --exclude="*.bin" \
  --exclude="*.parquet" \
  --exclude="*.jsonl" \
  --exclude="*.tgz" \
  --exclude="*.tar" \
  --exclude="*.zst" \
  -czf "$BUNDLE" .
sha256sum "$BUNDLE"
aws --endpoint-url https://pdx.s8k.io s3 cp "$BUNDLE" "s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/code-sync/$(basename "$BUNDLE")" --no-progress
'
```

Expected: upload succeeds and prints the bundle SHA256 without printing credentials.

- [ ] **Step 2: Pull the bundle on RunAI**

Run:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
set -euo pipefail
KEY=$(AWS_SHARED_CREDENTIALS_FILE=/workspace/interp/secrets/aws/credentials AWS_CONFIG_FILE=/workspace/interp/secrets/aws/config AWS_REQUEST_CHECKSUM_CALCULATION=when_required AWS_RESPONSE_CHECKSUM_VALIDATION=when_required /workspace/interp/.venv/bin/aws --endpoint-url https://pdx.s8k.io s3 ls s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/code-sync/ | awk "/qwen-rl-tier-plan/ {print \$4}" | tail -1)
BUNDLE=/workspace/interp/tmp/$KEY
TARGET=/workspace/interp/code/nano30b-nla-pilot-current-qwen-rl-tier-$(date -u +%Y%m%dT%H%MZ)
AWS_SHARED_CREDENTIALS_FILE=/workspace/interp/secrets/aws/credentials \
AWS_CONFIG_FILE=/workspace/interp/secrets/aws/config \
AWS_REQUEST_CHECKSUM_CALCULATION=when_required \
AWS_RESPONSE_CHECKSUM_VALIDATION=when_required \
/workspace/interp/.venv/bin/aws --endpoint-url https://pdx.s8k.io s3api get-object \
  --bucket team-ipp-trustworthy-ai \
  --key nano30b-nla-pilot/code-sync/$KEY \
  "$BUNDLE" >/tmp/qwen_rl_get_object.json
mkdir -p "$TARGET"
tar -xzf "$BUNDLE" -C "$TARGET"
ln -sfn "$TARGET" /workspace/interp/code/nano30b-nla-pilot-current
readlink -f /workspace/interp/code/nano30b-nla-pilot-current
'
```

Expected: `/workspace/interp/code/nano30b-nla-pilot-current` points at the new extracted target.

- [ ] **Step 3: Dry-run the fit canary on RunAI**

Run:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
set -euo pipefail
cd /workspace/interp/code/nano30b-nla-pilot-current
/workspace/interp/.venv/bin/python scripts/nano_rl_queue.py configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml --dry-run
'
```

Expected:

- Next item is `r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-tp2`.
- `rollout_batch_size=64`.
- `n_samples_per_prompt=8`.
- `global_batch_size=512`.
- `--rollout-num-gpus-per-engine 2`.
- One external SGLang endpoint: `127.0.0.1:31000`.

### Task 3: Run Tier 1 Fit Canary

**Files:**
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/rl_logbook.md`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml`

- [ ] **Step 1: Confirm RunAI is idle and has enough disk**

Run:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
set -euo pipefail
date -u +%Y-%m-%dT%H:%M:%SZ
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
ps -eo pid,etime,stat,cmd | grep -E "nano_rl_queue|roundtrip|train_actor|sglang.launch_server|ray::|raylet|gcs_server" | grep -v grep | grep -v defunct || true
df -h /workspace/interp /workspace/models /dev/shm
'
```

Expected: all GPUs near `4MiB`, no active process, and at least `250G` free on `/workspace/interp`.

- [ ] **Step 2: Launch the fit canary**

Run:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
set -euo pipefail
cd /workspace/interp/code/nano30b-nla-pilot-current
mkdir -p /workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale
LOG=/workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/queue_driver_tier1_fit_$(date -u +%Y%m%dT%H%MZ).log
nohup /workspace/interp/.venv/bin/python scripts/nano_rl_queue.py configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml --once > "$LOG" 2>&1 &
echo "$LOG"
'
```

Expected: command returns immediately and training continues on the cluster.

- [ ] **Step 3: Monitor the fit canary**

Run every 10 minutes:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
RUN=/workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/r33_component_rl_8h100_tier1_fit_rb64_n8_gb512_lr5e6_tp2
cd /workspace/interp/code/nano30b-nla-pilot-current
date -u +%Y-%m-%dT%H:%M:%SZ
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
/workspace/interp/.venv/bin/python scripts/nano_rl_queue.py configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml --status || true
python3 - <<'"'"'PY'"'"'
from pathlib import Path
import ast, re, statistics, time
p=Path("/workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/r33_component_rl_8h100_tier1_fit_rb64_n8_gb512_lr5e6_tp2/train.log")
text=p.read_text(errors="replace").replace("\r","\n") if p.exists() else ""
rollouts=[]; perfs=[]; steps=[]; bad=[]
for line in text.splitlines():
    low=line.lower()
    if any(s in low for s in ["traceback","exception","out of memory","oom","assertionerror","runtimeerror"]):
        bad.append(line)
    m=re.search(r"rollout (\d+): (\{.*\})", line)
    if m:
        try: rollouts.append((int(m.group(1)), ast.literal_eval(m.group(2))))
        except Exception: pass
    m=re.search(r"perf (\d+): (\{.*\})", line)
    if m and "update_weights_time" in m.group(2):
        try: perfs.append((int(m.group(1)), ast.literal_eval(m.group(2))))
        except Exception: pass
    m=re.search(r"step (\d+): (\{.*\})", line)
    if m:
        try: steps.append((int(m.group(1)), ast.literal_eval(m.group(2))))
        except Exception: pass
print("rollouts", len(rollouts), rollouts[-1][0] if rollouts else None)
if rollouts:
    raw=[d.get("rollout/raw_reward") for _, d in rollouts if d.get("rollout/raw_reward") is not None]
    print("raw_mean", statistics.mean(raw), "raw_min", min(raw), "raw_max", max(raw))
if perfs:
    print("last_perf", perfs[-1])
if steps:
    print("last_step", steps[-1])
print("bad_count", len(bad), "completed", "# completed_utc=" in text, "mtime", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime)) if p.exists() else None)
if bad:
    print("bad_tail", bad[-1][-500:])
PY
df -h /workspace/interp /workspace/models /dev/shm
'
```

Expected: 2 rollout/update cycles finish without OOM. If OOM happens, stop and document; do not unblock Tier 1 medium.

- [ ] **Step 4: Fit canary pass criteria**

Mark the canary passed only if:

- Queue item status is `complete`.
- No actor, SGLang, or critic OOM.
- No `tp_size` assertion.
- Peak actor memory remains below `94GB`.
- Step time is measured for both updates.
- Reward stats are finite and not all identical.
- `train/ppo_kl`, `train/kl_loss`, and rollout/policy logprob abs-diff are present or the missing fields are explicitly documented.
- Post-eval 64/64 report is produced with closed fraction at least `0.8` and usable fraction at least `0.95`.
- Temporary HF and actor DCP payloads are cleaned after eval.

### Task 4: Run Tier 1 Medium A, 16,384 Generated Rollouts

**Files:**
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/rl_logbook.md`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/current_state.md`

- [ ] **Step 1: Unblock Medium A only after canary passes**

Change only this queue item:

```yaml
  - name: r33-component-rl-8h100-tier1-medium-rb64-n8-gb512-lr5e6-rollout32
    status: pending
```

Keep all later items `blocked`.

- [ ] **Step 2: Launch Medium A**

Run:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
set -euo pipefail
cd /workspace/interp/code/nano30b-nla-pilot-current
LOG=/workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/queue_driver_tier1_medium32_$(date -u +%Y%m%dT%H%MZ).log
nohup /workspace/interp/.venv/bin/python scripts/nano_rl_queue.py configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml --once > "$LOG" 2>&1 &
echo "$LOG"
'
```

Expected: queue starts `r33-component-rl-8h100-tier1-medium-rb64-n8-gb512-lr5e6-rollout32`.

- [ ] **Step 3: Medium A gate**

Pass Medium A only if:

- It completes `32` updates and `16,384` generated rollouts.
- It does not OOM.
- Post-eval report exists.
- Parse health passes.
- AV-real beats all controls.
- Validation and test primary NMSE both match or beat the clean SFT baseline:
  - validation `<= 0.0001096800115192309`
  - test `<= 0.00012166367378085852`
- If validation improves but test regresses, stop and do not unblock Tier 1 confirm.

### Task 5: Run Tier 1 Confirm, 32,768 Generated Rollouts

**Files:**
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/rl_logbook.md`

- [ ] **Step 1: Unblock Tier 1 confirm only if Medium A passes**

Change only this queue item:

```yaml
  - name: r33-component-rl-8h100-tier1-confirm-rb64-n8-gb512-lr5e6-rollout64
    status: pending
```

Expected: no Tier 2 item is pending.

- [ ] **Step 2: Launch Tier 1 confirm**

Run:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
set -euo pipefail
cd /workspace/interp/code/nano30b-nla-pilot-current
LOG=/workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/queue_driver_tier1_confirm64_$(date -u +%Y%m%dT%H%MZ).log
nohup /workspace/interp/.venv/bin/python scripts/nano_rl_queue.py configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml --once > "$LOG" 2>&1 &
echo "$LOG"
'
```

- [ ] **Step 3: Tier 1 confirm gate**

Pass Tier 1 only if:

- It completes `64` updates and `32,768` generated rollouts.
- It remains stable without OOM.
- It beats or matches the clean SFT baseline on both validation and test.
- It is not worse than Medium A on test by more than `5e-7` absolute NMSE.
- Reward/response length does not drift into a verbosity shortcut: mean response length should remain below `150` with truncation max `0.0`.

If Tier 1 confirm passes, document that the next run is now Qwen-useful comparable by sample count. If it fails, stop before Tier 2 and diagnose LR/KL/reward normalization.

### Task 6: Run Tier 2 Minimum Hero, 76,800 Generated Rollouts

**Files:**
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/rl_logbook.md`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/current_state.md`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/experiment_logbook.md`

- [ ] **Step 1: Estimate runtime from Tier 1**

Compute:

```text
estimated_seconds = tier1_confirm_steady_step_seconds * 150 + post_eval_seconds
```

Do not launch if the estimate exceeds the available unattended window or if `/workspace/interp` has less than `300G` free.

- [ ] **Step 2: Unblock Tier 2 minimum hero**

Change only this queue item:

```yaml
  - name: r33-component-rl-8h100-tier2-minhero-rb64-n8-gb512-lr5e6-rollout150
    status: pending
```

- [ ] **Step 3: Launch Tier 2 minimum hero**

Run:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
set -euo pipefail
cd /workspace/interp/code/nano30b-nla-pilot-current
LOG=/workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/queue_driver_tier2_minhero150_$(date -u +%Y%m%dT%H%MZ).log
nohup /workspace/interp/.venv/bin/python scripts/nano_rl_queue.py configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml --once > "$LOG" 2>&1 &
echo "$LOG"
'
```

- [ ] **Step 4: Tier 2 minimum hero gate**

Pass Tier 2 minimum hero only if:

- It completes `150` updates and `76,800` generated rollouts.
- It beats clean SFT on validation and test.
- It beats the best Tier 1 result on test.
- AV-real beats all controls.
- Parse closed/usable fractions remain `1.0 / 1.0` or at least above `0.8 / 0.95`.
- Final lightweight artifacts are mirrored locally and to S3.

### Task 7: Optional Tier 2 Preferred Hero, 131,072 Generated Rollouts

**Files:**
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/experiment_logbook.md`

- [ ] **Step 1: Decide whether 256 updates are justified**

Unblock the 256-update item only if:

- Tier 2 minimum hero passes.
- Runtime is acceptable.
- Storage remains above `300G` free.
- The reward/test curves suggest continued improvement rather than saturation.

- [ ] **Step 2: Launch preferred hero**

Change:

```yaml
  - name: r33-component-rl-8h100-tier2-hero-rb64-n8-gb512-lr5e6-rollout256
    status: pending
```

Then run:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
set -euo pipefail
cd /workspace/interp/code/nano30b-nla-pilot-current
LOG=/workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/queue_driver_tier2_hero256_$(date -u +%Y%m%dT%H%MZ).log
nohup /workspace/interp/.venv/bin/python scripts/nano_rl_queue.py configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml --once > "$LOG" 2>&1 &
echo "$LOG"
'
```

Pass only if it beats the 150-update run and the clean SFT baseline on test.

### Task 8: If 8x H100 Throughput Is Too Slow, Design A 16-GPU Topology

**Files:**
- Create: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/superpowers/specs/2026-06-25-r33-rl-16gpu-throughput-design.md`

- [ ] **Step 1: Trigger condition**

Create the 16-GPU design only if Tier 1 fit canary steady step time is above `900s/update` or Tier 1 Medium A ETA exceeds `14h`.

- [ ] **Step 2: Compare topologies**

The design must compare these exact candidates:

```text
8x H100 current:
  actor=4, rollout=2 TP2, critic=2

16x H100 candidate A:
  actor=8, rollout=4 as two TP2 engines, critic=4

16x H100 candidate B:
  actor=8, rollout=2 TP2, critic=6

16x H100 candidate C:
  actor=6, rollout=4 as two TP2 engines, critic=6
```

The design must state whether `scripts/nano_rl_queue.py` and Miles can launch those multi-node shapes without code changes. If code changes are required, stop and plan them separately.

### Task 9: Documentation, Sync, And Cleanup After Every Completed Run

**Files:**
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/rl_logbook.md`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/current_state.md`
- Modify: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/experiment_logbook.md`

- [ ] **Step 1: Document each completed run**

Each run entry must include:

- Queue item.
- Run dir.
- Start/finish times.
- Exact generated rollout count.
- Step time mean.
- GPU peak memory by role.
- Reward mean/std/min/max.
- Response length mean/max and truncation.
- KL/logprob diagnostics.
- Round-trip report path.
- Validation/test primary NMSE.
- Clean SFT comparison.
- Pass/fail decision.
- Cleanup actions.

- [ ] **Step 2: Preserve lightweight artifacts**

For each completed run that has a round-trip report and has not already been archived, create a remote archive:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
set -euo pipefail
ROOT=/workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for RUN in "$ROOT"/r33_component_rl_8h100_*; do
  [ -d "$RUN" ] || continue
  [ -f "$RUN/.light_artifacts_uploaded" ] && continue
  find "$RUN" -maxdepth 1 -name "roundtrip_iter_*_report.json" -type f | grep -q . || continue
  BASENAME=$(basename "$RUN")
  ARCHIVE=/workspace/interp/tmp/${BASENAME}_light_artifacts_${STAMP}.tgz
  cd "$ROOT"
  tar -czf "$ARCHIVE" "$BASENAME"
  sha256sum "$ARCHIVE" | tee "$RUN/light_artifacts_${STAMP}.sha256"
  AWS_SHARED_CREDENTIALS_FILE=/workspace/interp/secrets/aws/credentials \
  AWS_CONFIG_FILE=/workspace/interp/secrets/aws/config \
  AWS_REQUEST_CHECKSUM_CALCULATION=when_required \
  AWS_RESPONSE_CHECKSUM_VALIDATION=when_required \
  /workspace/interp/.venv/bin/aws --endpoint-url https://pdx.s8k.io s3 cp "$ARCHIVE" "s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/sync_exports/${STAMP}_r33_rl_qwen_scale/${BASENAME}_light_artifacts_${STAMP}.tgz" --no-progress
  printf "%s\n" "s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/sync_exports/${STAMP}_r33_rl_qwen_scale/${BASENAME}_light_artifacts_${STAMP}.tgz" > "$RUN/.light_artifacts_uploaded"
  rm -f "$ARCHIVE"
done
'
```

Do not archive actor DCP checkpoint shards unless explicitly preserving a promoted checkpoint. Queue post-eval should have already removed temporary HF and actor DCP payloads; verify the archive is lightweight before keeping it.

- [ ] **Step 3: Verify cleanup**

Run:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
ps -eo pid,etime,stat,cmd | grep -E "nano_rl_queue|roundtrip|train_actor|sglang.launch_server|ray::|raylet|gcs_server" | grep -v grep | grep -v defunct || true
df -h /workspace/interp /workspace/models /dev/shm
'
```

Expected: GPUs idle, no active process, and `/workspace/interp` has enough free space for the next run.

---

## Self-Review

- Spec coverage: both requested sequential tiers are represented. Tier 1 reaches `16,384` then `32,768` generated rollouts at Qwen-style `512` generations/update. Tier 2 reaches `76,800` then `131,072` generated rollouts, comparable to Qwen useful saturation.
- Placeholder scan: no unresolved placeholder tokens, angle-bracket placeholders, or unspecified command remains.
- Type consistency: queue item names, rollout fields, training fields, paths, and command names match the existing queue runner schema and current 8x H100 config style.
- Safety check: only the Tier 1 fit canary starts as `pending`; every larger run starts `blocked`, so `--once` cannot accidentally launch Tier 1/Tier 2 before gates pass.
