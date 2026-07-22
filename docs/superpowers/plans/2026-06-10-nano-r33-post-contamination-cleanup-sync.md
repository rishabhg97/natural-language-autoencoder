# Nano R33 Post-Contamination Cleanup Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconcile local, RunAI, and S3 state after the Nano/Nemotron-H packed-boundary contamination finding so future R33 hero work starts from clean code, honest labels, and lightweight preserved evidence.

**Architecture:** Treat RunAI as the source for completed run evidence and the local repo as the source for code/docs. Preserve lightweight evidence before deleting heavy artifacts. Delete only model/checkpoint/HF payloads that are either contaminated, obsolete, or reproducible from preserved configs and logs.

**Tech Stack:** RunAI workspace `train`, project `trustworthy-ai-inference`, local repo `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot`, remote root `/workspace/interp/code/nano30b-nla-pilot-current`, S3 helper `scripts/nano_s3.py`.

---

### Task 1: Inventory And Safety Manifest

**Files:**
- Create: `artifacts/runai_sync/<timestamp>/runai_cleanup_manifest.json`
- Create: `artifacts/runai_sync/<timestamp>/runai_light_artifacts_manifest.json`

- [ ] **Step 1: Verify RunAI is idle**

Run:

```bash
/Users/rigarg/.runai/bin/2.116.4/runai workspace exec train -p trustworthy-ai-inference -- bash -lc 'ps -eo pid,etime,cmd | grep -E "nano_av|nano_ar|eval_nano|train_actor|run_nano|verify_nano|convert_fsdp" | grep -v grep || true; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader; df -h /workspace/interp /workspace/models /tmp /dev/shm'
```

Expected: no active Nano processes, GPUs idle, disk not near pressure.

- [ ] **Step 2: Record heavy artifact candidates**

Run:

```bash
/Users/rigarg/.runai/bin/2.116.4/runai workspace exec train -p trustworthy-ai-inference -- bash -lc 'du -xh --max-depth=2 /workspace/interp/outputs/nano30b-nla-pilot 2>/dev/null | sort -hr | head -100; find /workspace/interp/outputs/nano30b-nla-pilot -type f -size +500M -printf "%s %p\n" 2>/dev/null | sort -nr | head -120'
```

Expected: identify checkpoint/HF payloads separately from datasets and verifier/eval artifacts.

### Task 2: Preserve Lightweight Run Evidence

**Files:**
- Create: `artifacts/runai_sync/<timestamp>/runai_light_artifacts_<timestamp>.tgz`

- [ ] **Step 1: Build remote lightweight evidence archive**

Include `.json`, `.jsonl`, `.yaml`, `.yml`, `.txt`, `.log`, `.md`, `.csv`, `.out`, `.err`, and W&B metadata/log files. Exclude checkpoint directories, HF conversions, datasets, tensor/model files, and parquets.

- [ ] **Step 2: Stream archive to local**

Use `runai workspace exec` stdout streaming, then verify `tar -tzf` and compare the remote/local SHA-256 values.

### Task 3: Clean Heavy Contaminated Artifacts

**Files:**
- Modify remote-only: `/workspace/interp/outputs/nano30b-nla-pilot/...`
- Preserve local manifest: `artifacts/runai_sync/<timestamp>/runai_cleanup_manifest.json`

- [ ] **Step 1: Delete contaminated R33 full275k AR checkpoint payload**

Remove:

```text
/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-hero/nano-ar-r33-full275k-lr5e5-cosine-warmup25-gb192-mb96/checkpoints
```

Rationale: trained before the packed-boundary contamination fix; logs and eval metadata are preserved.

- [ ] **Step 2: Remove stale lock files**

Remove zero-byte or stale `*.lock` files only after confirming no matching queue process is active.

- [ ] **Step 3: Recheck disk**

Expected: `/workspace/interp` free space increases by roughly the deleted checkpoint size.

### Task 4: Sync Code State

**Files:**
- Remote code root: `/workspace/interp/code/nano30b-nla-pilot-current`
- S3 prefix: `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/code_sync/`

- [ ] **Step 1: Create local source archive**

Exclude `.git`, local heavy artifacts, caches, checkpoints, model files, and generated sync archives.

- [ ] **Step 2: Extract archive into the RunAI code copy**

Keep remote path stable so existing configs continue to use `/workspace/interp/code/nano30b-nla-pilot-current`.

- [ ] **Step 3: Upload the same source archive to S3 from RunAI**

Use remote credentials through `scripts/nano_s3.py` or AWS CLI without printing secrets.

### Task 5: Correct Documentation Labels

**Files:**
- Modify: `docs/experiment_logbook.md`
- Modify: `docs/nano_av_run_history.md`
- Modify: `docs/nano_av_job_tracker.md`
- Modify/Create: `docs/current_state.md`
- Modify/Create: `docs/architecture/audit_remediation_20260610.md`

- [ ] **Step 1: Add post-contamination status note**

Label pre-fix AR/AV checkpoints as contaminated-training evidence. Keep metrics as scouting signal only.

- [ ] **Step 2: Label cosine LR schedule bug**

Mark runs launched before the LR schedule remediation as potentially affected by the Miles `max_lr == min_lr` schedule bug where applicable.

- [ ] **Step 3: Record clean next steps**

State that clean R33 AR/AV hero proof requires patched code, remote-code patch report, packed-vs-padded check, and reruns of the best candidates.

### Task 6: Verify Final State

**Files:**
- Read-only verification over local and RunAI state.

- [ ] **Step 1: Verify no pending queue entries**

Run live-YAML queue status scan; backups may still contain old `pending` strings.

- [ ] **Step 2: Verify code archive checksum and remote marker**

Confirm RunAI contains the local sync marker and the S3 archive exists.

- [ ] **Step 3: Run lightweight tests**

Run dependency-light local tests and at least YAML parse checks for edited configs/docs if code changed.
