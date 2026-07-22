# Prompt: Nano30B NLA Super Audit And Optimization Review

You are an independent super auditor/verifier for the Nano30B Natural Language
Autoencoder pilot. Your job is to audit the entire codebase, docs, configs,
training/eval scripts, and run records from first principles. Do not assume the
current implementation is correct just because tests pass or prior runs looked
good. Do not rely on any prior optimization audit reports if you find them in
git history or local caches; use the primary code, primary docs, configs, eval
reports, and reference implementation files listed below.

Work in:

```bash
/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot
```

If you use RunAI, use the current local RunAI binary if available and do not
print secrets. Keep W&B offline. Do not launch long training, delete artifacts,
or start RL unless explicitly instructed. Prefer read-only inspection, bounded
tests, and short smoke/eval commands.

## Mission

Produce a hard correctness and performance audit that answers:

1. Is the Nano30B NLA implementation faithful to the Natural Language
   Autoencoder contract and the vendored reference implementation?
2. Are the Nano/Nemotron-H architecture adaptations correct for extraction,
   injection, AR loss, AV SFT, checkpoint conversion, and eval?
3. Are any training or eval metrics invalid because of leakage, packed-sequence
   contamination, LR schedule drift, split/doc overlap, prompt/sidecar mismatch,
   checkpoint-conversion bugs, or wrong controls?
4. What changes would improve correctness, throughput, memory use, checkpoint
   hygiene, and iteration speed?
5. What exact configs should be run next for clean R33 AR, clean R33 AV, and the
   final round-trip gate?

Treat the output as a review for a fresh agent who will continue the R33 hero
milestone.

## Core Scientific Background

The target task is a single-site Natural Language Autoencoder for:

```text
nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

Scientific contract:

```text
frozen target Nano: x, tau -> h_b = R_b^target(x)_tau
AV: h_b -> z
AR: z -> h_hat_b
loss/eval: h_hat_b should reconstruct h_b through the text bottleneck
```

The milestone is not complete until the actual round trip passes:

```text
h -> AV-generated explanation -> AR h_hat
```

Teacher-text AR metrics and AV real-vs-control losses are useful proxies, but
they are not proof of a successful NLA. Do not recommend RL until this
round-trip gate beats the mature R27 fallback.

Reference NLA naming:

- AV / actor: activation vector -> explanation text.
- AR / critic: explanation text -> activation vector.
- Released NLA convention normalizes vectors for direction-only comparison.
  Low normalized MSE means high cosine agreement. For normalized vectors,
  `MSE = 2 * (1 - cosine)`.

Reference implementation and paper-local docs:

- `external/REFERENCE_REPOS.md`
- `external/natural_language_autoencoders/README.md`
- `external/natural_language_autoencoders/docs/design.md`
- `external/natural_language_autoencoders/docs/inference.md`
- `external/natural_language_autoencoders/nla_inference.py`
- `external/natural_language_autoencoders/nla/`

Use these as the paper/reference baseline, but do not assume their Qwen/Gemma
wrapper assumptions apply directly to Nano.

## Nano30B Architecture Facts To Verify

Do not trust prose alone. Verify from loaded configs, module trees, and tests.
The current tracked Nano config snapshot is:

```text
runs/introspection/ar-r27-datagen-dryrun-20260528T230649Z/nano_tokenizer/config.json
```

Known working assumptions from that config:

```text
architectures: NemotronHForCausalLM
model_type: nemotron_h
hidden_size / d_model: 2688
num_hidden_layers: 52
num_attention_heads: 32
num_key_value_heads: 2
vocab_size: 131072
max_position_embeddings: 262144
num_experts_per_tok: 6
mamba_num_heads: 64
mamba_head_dim: 64
hybrid_override_pattern: MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
HF wrapper: .backbone
HF layers: .backbone.layers
HF final norm: .backbone.norm_f
HF embeddings: .backbone.embeddings
```

The `M` blocks are Mamba-style, `E` blocks are MoE, and `*` blocks are
attention/GQA. Audit all layer/boundary indexing carefully. R33 means boundary
R_33, not necessarily post-block 33. Off-by-one errors are high risk.

## Current Experiment Context

Start with these current-state docs:

- `README.md`
- `docs/current_state.md`
- `docs/runs/r33_gate_matrix.md`
- `docs/runs/r33_ar_hpo_202606.md`
- `docs/runs/r33_av_hpo_202606.md`
- `runs/registry/experiments.yaml`
- `docs/experiment_logbook.md`
- `docs/nano_av_run_history.md`
- `docs/nano_av_job_tracker.md`
- `docs/execution_log.md`
- `docs/nano30b-nla-core-plan.md`
- `docs/runai_miles_fsdp2_integration.md`
- `docs/architecture/audit_remediation_20260610.md`
- `docs/runbooks/runai_s3_sync.md`
- `docs/runbooks/runtime_monitoring.md`
- `docs/incidents/2026-06-longhorn-diskpressure.md`

Current decision state:

- R33 is the main scaling layer.
- R27 is the mature fallback.
- R34 AV was marginally better than R33 in a corrected 20k AV probe, but R34 AR
  was much weaker, so R33 remains the target.
- All pre-fix AR/AV training results are scouting evidence only because of
  confirmed packed-boundary contamination risk and LR schedule uncertainty.
- The R33 full275k AR checkpoint tree was deleted after preserving logs/eval
  evidence. Do not call it a clean hero checkpoint.

Key current artifacts:

```text
R33 dataset root:
/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_fullscan275396

Rows: 275,396
d_model: 2688
AR verifier: pass, empty explanations 0, nonfinite activations 0
AV verifier: pass, malformed responses 0, split/doc overlap 0

R33 critic init:
/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-critic-init/nano-ar-r33-critic-init
```

Important scouting results to verify from docs and, where possible, eval JSONs:

```text
R33 AR 20k:
validation/test teacher NMSE 0.381983 / 0.388301
validation/test source_raw NMSE 0.071066 / 0.076216

R34 AR 20k:
validation/test teacher NMSE 0.490728 / 0.501399

Mature R27 tuned AR fallback:
roughly validation/test teacher NMSE 0.441 / 0.437

R33 AV 20k corrected eval:
validation/test real NLL 1.040335 / 1.015130

R34 AV 20k corrected eval:
validation/test real NLL 1.037261 / 1.013677

R33 AR 100k pre-fix best scouting:
lr=5e-5, requested cosine, warmup=25, gb192/mb96
validation/test teacher NMSE 0.300924 / 0.292944

R33 AR full275k pre-fix scouting:
lr=5e-5, requested cosine, warmup=25, gb192/mb96
validation/test teacher NMSE 0.277565 / 0.276665
validation/test source_raw NMSE 0.096948 / 0.091568

R33 AV 20k LR smoke pre-fix scouting:
1e-6 real NLL 2.576588 / 2.562618
5e-6 real NLL 1.547347 / 1.531454
2e-5 real NLL 1.069623 / 1.044394
1e-4 real NLL 0.992138 / 0.971248
```

Use these only as context. Recompute or validate from source artifacts before
making strong claims.

## Known Correctness Issues Already Identified

Audit whether each is actually fixed, adequately tested, and enforced in
configs:

- Packed-boundary contamination in Nano/Nemotron-H training with packed
  microbatches. Check `seq_idx`, attention masks, sample boundaries, and
  packed-vs-padded equivalence.
- Miles LR schedule issue where requested cosine schedules could be neutered.
  Check final LR canaries and scheduler state.
- AV dynamic token cap mismatch: `max_tokens_per_gpu < max_sequence_tokens`
  must be explicit and verified.
- MoE router/mask plumbing and router fp32 preservation.
- Local-shard grad norm clipping vs old slow global clipping.
- Value-head gather optimization: only selected last-token rows should hit the
  value head and be stored when possible.
- Injection hook correctness, marker/neighbor checks, vectorization, and
  avoidance of accidental multi-marker updates.
- Data source copy behavior and activation-array copying overhead.
- Checkpoint conversion from DCP to HF, especially Nano remote-code safetensor
  layout and per-expert weight layout.
- W&B/system metrics logging and GPU/process memory telemetry.

Primary implementation files:

- `external/natural_language_autoencoders/nla/train_actor.py`
- `external/natural_language_autoencoders/nla/models.py`
- `external/natural_language_autoencoders/nla/loss.py`
- `external/natural_language_autoencoders/nla/injection.py`
- `external/natural_language_autoencoders/nla/data_source.py`
- `external/natural_language_autoencoders/nla/schema.py`
- `external/natural_language_autoencoders/nla/config.py`
- `external/natural_language_autoencoders/nla/arch_adapters.py`
- `external/natural_language_autoencoders/nla/audit_runtime.py`
- `external/natural_language_autoencoders/nla/remote_code_patches.py`
- `external/natural_language_autoencoders/nla/nemotron_moe.py`
- `external/natural_language_autoencoders/nla/system_metrics.py`
- `external/natural_language_autoencoders/tools/convert_fsdp_to_hf.py`
- `external/natural_language_autoencoders/nla/miles_patches/`

Primary Nano scripts:

- `scripts/nano_prefix_activation_extract.py`
- `scripts/nano_prefix_dataset_pipeline.sh`
- `scripts/nano_ar_r33_scaling_pipeline.sh`
- `scripts/nano_realdata_ar_build.py`
- `scripts/verify_nano_miles_ar_dataset.py`
- `scripts/verify_nano_miles_av_dataset.py`
- `scripts/nano_ar_hpo_queue.py`
- `scripts/nano_av_probe_queue.py`
- `scripts/nano_av_runner.py`
- `scripts/run_nano_av_miles_fsdp2_sft.sh`
- `scripts/eval_nano_ar_miles_checkpoint.py`
- `scripts/eval_nano_av_miles_checkpoint.py`
- `scripts/eval_nano_av_ar_roundtrip_gate.py`
- `scripts/nano_ar_hpo_study.py`
- `scripts/prune_nano_miles_checkpoints.py`

Important configs:

- `configs/README.md`
- `configs/nano_ar/hpo/r33_full275k_hero_queue.yaml`
- `configs/nano_ar/hpo/r33_full275k_lr5e5_cosine_warmup25_gb192_mb96.yaml`
- `configs/nano_ar/hpo/r33_100k_lr5e5_cosine_warmup25_gb192_mb96.yaml`
- `configs/nano_ar/hpo/r33_mini_probe_20k_lr2e5_cosine_128steps.yaml`
- `configs/nano_ar/hpo/r34_mini_probe_20k_lr2e5_cosine_128steps.yaml`
- `configs/nano_av/hpo/r33_20k_lr_smoke_queue.yaml`
- `configs/nano_av/hpo/r33_full275k_hero_queue.yaml`
- `configs/nano_av/hpo/r33_full275k_lr1e5_gb192_mb2_seq1152_dyn512.yaml`
- `configs/nano_av/layer_probe/r27_r34_av_20k_seq1152_queue.yaml`
- `configs/nano_av/layer_probe/r33_av_probe_20k_lr1e5_128steps_seq1152.yaml`
- `configs/nano_av/layer_probe/r34_av_probe_20k_lr1e5_128steps_seq1152.yaml`

## Required Audit Work

### 1. Reference-faithfulness audit

Compare the implementation against the vendored NLA reference:

- AV input-embedding injection contract.
- AR critic prompt/template and final-token value extraction.
- Sidecar metadata resolution and checkpoint sidecar precedence.
- `injection_scale`, `mse_scale`, normalization, normalized MSE, cosine, FVE/RRI
  interpretation.
- Dataset columns and split policy.
- DCP/HF conversion and inference/eval behavior.

Flag every difference as one of:

- intentional Nano adaptation,
- harmless implementation detail,
- likely correctness risk,
- confirmed bug.

### 2. Nano architecture and remote-code audit

Verify:

- `.backbone`, `.backbone.layers`, `.backbone.norm_f`, `.backbone.embeddings`
  adapter assumptions.
- Hidden-state boundary indexing, especially R27/R33/R34.
- Mamba, MoE, and attention mask behavior under packed training.
- Remote-code patcher coverage for checkpoint-local `modeling_nemotron_h.py`.
- Router fp32 behavior and router/load telemetry.
- MoE segmented implementation equivalence and possible grouped-GEMM speedups.
- Whether `NLA_PATCH_NEMOTRON_REMOTE_CODE=1` is enforced for clean runs.

### 3. Dataset and split audit

For AR and AV dataset builders/verifiers:

- Confirm that `doc_id`, token keys, prompt text, activation vectors, and
  explanations remain aligned.
- Confirm 275,396-row R33 dataset construction derives from the teacher-backed
  row/key table without generating new teacher explanations.
- Confirm split/doc overlap checks use doc-level split, not row-level leakage.
- Confirm prompt extraction from old R27 AR-SFT tables uses `<text>...</text>`
  consistently when deriving teacher keys.
- Confirm empty explanations, nonfinite activations, malformed responses, and
  sidecar mismatches fail hard.

### 4. Training-code audit

For AV and AR:

- Check FSDP/Miles launch arguments, sharding behavior, optimizer/scheduler
  state, exact-resume behavior, checkpoint pruning, and W&B offline logging.
- Verify LR schedules actually decay when requested.
- Verify microbatch/global batch/rollout batch semantics.
- Verify dynamic packed-token behavior for AV (`seq1152`, response cap,
  `max_tokens_per_gpu=512`) and whether it affects quality or contamination.
- Verify gradient clipping semantics and performance.
- Verify system metrics and router metrics are emitted without excessive
  overhead.
- Identify high-impact throughput bottlenecks with evidence.

### 5. Eval-code audit

For AR eval:

- Validate teacher, teacher_shuffled, blank, generic, mean, source_context, and
  source_raw controls.
- Validate train-only fitting of mean controls.
- Validate normalized metrics and rowwise win-rate computation.
- Validate bounded `512/512` eval sampling and reproducibility.

For AV eval:

- Validate real vs shuffled/zero/mean/no-injection controls.
- Validate DCP -> temporary HF -> eval -> cleanup path.
- Validate generation examples, parse/closure metrics, and prompt/template
  consistency.

For round trip:

- Validate `scripts/eval_nano_av_ar_roundtrip_gate.py`.
- Confirm it compares R33 against R27, includes AV parse/closure rate, and
  reports generated-text AR NMSE separately from teacher-text AR NMSE.

### 6. Config and next-run recommendations

Recommend improved clean configs for:

- fastest safe R33 AR smoke after fixes,
- R33 AR medium/hero confirmation,
- R33 AV LR confirmation,
- R33 AV medium/hero,
- R33 round-trip gate,
- optional layer sweep or boundary audit only if it can change the decision.

For each recommendation include:

- LR, warmup, schedule, global batch, microbatch, sequence length, token cap,
  save interval, eval interval, expected wall time, expected disk footprint, and
  what result would justify promotion.
- Which runs can use 1 GPU vs 2 GPUs.
- Which configs must not be used because they are stale, contaminated, or
  reproduce only historical behavior.
- Which checkpoints/logs should be preserved and which should be pruned.

## Suggested Verification Commands

Use these as starting points; adapt to the local or RunAI environment.

Local dependency-light tests:

```bash
python3 -m unittest \
  tests.test_nano_audit_remediation \
  tests.test_nano_ar_hpo_study \
  tests.test_nano_av_runner_validation \
  tests.test_nano_miles_launcher \
  -q
```

RunAI venv tests, if the workspace is available:

```bash
cd /workspace/interp/code/nano30b-nla-pilot-current
/workspace/interp/.venv/bin/python -m pytest \
  tests/test_nano_av_runner_spec.py \
  tests/test_nano_ar_hpo_queue.py \
  tests/test_nano_audit_remediation.py \
  tests/test_nano_av_runner_validation.py \
  tests/test_nano_miles_launcher.py \
  -q
```

Static checks:

```bash
git diff --check
python3 -m py_compile \
  external/natural_language_autoencoders/nla/audit_runtime.py \
  external/natural_language_autoencoders/nla/remote_code_patches.py \
  external/natural_language_autoencoders/nla/nemotron_moe.py \
  external/natural_language_autoencoders/nla/system_metrics.py \
  external/natural_language_autoencoders/nla/train_actor.py \
  external/natural_language_autoencoders/nla/models.py \
  external/natural_language_autoencoders/nla/loss.py \
  external/natural_language_autoencoders/nla/injection.py \
  external/natural_language_autoencoders/nla/data_source.py \
  scripts/nano_av_runner.py \
  scripts/nano_ar_hpo_study.py
```

Use `rg` heavily. When inspecting queue state, remember checked-in queue YAMLs
can lag remote state after launch. Prefer eval reports and run directories for
completed evidence.

## Deliverable

Write a new report. Do not use the deleted old audit filenames. Suggested path:

```text
docs/reviews/YYYY-MM-DD-nano-nla-super-audit.md
```

The report must include:

1. Executive verdict: clean/not clean for next R33 hero attempt.
2. Findings ordered by severity with file/line references, evidence, and impact.
3. Verification matrix: commands run, outputs, skipped checks, and why.
4. Correctness risks that remain unproven.
5. Training/eval optimization opportunities ranked by expected wall-time or
   memory impact.
6. Proposed clean R33 run plan with exact config changes or new config names.
7. Artifact/checkpoint cleanup guidance to avoid Longhorn disk pressure.
8. A short "what would invalidate this audit" section.

Be skeptical but constructive. A useful audit should be easy for the next agent
to execute: every major claim should point to a file, command, report, or
specific missing proof.
