# Audit Remediation 2026-06-10

This note records implementation status for the 2026-06 optimization audit
remediation pass. The original audit drafts were removed so future reviews start
from the primary code, configs, run records, and the fresh prompt under
`docs/prompts/`.

## Implemented

| Audit item | Implementation |
|---|---|
| C1 packed-boundary contamination | Runner guardrails reject silent packed AR-SFT. `NLAFSDPActor` can run SFT packed-vs-padded checks with `NLA_ASSERT_PACKED_EQUIV`. `nla.remote_code_patches` patches copied `modeling_nemotron_h.py` files to plumb `seq_idx` and `attention_mask` through known Nemotron-H patterns. |
| C1 duplicate-doc split leakage | `scripts/nano_dedup_teacher_keys.py` builds a deduplicated teacher-key table by first-300-token `token_ids_prefix` hash, keeping one doc per content group. `verify_nano_miles_ar_dataset.py` and `verify_nano_miles_av_dataset.py` now report duplicate content groups and fail on content-hash cross-split overlap. |
| C2 patcher default-on mutation | `NLACriticModel` and `prepare_critic_checkpoint.py` now copy checkpoint-local remote code verbatim. The Miles actor patch no longer calls `patch_nemotron_h_checkpoint_dir`, and `scripts/nano_av_runner.py` no longer exports `NLA_PATCH_NEMOTRON_REMOTE_CODE=1`; patching is an explicit preflight step only. |
| C2 patch reproducibility | Miles patch hunk headers were recounted, and `scripts/check_miles_patches.py` validates local hunk counts plus optionally applies all patches to a supplied pinned Miles checkout. |
| C3 round-trip gate | `eval_nano_av_ar_roundtrip_gate.py` now uses a positive default control margin, requires all controls, strips generated explanation tags before AR scoring, reports generation parse/closure stats, and checks baseline row identity plus rowwise baseline win rate. |
| N1 LR schedule neutering | Patch `0001` no longer forces `max_lr == min_lr`. `nano_ar_hpo_study.py` now records final LR and emits an LR-decay canary for non-constant schedules. |
| H1 queue status drift | Queue status constants are shared in `scripts/nano_queue_status.py`; AR/AV queues accept historical terminal states and AV has `--reset-active` parity. The full275k R33 AR queue item is recorded as completed pre-fix scouting evidence with checkpoint removed, not as cancelled. |
| H3 LR canary hard gate | `nano_ar_hpo_study.assert_lr_decay_canary_for_run()` fails non-constant schedules with flat, missing, or invalid final LR evidence. Both AR and AV queues call it after training and before eval/conversion. |
| H4 AV eval cleanup | `nano_av_probe_queue.py` removes temporary converted HF checkpoints in a `finally` block when cleanup is enabled, including failed conversion/eval paths, and can record AV HPO trials to a shared JSONL. |
| H5 split materialization cache | `nano_av_runner.py` can cache deterministic materialized train/validation/test splits behind `dataset.cache_materialized_splits`, keyed by source file signature, split fractions, seed, row limit, and final-batch padding policy. Cached manifests are rewritten to the current run directory before launch. |
| H6 materialized split verification | `dataset.verify_materialized_splits` runs the content-hash materialized-split verifier during `prepare_run`, writes `split_content_verify.json`, and fails before GPU launch on doc/content leakage across train/validation/test. |
| H7 bf16 AV eval conversion | The DCP-to-HF converter accepts `--torch-dtype bfloat16`; the AV queue can pass it via `converted_hf_dtype`, and the clean R33 AV queue defaults to bf16 temporary HF eval checkpoints. |
| C2 router balance telemetry | `RouterEntropyTracker` adds optional per-step router entropy, active expert count, and max expert fraction metrics through the W&B log path when `NLA_ROUTER_METRICS=1`. |
| C3 dynamic token inconsistency | `nano_av_runner.py` rejects `max_tokens_per_gpu < max_sequence_tokens` unless explicitly acknowledged with `allow_oversized_dynamic_batch`. |
| M1 norm/metric consistency | `nla_critic_loss` now logs cosine sum, pred/gold norm ratio, and value-head weight norm for direction-only MSE monitoring. |
| N2 MoE loop throughput | Added `nla.nemotron_moe.segmented_moe` and remote-code patching support to replace the per-expert `where` loop in known Nemotron-H `moe` methods; the helper restores the caller hidden-state dtype before returning. |
| E1 grad norm throughput | Patch `0005` uses `clip_grad_norm_local_shards` by default: local DTensor shard norms plus one scalar all-reduce, preserving clipping semantics. `--no-nla-local-grad-norm` keeps the old global path for debugging. |
| E2/E4 value-head gather | `NLACriticModel.forward(..., nla_value_indices=...)` gathers last-token hidden rows before `value_head`; `NLAFSDPActor` uses that path for critic SFT and stores only selected backbone rows. |
| E3 actor sync | The actor microbatch `cuda.synchronize()` is gated by embedding size and `NLA_SYNC_MICROBATCH`, so Nano can skip the Gemma-specific memory bound by default. |
| E5 injection hook | Injection now vectorizes valid marker/neighbor detection and assignment, removing the per-match Python loop. |
| E6 sample deepcopy | `NLADataSource` now shallow-copies samples and metadata for rollout emission instead of deep-copying activation arrays and prompt payloads. |
| N3 router fp32 | Patch `0003` restores router tensors/buffers to fp32 after the bf16 cast, and the remote-code patcher injects `_nla_keep_router_buffers_fp32` for checkpoint-local code. |
| N4 dead mask plumbing | The remote-code patcher forwards `attention_mask` through known block/mixer/attention call patterns, making mask-based C1 fixes effective where the source matches. |
| N5 timing debug | Runner rejects `timing_debug: true` for complete-performance configs; checked-in hero configs keep it disabled. |

## Operational Notes

- The remote-code fixes are **not** applied on model load/save or actor init.
  RunAI must run the explicit patcher preflight against checkpoint-local
  `modeling_nemotron_h.py` before new Nano jobs launch.
- The model-code patcher is pattern-based because the Nemotron-H source is
  checkpoint-local remote code, not a normal repo module. Treat the generated
  `modeling_nemotron_h.py.nla_patch_report.json` as required launch evidence.
- The segmented MoE helper removes the per-expert `where` scans but still loops
  over experts to preserve unused-parameter graph behavior. A grouped-GEMM MoE
  fork remains a possible later speedup after equivalence tests.
- `grad_norm_policy: clip` now maps to faithful local-shard clipping. Use
  `grad_norm_policy: global_clip` only when comparing against the old slow path,
  and `skip_diagnostic` only for diagnostics.
- The materialized split cache is opt-in and stores only deterministic split
  parquets/sidecars under the configured cache root. The run directory still
  receives its own copied split files and manifest, so evidence archives remain
  self-contained.

## Impact On Existing Runs

- Pre-fix AR/AV checkpoints are contaminated-training artifacts if they used
  packed Nano/Nemotron-H microbatches. Keep their heldout evals as scouting
  signal, not as clean hero proof.
- Pre-fix runs labeled `cosine` may also be affected by the Miles LR schedule
  issue. Treat those labels as requested configs unless the run has LR canary or
  final-LR evidence.
- The R33 full275k AR run
  `nano-ar-r33-full275k-lr5e5-cosine-warmup25-gb192-mb96` was reclassified as
  scouting evidence; its checkpoint tree was deleted on RunAI after preserving
  logs/eval/W&B metadata in
  `artifacts/runai_sync/20260610T234644Z/` and S3
  `sync_exports/20260610T234644Z/`.
- The next R33 hero attempt must be a fresh post-fix run with:
  checkpoint-local remote-code patch report, packed-vs-padded agreement or
  equivalent preflight, LR-decay canary evidence, and bounded heldout evals.
- The old `r33_prefix_fullscan275396` dataset is retained only as pre-fix
  provenance. The clean target is the deduplicated teacher-key set expected at
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_dedup_fullscan132996`,
  followed by fresh R33 activation extraction, AR/AV sidecar generation, and
  upgraded verifier reports.

## Verification

Local dependency-light checks:

```bash
python3 -m unittest tests.test_nano_audit_remediation tests.test_nano_ar_hpo_study tests.test_nano_av_runner_validation tests.test_nano_miles_launcher tests.test_nano_miles_av_dataset tests.test_nano_miles_ar_dataset tests.test_nano_dedup_teacher_keys -q
python3 -m unittest tests.test_nano_av_runner_spec.NanoAVRunnerSpecTests.test_prepare_run_can_reuse_cached_materialized_splits_with_rewritten_paths tests.test_nano_av_runner_spec.NanoAVRunnerSpecTests.test_prepare_run_fails_when_materialized_split_content_verification_finds_overlap tests.test_nano_av_probe_queue -q
python3 scripts/check_miles_patches.py
python3 -m py_compile external/natural_language_autoencoders/nla/audit_runtime.py external/natural_language_autoencoders/nla/remote_code_patches.py external/natural_language_autoencoders/nla/nemotron_moe.py external/natural_language_autoencoders/nla/system_metrics.py external/natural_language_autoencoders/nla/train_actor.py external/natural_language_autoencoders/nla/models.py external/natural_language_autoencoders/nla/loss.py external/natural_language_autoencoders/nla/injection.py external/natural_language_autoencoders/nla/data_source.py external/natural_language_autoencoders/tools/convert_fsdp_to_hf.py scripts/nano_av_runner.py scripts/nano_av_probe_queue.py scripts/nano_ar_hpo_study.py scripts/nano_dedup_teacher_keys.py scripts/check_miles_patches.py
```

RunAI-only validation still required before hero promotion:

- Apply Miles patches against the pinned checkout with
  `python scripts/check_miles_patches.py --miles-root /workspace/interp/code/miles-051cd15`
  and run the full test suite in `/workspace/interp/.venv/bin/python`.
- Prepare or patch a Nano checkpoint directory and inspect
  `modeling_nemotron_h.py.nla_patch_report.json`.
- Run the packed-vs-padded critic preflight with `NLA_ASSERT_PACKED_EQUIV=1`.
- Run a short AR/AV smoke with router metrics enabled and confirm no loss,
  throughput, or memory regression.
