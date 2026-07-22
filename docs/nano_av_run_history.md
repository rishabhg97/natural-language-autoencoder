# Nano30B NLA AV Run History

<!-- R33-HERO-BASELINE-PROTOCOL-INVALIDATED -->

> [!CAUTION]
> Publication status (`2026-07-16`): the corrected family-clean R33 SFT AV+AR
> pair is qualified for directional reconstruction and stored-snapshot
> functional recovery, and its validation signal passes through an independently
> initialized seed-`314159` AR. The archived `30.97% / 32.34%` RL comparison
> remains invalidated. See `docs/runs/r33_clean_sft_av_ar_20260715.md`.

This file records completed, failed, or stopped Nano30B AV work. The live queue is in `docs/nano_av_job_tracker.md`.

## Current Focus

The current milestone is the selected family-clean R33 AV and AR SFT
`iter_0001291` pair. Both component verifiers pass, the protocol-matched
AV-generated-text round trip passes on `512/512` validation/test rows, and the
stored-snapshot functional verifier passes. The pair manifest is qualified and
both model payloads are preserved on S3.

The independent AR replication, selected-run compute accounting, static
release audit, and deterministic no-weights candidate archive are also
complete. Human semantic review and legal/license approval remain pending.

Earlier packed AV/RL actor runs remain contaminated by the dropped
`position_ids` bug. The July 8 RL actor remains invalidated by its mixed
baseline protocol and is not the selected checkpoint. The canonical current
record is `docs/current_state.md`.

A bounded clean online joint HPO is active on 2026-07-17 from this qualified
pair. Its corrected critic repartition makes actor and critic sample accounting
explicit, and its paired 128-row evaluation is validation-only. This is an
experiment in progress, not a replacement checkpoint or an extension of the
qualified SFT claim. Queue and design details are in
`docs/runs/r33_online_joint_hpo8_20260717.md`.

## Qualified: Family-Clean R33 SFT AV+AR Pair (2026-07-15; Follow-Up 2026-07-16)

- AR: `iter_0001291`, SHA-256
  `5e792120ec1a00ebb4cf4abca50d2a6a962421ac4f45423479ae5061f4d2d760`
- AV: `iter_0001291`, model fingerprint
  `dcp_model_sha256:43346232d2fc043260ee903191e20cce07801903e1e7b7956f16022eb463386a`
- AR component validation: teacher directional MSE `0.281703`, cosine
  `0.859148`, FVE-NRM `0.584534`; raw centered R2 `-0.201696`
- AV component validation: real NLL `0.776775`, versus shuffled `1.311727`,
  zero `1.176494`, mean `1.237522`, and none `1.220974`
- round trip: validation/test directional MSE `0.307004 / 0.319225`, versus
  teacher `0.304714 / 0.302637`; closed/usable `100% / 100%`
- controls: all family-clustered lower bounds positive; test rowwise wins
  `99.61-100%`
- functional: candidate is teacher-level within family uncertainty and beats
  mean, zero, and shuffled under stored-snapshot counterfactual reinjection
- release: `r33-clean-sft-av-ar-iter1291-20260715`, pair manifest
  `qualified: true`
- evidence: `artifacts/runai_eval/r33-clean-sft-av-ar-qualified-20260715/`
- limitation: no raw-magnitude, exact fresh-forward, pristine historical test,
  R33-over-R27, independent AV-seed, external-generalization, or RL claim

## Publication Follow-Up (2026-07-16)

- independent AR: seed `314159`, 1,291 updates on 4 H100-NVL GPUs; component
  teacher directional MSE/cosine/FVE-NRM
  `0.286169 / 0.856916 / 0.577948`
- independent cross-critic gate: frozen selected-AV validation text
  directional MSE `0.310963`, teacher `0.308533`; 512 rows/250 families,
  `100%` closed and usable, all five controls passed, minimum rowwise win
  fraction `0.998047`
- independent checkpoint: 10 HF files, `38,462,226,688` bytes, directory
  SHA-256
  `c2eea74f5baccee97128617b05636187804c7e59aedc560d088dbf65d52f1925`;
  verified in internal S3 before redundant DCP/optimizer cleanup
- selected compute: primary AR `15.5467`, primary AV `107.6867`, independent
  AR `15.5122` H100-NVL GPU-hours; total `138.7456`
- compact curves: all 3,873 optimizer steps, SHA-256
  `7d9c22b989c594e546ec08648d0319c37caad69ad502d2badb635d41706c42a6`
- release candidate: 496 files, no weights, security gate passed; archive
  `6,859,370` bytes, SHA-256
  `3eb8e64ed0d9d61ed2d6b0694fbaf96b99051a63f2ce1a6c99372d93832e573a`
- audited/archive tree SHA-256:
  `df175c5f61cefbfc1a02451a7bd242ba69e1cb602cdd97ca4b8bd8fe9c263b77`
- internal S3: five release-candidate objects preserved under
  `publication/release-candidates/r33-clean-sft-av-ar-iter1291-20260716/`
- verification: full local suite `749 passed`; documentation consistency pass
- remaining gates: two blinded human reviewers, repository/base-model and
  teacher-API legal approval, and final notices; external data is needed only
  for a confirmatory generalization claim

## Clean Online Joint AV+AR Canary (2026-07-17)

- run: `r33-family-clean-online-joint-canary-update2-8h100-retry4`
- initialization: qualified family-clean R33 AV and AR SFT checkpoints
- topology: 8 H100 NVLs, partitioned as `4 actor / 3 critic / 1 rollout`
- training: two online actor+critic optimizer updates, each using 3 prompts x
  8 samples; actor/critic LR `1e-5 / 5e-6`; W&B offline
- systems result: both update-2 DCP checkpoints committed; step-0 critic
  reward/train equivalence was exact; no OOM or optimizer failure
- exact comparison: 64 family-stratified rows and 64 independent families,
  with matching stable row keys, dataset hashes, tokenizer, protocol, and five
  controls between online and clean SFT
- directional MSE: online `0.291993`, SFT `0.292173`; relative improvement
  `0.0618%`; paired wins `32/64`
- raw MSE: online `8.969927`, SFT `8.797533`; online is `1.96%` worse
- inference: directional family-bootstrap 95% CI
  `[-0.007704, 0.008597]`; one-sided sign-flip `p=0.4824`
- decision: strict promotion failed (`baseline_beaten: false`); retain this as
  online joint-training systems evidence, not as an improved NLA checkpoint
- evidence: `docs/runs/r33_online_joint_canary_20260717.md` and
  `artifacts/runai_eval/r33-online-joint-canary-evidence-20260717T0951Z/`

## Post-FSDP2 Run History

### completed: ar-sft-qwen-faithful-runner-dataset-and-eval-gates

- date_utc: `2026-05-28`
- class: implementation gate
- result: pass
- scope: RunAI `train-dev`

What changed:

- Added `training.objective: ar_sft` support to `scripts/nano_av_runner.py`.
- Added Nano AR configs under `configs/nano_ar/`.
- Added `scripts/verify_nano_miles_ar_dataset.py` for critic prompt, vector,
  sidecar, and doc-split contract checks.
- Added `scripts/eval_nano_ar_miles_checkpoint.py` for heldout reconstruction
  metrics against teacher, shuffled teacher, blank, generic, mean, and
  source-context/source-raw controls when present.
- Updated `NLACriticModel` and architecture adapters for Nano/Nemotron-H
  `.backbone` modules, `norm_f`, truncated `hybrid_override_pattern`, and
  Transformers 5.x save compatibility.
- Updated critic checkpoint save/export to preserve Nano remote-code files so
  HF checkpoints reload without manual repair.

Verification:

- AR dataset verifier passed on `99,570` rows, `d_model=2688`, finite
  activations, nonempty explanations, critic prompt suffix checks, and zero doc
  overlap for both `80/10/10` and `90/5/5`.
- Focused local non-torch tests: `29 passed`.
- Remote critic-architecture tests: `6 passed`.
- Remote eval metric tests: `2 passed`.
- Full local `pytest tests` was not run because the local temp test env lacks
  `torch`; the RunAI venv was used for torch/transformers-specific tests.

### completed: ar-sft-small-smoke-save-and-resume

- date_utc: `2026-05-28`
- experiment_class: `small-smoke`
- backend: Miles FSDP2
- rows: `96`
- split: doc-level `80/10/10`
- result: pass

Run:

- run_dir: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-small/nano-ar-miles-fsdp2-r27-small-smoke-save1-20260528T184247Z`
- checkpoint: `checkpoints/iter_0000001`
- checkpoint size: about `93G` exact-resume DCP payload plus about `31G` HF export
- train step 0: loss `1.419609`, FVE-vs-train-mean `-1.185470`

Resume:

- run_dir: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-small/nano-ar-miles-fsdp2-r27-resume-smoke-20260528T185445Z`
- loaded model, optimizer, LR scheduler, and rollout state from the checkpoint
  root
- advanced to step 1 with loss `1.324306`

Limitations:

- One-step heldout eval was intentionally not treated as success: teacher text
  did not beat the mean h baseline after one update.

### completed: ar-sft-medium-small-control-gate

- date_utc: `2026-05-28`
- experiment_class: `medium-small`
- backend: Miles FSDP2
- rows: `960`
- split: doc-level `80/10/10`
- result: pass after continuation

Rejected intermediates:

- One-epoch medium-small produced row-specific text signal but did not beat mean h.
- Five-epoch medium-small improved further but still did not beat mean h:
  validation teacher normalized MSE `0.902374` vs mean `0.875942`; test teacher
  `0.904764` vs mean `0.880934`.

Passing continuation:

- run_dir: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-medium/nano-ar-miles-fsdp2-r27-medium-small-continue-5ep-20260528T195213Z`
- init: HF checkpoint from the five-epoch medium run
- checkpoint: `checkpoints/iter_0000045`
- eval: `eval_iter_0000045_v128_t128_report.json`
- validation rows: `99`; test rows: `90`
- skipped control: `source_raw` because usable raw token IDs are not present

Validation normalized MSE:

- teacher: `0.795473`
- mean: `0.875942`
- source_context: `0.867701`
- teacher_shuffled: `0.988657`
- generic: `1.017015`
- blank: `1.029558`

Test normalized MSE:

- teacher: `0.795441`
- mean: `0.880934`
- source_context: `0.851662`
- teacher_shuffled: `0.995219`
- generic: `1.008652`
- blank: `1.016153`

Interpretation:

- The AR critic path has heldout row-specific reconstruction signal against
  every available requested control, including mean h.
- This is still a medium-small result, not the complete-performance 100k AR run.
- No RL run was launched.

### completed: ar-sft-complete-performance-fullscan-275k

- date_utc: `2026-05-30`
- experiment_class: `complete-performance`
- backend: Miles FSDP2
- status: completed; final checkpoint saved and bounded heldout eval passed
- config: `configs/nano_ar/fullscan_r27_miles_fsdp2.yaml`
- run_dir: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-fullscan/nano-ar-miles-fsdp2-r27-fullscan-275k-gb192-mb8-lr1e5-20260530T172928Z`
- dataset: `/workspace/interp/outputs/nano30b-nla-pilot/ar-r27-r30-fullscan-20260528T234403Z/R_27/ar_sft.parquet`
- verifier: `/workspace/interp/outputs/nano30b-nla-pilot/ar-r27-r30-fullscan-20260528T234403Z/R_27/ar_dataset_verify_runai.json`
- rows: `275,396`
- split: doc-level `90/5/5`
- train/validation/test rows: `247870/13761/13765`
- padded train rows: `247872`
- optimizer steps: `1291`
- batch: global `192`, micro `8`, rollout `192`
- lr: `1e-5`
- checkpoint plan: exact-resume final save at `iter_0001291`, `NLA_KEEP_LOCAL=1`
- final checkpoint: `checkpoints/iter_0001291`, about `123G`
- eval report: `eval_iter_0001291_v512_t512_report.json`
- W&B: offline for the wrapper, rollout manager, and actor runs
- first observed train metric: step `2`, loss `1.1862951914469402`,
  `fve_nrm=-0.7661219437917074`, grad norm `4.5625`
- final observed train metric: step `1290`, loss `0.4939449628194173`,
  `fve_nrm=0.2646290063858032`, grad norm `0.6484375`

Validation normalized MSE, `512` rows:

- teacher: `0.511466`
- mean: `0.886695`
- source_context: `0.632444`
- source_raw: `0.141013`
- teacher_shuffled: `1.153667`
- blank: `1.028922`
- generic: `1.033204`

Test normalized MSE, `512` rows:

- teacher: `0.505286`
- mean: `0.865295`
- source_context: `0.613429`
- source_raw: `0.137070`
- teacher_shuffled: `1.135018`
- blank: `1.010650`
- generic: `1.014764`

Verification before launch:

- The fullscan local handoff verified both R27 and R30 with `275,396` rows,
  `d_model=2688`, zero non-finite activation values, zero empty explanations,
  zero critic suffix failures, and zero document split overlap.
- The R27 parquet was uploaded to RunAI and sha256 matched the local artifact.
- The RunAI verifier passed on the uploaded R27 parquet and reported the expected
  `90/5/5` split counts.

Interpretation:

- This is the first complete-performance AR-SFT attempt at approximately Qwen AR
  data scale. Heldout teacher reconstruction beats mean, shuffled, blank,
  generic, and source-context controls on both validation and test. `source_raw`
  is stronger than teacher because it uses the raw token prefix/control context.
- This is not an NLL eval; AR-SFT uses the normalized MSE critic objective.
- No RL run was launched.

### launched: ar-sft-fullscan-v2048-winrate-eval-and-hpo-prep

- date_utc: `2026-05-31`
- class: eval/tuning prep
- status: v2048 eval complete; first HPO continuation probe running
- checkpoint under eval: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-fullscan/nano-ar-miles-fsdp2-r27-fullscan-275k-gb192-mb8-lr1e5-20260530T172928Z/checkpoints/iter_0001291`
- eval report target: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-fullscan/nano-ar-miles-fsdp2-r27-fullscan-275k-gb192-mb8-lr1e5-20260530T172928Z/eval_iter_0001291_v2048_t2048_winrates_report.json`
- eval controls: `teacher`, `teacher_shuffled`, `blank`, `generic`, `mean`,
  `source_context`, `source_raw`
- eval sample: validation `2048`, test `2048`
- local commit: `e8f824e` added AR rowwise win-rate reporting and LR schedule
  knobs for Miles launcher tuning

Result:

- validation teacher normalized MSE `0.503023`, cosine `0.748488`, FVE vs mean
  `0.416190`; rowwise teacher wins: mean `98.05%`, shuffled `99.90%`, blank
  `98.73%`, generic `98.93%`, source-context `77.54%`; source-raw wins over
  teacher `99.02%`
- test teacher normalized MSE `0.514389`, cosine `0.742806`, FVE vs mean
  `0.401452`; rowwise teacher wins: mean `97.46%`, shuffled `99.85%`, blank
  `98.10%`, generic `98.34%`, source-context `74.56%`; source-raw wins over
  teacher `99.41%`

Follow-up HPO result:

- tuning probe:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-hpo/nano-ar-r27-fullscan-continue-qwen-lr2e5-cosine-256steps-20260601T0019Z`
- config: `configs/nano_ar/hpo/r27_fullscan_continue_qwen_lr2e5_cosine_256steps.yaml`
- launch UTC: `2026-06-01T00:21Z`, PID `3259226`
- checkpoint: `checkpoints/iter_0001547`
- schedule: `lr=2e-5`, `min_lr=2e-6`, cosine decay, `50` warmup iters,
  `--finetune`, global batch `192`
- caveat: Miles `--finetune` restarted the rollout counter, so the nominal
  `256`-step probe ran as a longer `1547`-rollout fresh-schedule continuation
- v2048 validation teacher normalized MSE `0.436878`, cosine `0.781561`, FVE vs
  mean `0.492958`, teacher beats mean rowwise `98.54%`
- v2048 test teacher normalized MSE `0.450516`, cosine `0.774742`, FVE vs mean
  `0.475775`, teacher beats mean rowwise `97.71%`

Prepared tuning probes:

- `configs/nano_ar/hpo/r27_best1547_continue_lr1e5_cosine_256steps.yaml`
- `configs/nano_ar/hpo/r27_best1547_continue_lr5e6_cosine_256steps.yaml`
- `configs/nano_ar/hpo/r27_fullscan_continue_lr5e6_256steps.yaml`
- `configs/nano_ar/hpo/r27_fullscan_continue_qwen_lr2e5_cosine_256steps.yaml`
- `configs/nano_ar/hpo/r27_fullscan_continue_qwen_batch256_lr2e5_cosine_256steps.yaml`

Interpretation:

- The first HPO checkpoint has positive heldout improvement and is the current
  AR best, but it is not Qwen-level and not yet the final AR milestone. The next
  target is teacher normalized MSE `0.25-0.30` using bounded current-best
  continuation probes, without new data or RL. These probes are not final
  milestones because they use model/HF-only saves; a winning recipe should be
  rerun with exact-resume state.

### superseded: ar-sft-complete-performance-100k

- date_utc: `2026-05-28`
- experiment_class: `complete-performance`
- status: plan materialized, not launched; superseded by the 275,396-row
  fullscan R27 run
- plan_dir: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-hero/nano-ar-miles-fsdp2-r27-super-thinking-100k-hero-plan-20260528T202156Z`
- plan_file: `hero_run_plan.json`
- rows: `99,570`
- split: doc-level `90/5/5`
- train/validation/test rows: `89618/4978/4974`
- padded train rows: `89664`
- optimizer steps: `467`
- batch: global `192`, micro `8`, rollout `192`
- checkpoint plan: exact-resume final save at `iter_0000467`, `NLA_KEEP_LOCAL=1`
- eval plan: bounded validation/test reconstruction controls after final save

Decision:

- The plan is ready to launch as AR-SFT only. Do not start RL from it.

### completed: miles-fsdp2-import-and-contract-gates

- date_utc: `2026-05-27`
- class: implementation gate
- result: pass
- scope: RunAI `train-dev`

What changed:

- Made the vendored Qwen NLA stack importable with Miles on RunAI.
- Added dataset verification for Nano AV parquet contract.
- Added batched injection tests around `nla.injection.inject_at_marked_positions`.
- Added runner/config tests for the Miles/FSDP2 launcher path.

Verification:

- `import miles`
- `from nla.train_actor import NLAFSDPActor`
- `from nla.rollout.sft_actor import generate_rollout`
- `from nla.injection import inject_at_marked_positions`
- Local tests reached `30 passed, 48 subtests passed` before launch work.

### completed: small-smoke-miles-fsdp2-one-step-and-checkpoint

- date_utc: `2026-05-27`
- experiment_class: `small-smoke`
- backend: Miles FSDP2
- rows: `96`
- split: doc-level `80/10/10`
- result: pass

Outcomes:

- First true Miles/FSDP2 AV-SFT training path completed.
- Batched activation injection ran through the Qwen hook path.
- Checkpoint save succeeded.
- W&B offline artifacts were written.

Limitations:

- Scientific claims were intentionally withheld; this was a path, checkpoint, and stability gate.

### completed: medium-small-miles-fsdp2-training-gate

- date_utc: `2026-05-27`
- experiment_class: `medium-small`
- backend: Miles FSDP2
- rows: `960`
- split: doc-level `80/10/10`
- result: pass

Outcomes:

- Medium-small training ran through the real Miles/FSDP2 path.
- Checkpointing worked.
- Throughput was materially better than the sequential smoke harness once warmup was excluded.
- The run produced a checkpoint suitable for real-vs-control evaluation.

Limitations:

- The directional medium-small result is not directly comparable to full 100k training because it uses only 960 rows and short training.

### completed: medium-small-checkpoint-eval-real-vs-controls

- date_utc: `2026-05-27`
- experiment_class: `medium-small`
- checkpoint: medium-small final checkpoint
- eval sizes: `32` validation rows and `32` test rows
- result: pass, directional only

Validation NLL:

- real: `1.8323`
- shuffled: `1.9581`, gap `0.1258`
- zero: `1.9196`, gap `0.0873`
- mean: `1.9478`, gap `0.1155`
- no-injection: `2.0705`, gap `0.2382`

Test NLL:

- real: `1.7781`
- shuffled: `1.8741`, gap `0.0960`
- zero: `1.8412`, gap `0.0631`
- mean: `1.8725`, gap `0.0943`
- no-injection: `1.9756`, gap `0.1975`

Interpretation:

- This passed the row-specificity direction gate on both heldout splits.
- It did not reach the hero target of `>=0.30` real-vs-control gaps, and the small eval sample is not a complete-performance result.
- Generation parse/F1 was not run for this checkpoint.

### completed: full-optimizer-checkpoint-resume-smoke

- date_utc: `2026-05-27`
- experiment_class: `small-smoke`
- backend: Miles FSDP2
- result: pass

Outcomes:

- Full optimizer checkpoint payloads were saved and reloaded.
- Step count advanced after resume.
- W&B offline logging remained usable.
- This cleared the checkpoint/resume requirement before attempting full-data training.

### completed: fsdp2-batch-scaling-with-faithful-grad-norm

- date_utc: `2026-05-27`
- experiment_class: throughput diagnostic
- backend: Miles FSDP2
- result: pass with selected configuration `gb192/mb8`

Observed configurations:

| Global batch | Micro batch | Outcome | Notes |
| --- | --- | --- | --- |
| `96` | `8` | pass | about `157.2s/step`, about `0.61 examples/s` on the first measured step |
| `192` | `8` | pass | first step about `197.8s`; second-step diagnostic about `89.4s`, about `2.15 examples/s` after warmup |
| `384` | `8` | first step pass, hero failed on second step | OOM after Adam state allocation |

Interpretation:

- `gb384` has attractive one-step throughput but is not stable after optimizer state allocation.
- `gb192/mb8` is the current largest known stable full-data configuration.
- Faithful grad norm remains enabled; skipping it would change optimization behavior and is not the selected path.

### completed: artifact-cleanup-after-storage-expansion

- date_utc: `2026-05-27`
- result: pass
- storage: `/workspace/interp` expanded to about `1Ti`

Removed:

- stale medium model-only checkpoint payloads
- converted medium HF payloads no longer needed for active gates
- small-smoke full-optimizer checkpoint payload after resume was verified

Kept:

- reports
- logs
- active hero output directory
- code/config/docs

Latest observed free space:

- about `894G` free on `/workspace/interp`

### failed: hero-gb384-second-step-oom

- date_utc: `2026-05-27`
- experiment_class: `complete-performance`
- backend: Miles FSDP2
- global_batch_size: `384`
- micro_batch_size: `8`
- result: failed, CUDA OOM
- run_dir: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-hero/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gb384-mb8-save100-20260527T2035Z`

Evidence:

- Step 0 completed.
- Step 1 failed in backward after optimizer state had been allocated.
- Step 0 loss: `2.0007405`
- Step 0 grad_norm: `14.6875`
- Step 0 step_time: `212.1566s`

Decision:

- Do not use `gb384` for full-data hero until memory strategy changes.
- Continue with `gb192/mb8`.

### completed: gb192-two-step-full-data-diagnostic

- date_utc: `2026-05-27`
- experiment_class: throughput diagnostic
- backend: Miles FSDP2
- global_batch_size: `192`
- micro_batch_size: `8`
- result: pass
- run_dir: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-batch-scaling/nano-av-miles-fsdp2-medium-gradnorm-gb192-twostep-nosave-20260527T2040Z`

Evidence:

- Step 0 loss: `2.0214923`
- Step 0 grad_norm: `14.4375`
- Step 0 step_time: `198.7763s`
- Step 1 loss: `1.7771916`
- Step 1 grad_norm: `7.5625`
- Step 1 step_time: `89.4263s`
- peak reserved memory: about `137.3 GiB` per H200

Decision:

- Launch full-data hero at `gb192/mb8/save100`.

### failed: hero-gb192-save100-first-checkpoint

- date_utc: `2026-05-27`
- experiment_class: `complete-performance`
- backend: Miles FSDP2
- status: failed during first checkpoint save
- run_dir: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-hero/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gb192-mb8-save100-20260527T2050Z`
- first checkpoint target: `iter_0000100`
- latest completed train metric: step `99`, loss `1.0409398078918457`, grad_norm `1.2890625`
- process state at `2026-05-27T23:03:58Z`: exited; GPUs idle
- checkpoint payload: incomplete; `iter_0000100` exists but contains `0` files and total checkpoint tree size is about `28K`

Failure:

- `RuntimeError: NCCL Error 1: unhandled cuda error` inside `NLAFSDPActor.save_model()` via Miles FSDP checkpoint `dcp.save(...)`.
- The training path itself reached the first save boundary with stable loss/grad norm, but checkpointing failed before a usable artifact was written.

Decision:

- Do not count this as a checkpointed hero run.
- Do not run checkpoint eval or resume smoke from this incomplete payload.
- Next gate is a focused checkpoint-save remediation using the same Nano30B full-optimizer checkpoint path.

### completed: checkpoint-gloo-save-resume-remediation

- date_utc: `2026-05-27`
- experiment_class: checkpoint gate
- backend: Miles FSDP2
- result: pass

Patch:

- `external/natural_language_autoencoders/nla/miles_patches/0006_fsdp_checkpoint_gloo_pg.patch`
- Keeps FSDP training collectives on NCCL but passes a cached Gloo process group
  to `torch.distributed.checkpoint` save/load calls for metadata/object
  collectives.

Save smoke:

- run_dir: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-checkpoint-debug/nano-av-miles-fsdp2-checkpoint-gloo-fulloptim-gb8-mb4-save1-tokenized-20260527T232920Z`
- config: `ROW_LIMIT=8`, `GLOBAL_BATCH_SIZE=8`, `MICRO_BATCH_SIZE=4`,
  `SAVE_INTERVAL=1`, full optimizer state
- checkpoint: `checkpoints/iter_0000001`
- payload: complete model, optimizer, LR scheduler, RNG, NLA metadata, and
  latest checkpoint pointer
- size: about `177G`
- timing: about `9m23s` from Gloo checkpoint process-group creation to final
  "Saved checkpoint" log

Corrected resume smoke:

- run_dir: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-checkpoint-debug/nano-av-miles-fsdp2-checkpoint-gloo-resume-root-gb8-mb4-nosave-tokenized-20260527T2350Z`
- `LOAD_CHECKPOINT`: checkpoint root ending in `/checkpoints`, not the
  `iter_0000001` leaf
- result: loaded model, optimizer, and LR scheduler from `iter_0000001`, then
  trained step `1` with loss `1.8423114` and grad norm `11.25`

Operational note:

- An attempted resume using the `iter_0000001` leaf skipped DCP load because
  Miles expects the checkpoint root containing `latest_checkpointed_iteration.txt`.
  The launcher now rejects `LOAD_CHECKPOINT` values shaped like `iter_XXXXXXX`.

### failed: hero-gloo-gb192-save100-raw-source-startup

- date_utc: `2026-05-28`
- experiment_class: `complete-performance`
- backend: Miles FSDP2 with Gloo DCP metadata process group
- status: failed before training
- run_dir: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-hero/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gloo-gb192-mb8-save100-20260528T0100Z`
- attempted source: `/workspace/interp/artifacts/nano30b-nla-pilot/super-teacher-r27-100k-thinking-merged-20260525T2150Z/base_R27_super_thinking_99570_explained.parquet`
- failure: `KeyError: 'tokens'` in `NLADataSource`

Interpretation:

- The raw R27 source artifact has activation and source fields but is not the
  tokenized Miles AV-SFT prompt-data contract.
- The prior gb192 hero reached training because it used the tokenized
  `av_sft.parquet` source.

### completed: hero-gloo-tokenized-gb192-save100

- date_utc: `2026-05-28`
- experiment_class: `complete-performance`
- backend: Miles FSDP2 with Gloo DCP metadata process group
- status: completed
- run_dir: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-hero/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gloo-tokenized-gb192-mb8-save100-20260528T0110Z`
- dataset: `/workspace/interp/outputs/nano30b-nla-pilot/av-r27-100k-rslora-batch8-epoch1-v1/av-r27-99570-rslora-r192-broad-scale75-lr1e5-b8-e1-epoch-gen8-save-2gpu-offline-20260527T0250Z/av_sft.parquet`
- raw source artifact: `/workspace/interp/artifacts/nano30b-nla-pilot/super-teacher-r27-100k-thinking-merged-20260525T2150Z/base_R27_super_thinking_99570_explained.parquet`
- split: doc-level `90/5/5`
- train/validation/test rows: `89618/4978/4974`
- padded train rows: `89664`
- optimizer steps: `467`
- config: `GLOBAL_BATCH_SIZE=192`, `MICRO_BATCH_SIZE=8`,
  `ROLLOUT_BATCH_SIZE=192`, `SAVE_INTERVAL=100`, launched with
  `NLA_KEEP_LOCAL=3`
- checkpoint retention update: after the step-100 checkpoint finalized, the
  storage policy was tightened to minimum exact resume (`keep_full=1`) so only
  the latest completed full checkpoint is retained.
- final checkpoint: `checkpoints/iter_0000467`; older full checkpoints pruned
- HF eval checkpoint: `hf_iter_0000467`
- final train metric: step `466`, loss `0.9521`
- W&B run: `https://wandb.ai/rishabhga97/nano30b-nla-pilot/runs/dw7mp5sn`
- startup confirmation: step `1` completed at `2026-05-28T01:14:51Z` with
  train loss `1.8133697509765625`, grad norm `7.40625`, and step time
  `85.37s`

Heldout checkpoint eval:

- report: `eval_iter_0000467_v64_t64/report.json`
- local copy:
  `artifacts/runai_eval/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gloo-tokenized-gb192-mb8-save100-20260528T0110Z/eval_iter_0000467_v64_t64_report.json`
- eval counts: validation `64`, test `64`
- validation NLL: real `0.9046`, shuffled `1.3298`, zero `1.1743`,
  mean `1.1924`, no-injection `1.3441`
- validation gaps vs real: shuffled `0.4251`, zero `0.2697`,
  mean `0.2878`, no-injection `0.4395`
- test NLL: real `0.9565`, shuffled `1.3493`, zero `1.2160`,
  mean `1.2302`, no-injection `1.3577`
- test gaps vs real: shuffled `0.3928`, zero `0.2595`, mean `0.2736`,
  no-injection `0.4012`

Interpretation:

- The full-data AV-SFT training hero completed and the bounded heldout eval is
  row-specific: real activations beat shuffled, zero, mean, and no-injection
  controls on validation and test.
- This is not an AV+AR run. AR/critic SFT remains future work after AV-SFT.
- The zero and mean control gaps are positive but below the aspirational `0.30`
  target on the v64/t64 sample, so the next eval should be larger and use the
  cached-mean evaluator before making stronger claims.
- Post-audit status correction (`2026-06-10`): this checkpoint trained before
  the Nano/Nemotron-H packed-boundary contamination fix, so it is retained as
  mature R27 fallback/scouting evidence, not clean post-fix hero proof. Any
  future R27-vs-R33 claim should compare clean post-fix checkpoints and must
  include the AV-generated-text -> AR reconstruction gate.

Launch checks:

- RunAI GPUs were idle before launch.
- Tokenized AV-SFT source and sidecar existed.
- Live Miles source had patch `0006` applied.
- Heavyweight debug checkpoint payloads were pruned before launch, leaving about
  `893G` free on `/workspace/interp`.

### completed: r33-component-full-av-ar-hero

- date_utc: `2026-06-19`
- evidence_freeze_utc: `2026-06-21T15:57Z`
- experiment_class: `complete-performance`
- backend: Miles FSDP2, W&B offline
- status: selected internal R33 AV+AR hero milestone
- run_dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512`
- dataset:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_component_fullscan275396/av_sft_r33_component_fullscan275396.parquet`
- paired AR checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001289`
- AV checkpoint:
  `checkpoints/iter_0001291`
- config:
  `GLOBAL_BATCH_SIZE=192`, `MICRO_BATCH_SIZE=2`, sequence length `1152`,
  dynamic packed-token cap `512`, `lr=1e-4`, cosine, warmup `25`, injection
  scale `75`.

Corrected heldout AV eval:

- report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/eval_iter_0001291_v512_t512_gen8_report.json`
- eval counts: validation `512`, test `512`
- validation NLL: real `0.798672`, shuffled `1.331095`, zero `1.167483`,
  mean `1.241662`, no-injection `1.224772`
- test NLL: real `0.819993`, shuffled `1.361868`, zero `1.196865`,
  mean `1.287035`, no-injection `1.259839`

Round-trip gate:

- report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/roundtrip_iter_0001291_v256_t256_report.json`
- gate passed: `true`; `baseline_required=false`
- validation/test AV-real NMSE:
  `0.000109680 / 0.000121664`
- validation/test teacher-text NMSE:
  `0.000106810 / 0.000112370`
- parse health: closed fraction `1.0`, usable fraction `1.0` on both splits
- controls: AV-real beat `mean`, `av_mean`, `av_none`, `av_zero`, and
  `av_shuffled` by aggregate NMSE and rowwise win-rate thresholds.

Preservation and cleanup:

- local compact evidence archive:
  `artifacts/runai_sync/20260621T155000Z_r33_component_full_hero/20260621T155000Z_r33_component_full_hero_compact.tgz`
- archive SHA-256:
  `67063bf2ecb3c0face452410060aef42b29556e2d168ad52fc2dcb0933c7213b`
- S3 prefix:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/sync_exports/20260621T155000Z_r33_component_full_hero/`
- cleanup:
  deleted the superseded component-full AV smoke checkpoint payload (`59G`)
  after freezing evidence; kept selected AR `iter_0001289` and AV
  `iter_0001291` checkpoint payloads on RunAI.

Interpretation:

- This is the first selected clean component-full R33 AV+AR milestone with
  strong AV real-vs-control losses and a passing actual generated-text
  reconstruction gate.
- This remains an internal hero milestone. A row-matched clean R27 comparison
  should be run before claiming external R33-over-R27 superiority.

## R33 Corrected K3 RL Probes (2026-07-01 to 2026-07-02)

These probes start from the selected component-full R33 AV-SFT checkpoint and
use the selected R33 AR checkpoint as a frozen round-trip critic. Both use the
strict train-only R33 RL parquet (`247,700` verified rows), six actor H100s,
one critic H100, one managed SGLang rollout H100, K3 coefficient `1e-3`, exact
`gb384/mb32`, and eight generated responses per prompt.

### Selected: `lr=1e-5`, update 8

- run:
  `r33-corrected-k3-probe-lr1e5-update8-unifiedenv-retry1`
- training:
  eight updates completed; reward means
  `-0.5084, -0.3534, -0.3401, -0.3200, -0.3438, -0.2852, -0.3331,
  -0.3633`; drift `0.2632-0.3285`; K3 ended with a `10.9046` spike.
- valid round-trip gate:
  256 validation plus 256 test rows; AV-real NMSE
  `0.0001085655 / 0.0001195005` versus matched SFT
  `0.0001096657 / 0.0001216750`; closed/usable fractions `1.0 / 1.0`; all
  controls passed.
- disposition:
  selected for the 32-update confirmation. Update-8 actor checkpoint was last
  verified on the mounted PVC on `2026-07-02`; recheck after redeploy.
- evidence:
  S3 `rl_evidence/20260701T231539Z/`, SHA256
  `2029d86b49d6f72c9b0cd333839d9dd6ae40fd5ad889d000f8f4921d9d778419`.

### Not selected: `lr=2e-5`, update 8

- run:
  `r33-corrected-k3-probe-lr2e5-update8`
- training:
  eight updates completed; drift `0.2837-0.3420`; K3 spiked to `599.9822`
  and `102.9060` at the first two nonzero updates before recovering.
- invalid evaluation:
  the cache-backed 256/256 report is excluded because incremental cached
  tokens diverged from full-prefix tokens at generated token index 1.
- valid round-trip gate:
  replacement `legacy_batch` 64/64 report; AV-real NMSE
  `0.0001069578 / 0.0001179766` versus matched SFT
  `0.0001095636 / 0.0001207099`; closed/usable fractions `1.0 / 1.0`; all
  controls passed.
- selection comparison:
  versus the same 64 rows from `lr=1e-5`, validation was `1.64%` worse and
  test `3.86%` better. The combined `1.32%` mean advantage had only `45.31%`
  rowwise wins and a paired bootstrap interval crossing zero.
- disposition:
  not selected due to inconclusive quality separation and larger K3
  transients. Actor checkpoint deleted after evidence freeze.
- evidence:
  S3 `rl_evidence/20260702T025445Z/`, SHA256
  `c49bd2bf35d18773cd6361b621b9d8c60a8657b6b01211a125fb452d799a3469`.

These probes selected `lr=1e-5` for the later guarded confirmation and hero
line. They are superseded for checkpoint selection by the completed update-342
run below.

## R33 Corrected K3 RL Hero (2026-07-08)

### selected: `r33-corrected-k3-hero-lr1e5-update342-resume228-retry3`

- topology/config:
  six actor FSDP H100s, one SGLang rollout H100, one frozen R33 AR critic
  H100; constant LR `1e-5`, K3 coefficient `1e-3`, global batch `384`,
  microbatch `32`, eight samples per prompt, endpoint `342`.
- lineage:
  retry 2 produced model-only checkpoint `iter_0000228`; retry 3 restored its
  model, RNG, rollout counter, and dataset state, reset unavailable Adam state,
  and continued from rollout 228 through actor step 341.
- checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_hero/r33_corrected_k3_hero_lr1e5_update342_resume228_retry3/actor/iter_0000342`.
- final train record:
  loss `0.00183662`, K3 KL `1.836525`, grad norm `0.886719`, log-prob
  difference `0.311984`, entropy `0.789943`, LR `1e-5`.
- prerequisite gate:
  `64/64`, passed; validation/test relative improvement over SFT
  `25.01% / 28.32%`.
- promotion gate:
  `512/512`, passed; validation/test AV-real NMSE
  `0.000087528 / 0.000091176` versus matched SFT
  `0.000126796 / 0.000134752`; relative improvement
  `30.97% / 32.34%`; rowwise wins `83.40% / 88.67%`.
- parse/controls:
  usable generation `100% / 100%`; closed tags `99.02% / 99.41%`; AV-real
  beat teacher, mean, AV-mean, none, zero, and shuffled controls.
- provenance/statistics:
  exact baseline hashes and row identities matched; 52 documents/split;
  clustered confidence intervals positive; top-five improvement share only
  `6.74% / 6.30%`.
- completion:
  queue complete at `2026-07-08T03:20:52Z`; temporary HF cleaned; checkpoint
  and offline W&B retained.
- evidence:
  `docs/runs/r33_rl_hero_20260708.md`; local/S3 archive SHA-256
  `78cbf98d27188594c25cbf9c0d695f0b3b1754df978961585bbaa6fc178f0bc7`.

## Publication-Clean R33 SFT Recovery (2026-07-10)

### completed: deterministic family-clean R33 AR SFT

- run: `nano-ar-r33-publication-deterministic-family-clean-4gpu-unfusedtorchconv-expertscan-cudablock-lr5e5-warmup25-gb192-mb48`
- geometry: four H100-NVL GPUs, `gb192/mb48`, LR `5e-5`, cosine decay,
  warmup 25, unfused PyTorch causal convolution plus efficient Mamba scan
- completion: 1,291 updates; final loss/FVE `0.2625350 / 0.5302927`; no CUDA,
  OOM, or nonfinite training signal
- validation-only 512 gate: teacher NMSE `0.281703`, shuffled `0.968888`,
  blank `0.756098`, generic `0.781429`, mean `0.678041`, source context
  `0.301252`, source raw `0.083248`
- rowwise signal: teacher beat shuffled `512/512`, blank `504/512`, generic
  `508/512`, and mean `505/512`
- disposition at this boundary: selected clean AR SFT checkpoint; the later
  July 15 AV+AR qualification supersedes the then-pending promotion state

### historical July 10 snapshot: corrected packed R33 AV SFT eval pending

- run: `nano-av-r33-publication-deterministic-family-clean-8gpu-pospass-lr1e4-warmup25-gb192-mb2-dyn4096`
- geometry: eight H100-NVL GPUs, `gb192/mb2`, dynamic cap 4096, LR `1e-4`,
  cosine decay, warmup 25
- fail-closed live gate: packed/padded NLL `2.56366563 / 2.56583595`, maximum
  absolute/relative difference `0.02207184 / 0.00795042`, passed before step 0
- optimizer completion: 1,291 updates at `2026-07-10T14:09Z`; final loss,
  gradient norm, LR, and normalized router entropy were
  `0.683009 / 0.585938 / 1e-5 / 0.991240`
- dynamics: loss-window means declined from `1.488113` over steps 0-24 to
  `0.696919` over steps 1024-1290; all logged training and router scalars were
  finite, and all large gradient norms were confined to initial warmup
- disposition at this boundary: final checkpoint existed while conversion was
  active; the July 15 validation verifier later passed

### historical July 10 snapshot: clean validation AV->AR gate staged

- family coverage: every `13,766` validation row across 250 families and every
  `13,765` test row across 255 families is eligible; zero train exposure or
  cross-split-family exclusions
- protocol: validation-only 512 rows, family-stratified, 256 generation tokens,
  real/shuffled/zero/mean/no-injection controls, parse-health thresholds
- runtime: replacement immutable source `9b8b44f`; DCP fingerprint and tmpfs
  staging share one sequential pass, followed by local HF conversion,
  eight-worker generation, cached AR scoring, and `finally` cleanup
- dry-run: generation and score share protocol SHA-256
  `276d97c3e460218e24bf7bd751a94bd4c9ed55859cdf4c751fb2824338e8e1aa`;
  63 relevant RunAI tests pass
- disposition at this boundary: watcher PID `1506837` was waiting for the
  bounded AV eval. The July 15 validation and one-time test gates later passed

### historical July 10 snapshot: independent critic rebuild and clean AR lineage staged

- source: immutable commit `3676b93`
- critic gate: prefix-dataset queue requires exact init-manifest SHA-256
  `71cfb2bf243bbae720d0b2931a310b6b327a2922a59d5cf5165926fc988edcff`
- AR gate: validation-only independent AR queue dry-runs as
  `blocked_missing_critic_init`, which is the intended fail-closed state
- disposition: neither queue has launched; clean AV eval, protocol-matched SFT
  round trip, AV model preservation, and verified redundant-shard cleanup are
  prerequisites

This section intentionally preserves the last authenticated July 10 snapshot.
It is superseded for AV and round-trip status by the qualified July 15 entry,
and for independent AR by the completed July 16 follow-up at the top of this
file. Clean confirmatory RL remains unlaunched and is not part of the selected
supervised checkpoint claim.

## Pre-FSDP Context Summary

These legacy runs established that the AV signal exists, but they are not the current full-data training path.

- Best legacy batch1 lr1e-5 result: heldout real NLL `1.1707` vs shuffled `1.5454`, zero `1.4172`, mean `1.4671`, no-injection `1.4797`; generation parsed `7/8`, F1 about `0.4466`.
- Legacy batch4 200-step curve check: real `1.1406` vs shuffled `1.5170`, zero `1.3890`, mean `1.4476`, no-injection `1.4932`.
- Legacy batch8 small smoke: validation real `1.7841` beat all controls; test real `1.7244` beat all controls.
- The attempted legacy full batch8 epoch was manually stopped after about 4 hours because projected runtime remained multi-day; no report or checkpoint was produced, so it is a non-result.

## Removed From Active Planning

The following are intentionally absent from the live tracker:

- fixed-step smoke-harness substitutes for full-data training
- old rsLoRA/DoRA exploration jobs
- AR or critic jobs before AV-SFT FSDP2 is validated
- obsolete batch-size drafts superseded by the `gb192/mb8` hero
