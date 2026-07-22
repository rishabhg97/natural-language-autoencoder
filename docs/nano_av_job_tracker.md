# Nano30B NLA Job Tracker

<!-- R33-HERO-BASELINE-PROTOCOL-INVALIDATED -->

> [!CAUTION]
> Publication status (`2026-07-16`): the family-clean R33 SFT AV+AR pair is
> qualified for directional reconstruction and stored-snapshot functional
> recovery, with validation-only replication through an independently trained
> seed-`314159` AR. The archived `30.97% / 32.34%` RL comparison remains
> invalidated. See `docs/runs/r33_clean_sft_av_ar_20260715.md`.

This file tracks live or explicitly planned Nano30B AV, AR, round-trip, and RL
jobs. Completed and failed post-FSDP2 runs live in
`docs/nano_av_run_history.md`; old pre-FSDP smoke-harness runs are summarized
there as context instead of staying in the active queue.

## Current Status

The clean AR and AV queues are complete at `iter_0001291`. Their component
verifiers pass. The protocol-matched generated-text round trip and
stored-snapshot functional evaluation pass on 512 validation and 512 test
rows, and the checkpoint-pair manifest reports `qualified: true`. Both selected
model payloads are preserved on S3 and compact evidence is synced locally.

The independent AR, cross-critic validation gate, selected-run compute report,
redacted security audit, and deterministic no-weights candidate archive are
complete. The candidate and its manifest/audit/attestation are preserved as
five internal S3 objects.

The RunAI `train` workspace is currently running on eight H100 NVLs. A bounded
clean online joint AV+AR HPO queue is active from the final qualified SFT pair.
Only H1 is approved; H2-H4 remain unapproved. This does not change the existing
SFT release claim. Historical RL smoke/HPO work and the 342-update July 8 hero
remain non-publication evidence because the hero's SFT comparison used a
mismatched generation protocol. The new queue preserves the final pair's
dataset, family, row-identity, and generation-protocol claim boundary.

## Agent Update Protocol

When starting or finishing a Nano30B AV run, update this file and `docs/nano_av_run_history.md` in the same change set.

- Add a queue entry before launching a job.
- Keep `status`, `last_update_utc`, `run_dir`, and the pass or block gate current.
- Do not leave draft jobs in this file unless they are still planned.
- Keep old fixed-step smoke-harness substitutes out of the live queue.
- Do not claim scientific success from train loss alone; validation/test real-vs-control gaps are required.

## Experiment Classes

- `small-smoke`: `row_limit <= 96`, doc split `80/10/10`, path, memory, NaN, checkpoint, and resume only.
- `medium-small`: `row_limit <= 960`, doc split `80/10/10`, directional signal and throughput.
- `complete-performance`: current full dataset, `row_limit >= 90000`, doc split `90/5/5`, one full train split epoch.

## Operational Requirements

Every launched run must declare:

- `experiment_class`
- split fractions
- row limit
- W&B offline mode
- output run directory
- report JSON or checkpoint-eval plan
- checkpoint interval for long runs

Full-data runs require at least one exact-resume checkpoint. Current storage
policy is minimum exact-resume retention: keep the latest completed full
checkpoint payload, plus lightweight eval reports and optional HF conversion
artifacts as storage allows.

## Queue

### active: r33-family-clean-online-joint-hpo8

- status: `h1_retry5_running_joint_step_validation`
- last_update_utc: `2026-07-17T14:45:00Z`
- queue:
  `configs/nano_rl/hpo/r33_family_clean_online_joint_hpo8_queue_8h100.yaml`
- run: `r33-family-clean-online-joint-hpo8-h1-a1e5-c2e6-k3e3-retry5`
- topology: 8 H100 NVLs; `4 actor / 3 critic / 1 SGLang rollout`
- batch: `24 prompts x 8 samples = 192`; microbatch `2`; eight updates
- H1 parameters: actor/critic LR `1e-5 / 2e-6`, K3 coefficient `3e-3`
- critic contract: exact full-usable plan `192/192`, global parse filtering,
  DP/microbatch alignment, minimum usable retention `0.95`, no hidden local
  truncation
- eval: validation only, 128 family-stratified paired rows, five controls,
  clean-SFT baseline; test remains sealed
- W&B: offline actor and critic logs
- storage: final update-8 model-only actor/critic DCP; temporary actor HF under
  `/dev/shm`; H2-H4 require explicit approval and must not overlap checkpoints
- preflight correction: first initialization attempt stopped before rollout or
  optimizer because retention/JSON guards were not forwarded to Ray; preserved
  under `preflight_abort_20260717T1240Z`, no checkpoint
- retry evidence: retry 3 completed a healthy padded actor optimizer update;
  retry 4 retained all 168 usable critic rows, balanced them `56/56/56`, and
  passed exact reward/train MSE equivalence before exposing missing critic
  `bshd` width metadata. Neither produced a checkpoint.
- verification: combined RunAI queue/launcher/layout/repartition suite
  `110 passed`; `rl.sh` syntax check passed
- summary: `docs/runs/r33_online_joint_hpo8_20260717.md`

### complete-nonpromoted: r33-family-clean-online-joint-canary-update2

- status: `complete_systems_pass_quality_not_promoted`
- last_update_utc: `2026-07-17T09:51:00Z`
- run_id: `r33-family-clean-online-joint-canary-update2-8h100-retry4`
- topology: 8 H100 NVLs; `4 actor / 3 critic / 1 SGLang rollout`
- initialization: qualified family-clean R33 AV and AR SFT checkpoints
- training: two online joint updates; 3 prompts x 8 samples = 24 rollouts per
  update; actor/critic LR `1e-5 / 5e-6`; W&B offline
- checkpoint result: complete actor and critic DCP payloads at
  `iter_0000002`; no OOM or optimizer failure
- exact eval boundary: 64 family-stratified rows and 64 independent families;
  candidate and clean-SFT baseline have identical row keys, dataset hashes,
  tokenizer, generation protocol, and five control arms
- online/SFT directional MSE: `0.291993 / 0.292173`; relative directional
  improvement `0.0618%`; paired wins `32/64`
- online/SFT raw MSE: `8.969927 / 8.797533`; online is `1.96%` worse
- strict inference: directional family-bootstrap 95% CI
  `[-0.007704, 0.008597]`; one-sided sign-flip `p=0.4824`
- gates: provenance, family, parse, and controls pass; strict SFT-improvement
  gate fails; `baseline_beaten: false`
- decision: systems canary passed, but do not promote or scale this checkpoint;
  the qualified clean SFT pair remains canonical
- summary: `docs/runs/r33_online_joint_canary_20260717.md`
- local evidence:
  `artifacts/runai_eval/r33-online-joint-canary-evidence-20260717T0951Z/`
- S3 evidence:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/run-artifacts/r33-online-joint-canary-evidence-20260717T0951Z.tgz`

### complete: r33-family-clean-sft-av-ar-qualification

- status: `complete`
- last_update_utc: `2026-07-15T21:38:51Z`
- release_id: `r33-clean-sft-av-ar-iter1291-20260715`
- AR checkpoint: family-clean R33 `iter_0001291`, HF SHA-256
  `5e792120ec1a00ebb4cf4abca50d2a6a962421ac4f45423479ae5061f4d2d760`
- AV checkpoint: family-clean R33 `iter_0001291`, DCP model fingerprint
  `dcp_model_sha256:43346232d2fc043260ee903191e20cce07801903e1e7b7956f16022eb463386a`
- component gates: AR and AV validation verifiers pass on 512 rows
- round-trip gates: validation/test directional MSE `0.307004 / 0.319225`;
  teacher `0.304714 / 0.302637`; all controls pass
- parse gate: `100%` closed and usable on both splits
- functional gates: validation and test pass for stored-snapshot
  counterfactual reinjection
- pair manifest: `nano_nla_checkpoint_pair_manifest.v1`, `qualified: true`
- local evidence:
  `artifacts/runai_eval/r33-clean-sft-av-ar-qualified-20260715/`
- S3 evidence:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/sync_exports/r33_clean_sft_av_ar_qualified_20260715.tgz`
- next gate: none for the scoped SFT pair; require a pristine external test,
  row-matched R27, second AV seed, or clean RL only for the corresponding
  stronger claim

### complete: r33-publication-followup-and-release-candidate

- status: `complete_claim_bounded`
- last_update_utc: `2026-07-16T16:18:13Z`
- independent AR: seed `314159`, `iter_0001291`, component and cross-critic
  validation verifiers passed
- cross-critic result: AV-text directional MSE `0.310963`, teacher `0.308533`,
  512 rows/250 families, `100%` usable, all five controls passed
- selected training compute: `138.7456` H100-NVL GPU-hours across primary AR,
  primary AV, and independent AR
- public candidate: 496 files, no weights, security gate passed; archive
  SHA-256
  `3eb8e64ed0d9d61ed2d6b0694fbaf96b99051a63f2ce1a6c99372d93832e573a`
- archive/tree identity:
  `df175c5f61cefbfc1a02451a7bd242ba69e1cb602cdd97ca4b8bd8fe9c263b77`
- S3 preservation: five objects under
  `publication/release-candidates/r33-clean-sft-av-ar-iter1291-20260716/`
- verification: `749 passed`; claim-document consistency passed
- remaining non-job gates: blinded human review, legal/license approval, exact
  teacher-service terms, and final notices

### superseded: r33-scaling-100k-dataset-and-hpo

- experiment_class: `tuning-probe` prep and queued probes
- status: superseded by the verified component-full R33 SFT and RL hero path
- last_update_utc: `2026-06-07`
- historical note: this was the unlaunched 100k staging plan from
  `2026-06-07`; later component-full runs replaced it.
- prep script: `scripts/nano_ar_r33_scaling_pipeline.sh`
- expected RunAI output root:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_scaling_100k`
- planned R33 AR-SFT dataset:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_scaling_100k/ar_sft_r33_start10500_len10000.parquet`
- planned R33 AV-SFT dataset:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_scaling_100k/av_sft_r33_start10500_len10000.parquet`
- planned R33 critic init:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-critic-init/nano-ar-r33-critic-init`
- source slice: FineWeb `sample-10BT`, train docs `10500:20500`,
  `positions_per_doc=10`, expected rows about `99,570`
- AR queue: `configs/nano_ar/hpo/r33_100k_scaling_queue.yaml`
- AR configs:
  - `configs/nano_ar/hpo/r33_100k_lr2e5_cosine_gb192_mb8.yaml`
  - `configs/nano_ar/hpo/r33_100k_lr1e5_cosine_gb192_mb8.yaml`
- AR controls: `teacher`, `teacher_shuffled`, `blank`, `generic`, `mean`,
  `source_context`, `source_raw`
- AV queue: `configs/nano_av/hpo/r33_100k_scaling_queue.yaml`
- AV config:
  `configs/nano_av/hpo/r33_100k_lr1e5_gb192_mb2_seq1152_dyn512.yaml`
- AV controls: `real`, `shuffled`, `zero`, `mean`, `none`
- storage policy: model-only probe checkpoints, W&B offline logs, final eval
  reports retained; AV eval uses temporary DCP-to-HF conversion and cleanup
- promotion gate: do not call R33 a good NLA until the actual chained
  `h -> AV-generated explanation -> AR h_hat` reconstruction beats the mature
  R27 baseline materially.

Gate:

- Run `scripts/nano_ar_r33_scaling_pipeline.sh` on RunAI and rerun focused
  runner/queue tests in the RunAI venv before starting the AR queue.
- After both AR 100k probes complete, prefer the better heldout checkpoint for
  the round-trip gate only if teacher NMSE moves toward `0.25-0.30` and still
  beats all required controls on validation and test.

### completed: ar-r27-fullscan-eval-v2048-winrates

- experiment_class: `complete-performance` eval gate
- status: completed; larger heldout eval report written
- launched_utc: `2026-05-31T16:49Z`
- completed_utc: `2026-05-31T17:14Z`
- last_update_utc: `2026-06-01T00:18Z`
- pid: `2648297`
- checkpoint: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-fullscan/nano-ar-miles-fsdp2-r27-fullscan-275k-gb192-mb8-lr1e5-20260530T172928Z/checkpoints/iter_0001291`
- report: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-fullscan/nano-ar-miles-fsdp2-r27-fullscan-275k-gb192-mb8-lr1e5-20260530T172928Z/eval_iter_0001291_v2048_t2048_winrates_report.json`
- controls: `teacher`, `teacher_shuffled`, `blank`, `generic`, `mean`,
  `source_context`, `source_raw`
- eval rows: validation `2048`, test `2048`
- evaluator update: `scripts/eval_nano_ar_miles_checkpoint.py` reports
  rowwise teacher-vs-control win counts and win fractions in addition to mean
  normalized MSE, raw MSE, cosine, and FVE-vs-mean.
- validation teacher: normalized MSE `0.503023`, cosine `0.748488`, FVE vs
  mean `0.416190`; rowwise teacher beats mean `98.05%`, shuffled `99.90%`,
  blank `98.73%`, generic `98.93%`, and source-context `77.54%`; source-raw
  beats teacher `99.02%`
- test teacher: normalized MSE `0.514389`, cosine `0.742806`, FVE vs mean
  `0.401452`; rowwise teacher beats mean `97.46%`, shuffled `99.85%`, blank
  `98.10%`, generic `98.34%`, and source-context `74.56%`; source-raw beats
  teacher `99.41%`

Gate:

- The first fullscan checkpoint has stable heldout signal and is worth tuning,
  but it is not yet Qwen-level. Do not launch RL from this checkpoint.

### completed: ar-r27-fullscan-hpo-qwen-lr2e5-cosine-256steps

- experiment_class: `tuning-probe`
- status: completed; model/HF-only checkpoint and `2048/2048` heldout eval present
- launched_utc: `2026-06-01T00:21Z`
- completed_utc: `2026-06-01`
- pid: `3259226`
- config: `configs/nano_ar/hpo/r27_fullscan_continue_qwen_lr2e5_cosine_256steps.yaml`
- run_dir: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-hpo/nano-ar-r27-fullscan-continue-qwen-lr2e5-cosine-256steps-20260601T0019Z`
- resume root: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-fullscan/nano-ar-miles-fsdp2-r27-fullscan-275k-gb192-mb8-lr1e5-20260530T172928Z/checkpoints`
- resume start / target rollout: `1291` / `1547`
- training shape: global batch `192`, micro `8`, rollout `192`, `lr=2e-5`,
  `min_lr=2e-6`, cosine decay, `50` warmup iters, `--finetune`
- final checkpoint: `checkpoints/iter_0001547`
- eval report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-hpo/nano-ar-r27-fullscan-continue-qwen-lr2e5-cosine-256steps-20260601T0019Z/eval_iter_0001547_v2048_t2048_winrates_report.json`
- checkpoint policy: model/HF-only probe save with `--no-save-optim`,
  `NLA_KEEP_LOCAL=1`

Result:

- validation teacher normalized MSE `0.436878`, cosine `0.781561`, FVE vs mean
  `0.492958`; rowwise teacher beats mean `98.54%`, shuffled `99.90%`, blank
  `98.73%`, generic `99.37%`, and source-context `65.48%`; source-raw beats
  teacher `98.97%`
- test teacher normalized MSE `0.450516`, cosine `0.774742`, FVE vs mean
  `0.475775`; rowwise teacher beats mean `97.71%`, shuffled `99.90%`, blank
  `98.58%`, generic `98.93%`, and source-context `62.30%`; source-raw beats
  teacher `99.02%`
- Relative to the first fullscan v2048 baseline (`0.503023` validation /
  `0.514389` test), this improved heldout NMSE by about `13.2%` validation and
  `12.4%` test.

Caveat:

- Because Miles `--finetune` reset the rollout counter, the nominal
  `256`-step probe rendered as `--num-rollout 1547` and ran much longer than a
  bounded continuation. The result is useful as the current best AR checkpoint,
  but the launcher/config semantics must be fixed before launching more HPO.

### active: ar-r27-ar-milestone-tuning-to-0p25-0p30

- experiment_class: `tuning-probe`
- status: first bounded current-best probe running on RunAI `train-dev`
- launched_probe_utc: `2026-06-01T19:56:32Z`
- running_probe:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-hpo/nano-ar-r27-best1547-continue-lr1e5-cosine-256steps-20260601T195632Z`
- running_pid: `2179374`
- purpose: select a good AR-SFT milestone before any AV+AR/RL work. Target
  heldout teacher normalized MSE is `0.25-0.30`; current best is `0.436878`
  validation / `0.450516` test on `2048/2048`.
- current best resume root:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-hpo/nano-ar-r27-fullscan-continue-qwen-lr2e5-cosine-256steps-20260601T0019Z/checkpoints`
- configs:
  - `configs/nano_ar/hpo/r27_best1547_continue_lr1e5_cosine_256steps.yaml`
  - `configs/nano_ar/hpo/r27_best1547_continue_lr5e6_cosine_256steps.yaml`
  - `configs/nano_ar/hpo/r27_fullscan_continue_lr5e6_256steps.yaml`
  - `configs/nano_ar/hpo/r27_fullscan_continue_qwen_lr2e5_cosine_256steps.yaml`
  - `configs/nano_ar/hpo/r27_fullscan_continue_qwen_batch256_lr2e5_cosine_256steps.yaml`
- config note: probes reuse the already-vetted fullscan
  `splits/train_padded.parquet` rather than rematerializing doc splits
- checkpoint policy: model/HF-only probe saves with `--no-save-optim`; rerun the
  winning recipe as a complete exact-resume checkpoint before calling it a final
  AR milestone
- first-step metric: step `0` logged train loss `0.416706`, train FVE
  `0.379620`, grad norm `0.621094`, and step time `88.74s`
- rough_eta: first-step timing projects about `6-7h` for `256` bounded steps,
  plus final model/HF-only checkpoint save time

Gate:

- Fixed runner semantics must render finetune `resume_steps: 256` as
  `--num-rollout 256`, not `latest_checkpoint + 256`.
- Every launched probe must run W&B offline, save at the final bounded step, and
  pass a `512/512` heldout eval before any larger `2048/2048` confirmation.
- Green milestone: validation and test teacher NMSE both `<=0.30` with teacher
  beating mean/shuffled/blank/generic/source-context controls. Usable milestone:
  both `<=0.35` if multiple bounded probes plateau above the green target.
- Do not start RL until the AR milestone checkpoint and eval report are selected
  and documented.

### completed: ar-r27-miles-fsdp2-fullscan-275k-qwen-faithful

- experiment_class: `complete-performance`
- title: Nano30B AR-SFT Miles/FSDP2 R27 Qwen-faithful fullscan critic run
- backend: Miles FSDP2, `NLACriticModel`, Qwen custom MSE critic loss
- status: completed; final checkpoint saved and bounded heldout eval passed
- launched_utc: `2026-05-30T17:29Z`
- completed_utc: `2026-05-31T08:32Z`
- last_update_utc: `2026-05-31T09:52Z`
- config: `configs/nano_ar/fullscan_r27_miles_fsdp2.yaml`
- run_dir: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-fullscan/nano-ar-miles-fsdp2-r27-fullscan-275k-gb192-mb8-lr1e5-20260530T172928Z`
- pid: `3688351`
- local handoff: `runs/introspection/ar-r27-r30-fullscan-20260528T234403Z/handoff/AR_TRAINING_HANDOFF.md`
- local AR-SFT dataset: `runs/introspection/ar-r27-r30-fullscan-20260528T234403Z/handoff/R_27/ar_sft.parquet`
- RunAI AR-SFT dataset: `/workspace/interp/outputs/nano30b-nla-pilot/ar-r27-r30-fullscan-20260528T234403Z/R_27/ar_sft.parquet`
- RunAI verifier: `/workspace/interp/outputs/nano30b-nla-pilot/ar-r27-r30-fullscan-20260528T234403Z/R_27/ar_dataset_verify_runai.json`
- critic init: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-critic-init/nano-ar-r27-critic-init`
- source: FineWeb `HuggingFaceFW/fineweb`, config `sample-10BT`, split `train`, non-overlapping span `corpus_start=10500`, `corpus_length=27662`
- AR-SFT rows: `275,396`
- split: doc-level `90/5/5`
- train/validation/test rows: `247870/13761/13765`
- padded train rows: `247872`
- optimizer steps: `1291`
- global_batch_size: `192`
- micro_batch_size: `8`
- rollout_batch_size: `192`
- lr: `1e-5`
- loss: `custom_loss`, `nla.loss.nla_critic_loss`
- rollout function: `nla.rollout.sft_critic.generate_rollout`
- W&B: offline (`wandb/offline-run-20260530_173248-7yw7nd46`, `wandb/offline-run-20260530_173300-7yw7nd46`, `wandb/offline-run-20260530_173311-7yw7nd46`)
- checkpoint plan: exact-resume final checkpoint only, `save_interval=1291`,
  `NLA_KEEP_LOCAL=1`, optimizer/scheduler state required
- final checkpoint: `checkpoints/iter_0001291`, about `123G`
- eval report: `eval_iter_0001291_v512_t512_report.json`
- eval controls: `teacher`, `teacher_shuffled`, `blank`, `generic`, `mean`,
  `source_context`, and `source_raw`
- first observed optimizer metric: logged at step `2` with train loss
  `1.1862951914469402`, train `fve_nrm=-0.7661219437917074`, grad norm
  `4.5625`, and step time about `32.96s`
- latest compact live check: PID still running after about `8m`; GPUs active
  with about `83.6GiB` used per H200; latest observed train metric was step
  `8`, loss `1.0207316080729167`, train `fve_nrm=-0.5196358760197958`;
  no traceback/OOM/error lines found in the compact scan
- final observed train metric: step `1290`, loss `0.4939449628194173`,
  train `fve_nrm=0.2646290063858032`, grad norm `0.6484375`
- bounded validation eval, `512` rows: teacher normalized MSE `0.511466`,
  cosine `0.744267`, FVE vs mean `0.423177`; mean `0.886695`, shuffled
  `1.153667`, blank `1.028922`, generic `1.033204`, source-context
  `0.632444`, source-raw `0.141013`
- bounded test eval, `512` rows: teacher normalized MSE `0.505286`, cosine
  `0.747357`, FVE vs mean `0.416054`; mean `0.865295`, shuffled
  `1.135018`, blank `1.010650`, generic `1.014764`, source-context
  `0.613429`, source-raw `0.137070`

Gates:

- RunAI auth, `/workspace/interp` paths, free disk, idle train-dev GPUs, and no
  active Nano training process were checked before launch.
- Local and RunAI dataset verifiers passed: `275,396` rows, `d_model=2688`,
  finite vectors, nonempty explanations, critic prompt suffix checks, and zero
  doc overlap. The RunAI verifier reported `90/5/5` rows
  `247870/13761/13765`.
- Local-to-RunAI parquet transfer was sha256 verified:
  `76b78d2c34a251f004d53eb5d53766fa01879e2bf3744bc4d80d4fcc1d17825e`.
- First optimizer update completed without OOM or traceback in the matched log
  tail. This is only an engineering health signal; AR success still requires
  heldout reconstruction metrics against controls after the final checkpoint.
- Final heldout eval passed the AR reconstruction gate: teacher explanation
  reconstruction beats mean, shuffled, blank, generic, and source-context on
  both validation and test. `source_raw` is stronger than teacher because it
  uses the raw token prefix/control context.
- RL has not been launched.

### completed: ar-r27-miles-fsdp2-medium-small-qwen-faithful

- experiment_class: `medium-small`
- title: Nano30B AR-SFT Miles/FSDP2 R27 Qwen-faithful critic path
- backend: Miles FSDP2, `NLACriticModel`, Qwen custom MSE critic loss
- status: completed; medium-small heldout gate passed after continuation
- launched_utc: `2026-05-28T18:42Z`
- completed_utc: `2026-05-28T20:20Z`
- last_update_utc: `2026-05-28T20:21Z`
- source artifact: `/workspace/interp/artifacts/nano30b-nla-pilot/super-teacher-r27-100k-thinking-merged-20260525T2150Z/base_R27_super_thinking_99570_explained.parquet`
- AR-SFT dataset: `/workspace/interp/outputs/nano30b-nla-pilot/ar-r27-100k-qwen-faithful/ar_sft.parquet`
- dataset verifier: `/workspace/interp/outputs/nano30b-nla-pilot/ar-r27-100k-qwen-faithful/ar_dataset_verify_full_20260528.json`
- critic init: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-critic-init/nano-ar-r27-critic-init`
- final medium checkpoint: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-medium/nano-ar-miles-fsdp2-r27-medium-small-continue-5ep-20260528T195213Z/checkpoints/iter_0000045`
- final medium HF checkpoint: `checkpoints/iter_0000045/hf`
- final medium eval report: `eval_iter_0000045_v128_t128_report.json`
- W&B: offline for every launched run
- checkpoint retention: model/HF-only for medium; no optimizer shards
- source_raw control: skipped in eval because this AR parquet does not contain usable raw token-id sidecar rows

Gates:

- RunAI auth, paths, disk, and idle GPU/training state checked before launches.
- AR dataset contract passed: `99,570` rows, `d_model=2688`, finite vectors,
  nonempty explanations, critic prompt suffix checks, and zero doc overlap for
  both `80/10/10` and `90/5/5` splits.
- Small one-step smoke saved an exact checkpoint and HF checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-small/nano-ar-miles-fsdp2-r27-small-smoke-save1-20260528T184247Z/checkpoints/iter_0000001`.
- Resume smoke loaded model, optimizer, LR scheduler, and rollout state from
  that checkpoint root, then advanced one resumed step:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-small/nano-ar-miles-fsdp2-r27-resume-smoke-20260528T185445Z`.
- One-epoch medium-small did not beat mean h, so it was not accepted as a pass.
- Five-epoch medium-small improved but still did not beat mean h, so it was not
  accepted as a pass.
- Five-epoch continuation from the medium HF checkpoint passed heldout controls:
  validation teacher normalized MSE `0.795473` beat mean `0.875942`,
  shuffled `0.988657`, blank `1.029558`, generic `1.017015`, and
  source-context `0.867701`; test teacher `0.795441` beat mean `0.880934`,
  shuffled `0.995219`, blank `1.016153`, generic `1.008652`, and
  source-context `0.851662`.
- Validation/test teacher FVE vs mean: `0.091865` / `0.097048`.

Interpretation:

- AR-SFT is now a separate, checkpointed critic path. It is not part of the
  completed AV-SFT hero, and RL has not been launched.
- The medium-small gate now has row-specific heldout signal against available
  controls, including mean h.

### superseded: ar-r27-miles-fsdp2-hero-100k-qwen-faithful

- experiment_class: `complete-performance`
- status: planned only; not launched; superseded by the 275,396-row fullscan R27
  run above
- plan_dir: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-hero/nano-ar-miles-fsdp2-r27-super-thinking-100k-hero-plan-20260528T202156Z`
- plan_file: `hero_run_plan.json`
- config: `configs/nano_ar/hero_100k_miles_fsdp2.yaml`
- dataset: same 99,570-row AR-SFT parquet as the medium gate
- split: doc-level `90/5/5`
- train/validation/test rows: `89618/4978/4974`
- padded train rows: `89664`
- optimizer steps: `467`
- global_batch_size: `192`
- micro_batch_size: `8`
- rollout_batch_size: `192`
- lr: `1e-5`
- loss: `custom_loss`, `nla.loss.nla_critic_loss`
- rollout function: `nla.rollout.sft_critic.generate_rollout`
- checkpoint plan: exact-resume final checkpoint only, `save_interval=467`,
  `NLA_KEEP_LOCAL=1`, optimizer/scheduler state required for hero
- eval plan: bounded validation/test checkpoint eval after the final save,
  controls `teacher`, `teacher_shuffled`, `blank`, `generic`, `mean`,
  `source_context`, and `source_raw` if present

Launch only after explicitly deciding to spend the full-run time/storage; do
not start RL from this plan.

### completed: av-r27-miles-fsdp2-hero-gloo-tokenized-gb192-save100

- experiment_class: `complete-performance`
- title: Nano30B AV-SFT Miles/FSDP2 R27 Super-thinking 100k hero
- backend: Miles FSDP2 with Gloo DCP metadata process group
- status: completed; final checkpoint and heldout eval present
- launched_utc: `2026-05-28T01:10Z`
- completed_utc: `2026-05-28T15:16Z`
- last_update_utc: `2026-05-28T17:34Z`
- run_dir: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-hero/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gloo-tokenized-gb192-mb8-save100-20260528T0110Z`
- dataset: tokenized AV-SFT parquet derived from the 99,570-row R27 source:
  `/workspace/interp/outputs/nano30b-nla-pilot/av-r27-100k-rslora-batch8-epoch1-v1/av-r27-99570-rslora-r192-broad-scale75-lr1e5-b8-e1-epoch-gen8-save-2gpu-offline-20260527T0250Z/av_sft.parquet`
- raw source artifact: `/workspace/interp/artifacts/nano30b-nla-pilot/super-teacher-r27-100k-thinking-merged-20260525T2150Z/base_R27_super_thinking_99570_explained.parquet`
- model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- split: doc-level `90/5/5`
- train/validation/test rows: `89618/4978/4974`
- padded train rows: `89664`
- optimizer steps: `467`
- global_batch_size: `192`
- micro_batch_size: `8`
- rollout_batch_size: `192`
- injection_scale: `75`
- lr: `1e-5`
- save_interval: `100`
- eval_interval: external checkpoint eval after save
- W&B: offline
- checkpoint retention: `NLA_KEEP_LOCAL=1` for minimum exact-resume storage
- final checkpoint: `checkpoints/iter_0000467`, latest exact-resume checkpoint
  retained; older full checkpoints pruned
- HF eval checkpoint: `hf_iter_0000467`
- final train metric: step `466`, loss `0.9521`
- W&B sync: `https://wandb.ai/rishabhga97/nano30b-nla-pilot/runs/dw7mp5sn`
- heldout eval report: `eval_iter_0000467_v64_t64/report.json`
- heldout eval counts: validation `64`, test `64`
- validation NLL: real `0.9046`, shuffled `1.3298`, zero `1.1743`,
  mean `1.1924`, no-injection `1.3441`
- validation gaps vs real: shuffled `0.4251`, zero `0.2697`,
  mean `0.2878`, no-injection `0.4395`
- test NLL: real `0.9565`, shuffled `1.3493`, zero `1.2160`,
  mean `1.2302`, no-injection `1.3577`
- test gaps vs real: shuffled `0.3928`, zero `0.2595`, mean `0.2736`,
  no-injection `0.4012`
- interpretation: AV-SFT engineering hero completed and row-specific heldout
  signal is positive on the bounded eval. This is not an AV+AR run; AR/critic
  SFT has not started.

Gates:

- Patch `0006_fsdp_checkpoint_gloo_pg.patch` is applied in the live RunAI Miles
  source.
- Focused full-optimizer save and corrected resume smokes passed before launch.
- Checkpoint eval shows real activations beat shuffled, zero, mean, and
  no-injection controls on validation and test for the bounded v64/t64 eval.

### failed: av-r27-miles-fsdp2-hero-gloo-raw-source-startup

- experiment_class: `complete-performance`
- status: failed before training
- launched_utc: `2026-05-28T01:04Z`
- run_dir: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-hero/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gloo-gb192-mb8-save100-20260528T0100Z`
- attempted source: `/workspace/interp/artifacts/nano30b-nla-pilot/super-teacher-r27-100k-thinking-merged-20260525T2150Z/base_R27_super_thinking_99570_explained.parquet`
- failure: `KeyError: 'tokens'` in `NLADataSource`
- interpretation: the raw R27 source artifact is not directly the tokenized
  Miles AV-SFT prompt-data contract; use the tokenized `av_sft.parquet` source
  for Miles training or add an explicit raw-to-tokenized conversion gate.

### failed: av-r27-miles-fsdp2-hero-gb192-save100

- experiment_class: `complete-performance`
- title: Nano30B AV-SFT Miles/FSDP2 R27 Super-thinking 100k hero
- backend: Miles FSDP2
- status: failed at first checkpoint save
- launched_utc: `2026-05-27T20:50Z`
- last_update_utc: `2026-05-27T23:03:58Z`
- run_dir: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-hero/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gb192-mb8-save100-20260527T2050Z`
- pid: `944165`
- dataset: `/workspace/interp/artifacts/nano30b-nla-pilot/super-teacher-r27-100k-thinking-merged-20260525T2150Z/base_R27_super_thinking_99570_explained.parquet`
- model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- split: doc-level `90/5/5`
- train/validation/test rows: `89618/4978/4974`
- padded train rows: `89664`
- optimizer steps: `467`
- global_batch_size: `192`
- micro_batch_size: `8`
- rollout_batch_size: `192`
- injection_scale: `75`
- lr: `1e-5`
- save_interval: `100`
- eval_interval: external checkpoint eval after save
- W&B: offline
- checkpoint target: `iter_0000100`
- latest completed metric: step `99`, loss `1.0409398078918457`, grad_norm `1.2890625`
- failure time: about `2026-05-27T22:36Z`
- failure mode: `RuntimeError: NCCL Error 1: unhandled cuda error` inside `NLAFSDPActor.save_model()` -> Miles FSDP checkpoint `dcp.save(...)`
- checkpoint payload: incomplete; `iter_0000100` directory exists but contains `0` files and the full checkpoint tree is only about `28K`
- process state: exited; GPUs idle

Notes:

- `gb384` OOMed on the second optimizer step after Adam state allocation; `gb192` trained to the first save boundary but failed during distributed checkpoint save.
- Train loss and grad norm behaved normally through step 99, but this is not a scientific result because no checkpoint eval can be run.
- Storage is expanded and healthy: `/workspace/interp` had about `659G` free
  after the focused full-optimizer checkpoint smoke.

Gate:

- Fixed by routing DCP metadata/object collectives through a Gloo process group
  while leaving model/FSDP training collectives on NCCL.
- Do not relaunch the hero until the patched Miles source is applied on RunAI.

### completed: checkpoint-save-gloo-remediation

- experiment_class: checkpoint gate
- status: pass
- trigger: `av-r27-miles-fsdp2-hero-gb192-save100` failed at `iter_0000100`
- target: reproduce and fix Miles/FSDP2 DCP save under realistic full-optimizer payload conditions
- patch: `external/natural_language_autoencoders/nla/miles_patches/0006_fsdp_checkpoint_gloo_pg.patch`

Completed checks:

- Model-only save smoke passed and wrote a complete about `59G` model checkpoint.
- Full-optimizer save smoke passed and wrote a complete about `177G` checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-checkpoint-debug/nano-av-miles-fsdp2-checkpoint-gloo-fulloptim-gb8-mb4-save1-tokenized-20260527T232920Z/checkpoints/iter_0000001`.
- Corrected resume smoke passed with `LOAD_CHECKPOINT=<run_dir>/checkpoints`,
  loading model, optimizer, and LR scheduler before training step `1`.
- The first attempted resume command used the `iter_0000001` leaf and skipped
  DCP load; the launcher now rejects that shape.

Outcome:

- The original NCCL `scatter_object_list` checkpoint failure is fixed for the
  focused Nano30B full-optimizer payload.
- Relaunch the full-data hero only after applying patch `0006` in the active
  RunAI Miles source.

### launched: av-r27-miles-fsdp2-hero-gb192-save100-gloo

- experiment_class: `complete-performance`
- status: superseded by running entry above
- backend: Miles FSDP2 with Gloo DCP metadata process group
- config: same as failed `gb192/mb8/save100` hero unless a smaller save interval
  is chosen to get the first checkpoint sooner
- checkpoint expectation: each full optimizer checkpoint is about `177G`; rolling
  latest 3 needs about `531G` before logs and eval artifacts
- load/resume note: `LOAD_CHECKPOINT` must point to the checkpoint root
  containing `latest_checkpointed_iteration.txt`, not an `iter_XXXXXXX` leaf

### ready: eval-hero-iter100-real-vs-controls

- experiment_class: `complete-performance`
- status: blocked; checkpoint `iter_0000100` is incomplete
- target checkpoint: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-hero/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gb192-mb8-save100-20260527T2050Z/checkpoints/iter_0000100`
- evaluator: `scripts/eval_nano_av_miles_checkpoint.py`
- controls: real, shuffled, zero, mean, no-injection
- splits: validation and test
- required metrics: real/control NLLs, gaps, examples/sec, generation parse rate, content overlap/F1 where generation is enabled

Planned command shape:

```bash
runai workspace exec train-dev -p trustworthy-ai-inference -- bash -lc '
  source /workspace/interp/.venv/bin/activate
  cd /workspace/interp/code/nano30b-nla-pilot-nano-av-h200-20260523T031253Z
  export PYTHONPATH="$PWD/external/natural_language_autoencoders:$PWD/external/natural_language_autoencoders/Miles:$PWD:$PYTHONPATH"
  python scripts/eval_nano_av_miles_checkpoint.py \
    --input-parquet /workspace/interp/artifacts/nano30b-nla-pilot/super-teacher-r27-100k-thinking-merged-20260525T2150Z/base_R27_super_thinking_99570_explained.parquet \
    --checkpoint-dir /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-hero/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gb192-mb8-save100-20260527T2050Z/checkpoints/iter_0000100 \
    --experiment-class complete-performance \
    --split-fractions 0.90 0.05 0.05 \
    --injection-scale 75 \
    --report-json /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-hero/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gb192-mb8-save100-20260527T2050Z/eval_iter100_val_test_controls.json
'
```

Pass gate:

- Eval loads the checkpoint without conversion errors.
- Validation and test real NLL beat shuffled, zero, mean, and no-injection.
- If gaps are weak, keep training but label the result as an intermediate checkpoint, not a completed scientific success.

### ready: resume-hero-checkpoint-smoke

- experiment_class: `complete-performance`
- status: blocked; no complete full-data checkpoint exists
- purpose: prove the full-data checkpoint payload can resume, not just evaluate
- checkpoint: `iter_0000100`
- expected action: resume for a small bounded continuation in a separate run directory when the active run is no longer using both H200s

Pass gate:

- step count advances beyond the resumed checkpoint
- W&B offline logs continue cleanly
- checkpoint cleanup keeps rolling latest 3
- eval still loads after resumed save

### completed-gate: r33-prefix-fullscan275396-dataset

- experiment_class: `dataset-gate`
- status: passed
- last_update_utc: `2026-06-08T14:23Z`
- output root:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_fullscan275396`
- base parquet:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_fullscan275396/base_R33_r33_prefix_fullscan275396.parquet`
- AR-SFT parquet:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_fullscan275396/ar_sft_R33_r33_prefix_fullscan275396.parquet`
- AV-SFT parquet:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_fullscan275396/av_sft_R33_r33_prefix_fullscan275396.parquet`
- R33 critic init:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-critic-init/nano-ar-r33-critic-init`
- AR verifier:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_fullscan275396/verify_ar_R33_r33_prefix_fullscan275396.json`
- AV verifier:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_fullscan275396/verify_av_R33_r33_prefix_fullscan275396.json`

Validation:

- AR rows `275,396`, sidecar rows `275,396`, `d_model=2688`,
  nonfinite activations `0`, empty explanations `0`, suffix bad `0`,
  90/5/5 doc overlap `0`.
- AV rows `275,396`, sidecar rows `275,396`, `d_model=2688`,
  nonfinite activations `0`, malformed responses `0`, prompt marker bad `0`,
  90/5/5 doc overlap `0`.

Code notes:

- Prefix-key extraction from `token_ids_prefix` replaced the failed sharded
  FineWeb streaming approach.
- `scripts/nano_prefix_dataset_sidecar.py` now centralizes AR/AV training
  sidecars, including `kind: nla_dataset` and injection token/template metadata
  required by critic init and AV verification.

Next gate:

- Launch bounded R33 AR/AV HPO from the verified 275k artifacts.
- Do not claim NLA success until the chained
  `h -> AV-generated explanation -> AR h_hat` gate beats the mature R27
  baseline materially.

### running: r33-ar-100k-lr2e5-prefix275k-hpo

- experiment_class: `tuning-probe`
- status: training
- last_update_utc: `2026-06-08T14:36Z`
- queue: `configs/nano_ar/hpo/r33_100k_scaling_queue.yaml`
- config: `configs/nano_ar/hpo/r33_100k_lr2e5_cosine_gb192_mb8.yaml`
- run_id: `nano-ar-r33-100k-lr2e5-cosine-gb192-mb8`
- run_dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr2e5-cosine-gb192-mb8`
- source AR-SFT:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_fullscan275396/ar_sft_R33_r33_prefix_fullscan275396.parquet`
- row limit: `99,570` from the verified 275,396-row source
- expected checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr2e5-cosine-gb192-mb8/checkpoints/iter_0000467`
- eval: `512/512` with teacher, teacher_shuffled, blank, generic, mean,
  source_context, and source_raw controls
- startup health: reached step `5/467`; early loss after warmup around
  `0.95-0.97`; no train-log errors at `2026-06-08T14:36Z`
- pending paired stability config:
  `configs/nano_ar/hpo/r33_100k_lr1e5_cosine_gb192_mb8.yaml`

### ready-gate: r33-av-ar-roundtrip-evaluator

- experiment_class: `roundtrip-gate`
- status: ready, waiting for bounded R33 AR and AV checkpoints
- script:
  `scripts/eval_nano_av_ar_roundtrip_gate.py`
- purpose: score `h -> AV-generated explanation -> AR h_hat` directly instead
  of relying only on teacher-text AR NMSE and AV real-vs-control NLL proxies.
- generation controls: `real`, `shuffled`, `zero`, `mean`, `none`
- AR scoring variants: teacher prompt, AV-generated controls, mean activation
  control, rowwise win rates, and optional R27 baseline comparison.
- verification:
  - `tests/test_nano_av_ar_roundtrip_gate.py -q` -> `4 passed`
  - broader remote regression slice with the live-queue pending assertion
    deselected -> `38 passed, 1 deselected`
- run after:
  - R33 AR checkpoint has a passing bounded `512/512` teacher/control eval
  - R33 AV checkpoint has a passing bounded real-vs-control eval
  - a mature R27 baseline round-trip report is available or generated for the
    same bounded gate

Latest AR HPO poll at `2026-06-08T14:48:33Z`:

- status: training
- step: `21/467`
- latest loss: `0.8072489`
- checkpoint/eval reports: none yet

### completed: r33-ar-100k-lr2e5-prefix275k-hpo

- experiment_class: `tuning-probe`
- status: complete
- completed_at_utc: `2026-06-08T20:03:07Z`
- run_id: `nano-ar-r33-100k-lr2e5-cosine-gb192-mb8`
- run_dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr2e5-cosine-gb192-mb8`
- checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr2e5-cosine-gb192-mb8/checkpoints/iter_0000467`
- eval report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr2e5-cosine-gb192-mb8/eval_iter_0000467_v512_t512_winrates_report.json`
- final train loss: `0.331434`
- final train `fve_nrm`: `0.407104`

Bounded AR eval:

| Split | teacher NMSE | teacher_shuffled NMSE | blank NMSE | generic NMSE | mean NMSE | source_context NMSE | source_raw NMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.334868 | 0.938597 | 0.788234 | 0.781042 | 0.679221 | 0.388525 | 0.081792 |
| test | 0.323239 | 0.920332 | 0.768147 | 0.763415 | 0.659224 | 0.382368 | 0.076623 |

Decision:

- Passed bounded AR improvement gate versus R33 20k and mature R27 tuned AR.
- Not yet a final hero decision: run the planned `lr=1e-5` stability comparison
  before choosing the full-275k AR hero config.

### running: r33-ar-100k-lr1e5-prefix275k-hpo

- experiment_class: `tuning-probe`
- status: stopped/replaced
- started_at_utc: `2026-06-08T20:28:38Z`
- config: `configs/nano_ar/hpo/r33_100k_lr1e5_cosine_gb192_mb8.yaml`
- run_id: `nano-ar-r33-100k-lr1e5-cosine-gb192-mb8`
- run_dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr1e5-cosine-gb192-mb8`
- stopped_at_utc: `2026-06-08T20:36:28Z`
- reason: operator stopped the mb8 run before checkpoint to test
  `micro_batch_size=16` throughput. Partial artifacts/logs preserved.

### stopped: r33-ar-100k-lr1e5-prefix275k-mb16-hpo

- experiment_class: `tuning-probe`
- status: stopped/replaced
- started_at_utc: `2026-06-08T20:41:55Z`
- config: `configs/nano_ar/hpo/r33_100k_lr1e5_cosine_gb192_mb16.yaml`
- run_id: `nano-ar-r33-100k-lr1e5-cosine-gb192-mb16`
- run_dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr1e5-cosine-gb192-mb16`
- launch mode: direct runner, because the stopped queue watcher left a stale
  RunAI flock; the train command is the same command rendered from the validated
  config.
- eval handling: run bounded `512/512` AR eval manually after
  `checkpoints/iter_0000467` exists, then mark the queue item complete manually.
- early throughput:
  - confirmed `--micro-batch-size 16`
  - confirmed `num_microbatches=6`
  - no OOM through first completed steps
  - step 1 optimizer-step total about `26.6s`; step log interval `28s`
  - early peak observed memory about `103 GiB / 98 GiB` on GPU0/GPU1
- stopped_at_utc: `2026-06-08T20:56:58Z`
- reason: operator stopped before checkpoint to test single-GPU `gb192/mb192`
  feasibility.
- cleanup: partial artifacts removed.

### failed: r33-ar-100k-lr1e5-prefix275k-mb192-gpu1-hpo

- experiment_class: `tuning-probe`
- status: failed
- started_at_utc: `2026-06-08T20:58:58Z`
- failed_at_utc: `2026-06-08T21:01:16Z`
- config: `configs/nano_ar/hpo/r33_100k_lr1e5_cosine_gb192_mb192_gpu1.yaml`
- run_id: `nano-ar-r33-100k-lr1e5-cosine-gb192-mb192-gpu1`
- run_dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr1e5-cosine-gb192-mb192-gpu1`
- launch mode: direct runner, single GPU.
- confirmed wiring:
  - `--actor-num-gpus-per-node 1`
  - `--global-batch-size 192`
  - `--micro-batch-size 192`
  - `num_microbatches=1`
- failure: CUDA OOM before step 0; GPU0 reached `139.79 GiB` process memory
  with PyTorch allocated `137.41 GiB`.
- placement: GPU1 stayed idle, so the single-GPU placement path worked.
- cleanup: failed-run split/W&B artifacts removed; retained `train.log`,
  `run_spec.yaml`, and `run_plan.json`; run dir is now about `544K`.
- cluster status after cleanup: no active Nano processes; GPU0/GPU1 both idle
  at `4 MiB` used.

### failed: r33-ar-100k-lr1e5-prefix275k-single-gpu-halving

- experiment_class: `tuning-probe`
- status: failed
- purpose: halve single-GPU microbatch after the `mb192` OOM until a setting
  clears the first optimizer step.
- tested configs:
  - `configs/nano_ar/hpo/r33_100k_lr1e5_cosine_gb192_mb96_gpu1.yaml`
  - `configs/nano_ar/hpo/r33_100k_lr1e5_cosine_gb192_mb48_gpu1.yaml`
  - `configs/nano_ar/hpo/r33_100k_lr1e5_cosine_gb192_mb24_gpu1.yaml`
  - `configs/nano_ar/hpo/r33_100k_lr1e5_cosine_gb192_mb12_gpu1.yaml`
- results:
  - `mb96`: `num_microbatches=2`; OOM at `optimizer.step` Adam state init,
    PyTorch allocated `137.30 GiB`.
  - `mb48`: `num_microbatches=4`; OOM at `optimizer.step` Adam state init,
    PyTorch allocated `137.11 GiB`.
  - `mb24`: `num_microbatches=8`; OOM at `optimizer.step` Adam state init,
    PyTorch allocated `136.64 GiB`.
  - `mb12`: `num_microbatches=16`; OOM at `optimizer.step` Adam state init,
    PyTorch allocated `137.50 GiB`.
- conclusion: single-GPU AR training is blocked by optimizer-state/single-rank
  memory, not microbatch activation memory. Further microbatch halving is not
  expected to make this run fit without changing optimizer, FSDP sharding, or
  offload behavior.
- cleanup: heavy failed-run artifacts removed after each probe; logs/specs
  retained. Final cluster state after cleanup: no active matching Nano
  processes; both GPUs idle at `4 MiB`.

### stopped: r33-ar-100k-lr1e5-prefix275k-mb16-rerun

- experiment_class: `tuning-probe`
- status: stopped/replaced
- started_at_utc: `2026-06-08T22:33:41Z`
- config: `configs/nano_ar/hpo/r33_100k_lr1e5_cosine_gb192_mb16_rerun.yaml`
- run_id: `nano-ar-r33-100k-lr1e5-cosine-gb192-mb16-rerun`
- run_dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr1e5-cosine-gb192-mb16-rerun`
- launch mode: queue `--once`, two GPUs.
- confirmed wiring:
  - `--actor-num-gpus-per-node 2`
  - `--num-gpus-per-node 2`
  - `--global-batch-size 192`
  - `--micro-batch-size 16`
  - `num_microbatches=6`
- early memory: through the first microbatch loop and first optimizer-step
  scheduler timing without OOM; observed about `75 GiB / 86 GiB` on GPU0/GPU1.
- stopped_at_utc: `2026-06-08T22:38:00Z`
- reason: operator stopped before checkpoint to test the max legal two-GPU
  microbatch setting for `gb192`.
- cleanup: partial heavy artifacts removed; logs/specs preserved.

### complete: r33-ar-100k-lr1e5-prefix275k-mb96-2gpu

- experiment_class: `tuning-probe`
- status: complete
- started_at_utc: `2026-06-08T22:40:37Z`
- completed_at_utc: `2026-06-08T23:38:29Z`
- config: `configs/nano_ar/hpo/r33_100k_lr1e5_cosine_gb192_mb96_2gpu.yaml`
- run_id: `nano-ar-r33-100k-lr1e5-cosine-gb192-mb96-2gpu`
- run_dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr1e5-cosine-gb192-mb96-2gpu`
- launch mode: queue `--once`, two GPUs.
- confirmed wiring:
  - `--actor-num-gpus-per-node 2`
  - `--num-gpus-per-node 2`
  - `--global-batch-size 192`
  - `--micro-batch-size 96`
  - `num_microbatches=1`
- early result: no OOM through first optimizer initialization; logged steps
  `0-13` by `2026-06-08T22:44:25Z`.
- early losses: step 0 `1.3468`, step 1 `1.1237`, step 4 `1.1336`; these
  are warmup losses and not promotion evidence. Step 13 was `1.0902`.
- early memory: after Adam state init, process memory was about
  `99 GiB/GPU`, with allocator-reserved memory around `94-95 GiB` and observed
  max allocated around `82 GiB`.
- eval report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr1e5-cosine-gb192-mb96-2gpu/eval_iter_0000467_v512_t512_winrates_report.json`
- bounded `512/512` eval:
  - validation teacher NMSE `0.441161`, test teacher NMSE `0.419937`
  - validation source_raw NMSE `0.072254`, test source_raw NMSE `0.071656`
  - teacher beats shuffled on both splits, but source_raw remains much stronger
    as expected.
- conclusion: `mb96` is the preferred two-GPU throughput setting, but `lr=1e-5`
  is not the quality winner; it is worse than the prior R33 `100k lr=2e-5`
  checkpoint (`0.334868 / 0.323239` teacher NMSE).

### complete: r33-ar-100k-lr3e5-prefix275k-mb96

- experiment_class: `tuning-probe`
- status: complete
- started_at_utc: `2026-06-09T02:07:40Z`
- completed_at_utc: `2026-06-09T03:08:32Z`
- config: `configs/nano_ar/hpo/r33_100k_lr3e5_cosine_gb192_mb96.yaml`
- run_id: `nano-ar-r33-100k-lr3e5-cosine-gb192-mb96`
- run_dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr3e5-cosine-gb192-mb96`
- launch mode: queue `--once`, two GPUs.
- confirmed wiring:
  - `--actor-num-gpus-per-node 2`
  - `--num-gpus-per-node 2`
  - `--global-batch-size 192`
  - `--micro-batch-size 96`
  - `--lr 3e-5`
  - `num_microbatches=1`
- early result: first optimizer gate cleared without OOM; steady-state memory
  about `98-99 GiB/GPU`.
- eval report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr3e5-cosine-gb192-mb96/eval_iter_0000467_v512_t512_winrates_report.json`
- bounded `512/512` eval:
  - validation teacher NMSE `0.322812`, test teacher NMSE `0.313741`
  - validation source_raw NMSE `0.081054`, test source_raw NMSE `0.075954`
- conclusion: new best R33 100k AR checkpoint so far, narrowly better than the
  prior `lr=2e-5` result (`0.334868 / 0.323239` teacher NMSE).

### complete: r33-ar-100k-final-prehero-lr-schedule-probes

- experiment_class: `tuning-probe`
- status: complete
- purpose: finish the final small AR search around the current best
  `lr=3e-5`, then choose full275k hero AR params from all 100k evals.
- completed probes:
  - `nano-ar-r33-100k-lr2e5-cosine-warmup25-gb192-mb96`
    - completed_at_utc: `2026-06-09T17:15:07Z`
    - validation/test teacher NMSE: `0.348768 / 0.336298`
    - validation/test source_raw NMSE: `0.077899 / 0.073737`
    - conclusion: stable, but not a promotion candidate.
  - `nano-ar-r33-100k-lr3e5-cosine-warmup25-gb192-mb96`
    - completed_at_utc: `2026-06-09T18:17:01Z`
    - validation/test teacher NMSE: `0.321038 / 0.312018`
    - validation/test source_raw NMSE: `0.084906 / 0.078302`
    - conclusion: short warmup slightly improved the `lr=3e-5` result.
  - `nano-ar-r33-100k-lr4e5-cosine-gb192-mb96`
    - training checkpoint completed before the pod eviction; bounded eval was
      run from saved `iter_0000467` after workspace recovery.
    - eval_completed_at_utc: `2026-06-09T20:52:30Z`
    - validation/test teacher NMSE: `0.309019 / 0.301218`
    - validation/test source_raw NMSE: `0.086966 / 0.081296`
    - conclusion: improved over `lr=3e-5`, but later superseded by `lr=5e-5`.
  - `nano-ar-r33-100k-lr5e5-cosine-gb192-mb96`
    - completed_at_utc: `2026-06-09T22:18:20Z`
    - validation/test teacher NMSE: `0.301175 / 0.292956`
    - validation/test source_raw NMSE: `0.093239 / 0.086504`
    - conclusion: first R33 100k AR result to clearly enter the target band on
      test teacher NMSE.
  - `nano-ar-r33-100k-lr5e5-cosine-warmup25-gb192-mb96`
    - completed_at_utc: `2026-06-09T23:19:24Z`
    - validation/test teacher NMSE: `0.300924 / 0.292944`
    - validation/test source_raw NMSE: `0.094529 / 0.088689`
    - conclusion: narrowly best 100k AR candidate by teacher NMSE; selected for
      full275k AR hero promotion.
- cleanup: evaluated non-candidate R33 100k checkpoint trees were removed to
  reduce Longhorn pressure; retained W&B logs, train logs, eval reports, run
  specs, and the selected `lr5e-5 warmup25` checkpoint.

Blocked hero configs are prepared but must not be launched until HPO/eval gates
support promotion:

- `configs/nano_ar/hpo/r33_full275k_hero_queue.yaml`
- `configs/nano_av/hpo/r33_full275k_hero_queue.yaml`

### cleaned: runai-post-contamination-reset-20260610T234644Z

- status: complete
- purpose: reconcile RunAI/local/S3 state after the packed-boundary
  contamination finding and LR-schedule remediation.
- RunAI state before cleanup: workspace `train` running and idle; no active
  Nano train/eval/verify process; GPUs idle.
- queue cleanup: all live `status: pending` queue items were marked
  `cancelled`; backup queue YAMLs preserve the previous pending state.
- artifact evidence:
  - local:
    `artifacts/runai_sync/20260610T234644Z/runai_light_artifacts_20260610T234644Z.tar.gz`
  - S3:
    `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/sync_exports/20260610T234644Z/`
  - archive SHA-256:
    `7bb95e0f6ab98c3c6269f7217459af6f4bda14f1ee708356ae733fe273719db3`
- deleted contaminated checkpoint tree:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-hero/nano-ar-r33-full275k-lr5e5-cosine-warmup25-gb192-mb96/checkpoints`
- deleted bytes: `76,921,456,907`.
- additional cleanup:
  `artifacts/runai_sync/20260610T234644Z/runai_cleanup_manifest_20260611T0005Z.json`
- additional deleted bytes: `11,303,016,711` from reproducible pre-fix
  run-specific split/checkpoint-stub material.
- code sync:
  - local source archive:
    `artifacts/runai_sync/20260610T234644Z/source_code_20260610T234644Z.tgz`
  - source SHA-256: see the adjacent `.sha256` sidecar. Do not embed it here
    because this document is part of the source archive.
  - S3:
    `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/code_sync/20260610T234644Z/`
  - RunAI current symlink:
    `/workspace/interp/code/nano30b-nla-pilot-current -> /workspace/interp/code/nano30b-nla-pilot-sync-20260610T234644Z`
- final disk state after cleanup: `/workspace/interp` about `146G` used,
  `863G` free.
- next clean action: do not launch old queued jobs. Launch only post-fix AR/AV
  confirmation jobs with patch reports, LR canary evidence, and
  packed-vs-padded/preflight evidence.

### failed: r33-dedup-ar-packed-smoke-mb96

- status: failed
- failed_at_utc: `2026-06-11T11:00:34Z`
- config:
  `configs/nano_ar/hpo/r33_dedup_smoke_20k_lr2e5_cosine_warmup20_gb192_mb96.yaml`
- run_id:
  `nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb96`
- run_dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-smoke/nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb96`
- result: live step-0 reward-path/training-path MSE equivalence guard failed
  with `17.9%` max divergence under packed `mb96`.
- interpretation: packed AR training remains unsafe for clean
  Nano/Nemotron-H reruns; do not bypass the guard for promotion runs.

### running: r33-dedup-ar-clean-smoke-mb1-2gpu

- status: training
- started_at_utc: `2026-06-11T11:07:21Z`
- config:
  `configs/nano_ar/hpo/r33_dedup_smoke_20k_lr2e5_cosine_warmup20_gb192_mb1_2gpu.yaml`
- run_id:
  `nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb1-2gpu`
- run_dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-smoke/nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb1-2gpu`
- purpose: clean no-packed AR smoke on the strict content-dedup R33 dataset.
- first observed step: `train/loss=1.2053946`, `train/fve_nrm=-1.1587539`,
  LR `1e-6`, `perf/step_time=387.6s`.
- expected checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-smoke/nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb1-2gpu/checkpoints/iter_0000096`

### blocked: r33-dedup-av-roundtrip-smoke-lr1e4

- status: blocked
- config:
  `configs/nano_av/hpo/r33_dedup_av_20k_lr1e4_cosine_warmup5_gb192_mb1_seq1152_dyn1152.yaml`
- queue:
  `configs/nano_av/hpo/r33_dedup_clean_queue.yaml`
- run_id:
  `nano-av-r33-dedup-20k-lr1e4-cosine-warmup5-gb192-mb1-seq1152-dyn1152`
- study_task: `av_roundtrip`
- purpose: first clean R33 AV smoke using round-trip NMSE as the HPO metric.
- dependency:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-smoke/nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb1-2gpu/checkpoints/iter_0000096`
- baseline:
  `/workspace/interp/outputs/nano30b-nla-pilot/roundtrip/r27_baseline/r27_roundtrip_v64_t64_full_controls_prefix256_report.json`
- protocol: validation/test `64/64`, controls `real`, `shuffled`, `zero`,
  `mean`, and `none`, `max_new_tokens=256`, cached generation, close-rate
  gate `0.8`, usable-rate gate `0.95`, control margin `5e-5`, rowwise
  control-win gate `0.9`.
- verification: local and RunAI queue/config shards passed with `12 passed`.
- gate utility: `scripts/nano_queue_gate.py`.
- gate dry run on RunAI: `ready=false`, `changed=false` while AR was still
  `training`, the expected checkpoint was absent, and `eval_report` was not
  yet set.
- launch rule: use the gate to keep this blocked until the clean AR checkpoint
  and bounded AR eval exist; do not claim R33 NLA quality from AV NLL alone.

## Removed From Live Queue

The following are intentionally no longer queued here:

- legacy sequential gradient-accumulation smoke-harness runs
- fixed-step substitutes for full-data training
- DoRA NaN debugging
- AR or critic work before AV-SFT FSDP2 was validated
- obsolete batch-size drafts now superseded by `gb192` full-data hero

### selected: r33-dedup-av-ar-final-candidate-20260615

- status: selected for next hero-planning phase
- decision_at_utc: `2026-06-15T15:00Z`
- operator decision: assume the clean R33 AV+AR pair is the final candidate and
  skip a new row-matched R27 baseline before hero planning.
- AR checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-clean56k/nano-ar-r33-dedup-clean56k-lr5e5-cosine-warmup25-gb192-mb96-128step-padded/checkpoints/iter_0000128`
- AV checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-dedup-smoke/nano-av-r33-dedup-20k-lr1e4-cosine-warmup5-gb192-mb2-seq1152-dyn512-32steps/checkpoints/iter_0000032`
- AV eval report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-dedup-smoke/nano-av-r33-dedup-20k-lr1e4-cosine-warmup5-gb192-mb2-seq1152-dyn512-32steps/eval_iter_0000032_v512_t512_gen4_report.json`
- Round-trip report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-dedup-smoke/nano-av-r33-dedup-20k-lr1e4-cosine-warmup5-gb192-mb2-seq1152-dyn512-32steps/roundtrip_iter_0000032_v64_t64_report.json`
- Gate read: generated explanations parse cleanly (`closed=1.0`,
  `usable=1.0`, empty `0`), AV-real beats all in-run round-trip controls, and
  AV-real NMSE is `0.000128805 / 0.000135702` validation/test. The report-level
  `passed=false` is due to the non-row-identical R27 baseline comparison and is
  waived by operator decision for this candidate.
- Historical cleanup rule: keep these selected AR/AV checkpoints and compact
  evidence until cleanup and hero-run review. That review completed later on
  `2026-06-15`, superseding the hold on the full component-preserving AV hero
  queue item.
- Cleanup completed at `2026-06-15T15:15:56Z`: non-selected checkpoint/model
  payloads were removed from RunAI, including the stale R33 throughput
  checkpoint tree and non-selected critic-init `model.safetensors`. Selected AR
  `iter_0000128` and AV `iter_0000032` remain present. `/workspace/interp`
  now has about `760G` free.

### complete: r33-component-full-av-hero-lr1e4-warmup25

- experiment_class: `complete-performance`
- status: complete
- last_update_utc: `2026-06-21T15:57Z`
- queue:
  `configs/nano_av/hpo/r33_component_full_hero_queue.yaml`
- config:
  `configs/nano_av/hpo/r33_component_full_hero_lr1e4_cosine_warmup25_gb192_mb2_seq1152_dyn512.yaml`
- run_id:
  `nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512`
- run_dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512`
- expected checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291`
- training recipe:
  `lr=1e-4`, cosine, warmup `25`, `gb192/mb2`, sequence length `1152`,
  dynamic token cap `512`, injection scale `75`, W&B offline.
- corrected AV eval:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/eval_iter_0001291_v512_t512_gen8_report.json`
- round-trip report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/roundtrip_iter_0001291_v256_t256_report.json`
- final proof:
  - corrected AV eval `512/512` real NLL validation/test
    `0.798672 / 0.819993`, beating shuffled, zero, mean, and no-injection
    controls.
  - generated-text round-trip `256/256` gate passed with validation/test
    AV-real NMSE `0.000109680 / 0.000121664`, teacher-text NMSE
    `0.000106810 / 0.000112370`, and closed/usable parse fractions
    `1.0 / 1.0`.
  - AV-real beat `mean`, `av_mean`, `av_none`, `av_zero`, and `av_shuffled`
    round-trip controls on both validation and test.
- preservation:
  compact local evidence archive
  `artifacts/runai_sync/20260621T155000Z_r33_component_full_hero/20260621T155000Z_r33_component_full_hero_compact.tgz`
  and S3 prefix
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/sync_exports/20260621T155000Z_r33_component_full_hero/`.
- cleanup:
  the superseded component-full AV smoke checkpoint payload was deleted after
  evidence freeze, freeing `59G`; selected AR/AV hero checkpoints remain on
  RunAI.

### prepared: r33-stage1-independent-validity-512

- experiment_class: `validity-evaluation`
- status: running
- last_update_utc: `2026-06-30T22:47Z`
- queue:
  `configs/nano_rl/r33_component_stage1_roundtrip_queue_8h100.yaml`
- candidates:
  clean R33 component-full AV-SFT, fixed-AR RL update 16, and fixed-AR RL
  update 32.
- protocol:
  512 validation plus 512 test rows, stable source provenance, selected R33 AR
  scorer, generation shard resume, and temporary DCP-to-HF conversion in
  `/dev/shm`.
- reuse:
  SFT and update-16 reuse the existing 256/256 generations and create only the
  missing half; update-32 reuses its existing 512/512 generation.
- guardrail:
  evaluation only. No corrected RL probe or hero run may start until Stage 1
  functional, invariance, parser-health, and paired round-trip evidence is
  reviewed.
- implementation calibration:
  the hard identity gate is fresh R33 capture/reinjection logit equality.
  Historical stored-vector drift is retained as cross-topology evidence and
  `stored_gold` supplies the current-runtime replay floor.
- launch:
  generation/scoring watcher started at `2026-06-30T22:26Z`. SFT conversion
  completed, eight generation workers loaded, and shard counts reached
  `77-80/128` by `22:47Z` with no OOM or worker failure.
- continuation:
  `configs/nano_rl/r33_component_stage1_analysis_queue_8h100.yaml` is armed
  through `scripts/nano_queue_chain.py`. It starts only after every item in the
  first queue is complete and stops on any failed or blocked prerequisite.
- review guard:
  qualitative reports contain 50 validation and 50 test rows and remain
  promotion-blocking until every row has an explicit review decision.
- closure audit:
  update-32 validation/test coverage by token 192 was only
  `0.585938 / 0.519531`; token 256 reached `0.992188 / 0.982422` and is
  retained for corrected probes and hero planning.
- functional throughput:
  the analysis queue runs an update-16 `16/16`, batch-8 canary before both
  512/512 functional reports. It selects the 16 longest prefixes per split.
  Full batch 8 proceeds only when the canary exits cleanly; no silent fallback
  is allowed.
- panel lookup:
  provenance-key-first source scanning reduced the measured 1,024-row panel
  build from more than 100 seconds to 11 seconds on the full 275k parquet.
  The smoke panel remains review-pending by design.
- update32 qualitative review:
  complete at 50/50 rows per split. Validation/test flags are `0/1`; test row
  `262022` contains a repeated ISSN-like zero sequence. The reviewed report
  passes the 5% qualitative threshold, SHA256
  `d4476dd160587fc16db887884259a92da5f6a04f68adebf43460cf0a563e6a8a`.
- matched SFT baseline:
  complete at 512/512. Validation/test real NMSE is
  `0.000126796 / 0.000134752`; closed fraction is
  `0.978516 / 0.972656`; report SHA256
  `80a8222ee13ac7fa0172d0cf7c07ded64bb18c72b19870c9b9d5847d19fe23fe`.
- update32 paired preview:
  row identities exactly match SFT. Real NMSE is `26.76% / 28.69%` lower and
  paired bootstrap CIs are strictly positive on both splits. Round-trip Stage
  1 checks pass; composite promotion still awaits invariance and functional
  reports.

### superseded: r33-stage1-independent-validity-512

- status: `superseded_for_promotion`
- last_update_utc: `2026-07-03T14:26:14Z`
- reason:
  the historical Stage-1 queue remains useful diagnostic evidence, but the
  active recipe now uses the independently verified train-only R33 dataset,
  exact `gb384/mb32` actor batches, K3 KL, unified runtime preflight, and the
  corrected full-prefix generated-text gate. Do not use the historical
  update-32 preview to unblock the current hero queue.

### completed: r33-corrected-k3-probe-lr1e5-update8-unifiedenv-retry1

- experiment_class: `rl-hpo-probe`
- status: `complete_selected`
- completed_utc: `2026-07-01T23:13:07Z`
- topology:
  8 H100 NVLs split as 6 actor + 1 frozen critic + 1 managed SGLang rollout.
- dataset:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_rl_train_only/rl_r33_train_only.parquet`
  with verifier pass: `247,700` rows, `24,867` documents, layer `33`,
  `d_model=2688`, zero nonfinite rows, duplicates, or heldout overlap.
- recipe:
  `lr=1e-5`, K3 coefficient `1e-3`, `gb384/mb32`, 48 prompts x 8 samples,
  eight updates, constant LR, W&B offline.
- training:
  all eight updates and live synchronizations completed; drift
  `0.2632-0.3285`; final K3 `10.9046`; steady step time `155-159s`.
- valid gate:
  256/256 generated-text round trip passed. Validation/test AV-real NMSE
  `0.0001085655 / 0.0001195005` versus matched SFT
  `0.0001096657 / 0.0001216750`; closed/usable fractions `1.0 / 1.0`; all
  controls passed.
- checkpoint:
  update 8, `59G`, last verified present on the mounted PVC on `2026-07-02`.
  Reverify after workspace redeploy.
- evidence:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/rl_evidence/20260701T231539Z/r33_corrected_k3_probe_lr1e5_update8_unifiedenv_retry1_20260701T231539Z.tgz`,
  SHA256
  `2029d86b49d6f72c9b0cd333839d9dd6ae40fd5ad889d000f8f4921d9d778419`.

### completed: r33-corrected-k3-probe-lr2e5-update8

- experiment_class: `rl-hpo-probe`
- status: `complete_not_selected`
- completed_utc: `2026-07-02T02:52:28Z`
- recipe:
  same exact-384 topology as `lr=1e-5`, with actor `lr=2e-5`.
- training:
  all eight updates completed; drift `0.2837-0.3420`; K3 had early spikes
  `599.9822 / 102.9060` before recovering below `0.1`.
- invalid report:
  the cache-backed 256/256 report is diagnostic only. Cached tokens diverged
  from full-prefix tokens at generated token index 1 for batch sizes 5 and 1.
- valid gate:
  replacement 64/64 `legacy_batch` report passed. Validation/test AV-real
  NMSE `0.0001069578 / 0.0001179766` versus matched SFT
  `0.0001095636 / 0.0001207099`; closed/usable fractions `1.0 / 1.0`; all
  controls passed.
- selection:
  mixed versus `lr=1e-5`: validation `1.64%` worse, test `3.86%` better,
  combined mean `1.32%` lower, `45.31%` rowwise wins, paired bootstrap interval
  crossing zero. Not selected because the short comparison is inconclusive
  and its optimization transients are more severe.
- cleanup:
  actor checkpoint deleted after evidence freeze; logs, W&B, valid/invalid
  reports, generated text, and cache-equivalence logs retained.
- evidence:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/rl_evidence/20260702T025445Z/r33-corrected-k3-lr2e5-update8-evidence.tgz`,
  SHA256
  `c49bd2bf35d18773cd6361b621b9d8c60a8657b6b01211a125fb452d799a3469`.

### aborted: r33-corrected-k3-confirm-lr1e5-update32

- experiment_class: `rl-medium-confirmation`
- status: `aborted_before_update1_invalid_runtime`
- attempted_utc: `2026-07-03T18:00:51Z`
- stopped_utc: `2026-07-03T18:47Z`
- config:
  `configs/nano_rl/r33_component_corrected_k3_hpo_queue_8h100.yaml`
- selected recipe:
  `lr=1e-5`, K3 coefficient `1e-3`, exact `gb384/mb32`, eight samples per
  prompt, 32 updates, 6+1+1 H100 topology, W&B offline.
- invalidation:
  the separate live Miles runtime was stale and still used raw DTensor
  `clip_grad_norm_` plus `full_tensor()`. The queue source fingerprint did not
  cover that external directory. The attempt was stopped before update 1,
  produced no checkpoint or eval, and is excluded from scientific evidence.
- retained evidence:
  train/W&B/service logs, source provenance, and
  `queue_aborted_runtime.yaml` remain in the run directory.

### prepared: r33-corrected-k3-confirm-lr1e5-update32-runtimefix-retry1

- experiment_class: `rl-medium-confirmation`
- status: `training`
- started_utc: `2026-07-03T19:09:05Z`
- config:
  `configs/nano_rl/r33_component_corrected_k3_hpo_queue_8h100.yaml`
- correction:
  RunAI Miles actor and arguments were reconciled idempotently to local-shard
  gradient clipping. Their exact SHA256 values are queue-bound runtime
  contracts, and launch now writes `runtime_contracts.json`. SGLang staging
  releases only model tensor shards after health to avoid the 256 GB cgroup
  startup pressure. The retry uses a new run directory and W&B run ID.
- startup note:
  the first retry launch exposed that queue normalization dropped the nested
  release declaration. The already-loaded 13 tensor shards (63,156,683,832
  bytes) were unlinked manually before actor initialization reached its memory
  peak; a release manifest records the intervention. The parser is corrected
  and regression-tested for subsequent launches.
- required gate:
  compare updates 16 and 32 on 64/64 first, then run 512/512 only after the
  update-32 versus update-16 gate passes. The final gate requires exact dataset
  identity, document-clustered positive confidence intervals, more than 50%
  row wins, at least 10% relative improvement on both splits, healthy parse
  rates, all controls, bounded drift, and no repeated K3 instability.
- hero dependency:
  the fixed-AR hero queue requires both the missing Stage-2 gate and a missing
  independent cross-critic gate. Regenerate its actual training recipe only
  after the confirmation and those audits pass.

### platform: runai-train-workspace

- status: `running_idle`
- checked_utc: `2026-07-03T19:03:00Z`
- project: `trustworthy-ai-inference`
- observation:
  `train` is Running on 8 H100 NVLs. All GPUs report `4 MiB` used and zero
  utilization. No Nano training, evaluation, queue driver, or SGLang process
  is active.
- mounted storage:
  `/workspace/interp` `653G` used / `355G` free; `/workspace/models` `454G`
  free. Stale backup/core artifacts were removed without deleting experiment
  evidence.
- launch state:
  the runtime-fixed confirmation retry is training and the hero queue remains
  blocked pending its gates.

### failed: r33-corrected-k3-confirm-lr1e5-update32-runtimefix-retry1

- experiment_class: `rl-medium-confirmation`
- status: `failed_platform_host_memory_oom`
- started_utc: `2026-07-03T19:09:05Z`
- failed_utc: `2026-07-03T19:39:49Z`
- evidence:
  rollout 0 completed and rollout 1 began, but Kubernetes OOM-killed the pod
  at the 256 GB CPU-memory cgroup limit. No checkpoint or verifier report was
  produced, so this run is excluded from model-quality evidence.
- retained:
  train/service/W&B logs, source/runtime contracts, and release manifest.

### aborted: r33-corrected-k3-confirm-lr1e5-update32-runtimefix-retry2-mem512

- experiment_class: `rl-medium-confirmation`
- status: `aborted_host_memory_near_oom`
- config:
  `configs/nano_rl/r33_component_corrected_k3_hpo_queue_8h100.yaml`
- recipe:
  unchanged `lr=1e-5`, K3 `1e-3`, `gb384/mb32`, 48 prompts x 8 samples,
  32 updates, 6 actor + 1 critic + 1 rollout H100.
- platform correction:
  `train` was redeployed with an exact 512 GB CPU-memory cgroup while keeping
  the same 8 H100 NVLs and both PVCs.
- outcome:
  update 0 completed, but update 1 peaked at 511,997,681,664 of
  512,000,000,000 cgroup bytes. It was stopped cleanly before an OOM and
  produced no checkpoint or verifier report.
- promotion:
  superseded by the identical `retry3-mem768` recipe.

### prepared: r33-corrected-k3-confirm-lr1e5-update32-runtimefix-retry3-mem768

- experiment_class: `rl-medium-confirmation`
- status: `pending_workspace_redeploy`
- recipe:
  unchanged `lr=1e-5`, K3 `1e-3`, `gb384/mb32`, 48 prompts x 8 samples,
  32 updates, 6 actor + 1 critic + 1 rollout H100.
- sole correction:
  raise the RunAI CPU-memory request/limit from 512 GB to 768 GB.
- promotion:
  retry3 -> independent R33 AR critic -> Stage-2 validity/cross-critic queue
  -> armed 342-update hero. Every edge fails closed on queue/report status.

### prepared: r33-corrected-k3-hero-lr1e5-update342

- experiment_class: `rl-hero-qwen-comparable-budget`
- status: `blocked_on_unattended_promotion_chain`
- config:
  `configs/nano_rl/r33_component_corrected_k3_hero_342_queue_8h100.yaml`
- budget:
  342 updates x 384 generated samples = 131,328 rollouts.
- checkpoints/eval:
  retain updates 171 and 342; evaluate 171 at 64/64, 342 versus 171 at
  64/64, then 342 versus hardened SFT at 512/512 only after the short gate.

### failed: r33-corrected-k3-confirm-lr1e5-update32-runtimefix-retry3-mem768

- experiment_class: `rl-medium-confirmation`
- status: `failed_dynamic_metric_schema_after_rollout22`
- started_utc: `2026-07-03T20:53:59Z`
- failed_utc: `2026-07-03T22:27:08Z`
- durable artifact:
  model-only actor/critic `iter_0000016`; no bounded verifier report.
- failure:
  after the rollout-22 optimizer step, positional loss aggregation attempted
  to add local microbatch vectors with lengths 47 and 41.
- classification:
  observability aggregation bug, not an OOM or promotion result.

### completed: r33-corrected-k3-confirm-lr1e5-update32-runtimefix-retry4-keyedloss-mem768

- experiment_class: `rl-medium-confirmation`
- status: `complete_quality_gate_failed`
- recipe:
  unchanged `lr=1e-5`, K3 `1e-3`, `gb384/mb32`, 48 prompts x 8 samples,
  32 updates, 6 actor + 1 critic + 1 rollout H100, 768 GB host-memory limit.
- correction:
  dynamic telemetry is aggregated by key with per-metric denominators and a
  DP/CP-wide deterministic key union.
- runtime contract:
  actor SHA256 `7db9b4acfbc7af734dee736c8a549cdd5a3f6d31c46e4d7d53f8028b62357479`.
- result:
  all 32 updates completed without OOM or runtime failure. Update 16 improved
  matched 64/64 mean NMSE by `2.35% / 2.60%` validation/test, but clustered
  confidence intervals crossed zero. Update 32 was effectively flat versus
  update 16 on test (`0.24%`). The 512/512 promotion check skipped and no hero
  launched.

### completed: r33-retry4-update16-v512-hardened-diagnostic

- experiment_class: `roundtrip-eval-only`
- status: `complete_gate_passed`
- config:
  `configs/nano_rl/r33_retry4_update16_v512_diagnostic_queue_8h100.yaml`
- purpose:
  resume the valid update-16 64/64 rows to 512/512 and test the existing 10%
  hero margin with row wins, positive document-clustered paired confidence
  intervals, exact dataset identity, parse health, and all controls.
- safety:
  no training and no teacher generation; DCP conversion is temporary in
  `/dev/shm` and is deleted after scoring.
- orchestration correction:
  chained promotion now requires explicit `gate_passed: true`; false or
  missing gates fail closed. The independent critic source is pinned to
  `nano30b-nla-pilot-hero-current`.
- result:
  validation/test relative NMSE improvements over matched SFT are
  `14.53% / 14.86%`; clustered confidence intervals are strictly positive,
  row wins are `62.89% / 58.40%`, parse health is `100%`, dataset identity
  matches, and all controls are beaten. Update 16 is selected for Stage 2.

### running: r33-component-full-independent-critic-seed314159

- experiment_class: `ar-sft-independent-audit-critic`
- status: `failed_h100_cuda_oom`
- config:
  `configs/nano_ar/hpo/r33_component_full_independent_critic_queue.yaml`
- purpose:
  train an independently shuffled R33 AR critic on the same component split,
  then rescore SFT and selected update-16 text to detect reward-model-specific
  exploitation before hero launch.
- source correction:
  the fresh run uses `nano30b-nla-pilot-hero-current`; the earlier stale-source
  startup attempt is preserved separately and produced no checkpoint.
- failure:
  the two-GPU `mb96` recipe reached `91.19 GiB` on a 93.10-GiB H100 NVL and
  failed when the next backward allocation requested `2.42 GiB`. It completed
  only two optimizer steps and produced no checkpoint or eval report.

### failed: r33-component-full-independent-critic-seed314159-4gpu-mb48

- experiment_class: `ar-sft-independent-audit-critic`
- status: `failed_cuda_illegal_memory_access_step391`
- config:
  `configs/nano_ar/hpo/r33_component_full_independent_critic_retry_4gpu_queue.yaml`
- correction:
  use 4 FSDP H100s at `mb48` while preserving exact `gb192`, `lr=5e-5`,
  warmup 25, cosine decay, component split, and shuffle seed 314159. The
  per-update sample count and optimization recipe are unchanged.
- observed behavior:
  training was stable through step 391 at roughly 6 seconds/step and
  57-62 GiB/GPU. It then failed with `CUDA error: an illegal memory access was
  encountered`; the first synchronous frame was the optional router-entropy
  forward hook copying router indices to CPU. NCCL aborted only after the CUDA
  context was poisoned. No checkpoint or eval report exists.

### prepared: r33-component-full-independent-critic-seed314159-4gpu-mb48-norouter

- experiment_class: `ar-sft-independent-audit-critic`
- status: `pending_launch`
- config:
  `configs/nano_ar/hpo/r33_component_full_independent_critic_retry2_norouter_queue.yaml`
- correction:
  preserve the exact successful-through-step-391 training recipe and fresh
  4-GPU topology while disabling only optional router-entropy observability.
  The retry has a fresh run/W&B identity, and Stage 2 requires only its final
  `iter_0001289` checkpoint.

### failed: r33-component-full-independent-critic-seed314159-4gpu-mb48-norouter

- experiment_class: `ar-sft-independent-audit-critic`
- status: `failed_cuda_illegal_memory_access_rollout392`
- completed evidence:
  step 391 finished with loss `0.2966566`, FVE `0.4686700`, finite gradient
  norm `0.400390625`, and approximately 59 GiB/GPU. The next deterministic
  rollout failed in `nla/nemotron_moe.py` at the segmented GPU
  `torch.bincount` path. No checkpoint or eval report exists.
- conclusion:
  disabling router telemetry did not move the exact failure boundary, so the
  telemetry hook only synchronized the asynchronous fault in the prior run.

### prepared: r33-component-full-independent-critic-seed314159-4gpu-mb48-expertscan

- experiment_class: `ar-sft-independent-audit-critic`
- status: `pending_launch`
- config:
  `configs/nano_ar/hpo/r33_component_full_independent_critic_retry3_expertscan_queue.yaml`
- controlled change:
  preserve all data, seed, topology, batch, optimizer, LR, schedule, and eval
  settings while selecting `training.moe_routing_impl: expert_scan`. The
  stock-style per-expert route avoids GPU sort/bincount; H100 forward and all
  tested gradients match segmented routing, with zero maximum forward error.

### failed: r33-component-full-independent-critic-seed314159-4gpu-mb48-expertscan

- experiment_class: `ar-sft-independent-audit-critic`
- status: `failed_cuda_illegal_memory_access_rollout392`
- result:
  step 391 again completed normally (loss `0.2965752`, FVE `0.4688158`,
  gradient norm `0.37890625`). Rollout 392 then surfaced the asynchronous CUDA
  fault at expert-scan's first `torch.where` over router indices. No checkpoint
  or eval report exists.
- conclusion:
  segmented routing, router telemetry, and expert-scan routing are all ruled
  out as root causes; the fault originates earlier in the model forward.

### prepared: r33-independent-critic-cudablock-diag393

- experiment_class: `ar-sft-runtime-diagnostic`
- status: `pending_launch`
- config:
  `configs/nano_ar/hpo/r33_component_full_independent_critic_retry4_cudablock_diag_queue.yaml`
- purpose:
  run only through update 393 with `CUDA_LAUNCH_BLOCKING=1` so the true failing
  kernel is reported synchronously. Save full optimizer-bearing state at update
  384 to make subsequent diagnostics short and restartable. The corrected
  launch pins `lr_decay_iters=1289`; an initial run stopped at step 143 after
  detecting that an implicit 393-step decay horizon changed the trajectory.

### diagnostic-complete: r33-independent-critic-cudablock-diag393

- experiment_class: `ar-sft-runtime-diagnostic`
- status: `training_complete_eval_intentionally_stopped`
- result:
  the full-horizon run crossed rollout 392 under `CUDA_LAUNCH_BLOCKING=1`.
  Step 391 was loss `0.2985217`, FVE `0.4653295`; step 392 was loss
  `0.2844250`, FVE `0.4905775`. The exact batch failed in every asynchronous
  retry, so the fault is timing/race-sensitive rather than data or routing.
- checkpoint:
  full DCP state exists at iterations 384 and 393 on `/workspace/models`; the
  tracker selects 393. Longhorn remained at 262 GiB free.
- eval:
  the non-gating 64/64 eval was stopped during redundant NFS reconstruction;
  its partial log is retained and no quality result is claimed.

### loading: r33-independent-critic-cudablock-resume393

- experiment_class: `ar-sft-independent-audit-critic`
- status: `loading_optimizer_checkpoint`
- queue:
  `configs/nano_ar/hpo/r33_component_full_independent_critic_resume393_cudablock_queue.yaml`
- config:
  `configs/nano_ar/hpo/r33_component_full_independent_critic_seed314159_4gpu_mb48_cudablock_resume393.yaml`
- contract:
  restore iteration 393 model, optimizer, scheduler, and dataset offset; keep
  `lr_decay_iters=1289`, run 896 remaining updates with launch blocking, and
  save only `iter_0001289` model state. Stage 2 points to that exact final
  checkpoint and remains pending.

### archived-invalid: r33-independent-critic-cudablock-resume393-wrong-lr

- status: `stopped_no_checkpoint`
- finding:
  DCP restored scheduler epoch 393 correctly, but the Miles post-load policy
  forced every optimizer resume to constant LR `5e-5`. The run was stopped
  around step 410 and archived as
  `.wrong_scheduler_reset_20260704T225829Z`; it is not promotion evidence.

### running: r33-independent-critic-cudablock-resume393

- status: `training_corrected_cosine_resume`
- evidence:
  restored cosine at epoch 393 with live LR `4.1226691e-5`; logged step 393
  decayed to `4.1182339e-5`, and step 408 reached `4.0507070e-5`. Loss and FVE
  remain finite and in the prior trajectory band. W&B is offline.
- storage:
  retained recovery checkpoint `iter_0000393`; deleted redundant
  `iter_0000384`. Model-store free space is 255 GiB.
- promotion:
  the critic-to-Stage-2 and Stage-2-to-hero watchers are active. Stage 2 and
  hero remain pending and fail-closed on their declared reports.

### complete: r33-independent-critic-cudablock-resume393

- checkpoint: `iter_0001289` model-only DCP
- bounded eval:
  teacher NMSE `0.3208674/0.2924067` validation/test; source-raw
  `0.0942281/0.0800047`; shuffled, blank, generic, and mean controls strongly
  beaten.
- queue completion: `2026-07-05T01:39:17Z`

### passed: r33-corrected-confirm16-independent-roundtrip

- independent SFT baseline NMSE: `0.0001269159/0.0001344858`
- candidate NMSE: `0.0001081018/0.0001144881`
- relative improvement: `14.824%/14.870%`
- paired candidate wins: `66.99%/59.96%`
- gate evidence: positive doc-clustered 95% CIs, all controls beaten, 100%
  parse close/usable rates.
- next state: Stage 2 invariance/functional/closure/qualitative/composite
  checks running; hero remains pending.

### passed: r33-corrected-stage2

- combined cross-critic:
  passed after canonical row identity was corrected to dataset hash + parquet
  row index + document ID. The original failure compared optional
  `n_raw_tokens` enrichment as identity; the failed and corrected logs are
  both retained.
- invariance:
  format-normalized FVE retention `100.01%/99.96%`; unit-reordered retention
  `99.10%/98.77%` validation/test.
- closure:
  full close rate 100% on both splits; cap 192 closes `100%/99.80%` and meets
  the 95% requirement.
- functional caveat:
  reinjection identity passed all 1,024 rows; strict stored-activation replay
  remained outside tolerance on all rows but is not a composite-gate input.
- reports:
  both cross-critic and composite Stage 2 JSONs record `passed: true` with no
  composite blockers.

### guard-stopped: r33-corrected-k3-hero-lr1e5-update342

- started: `2026-07-05T02:57:23Z`
- topology: 6 actor + 1 rollout + 1 critic H100 NVL
- source fingerprint:
  `aef659279c9306f4818812b0b9eb0cbd24df0d857c562d323f4da221524c32a4`
- stopped: `2026-07-05T06:02:39Z`
- progress: 64 rollout batches (IDs 0-63), 24,576 generated responses, or
  `18.71%` of the planned hero budget. Normal actor logs cover steps 0-62.
- stop reason: configured actor guard fired at step 63 after consecutive
  `train/kl_loss` values `25.9797955` and `5.1258636` exceeded threshold
  `5.0`. No CUDA, OOM, NCCL, or rollout-service error was present.
- partial signal: first-ten to last-ten raw reward improved
  `-0.346596 -> -0.272935`; reward std fell `0.224467 -> 0.161619`; aggregate
  close/usable rates were `99.784%/99.752%`, with zero truncation.
- instability: KL median `0.6301`, p95 `14.3117`, maximum `233.0444`; large
  KL values coincided with gradient-norm spikes. Log-prob drift stayed below
  `0.304` versus guard `0.75`.
- artifacts: retained `train.log`, runtime/source contracts, SGLang logs, and
  four offline W&B role runs in the run directory.
- result: no checkpoint (first save was update 171), no post-eval, and no hero
  promotion claim. Stage 2 remains passed; hero remains open. No retry queued.

### guard-stopped: r33-corrected-k3-hero-lr1e5-update342-guard3-retry1

- config commit: `fe71bd8`
- queue:
  `configs/nano_rl/r33_component_corrected_k3_hero_342_guard3_retry1_queue_8h100.yaml`
- run directory:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_hero/r33_corrected_k3_hero_lr1e5_update342_guard3_retry1`
- started: `2026-07-06T16:42:47Z`
- topology/batch: six actor + one rollout + one frozen critic H100;
  48 prompts x 8 samples = global batch 384 with no dropped samples.
- training: actor LR `1e-5`, K3 coefficient `1e-3`, 342 updates, offline W&B.
- guards: KL above 5 for three consecutive steps; actor/rollout log-prob
  difference above 0.75 for two; gradient norm above 100 for two; existing
  parser and response-length guards retained.
- checkpoints: save 114/228/342, retain 114/342; evaluate update 114 at 64/64,
  update 342 versus 114 at 64/64, then update 342 versus SFT at 512/512.
- cleanup: obsolete core dumps, redundant critic iteration 393, and three
  superseded actor model-shard sets removed. Free space after cleanup:
  460 GiB Longhorn and 456 GiB model store. Logs/reports retained.
- verified start: SGLang health 200; four W&B role runs; rollout 0 generated
  all 384 samples (`98.44%` closed, `97.40%` usable, zero truncation);
  optimizer step 0 completed with KL `0`, gradient norm `1.1172`, log-prob
  difference `0.27250`, entropy `1.24026`, and LR `1e-5`.
- stopped: `2026-07-06T18:02:11Z` during rollout 26.
- stop reason: the inherited relative p95 rule fired after four ordinary
  increases `160.85 -> 163.70 -> 166.00 -> 170.55`; p95 remained below the
  256-token cap and truncation was zero.
- last completed actor step: update 25, KL `3.67496`, grad `4.5625`, log-prob
  difference `0.24050`; no CUDA/OOM/NCCL/model guard failure.
- result: no checkpoint or post-eval; superseded by the absolute length-cap
  retry below.

### guard-stopped: r33-corrected-k3-hero-lr1e5-update342-guard3-lengthcap-retry2

- config commit: `1087f5f`
- queue:
  `configs/nano_rl/r33_component_corrected_k3_hero_342_guard3_lengthcap_retry2_queue_8h100.yaml`
- run directory:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_hero/r33_corrected_k3_hero_lr1e5_update342_guard3_lengthcap_retry2`
- started: `2026-07-07T00:13:25Z`
- restart semantics: clean restart from selected AV-SFT actor because retry 1
  had no checkpoint; no stale optimizer or rollout state reused.
- length guards: p95 above 230 for two consecutive rollouts; truncated
  fraction above 5% for two. Relative monotonic rule removed.
- unchanged safeguards: close/usable parser gates, KL above 5 for three actor
  steps, log-prob difference above 0.75 for two, gradient norm above 100 for
  two, Stage-2 prerequisites, and all post-eval gates.
- verified start: SGLang health passed; four offline W&B role runs; rollout 0
  generated 384 samples with p95 `168.85`, zero truncation, `97.14%` close,
  and `96.61%` usable. Step 0 completed with KL `0`, grad `1.08594`, log-prob
  difference `0.27939`, entropy `1.20782`, and LR `1e-5`.
- stopped: `2026-07-07T13:52:32Z` during rollout 253.
- stop reason: response-length p95 exceeded the 230-token limit twice
  (`233.85`, `232.70`) despite zero truncation and healthy parsing.
- last completed actor step: update 252, KL `3.01199`, grad `1.46875`,
  log-prob difference `0.27546`; no model/runtime guard failure.
- retained checkpoint: actor update 228. No post-eval ran.

### complete: r33-corrected-k3-hero-lr1e5-update342-resume228-retry3

- queue:
  `configs/nano_rl/r33_component_corrected_k3_hero_342_resume228_retry3_queue_8h100.yaml`
- run directory:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_hero/r33_corrected_k3_hero_lr1e5_update342_resume228_retry3`
- started: `2026-07-07T17:52:05Z`
- continuation semantics: load retry-2 update 228 model/RNG/counter plus dataset
  state 227; restart unavailable Adam state; continue to absolute endpoint 342.
- guard change: response-length p95 remains logged but no longer aborts.
  Truncation above 5%, parser health, KL, gradient, and log-prob guards remain.
- verified continuation: all actor ranks loaded update 228, dataset state 227
  loaded, and rollout/optimizer step 228 completed. Reward mean `-0.20660`,
  p95 `222`, zero truncation, KL `0.92023`, grad `0.6875`, and log-prob
  difference `0.27336`.
- training completed: `2026-07-07T23:17:47Z`; actor step 341 wrote selected
  model-only checkpoint `actor/iter_0000342`.
- short gate: `64/64`, passed; validation/test relative improvement over
  matched SFT `25.01% / 28.32%`.
- promotion gate: `512/512`, passed; validation/test RL AV-real NMSE
  `0.000087528 / 0.000091176`, relative improvement `30.97% / 32.34%`,
  rowwise wins `83.40% / 88.67%`, and usable generations `100% / 100%`.
- provenance: baseline dataset hashes and exact row identities matched; all
  512 rows overlapped on both splits; document-clustered confidence intervals
  were positive across 52 documents/split.
- controls: AV-real beat teacher, mean, AV-mean, no-injection, zero, and
  shuffled variants by aggregate normalized MSE.
- queue completed: `2026-07-08T03:20:52Z`; temporary HF exports cleaned; no
  active Nano train/eval process and all eight GPUs returned to idle.
- evidence: `docs/runs/r33_rl_hero_20260708.md`; local/S3 lightweight archive
  SHA-256 `78cbf98d27188594c25cbf9c0d695f0b3b1754df978961585bbaa6fc178f0bc7`.
- decision: selected internal R33 RL hero checkpoint. External R33-over-R27
  claims still require a fresh row-matched R27 comparison and replication.

## 2026-07-09 Deterministic Publication Dataset Gate

### complete: r33-deterministic-full275396 extraction and verification

- source commit: `0dabaade33ee35a3ff7419d2f99be2551439ab13`.
- deterministic profile: deterministic algorithms on, TF32 off, cuDNN
  benchmarking off, matmul precision `highest`, cuBLAS workspace `:4096:8`,
  seed `20260709`.
- rows: base/AR-SFT/AV-SFT all `275,396`; `d_model=2688`.
- data checks: nonfinite activations `0`, empty explanations `0`, malformed AV
  responses `0`, bad AR suffixes `0`, bad AV prompt markers `0`.
- split checks: train/validation/test rows `247,865 / 13,766 / 13,765`;
  families `4,504 / 250 / 255`; document/family/content overlap all `0`.
- exact replication: a second complete eight-shard extraction produced the
  same merged Parquet SHA-256,
  `e3008a150831b8e894eac0de9f360a46823ffbfbd7cc73a9673f7e61e84521ac`.
- primary critic init: complete at
  `/workspace/models/nano30b-nla-pilot/publication/r33_deterministic_critic_init`.
- training status: not launched. Next jobs are the seed-`314159` independent
  critic init and clean family-disjoint AR/AV SFT queues.

### ready: r33 deterministic clean AR/AV SFT queues

- independent critic init: complete at
  `/workspace/models/nano30b-nla-pilot/publication/r33_deterministic_independent_critic_init_seed314159`.
- critic independence gate: `passed=true`; distinct value-head hash and changed
  router parameters, with the same base/data/layer/dtype as the primary.
- immutable launch code/config snapshot:
  `294cc1e1619c42ea454c8d29e9f477e7ae4d4322`; readiness-evidence full-tree
  fingerprint
  `1c7c0abbad68bebc426acff74ea1b14ed1559adacda28496d2b526154032c020`.
- remote tests: `65 passed`.
- primary AR dry-run: exit `0`, `2 GPU`, `gb192/mb96`, `lr=5e-5`,
  cosine/warmup 25, validation-only eval, queue remains `pending`.
- independent AR dry-run: exit `0`, same recipe with the seed-`314159`
  critic, validation-only eval, queue remains `pending`.
- AV dry-run: exit `0`, `2 GPU`, `gb192/mb2`, `lr=1e-4`,
  cosine/warmup 25, dynamic 512-token cap, validation-only eval, queue remains
  `pending`.
- AV queue fix: commits `4e23755` and `294cc1e` added a shared no-side-effect
  dry-run planner and JSON-safe plan output. Both pre-fix failure logs are
  retained.
- training status: not launched; all eight H100s idle after verification.
- evidence archive:
  `artifacts/runai_evidence/20260709_r33_publication_remediation/clean_sft_queue_readiness/r33_clean_sft_queue_readiness_evidence_20260709.tgz`,
  SHA-256
  `f6045b10e1e4573635c00cd49418137870b9309a2323faeb82db9e3adcb85c3a`.
- S3 status: upload pending RunAI egress-proxy recovery; local and RunAI copies
  are checksum-identical.

### retrying: r33 deterministic clean AR/AV SFT launch

- primary AR attempt 1: failed before step 0 because the default 10-minute
  c10d timeout expired during asymmetric cold checkpoint paging; no checkpoint.
- timeout fix: commit `658a2ef`, `distributed_timeout_minutes: 60`, RunAI
  focused tests `51 passed`.
- primary AR attempt 2: reached step 0 and passed reward/train equivalence, then
  OOMed in FSDP post-backward at `91.99 GiB` used plus a `1.21 GiB` request;
  no checkpoint.
- diagnosis: historical `2 GPU / mb96` geometry does not fit the 93.1 GiB
  H100-NVL memory envelope. The existing H100-safe geometry is `4 GPU / mb48`.
- next primary run: fresh `4gpu-mb48` identity, exact `gb192`, LR `5e-5`,
  cosine/warmup 25, deterministic family split, validation-only 512-row eval.
- AV attempt 1 used `2 GPU / gb192 / mb2 / dynamic 512`; its first actor pass
  completed all 48 packed microbatches in `465.1s`, then a true-capacity OOM
  occurred before the optimizer-step log (`91.68 GiB` allocated, `20 MiB`
  requested, about `7 MiB` free). No checkpoint was produced.
- AV retry uses a fresh `4 GPU / gb192 / mb2 / dynamic 512` identity on GPUs
  4-7. The global batch, token cap, data, LR, and validation protocol are
  unchanged; rank-local FSDP state and packed microbatch count are reduced.
- independent AR uses the same fresh `4gpu-mb48` geometry and starts after the
  primary AR releases GPUs 0-3.
- primary 4-GPU AR attempt: memory-safe and healthy through step 191, then the
  known asynchronous CUDA illegal-address race reappeared at the next update;
  final loss/FVE `0.323341 / 0.421504`, finite gradients, no checkpoint.
- corrected restart recipe: fresh IDs with `moe_routing_impl: expert_scan` and
  `cuda_launch_blocking: true` for primary AR, independent AR, and AV. This is
  the previously validated combination that crossed the deterministic fault
  boundary and completed the independent critic.
- S3 code sync is operational again through `https://pdx.s8k.io`; no secrets
  were printed.

### diagnosing: r33 clean AR synchronized Mamba fast-kernel failure

- protected primary AR result: step 191 completed at loss/FVE
  `0.3241694 / 0.4200212`; rollout 192 failed synchronously in
  `mamba_split_conv1d_scan_combined -> causal_conv1d_fwd` with a CUDA illegal
  memory access. No checkpoint exists.
- correction to prior diagnosis: direct `/proc/<actor>/environ` inspection on
  the live AV worker confirmed `expert_scan`, launch blocking, allocator, and
  observability variables were inherited. Explicit Miles `--train-env-vars`
  propagation is retained as deterministic hardening, not claimed as the AR
  root-cause fix.
- prepared primary/independent AR path: config-driven
  `mamba_kernel_mode: unfused_torch_conv`, which uses grouped PyTorch causal
  convolution and keeps the efficient Mamba chunk scan. Fresh identities avoid
  failed-run metric mixing.
- required preflight: four optimizer updates at the original `4 GPU / mb48 /
  gb192` geometry, no checkpoint, offline W&B.
- preflight result: passed all four updates and exited 0. Step 3 loss/FVE was
  `1.1304802 / -1.0225676`; reward/train MSE ratio remained exactly 1.0;
  peak allocated/reserved memory was `45.05/54.08 GiB` per H100. The only
  traceback was a post-training W&B actor-teardown broken pipe.
- deployed revision: commit `8d9917a`, RunAI tests `78 passed, 2 skipped`, S3
  archive SHA-256
  `d4ef1c23e52af0823214e45602bd0cbef5dc72fe9e4d1f9263440b4d1683c5ed`.
- primary AR full run: queue state `training` from
  `2026-07-09T21:02:46Z`, GPUs 0-3, target `iter_0001291` plus validation-only
  512-row bounded eval. It must cross the old rollout-192 boundary before the
  runtime mitigation is considered durable.
- matched dynamics: first 24 shared updates versus the fused-kernel attempt
  differ by at most `0.0024314` loss and `0.0043501` FVE with identical LR;
  step 30 reached loss/FVE `0.4479342 / 0.1985908`, finite gradients, and no
  CUDA/OOM/traceback signal.
- AV status: 4-GPU run remains healthy and was left running; step 26 loss
  `1.04451`, LR `9.99994e-5`, roughly 79 GiB/GPU, no CUDA/OOM signal.

### ready: corrected packed AV hero after fail-closed boundary proof

- The old `dyn512` full AV process was stopped around step 32 with no
  checkpoint; projected runtime was about 6.7 days.
- The first guarded packed-vs-padded probe failed before step 0 at mean NLL
  `2.58670735 / 2.86806154` and max absolute/relative difference
  `0.94716144 / 0.34476781`.
- Root cause: `NemotronHForCausalLM.forward` dropped packed `position_ids`
  before the backbone, so neither Mamba nor attention saw sample resets. The
  attention mask and seq-index ordering fixes alone were insufficient.
- Corrected model-code commit: `6abfe18`; identical deployed model-code hash
  across base and both critics:
  `dd5f82b9697307694d8e29b68066779b4c23a9e2a02095475a904bf6530f1e41`.
- Corrected `dyn2048` gate: `passed=true`, max absolute/relative difference
  `0.01182747 / 0.00395436`; steady step `131.72s`; post-Adam peak
  allocated/reserved `69.28 / 77.27 GiB`; exit `0`.
- Corrected `dyn4096` gate: `passed=true`, max absolute/relative difference
  `0.01632524 / 0.00704313`; steady step `67.33s`; post-Adam peak
  allocated/reserved `70.45 / 78.24 GiB`; exit `0`.
- Selected full AV config/queue:
  `configs/nano_av/publication/r33_family_clean_sft_8gpu_dyn4096.yaml` and
  `r33_family_clean_sft_8gpu_dyn4096_queue.yaml`.
- Deployment: commit `8a4ecfa`, S3 archive SHA-256
  `8959ec9115c6cc9a55f855e469dc706f06d339afba5b92650b3c02035823aa52`;
  no-side-effect queue dry-run exit `0`; `48` AV runner/queue tests passed.
- Queue state: `pending`. Do not launch until the active clean AR run has
  checkpointed and completed its validation-only 512-row eval.
- Historical correction: packed AV/RL actor checkpoints trained before
  `6abfe18` are internal scouting artifacts, not publication-clean models.
- Lightweight evidence archive (local and S3, no parquets/checkpoints):
  `artifacts/runai_evidence/20260709_r33_publication_remediation/packed_av_correctness/r33_packed_av_correctness_evidence_20260709.tgz`,
  SHA-256
  `ae5151248a796b0d81187c30a0080643059aaa2ee4d009066df7d31be506dce2`.
- Promotion watcher: PID `1017162`, source commit `3fa3b81`, currently waiting
  on the named AR queue item to reach `complete`; it launches the AV probe
  queue natively and does not bypass checkpoint/eval completion.
- Pre-hero cleanup: eight obsolete publication paths deleted after metadata
  freeze and manifest dry-run; four protected paths post-verified. Model-store
  free space rose from about `196 GiB` to `255 GiB`. Evidence:
  `artifacts/runai_evidence/20260709_r33_publication_remediation/prehero_cleanup/prehero_cleanup_evidence_20260709.tgz`,
  SHA-256
  `da7267438c9664090e11bb75d6d621c32223306dcd3f1ab9b8b1d1f662ec6769`.

### historical July 10 queue snapshot: publication-clean R33 AV after clean AR

- AR prerequisite: `complete` at `2026-07-10T01:13:44Z`; final checkpoint
  `iter_0001291`, final train loss/FVE `0.2625350 / 0.5302927`.
- AR validation gate: 512 rows; teacher NMSE `0.281703`, shuffled `0.968888`,
  blank `0.756098`, generic `0.781429`, mean `0.678041`, source context
  `0.301252`, source raw `0.083248`; no controls skipped.
- Promotion: watcher launched the eight-H100 AV queue at
  `2026-07-10T01:18:42Z`. The optimizer phase completed all 1,291 updates at
  `2026-07-10T14:09Z`; at the last authenticated RunAI check, DCP-to-HF
  conversion for the validation eval remained active.
- Live AV equivalence gate: `passed=true`; packed/padded mean response NLL
  `2.56366563 / 2.56583595`, maximum absolute/relative difference
  `0.02207184 / 0.00795042`.
- Final trajectory: step 1290 loss `0.683009`, gradient norm `0.585938`, LR
  `1e-5`, normalized router entropy `0.991240`; every one of 1,291 logged
  losses/gradients/LRs/router metrics is finite, with no CUDA, OOM, or
  traceback signal. Final-window mean loss was `0.696919`.
- Storage guard: clean AR HF and compact evidence are mirrored to S3 and
  verified at 10 objects / `38,462,226,607` bytes. Its redundant model and
  optimizer DCP copies were removed through the completed retention manifest.
- AR audit v2: passed with extraction block 33, 34 retained blocks, observed
  tensor blocks 0..33, absent LM head/final norm, finite `2688 x 2688` value
  head, and zero train/validation/test document overlap.
- Deferred round trip: clean family coverage passes for every validation/test
  row with zero exposure exclusions. Replacement immutable source `9b8b44f`
  passed `94` focused RunAI tests; protocol dry-run hash remains
  `276d97c3e460218e24bf7bd751a94bd4c9ed55859cdf4c751fb2824338e8e1aa`.
  Watcher PID `1506837` is waiting on AV queue completion; it stages and
  fingerprints DCP in one sequential pass, runs validation-only
  generation/scoring, and does not open test. Superseded watcher PID `1163925`
  from source `815db9e` was terminated without launching either queue item.
- Storage follow-up: the superseded primary and independent 36G critic
  initialization copies were deleted under a final manifest after compact
  reproducibility evidence was mirrored to S3. Selected AR and AV paths were
  post-verified intact. The earlier `285G` free figure predates the final AV
  checkpoint; the last authenticated observation was `108G` free on
  `/workspace/models` and `295G` on `/workspace/interp`.
- Independent critic/AR: source `3676b93` stages a manifest-hashed critic-init
  rebuild followed by the validation-only independent AR queue. The critic
  dry-run requires SHA-256
  `71cfb2bf243bbae720d0b2931a310b6b327a2922a59d5cf5165926fc988edcff`;
  the AR dry-run is intentionally `blocked_missing_critic_init`. Neither has
  launched.
- Status boundary: this preserves the July 10 observation before qualification.
  The completed July 15 queue entry at the top supersedes AV and round-trip
  status. The July 8 RL hero had completed, but no publication-protocol RL
  replication had launched from the later qualified SFT pair.
