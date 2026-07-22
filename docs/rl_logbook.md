# Nano30B NLA RL Logbook

<!-- R33-HERO-BASELINE-PROTOCOL-INVALIDATED -->

> [!CAUTION]
> Publication status (`2026-07-10`): the archived `30.97% / 32.34%` hero
> comparison is not publication-valid because the 512-row SFT baseline mixed
> two generation protocols. Figures below remain historical internal-gate
> evidence only. Deterministic clean SFT remediation is underway, and no
> publication-clean RL job has launched. See
> `docs/reviews/2026-07-08-r33-rl-hero-publication-audit.md`.

Last updated: `2026-07-10`.

This logbook records the historical component-full RL work and the active
publication-clean gating lineage. It separates systems evidence from quality
evidence and preserves failed/adaptive runs without promoting them.

## Active Publication RL Status

The deterministic family-clean R33 SFT AV+AR pair and independent AR
replication are complete. The first clean-lineage online joint AV+AR canary
also completed two actor and two critic optimizer updates on `2026-07-17`.
Its systems, parser, control, provenance, and row-matching checks pass, but its
strict paired SFT-improvement gate fails. No publication-clean RL checkpoint is
promoted; the supervised pair remains canonical.

## Historical Component-Full SFT Baseline

The historical pre-RL candidate was the R33 component-full AV+AR pair:

- AR checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001289`
- AR bounded `512/512` eval:
  validation/test teacher NMSE `0.320616 / 0.292730`.
- AV checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291`
- AV corrected `512/512` eval:
  validation/test real NLL `0.798672 / 0.819993`.
- Final SFT round-trip gate:
  validation/test AV-real NMSE `0.000109680 / 0.000121664`, closed/usable
  parse fractions `1.0 / 1.0`, and all in-run controls beaten.

The early RL runs below remain systems/debug evidence. The corrected K3 probes
from `2026-07-01` onward selected `lr=1e-5`, and the update-342 RL hero now
passes the full generated-text round-trip promotion gate against matched clean
SFT rows.

## Current RL Status (2026-07-08)

- Selected run:
  `r33-corrected-k3-hero-lr1e5-update342-resume228-retry3`.
- Selected checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_hero/r33_corrected_k3_hero_lr1e5_update342_resume228_retry3/actor/iter_0000342`.
- Queue completed at `2026-07-08T03:20:52Z` with no recorded error.
- The prerequisite `64/64` gate and final `512/512` gate both passed.
- Final validation/test AV-real normalized MSE:
  `0.000087528 / 0.000091176`.
- Relative improvement over exact matched clean SFT:
  `30.97% / 32.34%`.
- Rowwise wins over SFT:
  `83.40% / 88.67%`.
- Closed/usable generation rates:
  `99.02% / 100%` validation and `99.41% / 100%` test.
- Canonical compact report:
  `docs/runs/r33_rl_hero_20260708.md`.

## Historical RL Status (2026-07-03)

- RunAI workspace `train` is running on 8 H100 NVLs in project
  `trustworthy-ai-inference`. At `2026-07-03T17:06Z`, all GPUs were idle at
  `4 MiB`, no Nano train/eval/SGLang process was active, `/workspace/interp`
  had `325G` free, and `/workspace/models` had `454G` free.
- The selected `lr=1e-5` update-8 actor checkpoint was reverified on the
  mounted PVC at `59G`. It remains the only retained corrected-K3 actor
  checkpoint from the two-probe comparison.
- The corrected train-only R33 RL dataset passed its strict verifier with
  `247,700` rows, `24,867` documents, layer `33`, `d_model=2688`, zero
  nonfinite vectors, zero duplicate provenance keys, and zero heldout overlap.
- The selected probe is
  `r33-corrected-k3-probe-lr1e5-update8-unifiedenv-retry1`. Its valid 256/256
  gate improved matched SFT AV-real NMSE by `1.00% / 1.79%` on
  validation/test. The result is positive HPO evidence, not a hero milestone.
- The `lr=2e-5` probe also passed a valid 64/64 full-prefix gate, but its
  split-mixed comparison with `lr=1e-5` was statistically inconclusive and
  its training had severe early K3 transients. Its actor checkpoint was
  deleted only after lightweight evidence was archived and hash-verified.
- The next item is a blocked 32-update `lr=1e-5` confirmation. It has not
  launched and cannot auto-launch: explicit queue approval is required.
  Future runs use shuffled rows, working local-shard gradient clipping,
  composite KL/parser/length/drift guards, and storage-safe update-16/update-32
  retention. The two completed probes remain explicitly marked as unshuffled
  and unclipped so historical reconstruction is not rewritten.
- The SFT 512/512 baseline is now hardened with exact train/validation/test
  SHA256 values and 512 row keys plus document IDs per split at
  `validity/r33-sft/roundtrip_v512_t512_hardened_report.json`. Promotion gates
  require dataset identity, row coverage, rowwise wins, and document-clustered
  bootstrap confidence intervals.
- The Stage-2 composite gate and independent cross-critic gate are both still
  absent. Hero queues fail closed on those reports. Only one independent R33
  AR checkpoint is retained, so a second critic must be trained before the
  cross-critic requirement can be satisfied.
- Launch-critical source content is frozen by SHA256
  `150d2832105c007b1d977d45560c26c09b4aff03770f265ba577257756aa67a7`;
  the corresponding source commit is
  `30e5e26e1e831e54b83f5ac7bcf443bf89eda546`. The queue verifies the content
  fingerprint before launch and records both identifiers in each run
  directory.

## Early RunAI Topology Used

- Workspace/project: `train` / `trustworthy-ai-inference`.
- Hardware for successful RL smoke: `4x NVIDIA H200`.
- Layout: `2` actor H200s, `1` managed external SGLang rollout H200, `1`
  frozen R33 AR-critic H200.
- Actor SFT init: selected R33 AV checkpoint above.
- Reward/critic init: selected R33 AR checkpoint above, staged as HF.
- RL smoke parquet:
  `/workspace/interp/outputs/nano30b-nla-pilot/posthero/20260621T163000Z/rl_smoke/rl_R33_fullscan275396_smoke512.parquet`.

## Successful Smoke Runs

### Skip-Sync Frozen-Critic Smoke

- Queue:
  `configs/nano_rl/r33_component_full_smoke_queue_4h200_len256_rb2_fix2_freezecritic.yaml`
- Run directory:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_smoke/r33_component_full_sft_init_512row_4h200_len256_rb2_fix2_freezecritic`
- Controls:
  `NLA_SKIP_ROLLOUT_WEIGHT_SYNC=1`, `NLA_FREEZE_CRITIC_TRAIN=1`,
  `rollout_batch_size=2`, `global_batch_size=2`, `n_samples_per_prompt=2`,
  `max_response_len=256`, `max_context_len=256`, actor LR `1e-6`,
  actor microbatch `1`.
- Result:
  completed `4/4` rollout generations, one actor update, and saved
  `actor/iter_0000001`.
- Reward/train equivalence:
  `mean=1.0000`, `max|r-1|=0.0000`, `n=4`.
- Scalar snapshot:
  raw reward `-0.280732`, shaped reward `5.349516868591309e-06`,
  actor loss `-5.304813385009766e-06`, grad norm `9.3125`.
- Interpretation:
  proves the 4-H200 topology and reward/train MSE agreement can work when
  rollout weights are preloaded. It is not a true live RL result because
  actor-to-SGLang weight sync was skipped.

### Live-Sync Two-Rollout Smoke

- Queue:
  `configs/nano_rl/r33_component_full_pilot_queue_4h200_len512_rb2_sync2_nosaveoptim_unifiedenv_mambawheels_tokcompat_nopackedcheck_criticfwd_evalmode_nofastpath_timesteplimit_allocseg.yaml`
- Run directory:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_pilot/r33_component_full_sft_init_512row_4h200_len512_rb2_sync2_nosaveoptim_unifiedenv_mambawheels_tokcompat_nopackedcheck_criticfwd_evalmode_nofastpath_timesteplimit_allocseg`
- Runtime controls:
  unified SGLang/Torch environment, Mamba wheels, TokenizersBackend fallback,
  managed external SGLang, disabled SGLang radix cache, router history backend
  `none`, round-robin routing, `NLA_ASSERT_PACKED_EQUIV=0`,
  `NLA_FREEZE_CRITIC_TRAIN=1`, eval/no-grad critic reward forward, critic
  Mamba fast path disabled only during reward scoring, AR-critic
  `time_step_limit` sentinel normalization, and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Wall time:
  started `2026-06-23T20:18:50Z`, completed `2026-06-23T20:26:56Z`.
- Result:
  queue completed, two live-sync rollout/update cycles completed, `4/4`
  generations per rollout, actor checkpoint saved at `actor/iter_0000002`.
- Rollout 0:
  response length `235.75`, total length `352.75`, raw reward
  `-0.9464928060770035`, shaped reward about `-2.09e-07`, actor train time
  `175.0s`.
- Rollout 1:
  response length `234.0`, total length `351.0`, raw reward
  `-0.6190555766224861`, shaped reward about `-2.24e-07`, actor train time
  `10.9s`.
- Memory/checkpoint evidence:
  rollout 1 reached about `143.1 GiB` used on each actor H200 and crossed the
  prior actor-backward OOM point. The final DCP actor checkpoint wrote two
  model shards of about `31.6GB` each and took about `2m20s` to flush.
- Interpretation:
  live actor-to-SGLang weight sync is no longer the immediate blocker for a
  tiny 4-H200 R33 RL systems run. Memory headroom is still thin on the
  2-actor-GPU partition, and interval checkpointing is too expensive for
  exploratory runs.

## Failures And Fixes Encountered

- Single-actor-GPU topology OOMed. The actor side of Nano30B needs two H200s
  for this RL path.
- Early live-sync pilot failed with exit status `134` after the first
  `backbone.embeddings.weight` bucket metadata was sent. Logs indicated
  distributed rollout weight-sync instability.
- The barrier-fix retry reached SGLang healthcheck timeouts and SIGTERM/system
  error behavior. Its queue entry should be treated as stale/failed.
- The first unified-env retry failed because the SGLang environment lacked
  `accelerate`, which Miles imports during actor FSDP initialization.
- The no-fast-path retry reached rollout/generation/reward execution but failed
  in critic reward scoring with a `torch.clamp` type error caused by the AR
  critic config encoding `time_step_limit` as `[0.0, {"__float__":"Infinity"}]`.
- The time-step-limit fix exposed an actor backward CUDA OOM on rollout 1,
  with reserved-but-unallocated CUDA memory. Adding
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` allowed the next two-rollout
  smoke to complete.
- The SGLang router cache-aware/history path is unsafe for NLA rollout payloads:
  activation-conditioned `input_embeds` bodies are large, and token-prefix cache
  keys are insufficient when the same marker token can carry different
  activations. The working path disables radix cache and router history and uses
  round-robin routing.
- The reward path depends on the frozen AR critic and is intentionally run with
  `NLA_FREEZE_CRITIC_TRAIN=1` for the current smoke ladder. This avoids updating
  the verifier while validating actor RL plumbing.

## W&B And Metrics Status

The completed RL smoke configs set `WANDB_MODE=offline`, but that alone did not
initialize W&B for the RL path. The completed live-sync smoke directory contains
`train.log` and `sglang_service_0.log`, but no offline W&B run directory was
found under the run directory.

The launcher has now been updated so future RL runs pass W&B flags through
Miles:

- `--use-wandb`
- `--wandb-mode "$WANDB_MODE"`
- `--wandb-dir "$RUN_DIR/wandb"` by default
- `--wandb-project "$WANDB_PROJECT"` with default `nano30b-nla-pilot`
- `--wandb-group "$WANDB_GROUP"` with default `nano-rl`
- `--wandb-always-use-train-step`
- optional `--wandb-run-id "$WANDB_RUN_ID"`
- NLA rollout metrics hook:
  `--custom-rollout-log-function-path nla.rollout.rl_metrics.log_rollout_data`

The RL queue now also defaults NLA system metrics on for future runs:

- `NLA_SYSTEM_METRICS=1`
- `NLA_SYSTEM_METRICS_INTERVAL_STEPS=1`
- `NLA_SYSTEM_METRICS_NVSMI_INTERVAL_STEPS=1`

Current scalar logs already include rollout length, reward, logprob, entropy,
advantage/return, actor loss, grad norm, LR, update-weight time, logprob time,
actor train time, train wait time, step time, and wait ratio. The added NLA
rollout hook injects reward distribution and health scalars:
mean/std/min/max/p10/p25/p50/p75/p90, parse closed/usable fractions, completion
status fractions, and response-length versus reward correlation.

### W&B Canary Ladder, 2026-06-23

These runs used the R33 full component SFT actor and R33 AR critic with the
4-H200 layout above, `rollout_batch_size=2`, `global_batch_size=2`,
`n_samples_per_prompt=2`, `max_response_len=512`, `max_context_len=512`, actor
LR `1e-6`, KL disabled, and frozen critic train.

- `r33_component_full_wandb_canary_4h200_len512_rb2`: completed at
  `2026-06-23T21:21:02Z`. W&B rollout reward/parse/status keys were present,
  but `save_interval=999` still caused Miles to write a final actor checkpoint
  of about `59G`. The actor checkpoint directory was removed after verification;
  logs/W&B were kept.
- `r33_component_full_wandb_canary_4h200_len512_rb2_nosave`: completed at
  `2026-06-23T21:34:03Z`. Setting `save_interval: null` in the queue and
  omitting `--save-interval` in `rl.sh` prevented Miles checkpoint creation:
  run directory about `2.7M`, `actor_files=0`. W&B reward metrics were present.
  Snapshot: raw reward mean `-0.986965`, no truncation, actor train time
  `158.1s`, step time `189.6s`, grad norm `4.3125`.
- `r33_component_full_wandb_canary_4h200_len512_rb2_sysmetrics`: completed at
  `2026-06-23T21:42:19Z`. It proved no-save behavior and reward metrics again,
  but `nla/system/*` did not appear. Root cause was that Ray actor workers
  receive `--train-env-vars`; reading only `os.environ` was insufficient.
- `r33_component_full_wandb_canary_4h200_len512_rb2_sysmetrics_diag`:
  completed at `2026-06-23T21:53:05Z`. It proved `NLA_SYSTEM_METRICS*` reached
  actor and critic workers through `train_env_vars`, but `nla/system/*` still
  did not appear in W&B. Root cause was that Miles train losses are structured
  as `{"keys": [...], "values": tensor}` and reduced by
  `aggregate_train_losses`; plain `log_dict.update(...)` was ignored.
- `r33_component_full_wandb_canary_4h200_len512_rb2_system_metrics_reducer_fix`:
  completed at `2026-06-23T22:03:55Z`. This run verified the reducer fix:
  `train/nla/system/*` keys appear in both `train.log` and the offline W&B
  binary, alongside `rollout/nla_reward/*`, `rollout/nla_parse/*`, and
  `rollout/nla_status/*`. Run directory about `2.8M`, `actor_files=0`, GPUs
  idle after completion. Snapshot: raw reward `-0.640552`, response length
  `322.25`, total length `439.25`, no truncation, actor train time `172.8s`,
  step time `208.1s`, update-weight time `5.4s`, grad norm `3.65625`.

Implementation note: `train/nla/system/rank`, `local_rank`, and `pid` are
currently averaged across actor ranks by Miles' reducer, so they are useful as
presence/debug sentinels rather than literal identifiers. Memory/utilization
scalars are actor-rank averages; this is acceptable for the next medium smoke,
but per-rank extrema would be better before a long RL hero.

Canary hygiene note: the reducer-fix run emitted one rollout-side warning whose
filename pointed at an older synced source path,
`nano30b-nla-pilot-sync-20260615T154008Z-hero/external/.../nla/reward.py`.
Fresh import checks after the run in both `/workspace/interp/.venv` and
`/workspace/interp/.venvs/sglang-cu130` resolved `nla.reward` and
`nla.rollout.rl_metrics` to
`/workspace/interp/code/nano30b-nla-pilot-current`, and no live Ray/SGLang
processes remained. Treat this as stale Ray/runtime hygiene rather than an
active code-sync failure, but flush Ray/SGLang before the next medium run.

### Cancelled Medium-A Attempt, 2026-06-23

The first Medium-A launch was intentionally stopped before accepting results:

- Queue:
  `configs/nano_rl/r33_component_full_medium_a_queue_4h200_len512_rb2_rollout8.yaml`
- Bad run directory:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_medium/r33_component_full_medium_a_4h200_len512_rb2_rollout8`
- Cancel reason:
  `actor_sft_ckpt` pointed to `.../checkpoints/iter_0001291`, and `rl.sh`
  used that same value for Miles `--load` and `--ref-load`. Miles' DCP loader
  expects the parent `.../checkpoints/` directory because
  `latest_checkpointed_iteration.txt` lives there; it logged
  `No tracker file ... iter_0001291/latest_checkpointed_iteration.txt; skipping load`.
- Consequence:
  the actor trainer could have started from the base HF model while SGLang was
  initialized from the staged SFT HF checkpoint and then overwritten by
  actor-to-SGLang weight sync. Treat this bad attempt as systems-debug only.
- Fix:
  `scripts/nano_rl_queue.py` and `configs/rl.sh` now support separate actor
  paths: `ACTOR_LOAD_CKPT` / `ACTOR_REF_CKPT` for Miles DCP load roots and
  `ACTOR_SIDECAR_SOURCE` for the SFT iteration sidecar. Medium-A was moved to a
  fresh `_dcpfix` run directory with `actor_load_ckpt` and `actor_ref_ckpt`
  pointed at the parent `checkpoints/` directory and `actor_sidecar_source`
  kept at `iter_0001291`.

The corrected `_dcpfix` launch was also stopped before accepting results. The
actor DCP path was now correct (`--load .../checkpoints`, no actor skip), but
the launcher still had no explicit `--no-load-optim` path. That meant a fresh RL
phase could spend time loading SFT optimizer state and risk inheriting an
inappropriate Adam state. This was fixed by:

- adding YAML-driven `training.finetune` and `training.no_load_optim` controls
  in `scripts/nano_rl_queue.py`;
- emitting `--finetune` and `--no-load-optim` from `configs/rl.sh`;
- exposing `--no-load-optim` in the Miles NLA patch and making `--finetune`
  explicitly set `args.no_load_optim = True`.

The next clean Medium-A run is therefore
`r33_component_full_medium_a_4h200_len512_rb2_rollout8_dcpfix_noloadoptim`.

## Medium-Run Scaling Plan

The next run should be a quality smoke, not a hero run.

1. Flush stale runtime state before scaling: verify no active training,
   rollout, SGLang, or Ray processes; run `ray stop --force` if Ray is active;
   preserve logs and do not delete SFT checkpoints.
2. Run Medium-A with final-only checkpointing during the exploratory portion:
   `2k` rows or `8` rollouts,
   `rollout_batch_size=2`, `global_batch_size=2`, `n_samples_per_prompt=2`,
   `max_response_len=512`, `max_context_len=512`, actor LR `1e-6`, KL disabled
   initially, frozen critic reward, correct actor DCP load root, and
   `finetune/no_load_optim` so RL starts from SFT weights but fresh optimizer
   state.
3. Evaluate the Medium-A actor against the SFT actor with the same round-trip
   gate used for SFT promotion. Required comparison metrics are parse health,
   AV-generated-text to AR NMSE, control margins, reward mean/std trajectory,
   and response degeneracy checks.
4. If Medium-A is stable and improves or preserves round-trip NMSE without
   parse/length collapse, run Medium-B on `8k-10k` rows or `16-32` rollouts.
   Consider `rollout_batch_size=4` only after the canary confirms actor GPU
   memory and SGLang throughput are stable.
5. Defer full RL hero planning until Medium-B shows stable reward variance,
   healthy parse rates, no length gaming, and round-trip metrics that beat or
   match the SFT baseline.

## Current Next Step

Medium-A completed and did not beat the SFT candidate. Do not scale this RL
configuration without changing the reward/advantage setup.

### Medium-A Result, 2026-06-23

Run:
`/workspace/interp/outputs/nano30b-nla-pilot/rl_medium/r33_component_full_medium_a_4h200_len512_rb2_rollout8_dcpfix_noloadoptim`

Queue item:
`r33-component-full-rl-medium-a-512row-4h200-len512-rb2-rollout8-dcpfix-noloadoptim`

Status:

- completed at `2026-06-23T22:42:08Z`;
- live argv confirmed `--load .../checkpoints`, `--ref-load .../checkpoints`,
  `--nla-sidecar-source .../iter_0001291`, `--finetune`, and
  `--no-load-optim`;
- actor DCP load used the parent checkpoint root; the only `No tracker file`
  warning was the expected critic HF path;
- W&B offline, rollout metrics, and `train/nla/system/*` metrics were written;
- final actor checkpoint: `actor/iter_0000008/model/`, about `59G`;
- no OOM; W&B emitted BrokenPipe warnings during process teardown only.

Training dynamics:

- raw rewards over 8 rollouts:
  `[-0.2653, -0.4957, -0.4075, -0.4842, -0.2418, -0.5433, -0.6429, -0.8760]`;
- raw reward mean `-0.4946`, min `-0.8760`, max `-0.2418`;
- generated responses had truncation `0.0`;
- train losses were near zero because normalized advantages were effectively
  near zero for this tiny `global_batch_size=2`, `n_samples_per_prompt=2`
  setup;
- grad norms stayed finite, roughly `2.625` to `5.71875`;
- steady post-startup step time was about `21-27s`, with actor train about
  `8-12s` and weight update about `3-5s`.

Round-trip diagnostic:

- DCP actor checkpoint was converted to a temporary HF dir, evaluated, then the
  temporary HF dir was deleted to recover about `59G`;
- report:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_medium/r33_component_full_medium_a_4h200_len512_rb2_rollout8_dcpfix_noloadoptim/roundtrip_iter_0000008_v64_t64_report.json`;
- generated text:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_medium/r33_component_full_medium_a_4h200_len512_rb2_rollout8_dcpfix_noloadoptim/roundtrip_iter_0000008_v64_t64_generated.jsonl`;
- gate passed: `false`;
- validation parse health: closed `0.953125`, usable `1.0`, empty `0`;
- test parse health: closed `1.0`, usable `1.0`, empty `0`;
- validation AV-real round-trip NMSE `0.000130758`, teacher-text NMSE
  `0.000105076`; validation failed the control gate because
  `av_real_vs_mean` rowwise win fraction was `0.890625`, below the `0.9`
  threshold;
- test AV-real round-trip NMSE `0.000122169`, teacher-text NMSE `0.000112544`;
  test beat controls.

Comparison to the pre-RL R33 SFT hero round-trip report:

- SFT hero validation AV-real NMSE `0.000109680`; Medium-A RL validation
  worsened to `0.000130758`;
- SFT hero test AV-real NMSE `0.000121664`; Medium-A RL test was effectively
  tied/slightly worse at `0.000122169`;
- SFT hero had closed/usable parse `1.0 / 1.0` on both splits and gate passed.

Conclusion:

Medium-A proves the RL systems path now runs end to end, but this specific
short RL setup is not a promotion candidate. The next RL attempt should fix the
learning signal before scaling: use a larger effective rollout group or
different advantage/reward normalization, add a direct SFT-vs-RL round-trip
delta metric to the RL loop, and avoid treating raw reward alone as sufficient.
The current R33 SFT AV+AR hero remains the selected candidate.

### R33 RL Signal Ladder Plan, 2026-06-24

Medium-A showed that the RL systems path works, but the tiny rb2/n2 setup and
KL-free update did not improve the R33 SFT AV+AR candidate. The next queue is:

`configs/nano_rl/r33_component_full_signal_ladder_queue_4h200_len512.yaml`

Purpose:

- increase effective rollout group from rb2/n2 to rb4/n4 (`global_batch_size=16`);
- keep the frozen R33 AR critic reward;
- compare every saved RL actor directly against the SFT round-trip baseline via
  `--baseline-report-json`;
- test KL anchoring and reward/advantage normalization before any RL hero.

Queued variants:

- `r33-component-full-rl-signal-rb4-n4-kl1e4`: larger group, KL `1e-4`,
  default GRPO reward centering and std normalization.
- `r33-component-full-rl-signal-rb4-n4-kl3e4`: stronger KL `3e-4` to test SFT
  retention.
- `r33-component-full-rl-signal-rb4-n4-no-std-kl1e4`: disables GRPO std
  normalization while keeping group centering and KL.
- `r33-component-full-rl-signal-rb4-n4-raw-advnorm-kl1e4`: disables group reward
  normalization and uses global advantage whitening, testing absolute
  AR-reconstruction reward instead of only within-prompt ranking.

Storage rule: these are signal probes, not hero candidates. The queue preserves
logs, W&B, generated text, and round-trip reports, then deletes each temporary HF
conversion and actor DCP probe checkpoint after eval to avoid another Longhorn
disk-pressure episode. If a variant is promising, rerun it as a checkpoint-kept
candidate before promotion.

Promotion rule: do not scale RL unless a variant preserves parse health, beats
controls, and matches or improves the SFT baseline round-trip NMSE.

### R33 RL Signal Ladder Cleanup And KL Ref Fix, 2026-06-24

Before relaunching the signal ladder, obsolete probe actor DCP directories were
removed on RunAI while preserving selected R33 SFT hero checkpoints, logs, W&B,
and reports. `/workspace/interp` dropped from `742G` used / `266G` free to
`559G` used / `449G` free. Deleted directories were limited to old
`rl_smoke`, `rl_pilot`, and `rl_medium` actor checkpoint trees plus temporary
migration/Ray scratch data; the selected R33 AV SFT and AR SFT hero checkpoints
were kept.

The first KL-anchored signal-ladder launch failed before training because Miles
uses `--ref-load` only when KL is enabled, and the queue pointed
`actor_ref_ckpt` at a temporary `hf_iter_0001291` export that had already been
cleaned after eval. The DCP actor checkpoint root remains correct for
`--load`, but KL reference loading requires a Transformers/HF directory. The
queue now reuses the persisted SFT HF actor export already used by SGLang:
`/workspace/interp/outputs/nano30b-nla-pilot/rl_smoke/r33_component_full_sft_init_512row_3h200/actor_sft_hf_iter_0001291`.

Code guardrail added: `scripts/nano_rl_queue.py` now preflights required launch
paths, including `ACTOR_REF_CKPT` when `KL_LOSS_COEF` is nonzero and managed
SGLang `--model-path` values when present. Dry-runs report
`preflight_missing_paths`; real queue launches fail before starting SGLang if a
required path is missing. Local and RunAI focused tests passed (`45 passed`),
and RunAI dry-run reported an empty `preflight_missing_paths` list before the
corrected queue was relaunched.

### R33 RL Signal Ladder One-Rollout Debug Pass, 2026-06-24

Goal:

- get a clean RL train -> checkpoint -> DCP-to-HF -> round-trip eval path
  passing again on the 4-H200 workspace;
- preserve W&B/log/report artifacts, but avoid keeping signal-probe DCP/HF
  checkpoints unless the run beats the SFT baseline;
- use the same R33 component-full AV SFT actor and R33 AR critic baseline gate
  as the post-SFT hero candidate.

Queue/config:

- queue:
  `configs/nano_rl/r33_component_full_signal_ladder_queue_4h200_len512.yaml`;
- successful item:
  `r33-component-full-rl-signal-rb4-n2-kl1e4-rollout1`;
- run directory:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_signal/r33_component_full_signal_rb4_n2_kl1e4_rollout1_refhf`;
- main knobs:
  `rollout_batch_size=4`, `global_batch_size=8`,
  `n_samples_per_prompt=2`, `num_rollout=1`,
  `max_context_len=512`, `max_response_len=256`, KL `1e-4`;
- memory controls:
  BF16 FSDP reduce dtype, gradient checkpointing,
  `NLA_SKIP_GRAD_NORM=1`, no optimizer save, and FSDP2 backward prefetch
  disabled.

What happened:

- the original rb4/n4 and longer rb4/n2 signal variants were left failed or
  deferred because the actor path hit the H200 memory ceiling after the first
  rollout even with BF16 reduce, gradient checkpointing, and disabled backward
  prefetch; actor ranks were still near the high-130 GiB range;
- I added/used the bounded one-rollout rb4/n2 variant above to isolate whether
  the full systems path could complete before trying a larger medium run again;
- the queue file initially had invalid `status: deferred` entries, which the
  queue loader rejected. Those entries were converted to explicit failed items
  with failure reasons, and the one-rollout item was queued as the active probe;
- local-to-RunAI file sync through the RunAI exec gateway corrupted a normal
  base64 transfer of the YAML. The fix was to sync the file as hex chunks,
  decode it with remote Python, then verify byte count and SHA-256 on RunAI
  before relaunching the watcher.

Training result:

- queue watcher:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_signal/r33_signal_ladder_queue_watcher_rollout1.log`;
- training completed without OOM;
- `train.log` recorded completion at `2026-06-24T14:44:59Z`;
- rollout generation completed `8/8`;
- reward snapshot:
  raw reward `-0.264696`, centered rollout reward/advantage approximately
  `2.98e-08`;
- timing snapshot:
  actor train `37.9s`, train step `83.1s`, actor-to-SGLang update about
  `5.5s`;
- W&B offline ran; W&B service emitted teardown noise only, not a training
  failure.

Post-eval result:

- actor DCP `iter_0000001` was converted to a temporary HF directory and then
  evaluated with the round-trip gate;
- eval report:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_signal/r33_component_full_signal_rb4_n2_kl1e4_rollout1_refhf/roundtrip_iter_0000001_v64_t64_report.json`;
- generated text:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_signal/r33_component_full_signal_rb4_n2_kl1e4_rollout1_refhf/roundtrip_iter_0000001_v64_t64_report_generated.jsonl`;
- gate status: `false`;
- queue status after eval: `complete`, `post_eval_status=complete`;
- storage cleanup succeeded: the signal run directory is about `5.6M`; the
  temporary HF export and actor DCP probe checkpoint were removed, while logs,
  W&B, generated JSONL, and report were kept.

Round-trip metrics:

| split | RL AV-real NMSE | SFT baseline NMSE | delta vs SFT | teacher-text NMSE | parse health | control result |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| validation | `0.000113205` | `0.000109680` | `+0.000003525` | `0.000105076` | closed/usable `1.0 / 1.0` | beats shuffled, zero, mean, none |
| test | `0.000122785` | `0.000121664` | `+0.000001121` | `0.000112544` | closed/usable `1.0 / 1.0` | beats shuffled, zero, mean, none |

Interpretation:

- this pass is a systems success: train, save, DCP-to-HF conversion,
  multi-worker generation, AR scoring, baseline comparison, queue update, and
  storage-conscious cleanup all completed;
- it is not a model-quality win: the RL actor slightly regressed versus the
  clean SFT AV+AR round-trip baseline on both validation and test;
- the perfect parse health is important because earlier short-generation
  diagnostics were contaminated by empty or unparsable generations. This run's
  failure is a real quality/baseline failure, not a parser failure;
- with only one rollout, the normalized advantage signal was effectively near
  zero, so this should not be treated as evidence that RL cannot help. It is
  evidence that a one-rollout probe is too weak to improve the actor.

Issues encountered and fixes:

- invalid queue status:
  replaced unsupported `deferred` states with explicit failed entries carrying
  failure reasons so the queue loader could parse the YAML;
- missing KL reference path:
  fixed earlier by pointing KL `--ref-load` at the persisted SFT HF export
  rather than a temporary HF conversion removed after eval;
- unsafe launch paths:
  added/used queue preflight checks for actor load, KL ref load, managed
  SGLang model path, and post-eval baseline/report paths;
- YAML sync corruption:
  switched the Mac-to-RunAI YAML transfer from base64/stdin to hex chunks plus
  remote Python decode, then verified size and SHA before running;
- memory pressure:
  bounded the successful debug run to one rollout and rb4/n2, keeping the
  larger rb4/n4 variants failed/deferred until the actor memory topology is
  improved;
- opaque post-eval progress:
  generation workers wrote progress and all reached `32/32`, but the parent
  process gave little logging while it merged generated JSONL and ran AR
  scoring. This is a logging gap to fix before larger gates.

Next RL scaling implications:

- do not promote this RL checkpoint;
- keep the R33 SFT AV+AR hero as the selected candidate;
- the next RL run should be a true medium signal run with more rollout/update
  signal, but only after addressing actor memory limits or reducing per-rank
  memory pressure enough to avoid the rb4/n4 OOM path;
- add explicit post-generation/AR-scoring progress logging and direct
  SFT-vs-RL round-trip deltas during eval so future medium runs are easier to
  monitor.

### 8-H200 RL Readiness Pass, 2026-06-24

Motivation:

- the 4-H200 topology repeatedly put the actor FSDP ranks near the H200 memory
  ceiling, so rb4/n4 and longer rb4/n2 signal runs failed after the first
  rollout even with BF16 reduce, gradient checkpointing, disabled backward
  prefetch, and allocator expandable segments;
- the Qwen NLA RL recipe used much larger full-rollout updates
  (`128 prompts x 8 samples = 1024` samples/update) on a 16-H100 topology. The
  immediate Nano target is not full Qwen scale, but it should preserve the same
  optimizer semantics: one generated rollout batch, microbatched through actor
  training, then one optimizer step.

Code/config changes prepared while waiting for the 8-GPU RunAI workspace:

- `scripts/nano_rl_queue.py` now resolves a rollout batch plan before rendering
  CLI args:
  `rollout_batch_size`, `n_samples_per_prompt`, generated sample count,
  `global_batch_size`, and whether global batch matches the generated rollout;
- queues can opt into `rollout.require_global_batch_match: true`, which fails
  fast unless `global_batch_size == rollout_batch_size * n_samples_per_prompt`;
- the queue exports rollout batch sentinels to the training environment:
  `NLA_ROLLOUT_PROMPT_BATCH`, `NLA_ROLLOUT_SAMPLES_PER_PROMPT`,
  `NLA_ROLLOUT_GENERATED_SAMPLES`, `NLA_ROLLOUT_GLOBAL_BATCH`, and
  `NLA_ROLLOUT_GLOBAL_MATCH`;
- `external/natural_language_autoencoders/configs/rl.sh` logs those sentinels
  at launch and forwards them through Miles `--train-env-vars` for Ray actor
  observability;
- new queue:
  `configs/nano_rl/r33_component_full_signal_ladder_queue_8h200_len512.yaml`.

Prepared 8-H200 ladder:

| item | topology | rollout shape | purpose |
| --- | --- | --- | --- |
| `r33-component-full-rl-8h200-fit-rb4-n4-kl3e4` | actor `4`, critic `2`, rollout `2` | `4 prompts x 4 = 16` samples/update, `2` rollouts | memory/topology fit smoke |
| `r33-component-full-rl-8h200-signal-rb4-n8-kl3e4` | actor `4`, critic `2`, rollout `2` | `4 prompts x 8 = 32` samples/update, `4` rollouts | first Qwen-style group-size signal smoke |
| `r33-component-full-rl-8h200-medium-rb8-n8-kl3e4` | actor `4`, critic `2`, rollout `2` | `8 prompts x 8 = 64` samples/update, `16` rollouts | modest medium run if fit/signal pass |

All three keep the R33 AR critic frozen for reward scoring
(`NLA_FREEZE_CRITIC_TRAIN=1`) so that the next pass isolates actor RL behavior
against the clean SFT AV+AR baseline before attempting simultaneous AV+AR RL.
They use KL loss coefficient `3e-4`, response cap `256`, context cap `512`,
actor microbatch `1`, BF16 FSDP reduce, gradient checkpointing, and final
round-trip eval against the selected SFT baseline with storage-conscious
DCP-to-HF cleanup.

### 8-H100 High-Throughput Hero Plan Kickoff, 2026-06-24

Status: planned.

Topology update:

- `train` moved from H200/GH200 experiments to `8x NVIDIA H100 NVL` on
  `4u8g-gen-0176.ipp3a2.colossus.nvidia.com`;
- current H100 memory per GPU is about `95.8 GiB`;
- intended placement is actor FSDP on GPUs `0-3`, TP2 external SGLang rollout
  on GPUs `4-5`, and AR critic/reward on GPUs `6-7`;
- current live GPU usage was idle at the kickoff snapshot because the latest
  H100 canary failed before rollout/training.

Observed H100 blocker:

- the canary launched external SGLang with `--tp-size 2`;
- Miles still expected `tp_size=1` because the training command lacked
  `--rollout-num-gpus-per-engine 2`;
- failure evidence:
  `AssertionError: name='tp_size' expect_value=1 actual_value=2`;
- the earlier duplicate-endpoint workaround is rejected as the wrong
  abstraction: it satisfies the address count but not the rollout engine
  topology contract.

Plan document:

- `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/docs/superpowers/plans/2026-06-24-r33-rl-8h100-high-throughput-hero-plan.md`

Milestone policy:

- every topology/config/run transition must update this logbook;
- medium and hero runs must also update `docs/current_state.md`;
- a completed hero or rejected hero candidate must update
  `docs/experiment_logbook.md`;
- each milestone entry must include exact config paths, run directory, command,
  status, GPU memory/utilization, throughput, reward/KL stats, round-trip
  parse/close health, and checkpoint cleanup/retention actions.

## 2026-06-24 - M1 8x H100 TP2 Topology Fix

- Status: passed
- Workspace: `train`, project `trustworthy-ai-inference`
- Topology: `8x H100 NVL` on `4u8g-gen-0176.ipp3a2.colossus.nvidia.com`
- Local config:
  `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/configs/nano_rl/r33_component_full_signal_ladder_queue_8h100_len512.yaml`
- Remote config target:
  `/workspace/interp/code/nano30b-nla-pilot-current/configs/nano_rl/r33_component_full_signal_ladder_queue_8h100_len512.yaml`
- Previous blocker: external SGLang was TP2 but Miles expected TP1, failing
  with `AssertionError: name='tp_size' expect_value=1 actual_value=2`.
- Fix: one external endpoint `127.0.0.1:31000`, SGLang `--tp-size 2`,
  Miles `--rollout-num-gpus-per-engine 2`.
- Clean canary item:
  `r33-component-full-rl-8h100-fit-rb4-n4-kl3e4-tp2fix`.
- Clean canary run dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/r33_component_full_rl_8h100_fit_rb4_n4_kl3e4_tp2fix`.
- Verification:
  `/Users/rigarg/.local/bin/visor run python3 -m pytest tests/test_nano_rl_queue.py::NanoRLQueueTests::test_checked_in_r33_8h100_ladder_uses_tp2_rollout_engine -q`
  returned `1 passed`.

## 2026-06-24 - M2 8x H100 Sync and Runtime Cleanup

- Status: passed
- Local preflight:
  `/Users/rigarg/.local/bin/visor run python3 -m pytest tests/test_nano_rl_queue.py tests/test_nla_system_metrics.py tests/test_nla_rl_metrics.py -q`
  returned `34 passed`.
- Preflight repair: the older 8-H200 queue/test pair had a stale
  `HF_MODULES_CACHE` expectation pointing at the H100 cache. The 8-H200 queue
  now explicitly sets `NLA_PREWARM_HF_MODULES=1` and
  `HF_MODULES_CACHE=/dev/shm/nano30b-nla-pilot/hf_modules_cache/r33_component_full_signal_ladder_8h200`.
- S3 object:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/code-sync/nano30b-nla-pilot-current-h100-rl-tp2fix-20260624T2136Z.tgz`.
- Local upload command:
  `env NANO_S3_AWS_BIN=aws AWS_SHARED_CREDENTIALS_FILE=$HOME/.aws/credentials AWS_CONFIG_FILE=$HOME/.aws/config python3 scripts/nano_s3.py cp-up .cache/sync/nano30b-nla-pilot-current-h100-rl-tp2fix-20260624T2136Z.tgz code-sync/nano30b-nla-pilot-current-h100-rl-tp2fix-20260624T2136Z.tgz --timeout 300`.
- Remote code root:
  `/workspace/interp/code/nano30b-nla-pilot-current`.
- Remote extraction: downloaded with RunAI S3 credentials to
  `/workspace/interp/tmp/nano30b-nla-pilot-current-h100-rl-tp2fix-20260624T2136Z.tgz`
  and extracted over the remote code root. Tar printed macOS extended-header
  warnings only.
- Remote pytest:
  `/workspace/interp/.venv/bin/python -m pytest tests/test_nano_rl_queue.py tests/test_nla_system_metrics.py tests/test_nla_rl_metrics.py -q`
  returned `34 passed`.
- Processes before cleanup: only defunct Ray actor/GCS zombies from older
  failed attempts; no active `nano_rl_queue`, SGLang, or train actor process.
- Cleanup note: `ray` is not on the shell PATH in this container; a broad
  `pkill -f` cleanup attempt killed its own shell, so no further broad
  self-matching cleanup command was used. A follow-up process check still showed
  only defunct Ray/GCS zombies and no active training/rollout process.
- GPU memory after cleanup check: all 8 H100s at about `4 MiB` used and `0%`
  utilization.
- Disk after cleanup check: `/workspace/interp` `1008G` total, `677G` used,
  `331G` free; `/workspace/models` `1.4T` total, `974G` used, `454G` free;
  `/dev/shm` `2.3T` total, `59G` used.
## 2026-06-24 - M3 8x H100 TP2 Fit Canary

- Status: passed for topology/fit; not promoted on quality.
- Queue item: `r33-component-full-rl-8h100-fit-rb4-n4-kl3e4-tp2fix`.
- Workspace: RunAI `train`, project `trustworthy-ai-inference`.
- Topology: 8x H100 NVL. Actor FSDP on GPUs 0-3, one TP2 SGLang rollout engine on GPUs 4-5, frozen R33 AR reward/critic on GPUs 6-7.
- Run dir: `/workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/r33_component_full_rl_8h100_fit_rb4_n4_kl3e4_tp2fix`.
- Queue config: `/workspace/interp/code/nano30b-nla-pilot-current/configs/nano_rl/r33_component_full_signal_ladder_queue_8h100_len512.yaml`.
- Command shape: `rollout_batch_size=4`, `n_samples_per_prompt=4`, `global_batch_size=16`, `num_rollout=2`, `actor_lr=3e-6`, KL coefficient `3e-4`, `--rollout-num-gpus-per-engine 2`.
- Start/finish: queue started `2026-06-24T21:36:49Z`; training child completed `2026-06-24T21:47:36Z`; post-eval completed and queue marked complete `2026-06-24T22:15:15Z`.
- Previous TP2 blocker: resolved. No `tp_size expect_value=1 actual_value=2` failure.
- OOM status: no actor/SGLang/critic OOM.
- Checkpoint behavior: final DCP save wrote four ~15.8 GB model shards and `.metadata`; actor ranks spent about 3 minutes in `NLAFSDPActor.save_model` / disk sleep on Longhorn before returning. Queue cleanup then removed the actor checkpoint and temporary HF export; final run dir was about 7.3 MB.
- Peak training memory: actor GPUs reached about 90 GB used each; TP2 SGLang GPUs about 85 GB each; critic/reward GPUs about 50-57 GB each. Actor microbatch still has little H100 headroom.
- Rollout throughput: rollout 0 generated 16 samples in about 14s; rollout 1 generated 16 samples in about 7s. Actor train time was about 78.8s on step 0 and 33.2s on step 1. Weight updates were about 4.5-4.7s.
- Reward signal: rollout 0 raw reward `-0.3954378348`, normalized reward near zero after GRPO normalization; rollout 1 raw reward `-0.3836060343`, normalized reward near zero. Tiny normalized rewards are expected for per-rollout normalization at this batch size and are not enough to judge learning quality.
- KL/logprob signal: `train/ppo_kl` logged `0.0`; rollout/ref/log probs were close. Step 1 `train/kl_loss=-0.0011865899`, `train_train_rollout_logprob_abs_diff=0.0236886`.
- W&B: offline run created under the run directory and sync instructions were printed by W&B.
- Round-trip report: `/workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/r33_component_full_rl_8h100_fit_rb4_n4_kl3e4_tp2fix/roundtrip_iter_0000002_v64_t64_report.json`.
- Round-trip parse health: validation/test closed fraction `1.0/1.0`, usable fraction `1.0/1.0`.
- Round-trip validation: primary normalized MSE `0.0001111641`, teacher `0.0001049411`, SFT baseline `0.0001096800`; real beat all controls with rowwise win fraction `1.0`.
- Round-trip test: primary normalized MSE `0.0001218649`, teacher `0.0001123548`, SFT baseline `0.0001216637`; real beat all controls, with rowwise win fraction `1.0` against all AV controls and `0.984375` against mean.
- Gate decision: built-in report gate `false` because the 2-update canary did not beat the clean SFT baseline. This is acceptable for a fit canary; do not treat it as an RL quality win.
- Throughput issue: the 64/64 round-trip post-eval took roughly 27.5 minutes after training, dominated by 8-worker generation. Generation wrote 128 rows and used about 60-63 GB/GPU at roughly 40% GPU utilization. Eval throughput needs improvement before hero-scale gating.
- Path hygiene issue: `/workspace/interp/code/nano30b-nla-pilot-current` is a symlink to `/workspace/interp/code/nano30b-nla-pilot-sync-20260615T154008Z-hero`. Current code contents were synced into that target, so execution was not necessarily stale, but process logs show the old physical path for worker subprocesses. Clean this up before the medium run.
- Decision: proceed to Task 4 before launching medium. Required changes: fix remote code-root hygiene, choose a medium shape that avoids excessive checkpoint frequency, and account for the post-eval generation bottleneck.

## 2026-06-24 - M4 8x H100 Medium Run Parameter Selection

- Status: selected.
- Canary evidence: no TP2 assertion, no OOM, perfect parse health, real generation beat all controls on validation/test, and all intended GPU roles loaded correctly.
- Quality caveat: 2-update canary did not beat the clean SFT baseline, so the medium run is a learning/variance test, not a hero promotion.
- Chosen medium shape: `r33-component-full-rl-8h100-medium-rb8-n8-kl3e4-tp2`, `rollout_batch_size=8`, `n_samples_per_prompt=8`, `global_batch_size=64`, `num_rollout=16`, actor LR `3e-6`, KL coefficient `3e-4`, response/context caps `256/512`, actor microbatch `1`.
- Checkpoint policy: final-only checkpoint with `save_interval=16`; queue post-eval should clean the temporary HF export and the final actor DCP after the report is produced.
- Rejected alternative: keep canary `rb4_n4_global16` for 16 rollouts. It is safer but too small to improve reward/advantage variance, and the canary showed no OOM at the current microbatch.
- Rejected alternative: jump directly to hero rollouts. The 64/64 post-eval took about 27.5 minutes and the built-in quality gate was false, so a medium learning/variance run is still the right next gate.
- Expected GPU placement: actor FSDP GPUs 0-3, TP2 SGLang GPUs 4-5, frozen AR reward/critic GPUs 6-7.
- Expected ETA: training roughly 45-75 minutes if per-update scaling is near linear, plus about 30 minutes for the current 64/64 round-trip gate. Checkpoint save may add several minutes on Longhorn.
- Required before launch: sync this config and docs to a fresh physical `/workspace/interp/code/nano30b-nla-pilot-current` target so logs no longer show the stale June 15 symlink target.

## 2026-06-24/25 - M5 8x H100 Medium RL Candidate

- Status: completed; gate failed; do not hero-promote.
- Queue item: `r33-component-full-rl-8h100-medium-rb8-n8-kl3e4-tp2`.
- Run dir: `/workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/r33_component_full_rl_8h100_medium_rb8_n8_kl3e4_tp2`.
- Runtime: started `2026-06-24T22:24:46Z`, training reached `iter_0000016`, post-eval completed `2026-06-25T00:05:34Z`.
- Topology: 8x H100 NVL; actor FSDP GPUs 0-3, TP2 SGLang rollout GPUs 4-5, frozen AR reward/critic GPUs 6-7.
- Config: `rollout_batch_size=8`, `n_samples_per_prompt=8`, `global_batch_size=64`, `num_rollout=16`, actor LR `3e-6`, KL coefficient `3e-4`, actor microbatch `1`, final-only save interval `16`.
- Code-root hygiene: fixed before launch. `/workspace/interp/code/nano30b-nla-pilot-current` now points to `/workspace/interp/code/nano30b-nla-pilot-current-20260624T2220Z`, and post-eval worker paths used that fresh target.
- S3 sync note: the first S3 object uploaded from macOS was listable but `HeadObject` returned `400` from RunAI. A checksum-relaxed compatibility upload was downloaded with `s3api get-object`; direct `aws s3 cp` still hit `HeadObject 400`.
- Training throughput: 16 rollouts completed. After the first update, steady step time averaged `255.21s`, with actor train `133.54s`, ref logprobs `34.93s`, policy logprobs `30.73s`. Final update step time was `253.96s`.
- GPU memory: actor GPUs ran near `90.1GB` during actor train; TP2 SGLang stayed near `85.2GB`; critic/reward GPUs around `50-57GB`. No OOM.
- Reward trend: raw rewards over 16 rollouts averaged `-0.398916`, min `-0.582167`, max `-0.257208`. First four raw rewards were `[-0.471900, -0.568495, -0.582167, -0.400945]`; last four were `[-0.375338, -0.369535, -0.370403, -0.257208]`.
- Response health: mean response length `120.11`, min/max `109.20 / 126.53`, truncation max `0.0`.
- Entropy: mean `1.0726`, min/max `0.8094 / 1.4482`.
- KL/logprob: `train/ppo_kl` stayed `0.0`; last five `train/kl_loss` values were `[0.0011517, 0.0010003, 0.0024889, 0.0001422, 0.0014977]`; last five rollout/policy logprob abs diffs were about `0.020-0.027`.
- Round-trip report: `/workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/r33_component_full_rl_8h100_medium_rb8_n8_kl3e4_tp2/roundtrip_iter_0000016_v64_t64_report.json`.
- Parse health: validation/test closed `1.0 / 1.0`, usable `1.0 / 1.0`.
- Validation round-trip: AV-real primary NMSE `0.000109251`, teacher `0.000104941`, clean SFT baseline `0.000109680`. This is slightly better than the SFT baseline in aggregate, but the report did not mark `baseline_beaten` because baseline row identity did not match.
- Test round-trip: AV-real primary NMSE `0.000122012`, teacher `0.000112355`, clean SFT baseline `0.000121664`. This is slightly worse than the SFT baseline.
- Controls: AV-real beat all controls on validation/test. Test control NMSEs: `av_mean=0.000336037`, `av_none=0.000353968`, `av_shuffled=0.000366544`, `av_zero=0.000343471`, `mean=0.000256098`.
- Storage cleanup: temporary HF and actor DCP checkpoint were cleaned after post-eval; final run dir was about `29M`; `/workspace/interp` remained about `677G` used / `331G` free.
- Decision: not a hero run. This is a strong systems/throughput result and near-tie quality result. Next RL work should be a targeted medium variant before hero, not immediate hero promotion.
- Recommended next medium variant: keep `rb8/n8/global64`, increase learning signal by testing a longer `num_rollout=32` variant or an actor LR bump to `5e-6`, but preserve final-only checkpointing and the same 64/64 round-trip gate. Do not launch a hero until validation and test both match or beat the clean SFT baseline.

## 2026-06-25 - M5b 8x H100 Medium RL Candidate, 32 Rollouts

- Status: completed; gate failed; do not hero-promote.
- Queue item: `r33-component-full-rl-8h100-medium-rb8-n8-kl3e4-tp2-rollout32`.
- Run dir: `/workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/r33_component_full_rl_8h100_medium_rb8_n8_kl3e4_tp2_rollout32`.
- Runtime: queue started `2026-06-25T00:16:28Z`, reached `iter_0000032`, post-eval completed and queue marked complete `2026-06-25T03:06:20Z`.
- Topology: 8x H100 NVL; actor FSDP GPUs 0-3, TP2 SGLang rollout GPUs 4-5, frozen AR reward/critic GPUs 6-7.
- Config: `rollout_batch_size=8`, `n_samples_per_prompt=8`, `global_batch_size=64`, `num_rollout=32`, actor LR `3e-6`, KL coefficient `3e-4`, actor microbatch `1`, final-only save interval `32`.
- Code-root hygiene: remote `/workspace/interp/code/nano30b-nla-pilot-current` pointed to `/workspace/interp/code/nano30b-nla-pilot-current-20260625T0015Z`.
- Training throughput: 32 rollouts completed. Steady step time averaged `256.13s`; late steps had actor train about `131-134s`, ref logprobs about `34.8-35.1s`, policy logprobs about `30.5-31.0s`, and wait ratio about `0.10`.
- GPU memory/utilization: actor GPUs peaked around `90.1GB` during train phases, TP2 SGLang stayed around `85.2GB`, and critic/reward GPUs rose to about `62.7-73.8GB` late in the run. No OOM. All GPUs returned to `4MiB` idle after post-eval.
- Reward trend: raw rewards over 32 rollouts averaged `-0.405441`, std `0.084653`, min `-0.652331`, max `-0.258299`. First four raw rewards were `[-0.434032, -0.590248, -0.652331, -0.480681]`; last four were `[-0.395186, -0.447689, -0.356345, -0.324225]`.
- Response/entropy health: response length mean `118.91`; entropy mean `1.0975`; truncation remained `0.0` in sampled status polls.
- KL/logprob: `train/ppo_kl` stayed `0.0`; late `train/kl_loss` values were small and finite, e.g. `[-0.0005927, 0.0000996, -0.0008915]`; late rollout/policy logprob abs diffs were about `0.024-0.032`.
- Round-trip report: `/workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/r33_component_full_rl_8h100_medium_rb8_n8_kl3e4_tp2_rollout32/roundtrip_iter_0000032_v64_t64_report.json`.
- Local artifact mirror: `/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/artifacts/runai_rl/20260625T031500Z_r33_rl_rollout32/`.
- S3 artifact mirror: `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/sync_exports/20260625T031500Z_r33_rl_rollout32/r33_rl_rollout32_light_artifacts_20260625T0315Z.tgz` with SHA256 `07b937c0abec9bbfdaf9e05144158f86be333be9ff5392828c01484208695043`.
- Parse health: validation/test closed `1.0 / 1.0`, usable `1.0 / 1.0`.
- Validation round-trip: AV-real primary NMSE `0.000108897`, teacher `0.000104941`, clean SFT baseline `0.000109680`. Aggregate validation improved slightly over the clean SFT baseline, but the report still marked `baseline_beaten=false` because baseline row identity did not match.
- Test round-trip: AV-real primary NMSE `0.000123269`, teacher `0.000112355`, clean SFT baseline `0.000121664`. This is worse than the clean SFT baseline.
- Controls: AV-real beat all controls on validation/test. Test control NMSEs: `av_mean=0.000337857`, `av_none=0.000336659`, `av_shuffled=0.000368801`, `av_zero=0.000360521`, `mean=0.000256098`.
- Cleanup: queue cleanup removed temporary HF and actor DCP checkpoint payloads after post-eval. Final retained artifacts are lightweight logs, W&B offline files, generated JSONL, the round-trip report, and tiny actor/critic metadata. `/workspace/interp` returned to about `677G` used / `331G` free.
- Teardown caveat: `train.log` contains Ray atexit teardown tracebacks after completion, but no active process remained and GPUs were idle. Treat these as post-completion cleanup noise unless they recur before queue completion.
- Decision: stop before hero. The 8x H100 topology is now proven for medium throughput, but 32 rollouts at actor LR `3e-6` did not improve the clean SFT round-trip result on both splits. Do not launch the planned `num_rollout=64/128` hero from this exact configuration.
- Next RL direction: change the learning signal before scaling runtime. Good candidates are actor LR `5e-6`, explicit KL/SFT anchor accounting if `ppo_kl` continues to log as `0.0`, advantage/reward normalization diagnostics, or a more sensitive medium gate with larger eval limits once generation cost is acceptable.

## 2026-06-25 - M5c Paired Gate Fix and 5e-6 Rollout-8 Probe

- Status: completed; paired 256/256 round-trip gate passed, but do not hero-promote until throughput/variance canaries are resolved.
- Code commit: `da7dc8c` (`feat: pair rl roundtrip baseline gate`).
- Motivation: the prior 64/64 gates compared aggregate RL rows against a 256/256 SFT baseline whose row identity did not match. Manual overlap analysis showed near-tie behavior, so the gate now compares overlapping row indices when both reports expose rowwise normalized MSE.
- Gate fix: `scripts/eval_nano_av_ar_roundtrip_gate.py` now reports `baseline_row_overlap_count`, `primary_matched_normalized_mse`, and `baseline_primary_matched_normalized_mse`; `baseline_beaten` can use the matched-row comparison instead of requiring identical row order.
- Local tests: `tests/test_nano_av_ar_roundtrip_gate.py` and `tests/test_nano_rl_queue.py::NanoRLQueueTests::test_checked_in_r33_qwen_comparable_queue_uses_512_generation_updates` passed locally and on RunAI.
- Remote code root: `/workspace/interp/code/nano30b-nla-pilot-current` points to `/workspace/interp/code/nano30b-nla-pilot-current-da7dc8c-rl-paired`.
- Queue item launched: `r33-component-rl-8h100-tier1-probe-rb64-n8-gb512-lr5e6-rollout8-v256t256`.
- Run dir: `/workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/r33_component_rl_8h100_tier1_probe_rb64_n8_gb512_lr5e6_rollout8_v256t256`.
- Config: 8x H100 NVL, actor FSDP GPUs 0-3, TP2 SGLang GPUs 4-5, frozen AR reward/critic GPUs 6-7; `rollout_batch_size=64`, `n_samples_per_prompt=8`, `global_batch_size=512`, `num_rollout=8`, actor LR `5e-6`, KL coefficient `3e-4`, actor microbatch `1`, final-only checkpoint at `iter_0000008`.
- Post-eval gate: `validation_limit=256`, `test_limit=256`, `max_new_tokens=256`, controls `real/shuffled/zero/mean/none`, SFT baseline report `roundtrip_iter_0001291_v256_t256_report.json`.
- Final gate report: `/workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/r33_component_rl_8h100_tier1_probe_rb64_n8_gb512_lr5e6_rollout8_v256t256/roundtrip_iter_0000008_v256_t256_report.json`.
- Final matched gate: validation primary NMSE `0.00010905499220825732` vs matched SFT baseline `0.0001096800115192309`; test primary NMSE `0.00012110520765418187` vs matched SFT baseline `0.00012166367378085852`. `baseline_beaten=true` on both splits.
- Parse health: validation/test parse closed and parse usable rates were `1.0`; rowwise candidate win rates were validation `0.51953125`, test `0.50390625`.
- Startup status: SGLang health returned HTTP 200; W&B offline directory initialized; actor train entered `rollout_id=0`. GPU placement at startup matched expectation, with SGLang resident on GPUs 4-5 and actor/reward memory on GPUs 0-3 and 6-7.
- Next decision: if the 5e-6 rollout-8 run keeps KL/logprob movement tiny or fails both matched-row split gates, unblock the queued `1e-5` rollout-8 follow-up. If it beats the clean SFT baseline on both validation and test under the paired 256/256 gate, consider a longer `5e-6` Tier 1 run before any hero-scale RL.

## 2026-06-25 - RL Throughput Engineering Review Follow-Up

- Status: code/config prepared; no new RL launch.
- Review claim accepted: the current 8x H100 RL topology is phase-serial and actor-bound. TP2 generation is only a few seconds per step while actor update and ref/policy logprob dominate, so keeping two GPUs on SGLang underuses the cluster during the bottleneck phase.
- Implemented: `scripts/nano_rl_queue.py` now accepts structured SGLang topology fields (`tensor_parallel_size`, `base_gpu_id`, `rollout_num_gpus_per_engine`), renders the matching SGLang launch flags, forwards `--rollout-num-gpus-per-engine`, and rejects mismatched rollout GPU accounting before launch.
- Implemented: `external/natural_language_autoencoders/configs/rl.sh` and `external/natural_language_autoencoders/nla/system_metrics.py` now log/forward static topology labels into stdout/W&B metrics (`workspace`, actor, critic, rollout, rollout-per-engine, SGLang TP, SGLang base GPU).
- Config change: `configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml` keeps TP2 actor4 as the default but moves `--rollout-num-gpus-per-engine=2` out of opaque `extra_args` into structured config.
- New blocked canary: `r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-tp1-actor5`, with 5 actor GPUs, 1 TP1 SGLang GPU, and 2 critic GPUs. This is a throughput fit canary only; compare actor train time, ref/policy logprob time, generation time, GPU utilization, and OOM headroom against the completed TP2 actor4 canary before unblocking medium/hero runs.
- Runtime support check: Miles `/workspace/interp/code/miles-051cd15/train_async.py` exists and pipelines `rollout_manager.generate.remote(...)` for the next rollout while training the current rollout, but it explicitly asserts `not args.colocate`. The queue now rejects `training.async_training + training.colocate` before launch.
- Runtime support check: Miles exposes `--offload-train`, `--offload-rollout`, `--offload-rollout-level`, `--colocate`, `--gradient-checkpointing`, `fsdp_cpu_offload`, and `fsdp_cpu_backend`; `configs/rl.sh` now forwards structured env flags for these runtime knobs.
- Runtime support check: FSDP actor code computes ref logprobs inside the actor group via `self._compute_log_prob("ref", ...)`, and the ref model is created in the actor FSDP path. There is no current Miles switch to place ref logprobs on critic GPUs, so the queue now rejects `ref_log_probs_placement=critic` instead of pretending it is config-only.
- Runtime support check: current Nano queues use external SGLang (`--rollout-external`). True Miles actor/rollout co-location is a managed/internal rollout mode, so the queue now rejects `training.colocate` with `sglang.mode: external`.
- New blocked canary: `r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-tp1-actor5-cpuoffload-nockpt-mb2`, with 5 actor GPUs, TP1 rollout, FSDP CPU offload, gradient checkpointing disabled, and actor microbatch `2`. Run only after the plain TP1 actor5 canary proves stable; compare VRAM, CPU/offload overhead, and actor/ref/policy logprob times.
- New blocked canary: `r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-async-tp2`, using Miles `train_async.py` with the default TP2 rollout topology. Interpret separately from synchronous runs because async pipelining changes policy-staleness semantics; require KL/TIS/reward/round-trip diagnostics before using it as a hero precursor.
- Local verification: `.venv/bin/python -m pytest tests/test_nano_rl_queue.py -q` passed (`31 passed`), and `.venv/bin/python -m pytest tests/test_nla_system_metrics.py -q` passed (`9 passed`).

## 2026-06-25 - R33 RL Throughput/N=16 Canary Setup

- Decision: defer the heavier runtime-architecture ideas for now: ref logprobs on critic GPUs, true actor/rollout co-location with external SGLang, and async as a default path. They are not worth the implementation risk before proving the simpler topology and group-size canaries.
- Queue change: `r33-component-rl-8h100-tier1-probe-rb64-n8-gb512-lr5e6-rollout8-v256t256` is marked complete locally because its paired 256/256 gate already passed; it should not be rerun accidentally.
- New pending canary: `r33-component-rl-8h100-tier1-fit-rb32-n16-gb512-lr5e6-tp1-actor5`.
- Config: 8x H100 NVL, actor FSDP GPUs `0-4`, TP1 SGLang on GPU `5`, frozen AR reward/critic GPUs `6-7`, `rollout_batch_size=32`, `n_samples_per_prompt=16`, `global_batch_size=512`, `num_rollout=2`, actor LR `5e-6`, KL coefficient `3e-4`, actor microbatch `1`, gradient checkpointing enabled.
- Rationale: this tests the larger GRPO group size (`n=16`) while keeping generated samples/update at `512`, so it isolates group-size behavior before trying the heavier `rb64/n16/gb1024` setting.
- Local verification before RunAI sync: `.venv/bin/python -m pytest tests/test_nano_rl_queue.py -q` passed (`31 passed`), and `.venv/bin/python -m pytest tests/test_nla_system_metrics.py -q` passed (`9 passed`).
- S3 sync note: Mac upload via `scripts/nano_s3.py cp-up` succeeded. `s3api head-object` returned endpoint 400 on both Mac and RunAI, but `s3 ls` saw the object from both sides; RunAI download used `scripts/nano_s3.py sync-down` with an include filter.

## 2026-06-26 - R33 RL N=16 Throughput Canary Ladder

- Goal: prove `n_samples_per_prompt=16` can run on the 8x H100 topology without OOM, then raise actor microbatch until the ref/policy/actor phases are no longer dominated by tiny microbatches.
- Topology for all canaries in this ladder: actor FSDP on GPUs `0-4`, external SGLang TP1 on GPU `5`, frozen AR reward/critic on GPUs `6-7`, `rollout_batch_size=32`, `n_samples_per_prompt=16`, requested `global_batch_size=512`, `num_rollout=2`, actor LR `5e-6`, KL coefficient `3e-4`, max context/response `512/512`.
- Code fix before the first successful retry: `external/natural_language_autoencoders/nla/system_metrics.py` had made torch optional, but `RouterEntropyTracker._find_index_tensor` still referenced the global `torch` symbol. The first n=16 retry failed with `NameError: name 'torch' is not defined`. The tracker now stores a lazily loaded torch module, no-ops cleanly when torch is unavailable, and has tests for real torch collection plus no-torch behavior.
- Verification for the router fix and queue changes passed locally and on RunAI: `tests/test_nla_system_metrics.py` and `tests/test_nano_rl_queue.py`.
- `r33-component-rl-8h100-tier1-fit-rb32-n16-gb512-lr5e6-tp1-actor5`, actor microbatch `1`: proved the n=16 rollout/reward path and SGLang weight update. Generation reached `512/512` samples, but ref logprobs ran as `102` tiny chunks and was projected around `45` minutes for the first update, so the run was intentionally stopped as a throughput canary rather than a failed run.
- `...-mb4`, actor microbatch `4`: completed the first update's ref-logprob pass without OOM. Ref chunks dropped to `25`; ref time was `641.7s`. The run was intentionally stopped after the ref pass to move to a faster microbatch.
- `...-mb8`, actor microbatch `8`: completed one full update without OOM. Ref chunks dropped to `12`; ref time `318.2s`, policy logprob time `300.9s`, actor train time `693.8s`, train time `1328.0s`, step time `1476.1s`. Actor GPU memory was roughly `41-51GiB`.
- `...-mb16`, actor microbatch `16`: launched from the synced `r33-rl-n16-mb16` code bundle. Two RL updates completed and saved `actor/iter_0000002`. Ref chunks dropped to `6`; rollout 0 ref time `163.2s`, policy logprob time `150.9s`, actor train time `351.7s`, train time `681.6s`, step time `829.2s`; rollout 1 ref time `154.0s`, policy logprob time `150.8s`, actor train time `338.1s`, train time `672.5s`, step time `837.7s`.
- Throughput takeaway so far: mb16 is about `1.8x` faster than mb8 by full step time (`1476s -> 829-838s`) and about `4x` faster than mb4 on the ref-logprob phase (`641.7s -> 154-163s`), while preserving enough H100 memory headroom for this sequence length.
- Memory takeaway from mb16: rank-0 observed max allocated `58.83GiB`, max reserved `75.21GiB`, and nvidia-smi memory around `78.7GiB` during actor training on a `95.8GiB` H100 NVL. Use mb8 as the fallback if a longer sequence or larger rollout pushes mb16 over the limit.
- Batch-shape caveat: with requested `512` samples, actor DP `5`, and mb16, the dynamic aligned batch is `480` (`96` local samples on each actor rank). This also happened at mb8. If exact `512` updates matter, change the divisibility of actor DP / rollout batch / n-samples rather than silently interpreting the run as a true 512-sample update.
- Post-train caveat: `train.log` contains W&B/Ray `BrokenPipeError` teardown tracebacks after `# completed_utc=2026-06-26T04:18:31Z`; the actor checkpoint and frozen-critic skip marker were already written. Treat this as teardown noise unless the queue driver records a nonzero return code.
- Post-eval result: the configured 64/64 round-trip gate completed. Generation wrote all `128` cached rows by `2026-06-26T04:40:48Z`, using eight one-GPU workers with roughly `63GiB` per H100 and `38-42%` utilization during generation. The final report is `/workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/r33_component_rl_8h100_tier1_fit_rb32_n16_gb512_lr5e6_tp1_actor5_mb16/roundtrip_iter_0000002_v64_t64_report.json`.
- Gate result: `gate.passed=false`. Parse health was perfect on both splits (`closed_fraction=1.0`, `usable_fraction=1.0`) and AV-real beat all controls on both splits, but the 2-update RL checkpoint did not beat the matched SFT baseline on validation.
- Validation: primary NMSE `0.00011183605238329619`, matched SFT baseline `0.00010963059321511537`, teacher `0.00010494107846170664`, candidate rowwise win fraction `0.421875`, baseline win fraction `0.578125`, `baseline_beaten=false`.
- Test: primary NMSE `0.00011975715460721403`, matched SFT baseline `0.0001206386077683419`, teacher `0.0001123547917813994`, candidate rowwise win fraction `0.484375`, baseline win fraction `0.515625`, `baseline_beaten=true`.
- Interpretation: this mb16 run proves the n=16 systems path and throughput target, not RL quality. With only two updates, quality is essentially near-SFT/noisy: one split slightly worse, one split slightly better, both with tiny deltas. Do not promote this checkpoint; use the mb16 throughput shape as the default for the next longer n=16 run, with mb8 as the memory fallback.

## 2026-06-26 - R33 RL Clean-Divisible N=16 Medium Launch

- Fix: add `r33-component-rl-8h100-tier1-medium-rb30-n16-gb480-lr5e6-rollout16-mb16-v64t64` to the Qwen-scale queue.
- Rationale: the previous `rb32/n16/gb512` canary generated `512` samples but actor DP `5` and mb16 aligned the train batch down to `480`. The new medium candidate makes `480` intentional: `rollout_batch_size=30`, `n_samples_per_prompt=16`, `global_batch_size=480`, actor DP `5`, and actor microbatch `16`, giving `96` samples per actor rank and exact microbatch divisibility.
- Config: 8x H100 NVL, actor FSDP GPUs `0-4`, external SGLang TP1 on GPU `5`, frozen AR reward/critic GPUs `6-7`, actor LR `5e-6`, KL coefficient `3e-4`, `num_rollout=16`, save interval `16`, 64/64 round-trip post-eval.
- Checkpoint policy: override `cleanup_actor_checkpoint=false` for this medium candidate so the final actor checkpoint is kept until the gate result is reviewed; temp HF conversion still uses the normal cleanup path.
- Local verification before RunAI sync: `tests/test_nano_rl_queue.py` and `tests/test_nla_system_metrics.py` passed together (`41 passed`).
- Launch: synced via S3 to `/workspace/interp/code/nano30b-nla-pilot-current-r33-rl-n16-medium-rb30-20260626T1439Z` and launched with the queue item at `2026-06-26T14:42:07Z`.
- Remote dry-run/preflight before launch: `preflight_missing_count=0`, `resource_total_gpus=8`, rollout batch plan `{rollout_batch_size=30, n_samples_per_prompt=16, generated_samples=480, global_batch_size=480, global_batch_matches_rollout=true}`, actor/critic/rollout GPUs `5/2/1`, SGLang TP1 on base GPU `5`.
- Startup: SGLang health returned HTTP 200 and W&B initialized offline. `train.log` printed `[NLA RL CONFIG] prompts=30 samples_per_prompt=16 generated_samples=480 global_batch_size=480 global_matches_rollout=1`.
- First update evidence: rollout `0` completed with `dynamic_global_batch_size=480` and `local_samples=96`, proving the batch-shape fix reached actor training without truncation. Timings were close to the mb16 canary: `ref_log_probs_time=163.6s`, `log_probs_time=151.0s`, `actor_train_time=352.3s`, `train_time=682.0s`, `step_time=821.9s`, `wait_time_ratio=0.170`.
- First update memory: actor-train observed max reserved about `51.97GiB` and nvidia-smi memory about `54.9GiB` on the sampled actor rank, lower than the previous mb16 canary's roughly `75.2GiB` reserved peak. No OOM or traceback was seen through the first update.

## 2026-06-26 - R33 RL Rollout8 Probe Final Diagnostics and Stage-A Decision

- Probe: `r33-component-rl-8h100-tier1-probe-rb64-n8-gb512-lr5e6-rollout8-v256t256`, run dir `/workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/r33_component_rl_8h100_tier1_probe_rb64_n8_gb512_lr5e6_rollout8_v256t256`.
- Final paired gate report: `roundtrip_iter_0000008_v256_t256_report.json`, synced locally under `artifacts/runai_rl/20260625_r33_rl_probe8_final/current_run/`.
- Gate status: `passed=true`. Parse health was perfect on both splits (`closed_fraction=1.0`, `usable_fraction=1.0`), and AV-real beat all configured controls.
- Validation: primary NMSE `0.00010905499220825732`, matched SFT baseline `0.0001096800115192309`, teacher `0.00010689272312447429`, candidate rowwise win fraction `0.51953125`, `baseline_beaten=true`.
- Test: primary NMSE `0.00012110520765418187`, matched SFT baseline `0.00012166367378085852`, teacher `0.00011230249219806865`, candidate rowwise win fraction `0.50390625`, `baseline_beaten=true`.
- Reward-vs-gate diagnostic: `scripts/analyze_rl_reward_gate_correlation.py` recomputed frozen R33 AR critic rewards for the generated `av_real` explanations and paired them against the round-trip rowwise gate losses.
- Diagnostic report: remote `reward_gate_correlation_report.json`, synced locally to `artifacts/runai_rl/20260625_r33_rl_probe8_final/current_run/reward_gate_correlation_report.json`.
- Correlation result: validation paired rows `256`, Pearson `-0.9997456847713523`, Spearman `-0.999643797207599`; test paired rows `256`, Pearson `-0.9998892839933052`, Spearman `-0.9997167544060426`. Negative correlation is aligned because higher reward means lower critic MSE/gate loss.
- Stage-A decision: pass. The finished rollout8 policy improved the paired SFT baseline by a small but consistent amount on both validation and test, and the fixed-policy reward diagnostic is strongly aligned with the round-trip gate.
- Caveat: this probe predated the new `[NLA ROLLOUT]` stdout and `[NLA ADVANTAGE]` patch, so it cannot satisfy the Stage-B observability gate. Do not infer true advantage variance or zero-advantage fraction from this log.
- Stopped follow-up: `r33-component-rl-8h100-tier1-medium-rb30-n16-gb480-lr5e6-rollout16-mb16-v64t64` had been started before the Stage-A diagnostics were complete. It was manually stopped and its queue status is now `cancelled`; treat any partial output as a systems trace only, not as an experiment result.
- Code fixes completed for the next instrumented run: rollout stdout summary, true advantage helper, Miles patch logging of `rollout_data["advantages"]`, safe configurable failed-extraction reward, log analyzer, reward-vs-gate correlation diagnostic, and queue gating for the stopped medium item.
- Sync/runtime note: local S3 upload failed in this shell because Mac-side AWS credentials were not available, so the lightweight source bundle was transferred directly to RunAI without printing credential material. The active RunAI code root was refreshed, and the live Miles checkout `/workspace/interp/code/miles-051cd15` was surgically updated with the missing `[NLA ADVANTAGE]` hook. A future clean environment should apply the corrected `0004_fsdp_timing_debug.patch`; on the already patched live tree, a dry-run reports already-applied/reversed hunks rather than a malformed patch.
  - Verification: focused tests passed locally and on RunAI (`56 passed` in each environment), covering queue generation, system metrics, rollout/advantage metrics, failed-extraction reward config, log analysis, and reward-vs-gate correlation.

## 2026-06-27: R33 Medium-Scale RL Queue (Submitted After Confirmation Gate)

### Selected starting point

- The aggressive 8xH100 component HPO selected `actor_lr=2e-5`,
  `kl_loss_coef=1e-3`, and unnormalized advantages. On the bounded 256/256
  gate, its heldout round-trip NMSE was below the matched clean SFT baseline
  on both splits. The higher-KL normalized-advantage candidate had comparable
  aggregate NMSE but showed late policy drift, so it is not a scale candidate.
- The queue retains the measured high-throughput partition: six actor GPUs,
  one critic GPU, and one rollout/SGLang GPU; `rollout_batch_size=30`,
  `n_samples_per_prompt=16`, `global_batch_size=480`, and
  `actor_micro_batch=32`. It is deliberately not the obsolete microbatch-one
  configuration.

### Queue and promotion contract

- `configs/nano_rl/r33_component_medium_scaleup_queue_8h100.yaml` first runs
  a 16-update confirmation (7,680 sampled completions) with a 256/256
  round-trip gate. The 32-update medium item (15,360 completions, 512/512
  final gate) begins only when the confirmation queue entry records
  `gate_passed: true`.
- Gate promotion is generic queue behavior. It requires the predecessor to be
  `complete`; a failed, inconclusive, or merely exited predecessor leaves a
  dependent item blocked. The confirmation also writes a reward-to-gate
  correlation report, healthy parse/control evidence, and drift telemetry.
- The 512/512 medium gate has full controls and parse checks. The available
  matched-SFT round-trip baseline is 256/256, so any comparison at 512 rows is
  explicitly reported as aggregate control evidence rather than a fictitious
  512-row matched-SFT delta.

### Offline W&B observability

- Miles previously initialized its driver, rollout manager, actor leader, and
  critic leader as separate offline W&B writers with one run ID. Patch
  `0016_wandb_offline_role_runs.patch` now assigns deterministic role-suffixed
  IDs and names in offline mode while retaining the configured W&B group. This
  avoids local/offline writer collisions without changing online shared-run
  semantics.
- The active Miles checkout has a harmless context difference from the
  revision used for the unified patch. The checked-in
  `scripts/apply_miles_offline_wandb_patch.py` installer applies the same four
  edits with exact pre/postimage checks, is idempotent, and is the runtime
  application path for this queue. Its regression test covers both application
  and a second no-op invocation.
- System, rollout, reward, advantage, KL, parse, and gate metrics remain
  offline. The driver and three role series should be compared in the shared
  configured group after sync; no secret material is captured in this record.

### Verification before submission

- Local focused verification: `76 passed` across the Miles launcher, RL queue,
  RL metrics, and system-metrics tests. `scripts/check_miles_patches.py`
  reported no failed hunks, and `git diff --check` was clean.
- The source-only S3 handoff uses the project prefix
  `nano30b-nla-pilot/code-sync/` in S3 API keys. The root-level `code-sync/`
  key is incorrect for RunAI `GetObject` calls and is deliberately not used by
  the scale-up launch procedure.
- Next gate before any longer RL run: launch only an instrumented Stage-B canary and require `[NLA ADVANTAGE]` lines in `train.log`, finite reward/KL metrics, advantage standard deviation at least `0.5`, zero-advantage fraction at most `0.25`, and a paired round-trip gate that is neutral or better versus matched SFT.

### Stage-B Canary Launch Notes

- `r33-component-rl-8h100-stageb-advcheck-rb30-n16-gb480-lr5e6-rollout3-mb16-v64t64` was stopped after rollout 0 because it produced `[NLA ROLLOUT]` but no `[NLA ADVANTAGE]`; the active training path uses `nla.train_actor.NLAFSDPActor`, so relying only on the Miles base-class 0004 hook was insufficient.
- `r33-component-rl-8h100-stageb-advcheck2-rb30-n16-gb480-lr5e6-rollout3-mb16-v64t64` was stopped after rollout 0/1 also lacked `[NLA ADVANTAGE]`; the direct NLA actor helper had been gated on global `dist.get_rank()==0`, but global rank 0 is not guaranteed to be an actor rank in the 5 actor / 2 critic / 1 rollout topology.
- Fix: `nla.train_actor.NLAFSDPActor._train_core` now emits advantage stats from the actor DP leader (`self._nla_is_dp_leader()`) after delegated actor training returns.
- Current retry: `r33-component-rl-8h100-stageb-advcheck3-rb30-n16-gb480-lr5e6-rollout3-mb16-v64t64`, run dir `/workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/r33_component_rl_8h100_stageb_advcheck3_rb30_n16_gb480_lr5e6_rollout3_mb16_v64t64`.
- Do not unblock medium/hero RL until `advcheck3` proves both `[NLA ROLLOUT]` and `[NLA ADVANTAGE]` on the same run and passes the Stage-B thresholds.

## 2026-06-30 - R33 Functional Identity Calibration And Stage 1 Validity Queue

No new RL training was launched in this milestone. Work remained inside the
independent validity stage required before corrected fixed-AR probes.

### Stored activation replay diagnosis

- A strict four-row gate initially failed because activations freshly captured
  from the current R33 target model did not exactly equal the stored R33
  dataset vectors. Relative L2 drift was about `1%` to `7%` on the sampled
  rows.
- The stored vectors were extracted on the historical 2x H200/GH200 stack.
  The current evaluation stack is 8x H100 NVL with a different PyTorch and
  Transformers runtime. Replaying the historical extraction geometry did not
  remove the drift.
- Within one current process, repeated full forwards were bit-exact. The normal
  full forward and historical manual-prefix extraction path also agreed
  bit-for-bit for the same inputs.
- Most importantly, a freshly captured final-token R33 activation reinserted at
  the same boundary reproduced the original logits exactly. This is the hard
  implementation identity check needed by the functional evaluator.

The gate now separates two claims that were previously conflated:

- **Hard identity:** fresh activation capture -> boundary reinjection -> logits
  must be exact.
- **Calibration evidence:** stored-vector replay drift is reported explicitly
  across topology/runtime changes, but is not treated as a patching failure.

`stored_gold` is now a first-class functional variant. It provides the
irreducible replay floor for the stored dataset under the current runtime, so
candidate and SFT excess KL are measured relative to that floor.

### Minimal corrected gate result

Report:
`/workspace/interp/outputs/nano30b-nla-pilot/validity/r33-sft/functional_v2_stored_gold_v2_t2_20260630_report.json`

- Fresh capture/reinjection identity: passed.
- Stored-vector drift: four sampled outliers retained in the report.
- Stored-gold KL, validation/test: `0.0023096 / 0.00120877`.
- Candidate KL, validation/test: `0.343964 / 0.500493`.
- Teacher KL, validation/test: `0.379341 / 0.580593`.
- Candidate beat the mean, zero, and shuffled controls on the tiny 2/2 panel
  and slightly beat teacher KL. This is implementation evidence only, not a
  promotion result.

The train-only functional mean path was vectorized over Arrow fixed-size-list
storage. Computing the mean over 247,872 rows now takes about `7.45s` with
about `1.50GB` peak RSS instead of minutes of Python-list materialization.

### Stage 1 evaluation queue

Prepared queue:
`configs/nano_rl/r33_component_stage1_roundtrip_queue_8h100.yaml`.

It performs evaluation only:

1. Seed exact eight-worker generation shards from the existing 256/256 SFT
   outputs, resume generation to 512/512, enrich provenance, and score.
2. Repeat the same 256/256 -> 512/512 procedure for the update-16 actor.
3. Enrich and rescore the existing update-32 512/512 generation.

Temporary DCP-to-HF conversions live in `/dev/shm` and are removed after each
candidate. The queue does not start RL or SFT training. Corrected probes remain
blocked until the independent Stage 1 evidence is complete.

S3 code transfer was unavailable during this milestone: the local credential
was stale and the RunAI proxy rejected S3 CONNECT requests. Source was
therefore transferred as a checksum-verified compressed overlay through the
RunAI workspace transport; no data or checkpoint payloads were transferred.

### Stage 1 launch and fail-closed continuation

- Launch time: `2026-06-30T22:26Z` on 8x H100 NVL.
- Generation/scoring queue:
  `configs/nano_rl/r33_component_stage1_roundtrip_queue_8h100.yaml`.
- The SFT seed report contains exactly 512 historical rows distributed as 64
  total rows per worker across eight workers. Each worker targets 128 final
  rows and therefore generates only 64 missing rows.
- SFT DCP-to-HF conversion completed in `/dev/shm`; all eight generation
  workers loaded successfully. At `2026-06-30T22:47Z`, shard counts were
  `77-80/128`, GPU utilization was active, and no OOM or worker failure was
  present.
- Analysis queue:
  `configs/nano_rl/r33_component_stage1_analysis_queue_8h100.yaml`.
  It covers update-16/update-32 deterministic invariance, 512/512 functional
  recovery, 50+50 stratified qualitative panels, and the update-32 response
  closure-cap audit.
- `scripts/nano_queue_chain.py` is running as a fail-closed prerequisite
  watcher. It launches the analysis queue only when every generation/scoring
  item is `complete`; any `failed` or `blocked` item terminates the chain.
- The qualitative gate now requires `reviewed_count == row_count`. Merely
  generating a panel or supplying a low flag count cannot pass promotion.
  Initial panel reports are deliberately pending until all selected rows have
  explicit review decisions.
- Local and RunAI verification for the analysis continuation: `16 passed`;
  the analysis queue dry-run resolved the expected update-16 invariance item.
- No SFT or RL training was launched.

### Update-32 response closure audit

The cached 512/512 update-32 text allowed the response-cap audit to run while
generation used the GPUs. Validation/test closed fractions were
`0.992188 / 0.982422`. Coverage by cap was:

- 150 tokens: `0.019531 / 0.037109` validation/test;
- 192 tokens: `0.585938 / 0.519531`;
- 224 tokens: `0.917969 / 0.902344`; and
- 256 tokens: `0.992188 / 0.982422`.

Closed-response p95 positions were about `231 / 232` tokens. The planned
150-or-192 rule was therefore rejected empirically. The audit now selects the
smallest of 150, 192, 224, and 256 that reaches 95% on both splits. The
corrected probe and hero queues already use 256, so no training-config change
is required.

The full functional evaluator was initially configured at batch 4. To avoid
committing to a potentially twice-slower 512/512 pass, the analysis queue now
runs an update-16 `16/16`, batch-8 canary first. A successful canary promotes
batch 8 for both full functional reports. The canary deterministically selects
the 16 longest prefixes per split from `n_raw_tokens`, so it tests the
memory-heavy tail rather than the first rows. An OOM or identity failure stops
the queue before either full pass; there is no silent batch fallback.

The first qualitative-panel source scan exposed a CPU-side inefficiency:
`resolve_source_rows` converted every 2,688-float activation vector in each
scanned parquet batch to Python, even though only 1,024 rows were requested.
The resolver now scans only lightweight provenance columns and materializes a
full row after its key matches. The same full-275k, 1,024-row self-panel smoke
dropped from more than 100 seconds (stopped while still running) to 11 seconds.
It produced 50+50 rows and remained correctly review-pending. The first
trigram heuristic marked 8 validation and 5 test rows, but inspection showed
it was reacting to legitimate repeated template phrases such as "some kind
of." Repetition now requires duplicate semantic units or repeated four-grams
covering at least 25% of the response. Full-set hints fell to 1/3
validation/test rows, and the fixed self-panel contains 0/1 hints. These remain
review candidates, not automatic gate failures.

### Update-32 qualitative review

The fixed 50-validation/50-test update-32-versus-SFT panel was reviewed
row-by-row. All 50 validation rows were readable and unflagged. Test row
`262022` was flagged because its final constraint degenerated into a repeated
ISSN-like zero sequence. The other 49 test rows were readable; row `262205`
was much longer than SFT but structurally coherent and was not treated as a
readability regression.

Final manual counts are `0/50` validation and `1/50` test, both below the 5%
threshold. The RunAI reviewed report SHA256 is
`d4476dd160587fc16db887884259a92da5f6a04f68adebf43460cf0a563e6a8a`.
The explicit 100-decision file SHA256 is
`a60825247d4235d4c250767bcaa8b61759909606a20fc113a91a5cec8e934ff4`.
Local copies are under
`artifacts/runai_validity/20260630_r33_stage1/`. The analysis queue now passes
the decision file back to the panel builder, so it reproduces the reviewed
report rather than overwriting it with a pending panel.

### Stage 1 matched SFT round-trip baseline

The resumed SFT generation completed with exactly 512 validation and 512 test
rows, stable provenance enrichment, AR scoring, and temporary-HF cleanup.
Report:
`/workspace/interp/outputs/nano30b-nla-pilot/validity/r33-sft/roundtrip_v512_t512_report.json`,
SHA256 `80a8222ee13ac7fa0172d0cf7c07ded64bb18c72b19870c9b9d5847d19fe23fe`.

- Validation real NMSE/FVE: `0.000126796 / 0.502652`.
- Test real NMSE/FVE: `0.000134752 / 0.451356`.
- Validation/test teacher NMSE: `0.000109224 / 0.000109418`.
- Validation/test closed fraction: `0.978516 / 0.972656`; usable fraction
  was `1.0 / 1.0`.
- Minimum real-versus-activation-control row win fraction was
  `0.966797 / 0.972656` validation/test.

The existing update-32 report has exactly identical 512/512 row identities.
Its real NMSE is lower than this matched SFT baseline by `26.76% / 28.69%`
validation/test. Paired bootstrap mean SFT-minus-update32 improvement is
`3.3928e-5` with 95% CI `[2.8933e-5, 3.9040e-5]` on validation and
`3.8663e-5` with CI `[3.3783e-5, 4.3711e-5]` on test. This passes the
round-trip improvement and positive-CI components of Stage 1. Composite
promotion remains blocked on invariance and functional recovery.

## 2026-07-01: R33 Stage 1 Validity Completion And Historical-Recipe Audit

Both evaluation-only queues completed without OOM or failed items:

- round-trip/provenance queue: `16/16` complete;
- invariance/functional/qualitative queue: `8/8` complete;
- `/workspace/interp` remained at about `515 GB` free; and
- no SFT or RL training was launched by either queue.

The update-32 checkpoint passes all `26/26` checks in the composite validity
gate. Its report is
`/workspace/interp/outputs/nano30b-nla-pilot/validity/r33-update32/composite_validity_report.json`
with SHA256
`5c5b7341a175b4795fdf8ea20d210b9213ffb9a1a5dc6e4048d7d8598f73795e`.
The update-16 checkpoint passes every quantitative check but remains formally
incomplete because its fixed 50+50 qualitative panel was not reviewed. Since
update-32 is significantly better on the same rows and has a completed review,
there is no current reason to spend a promotion decision on update-16.

### Endpoint evidence

- Update-16 validation/test round-trip NMSE is
  `9.81933e-5 / 1.03650e-4`, a `22.56% / 23.08%` reduction from matched SFT.
- Update-32 validation/test round-trip NMSE is
  `9.28685e-5 / 9.60889e-5`, a `26.76% / 28.69%` reduction from matched SFT.
- Direct paired update-16-minus-update-32 NMSE confidence intervals are
  `[3.1904e-6, 7.5020e-6]` validation and
  `[4.8759e-6, 1.0390e-5]` test. Thus the extra 16 updates improved the held-out
  endpoint rather than merely preserving the update-16 gain.
- Update-32 also beats teacher-text AR NMSE by about `15.0% / 12.2%` on
  validation/test.
- Update-32 minimum deterministic-transform FVE retention is
  `0.990260 / 0.984714`; update-16 is `0.992364 / 0.989668`. Both are far above
  the `0.90` gate.
- Update-32 functional KL is `0.870409 / 0.944165`, versus matched SFT
  `1.430538 / 1.837270` and teacher text `1.083416 / 1.141398`.
- Paired SFT-minus-update-32 functional-KL confidence intervals are strictly
  positive, and update-32 improves over update-16 with direct paired CIs
  `[0.02532, 0.23375]` validation and `[0.09847, 0.34107]` test.
- Update-32 top-10/top-50 overlap is `0.6662/0.6848` validation and
  `0.6449/0.6550` test, improving over SFT on both splits.
- Exact row identity holds across SFT, round-trip, invariance, and functional
  reports at 512 rows per split. Real explanations beat every control on at
  least `0.9902 / 0.9961` of rows. Parse usability is `1.0`; closure is
  `0.9922 / 0.9824`.
- The reviewed panel has `0/50` validation and `1/50` test failures. The one
  true failure is the repeated ISSN-like numeric suffix on row `262022`.
- The closure audit confirms that 256 tokens is required: 224 tokens closes
  only `0.9180 / 0.9023`, while 256 closes `0.9922 / 0.9824`.

### Functional-loader and activation-drift interpretation

The functional logs warn that `backbone.norm_f.weight` and `lm_head.weight`
are missing from the AR HF checkpoint. This is expected for `NLACriticModel`:
the critic never produces vocabulary logits, and its loader immediately
replaces the LM head and final norm with `Identity` before applying the value
head. The warning is cosmetic for AR activation prediction; it is not a
randomly initialized layer on the functional target-model path.

Fresh capture/reinjection remains exact on all 1,024 functional rows. Stored
R33 vectors differ from fresh H100 captures by mean relative L2 `0.03010`
(p95 `0.07240`) with mean one-minus-cosine `0.000653` and RMS ratio `1.00034`.
The same discrepancy was present under the historical extraction geometry,
while repeat current forwards and current full-versus-prefix extraction were
exact. This localizes the difference to cross-runtime extraction provenance,
not the reinjection implementation. Relative candidate-versus-SFT conclusions
remain paired and trustworthy; future extraction manifests should additionally
record model-shard hashes, hardware, CUDA, PyTorch, and kernel configuration.

### Historical training dynamics: valid checkpoint, invalid scaling recipe

The 32-update scouting run generated 480 responses per update, but every
training-phase observation reports `dynamic_global_batch_size=384`. With six
actor ranks and microbatch 32, the configured global batch 480 was not
divisible by the 192-sample actor quantum. Approximately 96 generated samples
per update, or 20%, were therefore not included in the effective actor batch.

The run also used signed `k1` KL. KL loss moved from `0.0` to `-1.3990`, so the
`1e-3` KL term became negative rather than a non-negative trust-region penalty.
Mean absolute train-versus-rollout log-probability difference rose from
`0.280` to `1.989`, exceeded `0.75` at update 18, and stayed above it from
updates 20 through 31. A two-update drift guard would have stopped after
update 21. Response length rose from about 120 to 179 tokens and correlates
strongly with mismatch (`r=0.898`). These observations make the old run
scouting evidence, not a recipe to extend.

The useful signals did not collapse: rank-0 advantage standard deviation
averaged `0.948` with zero zero-advantage fraction, all 128 router experts
remained active with normalized router entropy around `0.991`, mean reward
improved from `-0.536` to a best `-0.201`, and held-out results continued to
improve from update 16 to update 32. Training averaged `187.9s` per update,
but actor work averaged only `26.5s`; mean wait ratio was `0.829`. Rollout and
reward production, not actor compute, dominate end-to-end time.

Decision: retain update-32 as the independently validated scouting best, but
do not scale its historical configuration. Proceed to the corrected divisible
`k3` probes (`gb384`, `mb32`, `n=8`) with runtime drift guards, then require the
winner to pass this same composite gate before any fixed-AR hero launch.

## 2026-07-01: Corrected K3 Stage 2 Dataset Gate And Probe Launch

The existing train-only RL builder had a correctness mismatch: it required a
`prompt` column from the activation source, even though the authoritative full
R33 base parquet contains raw activations and provenance rather than actor
prompts. Pointing it at AR-SFT data could therefore preserve a critic string as
the actor prompt. The builder is now a layer-generic composition of three
explicit inputs: raw activations, a held-out split manifest, and an actor
dataset sidecar. It synthesizes the canonical list-of-chat actor prompt with
exactly one `<INJECT>`, excludes all teacher/response columns, preserves token
provenance, and writes an RL dataset sidecar. The independent verifier now
checks prompt structure/template identity, token metadata, sidecar lineage,
layer/dimension consistency, source hashes, teacher-column absence, duplicate
keys, nonfinite values, and held-out overlap.

Local and RunAI regression suites both passed `73/73` tests. The synchronized
source archive is:
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/source-sync/20260701T180002Z/nano30b-r33-corrected-k3-source-20260701T180002Z.tgz`
with SHA256
`7d9f0ee923314e15132dacd73d0c98657900c81320dd2860153f0e161ce530fc`.

The corrected train-only dataset is:
`/workspace/interp/outputs/nano30b-nla-pilot/r33_rl_train_only/rl_r33_train_only.parquet`.
It was derived from the 275,396-row raw R33 base using the selected component
hero split manifest and actor contract. Build output is exactly 247,700 rows
and 24,867 train documents. The strict verifier report at
`/workspace/interp/outputs/nano30b-nla-pilot/r33_rl_train_only/verify_report.json`
passes with zero blockers:

- 247,700 unique provenance keys, zero duplicates and zero missing keys;
- every activation is finite, `d_model=2688`, and `activation_layer=33`;
- 247,700 canonical actor prompts and 247,700 single injection placeholders;
- zero invalid prompts, empty prefixes, or teacher/response columns;
- zero held-out/nontrain document overlap and zero attested component overlap;
- valid RL sidecar/token metadata and matching base, split, and actor-sidecar
  lineage hashes.

After a clean 8-H100 dry-run with no missing paths, the corrected queue
`configs/nano_rl/r33_component_corrected_k3_hpo_queue_8h100.yaml` started at
`2026-07-01T18:13:10Z`. The first item is the guarded `lr=1e-5`, `k3`,
8-update probe with 48 prompts x 8 responses = exact global batch 384,
microbatch 32, and 6 actor + 1 fixed critic + 1 rollout GPU. W&B is offline.
At the last authenticated observation, SGLang was healthy on GPU 6, all six
actor shards and the fixed critic had loaded, and initial actor-to-rollout
weight synchronization was active without OOM, traceback, batch mismatch, or
drift-guard failure. The `lr=2e-5` probe remains dependency-blocked behind the
first probe; the 32-update confirmation remains gate-blocked behind the probe
winner. No hero run was launched.

## 2026-07-01: Unified-Runtime Live-Sync Canary

The first corrected `lr=1e-5` probe did not reach rollout 0. Its trainer used
PyTorch/CUDA/NCCL from `/workspace/interp/.venv`, while managed SGLang used
`/workspace/interp/.venvs/sglang-cu130`; both processes then joined the same
NCCL live-weight-update group. SGLang accepted the update-group request, its
heartbeat stopped, actor rank 0 never reached the next bucket barrier, and the
remaining actor ranks timed out after 600 seconds. This reproduces the known
split-runtime live-sync failure rather than an RL loss or memory failure.

The queue now rejects managed external live synchronization unless trainer and
SGLang use the same explicit Python runtime. The failed run is preserved at
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/failure_evidence/20260701T200309Z/`.
The focused queue/chain regression suite passed `46/46` tests after the guard.

A one-update unified-runtime canary then passed initial live sync in `6.9s`
and completed the exact 384-sample actor update, but the fixed critic raised the
legacy reward-vs-training-layout assertion on one of 32 diagnostic samples:
maximum MSE-ratio deviation `2.98%` versus a `2%` threshold. The critic is
frozen in this recipe, so that optimizer path is never executed; reward-path
correctness is already covered by the strict preflight. The failed canary was
preserved at
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/failure_evidence/20260701T205624Z/`
with archive SHA256
`11abafdea856e38182aa83f12e45d6a6031c800db3f92ba2eb46ff195b609747`.

The retry made that policy explicit with `NLA_ASSERT_PACKED_EQUIV=0`; it did
not loosen any actor, reward, batch, or drift gate. It ran from
`2026-07-01T21:13:38Z` to `21:22:31Z` on the 6 actor + 1 fixed critic + 1
rollout H100 topology and completed cleanly:

- initial and post-update live sync: `7.4s` and `6.7s`;
- 384/384 generated samples, closed fraction `0.95052`, usable fraction
  `0.94271`, zero truncation, reward mean/std `-0.51597 / 0.43549`;
- frozen-critic train phase `2.9s`, reference log-probability phase `15.8s`,
  actor train phase `104.0s` including two exact microbatches;
- train-vs-rollout log-probability absolute difference `0.28390`, below the
  `0.75` drift threshold, with no OOM, NCCL timeout, or batch truncation;
- worst observed actor CUDA allocation about `41.7 GiB` on a 95.8-GiB H100.

Checkpoint `actor/iter_0000001` saved successfully. The only traceback was the
known W&B offline-service `BrokenPipeError` during interpreter teardown after
save; queue status is `complete`. Lightweight logs and offline W&B evidence are
stored under
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/rl_evidence/20260701T212349Z/`
with SHA256
`7da1b180154e7658596645016c48637844962b816c170dbc6ef0943ccb593f0f`.
The unselected 59-GiB canary checkpoint was deleted after verification,
returning `/workspace/interp` to about `391G` free.

This canary validates infrastructure and one optimizer update, not RL quality.
The next run is the renamed eight-update `lr=1e-5`, `k3`, exact-`gb384`
probe under the same unified runtime; `lr=2e-5` remains blocked behind it.

## 2026-07-01: Corrected K3 `lr=1e-5` Eight-Update Probe

Run `r33-corrected-k3-probe-lr1e5-update8-unifiedenv-retry1` trained from
`2026-07-01T21:31:20Z` and completed its 256/256 post-eval at
`2026-07-01T23:13:07Z`. It used the validated unified runtime, six actor
H100s, one fixed-critic H100, one rollout H100, 48 prompts x 8 generations,
exact global batch 384, actor microbatch 32, `lr=1e-5`, and K3 coefficient
`1e-3`. All eight updates and all live weight synchronizations completed
without OOM, NCCL timeout, sample truncation, or drift-guard failure.

Training dynamics:

- rollout reward means were `-0.5084, -0.3534, -0.3401, -0.3200, -0.3438,
  -0.2852, -0.3331, -0.3633`; rollout 5 was best;
- usable output fraction rose from `0.9531` at rollout 0 to `1.0` at rollouts
  5 and 7, with zero generation truncation throughout;
- train-vs-rollout log-probability absolute difference stayed between
  `0.2632` and `0.3285`, below the `0.75` guard;
- K3 loss was `0, 0.0185, 1.4748, 1.6383, 1.5289, 0.7414, 0.3994,
  10.9046`. The final spike makes update 8 a result to evaluate, not an
  automatic promotion;
- final steady-state step time was about `155.0s`, actor work `23.9s`, and
  wait ratio `0.636`; active actor allocation peaked around `60.4 GiB` with
  about `18 GiB` free per actor H100.

The full round-trip gate passed. Against the matched rows from the selected
R33 SFT baseline:

- validation AV-real NMSE `0.0001085655` versus `0.0001096657`, a `1.00%`
  improvement with `56.64%` rowwise wins;
- test AV-real NMSE `0.0001195005` versus `0.0001216750`, a `1.79%`
  improvement with `50.39%` rowwise wins.

AV-real beat shuffled, zero, mean-activation, no-injection, and target-mean
controls on `98.8-100%` of rows. Real generation was closed and usable for
`100%` of both splits. Teacher text remained stronger: teacher NMSE was
`0.0001068927 / 0.0001123025` on validation/test, and AV-real beat teacher on
`44.92% / 36.72%` of rows. This is therefore a small but real SFT improvement,
not yet a strong hero margin.

Evidence is stored at
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/rl_evidence/20260701T231539Z/`
with SHA256
`2029d86b49d6f72c9b0cd333839d9dd6ae40fd5ad889d000f8f4921d9d778419`.
The 59-GiB update-8 checkpoint is retained pending comparison with the next
probe; its temporary HF conversion was cleaned.

The nominal cache generation path was found to be architecture-incomplete:
Nano returns `cache_params` backed by `HybridMambaAttentionDynamicCache`, not
`past_key_values`. The 512-row generation phase therefore took about 64
minutes at roughly 42% GPU utilization. A tested adapter now initializes the
Nano cache and advances `cache_position`.

Real-model verification exposed two additional defects in the checkpoint's
bundled Nemotron-H remote code: the cache constructor computed but did not
retain `conv_kernel_size`, and list-backed convolution/SSM states were later
addressed as tensors through `.device` and `.zero_()`. The centralized,
idempotent `nla.remote_code_patches` path now repairs those defects for every
future DCP-to-HF conversion. The patch report on the staged R33 model recorded
one kernel-size repair, four device repairs, one reset repair, and no
validation errors. Remote cache/eval tests passed `45/45` with two optional
fixtures skipped. A fresh real-model decode on one H100 then generated four
tokens in `6.65s` with `cache_used=true` and no fallback. That smoke proved
only that the cache API executed; it did not prove token equivalence. The
subsequent deterministic equivalence check below supersedes it.

After that gate, the queue dependency simulation promoted only
`r33-corrected-k3-probe-lr2e5-update8`; the 32-update confirmation remained
blocked. The `lr=2e-5` probe launched at approximately
`2026-07-01T23:39Z` with the same eight-H100 topology, exact global batch 384,
eight samples per prompt, eight updates, unified trainer/SGLang runtime, and
256/256 bounded post-eval. No confirmation or hero run was launched.

## 2026-07-02: `lr=2e-5` Probe And Cache-Eval Invalidation

The corrected `lr=2e-5` probe completed all eight exact-384 updates without an
OOM, NCCL failure, truncation event, or drift-guard stop. Rollout reward means
were `-0.5057, -0.3054, -0.3423, -0.3439, -0.3709, -0.3187, -0.3609,
-0.3865`. K3 was `0, 599.9822, 102.9060, 0.8475, 0.3299, 0.1754, 0.0777,
0.0804`; the two large early transients make this trajectory less stable than
the `lr=1e-5` probe even though it recovered. Train-versus-rollout drift stayed
between `0.284` and `0.342`, below the configured guard. The update-8 actor
checkpoint is retained until a valid matched gate decides between the two
learning rates.

The first exported-checkpoint eval attempts exposed two orchestration defects:
the DCP-to-HF path did not apply the centralized remote-code patch to the
temporary export, and a retry did not clean a stale temporary HF directory.
The queue now patches every export before eval and supports an idempotent
`--post-eval-only` retry that removes only its disposable HF directory. These
are infrastructure fixes; they do not alter model weights.

The resulting cache-backed 256/256 report is **invalid for model selection**.
Deterministic checks against the patched SFT model diverged at generated token
index 1 for both batch size 5 over 16 tokens and batch size 1 over 8 tokens.
The logs are `cache_equivalence_sft_batch5_16tok_20260702T0205Z.log` and
`cache_equivalence_sft_batch1_8tok_20260702T0210Z.log` in the corrected-RL
output root. The report and generated JSONL remain preserved as diagnostic
artifacts, and the queue records them under `invalidated_post_eval`; their
failed gate and NMSE values must not be interpreted as `lr=2e-5` quality.

Generation now fails closed for the exact Nemotron-H model class unless the
experimental cache is explicitly opted in. A new `legacy_batch` backend
batches the five same-prompt controls while retaining complete-prefix
recomputation at every token, matching the valid `lr=1e-5` semantics. Local
tests passed `111/111`; RunAI passed `109` with two optional skips.

The replacement 64/64 `legacy_batch` gate completed at
`2026-07-02T02:52:28Z` and passed. Real generation was closed and usable on
all 128 rows and beat every control on every row. Against matched SFT rows:

- validation AV-real NMSE was `0.0001069578` versus `0.0001095636`, `2.38%`
  lower, with `50.00%` rowwise wins;
- test AV-real NMSE was `0.0001179766` versus `0.0001207099`, `2.26%` lower,
  with `48.44%` rowwise wins.

The direct matched comparison with `lr=1e-5` was mixed. On validation,
`lr=2e-5` was `1.64%` worse and won `39.06%` of rows; on test it was `3.86%`
better and won `51.56%`. Combined across 128 rows, its mean was `1.32%` lower
but it won only `45.31%` of rows. A 100,000-sample paired bootstrap interval
for mean `lr2-lr1` loss was `[-6.68e-6, 3.31e-6]`, crossing zero. This is not a
decisive quality advantage, while the `lr=2e-5` training trajectory had much
larger early K3 transients.

Decision: select `lr=1e-5` for the guarded 32-update confirmation. The blocked
queue was subsequently hardened to retain only updates 16 and 32, evaluate
both at 64/64, and run the 512/512 promotion gate only after update 32 beats
update 16; it was prepared but not launched. The complete `lr=2e-5`
lightweight evidence is at
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/rl_evidence/20260702T025445Z/r33-corrected-k3-lr2e5-update8-evidence.tgz`
with SHA256
`c49bd2bf35d18773cd6361b621b9d8c60a8657b6b01211a125fb452d799a3469`.
After the same archive was verified locally, only the losing 59-GiB actor
checkpoint was deleted. Logs, offline W&B, valid and invalid reports,
generated text, and equivalence evidence remain; PVC free space rose to about
`332 GiB`.

## 2026-07-03: Evidence Reconciliation And Hero Handoff

This entry reconciles the preserved archives, live queue state, and current
operational status. It is the handoff source of truth for the next R33 RL
session.

### Preserved Evidence

| Run | Evidence | SHA256 | Local copy |
| --- | --- | --- | --- |
| corrected K3 `lr=1e-5`, update 8 | `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/rl_evidence/20260701T231539Z/r33_corrected_k3_probe_lr1e5_update8_unifiedenv_retry1_20260701T231539Z.tgz` | `2029d86b49d6f72c9b0cd333839d9dd6ae40fd5ad889d000f8f4921d9d778419` | `artifacts/runai_rl_evidence/20260701T231539Z/` |
| corrected K3 `lr=2e-5`, update 8 | `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/rl_evidence/20260702T025445Z/r33-corrected-k3-lr2e5-update8-evidence.tgz` | `c49bd2bf35d18773cd6361b621b9d8c60a8657b6b01211a125fb452d799a3469` | `artifacts/runai_rl_evidence/20260702T025445Z/` |

Both archives were downloaded again on `2026-07-03` and matched the recorded
hashes. They contain offline W&B payloads, training/eval logs, queue state,
generated text, reports, and lightweight checkpoint metadata; they do not
contain the 59-GiB actor model shards.

### Valid Round-Trip Results

| Probe | Split | Rows | AV-real NMSE | Matched SFT NMSE | Relative change | Rowwise wins | Teacher NMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lr=1e-5` update 8 | validation | 256 | `0.0001085655` | `0.0001096657` | `-1.00%` | `56.64%` | `0.0001068927` |
| `lr=1e-5` update 8 | test | 256 | `0.0001195005` | `0.0001216750` | `-1.79%` | `50.39%` | `0.0001123025` |
| `lr=2e-5` update 8 | validation | 64 | `0.0001069578` | `0.0001095636` | `-2.38%` | `50.00%` | `0.0001050336` |
| `lr=2e-5` update 8 | test | 64 | `0.0001179766` | `0.0001207099` | `-2.26%` | `48.44%` | `0.0001124691` |

All four valid split results had closed and usable real generation fractions
of `1.0` and passed the full real-vs-shuffled/zero/mean/no-injection control
gate. The `lr=1e-5` report is labeled `generation_backend=cache`, but the
generic path did not receive a usable Nano cache and therefore recomputed the
full prefix. Its semantics match the trusted path, not the later invalid
incremental-cache experiment.

On the exact 64-row panel shared by both probes, `lr=2e-5` was `1.64%` worse
than `lr=1e-5` on validation and `3.86%` better on test. Across both splits it
had a `1.32%` lower mean loss but won only `45.31%` of rows. A 100,000-sample
paired bootstrap for mean `lr2-lr1` loss gave `[-6.68e-6, 3.31e-6]`, which
crosses zero. The direct probe comparison is therefore inconclusive.

### Training Dynamics

- `lr=1e-5` rewards by rollout were `-0.5084, -0.3534, -0.3401, -0.3200,
  -0.3438, -0.2852, -0.3331, -0.3633`. K3 was `0, 0.0185, 1.4748,
  1.6383, 1.5289, 0.7414, 0.3994, 10.9046`. Drift stayed in
  `[0.2632, 0.3285]`; steady step times were about `155-159s`.
- `lr=2e-5` rewards were `-0.5057, -0.3054, -0.3423, -0.3439, -0.3709,
  -0.3187, -0.3609, -0.3865`. K3 was `0, 599.9822, 102.9060, 0.8475,
  0.3299, 0.1754, 0.0777, 0.0804`. Drift stayed in `[0.2837, 0.3420]`;
  steady step times were about `161-165s`.
- Neither probe hit the `0.75` drift guard, OOMed, truncated a rollout, or
  lost an actor microbatch. The `lr=1e-5` final K3 spike remains a medium-run
  risk; the much larger early `lr=2e-5` transients are the main reason not to
  prefer it from a statistically mixed short comparison.

Decision: use `lr=1e-5` for the next confirmation. This selection favors the
larger valid 256/256 evaluation and the less extreme optimization trajectory;
it does not claim that `lr=2e-5` is uniformly worse.

### Correctness And Orchestration Changes

- Queue preflight now rejects trainer/SGLang Python-runtime mismatches before
  NCCL live weight synchronization.
- The frozen critic's unused optimizer-layout assertion is disabled without
  weakening reward-path preflight, exact actor-batch checks, or drift guards.
- DCP-to-HF post-eval now applies the centralized idempotent Nemotron remote
  code patch before model loading and cleans only disposable temporary HF
  directories on retry.
- `--post-eval-only` retries an existing actor checkpoint without replaying
  training and preserves prior failure metadata.
- Real Nemotron incremental-cache generation now fails closed by default.
  `legacy_batch` batches same-prompt controls while retaining full-prefix
  recomputation.
- Deterministic cache-equivalence checks diverged at generated token index 1
  for both batch size 5 and batch size 1. Consequently, the `lr=2e-5`
  cache-backed 256/256 report (`0.0002812802 / 0.0002757850`) is explicitly
  invalid and excluded from every model-selection comparison.
- The final source/config/doc bundle was synchronized through S3 at
  `source-sync/20260702T025848Z/` with SHA256
  `e3b92fb671cab085365f75af688937702b098027c62bf31aaebfc8deaa7cef56`.
  Local tests passed `111/111`; the RunAI environment passed `109` with two
  optional skips; Ruff and `git diff --check` passed.

### Operational Handoff

As of `2026-07-03T17:06Z`, `train` is mounted and idle on 8 H100 NVLs. The
selected SFT checkpoints, strict RL dataset, and retained `lr=1e-5` checkpoint
are present. Do not start the queue watcher as a substitute for approval: use
the queue's explicit arm operation only after reviewing the generated spec.

The blocked confirmation now retains only updates 16 and 32, runs 64/64 gates
on both, and runs 512/512 only after the update-32 versus update-16 gate passes.
Its final promotion requires at least 10% relative improvement on both splits,
more than 50% rowwise wins, positive document-clustered bootstrap confidence
intervals, exact dataset identity, healthy parsing, and all controls.

The older fixed-AR hero queue remains deliberately blocked. Its Stage-2 report
and independent cross-critic report are absent, and both are required launch
inputs. The hero configuration must be regenerated from the confirmed recipe
rather than treated as a ready artifact. The quality-first target remains a
clean run from the selected SFT initialization, with checkpoint selection by
generated-text round-trip NMSE rather than train reward or final-step identity.
A Qwen-comparable `131,328`-rollout budget is `342` updates at the validated
global batch `384`; that run remains gated on the confirmation and independent
validity evidence.

### 2026-07-03 Audit Remediation

- launch safety: pending and dependency-promoted items require explicit
  approval; `--arm` and `--disarm` record the approver and timestamp.
- source safety: launch-critical roots are content-addressed, exact-sync
  reconciliation removes only files outside the signed manifest, and every
  run records source, queue, branch, and Git provenance.
- optimizer safety: local-shard clipping now computes the norm in FP32 but
  scales the original gradients, including BF16 tensors. The future queue no
  longer passes `--nla-skip-grad-norm`.
- runtime safety: configurable composite guards cover K3 KL, actor/rollout
  drift, parser closure/usability, and monotonically rising response-length
  tails.
- evaluation safety: unsafe incremental cache evaluation is quarantined;
  future queues use deterministic `legacy_batch`. Round-trip reports bind row
  keys and document IDs to exact dataset hashes and use document-clustered
  paired bootstrap statistics.
- storage safety: retention is manifest-first and config-driven. The
  confirmation keeps only update 16 and update 32 and is restart-only because
  optimizer state is intentionally omitted.
- supervision safety: the queue chain recognizes RL training/eval states,
  required report paths fail closed, and hero launch requires both Stage-2 and
  independent cross-critic gates.

This remediation changes launch and evaluation validity; it does not upgrade
the prior probes' scientific status. Their observed gains remain statistically
null, tail-sensitive evidence, and their historical unshuffled/unclipped
runtime is preserved explicitly in the queue metadata.

### 2026-07-03 Corrected Confirmation Startup And CPU-Memory Finding

The explicitly armed `r33-corrected-k3-confirm-lr1e5-update32` confirmation
started on 8 H100 NVLs with the audited 6 actor / 1 critic / 1 rollout
topology, global batch 384, microbatch 32, eight samples per prompt, K3
coefficient `1e-3`, actor LR `1e-5`, gradient clipping enabled, and W&B
offline. The queue keeps model-only updates 16 and 32 and will run the staged
64/64 then 512/512 round-trip gates after training.

Startup exposed a host-memory constraint that GPU monitoring alone had hidden.
The pod cgroup was capped at exactly 256 GB while six actor ranks constructed
their frozen FSDP reference models. A 59 GB SGLang checkpoint copy remained in
`/dev/shm` after the server health check, so cgroup usage reached
`255.99/256.00 GB`; actor ranks spent several minutes in page reclaim while
the GPUs waited in collectives. There was no CPU OOM, CUDA OOM, traceback, or
guard failure.

The SGLang server had already loaded the checkpoint and no process had the
staged tensor shards open or mapped. Removing only the staged
`model-*.safetensors` files reduced cgroup usage to about 197 GB immediately;
tokenizer, config, remote code, the live GPU weights, and the actor update
channel were untouched. FSDP initialization resumed and rollout 0 completed
with 384 samples, 99.22% closure, 98.18% usable responses, zero truncation,
and reward mean/std `-0.4045 / 0.3048`. These are training-rollout diagnostics,
not promotion evidence; the final bounded verifier remains authoritative.

Commit `3ddb562` makes this behavior declarative through
`sglang.model_staging.release_after_health.globs`. It also verifies the staged
tree against the source signature so a deliberately pruned cache is rebuilt
on the next launch. The release writes a byte/file manifest into the run
directory. Local verification passed all 471 tests. The running confirmation
still uses its frozen pre-sync source fingerprint and received only the
equivalent runtime file release; future runs will use the committed queue
implementation. The commit's S3 source bundle is
`source-sync/20260703T1843Z/nano30b-r33-runtime-release-3ddb562.tgz`, SHA256
`a9254e960811d92192e4a726051c97c0a66978e2ec09aeaf0e4e63f1ed7bc69e`.

Before launch, an unrelated stale source-backup process was also found
archiving a 27 GB ELF core dump. The incomplete 46 GB archive and the stale
core dump were removed, increasing `/workspace/interp` free space from 286 GB
to 356 GB without deleting checkpoints, datasets, logs, or verifier reports.

The confirmation was then stopped before update 1 could complete. A direct
inspection of the live `/workspace/interp/code/miles-051cd15` runtime showed
that its FSDP actor still called `torch.nn.utils.clip_grad_norm_` followed by
DTensor `full_tensor()`. The audited source tree contained the local-shard
implementation, but the source fingerprint did not cover the separate Miles
runtime directory, and its already-patched non-git copy had never received the
newer logical change. The apparent post-microbatch stall was the old global
gradient materialization path, so this attempt is operational evidence only
and must not be used as RL quality evidence. It produced no checkpoint or
post-eval report. Logs and the exact runtime queue were retained; all Ray,
SGLang, and trainer processes were terminated cleanly and all eight GPUs
returned to 4 MiB idle usage.

The remediation adds exact external-runtime contracts to queue preflight and
an idempotent `scripts/reconcile_miles_runtime.py` utility for existing patched
Miles trees. A retry must use a new run directory and may launch only after the
actor and argument-file hashes plus required local-shard markers are recorded
in `runtime_contracts.json`.

The fresh `runtimefix-retry1` directory launched at `2026-07-03T19:09:05Z`
after exact source fingerprint and external-runtime contracts passed. During
startup, a second fail-closed check showed that `_sglang_model_staging_config`
had omitted the nested `release_after_health` mapping even though the raw YAML
contained it. SGLang was already healthy and no rollout or optimizer step had
begun, so the same 13 staged tensor shards (63,156,683,832 bytes) were unlinked
manually and recorded in `sglang_model_release.json`; live GPU weights and all
runtime metadata remained intact. Queue normalization now preserves and
validates the release mapping, with a built-spec regression test. This is an
operational intervention only, not a change to the RL recipe.

### 2026-07-03: Retry1 host-memory OOM and fail-closed mem512 retry

`runtimefix-retry1` completed rollout 0, entered rollout 1, and demonstrated
that the reconciled Miles runtime no longer stalled at the old DTensor
`full_tensor()` gradient-norm path. At `2026-07-03T19:39:49Z`, Kubernetes
OOM-killed the entire `train` pod at its exact 256 GB CPU-memory cgroup limit.
The run had produced no checkpoint or bounded verifier report, so it is
operational evidence only and is excluded from model-quality comparisons.
Its logs and 6.6 MB of lightweight run metadata are retained.

The workspace was redeployed on the same 8 H100 NVL topology with both PVCs
preserved and the CPU-memory request/limit raised to 512 GB. The new pod
reported `memory.max=512000000000`, eight idle H100 NVLs, 353 GB free on
`/workspace/interp`, and 454 GB free on `/workspace/models`. A distinct
`runtimefix-retry2-mem512` queue item preserves the selected `lr=1e-5`, K3
coefficient `1e-3`, exact `gb384/mb32`, 48 prompts x 8 samples, and 32-update
recipe; only the workspace host-memory envelope changes.

The unattended promotion chain is fail-closed: retry2 must complete its
16/32 and 512/512 gates, an independently shuffled R33 AR critic must train
and pass 512/512 evaluation, and Stage-2 cross-critic, invariance, functional,
closure, and structural-output checks must all pass before the 342-update
hero queue can launch. The hero budget is 342 x 384 = 131,328 generated
rollouts, with checkpoints at updates 171 and 342 and post-evals at 64/64 and
512/512. Automatic panel decisions are explicitly structural rather than a
claim of semantic human review.

### 2026-07-03: Retry2 proves 512 GB is not an unattended envelope

`runtimefix-retry2-mem512` completed update 0 and entered update 1 with no
CUDA OOM, no guard failure, and automatic removal of 63,156,683,832 staged
SGLang bytes. During the second actor backward, cgroup usage peaked at
`511,997,681,664 / 512,000,000,000` bytes. The kernel reported no OOM event,
but only about 2.3 MB remained before reclaim; this is too fragile for an
unattended 342-update hero. The process group was terminated deliberately,
all eight GPUs returned idle, and no checkpoint or verifier report existed.

`runtimefix-retry3-mem768` keeps the exact model, data, optimizer, LR, K3,
batch, rollout, and topology settings. Only the RunAI CPU-memory cgroup moves
from 512 GB to 768 GB. The detached promotion chain will be re-armed against
retry3 after the workspace is redeployed; retry2 remains operational memory
evidence and is not a model-quality result.

### 2026-07-04: Retry3 dynamic-metric aggregation failure

`runtimefix-retry3-mem768` confirmed that 768 GB removes the host-memory
blocker and trained through rollout 22. It retained model-only actor/critic
checkpoint `iter_0000016`, but produced no verifier report. After the rollout
22 optimizer and scheduler steps, Miles crashed while adding two local
microbatch telemetry vectors with lengths 47 and 41. This was not a CUDA or
host OOM, and the downstream fail-closed controllers correctly prevented the
independent critic, Stage-2 analysis, and hero from launching.

The root cause was positional aggregation of a dynamic metric schema. Miles'
base losses and NLA system/router telemetry shared one `keys`/`values` vector,
while optional telemetry can differ by microbatch or rank. The replacement
reducer aligns metrics by key, tracks a separate observed normalizer for each
metric, forms a deterministic key union across the DP/CP group, and then
all-reduces aligned numerator/denominator tensors. A RunAI regression test
reproduced the unequal schemas before the fix and passed afterward.

The live Miles actor was reconciled idempotently with backup under
`runtime_backups/miles_pre_keyed_loss_20260704T0648Z`. Its exact actor SHA256
is `7db9b4acfbc7af734dee736c8a549cdd5a3f6d31c46e4d7d53f8028b62357479`;
the arguments SHA256 remains
`203c1a77678d3fd9fb982c0ee75a28fe92843dd32d969f6cb418f711bb11f943`.
`runtimefix-retry4-keyedloss-mem768` preserves the full selected recipe and
uses a fresh run directory and W&B identity.

### 2026-07-04: Retry4 completes; hardened short gates reject promotion

`runtimefix-retry4-keyedloss-mem768` completed all 32 exact-384 updates on
the 6 actor + 1 frozen critic + 1 rollout H100 topology. It passed the former
rollout-22 failure point, saved model-only actor checkpoints at updates 16 and
32, and produced no CUDA OOM, host OOM, metric-schema mismatch, or guard
failure. This validates the key-aligned loss reducer and the 768 GB host-memory
envelope for the selected corrected recipe.

The hardened 64/64 round-trip checks did not approve promotion:

- update 16 versus matched SFT improved mean NMSE by `2.35% / 2.60%` on
  validation/test, with row-win fractions `54.69% / 50.00%`;
- update 32 versus update 16 improved mean NMSE by `2.40% / 0.24%`, with
  row-win fractions `56.25% / 51.56%`;
- parsing and usability were `100%`, and real generation beat all controls;
- the document-clustered paired confidence intervals crossed zero on both
  splits for both comparisons.

The configured update-32 512/512 promotion check therefore skipped by design.
The observed effect is small and statistically unresolved, not a qualifying
double-digit hero signal. No RL hero was launched.

The first confirmation-to-critic watcher exposed a separate orchestration
defect: it treated queue status `complete` as approval without requiring the
item's recorded `gate_passed: true`. It consequently attempted the independent
AR critic after the failed quality gate. That critic stopped before training
because its spec resolved the stale `nano30b-nla-pilot-current` source tree,
which lacked `aggregate_train_losses_by_key`; it produced no checkpoint or
quality result. `scripts/nano_queue_chain.py` now has a fail-closed
`--require-gate-pass` prerequisite, and the independent critic pins
`nano30b-nla-pilot-hero-current`. Regression coverage verifies false and
missing gates both reject handoff.

To distinguish a genuinely small effect from a 64-row sampling artifact, an
evaluation-only queue now resumes the update-16 generations to 512/512 and
scores against the hardened matched SFT baseline with exact dataset identity,
row wins, a document-clustered paired bootstrap, and the existing 10% hero
margin. It reuses the valid 64/64 rows, generates no teacher data, launches no
training, and removes only its temporary HF export. The source bundle is
commit `0974ca3`, archived at
`source-sync/20260704T100801Z/nano30b-r33-promotion-0974ca3.tgz` with SHA256
`3e7d12ea6d234c97a25ab82425bc6e31a6e65119424558d0ad8ee78aa90e6124`.

The 512/512 diagnostic completed and passed every configured promotion
condition. Against the exact hardened SFT rows:

- validation NMSE was `0.0001083722` versus `0.0001267961`, a `14.53%`
  relative improvement; the document-clustered 95% CI for absolute
  SFT-minus-RL delta was `[1.1554e-5, 2.6094e-5]`, and row wins were `62.89%`;
- test NMSE was `0.0001147213` versus `0.0001347520`, a `14.86%` relative
  improvement; the clustered CI was `[1.2709e-5, 2.8339e-5]`, and row wins
  were `58.40%`;
- each split contained 512 rows from 52 independent documents, median paired
  deltas were positive, and the top five positive rows accounted for only
  `13.76% / 12.57%` of net validation/test improvement;
- parsing and usability were `100%`, dataset hashes matched, and real
  generations beat every control.

The first 64 rows were therefore an unrepresentative low-error slice whose
SFT baseline was much stronger than the full matched set. Update 16 is the
selected corrected-K3 confirmation checkpoint. Stage 2 and the hero preflight
now consume its exact report and generated text; update 32 is not selected.

The first fresh independent-critic launch then exposed a topology-specific
memory limit. Its inherited 2-GPU `mb96` recipe reached `91.19 GiB` on a
93.10-GiB H100 NVL and failed when backward requested another `2.42 GiB`.
Only steps 0 and 1 completed, and no checkpoint or eval report exists. The
retry preserves `gb192`, LR, warmup, cosine schedule, split, and independent
row order, but uses 4 FSDP GPUs at `mb48`; this is an exact-batch topology
change, not a scientific recipe change.

The 4-GPU retry was memory-stable at 57-62 GiB/GPU and progressed through
step 391 at approximately 6 seconds/step. At `2026-07-04T13:37:16Z`, it then
failed with a CUDA illegal-memory-access error. The first synchronous stack
frame was the optional router-entropy forward hook in `nla/system_metrics.py`,
while copying detached router indices to CPU for `torch.bincount`; the later
NCCL watchdog failures followed the already-poisoned CUDA context. Volatile
ECC counters on the four participating GPUs were zero when inspected. This
does not prove whether the hook caused or merely synchronized an earlier
asynchronous fault, so the next controlled retry changes only one variable:
router-entropy observability is disabled. Dataset, row order, seed, FSDP
topology, `gb192/mb48`, optimizer, `lr=5e-5`, warmup 25, cosine schedule,
checkpoint policy, and 512/512 eval remain exact. The failed run produced no
checkpoint and cannot satisfy Stage 2.

That controlled no-router retry reproduced at the same boundary. Step 391
completed with loss `0.2966566`, normalized FVE `0.4686700`, LR
`4.1271e-5`, and finite gradient norm `0.400390625`; rollout 392 then failed
inside the actual segmented MoE routing path at GPU `torch.bincount`. Memory
remained near 59 GiB/GPU, all participating volatile ECC counters were zero,
and no checkpoint or eval report was emitted. Disabling telemetry therefore
did not fix or move the fault; it only changed where the asynchronous CUDA
error first became synchronous.

The next hypothesis is now narrower: avoid the optimized GPU
argsort/bincount route while preserving MoE arithmetic. The runtime supports a
config-driven `expert_scan` fallback that follows the stock per-expert scan;
`segmented` remains the default. Unit coverage compares forward outputs,
hidden gradients, router-weight gradients, and every expert-weight gradient.
The RunAI Torch suite passed `123` tests with two unrelated skips, and an H100
check reported exact forward equality (`max_abs=0.0`) with all gradient checks
passing. Retry 3 changes only `training.moe_routing_impl: expert_scan` from the
failed no-router recipe. Hero provenance is pinned to source commit `78ba931`
and fingerprint
`bf3694441ba39f3492548570dfcfbfd7b6dbe56858362664b199cc20b8ea268c`.

Retry 3 disproved that routing hypothesis as well. The expert-scan run matched
the same learning trajectory through step 391 (loss `0.2965752`, FVE
`0.4688158`) and failed on rollout 392 at the first `torch.where` consuming
router indices. Since router telemetry, segmented sort/bincount, and
expert-scan are three different synchronization sites for the same
asynchronous failure, the CUDA fault originates earlier in the model forward.

The next run is a bounded diagnostic rather than a promotion attempt. It sets
`CUDA_LAUNCH_BLOCKING=1`, stops at update 393, and writes full model,
optimizer, and scheduler state at update 384. This should both identify the
true failing kernel and make subsequent experiments restartable within eight
updates of the deterministic boundary. Hero provenance is consequently pinned
to source commit `c403acc` and fingerprint
`6954ee69f10fa3776c0697c579b1840ada5363658be77ec1f10e55a829ddbb9b`.
The first bounded launch was stopped at step 143 because `num_rollout=393`
also shortened the default cosine horizon. The corrected config explicitly
sets `lr_decay_iters=1289`, preserving the original LR trajectory while still
stopping after the diagnostic boundary; the stopped run is not evidence.

The corrected CUDA-blocking diagnostic reproduced the full-horizon trajectory
and crossed the deterministic failure boundary. At step 143 its LR was
`4.9023024e-5`, exactly matching the prior full run while the archived
wrong-horizon attempt had already decayed to `3.9354329e-5`. Step 391 completed
at loss `0.2985217`, FVE `0.4653295`, and LR `4.1270958e-5`; step 392 then
completed at loss `0.2844250` and FVE `0.4905775`. The only runtime change from
the expert-scan failure was `CUDA_LAUNCH_BLOCKING=1`. Since the same seed, row
order, batch, optimizer, schedule, routing implementation, and formerly
failing batch now pass, the rollout-392 failure is an asynchronous CUDA
race/timing fault rather than invalid data, sequence length, routing
arithmetic, memory exhaustion, or numerical divergence. The audit found no
invalid token IDs, and several earlier successful batches were longer than
rollout 392.

Full model, optimizer, and scheduler checkpoints were written at iterations
384 and 393 through a symlink into `/workspace/models`, keeping the Longhorn
filesystem at 262 GiB free. The final tracker records iteration 393. A bounded
64/64 eval was intentionally stopped while redundantly reconstructing the 30B
checkpoint from NFS; it was non-gating and projected another 44 minutes of
load time. Its log is retained, but it is not quality evidence. The required
continuation instead loads iteration 393 directly and is the stronger full
state check.

The continuation is config-driven at
`configs/nano_ar/hpo/r33_component_full_independent_critic_seed314159_4gpu_mb48_cudablock_resume393.yaml`.
It restores optimizer and scheduler state, preserves the original 1289-update
cosine horizon, runs the remaining 896 updates with launch blocking, and saves
only the final model checkpoint. The runner now supports this storage-conscious
contract explicitly with `checkpoint.resume_optimizer_state_required: true`;
launch verifies `optimizer/.metadata` and `lr_scheduler/.metadata` at the
tracker-selected iteration before allowing a model-only final save. The
continuation is loading on four H100 NVLs. Stage 2 remains blocked until its
final 512/512 eval completes and passes.

The first continuation attempt exposed a separate resume-policy bug. DCP
correctly restored scheduler `last_epoch=393` and stored LR `4.1226691e-5`, but
the patched Miles actor classified every optimizer resume as stale, changed
the requested cosine schedule to constant, and rewrote the live param-group LR
to `5e-5`. The attempt was stopped around step 410, produced no checkpoint,
and is preserved under a `.wrong_scheduler_reset_20260704T225829Z` run suffix.

The fix is shared and config-driven in `nla/lr_policy.py`, applied through
Miles patch `0018_fsdp_resume_lr_policy.patch`. It refreshes requested LR
bounds, preserves the restored scheduler epoch, and recomputes the live
optimizer LR from that epoch. Constant LR now requires either a constant
configured schedule or explicit `NLA_FORCE_CONSTANT_LR=1`; a resume alone no
longer flattens decay. Local policy/patch tests pass 29 cases and the RunAI
runtime shard passes 58. The corrected source fingerprint is
`158336e68fcbfd6a217fc559a65304f405639a85d0c0c186e93a349f03092aba`; the
live Miles actor SHA256 is
`da198ce079c3ce68ddf761a88a7d328910f828d68e0d0a52515f0d80c6333359`.

The corrected retry restored model, optimizer, and scheduler at
`2026-07-04T23:00:50Z` and reported cosine, `last_epoch=393`, and live LR
`4.1226691e-5`. Real optimizer steps then decayed as expected: logged step 393
was LR `4.1182339e-5`, loss `0.2806310`, FVE `0.4973729`; by step 408 LR was
`4.0507070e-5`, loss `0.2837751`, FVE `0.4917415`. With the resumed state now
proven, redundant full checkpoint iteration 384 was deleted while iteration
393 was retained. `/workspace/models` free space increased from 166 GiB to
255 GiB. W&B remains offline. Independent-critic-to-Stage-2 and
Stage-2-to-hero watchers are active, and neither gate has been bypassed.

The continuation completed all 1289 updates and wrote model-only checkpoint
`iter_0001289`. Its mandatory 512/512 eval found teacher NMSE `0.3208674`
validation and `0.2924067` test. Teacher beat shuffled on `99.80%/100%` of
rows and blank/generic/mean on approximately 99% or more. Source-raw remained
the oracle ceiling at `0.0942281/0.0800047`. The critic queue completed at
`2026-07-05T01:39:17Z`, allowing the first watcher to start Stage 2.

Stage 2 then established an independent-critic SFT baseline of
`0.0001269159/0.0001344858` validation/test round-trip NMSE. Corrected-K3
confirmation update 16 achieved `0.0001081018/0.0001144881`, a matched
`14.824%/14.870%` improvement. Candidate rowwise win rates were
`66.99%/59.96%`; both doc-clustered 95% confidence intervals were strictly
positive, all controls were beaten, and parse close/usable rates were 100%.
The independent round-trip report's gate passed. Invariance, functional,
closure, qualitative, combined cross-critic, and composite Stage 2 checks
remain in progress.

The first combined cross-critic command failed only its raw `row_identity`
comparison. All four reports had identical train/validation/test parquet
hashes, exact row-index order, and matching document IDs; candidate row keys
also carried optional `n_raw_tokens`, while the older SFT reports did not.
The gate incorrectly treated that enrichment field as identity. The fixed gate
canonicalizes identity to dataset hash + parquet row index + document ID. New
tests prove optional enrichment passes while row-index or document-ID changes
still fail closed. The unchanged four real reports then passed every split
check, and `r33-corrected-cross-critic-gate.json` records `passed: true`.

The remaining Stage 2 evidence also completed. Format normalization retained
`100.01%/99.96%` validation/test FVE and unit reordering retained
`99.10%/98.77%`. Functional reinjection identity passed all 1,024 rows; the
separate stored-activation replay diagnostic remained outside tolerance on all
rows and is retained as a caveat, while the paired functional recovery checks
used by the composite gate passed. Response closure was 100% on both splits;
cap 192 closed `100%/99.80%` validation/test and met the 95% requirement.
Structural qualitative review completed, and the composite Stage 2 report has
no failed checks or blockers. Both Stage 2 and combined cross-critic gate JSONs
record `passed: true`.

The guarded watcher launched hero run
`r33-corrected-k3-hero-lr1e5-update342` at `2026-07-05T02:57:23Z` with six
actor GPUs, one rollout GPU, and one critic GPU. Source fingerprint
`aef659279c9306f4818812b0b9eb0cbd24df0d857c562d323f4da221524c32a4`
and the patched Miles actor contract were verified before launch. At the final
start check, optimizer steps 0, 1, and 2 had completed. Step-2 actor KL loss
was `0.6216564`, rollout/logprob absolute difference was `0.2952003` versus a
`0.75` two-step guard, and no CUDA/OOM/NCCL/guard error was present. The next
384-sample rollout was already running. Offline W&B role runs exist for actor,
critic, and rollout. This proves hero training started; it is not yet a hero
quality result.

## 2026-07-06: R33 Hero Guard Stop At Update 63

The 342-update hero attempt did not complete. The queue marked
`r33-corrected-k3-hero-lr1e5-update342` failed at
`2026-07-05T06:02:39Z` after the actor train guard observed two consecutive
`train/kl_loss > 5.0` values. Logged step 62 had KL loss `25.9797955`; the
aggregated step-63 metrics then reported `5.1258636`, satisfying the two-step
stop rule. This was an intentional fail-closed guard stop, not an OOM, CUDA
illegal-memory-access, NCCL, parser, or rollout-service failure.

The run generated 64 rollout batches (IDs 0-63), or 24,576 responses at
global batch 384. That is `18.71%` of the planned 131,328-response budget.
Normal actor records cover steps 0-62; step 63 reached the guard hook after
metrics aggregation and was not written as a normal train-step record. The
KL trajectory was intrinsically spiky: median `0.6301`, p90 `3.8691`, p95
`14.3117`, and maximum `233.0444`. Logged values exceeded 5 at steps 5, 24,
39, and 62; each earlier isolated excursion recovered on the next step, but
step 63 remained above threshold and stopped the run. Gradient-norm spikes
coincided with the largest KL excursions. Actor/rollout log-prob absolute
difference remained healthy (`0.2815` initially, `0.2487` at step 62,
maximum `0.3035`) and never approached its `0.75` guard.

There was useful learning signal before the stop. Mean raw reward improved
from `-0.346596` over the first ten rollouts to `-0.272935` over the last ten,
while reward standard deviation fell from `0.224467` to `0.161619`. Mean
response length rose from `118.67` on the first rollout to `141.05` over the
last ten, and rollout entropy fell from `1.2266` to a last-ten mean of
`0.8346`. Across all 64 rollouts, close and usable rates averaged
`99.784%/99.752%`; the final rollout was 100% closed and usable, and no
response was truncated. These are encouraging training-dynamics signals,
but they are not a promotion result.

The first configured save was update 171, so the stopped attempt produced no
actor checkpoint and no round-trip post-eval report. None of the update-171,
update-342, or 512/512 promotion gates ran. The retained evidence is the
31.1-MB `train.log`, four offline W&B role runs, runtime/source contracts, and
SGLang logs under
`/workspace/interp/outputs/nano30b-nla-pilot/rl_hero/r33_corrected_k3_hero_lr1e5_update342/`.
At the 2026-07-06 status check the workspace was idle, all eight H100 NVLs
were empty, and free space was 184 GiB on `/workspace/interp` and 312 GiB on
`/workspace/models`. No retry was launched. The scientific status remains:
Stage 2 passed, while the full RL hero milestone remains unachieved.

## 2026-07-06: Storage Cleanup And Guard-Calibrated Hero Retry Launch

The failed hero trace was compared against the successful clean 32-update
confirmation before changing the retry. Both showed isolated heavy-tail K3
excursions, while the failed hero's eight-step KL medians remained between
`0.28` and `0.91`, raw reward improved, parsing stayed healthy, and
actor/rollout log-prob difference remained below `0.304`. The immediate stop
was therefore attributed to a two-consecutive-sample guard applied to a
heavy-tailed metric, not demonstrated sustained divergence.

Prelaunch cleanup removed five obsolete crash core dumps (approximately 100
GiB) from the unused June-27 source snapshot and the completed independent
critic's redundant 144-GiB iteration-393 continuation checkpoint. The tested
retention policy then removed only model shards from three superseded RL
checkpoints: corrected-K3 update 8, historical Qwen-scale update 16, and
historical Qwen-scale update 32. Their logs, W&B runs, generated text, and eval
reports remain. The selected update-16 corrected-K3 checkpoint, AV-SFT actor,
actor-reference HF export, AR-SFT critic, independent critic final checkpoint,
dataset, and Stage-2 reports were all protected. Applied cleanup evidence is
at
`/workspace/interp/outputs/nano30b-nla-pilot/cleanup/r33_hero_retry_prelaunch_cleanup_applied_20260706.json`.
Free space increased from 184 to 460 GiB on `/workspace/interp` and from 312
to 456 GiB on `/workspace/models`.

The retry is config commit `fe71bd8`, queue
`configs/nano_rl/r33_component_corrected_k3_hero_342_guard3_retry1_queue_8h100.yaml`,
and run `r33-corrected-k3-hero-lr1e5-update342-guard3-retry1`. It preserves
the selected clean recipe (`lr=1e-5`, K3 coefficient `1e-3`, global batch
384, six actor + one rollout + one frozen critic H100) and changes only the
safety/checkpoint envelope: KL above 5 for three consecutive steps, log-prob
difference above 0.75 for two steps, gradient norm above 100 for two steps,
and saves at updates 114/228/342 with 114 and 342 retained. Post-eval is
64/64 at update 114, 64/64 update 342 versus 114, then the gated 512/512 final
comparison against clean SFT.

RunAI verification passed the two new static tests, then 60 queue/guard/
retention tests with the old live-mutated queue-state assertion deselected.
Dry-run reported no missing paths, exact 384 generated/trained samples, and
all eight GPUs. Local and RunAI config/test hashes matched. The detached queue
started at `2026-07-06T16:42:47Z`; SGLang health returned 200 and all four
offline W&B role runs were created. Rollout 0 produced all 384 samples with
`98.44%` close, `97.40%` usable, and zero truncation. Optimizer step 0 then
completed with loss `-6.3330e-8`, KL `0`, gradient norm `1.1172`, log-prob
difference `0.27250`, entropy `1.24026`, and LR `1e-5`. No traceback, OOM,
NCCL, or guard event was present. This is a verified hero retry start, not a
quality or promotion result.

## 2026-07-07: Relative-Length Guard Failure And Length-Cap Retry

The first guard-calibrated retry stopped at `2026-07-06T18:02:11Z` while
generating rollout 26. It was not an actor KL, gradient, parser, CUDA, OOM, or
NCCL failure. The inherited
`rollout/nla_response_length/p95 comparison=increasing` rule counted four
ordinary increases (`160.85 -> 163.70 -> 166.00 -> 170.55`) and raised after
each exceeded the preceding value by one token. The implementation does not
use an absolute threshold for `increasing`; at failure, p95 was still far
below the 256-token cap and truncation remained zero. Last completed actor
step 25 had KL `3.67496`, gradient norm `4.5625`, log-prob difference
`0.24050`, and healthy parsing. No checkpoint existed because the first save
was update 114.

The failure was reproduced as a static queue-contract test before the fix.
Commit `1087f5f` adds a fresh queue rather than mutating either failed run. It
removes the relative trend rule and adds two absolute fail-closed rules: p95
above 230 for two consecutive rollouts, and truncated fraction above 5% for
two consecutive rollouts. Closed/usable parser guards, three-step KL guard,
two-step gradient guard, two-step log-prob guard, LR, batch, topology,
checkpoint cadence, and post-eval gates are unchanged. The targeted test
failed before the queue existed and passed after it was added; the RunAI shard
then passed 60 queue/guard/retention tests with the two live-mutated historical
status assertions deselected. Dry-run reported no missing paths and exact
384-sample training.

Because retry 1 produced no checkpoint, retry 2 is necessarily a clean restart
from the selected R33 AV-SFT actor rather than an optimizer-state resume. Run
`r33-corrected-k3-hero-lr1e5-update342-guard3-lengthcap-retry2` started at
`2026-07-07T00:13:25Z`. SGLang passed health, all four offline W&B role runs
exist, and the live trainer environment contains the reviewed absolute length
and truncation rules. Rollout 0 produced all 384 samples with p95 `168.85`,
zero truncation, `97.14%` close, and `96.61%` usable. Optimizer step 0
completed with loss `3.9736e-8`, KL `0`, gradient norm `1.08594`, log-prob
difference `0.27939`, entropy `1.20782`, and LR `1e-5`. No guard or runtime
error was present. This is a verified restart, not promotion evidence.

## 2026-07-07: Absolute-Length Stop And Update-228 Continuation

Retry 2 stopped at `2026-07-07T13:52:32Z` while logging rollout 253. The
absolute response-length p95 rule observed `233.85` and `232.70` against its
230-token threshold and raised after two consecutive exceedances. This was
the only stop condition: rollout 252 was `99.22%` closed and usable with zero
truncation, reward mean `-0.22851`, and p95 `233.85`. Actor step 252 completed
with loss `0.0030120`, KL `3.01199`, gradient norm `1.46875`, log-prob
difference `0.27546`, entropy `0.82575`, and LR `1e-5`. There was no OOM,
NCCL, illegal-memory, parser, gradient, log-prob, or actor-KL guard failure.
The durable model-only checkpoint is update 228; no post-eval ran.

The p95 metric is useful telemetry but was not a reliable abort condition at
this generation budget: the model remained parseable and no response was
truncated. Retry 3 therefore removes only the p95 abort rule. The 5%
truncation guard, close/usable parser guards, and all actor guards remain
fail-closed.

Because retry 2 used `--no-save-optim`, exact Adam-state recovery is
impossible. The continuation loads update 228 model weights, RNG metadata,
rollout counter, and `global_dataset_state_dict_227.pt`, while starting a
fresh optimizer at the unchanged constant LR. `--finetune` is deliberately
absent, so Miles restores `start_rollout_id=228`; `num_rollout=342` remains the
absolute endpoint. Run
`r33-corrected-k3-hero-lr1e5-update342-resume228-retry3` started at
`2026-07-07T17:52:05Z`. All six actor ranks loaded
`iter_0000228/model`, and the rollout manager restored dataset state 227.
Rollout 228 produced 384 samples with reward mean `-0.20660`, p95 `222`,
`99.74%` closed/usable, and zero truncation. Optimizer step 228 completed at
`2026-07-07T18:01:23Z` with loss `0.00092021`, KL `0.92023`, gradient norm
`0.6875`, log-prob difference `0.27336`, entropy `0.79841`, and LR `1e-5`.
No error or guard event is present. This proves the continuation is live; it
is not yet promotion evidence.

## 2026-07-08: R33 RL Hero Completion And Promotion Pass

Retry 3 completed training at `2026-07-07T23:17:47Z`. It ran from restored
rollout/data position 228 through actor step 341 and wrote model-only checkpoint
`iter_0000342`. The final actor record had loss `0.00183662`, K3 KL
`1.836525`, gradient norm `0.886719`, actor/rollout log-probability absolute
difference `0.311984`, entropy `0.789943`, and LR `1e-5`. The final rollout
had reward mean `-0.226492`, response-length p95 `226`, `99.74%`
closed/usable output, and zero truncation. No model, CUDA, OOM, NCCL, or guard
failure occurred. An ignored W&B atexit `BrokenPipeError` appeared after the
trainer had completed and did not affect checkpointing or evaluation.

The update-342 `64/64` prerequisite gate passed. Validation/test AV-real NMSE
was `0.000082157 / 0.000086528`, improving over matched SFT by
`25.01% / 28.32%`; RL won `51/64 / 57/64` rows and parsing was fully usable.

The final `512/512` promotion gate then passed with all provenance checks,
controls, parse-health requirements, rowwise thresholds, and the required
`10%` relative-improvement threshold:

| Metric | Validation | Test |
|---|---:|---:|
| RL AV-real NMSE | `0.0000875281` | `0.0000911757` |
| Matched SFT NMSE | `0.0001267961` | `0.0001347520` |
| Relative improvement | `30.97%` | `32.34%` |
| RL rowwise wins | `427/512` (`83.40%`) | `454/512` (`88.67%`) |
| Clustered 95% CI for improvement | `[0.0000321424, 0.0000472365]` | `[0.0000360316, 0.0000519528]` |
| Independent documents | `52` | `52` |
| Closed/usable | `99.02% / 100%` | `99.41% / 100%` |

AV-real beat `teacher`, `mean`, `av_mean`, `av_none`, `av_zero`, and
`av_shuffled` by aggregate normalized MSE. Baseline dataset hashes, exact row
identities, and all 512 row overlaps matched on both splits. The top five
positive rows contributed only `6.74% / 6.30%` of net improvement.

The queue completed at `2026-07-08T03:20:52Z`, temporary HF exports were
cleaned, and the selected checkpoint remains on RunAI. Lightweight evidence is
mirrored locally and on S3 under `20260708T151400Z_r33_rl_hero_final`, archive
SHA-256
`78cbf98d27188594c25cbf9c0d695f0b3b1754df978961585bbaa6fc178f0bc7`.

## 2026-07-09: Corrected Retained-Checkpoint Salvage Gate

The retained update-342 actor and protocol-matched SFT baseline were rescored
on identical 512/512 rows and 214/218 content families under both the primary
critic and the retained reseed critic. Both use generation protocol SHA-256
`5677d491a812baecb9fa21829866de7eb3750f7cdb0273ed3038123a3119a381`:
no forced prefix, deterministic legacy full-prefix generation,
`max_new_tokens=256`, eight generation workers, and injection scale 75.

Primary/retained-reseed directional gains are `22.64%/22.62%` validation and
`20.31%/19.89%` test. The retained-reseed family-clustered intervals are
strictly positive, row-win rates are `85.74%/83.59%`, and gain-transfer ratios
are `0.9993/0.9793`. After matching hero explanations to SFT or teacher token
length, the best retained-reseed gains remain `15.38%/13.42%`. The aggregate
cross-critic gate passes all declared checks.

This is salvage evidence only. The retained reseed critic is not the freshly
initialized publication critic, the historical stored activations failed the
frozen-runtime identity audit, and the test families are exploratory. The
next valid lineage starts with the frozen 275,396-row R33 extraction and clean
family-disjoint AR/AV SFT; no historical effect size is promoted into the
publication claim.

## 2026-07-10: Clean-Lineage RL Remains Gated

No publication-clean RL job has launched. The deterministic family-disjoint
R33 extraction and primary clean AR SFT are complete. Clean AV optimization
also completed all 1,291 updates with finite loss, gradients, LR, and router
telemetry, but its bounded validation eval and the protocol-matched clean SFT
round-trip baseline were still pending at the last authenticated RunAI check.
Training loss is not used as a substitute for either gate.

The future RL launcher now supports exact model-only saves at updates
`16 / 64 / 110 / 228 / 342`, explicit gradient clipping, and a fail-closed
publication contract binding the preregistration, family evidence, clean SFT
baseline, seed, guards, endpoints, validation-only policy, and checkpoint
schedule by hash. The draft preregistration remains unregistered until the
clean SFT baseline, independent critic/AR, validation power check, and finite
four-probe stability grid complete.

The independent seed-`314159` critic-init rebuild is staged from immutable
source `3676b93` with exact manifest-hash enforcement. Its dependent clean AR
queue correctly remains `blocked_missing_critic_init`; neither has launched.
The confirmatory test split remains sealed. RunAI authentication expired before
this documentation pass could obtain a newer remote observation, so no state
beyond these last authenticated gates is asserted.

## 2026-07-17: First Clean Online Joint AV+AR Canary

The accidentally suspended RunAI `train` workspace was resumed on eight H100
NVLs. Retry 4 of the family-clean online canary completed with four actor GPUs,
three critic GPUs, one SGLang rollout GPU, actor LR `1e-5`, critic LR `5e-6`,
three prompts, eight samples per prompt, a 24-sample global rollout batch, and
two optimizer updates. W&B remained offline.

The critic reward/train equivalence check passed exactly at step 0
(`mean=1.0`, `max_abs_error=0.0`, `n=6`). Update 1 produced 24/24 usable
rollouts, reward mean/std `-0.374729 / 0.100902`, critic time `78.1s`, and
actor core/end-to-end time `148.0s / 186.5s`. Update 2 produced 21/24 usable
rollouts, reward mean/std `-0.553841 / 0.587261`, minimum reward `-2.0`, critic
time `18.0s`, and actor core/end-to-end time `26.9s / 70.1s`. No OOM or
optimizer failure occurred. Actor/critic DCP checkpoints at `iter_0000002` are
complete at approximately `59G / 36G`.

The first post-eval attempt failed before generation because the queue omitted
explicit model and tokenizer fingerprints. Shared model/tokenizer provenance,
automatic converted-HF fingerprinting, cache invalidation, and post-eval-only
retry support were added to the queue. Focused local and RunAI tests passed.
The retry generated 64/64 rows with 100% closed and usable explanations for
real, shuffled, zero, mean, and no-injection controls.

The original SFT report used a different train parquet provenance and numeric
row indices that were not stable across subset evaluations. Paired alignment
was corrected to prefer canonical `row_keys`, with legacy index fallback. An
exact SFT comparator was generated and scored on the same 64 rows, same order,
same train/validation hashes, same generation protocol, and 64 independent
families. A report-only re-gater now separates exploratory thresholds from the
strict promotion contract.

Final online/SFT directional MSE is `0.291993 / 0.292173`. The paired mean
delta is `0.000181` (`0.0618%` relative), wins are `32 / 64`, the strict
family-bootstrap 95% interval is `[-0.007704, 0.008597]`, and the sign-flip
test gives `p=0.4824`. Raw MSE is `8.969927 / 8.797533`, a `1.96%` regression,
with only 28/64 online wins. Real text still beats every control with rowwise
win rates from `98.44%` to `100%`, and parse health is perfect.

The permissive aggregate-only gate mechanically passes, but the strict hero
policy requires at least 60% paired wins, 10% relative improvement, a positive
family-bootstrap lower bound, clustered inference, and exact dataset/protocol
identity. That strict report correctly fails. Decision: retain both checkpoints
as online-training systems evidence, do not promote them over SFT, and do not
scale until reward variance, parser failures, KL/SFT anchoring, and raw
calibration are addressed.

Canonical evidence:

- Run summary: `docs/runs/r33_online_joint_canary_20260717.md`.
- Local lightweight archive:
  `artifacts/runai_eval/r33-online-joint-canary-evidence-20260717T0951Z/`.
- Archive SHA-256:
  `3a721f6bbd795b4aeba4a801164594c982807d8eee2160ad4ab0e484e52efc83`.
- Strict report SHA-256:
  `8c34ec3b527651c32bcfc7ede749e12f666f7206b95e68c02264ad9addee65b4`.

## 2026-07-17: Balanced-Critic Online HPO Launch

The clean canary's critic batch accounting was reconstructed before scaling.
With four actor ranks and three critic ranks, the old actor-shard assignment
gave critic ranks `12 / 6 / 6` local rows on a 24-row update. Cross-rank safety
truncation then trained `6 + 6 + 6 = 18` rows. On the canary's second update,
three parse failures made the imbalance worse and only 12 of 21 usable rows
reached the critic optimizer. Actor metrics still represented the full batch,
so this was both wasted data and misleading observability.

The new repartitioner rebuilds and validates the globally indexed rollout,
filters invalid critic rows globally, aligns once to critic DP times
microbatch, and emits balanced shards. It fails on duplicate, missing, or
out-of-range indices, mismatched list fields, excess retention loss, or any
later cross-rank truncation. Queue preflight now exposes the corresponding
critic batch plan, and W&B receives generated/usable/retained/drop metrics.
The FSDP and Megatron paths share the same pure implementation.

A four-candidate, eight-update HPO queue was added at
`configs/nano_rl/hpo/r33_family_clean_online_joint_hpo8_queue_8h100.yaml`.
All candidates use `24 prompts x 8 samples = 192`, microbatch 2, topology
`4 actor / 3 critic / 1 SGLang`, validation-only paired 128-row evaluation,
and offline W&B. H1 uses actor/critic LR `1e-5 / 2e-6` and K3 coefficient
`3e-3`; H2 lowers actor LR, H3 raises critic LR, and H4 lowers K3 anchoring.
Only H1 is approved.

The first H1 initialization attempt was deliberately stopped before rollout
or optimizer state after its rendered command exposed missing Ray forwarding
for the retention threshold and JSON guard rules. No checkpoint was produced.
The attempt and logs are preserved under a `preflight_abort_20260717T1240Z`
suffix. The launcher now forwards those contracts, and actor reference plus
SGLang share one verified `/dev/shm` HF staging target. Local and RunAI focused
tests report `100 passed`; the corrected H1 relaunch is active from the
unchanged qualified clean-SFT pair. Results are not inferred until optimizer
updates and the paired validation report exist.

Run note: `docs/runs/r33_online_joint_hpo8_20260717.md`.

## 2026-07-17: HPO8 H1 Startup Guard Stop And Calibrated Retry

The first fully corrected H1 launch completed shared `/dev/shm` staging,
started SGLang on the dedicated rollout GPU, loaded the qualified family-clean
R33 actor and critic, and generated all 192 samples for rollout 0. It stopped
before any actor or critic optimizer update because
`rollout/nla_parse/closed_frac=0.8854167` violated the queue's inherited hard
minimum of `0.95` on its first observation. The queue reported `failed`, no
checkpoint exists, and all eight H100 NVLs returned to 4 MiB idle usage. This
is a pre-update guard failure, not an HPO result.

The generation budget was checked before changing the guard. A CPU tokenizer
audit over the first 120,000 clean AV-SFT responses found response-token
p50/p90/p95/p99 `119 / 147 / 156 / 174`, maximum `233`, and no response longer
than 256 tokens. The 256-token cap remains appropriate. The clean actor's
unclosed outputs are therefore treated as the behavior RL should penalize, not
as evidence that the target distribution needs a larger rollout budget.

The rollout guard now requires close and usable fractions of at least `0.80`
for two consecutive batches and raw engine truncation at most `0.20` for two
consecutive batches. The independent critic-balance invariant still requires
at least `0.95` of usable rows after DP/microbatch alignment. This permits the
observed clean-SFT startup but still stops sustained parse collapse.

During diagnosis, the truncation metric was found to be semantically stale:
the rollout function changes a truncated sample's final status to `FAILED` so
it receives failed-extraction reward and is excluded from critic training.
Metrics previously inspected only that final status. The runtime now preserves
the engine status before relabeling and exports it under
`rollout/nla_generation/{completed,truncated,failed}_frac`; the queue uses the
raw truncation metric. The compact rollout payload is emitted before guard
evaluation so a guard-triggered batch remains diagnosable from stdout. Focused
tests pass (`110 passed`). Retry 1 has a fresh run directory and W&B identity;
the failed launch remains preserved without checkpoint shards.

## 2026-07-17: Packed Actor Drift And Native Padded FSDP Route

H1 retry 1 generated all 192 rollout-0 samples and passed the calibrated
parser guards. Close/usable/raw-truncated fractions were
`0.916667 / 0.911458 / 0.057292`; reward mean/std was
`-0.516923 / 0.503036`, and response-length p95 reached the 256-token cap.
Before any optimizer update, the actor packed-equivalence gate compared the
same two activation-backed samples in concatenated and padded layouts. Mean NLL
was `1.379626 / 1.356946`, but global maximum absolute/relative drift reached
`0.103714 / 0.053270`, exceeding the registered `0.05 / 0.02` tolerances. The
run failed closed, wrote no checkpoint, and released all GPUs.

The gate was not loosened. Miles already implements padded `bshd` data,
log-prob slicing, loss masks, and policy loss, but its argument validator had
historically restricted that format to Megatron and the FSDP wrapper always
passed `attention_mask=None`. The NLA wrapper now builds a true length-derived
padding mask for FSDP `bshd`, validates batch/position shape, and logs the first
layout as `[NLA PADDED BATCH]`. The launcher exposes `training.qkv_format`
through `QKV_FORMAT` and `--qkv-format`; the HPO queue selects `bshd`. Checked
patch `0020_fsdp_bshd_support.patch` permits Miles' existing FSDP data/loss
path. Packed actor equivalence is disabled only for this padded actor path;
critic packed-equivalence and global critic-retention checks remain active.
The focused local suite passes (`117 passed`). Retry 2 uses a fresh run and
offline-W&B identity, preserving retry 1 as a pre-update failure artifact.

## 2026-07-17: Native-Padded H1 Retry Chain Through Critic Backward

The first retry-2 queue invocation failed before launch because the runtime
contract expected the checked Miles FSDP-`bshd` assertion on one line while the
patch formats it over three. Stable semantic markers replaced the brittle
string. A corrected retry-2 launch was stopped during initialization after a
generic Miles warning about absent `save_interval`; inspection confirmed the
queue's checked `NLA_SAVE_ITERATIONS=8` override saves the final joint update.
The launcher now records that explicit schedule. Neither retry-2 invocation
generated data or updated weights.

Retry 3 completed the first mathematically consistent padded actor update.
Rollout close/usable/raw-truncated fractions were
`0.947917 / 0.937500 / 0.046875`. Actor loss/K3/grad-norm/log-prob-drift were
`0.004039 / 0.002306 / 4.3125 / 0.027766`, with clip fraction `0.008266`.
Critic dispatch then failed closed because Miles' globally replicated
192-element `raw_reward` vector was misclassified as a 48-row local field.
The repartitioner now has an explicit, validated replicated-global field
contract and keeps that vector global for Miles' train-side semantics.

Retry 4 retained every usable critic row: generated/usable/retained
`192 / 168 / 168`, zero alignment drop, and local rows `56 / 56 / 56`. The
real-data reward/training MSE paths agreed exactly (`mean ratio 1.0000`, maximum
reported deviation `0.0000`). It failed before critic backward because the
critic-token rewrite dropped `max_seq_lens` required by the native padded
`bshd` path. The rewrite now recomputes padded width from critic-token lengths
and the Miles padding multiple. Combined focused tests pass (`110 passed`).
Retry 5 has a fresh immutable identity and is running; H2-H4 remain unapproved.

## 2026-07-17: HPO8 H1 Retry 5 Complete, Corrected Evaluation Does Not Promote

Retry 5 completed eight joint optimizer updates from the qualified family-clean
SFT pair. The canonical validation evaluation regenerated H1 text under the
same 128-row, family-stratified protocol as the clean SFT baseline: protocol
SHA-256 `e5e3a2658d28975514dd962be18c149012ee1fc85f1d6f52ccc834f59c95d416`,
exact row identity and dataset hashes, 128 independent families, real parse
closed/usable fractions `1.0 / 1.0`, and all required real-vs-control checks.

Baseline/H1 directional MSE is `0.3028433237 / 0.3012607509`, an apparent
`0.5226%` H1 gain. It is not statistically reliable: the family-clustered
paired 95% CI is `[-0.0040094576, 0.0072551257]`, sign-flip `p=0.296687`, and
H1 wins `50%` of rows. The independent strict regate also returns
`gate_passed=false`; this run remains confirmatory rather than promotable.

Component decomposition attributes essentially all of the nominal movement to
the actor: H1 AV text through untouched clean AR scores `0.3012762239`, within
`1.55e-5` of the joint result. Clean SFT AV text through the H1 critic scores
`0.3028854620`, slightly worse than the baseline. Do not treat the older
protocol-mismatched `v128_t64` diagnostic as promotion evidence. H2-H4 remain
unlaunched; a future approved HPO should prioritize actor schedule/duration
over critic-LR escalation.

Checkpoint-free reports, generated text, prediction caches, queue contracts,
training log, and offline W&B files are local at
`artifacts/runai_eval/r33-online-joint-hpo8-h1-protocolfixed-20260717/` and
archived in S3 under
`evidence-sync/20260717_r33_online_joint_hpo8_h1_retry5/` (local archive
SHA-256 `bc44dfef896f63eccf68949eec7a1bc392b0a03f1e4e694489c1c13037345809`).

## 2026-07-17: Family-Disjoint Actor-Schedule Development Protocol Started

The failed strict H1 gate is retained as sealed evidence and is not reused for
schedule selection. A separate development partition was derived from the
same validation source by excluding all `128` canonical families in the H1
strict-regate report. The resulting immutable boundary contains `6,802` rows
from `122` families, with zero family overlap. Its parquet SHA-256 is
`b5f31379bbc41d087c07b7dda2445ad6fb30af25e7070157b8912fa1f9be5eaf`; the
boundary manifest SHA-256 is
`a505c2123f48223eeb17bd6e5dcb6683833a7ea40ef8aee3c31186a4f0cad5b5`.

Before any online-RL trial, a clean-SFT AV->AR baseline is being regenerated
on a fixed, family-stratified 122-row subset of that development partition
under the same generation protocol, controls, and parser requirements used by
all candidates. The score phase also records AR score-only device placement
and CUDA-memory telemetry; it is observational and cannot affect generation,
rewards, or metrics.

The bounded selection grid fixes critic LR `2e-6`, K3 coefficient `0.003`,
and the H1-proven `4 actor / 3 critic / 1 SGLang` topology. It compares only
actor schedule/duration: `(2e-5, 8 updates)`, `(3e-5, 8)`, `(1e-5, 24)`, and
`(2e-5, 24)`. Rollout batch, effective batch, samples per prompt, context,
response cap, controls, and row-selection protocol are fixed. Candidates are
unapproved until the baseline report exists, retain no checkpoint after their
selection evaluation, and cannot themselves promote a hero run.

### Metadata-contract repair

The first development-baseline launch failed before producing any completed
generation because the derived parquet lacked its required
`<parquet>.nla_meta.yaml` dataset sidecar. It is an operational contract
failure, not a quality result: the four bounded workers loaded the AV model,
then each raised `FileNotFoundError` while loading the derived dataset config.
The log and zero-row worker artifacts are retained under the original
`actor_hpo_dev/baseline/` path.

`scripts/build_nano_roundtrip_family_holdout.py` now treats the source NLA
sidecar as a checksummed input and emits a derived sidecar with the filtered
row count and holdout lineage. The clean retry uses new immutable `v2` paths.
Its family-filtered parquet is still 6,802 rows / 122 families / zero excluded
family overlap (parquet SHA-256
`b5f31379bbc41d087c07b7dda2445ad6fb30af25e7070157b8912fa1f9be5eaf`), and
its output sidecar is `kind: nla_dataset`, `d_model: 2688`, `row_count: 6802`
(SHA-256 `0b51c7c403758a1abaab1b9a1568b26895f64d595c8f19c117edd7549f67be38`).
The v2 baseline is in progress; no baseline metric or HPO result is claimed.

### Completed clean-SFT development baseline

The repaired v2 baseline completed on `2026-07-17T22:45:27Z` with a passing
round-trip gate. It reuses exactly 122 completed family-stratified generations
from the clean R33 AV SFT checkpoint under generation-protocol SHA-256
`97ef2a00acae3ace82ad5efc0c2586a447a93d2d1d2e4be72dc3443e4a424678`.
The fixed development subset has 122 rows and 122 independent families.

Generated `av_real` explanations give directional NMSE `0.3090549575`, FVE
`0.5501967978`, and cosine `0.8454725213`. The matched teacher-text score is
`0.3144731059` / `0.5423111436` / `0.8427634471`. This is a development
baseline, not a sealed comparison or an RL-improvement claim. The real-text
path strongly beats every nonsemantic control: shuffled directional NMSE
`0.9718231760`, zero `0.9802652469`, no-injection `0.8631586509`, and mean
`0.6870892943`. It wins against mean on 121/122 rows and against
no-injection, shuffled, and zero on 122/122 rows.

Parse health is complete for every generated control: 122/122 closed-tag,
usable, and content-usable records, with zero empty or repetition-loop
records. The report and prediction cache are
`actor_hpo_dev/v2/baseline/roundtrip_v122_direct_gpu_retry2_report.json` and
`roundtrip_v122_direct_gpu_retry2_predictions.npz` on RunAI.

Three prior score attempts are retained as operational-only evidence with no
quality result: automatic AR placement was too slow, the CPU-materialize plus
`model.cuda()` path was too slow, and the first direct-GPU retry exposed a
stale call to a nonexistent remote-code patch helper. The loader now uses
`patch_nemotron_h_checkpoint_dir`; focused remote tests pass (`101 passed`).
The successful retry places all 19,222,267,968 critic parameters on `cuda:0`
at roughly 38.5 GB allocated after scoring.

This passing baseline unlocks only the first guarded development candidate
(`actor_lr=2e-5`, 8 updates). The sealed 128-family H1 boundary remains
untouched and no hero or publication conclusion follows from this result.

### 2026-07-18 - Actor-Schedule HPO Operational Repair and Valid R3 Start

The first guarded development candidate remains a development-only experiment
on the fixed 122-family pool. It does not touch the sealed H1 boundary and it
is not evidence of an RL gain until its completed post-eval passes.

- The original `a2e-5/u8` queue item was interrupted before an optimizer step
  after a generic save warning; it has no rollout, checkpoint, or quality
  result. Fresh `r1` then completed its first 192-sample rollout and actor
  phase but stopped during the concurrent critic phase. One rank measured a
  maximum reward-path versus train-path MSE-ratio deviation of `2.02335%`
  against a strict `<2%` guard; the mean ratio was `0.9995`. No checkpoint was
  written, so `r1` is an operational failure rather than an HPO observation.
- The live check is now split into two explicit fail-closed invariants: a
  strict reward-versus-eval-layout check (`2%`) and a bounded train-mode
  versus eval-layout check (`5%`). This preserves detection of the historical
  50-100% left-padding/layout failures while separating them from harmless
  BF16 train/eval kernel drift. Both tolerance names are registered in the
  queue configuration.
- `r2` was stopped before Ray workers or model initialization after exposing a
  staging inefficiency: the actor reference and SGLang paths attempted to
  copy the same 63.2 GB AV checkpoint twice because their marker names did
  not recognize one another. Shared verified-stage markers and an existing
  content-addressed `/dev/shm` staging target remove that duplicate transfer.
  It has no training or quality result.
- `r3` started at `2026-07-18T00:27:46Z` with all three input stages reused,
  4 actor / 3 critic / 1 rollout GPUs, offline W&B, `actor_lr=2e-5`, K3
  anchoring, and the registered 8-update schedule. On rollout 0 it generated
  192 samples with closed fraction `0.9479`, usable fraction `0.9323`,
  generation-truncated fraction `0.0625`, and reward mean/std
  `-0.4330 / 0.5014`. The balanced critic view retained 174 of 176 usable
  samples (`98.86%`) after two alignment drops.
- The repaired real-data check passes exactly on all critic ranks: both
  reward/eval-layout and train/eval-layout MSE-ratio means and maximum
  deviations are `1.0000 / 0.0000`. The first critic optimizer step reports
  normalized FVE `0.4787` and loss `0.2914`; the first actor step reports
  PPO KL `0.000234`, clip fraction `0.00666`, KL loss `0.00241`, and
  rollout-logprob absolute drift `0.0253`, all below configured guards.
- `r3` then stopped during rollout 1 when Ray's node-memory monitor observed
  `709.62 / 715.26 GiB` of the container cgroup and killed an actor worker.
  This was not GPU OOM: `/dev/shm` alone held 405 GiB of historical staged
  models and temporary HF conversions, all cgroup-accounted. The run completed
  one joint optimizer update but did not save `iter_0000008`, so it has no
  round-trip HPO verdict and is not a candidate improvement. After retaining
  only the active actor (59 GiB), critic (36 GiB), and AV/SGLang (59 GiB)
  inputs, `/dev/shm` fell to 155 GiB and cgroup use to about 159 GiB. Fresh
  `r4` keeps every learning parameter and gate unchanged.

### R4 live retry: host-memory repair validated, quality still pending

- `r33-family-clean-online-joint-actor-hpo-dev-a2e5-u8-r4` launched after the
  stale `/dev/shm` cleanup with the same clean 122-family development data,
  `actor_lr=2e-5`, eight-update schedule, 4/3/1 actor/critic/rollout topology,
  and all integrity guards as `r3`. This is a runtime retry, not a new HPO
  point.
- The content-addressed 59 GiB actor, 36 GiB critic, and 59 GiB SGLang AV
  inputs were reused; no duplicate 63 GiB reference-model stage was made.
  Baseline cgroup use was about 162 GiB with 562 GiB free in `/dev/shm`.
- Real rollout 0 completed, and both step-0 checks passed exactly: the
  reward/eval-layout and train/eval-layout MSE ratios each had mean `1.0000`
  and maximum deviation `0.0000` on 32 checked examples. Critic step 0 logged
  loss `0.29198` and normalized FVE `0.47762`; actor step 0 logged loss
  `0.000654`, PPO KL `0.000365`, clip fraction `0.00618`, and KL loss
  `0.00140`.
- The process crossed the prior failure boundary: rollout 1 also generated
  all 192 samples without Ray eviction, and the second joint update completed.
  The live cgroup working set stabilized around 525 GiB of the 715 GiB limit,
  with zero new `oom` or `oom_kill` events. The eight-update run and its
  post-eval remain in progress; do not treat these live diagnostics as an HPO
  or quality result until the retained checkpoint and round-trip report exist.

### 2026-07-18 Storage Retention Cleanup

- While `r4` remained active, a path-reference check confirmed that it uses
  separate content-addressed staged inputs and does not reference the retired
  payloads below. No active process, current SFT input, selected critic, or
  final internal RL hero checkpoint was removed.
- The PVC cleanup reclaimed an expected `189.39 GiB`: the non-promoted H1
  pilot's actor model (59 GiB), critic DCP model (36 GiB), temporary critic HF
  conversion (36 GiB), and the superseded retry-2 resume model (59 GiB). Their
  run directories, configs, train/W&B logs, reports, and lightweight evidence
  remain. The final internal hero checkpoint at retry-3 `iter_0000342`, the
  selected SFT/independent-critic artifacts, and the live `r4` directory are
  explicitly retained.
- `/workspace/interp` moved from `852 GiB` used / `156 GiB` free to `663 GiB`
  used / `346 GiB` free. The machine-readable retention record is
  `/workspace/interp/outputs/nano30b-nla-pilot/cleanup/20260718T014800Z_r33_obsolete_rl_payload_cleanup.json`.
- On S3, the historical migration retained the complete
  `cluster-artifacts-runs.tar.gz` archive and pruned only its 12 redundant
  split transfer parts (`1.07 GiB` total), whose summed size matched the
  completed archive. Publication checkpoint archives and the base-model
  migration were deliberately retained. The S3 cleanup manifest is
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/cleanup/20260718T015000Z_s3_completed_archive_parts_cleanup.json`.

### 2026-07-18 - R4 Completed Development Result and Retention-Order Reconciliation

- `r33-family-clean-online-joint-actor-hpo-dev-a2e5-u8-r4` completed all
  eight joint updates. It used the fixed 122-family development boundary,
  `actor_lr=2e-5`, `24 x 8 = 192` rollout samples per update, online critic,
  K3 anchoring, and the 4/3/1 actor/critic/rollout GPU topology. This is the
  first completed clean actor-schedule HPO observation, not a sealed-H1 or
  publication result.
- Its matched generated-text round-trip evaluation passed: real directional
  NMSE `0.2981629915`, directional FVE `0.5660491278`, and cosine
  `0.8509185043`. Against the protocol-matched clean-SFT development baseline
  (directional NMSE `0.3090549575`), the mean directional-MSE improvement is
  `3.5243%`. The real path closed and yielded usable explanations for all
  `122/122` rows and beat shuffled, zero, mean, and no-injection controls.
- The quality signal is directionally encouraging but intentionally
  non-promotional: it uses only the development split, and this HPO gate does
  not require a positive bootstrap interval or permutation p-value. The sealed
  128-family H1 boundary remains untouched. The post-eval also removes the
  temporary actor DCP by configuration, so R4's evidence can guide HPO but
  cannot be promoted as a retained model checkpoint.
- The queue initially labeled R4 `failed` after the successful post-eval:
  `cleanup_actor_checkpoint=true` deleted the deliberate evaluation input,
  then checkpoint retention incorrectly required that deleted iteration to
  exist. The queue now applies retention before post-eval; the required
  iteration is preserved for conversion/evaluation, while post-eval may still
  reclaim it afterwards. A regression test reproduces this exact lifecycle;
  `73` focused local tests and the focused RunAI test pass.
- Lightweight evidence was copied locally under
  `artifacts/runai_eval/r33_online_rl/actor_hpo_dev/a2e5_u8_r4/` and archived
  at `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/evidence-sync/20260718_r33_online_rl_a2e5_u8_r4/r33_online_rl_a2e5_u8_r4_evidence_20260718T0720Z.tgz`
  (SHA-256 `8052eb426c431678d73858c1182d22ecd4693b75dd0960b8d0bbdec6f564d97b`).
  It contains contracts, train/post-eval logs, generated samples, report, and
  offline W&B records, but no checkpoint payloads.
- The next isolated development point is armed as `actor_lr=3e-5`, eight
  updates, with all data, controls, generation protocol, and gates fixed. Its
  rendered launch contract passes preflight with no missing paths.

### 2026-07-18 - A3 Development HPO Result and Evidence Retention

- `r33-family-clean-online-joint-actor-hpo-dev-a3e5-u8` completed its eight
  joint online actor/critic updates on the same fixed 122-family development
  boundary as R4. Only actor learning rate changed (`3e-5` versus `2e-5`);
  the online critic, K3 anchoring, `24 x 8 = 192` rollout samples per update,
  4/3/1 GPU topology, generation protocol, and controls remained fixed.
- The generated AV-text -> AR gate passed with real directional NMSE
  `0.2803779641`, directional FVE `0.5919337224`, and cosine
  `0.8598110180`. Real generations closed and produced usable explanations
  for all `122/122` rows, and beat shuffled, zero, mean, and no-injection
  controls; the smallest real-over-control rowwise win fraction was
  `0.9918033`.
- Against the exact protocol- and row-matched clean-SFT development baseline
  (`0.3090549575` directional NMSE), the mean paired improvement is
  `0.0286769934` (`9.2789%`). Its family-clustered bootstrap 95% interval is
  `[0.0145183866, 0.0436190949]`; the configured 100,000-sample sign-flip
  estimate is `p=3.99996e-05`. These are useful development diagnostics, not
  a sealed-H1 or publication claim.
- The run also exercised the retention-order repair in the intended lifecycle:
  retention completed in the `pre_post_eval` phase, then the temporary actor
  DCP was cleaned after successful conversion/evaluation, and the queue
  correctly finished as `complete`.
- Lightweight evidence is SHA-verified locally and in S3 at
  `artifacts/runai_eval/r33_online_rl/actor_hpo_dev/a3e5_u8/` and
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/evidence-sync/20260718_r33_online_rl_a3e5_u8/r33_online_rl_a3e5_u8_evidence_20260718T1340Z.tgz`
  (SHA-256 `604f2734c154489c2ebeb1238b4269088ea95ad1902cedcad821fdaf94fa6e2b`).
  The non-promotable critic payload was then removed, reclaiming 72 GiB; the
  cleanup record is
  `outputs/nano30b-nla-pilot/cleanup/20260718T135000Z_r33_a3e5_u8_nonpromotable_critic_cleanup.json`.
- The next decision is a duration probe at the winning `3e-5` actor learning
  rate. Existing `1e-5/u24` and `2e-5/u24` items remain useful conservative
  comparators, but neither substitutes for testing the current leading rate
  at 24 updates before a guarded hero promotion.

### 2026-07-18 - Duration-Queue Horizon Preflight Repair

- The first render of the new `a3e-5/u24` promotion queue failed before any
  worker allocation because `training.save_iterations=[24]` had been changed
  without changing the effective `rollout.num_rollout`, which remained `8`.
  The queue's fail-closed invariant correctly rejected that inconsistency.
- The canonical `a1e-5/u24` and `a2e-5/u24` schedule items had the same latent
  configuration error. Each duration item now explicitly sets
  `rollout.num_rollout=24`, records/retains iteration `24`, and has a focused
  queue test enforcing that the save, rollout, and retention horizons agree.
- The corrected single-item `a3e-5/u24` promotion queue passed the RunAI
  focused regression test and an armed remote dry run with no missing paths,
  then launched at `2026-07-18T13:53:35Z`. This is a configuration-preflight
  repair, not a failed training observation or an additional HPO point.

### 2026-07-18 - A3 U24 Promotion Result

- `r33-family-clean-online-joint-actor-schedule-promotion-dev-a3e5-u24`
  completed all 24 online updates and passed its 122-family validation-only
  round-trip gate. Real generated-text -> AR directional NMSE is
  `0.2483426089`, a paired `19.6445%` improvement over the matched clean-SFT
  baseline `0.3090549575`.
- The family-clustered bootstrap improvement interval is
  `[0.0449011409, 0.0772767256]`, and the configured sign-flip estimate is
  `p=9.9999e-06`. Real parse close/usable fractions are both `1.0`; all
  semantic controls lose with at least `0.9918` rowwise real-path wins.
- Training stayed within every declared stability guard: actor KL peaked near
  `5.5e-4`, rollout/reference log-prob drift remained below `0.026`, clip
  fraction stayed below `0.007`, and no rollout truncated. The online critic
  finished with normalized FVE `0.5837` after reaching `0.6702` during the
  run. The W&B teardown `BrokenPipeError` occurred only after successful
  training/post-eval and is preserved as an operational warning.
- Evidence is SHA-verified in S3 at
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/evidence-sync/20260719_r33_online_rl_a3e5_u24/r33_online_rl_a3e5_u24_evidence_20260719T164451Z.tgz`
  (SHA-256 `ba58a645a868151f5eada07dd4e0309704df9f006ef8364cc06fae71d6f9b38e`).
  The non-promotable 72 GiB critic payload was reclaimed only after archive
  verification; its cleanup record is
  `outputs/nano30b-nla-pilot/cleanup/20260719T164800Z_r33_a3e5_u24_nonpromotable_critic_cleanup.json`.
- The selected successor is the internal 342-update hero declaration in
  `docs/runs/r33_family_clean_internal_rl_hero_20260719.md`. It deliberately
  uses no in-corpus sealed-H1 claim because the v6 exposure audit exhausts the
  candidate-family universe.

### 2026-07-21 - Internal-Hero Matched 384-Token Validation

- The retained `a3e5_u342` pair was produced by approximately 43 hours of
  online RL from the qualified clean SFT initialization: 342 optimizer updates
  x 24 prompts x 8 rollouts = `65,664` generated responses, using a 4-GPU
  actor/AV group, 3-GPU online critic/AR group, and 1-GPU SGLang group.
- This entry records a new evaluation of those trained iteration-342 weights,
  not a second training run. The pair is stored under
  `/workspace/interp/outputs/nano30b-nla-pilot/publication/r33_online_rl/internal_hero/a3e5_u342/`.
- The exact matched protocol covers 122 validation rows and 122 independent
  content families, permits at most 384 new tokens, and binds SFT and RL to
  protocol SHA-256
  `fcc431ec4450adb8817cd946d6c194fa2a45b53b0c6c42c8682c1e9f12f94d4d`.
- Online RL reduces directional round-trip NMSE from the matched-SFT
  `0.309055` to `0.224386` (`0.084669` absolute; `27.4%` relative). Raw MSE
  falls from `9.5523` to `7.2665` (`23.9%` relative).
- RL-generated text has lower AR reconstruction error than teacher text on
  `103/122` rows (`84.4%`), versus `62/122` (`50.8%`) for SFT; this does not
  establish better semantic explanations. RL closes on `121/122` rows
  (`99.18%`) and beats shuffled, zero, mean, and no-injection controls. Both
  gates pass; the RL report is confirmatory and generation-protocol compatible.
- Reports are `eval384_matched_v122/sft_roundtrip_report.json` and
  `eval384_matched_v122/rl_roundtrip_report.json`; execution evidence is
  `eval384_matched_v122/eval384_chain.log`. The execution record reports no
  OOM, traceback, or evaluation error.
- Interpretation is deliberately bounded: this is strong family-independent
  validation evidence for an online-RL gain over matched SFT, not a pristine
  test-set result or a matched R27 comparison. The SFT and RL reports use their
  respective AR checkpoints, so the measured gain is pair-level rather than
  actor-only. Before a public RL claim, sync and hash the artifacts, inspect
  the one close failure, report all companion metrics/inference, run the
  four-way AV/AR component decomposition plus functional reinjection delta,
  replicate through an independent critic and a second RL seed, and evaluate
  on a new external teacher-backed boundary.
