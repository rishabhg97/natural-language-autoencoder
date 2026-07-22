# RunAI Miles/FSDP2 NLA Integration

Date: 2026-05-28

Status update (`2026-07-10`): this document records the original Miles/FSDP2
integration milestone. The R33 corrected-K3 RL hero `iter_0000342` is retained
as a historical systems artifact, but its baseline mixed generation protocols
and it is not a current release candidate. See
`docs/current_state.md` and `docs/runs/r33_rl_hero_20260708.md`.

## Historical Integration Status

Phase 1 import gate passes on RunAI `train-dev` after installing Miles from the
pinned source and applying the NLA patches. Phase 2 dataset contract passes on a
full 99,570-row AV-SFT parquet from the latest held batch8 run directory.

Small and medium-small Miles/FSDP2 gates proved the real NLA actor path can
train with batched activation transport, produce row-specific validation/test
likelihood gaps, save checkpoints, and resume with full optimizer/scheduler
state. The complete-performance AV-SFT hero also completed on `2026-05-28`:
one full train split epoch, global batch `192`, micro batch `8`, and final exact
resume checkpoint `iter_0000467`.

The completed hero is AV-SFT only (`R_27` activation h -> explanation z). It is
not an AV+AR run. AR/critic SFT is a separate path: medium-small AR-SFT passed
heldout reconstruction controls, and a 275,396-row R27 fullscan AR-SFT run was
launched on `2026-05-30`. RL has not started.

Final bounded checkpoint eval on `hf_iter_0000467` with validation/test counts
`64/64` shows row-specific heldout signal: real activation NLL beats shuffled,
zero, mean, and no-injection on both validation and test. Validation real NLL is
`0.9046`; test real NLL is `0.9565`. The zero/mean gaps are positive but below
the aspirational `0.30` target on this bounded eval, so run a larger eval before
making stronger scientific claims.

Important distinction: the durable artifact below is the explained source
dataset, not the final AV-SFT Miles dataset:

```text
/workspace/interp/artifacts/nano30b-nla-pilot/super-teacher-r27-100k-thinking-merged-20260525T2150Z/base_R27_super_thinking_99570_explained.parquet
```

It has `activation_vector` and explanations, but it does not have Qwen AV-SFT
`prompt` and `response` columns. The verifier correctly rejects it. A Miles
AV-SFT run should use a materialized `av_sft.parquet`, for example:

```text
/workspace/interp/outputs/nano30b-nla-pilot/av-r27-100k-rslora-batch8-epoch1-v1/av-r27-99570-rslora-r192-broad-scale75-lr1e5-b8-e1-epoch-gen8-save-2gpu-offline-20260527T0250Z/av_sft.parquet
```

For the original 100k AR-SFT plan, use the Qwen-faithful critic dataset produced
from the same 99,570-row source:

```text
/workspace/interp/outputs/nano30b-nla-pilot/ar-r27-100k-qwen-faithful/ar_sft.parquet
```

The AR dataset verifier passed with `d_model=2688`, finite activations,
nonempty explanations, sidecar checks, and zero document overlap. It is a
text/explanation z -> activation h reconstruction contract, not an AV prompt
contract.

For the fullscan AR-SFT run, use the R27 handoff dataset copied to RunAI:

```text
/workspace/interp/outputs/nano30b-nla-pilot/ar-r27-r30-fullscan-20260528T234403Z/R_27/ar_sft.parquet
```

The fullscan R27 dataset has `275,396` rows. The uploaded parquet sha256 matched
the local handoff artifact, and the RunAI verifier passed with `d_model=2688`,
zero non-finite activation values, zero empty explanations, zero critic suffix
failures, and zero doc split overlap.

## RunAI Environment Setup

Inside `train-dev`:

```bash
source /workspace/interp/.venv/bin/activate
cd /workspace/interp/code
```

Miles was installed as a source checkout:

```text
/workspace/interp/code/miles-051cd15
```

RunAI could not clone GitHub directly (`CONNECT tunnel failed, response 403`),
so the pinned `radixark/miles@051cd15` source was transferred from the Mac into
`/workspace/interp/code`. Apply patches in order:

```bash
cd /workspace/interp/code/miles-051cd15
patch -p1 < /workspace/interp/code/nano30b-nla-pilot-current/external/natural_language_autoencoders/nla/miles_patches/0001_miles_nla_integration.patch
patch -p1 < /workspace/interp/code/nano30b-nla-pilot-current/external/natural_language_autoencoders/nla/miles_patches/0002_train_py_nla_hooks.patch
patch -p1 < /workspace/interp/code/nano30b-nla-pilot-current/external/natural_language_autoencoders/nla/miles_patches/0003_fsdp_sft_import_fallbacks.patch
patch -p1 < /workspace/interp/code/nano30b-nla-pilot-current/external/natural_language_autoencoders/nla/miles_patches/0004_fsdp_timing_debug.patch
patch -p1 < /workspace/interp/code/nano30b-nla-pilot-current/external/natural_language_autoencoders/nla/miles_patches/0005_fsdp_skip_grad_norm_debug.patch
patch -p1 < /workspace/interp/code/nano30b-nla-pilot-current/external/natural_language_autoencoders/nla/miles_patches/0006_fsdp_checkpoint_gloo_pg.patch
```

Editable installs:

```bash
python -m pip install --no-deps -e /workspace/interp/code/miles-051cd15
python -m pip install --no-deps -e /workspace/interp/code/nano30b-nla-pilot-current/external/natural_language_autoencoders
python -m pip install "ray[default]" ring_flash_attn orjson omegaconf pybase64 blobfile pylatexenc qwen_vl_utils "sglang-router>=0.2.3" "mcp[cli]" memray
```

`ring_flash_attn` imports are patched to no-op when incompatible with
Transformers 5.9.0 because Nano AV-SFT uses `context_parallel_size=1`; the ring
attention path is not used. SGLang weight-sync imports are patched to raise only
if used; `--debug-train-only` SFT does not start SGLang.

## Gates Run

Import gate:

```bash
cd /workspace/interp/code/nano30b-nla-pilot-current
python scripts/check_miles_nla_imports.py \
  --report-json /workspace/interp/outputs/nano30b-nla-pilot/miles_import_gate_20260527.json
```

Result: `ok: true` for:

```text
miles
nla.train_actor.NLAFSDPActor
nla.rollout.sft_actor.generate_rollout
nla.injection.inject_at_marked_positions
```

Dataset contract gate on full AV-SFT parquet:

```bash
python scripts/verify_nano_miles_av_dataset.py \
  /workspace/interp/outputs/nano30b-nla-pilot/av-r27-100k-rslora-batch8-epoch1-v1/av-r27-99570-rslora-r192-broad-scale75-lr1e5-b8-e1-epoch-gen8-save-2gpu-offline-20260527T0250Z/av_sft.parquet \
  --expected-rows 99570 \
  --tokenizer-model /workspace/interp/models/nano-30b-a3b-bf16-hf \
  --report-json /workspace/interp/outputs/nano30b-nla-pilot/nano_miles_av_dataset_verify_full_avsft_20260527.json
```

Result:

```text
rows: 99,570
d_model: 2,688
nonfinite activations: 0
malformed responses: 0
prompt marker failures: 0 / 99,570 checked
80/10/10 doc split: 79,647 train / 9,961 validation / 9,962 test / 0 doc overlap
90/5/5 doc split: 89,618 train / 4,978 validation / 4,974 test / 0 doc overlap
```

AR dataset contract gate on full AR-SFT parquet:

```bash
python scripts/verify_nano_miles_ar_dataset.py \
  /workspace/interp/outputs/nano30b-nla-pilot/ar-r27-100k-qwen-faithful/ar_sft.parquet \
  --expected-rows 99570 \
  --expected-d-model 2688 \
  --report-json /workspace/interp/outputs/nano30b-nla-pilot/ar-r27-100k-qwen-faithful/ar_dataset_verify_full_20260528.json
```

Result:

```text
rows: 99,570
d_model: 2,688
nonfinite activations: 0
empty explanations: 0
critic prompt suffix bad_count: 0 / 256 checked
80/10/10 doc split: 79,647 train / 9,961 validation / 9,962 test / 0 doc overlap
90/5/5 doc split: 89,618 train / 4,978 validation / 4,974 test / 0 doc overlap
```

AR fullscan R27 dataset contract gate on RunAI:

```bash
python scripts/verify_nano_miles_ar_dataset.py \
  /workspace/interp/outputs/nano30b-nla-pilot/ar-r27-r30-fullscan-20260528T234403Z/R_27/ar_sft.parquet \
  --expected-d-model 2688 \
  --expected-rows 275396 \
  --tokenizer-model /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-critic-init/nano-ar-r27-critic-init \
  --report-json /workspace/interp/outputs/nano30b-nla-pilot/ar-r27-r30-fullscan-20260528T234403Z/R_27/ar_dataset_verify_runai.json
```

Result:

```text
rows: 275,396
d_model: 2,688
nonfinite activations: 0
empty explanations: 0
critic prompt suffix bad_count: 0
80/10/10 doc split: 220,350 train / 27,530 validation / 27,516 test / 0 doc overlap
90/5/5 doc split: 247,870 train / 13,761 validation / 13,765 test / 0 doc overlap
```

AR fullscan R27 complete-performance launch:

```text
config: configs/nano_ar/fullscan_r27_miles_fsdp2.yaml
run_dir: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-fullscan/nano-ar-miles-fsdp2-r27-fullscan-275k-gb192-mb8-lr1e5-20260530T172928Z
train rows: 247,870, padded to 247,872
optimizer steps: 1,291
batch: global 192, micro 8, rollout 192
lr: 1e-5
checkpoint: exact-resume final save only, save_interval=1291, NLA_KEEP_LOCAL=1
wandb: offline
first observed metric: step 2 loss 1.1862951914469402, fve_nrm -0.7661219437917074, grad_norm 4.5625
final checkpoint: checkpoints/iter_0001291, about 123G
final observed metric: step 1290 loss 0.4939449628194173, fve_nrm 0.2646290063858032, grad_norm 0.6484375
eval report: eval_iter_0001291_v512_t512_report.json
```

Final checkpoint heldout reconstruction eval:

```text
validation rows: 512
validation teacher normalized MSE: 0.511466
validation teacher cosine_mean: 0.744267
validation teacher FVE vs mean: 0.423177
validation controls: mean 0.886695, shuffled 1.153667, blank 1.028922, generic 1.033204, source_context 0.632444, source_raw 0.141013

test rows: 512
test teacher normalized MSE: 0.505286
test teacher cosine_mean: 0.747357
test teacher FVE vs mean: 0.416054
test controls: mean 0.865295, shuffled 1.135018, blank 1.010650, generic 1.014764, source_context 0.613429, source_raw 0.137070
```

This is not an NLL eval; AR-SFT uses the normalized MSE critic objective.
Teacher explanations beat mean, shuffled, blank, generic, and source-context on
both validation and test. `source_raw` is stronger than teacher because it uses
the raw token prefix/control context.

Follow-up AR tuning work started on `2026-05-31`:

- `scripts/eval_nano_ar_miles_checkpoint.py` now emits rowwise teacher-vs-control
  win counts/fractions for Qwen-style QC comparisons.
- `scripts/nano_av_runner.py` now renders optional Miles LR schedule flags
  (`--min-lr`, `--lr-decay-style`, warmup knobs) and `checkpoint.finetune: true`
  as `--finetune` with `--load`, so continuation probes can test fresh LR
  schedules without reloading the old LR scheduler.
- A larger detached eval completed on the final checkpoint:
  `eval_iter_0001291_v2048_t2048_winrates_report.json`, validation/test
  `2048/2048`, controls `teacher`, `teacher_shuffled`, `blank`, `generic`,
  `mean`, `source_context`, and `source_raw`.
- v2048 result: validation teacher normalized MSE `0.503023`, FVE vs mean
  `0.416190`, teacher rowwise beats mean `98.05%`; test teacher normalized MSE
  `0.514389`, FVE vs mean `0.401452`, teacher rowwise beats mean `97.46%`.
- The first HPO continuation probe from `iter_0001291` completed with the
  Qwen-style LR schedule at the proven Nano gb192 shape. Its checkpoint
  `iter_0001547` is the current best AR checkpoint, but the run was longer than
  intended because Miles `--finetune` reset the rollout counter while the runner
  rendered `--num-rollout latest+resume_steps`.
- `iter_0001547` v2048 result: validation teacher normalized MSE `0.436878`,
  FVE vs mean `0.492958`, teacher rowwise beats mean `98.54%`; test teacher
  normalized MSE `0.450516`, FVE vs mean `0.475775`, teacher rowwise beats mean
  `97.71%`.
- HPO continuation configs under `configs/nano_ar/hpo/` are storage-conscious
  tuning probes and use model/HF-only saves; rerun the winning recipe with
  exact-resume state before calling it the final AR milestone. Bounded finetune
  probes must render `resume_steps` directly as `--num-rollout`.

AR-SFT medium-small gate:

```text
run_dir: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-medium/nano-ar-miles-fsdp2-r27-medium-small-continue-5ep-20260528T195213Z
checkpoint: checkpoints/iter_0000045
eval_report: eval_iter_0000045_v128_t128_report.json
```

The passing checkpoint was a continuation from the five-epoch medium HF
checkpoint. Validation teacher normalized MSE `0.795473` beat mean `0.875942`,
source-context `0.867701`, shuffled `0.988657`, generic `1.017015`, and blank
`1.029558`. Test teacher `0.795441` beat mean `0.880934`, source-context
`0.851662`, shuffled `0.995219`, generic `1.008652`, and blank `1.016153`.
`source_raw` was skipped because the AR parquet does not contain usable raw
token ID rows.

Small Nano30B Miles/FSDP2 train/checkpoint gates:

```text
retry8: nano-av-miles-fsdp2-small-one-step-nosaveoptim-retry8-adamforeachfalse-20260527T2035Z
run_dir: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-small/nano-av-miles-fsdp2-small-one-step-nosaveoptim-retry8-adamforeachfalse-20260527T2035Z
config: rows=96, global_batch=8, micro_batch=1, adam_foreach=false, no_save_optim=true, save_interval=1
result: step 0 loss 2.574970, grad_norm 16.0, step_time 131.811s, model-only FSDP checkpoint written
cleanup: checkpoint payload removed 2026-05-27 to reclaim PVC space; lightweight logs/config retained

retry10: nano-av-miles-fsdp2-small-resume-modelonly-retry10-skipemptyoptimlrs-20260527T2057Z
run_dir: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-small/nano-av-miles-fsdp2-small-resume-modelonly-retry10-skipemptyoptimlrs-20260527T2057Z
config: resumed from retry8 model-only checkpoint
result: skipped missing optimizer/LR scheduler metadata, loaded rollout state, step 1 loss 2.511452, grad_norm 13.125, step_time 139.233s

retry11: nano-av-miles-fsdp2-small-one-step-gb8-mb4-nosave-retry11-20260527T1405Z
run_dir: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-small/nano-av-miles-fsdp2-small-one-step-gb8-mb4-nosave-retry11-20260527T1405Z
config: rows=96, global_batch=8, micro_batch=4, adam_foreach=false
result: one true local batch per rank; train microstep progress completed in 25.77s, full actor_train_time 96.948s, step_time 99.972s, loss 2.073258
cleanup: partial final checkpoint save was stopped and its incomplete checkpoint dir was removed

retry12: nano-av-miles-fsdp2-small-one-step-gb8-mb4-adamforeach-nosave-retry12-20260527T1414Z
run_dir: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-small/nano-av-miles-fsdp2-small-one-step-gb8-mb4-adamforeach-nosave-retry12-20260527T1414Z
config: rows=96, global_batch=8, micro_batch=4, adam_foreach=true, no --save
result: no immediate OOM, but post-backward update remained running for >4 minutes with no step log; manually stopped

retry13: nano-av-miles-fsdp2-small-one-step-gb8-mb4-timing-nosave-retry13-20260527T1500Z
run_dir: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-small/nano-av-miles-fsdp2-small-one-step-gb8-mb4-timing-nosave-retry13-20260527T1500Z
config: rows=96, global_batch=8, micro_batch=4, adam_foreach=false, no --save, nla_timing_debug=true
result: invalid timing-patch run; crashed before forward because the first timing patch changed the subclass `_train_step` call contract

retry14: nano-av-miles-fsdp2-small-one-step-gb8-mb4-timing-nosave-retry14-20260527T1507Z
run_dir: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-small/nano-av-miles-fsdp2-small-one-step-gb8-mb4-timing-nosave-retry14-20260527T1507Z
config: rows=96, global_batch=8, micro_batch=4, adam_foreach=false, no --save, nla_timing_debug=true
result: step 0 loss 2.073258, actor_train_time 113.398s, step_time 114.987s
timing: forward 8.98s rank0 / 8.32s rank1; backward 16.26s / 16.27s; microbatch_total 25.38s / 25.40s; clip_grad_norm_raw 86.56s / 67.88s; optimizer_step 1.38s / 1.40s

retry15: nano-av-miles-fsdp2-small-one-step-gb8-mb4-timing-skipgradnorm-nosave-retry15-20260527T1516Z
run_dir: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-small/nano-av-miles-fsdp2-small-one-step-gb8-mb4-timing-skipgradnorm-nosave-retry15-20260527T1516Z
config: rows=96, global_batch=8, micro_batch=4, adam_foreach=false, no --save, nla_timing_debug=true, nla_skip_grad_norm=true
result: step 0 loss 2.073258, actor_train_time 27.942s, step_time 30.100s
timing: forward 8.30s / 8.31s; backward 16.36s / 16.36s; microbatch_total 24.81s / 24.81s; grad norm skipped; optimizer_step 3.00s / 3.06s

gb16/mb4: nano-av-miles-fsdp2-small-one-step-gb16-mb4-timing-skipgradnorm-nosave-20260527T1534Z
run_dir: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-small/nano-av-miles-fsdp2-small-one-step-gb16-mb4-timing-skipgradnorm-nosave-20260527T1534Z
config: rows=96, global_batch=16, micro_batch=4, adam_foreach=false, no --save, nla_timing_debug=true, nla_skip_grad_norm=true
result: step 0 loss 2.104505, actor_train_time 35.704s, step_time 37.333s, peak_reserved ~=126.8 GiB

gb16/mb8: nano-av-miles-fsdp2-small-one-step-gb16-mb8-timing-skipgradnorm-nosave-20260527T1546Z
run_dir: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-small/nano-av-miles-fsdp2-small-one-step-gb16-mb8-timing-skipgradnorm-nosave-20260527T1546Z
config: rows=96, global_batch=16, micro_batch=8, adam_foreach=false, no --save, nla_timing_debug=true, nla_skip_grad_norm=true
result: step 0 loss 2.001115, actor_train_time 35.663s, step_time 37.257s, peak_reserved ~=123.2 GiB

gb32/mb8: nano-av-miles-fsdp2-small-one-step-gb32-mb8-timing-skipgradnorm-nosave-20260527T1549Z
run_dir: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-small/nano-av-miles-fsdp2-small-one-step-gb32-mb8-timing-skipgradnorm-nosave-20260527T1549Z
config: rows=96, global_batch=32, micro_batch=8, adam_foreach=false, no --save, nla_timing_debug=true, nla_skip_grad_norm=true
result: step 0 loss 1.987011, actor_train_time 36.131s, step_time 37.806s, peak_reserved ~=126.9 GiB

gb64/mb8: nano-av-miles-fsdp2-small-one-step-gb64-mb8-timing-skipgradnorm-nosave-20260527T1552Z
run_dir: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-small/nano-av-miles-fsdp2-small-one-step-gb64-mb8-timing-skipgradnorm-nosave-20260527T1552Z
config: rows=96, global_batch=64, micro_batch=8, adam_foreach=false, no --save, nla_timing_debug=true, nla_skip_grad_norm=true
result: step 0 loss 1.984885, actor_train_time 42.823s, step_time 44.676s, peak_reserved ~=126.9 GiB

gb96/mb8: nano-av-miles-fsdp2-small-one-step-gb96-mb8-timing-skipgradnorm-nosave-20260527T1557Z
run_dir: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-small/nano-av-miles-fsdp2-small-one-step-gb96-mb8-timing-skipgradnorm-nosave-20260527T1557Z
config: rows=96, global_batch=96, micro_batch=8, adam_foreach=false, no --save, nla_timing_debug=true, nla_skip_grad_norm=true
result: step 0 loss 1.991807, actor_train_time 54.536s, step_time 56.845s, peak_reserved ~=126.9 GiB

batch_scaling_report: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-small/batch_scaling_skipgradnorm_20260527T1605Z.json

medium-small: nano-av-miles-fsdp2-medium-small-gb96-mb8-interactive-save3-modelonly-numrollout9-20260527T1745Z
run_dir: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-medium/nano-av-miles-fsdp2-medium-small-gb96-mb8-interactive-save3-modelonly-numrollout9-20260527T1745Z
config: row_limit=960, doc split 80/10/10, train=771 padded to 864, validation=99, test=90, global_batch=96, micro_batch=8, num_rollout=9, save_interval=3, no_save_optim=true, nla_timing_debug=true, nla_skip_grad_norm=true
result: completed one padded train epoch and wrote rolling model-only FSDP checkpoints iter_0000003, iter_0000006, iter_0000009
checkpoint_sizes: each checkpoint ~63.18 GB raw bytes / 59 GiB du; total checkpoint dir ~177 GiB
training_summary: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-medium/nano-av-miles-fsdp2-medium-small-gb96-mb8-interactive-save3-modelonly-numrollout9-20260527T1745Z/medium_small_training_summary.json
throughput: non-checkpoint steps after warmup were 42.93s, 45.66s, 47.78s, 46.24s, 47.49s for global batch 96; checkpoint saves added about 192s, 213s, and 397s respectively
resume_smoke: nano-av-miles-fsdp2-medium-small-resume-one-step-from-iter9-20260527T1732Z loaded iter_0000009/model on both ranks, skipped missing optimizer/lr scheduler state, advanced to rollout 9, and finished a no-save step with loss 1.486620 and step_time 56.697s
zero_step_resume_note: --num-rollout 0 is not a valid load-only smoke for this Miles FSDP2 path because the LR scheduler asserts lr_decay_steps > 0 before checkpoint load
```

Throughput implication: retry14 proves the dominant bottleneck is FSDP full
gradient-norm clipping, not forward/backward. Retry15 shows that explicitly
skipping that norm drops the no-save gb8/mb4 step from 114.99s to 30.10s. That
is a major diagnostic win. Larger no-save batches stayed stable up to gb96/mb8
on the 96-row small slice, improving projected train-only epoch time from about
3.9 days at gb8 to about 14.7 hours at gb96 before eval/checkpoint overhead.
Do not launch hero until the skip-grad-norm strategy is validated for stability
and row-specificity, and checkpoint/resume/eval gates pass.

## Small/Medium/Hero Launch

## Config-Driven Runner Spine

The next runner path should use checked-in specs instead of adding more
environment-variable launch variants. Initial specs:

```text
configs/nano_av/small_smoke.yaml
configs/nano_av/medium_small_miles_fsdp2.yaml
configs/nano_av/hero_100k_miles_fsdp2.yaml
configs/nano_av/diagnostics/batch_scaling.yaml
configs/nano_ar/small_smoke.yaml
configs/nano_ar/medium_small_miles_fsdp2.yaml
configs/nano_ar/hero_100k_miles_fsdp2.yaml
configs/nano_ar/diagnostics/resume_smoke.yaml
```

Validation and command rendering live in:

```text
scripts/nano_av_runner.py
scripts/nano_av_materialize_splits.py
scripts/eval_nano_av_miles_checkpoint.py
scripts/verify_nano_miles_ar_dataset.py
scripts/eval_nano_ar_miles_checkpoint.py
```

The materializer writes explicit doc-level `train.parquet`,
`validation.parquet`, `test.parquet`, `train_padded.parquet`, sidecars, and
`split_manifest.json`. Use `train_padded.parquet` for Miles when the spec sets
`final_batch_policy: pad_with_train_duplicates`; the manifest records the exact
duplicate count so one-epoch coverage is auditable.

Example validation:

```bash
python scripts/nano_av_runner.py configs/nano_av/medium_small_miles_fsdp2.yaml
```

AR validation:

```bash
python scripts/nano_av_runner.py configs/nano_ar/medium_small_miles_fsdp2.yaml
```

Example run-plan preparation:

```bash
python scripts/nano_av_runner.py \
  configs/nano_av/medium_small_miles_fsdp2.yaml \
  --prepare \
  --run-id nano-av-miles-fsdp2-medium-small-gb96-mb8-20260527TBD
```

That creates:

```text
<output_root>/<run_id>/run_spec.yaml
<output_root>/<run_id>/run_plan.json
<output_root>/<run_id>/splits/train.parquet
<output_root>/<run_id>/splits/train_padded.parquet
<output_root>/<run_id>/splits/validation.parquet
<output_root>/<run_id>/splits/test.parquet
<output_root>/<run_id>/splits/split_manifest.json
```

The prepared plan computes `num_rollout` from the selected train parquet row
count divided by `global_batch_size`. This is required for Miles FSDP2 SFT:
passing only `--num-epoch 1` can parse successfully but execute zero optimizer
steps on this path.

Legacy shell launchers remain available for reproduction, but new Miles runs
should move toward the spec/materializer/runner path.

Resume smoke note: do not use `--num-rollout 0` for a load-only check. Miles'
FSDP LR scheduler asserts before checkpoint load when `num_rollout == 0`. Use a
one-step no-save resume instead: set `checkpoint.resume_from` in
`configs/nano_av/diagnostics/resume_smoke.yaml`, prepare the plan on RunAI, and
let `training.resume_steps: 1` compute `num_rollout = latest + 1`.

Checkpoint process-group note: full-data `gb192/mb8/save100` reached step 99 and
failed at the first save boundary before writing any shard files:

```text
RuntimeError: NCCL Error 1: unhandled cuda error
NLAFSDPActor.save_model() -> miles/backends/fsdp_utils/checkpoint.py -> dcp.save(...)
torch.distributed.checkpoint utils.py scatter_object_list(...)
```

The incomplete `iter_0000100` directory contained `0` files. Patch
`0006_fsdp_checkpoint_gloo_pg.patch` keeps model/FSDP training on NCCL but
passes a cached Gloo process group to `torch.distributed.checkpoint` so DCP
metadata/object collectives do not use the default NCCL group.

Remediation result: the focused `gb8/mb4` full-optimizer checkpoint smoke passed
on 2026-05-27. It saved a complete `iter_0000001` payload with model shards,
optimizer shards, LR scheduler state, RNG, NLA metadata, and the latest pointer:

```text
/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-checkpoint-debug/nano-av-miles-fsdp2-checkpoint-gloo-fulloptim-gb8-mb4-save1-tokenized-20260527T232920Z/checkpoints/iter_0000001
```

The checkpoint is about `177 GiB`. Save time from Gloo group creation to the
final "Saved checkpoint" log was about `9m23s`; the model-only save was about
`59 GiB` and took about `3m40s`. The corrected resume smoke used
`LOAD_CHECKPOINT=<run_dir>/checkpoints` and loaded model, optimizer, and LR
scheduler before training the next rollout. Do not pass an `iter_XXXXXXX` leaf
as `LOAD_CHECKPOINT`; Miles expects the checkpoint root containing
`latest_checkpointed_iteration.txt`.

Checkpoint eval note: `scripts/eval_nano_av_miles_checkpoint.py` currently
expects a Hugging Face format checkpoint. The medium-small checkpoint is FSDP
DCP format, so convert `iter_0000009/model` first. Nano/Nemotron-H must preserve
the origin remote-code safetensors layout (`backbone.*`, split per-expert
weights). A built-in-HF-style `model.*` / packed-expert conversion loaded with
random-initialized backbone parameters and produced unusable NLL near 13. The
converter now uses the origin `model.safetensors.index.json` layout when it
matches the DCP keys exactly.

```bash
python /workspace/interp/code/nano30b-nla-pilot-current/external/natural_language_autoencoders/tools/convert_fsdp_to_hf.py \
  --input-dir /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-medium/nano-av-miles-fsdp2-medium-small-gb96-mb8-interactive-save3-modelonly-numrollout9-20260527T1745Z/checkpoints/iter_0000009 \
  --origin-hf-dir /workspace/interp/models/nano-30b-a3b-bf16-hf \
  --output-dir /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-medium/nano-av-miles-fsdp2-medium-small-gb96-mb8-interactive-save3-modelonly-numrollout9-20260527T1745Z/hf_iter_0000009

python scripts/eval_nano_av_miles_checkpoint.py \
  --hf-checkpoint /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-medium/nano-av-miles-fsdp2-medium-small-gb96-mb8-interactive-save3-modelonly-numrollout9-20260527T1745Z/hf_iter_0000009 \
  --train-parquet /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-medium/nano-av-miles-fsdp2-medium-small-gb96-mb8-save3-modelonly-numrollout9-20260527T1725Z/splits/train.parquet \
  --validation-parquet /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-medium/nano-av-miles-fsdp2-medium-small-gb96-mb8-save3-modelonly-numrollout9-20260527T1725Z/splits/validation.parquet \
  --test-parquet /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-medium/nano-av-miles-fsdp2-medium-small-gb96-mb8-save3-modelonly-numrollout9-20260527T1725Z/splits/test.parquet \
  --validation-limit 32 \
  --test-limit 32 \
  --generation-examples 0 \
  --report-json /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-medium/nano-av-miles-fsdp2-medium-small-gb96-mb8-interactive-save3-modelonly-numrollout9-20260527T1745Z/eval_iter_0000009_val32_test32.json
```

Small smoke, using the real Miles path and first 96 rows:

```bash
cd /workspace/interp/code/nano30b-nla-pilot-current
INPUT_PARQUET=/workspace/interp/outputs/nano30b-nla-pilot/av-r27-100k-rslora-batch8-epoch1-v1/av-r27-99570-rslora-r192-broad-scale75-lr1e5-b8-e1-epoch-gen8-save-2gpu-offline-20260527T0250Z/av_sft.parquet \
OUTPUT_ROOT=/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-small \
EXPERIMENT_CLASS=small-smoke \
ROW_LIMIT=96 \
GLOBAL_BATCH_SIZE=8 \
MICRO_BATCH_SIZE=1 \
ROLLOUT_BATCH_SIZE=8 \
MAX_STEPS=1 \
SAVE_INTERVAL=1 \
WANDB_MODE=offline \
scripts/run_nano_av_miles_fsdp2_sft.sh
```

For no-save optimizer/throughput diagnostics only, set:

```bash
DISABLE_CHECKPOINT_SAVE=1
NLA_TIMING_DEBUG=1
NLA_SKIP_GRAD_NORM=1   # diagnostic only until medium-small stability is proven
```

The launcher rejects `DISABLE_CHECKPOINT_SAVE=1` for `complete-performance`.

Medium-small config gate:

```bash
python scripts/nano_av_runner.py \
  configs/nano_av/medium_small_miles_fsdp2.yaml \
  --prepare \
  --run-id nano-av-miles-fsdp2-medium-small-gb96-mb8-20260527TBD \
  --report-json /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-medium/medium_small_prepare_20260527TBD.json

python - <<'PY'
import json, subprocess
plan = json.load(open("/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-medium/nano-av-miles-fsdp2-medium-small-gb96-mb8-20260527TBD/run_plan.json"))
print(" ".join(plan["command"]))
subprocess.run(plan["command"], check=True)
PY
```

Hero launch should use the selected faithful batch configuration from the
batch-scaling gate. At current gb384 timing the full 90% train split is about
234 optimizer steps, so a 500-step save interval would not create intermediate
checkpoints. Use a shorter interval for the first hero attempt:

```bash
EXPERIMENT_CLASS=complete-performance
ROW_LIMIT=99570
TRAIN_FRACTION=0.9
VALIDATION_FRACTION=0.05
TEST_FRACTION=0.05
TRAIN_EPOCHS=1
GLOBAL_BATCH_SIZE=384
MICRO_BATCH_SIZE=8
ROLLOUT_BATCH_SIZE=384
SAVE_INTERVAL=100
EVAL_INTERVAL=100
WANDB_MODE=offline
```

## Known Blockers / Next Gates

- Do not launch the hero run until diagnostic artifacts are pruned and explicit
  full-data split parquets are materialized.
- 2026-05-27 storage expansion unblocked checkpoint experiments:
  `/workspace/interp` is 1008 GiB total with about 719 GiB free immediately
  after expansion.
- 2026-05-27 cleanup removed heavyweight diagnostic checkpoint/HF payloads while
  retaining logs and JSON reports. `/workspace/interp` had about 896 GiB free
  afterward.
- Corrected medium-small checkpoint eval passed the row-specificity gate on a
  32 validation / 32 test slice after raw-layout DCP->HF conversion:
  validation real 1.8323 vs shuffled 1.9581, zero 1.9196, mean 1.9478,
  no-injection 2.0705; test real 1.7781 vs shuffled 1.8741, zero 1.8412,
  mean 1.8725, no-injection 1.9756.
- Full optimizer-state checkpoint smoke passed on a 96-row one-step run with
  grad norm enabled. One checkpoint is about 177 GiB:
  model shards about 63 GiB, optimizer shards about 126 GiB, plus LR scheduler,
  RNG, NLA metadata, and run metadata. Resume from that checkpoint loaded model,
  optimizer, and LR scheduler on both ranks and advanced from latest iteration 1
  to training step 1.
- Faithful grad-norm timing is much slower than the diagnostic skip-grad-norm
  path. Global batch 96 took about 157.2s/step for 96 examples. Global batch 192
  took about 197.8s/step for 192 examples. Global batch 384 took about
  186.5s/step for 384 examples, making gb384 the current best throughput probe
  at about 2.06 examples/sec. Peak memory stayed in the same band, about
  118 GiB allocated and 127 GiB reserved per H200 before clearing.
- `clip_grad_norm_` remains the main faithful-path bottleneck: observed raw
  grad norm timings were about 70-92s at gb96, 67-88s at gb192, and 52s at
  gb384. Skipping grad norm is useful for diagnostics but is not Qwen-faithful
  when norms are above the clipping threshold.
- `GLOBAL_BATCH_SIZE=384`, `MICRO_BATCH_SIZE=8`, `ROLLOUT_BATCH_SIZE=384`
  is not hero-stable as-is. The first complete-performance attempt completed
  optimizer step 0, then failed with CUDA OOM on rollout 1 backward after Adam
  state existed.
- Current follow-up candidate: prove `GLOBAL_BATCH_SIZE=192`,
  `MICRO_BATCH_SIZE=8`, `ROLLOUT_BATCH_SIZE=192` with a two-step faithful
  diagnostic before launching the next hero.
- The gb192 two-step diagnostic passed with no CUDA OOM. The active
  complete-performance run is now gb192/mb8/save100:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-hero/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gb192-mb8-save100-20260527T2050Z`.
- Because gb384 gives only about 234 optimizer steps for the 89,618-row train
  split, set `SAVE_INTERVAL` and `EVAL_INTERVAL` around 100 or 125 steps for the
  first complete-performance run. This should produce at least one usable
  intermediate full optimizer checkpoint before the final checkpoint.
- The first complete-performance run was launched detached on 2026-05-27 at
  about 20:32Z:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-hero/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gb384-mb8-save100-20260527T2035Z`.
- The full-model update path is the main bottleneck. Timing retry14 measured
  `clip_grad_norm_` at 68-87s, versus about 25s for forward+backward. Timing
  retry15 proves an explicit `--nla-skip-grad-norm` diagnostic path reduces
  no-save step time to about 30s, but this changes training semantics and should
  remain diagnostic only unless a separate Qwen-faithful replacement is designed.
- `adam_foreach=true` no longer immediately OOMed in the no-save retry, but it
  did not complete a step within several minutes and was stopped. It is not a
  hero-ready optimizer setting.
- Retry8 wrote a model-only checkpoint with `--no-save-optim`; retry10 proved
  model-only load/continue after patching Miles to skip empty optimizer and LR
  scheduler checkpoint dirs. The later full optimizer smoke wrote a complete
  model/optimizer/LR-scheduler/RNG/NLA checkpoint and the resume smoke loaded it
  successfully.
- PVC space is no longer the blocker after expansion, but full optimizer
  checkpoints are large. One 2-rank Nano30B Adam checkpoint is about 177 GiB:
  about 59 GiB model shards plus about 118 GiB Adam optimizer shards. The
  minimum exact-resume policy is `NLA_KEEP_LOCAL=1`, which keeps only the latest
  completed full checkpoint and reduces steady-state checkpoint storage to about
  177 GiB, with a temporary peak around 354 GiB while the next checkpoint is
  being written.
- The launcher records `EVAL_INTERVAL`, but Miles in-process eval is not wired
  for `nla.rollout.sft_actor.generate_rollout` because that rollout asserts
  `evaluation=False`. Use a separate checkpoint eval script until a Miles eval
  function is implemented.
- The current launcher slices the first `ROW_LIMIT` rows for small/medium runs.
  Before hero, materialize explicit train/validation/test parquets from the
  doc-level split so training uses only the train split and eval uses validation
  and test.
- Resume smoke is complete for full optimizer/scheduler state.
- Checkpoint cleanup should be verified with `NLA_KEEP_LOCAL=1` during the hero;
  do not run full data without at least one completed exact-resume checkpoint.
