# R33 Hero Handoff - 2026-06-22

Historical status: superseded first by the July 8 internal RL hero and then by
the publication-correctness remediation. Use `docs/current_state.md` for the
active lineage and `docs/runs/r33_publication_preregistration.md` for the
future confirmatory protocol. The July 8 hero document is historical only.

This is the compact handoff for the `2026-06-22` Nano30B NLA state. The
detailed chronology remains in `docs/experiment_logbook.md`.

Last updated with RL smoke state: `2026-06-23`.

## Historical Decision At Handoff

R33 component-full AV+AR SFT is the selected internal hero milestone.

R27 remains the mature fallback and nearest-valid baseline. Do not make an
uncaveated external claim that R33 beats R27 until a fresh row-matched clean R27
comparison is rerun or the nearest-valid caveat is kept attached.

Do not relaunch stale dedup queues. They are superseded by the component-full
hero path.

## Selected Artifacts

Dataset:

- `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_component_fullscan275396`
- Rows: `275,396`
- `d_model=2688`
- Nonfinite activations: `0`
- Empty explanations: `0`
- Materialized split doc/content overlap: `0`

AR hero:

- Run id:
  `nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96`
- Checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001289`
- Eval report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/eval_iter_0001289_v512_t512_winrates_report.json`
- Validation/test teacher NMSE: `0.320616 / 0.292730`
- Validation/test source_raw NMSE: `0.095084 / 0.080078`

AV hero:

- Run id:
  `nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512`
- Checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291`
- Corrected AV eval:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/eval_iter_0001291_v512_t512_gen8_report.json`
- Validation/test real NLL: `0.798672 / 0.819993`
- Validation/test shuffled NLL: `1.331095 / 1.361868`
- Validation/test zero NLL: `1.167483 / 1.196865`
- Validation/test mean NLL: `1.241662 / 1.287035`
- Validation/test no-injection NLL: `1.224772 / 1.259839`

Round-trip gate:

- Report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/roundtrip_iter_0001291_v256_t256_report.json`
- Gate passed: `true`
- `baseline_required=false`
- Validation/test AV-real NMSE: `0.000109680 / 0.000121664`
- Validation/test teacher-text NMSE: `0.000106810 / 0.000112370`
- Parse closed/usable fraction: `1.0 / 1.0`

## R33 vs R27 Read

Nearest-valid R27 `256/256` full-control baseline:

- Report:
  `/workspace/interp/outputs/nano30b-nla-pilot/roundtrip/r27_baseline/r27_roundtrip_v256_t256_full_controls_prefix256_report.json`
- Validation/test AV-real NMSE: `0.000180003 / 0.000175571`
- Validation/test teacher-text NMSE: `0.000157706 / 0.000155285`
- Parse closed/usable fraction: `1.0 / 1.0`

Comparison:

- R33 validation AV-real NMSE is `39.1%` lower than the nearest-valid R27
  report.
- R33 test AV-real NMSE is `30.7%` lower than the nearest-valid R27 report.
- Caveat: this is not a fresh row-identical R27 rerun with preserved post-fix
  R27 checkpoints.

## Cleanup State

RunAI cleanup on `2026-06-22T15:26:15Z`:

- Deleted superseded R33 dedup AR checkpoint payload: `72G`.
- Deleted superseded R33 dedup AV checkpoint payload: `59G`.
- Preserved selected component-full AR and AV checkpoints.
- Cancelled stale dedup AR queue items and stale blocked throughput-ladder
  items.
- `/workspace/interp` improved from `622G used / 386G free` to
  `492G used / 516G free`.

Cleanup manifest:

- `/workspace/interp/outputs/nano30b-nla-pilot/cleanup_manifests/20260622T152615Z_posthero_workspace_cleanup.txt`

## Verification Snapshot

Local dependency-light focused shard:

```bash
.venv/bin/python -m pytest \
  tests/test_nano_av_generation.py \
  tests/test_nano_av_ar_roundtrip_gate.py \
  tests/test_nano_roundtrip_eval_config.py \
  tests/test_nano_roundtrip_queue.py \
  tests/test_nano_av_probe_queue.py \
  tests/test_nano_ar_hpo_queue.py \
  tests/test_nano_ar_hpo_study.py \
  tests/test_nano_queue_gate.py \
  tests/test_nano_miles_launcher.py -q
```

Result: `78 passed`.

RunAI full focused shard:

```bash
CUDA_VISIBLE_DEVICES= /workspace/interp/.venv/bin/python -m pytest \
  tests/test_nano_av_generation.py \
  tests/test_nano_av_ar_roundtrip_gate.py \
  tests/test_nano_roundtrip_eval_config.py \
  tests/test_nano_roundtrip_queue.py \
  tests/test_nano_av_probe_queue.py \
  tests/test_nano_ar_hpo_queue.py \
  tests/test_nano_ar_hpo_study.py \
  tests/test_nano_queue_gate.py \
  tests/test_nano_critic_model_arch.py \
  tests/test_nano_miles_launcher.py -q
```

Result: `88 passed, 5 warnings`.

Current RunAI health after cleanup:

- Workspace: `train`, running.
- GPUs: 2x H200 idle, `4 MiB` used each.
- No active Nano train/eval/round-trip/RL process.
- `/workspace/interp`: `492G` used, `516G` free.

## Next Recommended Step

Choose one of two paths:

1. Fresh R27 comparison path:
   rerun a row-matched clean R27 AV+AR comparison if the next output needs an
   external R33-over-R27 claim.
2. RL path:
   use the now-proven 4-H200 actor/rollout/critic topology for a larger
   no-interval-checkpoint R33 RL quality smoke with frozen AR reward and
   post-run round-trip/AR reconstruction evals.

If the goal is research momentum rather than external proof language, take the
RL path next. If the goal is publication/audit defensibility, run the fresh R27
comparison first.

## RL Smoke Update - 2026-06-23

The R33 SFT hero gate remains the clean quality baseline. The first 4-H200 RL
systems smoke is partially unblocked, but there is not yet a quality-bearing RL
result.

Completed:

- Queue:
  `configs/nano_rl/r33_component_full_smoke_queue_4h200_len256_rb2_fix2_freezecritic.yaml`
- Run dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_smoke/r33_component_full_sft_init_512row_4h200_len256_rb2_fix2_freezecritic`
- Topology:
  `2` actor H200s, `1` rollout/SGLang H200, `1` critic H200.
- Caveats:
  `NLA_SKIP_ROLLOUT_WEIGHT_SYNC=1` and `NLA_FREEZE_CRITIC_TRAIN=1`.
- Result:
  rollout `4/4`; actor step `0` completed; reward/train MSE equivalence
  `mean=1.0000`, `max|r-1|=0.0000`, `n=4`; actor checkpoint saved at
  `actor/iter_0000001`.
- Snapshot scalars:
  raw reward `-0.280732`, shaped reward `5.349516868591309e-06`, actor loss
  `-5.304813385009766e-06`, grad norm `9.3125`.

Superseded failed live-sync attempts:

- Live actor-to-rollout SGLang weight sync failed in the no-save-optim pilot
  after the first `backbone.embeddings.weight` bucket metadata transfer.
- The barrier-fix retry is stale in queue state but logs show SIGTERM/SGLang
  healthcheck timeouts.
- The unified-env retry failed with `ModuleNotFoundError: No module named
  'accelerate'` during Miles FSDP actor init.

Completed live-sync follow-up:

- Queue:
  `configs/nano_rl/r33_component_full_pilot_queue_4h200_len512_rb2_sync2_nosaveoptim_unifiedenv_mambawheels_tokcompat_nopackedcheck_criticfwd_evalmode_nofastpath_timesteplimit_allocseg.yaml`
- Run dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_pilot/r33_component_full_sft_init_512row_4h200_len512_rb2_sync2_nosaveoptim_unifiedenv_mambawheels_tokcompat_nopackedcheck_criticfwd_evalmode_nofastpath_timesteplimit_allocseg`
- Runtime:
  unified SGLang/Torch env, Mamba wheels, TokenizersBackend fallback,
  indexed/eval-mode AR reward forward, Mamba fast-path disabled only for
  critic reward scoring, `time_step_limit` sentinel normalization,
  `NLA_FREEZE_CRITIC_TRAIN=1`, and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Result:
  completed `2026-06-23T20:26:56Z`; two live-sync rollout/update cycles;
  `4/4` generations per rollout; actor checkpoint saved at
  `actor/iter_0000002`.
- Caveat:
  still a systems smoke, not a quality-bearing RL result. Rollout 1 actor
  training peaked near `143.1 GiB` on each actor H200, and interval
  checkpointing wrote about `63GB` of actor model shards.

Next RL work should run a larger no-interval-checkpoint quality smoke with
allocator segmentation retained, then evaluate round-trip/AR reconstruction
before any hero-scale RL run.
