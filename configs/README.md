# Configs Map

Configs are source-controlled experiment plans. They should not be the only
record of mutable runtime state after launch.

## Layout

| Directory | Purpose |
|---|---|
| `nano_ar/hpo/` | AR HPO, continuation, and hero configs |
| `nano_ar/layer_sweep/` | AR layer-sweep queues/configs |
| `nano_ar/diagnostics/` | AR diagnostic configs |
| `nano_av/hpo/` | AV HPO and hero configs |
| `nano_av/layer_probe/` | AV layer-probe configs and queues |
| `nano_av/diagnostics/` | AV diagnostic configs |
| `nano_rl/` | Config-driven RL probes, guarded confirmations, hero queues, post-eval gates, and retention policies |

Current selected RL config:

`nano_rl/r33_component_corrected_k3_hero_342_resume228_retry3_queue_8h100.yaml`

## Queue State Rule

Checked-in queue YAMLs are plans. Once a queue is launched on RunAI, the remote
queue file, run directory, eval report, and registry entry become the evidence
source. Local queue YAML status can lag.

For decision-changing runs, update:

- `runs/registry/experiments.yaml`,
- the relevant file under `docs/runs/`,
- `docs/experiment_logbook.md` only for long chronology,
- `docs/nano_av_job_tracker.md` for live queue state.

## Artifact Rule

Do not commit:

- checkpoints,
- optimizer shards,
- temporary HF conversions,
- parquets,
- activations,
- W&B payloads,
- large logs.

Do commit:

- configs,
- queue templates,
- verifier code,
- compact eval summaries,
- run registry entries,
- docs that explain decisions.

## Safety Flags

These flags are intentionally explicit because the 2026-06 optimization audit
found correctness risks that should not be silent:

| Flag | Scope | Meaning |
|---|---|---|
| `training.allow_packed_critic_training: true` | Historical AR-SFT configs | Legacy acknowledgement from the pre-2026-06-12 packed-THD critic path. Current AR-SFT critic training uses padded masked microbatches, so new batched configs should not need this flag. |
| `training.allow_oversized_dynamic_batch: true` | AV/AR dynamic batching | Acknowledges `max_tokens_per_gpu < max_sequence_tokens`. This relies on verified Miles behavior that over-budget samples become oversized single-sample microbatches rather than being dropped. |
| `training.grad_norm_policy: clip` | AV/AR Miles FSDP2 | Uses faithful local-shard gradient clipping from the audit remediation. This is the default for production configs. |
| `training.grad_norm_policy: global_clip` | AV/AR Miles FSDP2 | Forces the old global `clip_grad_norm_` path for comparison/debugging. It is expected to be much slower on Nano MoE. |
| `training.system_metrics.router_entropy: true` | AV/AR Nano MoE | Enables router entropy/load telemetry through W&B/system metrics. Use on smokes and probes before changing router losses. |
| `training.timing_debug: false` | complete-performance | Hero/complete-performance runs must keep sync-heavy timing debug disabled. Use timing debug only for probes. |

New Nano runs should keep `NLA_PATCH_NEMOTRON_REMOTE_CODE=1` in their generated
run plan so copied `modeling_nemotron_h.py` files receive the C1/N2/N3/N4 audit
remediations before training. Batched AR-SFT runs should use the padded critic
forward path; packed THD critic training is reserved for explicit diagnostics.

For RL, response-length p95 is telemetry rather than a universal abort
condition. The selected hero retained fail-closed parser, truncation, KL,
gradient, and actor/rollout log-probability guards. Two earlier p95 rules
stopped otherwise healthy runs, first through relative monotonic growth and
then through an absolute 230-token threshold despite zero truncation.
