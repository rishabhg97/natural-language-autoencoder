# R33 Clean Online Joint AV+AR Canary

Date: 2026-07-17 UTC

## Decision

The first clean-lineage online actor+critic canary is a systems success and a
quality non-promotion. Both AV and AR received two online optimizer updates,
checkpointed, and passed exact generated-text evaluation plumbing. The strict
paired gate does not show improvement over the clean SFT pair, so SFT remains
the canonical checkpoint.

## Training Contract

- Run: `r33-family-clean-online-joint-canary-update2-8h100-retry4`.
- Hardware: 8 H100 NVLs (`4 actor / 3 critic / 1 SGLang rollout`).
- Initialization: qualified family-clean R33 AV and AR SFT checkpoints.
- Rollout: 3 prompts x 8 samples = 24 samples per update.
- Updates: 2.
- Actor/critic LR: `1e-5 / 5e-6`, constant schedule.
- W&B: offline actor and critic logs.
- Critic step-0 reward/train equivalence: exact pass.

Update 1 had 24/24 usable rollouts and reward mean/std
`-0.374729 / 0.100902`. Update 2 had 21/24 usable rollouts and reward mean/std
`-0.553841 / 0.587261`, with minimum reward `-2.0`. The actor and critic DCP
checkpoints at update 2 are complete; ignored W&B atexit pipe errors occurred
only after checkpoint commit.

## Evaluation Contract

The final comparison uses 64 family-stratified validation rows, 64 independent
families, and the same rows in the same order for online and SFT. Dataset
hashes, tokenizer identity, generation settings, and protocol SHA-256 match
exactly. Generation is deterministic with a 256-token budget and five arms:
real, shuffled, zero, mean injection, and no injection. All 64 rows are closed
and usable in every arm.

The evaluator now aligns paired reports by canonical `row_keys`. Numeric row
indices remain only as a legacy fallback. A report-only re-gater binds the
candidate/baseline report hashes and declared thresholds without rerunning
models.

## Results

| Metric | Online update 2 | Clean SFT | Online effect |
|---|---:|---:|---:|
| Directional MSE | `0.291993` | `0.292173` | `0.0618%` lower |
| Raw MSE | `8.969927` | `8.797533` | `1.96%` worse |
| Centered raw R2 | `-0.238289` | `-0.214490` | worse |
| Norm ratio mean | `1.504606` | `1.494323` | farther from 1 |
| Directional paired wins | `32/64` | `32/64` | tied |
| Raw paired wins | `28/64` | `36/64` | worse |

Directional family-bootstrap 95% CI is
`[-0.007704, 0.008597]`; one-sided sign-flip `p=0.4824`. Raw paired mean delta
(baseline minus online) is `-0.172394`, with one-sided `p=0.9597`.

Online real text still beats all controls: directional MSE is `0.805388`
for AV-mean, `0.800908` for no injection, `0.959201` for shuffled, `0.953550`
for zero, and `0.641089` for activation mean. Rowwise real-control win rates
are `98.44%` to `100%`.

## Promotion Gate

The exploratory aggregate-only gate passes because it required only a
nonnegative mean directional delta. It is retained but is not the decision
artifact. The strict policy requires:

- at least 60% paired wins;
- at least 10% relative directional improvement;
- positive family-bootstrap lower bound;
- clustered family inference;
- exact dataset, row, and generation protocol identity;
- parse and control gates.

The strict report passes provenance, family, parse, and control checks but
fails SFT improvement. Final decision: `passed: false`.

## Evidence

- Local:
  `artifacts/runai_eval/r33-online-joint-canary-evidence-20260717T0951Z/`.
- S3 archive:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/run-artifacts/r33-online-joint-canary-evidence-20260717T0951Z.tgz`.
- Archive SHA-256:
  `3a721f6bbd795b4aeba4a801164594c982807d8eee2160ad4ab0e484e52efc83`.
- Strict report SHA-256:
  `8c34ec3b527651c32bcfc7ede749e12f666f7206b95e68c02264ad9addee65b4`.

## Next Experiment

Do not scale this checkpoint directly. The next bounded online probe should
increase effective rollout count, reduce reward variance, retain explicit
KL/SFT anchoring, account for parser failures in reward, and track directional
and raw reconstruction deltas during training. Promotion must remain based on
the strict held-out paired gate, not rollout reward.
