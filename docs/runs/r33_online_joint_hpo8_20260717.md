# R33 Clean Online Joint AV+AR HPO8

Date: 2026-07-17 UTC

## Objective

Run a bounded, validation-only online joint AV+AR search from the qualified
family-clean R33 SFT pair. The search metric is the paired generated-text
round-trip effect against the clean SFT baseline, not rollout reward or AV NLL.
The test split remains unopened.

## Canary Finding And Fix

The two-update clean canary exposed an asymmetric data-parallel repartition
bug. Four actor ranks produced four local shards, while three critic ranks
received actor shards in a `2 / 1 / 1` pattern. A later cross-rank truncation
reduced every critic rank to the smallest local count. The critic therefore
trained on only `18/24` rows at update 1 and `12/21` usable rows at update 2,
even though the actor trained on the full rollout batch.

The runtime now reconstructs the globally indexed actor batch on every critic
rank, validates exact coverage and field lengths, filters rows without critic
tokens once globally, aligns once to `critic_dp * micro_batch`, and distributes
the retained rows evenly. It records generated, usable, retained, parse-drop,
alignment-drop, and per-rank counts in stdout and W&B. Further local
truncation is an assertion failure. Online runs also enforce a configurable
minimum retained fraction through
`training.min_critic_retained_fraction`.

Queue preflight now reports critic divisibility in the immutable launch spec.
The HPO shape is exact when all rows parse: 192 generated, 192 actor-trained,
and 192 critic-retained rows (`3 critic ranks * microbatch 2`). Focused local
and RunAI tests pass (`100 passed` before the guard-observability revision).

## HPO Matrix

All candidates use 24 prompts, eight samples per prompt, global batch 192,
microbatch 2, eight optimizer updates, constant LR, K3 KL loss, 256 response
tokens, 512 context tokens, four actor GPUs, three critic GPUs, and one SGLang
GPU. W&B is offline.

| Candidate | Actor LR | Critic LR | K3 coefficient | Purpose |
|---|---:|---:|---:|---|
| H1 | `1e-5` | `2e-6` | `3e-3` | primary candidate |
| H2 | `5e-6` | `2e-6` | `3e-3` | lower actor LR |
| H3 | `1e-5` | `5e-6` | `3e-3` | higher critic LR |
| H4 | `1e-5` | `2e-6` | `1e-3` | weaker SFT/KL anchor |

Only H1 is approved. H2-H4 remain pending and cannot launch without explicit
approval. Storage policy permits one final actor/critic checkpoint pair at a
time; no optimizer state is saved.

## Evaluation Contract

Each candidate is evaluated on 128 family-stratified validation rows with
selection seed `20260709`. These rows are an exact prefix of the qualified
512-row clean-SFT report, enabling paired row-identity comparison. Evaluation
uses deterministic 256-token generation, real/shuffled/zero/mean/no-injection
controls, 100% local model loading, family-clustered bootstrap inference, and
generation-protocol and dataset-provenance matching. Temporary actor HF output
lives under `/dev/shm` and is removed after evaluation.

## Launch Record

The first H1 launch was stopped before rollout or any optimizer update. Its
rendered Miles command revealed that `rl.sh` did not forward the queue's critic
retention threshold or JSON train/rollout guards into Ray worker environments.
It also loaded the actor reference directly from network model storage, making
four ranks fault through the 13-shard model independently. The attempt is
preserved as
`hpo8_h1_a1e5_c2e6_k3e3_preflight_abort_20260717T1240Z`; it is not a run result
and contains no checkpoint.

The launcher now forwards all three runtime contracts. Actor-reference input
staging and SGLang model staging share one verified `/dev/shm` target, so the
first corrected launch copies the HF model once and later candidates reuse it.
H1 was relaunched from the unchanged qualified SFT initialization. It generated
all 192 rollout samples, then stopped before optimizer update 0 because the
startup close rate was `0.8854167` while the queue required `0.95` on the first
batch. This was a parser guard stop, not an OOM, worker crash, or optimizer
failure. No checkpoint was written and all eight GPUs returned idle.

The guard was calibrated against the clean initialization rather than simply
removed. A CPU tokenizer audit over the first 120,000 family-clean SFT target
responses found token-length p50/p90/p95/p99 `119 / 147 / 156 / 174`, maximum
`233`, and zero targets above 256 tokens. The 256-token rollout budget is
therefore retained: increasing it would primarily permit malformed generations
to ramble rather than cover the teacher distribution. Close and usable guards
now require at least `0.80` for two consecutive batches; raw generation
truncation may not exceed `0.20` for two consecutive batches. Combined with
the `0.95` retention-of-usable invariant, this bounds critic training to at
least about 76% of generated rows while allowing the failed-extraction reward
to improve formatting from the clean SFT start.

The stop also exposed an observability bug: `nla_generate` relabeled truncated
generations as `FAILED` before rollout metrics ran, so the historical
post-processing `truncated_frac` could be zero even for engine truncations.
The raw engine status is now preserved separately as
`rollout/nla_generation/*`, the queue guards the raw truncation metric, and the
compact `[NLA ROLLOUT]` payload is printed before a guard may raise. The focused
suite passes (`110 passed`). The failed H1 directory is immutable; the corrected
candidate uses fresh `...h1...retry1` run and W&B identities.

Retry 1 passed the recalibrated parser guards. Its first 192-sample rollout was
`0.916667` closed, `0.911458` usable, and `0.057292` raw-truncated, with reward
mean/std `-0.516923 / 0.503036` and response-length p95 exactly 256. It then
stopped before update 0 at the actor packed-equivalence gate: packed/padded
mean NLL was `1.379626 / 1.356946`, while the global maximum absolute/relative
sample drift was `0.103714 / 0.053270`. No optimizer step or checkpoint exists.

This is a real batching-correctness finding. Miles' default FSDP `thd` path
concatenates a microbatch into one sequence and relies on patched position-reset
handling for attention and Mamba boundaries. The observed drift is too large
to dismiss for a publication-oriented run. Retry 2 therefore uses Miles'
existing padded `bshd` data and loss path consistently for reference log-probs,
actor log-probs, and actor training. The NLA FSDP wrapper now supplies a
length-derived two-dimensional attention mask and validates the padded batch
layout. A checked Miles compatibility patch permits its already-implemented
`bshd` path for FSDP; the packed actor gate is disabled only because the actor
is no longer packed. Critic equivalence and critic row-retention gates remain
enabled. Focused tests pass (`117 passed`).

Retry 2 first failed preflight because its runtime contract expected a
single-line Miles assertion while the checked patch formatted the same
assertion over multiple lines. The contract now checks stable semantic
fragments. A corrected retry-2 launch was then stopped during initialization
after a generic Miles warning suggested that no checkpoint would be saved.
Inspection proved the warning inapplicable: this queue uses the checked
`NLA_SAVE_ITERATIONS=8` override in `nla.save_schedule`. The launcher now emits
`[NLA SAVE SCHEDULE] explicit_iterations=8 miles_save_interval=disabled`.
Neither retry-2 attempt generated a rollout or performed an optimizer update.

Retry 3 completed a full padded actor optimizer update. Rollout 0 was
`0.947917` closed, `0.9375` usable, and `0.046875` raw-truncated. Actor metrics
were loss `0.004039`, K3 loss `0.002306`, gradient norm `4.3125`, policy clip
fraction `0.008266`, and train-vs-rollout log-prob drift `0.027766`. It then
failed before critic optimization because Miles replicates the global
`raw_reward` vector on every actor shard, while the first repartitioner treated
every list as row-local. The generic repartitioner now validates and preserves
replicated global list fields. No checkpoint was written.

Retry 4 proved that repair on live data. Of 192 generated rows, 168 were usable;
all 168 were retained with zero alignment loss and split evenly as 56 rows per
critic rank. The step-0 reward-vs-training MSE ratio was exactly `1.0000` with
zero reported maximum deviation. Critic backward then failed before its first
microbatch because the critic-token rewrite removed Miles' `bshd`
`max_seq_lens` metadata. The critic view now rebuilds padded widths from the
rewritten token rows using the configured padding multiple. Combined focused
tests pass (`110 passed`).

Retry 5 uses a fresh immutable run/W&B identity. It is the first attempt to
complete joint actor and critic optimization through update 0, and the first
four joint updates completed without a guard, OOM, or numerical failure.
Rollouts 0 through 3 had reward means
`-0.514659 / -0.413494 / -0.499352 / -0.375198`, close fractions
`0.9323 / 0.9427 / 0.9375 / 0.9792`, usable fractions
`0.9219 / 0.9323 / 0.9271 / 0.9531`, and raw truncation fractions
`0.0729 / 0.0469 / 0.0521 / 0.0156`. Because each update samples a different
prompt batch and failed parses receive reward `-2`, these reward means are
diagnostics rather than a validation trajectory.

Across the first four updates, critic training FVE remained positive in
`[0.3363, 0.4866]` with gradient norm in `[0.6719, 0.7695]`. Actor policy clip
fraction fell from `0.00788` to `0.00408`, train-vs-rollout log-prob drift fell
from `0.02820` to `0.02134`, and update-3 gradient norm returned to `0.7656`
after two batch-specific clipped gradients. The K3 term contributed about
`0.6-2.4%` of total actor loss over these updates, so the lower-K3 H4 candidate
is not currently the highest-information follow-up. These observations are
provisional; no HPO or quality conclusion is allowed until update 8 and the
paired validation report exist.

Steady-state elapsed time was about `447 s` per 192-sample update. Actor and
critic training overlap, but observed H100-NVL memory reached about
`90.5 / 95.8 GiB` for actor ranks and `92.6 / 95.8 GiB` for critic ranks, so
larger microbatches are not a safe H100 lever. The superseded, non-promoted
two-update canary model/HF payloads were removed after confirming its reports,
logs, predictions, and S3 evidence archive; free PVC space rose from `158 GiB`
to about `288 GiB` before the final H1 save.

The shared rollout logger and `scripts/analyze_nla_rl_run.py` now have
observability-only additions for usable-only reward statistics plus compact,
role-aware actor, critic, rollout, and timing trajectories. Focused tests pass
(`9 passed`). These changes do not affect the active retry-5 process and must
be synced before a later candidate is launched.

## 2026-07-17 - Retry 5 Completion and Corrected Paired Round-Trip Result

Retry 5 completed all eight joint optimizer updates from the qualified
family-clean SFT pair. Its canonical validation result is the corrected
128-row, 128-independent-family protocol, not the earlier `v128_t64`
diagnostic whose generation protocol did not match the baseline.

- The exact matched protocol hash is
  `e5e3a2658d28975514dd962be18c149012ee1fc85f1d6f52ccc834f59c95d416`.
  It has identical row keys, dataset hashes, controls, seed, generation budget,
  and tokenizer identity for baseline and candidate.
- The generated H1 file contains 128 rows. The primary real path has closed and
  usable parse fractions of `1.0`; all required real-vs-control checks pass.
- Clean SFT baseline directional MSE: `0.3028433237`. Joint H1 directional MSE:
  `0.3012607509`, a nominal `0.0015825728` (`0.5226%`) improvement. This does
  not pass the paired gate: family-clustered 95% CI
  `[-0.0040094576, 0.0072551257]`, sign-flip `p=0.296687`, and H1 wins exactly
  `50%` of rows.
- Decomposition makes the mechanism clear. H1 AV text with the untouched clean
  AR critic yields `0.3012762239`, nearly identical to joint H1. Clean SFT AV
  text through the H1 critic yields `0.3028854620`, effectively flat/slightly
  worse than baseline. The eight-update nominal movement is therefore actor-side,
  not evidence for increasing critic LR.
- The strict independent regate reproduces `gate_passed=false` and
  `publication_status=confirmatory`. H1 is a valid bounded online-RL pilot, not
  a promotion or hero-scale candidate.
- Evidence is present locally under
  `artifacts/runai_eval/r33-online-joint-hpo8-h1-protocolfixed-20260717/`
  (73 lightweight files; checkpoint-free), and in S3 at
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/evidence-sync/20260717_r33_online_joint_hpo8_h1_retry5/r33_online_joint_hpo8_h1_retry5_evidence_20260717.tgz`.
  The locally verified archive SHA-256 is
  `bc44dfef896f63eccf68949eec7a1bc392b0a03f1e4e694489c1c13037345809`.

No H2-H4 or larger online-RL run was launched from this result. If another
bounded HPO trial is approved, actor learning rate and/or update budget is the
more informative axis; critic-LR escalation is deprioritized by the critic-only
decomposition.

## Canonical Files

- Queue: `configs/nano_rl/hpo/r33_family_clean_online_joint_hpo8_queue_8h100.yaml`
- Repartition module:
  `external/natural_language_autoencoders/nla/critic_repartition.py`
- FSDP integration:
  `external/natural_language_autoencoders/nla/train_actor.py`
- Launcher: `external/natural_language_autoencoders/configs/rl.sh`
- Run root:
  `/workspace/interp/outputs/nano30b-nla-pilot/publication/r33_online_rl/`
