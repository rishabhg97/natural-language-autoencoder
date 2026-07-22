# R33 Confirmatory Replication Preregistration

Status: **draft; test split sealed; no confirmatory RL launch authorized**

This document freezes the analysis shape for a clean R33 RL replication. It is
not yet the final registration. The deterministic dataset and clean primary AR
are complete, and clean AV has completed optimization, but its bounded
validation report and protocol-matched SFT round-trip baseline are still
pending. The independent clean critic/AR, validation power calculation, and
bounded stability grid are also pending. Finalization changes this status to
`registered`, records all artifact hashes, and occurs before confirmatory RL.
This is a study-plan record, not a separate runtime mode. Registered and
exploratory jobs use the same queue, source contract, guards, and evaluator.

## Claim And Hypothesis

The primary claim is narrow: starting from the deterministic family-clean R33
AV SFT checkpoint, RL improves activation-direction reconstruction from
AV-generated explanations relative to the exact protocol-matched SFT checkpoint.
The primary endpoint is family-clustered paired directional reconstruction gain
under a separately seeded, clean R33 critic. The experiment does not claim raw
activation recovery, R27 superiority, or Qwen equivalence unless their separate
publication gates are later satisfied.

## Sealed Data Policy

- Hyperparameters, guard calibration, checkpoint selection, and stopping
  decisions use training telemetry and validation families only.
- The confirmatory test split remains unopened until one checkpoint and one
  analysis specification are locked.
- Generation uses the same model/tokenizer fingerprints, prompt, stop text,
  deterministic decoding, token budget, controls, and parser for SFT and RL.
- Candidate selection is family-stratified and inference is clustered by content
  family. Row-order first-N selection is forbidden.

## Stability Grid

The bounded clean-SFT grid contains exactly four 16-update runs. All other
parameters are held fixed: actor LR `1e-5`, global batch `384`, actor microbatch
`32`, eight samples per prompt, six actor GPUs, one rollout GPU, one frozen
critic GPU, 256 response tokens, constant LR, and seed `161803`.

| ID | KL estimator | KL coefficient |
|---|---|---:|
| P1 | `k3` | `5e-4` |
| P2 | `k3` | `1e-3` |
| P3 | `k3` | `2e-3` |
| P4 | `low_var_kl` | `1e-3` |

Every probe saves update 16 only and evaluates 256 validation rows with the
full real/shuffled/zero/mean/no-injection control set. A probe is ineligible if
any predeclared guard fires. Among eligible probes, choose the largest
family-clustered validation directional gain; if estimates are within one
bootstrap standard error, choose the lower KL coefficient, then `k3` as the
deterministic tie break. No additional grid points may be added after results
are seen without registering a new experiment family.

Before this RL grid, the selected clean AR and AV recipes each receive one
bounded validation-only neighbor check around their inherited learning rate.
The purpose is to verify that the post-fix clean path has not shifted the local
optimum; these checks do not consume test. AR, AV, and RL must then use one
recorded Mamba kernel profile. If the existing fused-AV/unfused-AR checkpoints
are retained, a validation-only cross-kernel delta report is required instead
of asserting kernel equivalence.

## Guard Policy

The queue must materialize and hash the exact guard policy. A violation aborts
the run; thresholds are not relaxed in place.

- actor/rollout log-prob absolute difference: maximum `0.75` for two
  consecutive updates;
- actor KL loss: maximum `5.0` for two consecutive updates;
- actor gradient norm: maximum `10.0` for two consecutive updates;
- parsed close fraction: minimum `0.95` on every rollout;
- usable explanation fraction: minimum `0.99` on every rollout;
- truncated response fraction: maximum `0.05` on every rollout;
- nonfinite loss, reward, KL, gradient, or CUDA/NCCL failure: immediate abort.

The stability grid is intended to select a recipe that avoids recurrent KL and
gradient excursions, not one that merely survives a higher abort threshold.

## Confirmatory Runs

- Fresh actor/rollout seeds: `271828` and `57721`, launched as two independent
  queue items with separate run identities and checkpoints.
- Planned budget: 342 optimizer updates at global batch 384 and eight samples
  per prompt for each seed, or 131,328 generated samples per run.
- Model-only checkpoints: updates `16`, `64`, `110`, `228`, and `342`.
- Checkpoint selection: maximize the predeclared primary validation endpoint;
  within one bootstrap standard error, choose the earliest checkpoint.
- Stopping: guard failure ends that registered run. A changed recipe or guard is
  a new study, never a resume carrying the same confirmatory identity.
- Administrative retry cap: one relaunch is allowed only when failure occurs
  before the first optimizer update and the frozen launch contract is
  byte-identical. Any failure after update 0 consumes that seed's registration.

## Endpoints And Multiplicity

Primary endpoint: family-clustered paired directional reconstruction gain over
protocol-matched SFT under the independently seeded clean critic, with a
positive 95% interval in each registered seed. The pooled effect is reported as
secondary and does not rescue a failed seed.

Mandatory secondary endpoints are primary-critic directional gain, raw MSE,
centered R2, cosine similarity, vector norm calibration, mean-predictor
comparison, length-matched gain, parse/closure/truncation rates, and transfer
ratio from primary to independent critic. Holm correction is applied across
confirmatory secondary hypotheses. Unadjusted effect sizes and intervals remain
descriptive.

Before launch, a simulation using validation-family paired residuals must show
at least 0.80 power for a 10% relative directional gain at the registered family
count. If power is below 0.80, the confirmatory run does not launch; the remedy
must increase independent family evidence or narrow the claim before a new
registration.

The simulation inputs, code hash, seed, family count, effect grid, and output
JSON must be archived before either confirmatory queue item is launched.
`scripts/nano_clustered_power.py` consumes the paired candidate and SFT
round-trip reports, aggregates residuals by content family, and emits the
hash-bound pass/fail artifact used by the queue.

## Required Telemetry And Evidence

W&B remains offline during execution. Separate actor, rollout, critic, and
system streams must preserve reward distribution, emitted scalar KL metrics,
entropy, policy-gradient and KL-loss components, gradient norms, configured clipping,
response lengths, parse/truncation/repetition rates, throughput, GPU/system
metrics, and checkpoint-eval pointers. The immutable launch contract binds the
source, Miles runtime and patches, environment, data/family manifests, clean SFT
baseline, guard policy, seed, config, and checkpoint schedule by SHA-256.

The queue requires passing, hash-pinned family-seal and kernel-compatibility
reports for every registered run. Confirmatory runs additionally require the
passing clustered-power report; missing or failed reports abort before launch.

Per-token KL quantiles are not emitted by the current Miles loss path and are
therefore not part of the registered telemetry contract. They may be added in
a later study, but their absence cannot be described as missing registered
evidence for this one.

The one-shot test report is admissible only after a selection-lock artifact
records the chosen checkpoint, baseline hash, independent-critic hash,
generation protocol hash, analysis code hash, and multiplicity plan.
