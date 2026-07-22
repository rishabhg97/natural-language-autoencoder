# R33 Validity-First Fixed-AR Hero Design

## Status

Approved direction: validate the existing R33 RL gain independently, correct
the fixed-AR RL mechanics, and only then scale the validated recipe. A direct
port of simultaneous AV+AR training is deferred until it can be adapted to
Nano30B's hybrid architecture and available GPU topology.

## Context

The clean R33 SFT baseline has validation/test AV-to-AR normalized MSE of
approximately `1.0968e-4 / 1.2166e-4`. The 32-update fixed-AR RL run improves
the matched rows by approximately `16.8% / 19.3%`, reaches FVE of approximately
`0.636 / 0.609`, beats all existing activation controls, and preserves high
parse health.

That result is promising but not independently conclusive:

- RL reward and final round-trip evaluation use the same frozen R33 AR.
- The current shuffled control is globally sampled and can be too easy.
- The evaluator does not test whether reconstructed activations preserve the
  target model's output behavior.
- Meaning-preserving transformations are not tested, so AR-specific shorthand
  or formatting codes remain possible.
- Miles currently defaults to signed `k1` KL. The 32-update run develops
  negative KL loss and large train-versus-rollout log-probability drift.
- The configured global batch of 480 is truncated to 384 because it is not
  divisible by six actor ranks times microbatch 32.
- The RL smoke dataset contains only 512 rows, so the medium run revisits a
  small prompt pool.

The official NLA reference trains AV and AR together, uses group size 8, caps
rollouts at 150 tokens, and evaluates downstream behavior and semantic
transformations. Those choices are evidence, not a drop-in configuration for
Nano30B. This design first establishes a trustworthy fixed-AR result, then
uses it as the base for architecture-aware joint training later.

## Goals

1. Determine whether the existing R33 RL improvement survives independent,
   model-grounded evaluation.
2. Remove known KL, batching, and length-drift risks from fixed-AR RL.
3. Select a stable short-run configuration using round-trip, functional, and
   language-quality evidence.
4. Scale that configuration on the full clean R33 training split through
   bounded milestones.
5. Preserve reproducibility, lightweight local evidence, and enough RunAI
   storage headroom to avoid Longhorn pressure.

## Non-Goals

- Do not train the AR jointly with the AV in this program.
- Do not generate a replacement 275k teacher-label dataset.
- Do not start a large run from the current signed-`k1` configuration.
- Do not promote a checkpoint using AV NLL or same-AR round-trip NMSE alone.
- Do not retain every HPO or milestone checkpoint.

## Program Overview

The program has three ordered stages:

1. **Validity evaluation:** compare SFT, 16-update RL, and 32-update RL without
   additional training.
2. **Corrected fixed-AR RL:** run bounded mechanics and learning-rate probes,
   then a 32-update confirmation.
3. **Fixed-AR hero scaling:** train on the full clean R33 train split through
   64, 128, and 256-update gates.

Stage 3 is blocked until the selected Stage 2 checkpoint passes every hard
gate in this document.

## Component Boundaries

### Evaluation Core

Reusable evaluation helpers will own:

- normalized activation metrics;
- paired row alignment;
- deterministic paired bootstrap confidence intervals;
- semantic transformations;
- functional activation patching metrics;
- generation-health metrics; and
- promotion-gate evaluation.

Command-line entry points and queue code will only load configuration, invoke
these helpers, write reports, and return a meaningful exit code. Nano-specific
model loading and R33 boundary hooks stay outside the vendored generic NLA
package.

### Functional Recovery Harness

The functional evaluator consumes:

- an AV-generated explanation JSONL;
- the selected AR checkpoint;
- a source base parquet containing exact `token_ids_prefix` provenance;
- the Nano30B target checkpoint; and
- a YAML evaluation configuration.

For every row it will:

1. Resolve the source row by stable provenance keys.
2. Recompute the stored R33 activation from `token_ids_prefix`.
3. Fail the row if recomputation exceeds the extraction-identity tolerances:
   relative L2 `1e-2`, max absolute error `1e-2`, or one-minus-cosine `1e-4`.
4. Reconstruct `h_hat` from the explanation with the AR.
5. Normalize `h_hat` and rescale it to the original activation norm.
6. Run Nano30B with a forward hook that replaces only the final-token R33
   boundary activation while leaving every earlier token unchanged.
7. Compare patched next-token logits with the unmodified model.

An identity canary patches the recomputed original activation back into the
same boundary. Its logits must match the unmodified forward pass within the
configured floating-point tolerance before candidate scores are trusted.

Reported functional metrics are:

- `KL(original || patched)`;
- Jensen-Shannon divergence;
- Pearson correlation over logits;
- top-10 and top-50 token-set overlap; and
- original top-1 token rank under the patched logits.

Controls use teacher explanation, SFT AV explanation, shuffled activation,
train mean activation, and zero activation. Within-document shuffle is required
for every row whose document contributes at least two eligible positions; the
report records eligible and ineligible row counts rather than silently falling
back to global shuffle.

### Semantic-Invariance Harness

The same generated explanations are scored after:

- sentence and bullet reordering;
- formatting and whitespace normalization; and
- light paraphrasing supplied through a versioned JSONL transform interface.

The deterministic transforms are required in Stage 1. Before Stage 3, light
paraphrases will be generated with a fixed local model and prompt, and 50
stratified examples will be manually audited for meaning preservation. The
transform JSONL records source hash, transform type, model, prompt hash, seed,
and transformed text.

The evaluator must never silently fall back to the original explanation when
a requested transform is absent or malformed.

### RL Objective Configuration

All adjustable behavior is expressed in YAML and rendered by the existing RL
queue. The launcher will expose and validate:

- KL estimator type;
- KL coefficient;
- actor and critic learning rates;
- group size;
- prompt batch, generated batch, global batch, and actor microbatch;
- response and context caps;
- reward transform;
- failed-parse reward;
- policy-drift abort thresholds; and
- checkpoint retention policy.

The queue must reject a run unless all three conditions hold:

```text
generated_samples == rollout_batch_size * n_samples_per_prompt
global_batch_size == generated_samples
global_batch_size % (actor_data_parallel_size * actor_micro_batch) == 0
```

Actor code may retain defensive alignment checks, but a production run must
not depend on silent sample truncation.

## Stage 1: Independent Validity Evaluation

### Candidates

Evaluate these checkpoints on identical held-out rows:

1. clean R33 AV-SFT;
2. the 16-update `lr=2e-5, KL=1e-3` confirmation; and
3. the 32-update `lr=2e-5, KL=1e-3` medium checkpoint.

Use 512 validation and 512 test rows for primary metrics. Use a fixed
stratified qualitative panel of at least 50 rows covering document type,
token position, activation norm, and explanation length.

### Required Reports

Each candidate receives:

- the existing real-versus-control round-trip report;
- a semantic-invariance report;
- a functional recovery report;
- generation-health statistics;
- paired comparisons against SFT; and
- a manifest containing checkpoint, dataset, code revision, configuration,
  and row identities.

### Stage 1 Interpretation

The existing 32-update checkpoint is considered independently validated only
if all of the following hold on validation and test:

- round-trip NMSE is at least 5% lower than SFT on matched rows;
- the paired bootstrap 95% confidence interval for `SFT NMSE - candidate NMSE`
  is strictly above zero;
- mean functional KL is lower than SFT and its paired bootstrap 95% confidence
  interval for `SFT KL - candidate KL` is strictly above zero;
- top-10 and top-50 overlap do not regress by more than one percentage point;
- deterministic semantic transforms retain at least 90% of untransformed FVE;
- real explanations beat every activation control on at least 90% of rows;
- usable parse fraction is at least 0.99 and closed fraction at least 0.95;
- no injection-marker or CJK leakage is observed; and
- no more than 5% of the qualitative panel is flagged for repeated
  encoded-looking strings or a clear readability regression relative to SFT.

Failure does not invalidate the SFT checkpoint. It identifies which evidence
or objective must be repaired before fixed-AR RL can scale.

## Stage 2: Corrected Fixed-AR RL

### Mechanical Corrections

Before new training:

1. Expose Miles `--kl-loss-type` through the shell launcher and YAML queue.
2. Use pointwise non-negative `k3` KL for new probes.
3. Add strict batch-divisibility preflight and report the effective trained
   batch explicitly.
4. Keep the current `-MSE` reward so the KL and batching corrections are not
   confounded with reward-shape changes.
5. Audit existing generation lengths and choose the smallest tested cap that
   closes at least 95% of rows on both held-out splits. Test 150, 192, 224,
   and 256 tokens; retain 256 when no smaller cap meets the threshold.
6. Preserve a failed or truncated generation penalty that cannot outrank any
   valid orthogonal reconstruction.

### Probe Matrix

Use the known `6 actor + 1 critic + 1 rollout` topology and a divisible batch:

```text
n_samples_per_prompt: 8
rollout_batch_size: 48
global_batch_size: 384
actor_micro_batch: 32
KL type: k3
KL coefficient: 1e-3
updates: 8
```

Run two actor learning rates:

- `1e-5`; and
- `2e-5`.

Evaluate both at 64 validation and 64 test rows with all controls and the
functional evaluator. Promote at most one to a 32-update confirmation with
256 validation and 256 test rows. The confirmation saves checkpoints at
updates 8, 16, 24, and 32 so early stopping can select a non-final checkpoint.

### Runtime Stability Guards

Abort further optimizer steps and preserve the latest completed checkpoint if
either condition persists for two consecutive updates:

- mean absolute train-versus-rollout log-probability difference exceeds
  `0.75`; or
- non-finite reward, advantage, loss, KL, gradient, or model parameters are
  observed.

Also fail promotion when:

- response-cap hit rate exceeds 2%;
- usable parse fraction falls below 0.99;
- entropy falls more than 25% below the mean of the first four updates; or
- the effective trained batch differs from the configured global batch.

The Stage 2 winner must pass the full Stage 1 hard gates before Stage 3 is
unblocked.

## Stage 3: Fixed-AR Hero Scaling

### Dataset

Build an RL parquet from the verified R33 component-full base dataset and its
existing document-disjoint split manifest. It requires activation vectors,
prompt, stable provenance, and exact token-prefix provenance; it does not
require new teacher explanations.

Only the clean training split is eligible for RL. Validation and test document
IDs are hard exclusions. The verifier must report:

- row and unique-document counts;
- activation dimension 2688;
- zero non-finite vectors;
- zero train, validation, and test document overlap;
- zero content-component overlap when component IDs are available; and
- source manifest and hash lineage.

### Topology Canary

Benchmark two one-update layouts before the long run:

1. known-safe `6 actor + 1 critic + 1 rollout`; and
2. candidate `5 actor + 1 critic + 2 independent rollout engines`.

Both use a global batch divisible by actor data parallelism and microbatch.
The second layout is selected only if it fits without OOM, trains every
generated sample, and improves end-to-end samples per second by at least 15%.
Otherwise the known-safe layout remains the hero topology.

### Scale Ladder

Run one uninterrupted job capped at 256 updates, with model-only checkpoints
at cumulative updates 64, 128, and 256. Online drift guards can stop the job
before the cap without resetting optimizer state. After the job stops, evaluate
the saved checkpoints sequentially. For each checkpoint:

1. convert only the candidate checkpoint to temporary HF;
2. run 256/256 cadence evaluation;
3. run 512/512 full evaluation before final promotion;
4. run semantic and functional gates;
5. compare against clean SFT and the selected Stage 2 checkpoint; and
6. mark it promotable only if it passes every hard gate and either round-trip
   NMSE or functional KL improves over the previous best with a paired 95%
   confidence interval strictly above zero.

The hero is the best passing checkpoint, not automatically update 256.

## Storage and Cleanup Contract

Measured storage at design time:

- `/workspace/interp`: 516 GB free;
- `/workspace/models`: 454 GB free;
- `/dev/shm`: 2.2 TB free;
- Mac data volume: 207 GiB free;
- actor model-only checkpoint: approximately 59 GB; and
- selected AR DCP plus HF: approximately 72 GB.

The program must:

- create temporary HF conversions under `/dev/shm` and remove them in a
  `finally` path;
- retain at most the current best and one challenger actor checkpoint after an
  evaluation stage closes; the four Stage 2 confirmation checkpoints and
  three Stage 3 milestone checkpoints may coexist only during their active
  sequential evaluation window and are pruned immediately after selection;
- keep W&B offline logs, generated JSONL, reports, manifests, and configs;
- preserve `--no-save-optim` for ordinary probes and milestones;
- allow at most one explicitly configured rolling optimizer-bearing recovery
  checkpoint if exact-resume protection is required;
- upload selected model-only checkpoints to the project S3 prefix before local
  deletion; and
- sync only lightweight evidence to the Mac unless a specific heavy
  checkpoint is requested.

Cleanup must never delete the last known-good SFT checkpoint, selected AR,
current best RL checkpoint, or evidence from a failed run.

## Error Handling

- Missing provenance, transform records, source rows, or checkpoint files are
  hard failures.
- Identity-canary failure blocks functional scoring.
- A failed eval preserves generated text and partial reports.
- A failed training run preserves logs, the latest complete checkpoint, queue
  state, GPU and disk snapshots, and the exact rendered command.
- Queue dependencies prevent Stage 3 from starting on an inconclusive or
  failed Stage 2 gate.
- Cleanup is explicit and manifest-driven; glob-only checkpoint deletion is
  forbidden.

## Testing Strategy

### Unit Tests

- deterministic semantic transformations and source hashes;
- paired bootstrap reproducibility and confidence-interval direction;
- functional metric calculations on synthetic logits;
- activation rescaling and final-token-only hook replacement;
- identity-canary pass and fail behavior;
- batch-divisibility validation;
- KL-type rendering and validation;
- drift-guard state transitions;
- retention-manifest selection; and
- cleanup behavior on success and injected failure.

### Integration Tests

- tiny fixture parquet through transform, AR scoring, and report merge;
- four-row Nano boundary identity and functional patch smoke on RunAI;
- one-update queue dry run for each topology;
- temporary HF cleanup verified after both successful and failed eval; and
- end-to-end gate dependency proving Stage 3 cannot launch early.

### Experiment Verification

Every completed milestone records:

- code revision and dirty-state manifest;
- rendered YAML and command;
- dataset and checkpoint hashes;
- verifier and gate reports;
- offline W&B run paths;
- GPU memory, utilization, phase timing, and disk snapshots; and
- a concise logbook entry with decision and next action.

## Documentation

The following are updated after each milestone:

- `docs/rl_logbook.md` for systems and RL dynamics;
- `docs/experiment_logbook.md` for scientific results and decisions;
- the run registry for machine-readable status; and
- the job tracker while a queue is active.

Historical signed-`k1` runs remain labeled as scouting evidence and are never
silently mixed with corrected-`k3` results.

## Future Architecture-Adapted Joint Training

After a fixed-AR hero passes independent gates, the next design can adapt the
paper's simultaneous AV+AR objective to Nano30B. The likely starting point is
alternating block-coordinate training:

1. freeze AR and update AV for a bounded block;
2. freeze AV and refresh AR on generated explanations plus supervised replay;
3. evaluate with a held-out frozen AR and target-model functional metrics; and
4. promote only when both reconstruction and semantic validity improve.

This avoids assuming that the reference model's co-resident optimizer and GPU
layout transfer directly to Nano30B's large hybrid MoE architecture.
