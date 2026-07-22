# Nano30B AR Revised Improvement Design

## Purpose

Improve Nano30B AR-SFT from the current plateau around `0.437-0.44` teacher
normalized MSE toward a useful AR milestone, without starting RL, AV+AR tuning,
or large new activation generation. This revision incorporates external review
feedback on the previous geometry/staging-only design.

The revised design is diagnostic-first:

1. audit correctness assumptions that could fake an AR plateau
2. estimate the information ceiling of the current teacher explanations
3. fit closed-form and frozen-feature baselines before more SGD
4. only then run bounded training probes selected by the diagnostics

The phase still uses the existing `275,396` row R27 AR-SFT dataset, Qwen-faithful
critic contract, `NLACriticModel`, and standard heldout controls.

## Current Evidence

The latest wide sweep did not find an optimizer-only escape. The best confirmed
quick point remains `r27-best1547-lr2e5-iter0000256` with `512/512` heldout test
teacher NMSE `0.4374333322`. Wide `best1547` probes stayed near `0.44`, and
fullscan escape probes were worse at about `0.47-0.49`.

The strongest diagnostic clue is the prompt/control gap:

| input/control | test NMSE | approximate cosine |
| --- | ---: | ---: |
| teacher explanation | `0.4374` | `0.7813` |
| source-context | `0.5146` | `0.7427` |
| source-raw | `0.1302` | `0.9349` |
| target | `0.3000` | `0.8500` |
| target | `0.2500` | `0.8750` |

Under the normalized objective, `NMSE ~= 2 * (1 - cosine)`. Source-raw shows
that the critic architecture can reconstruct Nano R27 well when the source
information is present. It does not prove the teacher explanation contains
enough information to reach `0.25-0.30`; it proves the target is representable
and that teacher AR should be diagnosed against information and readout
ceilings, not only optimizer settings.

## Success Criteria

Primary metric:

- `512/512` validation and test teacher normalized MSE from the standard AR eval.

Promotion thresholds:

- `<0.40` test teacher NMSE is a meaningful improvement and triggers
  confirmation.
- `<0.35` on both validation and test is a usable AR milestone candidate.
- `0.25-0.30` on both validation and test remains the green target.

Required controls:

- teacher beats mean, teacher-shuffled, blank, generic, and source-context.
- source-raw remains diagnostic/oracle-style context, not the AR success target.
- paired bootstrap confidence intervals are reported before promotion.

Safety criteria:

- Do not start RL.
- Do not start AV+AR tuning.
- Do not claim AR success from training loss alone.
- Keep W&B offline for launched runs.
- Keep checkpoint retention storage-conscious.

## Phase 0: Correctness Audit

Before any new training, add a hard audit report for implementation assumptions
that could masquerade as a `0.44` plateau.

Audit fields:

- boundary name, expected zero-based last layer, critic config hidden layers, and
  gold activation boundary
- final norm stripping or bypass status for Nano, including `norm_f`
- critic suffix text and token IDs
- last real prompt token ID, decoded token, and prompt/explanation token counts
- chat template mode and any reasoning-control setting used to build critic
  prompts
- `mse_scale` and activation normalization mode
- value-head identity check: `||W - I||_F / ||I||_F`
- step-0 or current checkpoint pred/backbone/gold norm ratios when available
- eval row IDs, doc IDs, and split-overlap status

Hard expectations for R27:

```text
boundary_name = R27
zero_based_last_layer_index = 27
critic_config_num_hidden_layers = 28
gold_activation_boundary = R27
```

Correction recorded 2026-07-10: Nano boundary labels use the extractor's
zero-based decoder-module index directly. `R27` hooks block 27 and captures its
post-block residual stream, so a critic reproducing that boundary retains
blocks 0 through 27 inclusive (`28` blocks). Likewise, `R33` means block index
33 and a 34-block critic. The earlier 26/27 values treated the label as a
one-based ordinal and were inconsistent with the extractor and K+1 critic
implementation.

If any audit item fails, stop AR improvement work and fix the contract before
training or tuning.

## Phase 1: Information-Ceiling Diagnostics

This phase answers whether current teacher text can plausibly reach `0.25-0.30`.
It does not require new activations or model training.

### Duplicate And Near-Duplicate Explanation Floor

Canonicalize teacher explanations by lowercasing, removing XML tags, normalizing
whitespace, and stripping boilerplate. For groups with at least two rows, compute
the best constant prediction for that text group:

```text
h_group = normalize(mean(normalize(h_i)))
group_loss = mean_i NMSE(h_group, h_i)
```

Also compute near-duplicate floors using text embeddings or frozen critic hidden
states. For each validation explanation, retrieve nearest train explanations and
predict the mean activation of retrieved examples.

Interpretation:

- If this floor sits around `0.40-0.45`, teacher text is likely the bottleneck.
- If it reaches `0.25-0.30`, current AR training/readout is underperforming.

### Hard-Negative Diagnostics

For each validation row, compare the true gold vector against hard negatives:

- same document, different target token
- same target token string, different document
- same position bucket, different document
- same canonical explanation cluster
- same source domain

Report whether teacher AR predictions are closer to true gold than these
negatives. This catches models that beat random shuffles but fail token-local
disambiguation.

### Token, Position, And Local-Window Baselines

Build non-training baselines that expose missing low-level information:

- target token string only
- target token ID only
- position bucket only
- previous/next local token window
- target token plus position
- teacher explanation plus target token
- teacher explanation plus local token window

If teacher plus compact token/local hints sharply improves NMSE, the current
teacher explanation is missing activation-relevant lexical or positional
content.

### Specificity And Bucket Report

For each row, compute and report:

- explanation length
- boilerplate/generic score
- whether target token string appears in the explanation
- whether local context appears
- source domain and language when available
- source length
- target position bucket
- target norm bucket

Report teacher NMSE by bucket. This directs future data enrichment only if the
current teacher text is proven insufficient.

## Phase 2: Frozen-Feature And Closed-Form Baselines

This phase distinguishes "features lack information" from "SGD/head training is
weak."

Extract bounded feature artifacts:

- `q_teacher`: critic hidden/readout feature for teacher explanation prompt
- `q_source_context`: feature for source-context prompt
- `q_source_raw`: feature for raw source token prompt
- gold activation vector
- row metadata needed for buckets and hard negatives

Initial extraction may use a bounded train subset plus standard validation/test
samples. Full-train frozen-feature extraction is not part of the first pass.

Fit closed-form or cheap heads:

- ridge regression
- Procrustes/orthogonal map diagnostic
- CCA-style diagnostic if practical
- linear without bias
- linear with bias
- `mu + residual linear`
- RMSNorm plus linear
- final-token readout
- mean-pool over explanation span
- last-k-token pooling
- attention-pooling if cheap enough

Interpretation:

- If frozen features plus ridge/readout beat `0.40`, train a matching head/probe.
- If source-raw frozen features are excellent but teacher frozen features remain
  poor, the issue is teacher-text information or prompt/readout.
- If pooling beats final-token readout, the current last-token-only AR readout is
  a bottleneck.

## Phase 3: Subspace-Aware Geometry Diagnostics

Keep PCA diagnostics, but frame them correctly: the normalized loss is angular,
not raw coordinate MSE. The goal is subspace-aware evaluation and auxiliary
losses, not replacing the main normalized objective.

Report:

- target PCA spectrum and variance cutoffs
- residual energy by PC band for teacher, source-context, source-raw, mean, and
  kNN baselines
- top-PC and tail-PC errors by metadata bucket

Candidate PC bands:

- top `1-16`
- `17-64`
- `65-256`
- tail

Use this report to decide whether PCA-shrinkage auxiliary losses are worth a
training probe. Avoid native-coordinate diagonal std losses as a first-choice
method because they are not rotation-invariant and may overweight noise.

## Phase 4: Bounded Training Probes

Only launch bounded training after Phases 0-3 identify a plausible lever.

Recommended probe families:

1. ridge-init or identity-init head warmup
   - initialize value head from the best closed-form map if supported
   - otherwise head-only warmup with identity init verified

2. head-high/backbone-low optimization
   - separate value-head and backbone learning rates
   - optional layerwise LR decay
   - preserve current behavior when parameter-group fields are unset

3. readout/pooling probe
   - final-token baseline vs selected pooling readout from Phase 2
   - bounded `128-256` steps

4. source-raw geometry curriculum
   - Stage A: train on source-raw prompt to learn R27 geometry
   - Stage B: mix source-raw and teacher explanation
   - Stage C: teacher-only fine-tune
   - final eval remains teacher-only

5. auxiliary loss probes
   - norm-control auxiliary:
     `lambda_norm * (log||pred|| - log||gold||)^2`
   - PCA-shrinkage residual auxiliary
   - in-batch contrastive loss with hard negatives when available

All probes should start with `512/512` evals. Run `2048/2048` only after a clear
paired improvement at `512/512`.

## Data Flow

```text
existing AR train/validation/test parquet
  -> Phase 0 correctness audit report
  -> Phase 1 information-ceiling reports
  -> Phase 2 frozen-feature artifacts and closed-form baselines
  -> Phase 3 subspace diagnostics
  -> selected bounded probe configs
  -> queue/watch one probe at a time on RunAI train
  -> standard 512/512 eval reports with prediction dumps and bootstrap CIs
  -> logbook summary and promotion decision
```

## Implementation Boundaries

Do:

- reuse existing AR YAML config and queue patterns
- keep the standard eval controls unchanged
- add row-level prediction dumps for diagnostics
- make all diagnostics JSON-backed and reproducible
- add focused tests for math, parsing, and config behavior

Do not:

- generate large new AR data in this phase
- change the teacher explanation template contract during diagnostics
- start RL or AV+AR tuning
- run long `2048/2048` evals before a `512/512` win
- save optimizer shards for every probe

## Testing

Tests should cover:

- R27 boundary/off-by-one audit logic
- suffix/last-token metadata extraction on toy prompts
- identity-head distance calculation
- duplicate explanation floor math
- kNN/constant group baseline math on synthetic vectors
- hard-negative and bucket metric aggregation
- PCA band residual accounting
- ridge/Procrustes baseline math on known synthetic systems
- config defaults preserving existing AR behavior
- parameter-group config rendering only when requested

## Expected Outcomes

If Phase 0 fails, the next milestone is contract repair, not training.

If Phase 1 floors are near `0.40-0.45`, the current teacher text is likely the
main bottleneck. The next design should consider compact token/local hints or
teacher explanation enrichment.

If Phase 2 baselines beat `0.40`, the next training run should implement the
matching head/readout/staging recipe.

If source-raw curriculum improves teacher-only eval, promote that as the main
optimization path while preserving teacher-only final evaluation.

If none of the diagnostics find a lever, continuing broad LR/schedule sweeps is
not justified.
