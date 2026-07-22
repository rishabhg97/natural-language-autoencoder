I’m using the writing-plans skill to create the implementation plan.

# Nano30B NLA AV/AR Scale-Up Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Train a Qwen-faithful Nano30B NLA warm-start system with both a usable AV actor `h -> z` and an AR critic `z -> h_hat`, validated by reconstruction, row-specific controls, and round-trip tests.

**Architecture:** Follow the Qwen NLA recipe: supervised actor warm-start first, supervised critic reconstruction next, then actor-generated round-trip validation. We treat the successful 30K AV run as proof that the Nano injection path is faithful enough to scale, but not yet evidence that explanations carry reconstructable activation information.

**Tech Stack:** Nano30B BF16, RunAI H200 workspace, Super teacher labels, parquet datasets, Qwen NLA reference code, local Nano adaptation scripts.

---

**Current Baseline**

We have a real AV signal on 29,913 Super-thinking rows.

Best current AV run:

```text
Dataset: 29,913 usable rows
Split: doc-level, 26,920 train / 2,993 heldout
Trainable: lm_head only
Steps: 10,000
Best config: scale 75, lr 1e-4, max_target_tokens 192

Heldout NLL:
real h:      1.6051
shuffled h:  1.7151
zero h:      1.6812
mean h:      1.7029
no inject:   1.8309
```

This is good because real activation beats every control. It is not enough because decoded content is still weak and we have not yet shown `z -> h_hat` reconstruction, which is the core NLA critic proof.

The plan below treats this as the starting line, not the finish line.

---

## Stage 0: Lock Down Evaluation And Data Contracts

**Objective:** Before scaling more, make sure every future AV/AR run reports comparable, Qwen-style metrics.

**Files likely involved:**

```text
/Users/rigarg/.codex/worktrees/c422/research-projects/nano30b-nla-pilot/scripts/nano_av_warmstart_smoke.py
/Users/rigarg/.codex/worktrees/c422/research-projects/nano30b-nla-pilot/scripts/nano_realdata_stage3_build.py
/Users/rigarg/.codex/worktrees/c422/research-projects/nano30b-nla-pilot/scripts/nano_ar_*.py
/Users/rigarg/.codex/worktrees/c422/research-projects/nano30b-nla-pilot/tests/test_nano_harness.py
```

**Required dataset schema:**

For AV:

```text
source_text / prefix text
doc_id
row_id
activation_vector
activation_layer
activation_position
teacher_model
teacher_prompt_version
teacher_explanation
target_text = <explanation>{z}</explanation>
```

For AR:

```text
teacher_explanation
critic_prompt
activation_vector target
activation_layer
activation_position
doc_id
row_id
```

**Validation checks every dataset must pass:**

```text
No train/heldout doc overlap
No duplicate row IDs across splits
Activation dim exactly expected for Nano residual stream
Activation layer matches requested R_27 / R_30
All activation rows finite
Teacher explanation non-empty
Parsed explanation has expected tags or clean fallback parse
Target token length distribution reported
Drop reasons counted explicitly
```

**Run-level checks every training run must report:**

```text
Train rows / heldout rows / test rows
Doc counts per split
Target length p50 / p90 / p99
Activation norm p50 / p90 / p99
Training loss curve
Heldout real vs shuffled vs zero vs mean vs no injection
Generated examples for each control
Saved trainable checkpoint
Exact command and env path
```

**Target to proceed:**

```text
Dataset blocker count: 0
Run manifest complete
All controls reported
No silent parse/drop behavior
```

---

## Stage 1: Generate A Larger, Cleaner Super Teacher Dataset

**Objective:** Move beyond 30K rows. The current AV result says scale matters, so the next teacher-label target should be 100K minimum, with a 200K stretch goal if throughput is acceptable.

**Primary teacher choice:** Super with high reasoning/token budget.

Reasoning: we want best mathematical/semantic explanation quality. Cost is not the bottleneck. Super is acceptable and cheaper for the user. High reasoning is preferable here because we want explanations that reveal latent computational structure, not terse summaries.

**Prompt target:**

Keep the teacher prompt in a separate txt file, versioned, for repeatability:

```text
/Users/rigarg/.codex/worktrees/c422/research-projects/nano30b-nla-pilot/prompts/super_teacher_nla_v2.txt
```

The prompt should ask for:

```text
A concise but information-dense explanation of what information in the prefix is most relevant to the model's next-token computation.
No chain-of-thought transcript.
No generic summary.
Output exactly one <explanation>...</explanation> block.
```

Important: “high reasoning” does not mean we want hidden reasoning printed. We want the model to use more reasoning internally and emit a cleaner compressed explanation.

**Data sizes:**

Step 1A:

```text
Generate 100K teacher explanations
Use existing source prefixes if available
Checkpoint every 500-1000 rows
Write chunked parquet/jsonl outputs
```

Step 1B:

```text
If parse/drop rate < 2% and speed is acceptable, extend to 200K
```

**Validation targets:**

```text
Parse success: >= 98%
Empty explanation rate: < 0.5%
Malformed XML/tag fallback rate: < 2%
Median explanation length: 40-160 tokens
p99 explanation length below AV max target cap or intentionally truncated
No repeated boilerplate collapse
```

**Quality audit:**

Sample 100 explanations and label:

```text
specific-to-prefix
generic summary
format violation
overlong
empty/useless
```

Target:

```text
specific-to-prefix >= 85%
generic/useless <= 10%
format violation <= 2%
```

---

## Stage 2: Extract Activations At Scale

**Objective:** Build activation targets for the same rows, with R_27 primary and backup layers/positions for future analysis.

**Primary activation target:**

```text
Nano residual R_27
Final source prefix token
```

**Backup extraction:**

```text
R_24
R_27
R_30
final source token
optionally one earlier semantic anchor position if cheap
```

R_27 remains the main NLA path. R_30 is backup because the user already requested it and because later-layer residuals may align better with language explanations. R_24 gives an earlier comparison point.

**Run plan:**

```text
Extract activations for all teacher-labeled rows
Store one canonical parquet per layer/position
Keep a compact merged training parquet for R_27
Do not regenerate teacher labels during activation extraction
```

**Validation targets:**

```text
Activation extraction success: >= 99.5%
Non-finite rows: 0
Dim consistency: 100%
Layer metadata consistency: 100%
Activation norm distribution stable across chunks
```

**Performance target:**

Given prior cluster throughput, 50K activations should be comfortably sub-hour on one A100-class GPU. On H200, the target should be faster, but exact speed depends on input length and batching. The key target is not raw speed; it is reliable chunked extraction with restartability.

---

## Stage 3: Build Canonical AV And AR Parquets

**Objective:** Produce stable training files for both actor and critic.

**Outputs:**

```text
runs/stage3/super-thinking-v2-100k/av_sft_r27.parquet
runs/stage3/super-thinking-v2-100k/ar_sft_r27.parquet
runs/stage3/super-thinking-v2-100k/manifest.json
```

**AV target:**

```text
input activation: h at R_27
output text: <explanation>{z}</explanation>
```

**AR target:**

```text
input text: z inside critic template
output target: h at R_27
```

**Splits:**

Use doc-level splits only.

```text
Train: 90%
Validation: 5%
Test: 5%
```

The previous run used train/heldout. Going forward we need a true test split that remains untouched until promotion decisions.

**Promotion gates:**

```text
0 doc overlap
0 malformed activation rows
drop count fully explained
same rows available in AV and AR views
```

---

## Stage 4: AV Actor Training, Scaled Properly

**Objective:** Train a stronger AV actor that converts `h -> z` with row-specific dependence and better decoded content.

**Baseline to beat:**

```text
Current 30K / 10K-step AV:
heldout real NLL: 1.6051
gap vs shuffled: +0.1100
gap vs zero:     +0.0761
gap vs mean:     +0.0978
gap vs none:     +0.2258
example F1:      ~0.25
```

**Training configs:**

Step 4A, 30K reproduction:

```text
Purpose: ensure new code/env still reproduces current result
Rows: existing 29,913
Steps: 10K
Config: scale 75, lr 1e-4, lm_head only
Target: within +/- 0.03 NLL of current run
```

Step 4B, 100K tuning:

Run parallel H200 jobs:

```text
Config A:
scale 75
lr 1e-4
steps 30K
trainable lm_head only

Config B:
scale 150
lr 1e-4
steps 30K
trainable lm_head only

Config C:
scale 75
lr 5e-5
steps 30K
trainable lm_head only

Config D:
scale 75
lr 1e-4
steps 30K
trainable final block + lm_head only, if memory allows
```

I would keep the first serious scaled run mostly faithful and avoid adding LoRA/PEFT unless the lm_head-only actor clearly plateaus. The current result already proves lm_head-only works enough to scale.

Step 4C, final AV:

```text
Rows: 100K or 200K
Steps: 1-2 epochs equivalent, not tiny overfit steps
Batching: whatever fits without changing recipe too much
Best config from Step 4B
Save trainable state and manifest
```

**AV validation metrics:**

Teacher-forced:

```text
heldout/test NLL real
heldout/test NLL shuffled
heldout/test NLL zero
heldout/test NLL mean
heldout/test NLL no injection
```

Generation:

```text
exact tag validity
content F1 / ROUGE against teacher z
BERTScore or embedding similarity if easy
qualitative examples
control degradation examples
```

**Target AV performance:**

Minimum acceptable:

```text
Test real NLL <= 1.55
real beats shuffled by >= 0.12
real beats zero by >= 0.08
real beats mean by >= 0.10
real beats no-injection by >= 0.25
valid <explanation> format >= 95%
```

Strong target:

```text
Test real NLL <= 1.40
real beats shuffled by >= 0.18
real beats no-injection by >= 0.35
generated explanation F1 >= 0.40
controls visibly degrade on most sampled rows
```

Critical target:

```text
Real h must beat every control on at least 80% of heldout examples by per-row NLL.
```

That per-row win-rate matters more than aggregate NLL, because aggregate NLL can hide generic-language gains.

---

## Stage 5: AR Critic Training, Qwen-Style

**Objective:** Train the real NLA critic: `z -> h_hat`, where the target is Nano R_27 activation.

This is the main blocker now. Until this works, we do not know whether teacher explanations actually preserve the information needed to reconstruct Nano activations.

**Qwen recipe adaptation:**

Qwen NLA critic uses:

```text
Truncated base model up to target layer
No final language modeling objective
Linear(d, d) reconstruction head
Normalized MSE on target activation
Evaluation against shuffled/random/mean controls
```

Nano adaptation:

```text
Input: critic prompt containing z
Backbone: truncated Nano through the layer corresponding to R_27 geometry
Head: Linear(2688, 2688), or equivalent Nano residual dim
Loss: normalized MSE against target h
Metric: cosine similarity, NMSE, FVE/RRI
```

**Important geometry test before training:**

Run source-prefix oracle:

```text
x -> truncated Nano -> h_hat_source
compare h_hat_source to stored h at R_27
```

Target:

```text
cosine >= 0.99 if same boundary
NMSE near zero relative to activation norm
```

If this fails, the issue is not training. It means our layer boundary, residual stream capture point, token position, or normalization is wrong.

**AR training configs:**

Step 5A, source oracle sanity:

```text
Rows: 512
No training
Verify stored h can be reproduced from x
```

Step 5B, 3K critic smoke:

```text
Rows: 3K
Steps: 2K-5K
Trainable: reconstruction head only first
Backbone frozen
```

Step 5C, 30K critic:

```text
Rows: existing 29,913
Steps: 20K-50K
Trainable: head only, then optionally last 1-2 truncated blocks if needed
```

Step 5D, 100K critic:

```text
Rows: 100K
Steps: 1 epoch equivalent
Best config from 30K
```

**Controls:**

Every AR eval must compare:

```text
correct z -> h
shuffled z -> h
blank z -> h
generic z -> h
mean activation baseline
random row baseline
source oracle x -> h, if included
```

**Metrics:**

```text
NMSE normalized by target activation norm
cosine(h_hat, h)
FVE = 1 - MSE_model / MSE_mean_baseline
correct-vs-shuffled win rate
correct-vs-blank win rate
correct-vs-mean win rate
```

**Target AR performance:**

Minimum acceptable 30K:

```text
FVE > 0.05 on heldout
correct cosine beats shuffled by >= 0.05
correct beats shuffled on >= 65% of rows
correct beats blank/generic on >= 70% of rows
```

Strong 100K target:

```text
FVE > 0.15
median cosine >= 0.35
correct beats shuffled on >= 75% of rows
correct beats mean on >= 80% of rows
```

Qwen-level target, eventually:

```text
correct beats shuffled/random/mean on >= 95% rows
large cosine gap vs controls
```

We should not expect Qwen-level numbers immediately. Qwen’s released positive control is much stronger: correct AR median cosine around `0.92` and correct beating controls on essentially all rows. Our first Nano target should be: prove nontrivial reconstructability, then scale.

---

## Stage 6: Round-Trip AV -> AR Validation

**Objective:** Test the actual NLA loop, not just isolated AV or AR.

Pipeline:

```text
x -> Nano -> h
h -> AV -> z_generated
z_generated -> AR -> h_hat
compare h_hat to h
```

Compare against:

```text
teacher z -> AR -> h_hat
AV z from real h
AV z from shuffled h
AV z from zero h
AV z with no injection
blank/generic z
```

**This is the decisive test.**

Expected ordering:

```text
source oracle best
teacher z next
AV(real h) next
AV(shuffled/zero/no-injection) worse
blank/generic worst
mean baseline lowest
```

**Promotion target:**

Minimum:

```text
teacher z AR FVE > 0
AV(real h) AR FVE > controls
AV(real h) beats AV(shuffled h) on >= 60% rows
```

Strong:

```text
teacher z FVE > 0.15
AV(real h) preserves at least 50% of teacher-z AR gain
AV(real h) beats all AV controls on >= 70% rows
```

If teacher z works but AV-generated z fails, the blocker is actor generation quality.

If teacher z fails, the blocker is explanation content or critic geometry/training.

If source oracle fails, the blocker is Nano truncation/activation boundary.

---

## Stage 7: Layer And Position Ablations

**Objective:** Use extra activation extraction to find the easiest NLA target, without turning this into an endless search.

Run only after R_27 baseline is understood.

Compare:

```text
R_24 final token
R_27 final token
R_30 final token
```

For each layer:

```text
AV heldout real-vs-control NLL
AR FVE / cosine / win rate
round-trip FVE
```

**Decision rule:**

Promote a layer only if it wins on AR or round-trip, not just AV NLL.

Reason: later layers may be easier to decode into text but harder to reconstruct faithfully. The critic decides whether it is useful for NLA.

---

## Stage 8: Final Larger Training Run

**Objective:** Train the best AV and AR pair on the largest clean dataset available.

Preferred final dataset:

```text
100K minimum
200K stretch
doc-level train/val/test
R_27 primary unless ablations identify a better layer
```

Final AV:

```text
Best scale/lr/trainable subset from Stage 4
Train until heldout NLL plateaus
Early stop by validation real-vs-control gap, not train loss
```

Final AR:

```text
Best critic config from Stage 5
Train until validation FVE/cosine plateaus
Early stop by correct-vs-shuffled win rate
```

Final report should include:

```text
Dataset rows and drops
Teacher parse quality
Activation extraction stats
AV metrics
AR metrics
Round-trip metrics
Layer ablation if run
Failure examples
Best checkpoint paths
Exact commands
```

---

## Decision Tree

**If AV improves but AR fails:**

The explanations are fluent but not reconstructive. Next move is better teacher prompt, richer z, or target a different layer.

**If AR works with teacher z but AV-generated z fails:**

The critic is good, teacher labels are useful, but actor generation is weak. Next move is stronger AV training, more trainable parameters, longer training, or better decoding.

**If both AV and AR work independently but round-trip fails:**

The AV actor is optimizing teacher-likelihood but losing reconstruction-critical details. Next move is Qwen-style RL or critic-guided actor training.

**If source oracle fails:**

Stop all training. Fix Nano geometry, layer boundary, token position, or residual capture.

**If R_30 beats R_27 on AR and round-trip:**

Switch primary target to R_30, but keep R_27 results as the paper-aligned baseline.

---

## Immediate Next Execution Order

1. Reproduce current 30K AV run once in the expanded RunAI setup.
2. Implement/source-check AR critic geometry with the source-prefix oracle.
3. Build canonical 30K AR parquet from the existing Super-thinking data.
4. Run 3K AR smoke.
5. Run 30K AR full critic if smoke beats controls.
6. Start 100K teacher generation and activation extraction in parallel.
7. Run 100K AV and AR once data is ready.
8. Run round-trip validation.
9. Only then consider RL-style actor optimization.

The single most important next step is the AR critic. The AV result is encouraging, but the NLA claim lives or dies on whether `z` reconstructs `h` better than controls.