# A Natural-Language Bottleneck for Nano30B Residual Activations: Directional Reconstruction and Stored-Snapshot Functional Recovery at Boundary R33

<!-- R33-HERO-BASELINE-PROTOCOL-INVALIDATED -->

Status: internal manuscript draft (`2026-07-16`). Not cleared for external
distribution; see `docs/publication/r33_nano30b_nla_open_issues.md` for
blockers. Every number in this draft is traceable to a hash-bound JSON
artifact via `docs/publication/r33_nano30b_nla_evidence_table.md`.

---

## Abstract

We train and evaluate a supervised Natural Language Autoencoder (NLA) pair
for the hybrid Mamba/attention/MoE model NVIDIA-Nemotron-3-Nano-30B-A3B
("Nano30B") at residual boundary R33. An activation verbalizer (AV) maps a
stored residual-stream activation `h` (d=2688) to a natural-language
explanation `z`; an activation reconstructor (AR) maps `z` back to an
estimate `h_hat`. The primary metric is directional reconstruction error,
`||unit(h_hat) - unit(h)||^2 = 2(1 - cos(h_hat, h))`, which is invariant to
prediction scale. On a deterministic stored activation snapshot with
content-family-disjoint splits (512 rows per split; 250 validation / 255
exploratory-test families), AV-generated text reaches directional MSE
0.307004 (validation) and 0.319225 (exploratory test), close to teacher-text
reconstruction (0.304714 / 0.302637), with 512/512 closed and parseable
generations on both splits. Every registered semantic and activation control
is beaten with positive content-family-clustered 95% bootstrap intervals and
rowwise win rates of 99.61–100%. Reinjecting `h_hat` into a stored
counterfactual snapshot recovers next-token behavior at teacher-text level
within family-clustered uncertainty (e.g., exploratory-test
KL(original‖patched) 0.9495 vs. teacher 0.9701, versus 4.12–9.53 for mean,
zero, and shuffled controls). An independently initialized and independently
trained AR (seed 314159) reproduces the validation result (AV text 0.310963
vs. teacher 0.308533; all controls passed). We report equally prominent
negative results: native centered raw R² is negative on both splits (−0.327 /
−0.335), so raw-magnitude recovery is not established (a validation-fitted
single scalar, 0.560604, raises exploratory-test centered raw R² to 0.478 but
is post-hoc); strict fresh-forward activation identity fails, so all claims
are scoped to the stored snapshot; and a project-wide exposure audit shows
all 5,009 in-corpus content families were exposed to selection or evaluation,
so no in-corpus confirmatory boundary exists. A previously reported RL
improvement (July 8) was invalidated by a mixed-protocol baseline and is not
claimed. The contribution is a transparent, verifier-bound account of an
activation-to-language bottleneck that preserves strong row-specific
directional and downstream functional information under a deterministic
stored-snapshot protocol.

## 1. Introduction

Whether the internal activations of a large language model can be re-encoded
as natural language — and decoded back with enough fidelity to matter — is a
concrete, falsifiable question for interpretability research. The Natural
Language Autoencoder (NLA) formulation [1] operationalizes it as a
round-trip: a verbalizer model turns an activation vector into text, and a
reconstructor model turns that text back into a vector, with reconstruction
scored against the original activation.

This paper documents a supervised NLA pair for Nano30B, a hybrid
Mamba-2/attention/MoE architecture, at one residual boundary (R33). We define
the system as:

```
h      = stored Nano30B R33 activation (d_model = 2688)
z      = AV-generated natural-language explanation of h
h_hat  = AR reconstruction of h from z
```

Three properties distinguish this study from a conventional results report:

1. **Every headline number is bound to a fail-closed verifier.** Each claim
   below cites a JSON report and an independent verifier JSON, both
   SHA-256-fingerprinted, together with the exact checkpoint fingerprints
   and dataset hashes used to produce it.
2. **The claim boundary is explicit and narrow.** The evidence supports
   *directional* reconstruction on a *stored snapshot* under a
   *family-disjoint but historically exposed* evaluation boundary. It does
   not support raw-magnitude recovery, fresh-forward identity, confirmatory
   generalization, layer-superiority, or RL improvement.
3. **Failures are part of the result.** The current protocol exists because
   earlier results — including a promoted RL headline — were invalidated by
   training-path contamination, a mixed-protocol baseline, and near-duplicate
   evaluation leakage. Section 12 reconstructs that history from the audit
   record.

Our thesis is deliberately modest: under a deterministic stored-snapshot
protocol, AV-generated text carries enough row-specific information about
`h` for a paired AR to recover its direction almost as well as from
human-teacher text, and for the reconstruction to substitute functionally
for the original activation in a stored counterfactual forward pass. The
manuscript is written to remain useful even if a reader rejects the broader
NLA interpretation, as a documented case study of an activation-to-language
bottleneck with controls, replication, and a complete bug history.

## 2. Related work

**Natural Language Autoencoders.** The method adapted here was introduced in
"Natural Language Autoencoders Produce Unsupervised Explanations of LLM
Activations" (Fraser-Taliente, Kantamneni, Ong et al., Transformer Circuits,
2026) [1], with an accompanying Apache-2.0 reference implementation [2]. The
reference recipe trains AV ("actor") and AR ("critic") pairs for
Qwen2.5-7B-Instruct, Gemma-3-12B/27B-IT, and Llama-3.3-70B-Instruct at layers
roughly two-thirds through the model, injects the activation as a single
input-embedding token at a fixed scale (150 in the published recipe), scores
round trips with the same scale-invariant `2(1 - cos)` metric used here, and
follows supervised training with joint AV+AR reinforcement learning.

**This project is an adaptation, not a reproduction.** The vendored tree
under `external/natural_language_autoencoders/` is a production fork whose
`NANO_FORK.md` mandates exactly this disclosure. Material divergences,
enumerated in `docs/methods/measurement_contract.md` and `NANO_FORK.md`:
Nemotron-H architecture adapters (hybrid Mamba-2 cache handling, `.backbone`
module paths, MoE routing); injection scale 75 rather than 150 (inherited
from the R27 line; never re-derived for R33); project-specific directional
FVE definitions; and a frozen-AR-critic RL design, which inverts the upstream
recipe's explicit rationale for jointly training the critic. No claim of
method parity with [1] is made, and no comparison of our numbers to the
Qwen/Gemma/Llama results is meaningful without a method-by-method audit
(different models, data, activation geometry, and metric baselines).

**Base model and data.** Nano30B is NVIDIA-Nemotron-3-Nano-30B-A3B-BF16, a
Nemotron-H-family hybrid Mamba-2/attention/MoE causal LM [3, TODO: confirm
the correct primary citation for the Nemotron-3 Nano model family before
external use]. Source text is FineWeb (`HuggingFaceFW/fineweb`,
`sample-10BT`) [4]; teacher explanations were generated with
`nemotron-3-super-v3` through an NVIDIA inference API (Section 6). Related
interpretability techniques that also read internal states into text —
probing, patchscopes-style decoding, sparse-autoencoder feature labeling —
are surveyed in [1]; we do not re-survey them here. [TODO: add primary
citations for Mamba-2 (Dao & Gu, 2024) and any surveyed methods if this
section is expanded; do not cite from memory without verification.]

## 3. Nano30B architecture context and the R33 boundary

The project's architecture notes (`docs/nano30b-nla-core-plan.md`) record
Nano30B as a 52-block `NemotronHForCausalLM` with hidden size 2688 and block
pattern `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME` (`M` =
Mamba-2, `E` = MoE, `*` = grouped-query attention). The "30B-A3B" name
encodes a ~30B-parameter MoE with a smaller active-parameter footprint; the
repository does not state exact total/active parameter counts, and we do not
assert them. The residual stream is defined by
`R_0 = Embed(x)`, `R_{i+1} = R_i + F_i(RMSNorm_i(R_i))`.

**Boundary convention.** The repository contains two conflicting label
conventions, and we report both rather than silently choosing one. The
deployed extractor that produced the stored dataset
(`scripts/nano_ar_layer_sweep.py::_forward_selected_boundaries`, invoked by
`nano_prefix_activation_extract.py`) labels the residual state after `b`
blocks as `R_b` (`boundary = layer_idx + 1` after running `blocks[layer_idx]`;
`R_0` = embedding output), matching the core plan's definition
(`R_{i+1} = R_i + F_i(RMSNorm_i(R_i))`, "extracting after zero-based block k:
b = k + 1"). Under this convention the stored target is the residual stream
**after the first 33 of 52 blocks** — the output of zero-based block 32, a
Mamba-2 block, immediately before the attention block at index 33 — which is
exactly the June research memo's "post-Mamba, pre-attention" reading. The
functional and identity diagnostics hook the same module
(`layers[boundary − 1]`), so every activation-facing code path is internally
consistent. A convention "correction" recorded on 2026-07-10
(`docs/superpowers/specs/2026-06-03-nano-ar-geometry-staging-design.md`;
`nano_ar_correctness_audit.v2`) instead asserts that R33 hooks module index
33 directly and pairs it with the AR critic's geometry — the shipped AR
checkpoint physically retains blocks 0–33 (34 blocks, audit-verified) with
the LM head and final norm stripped. The two facts that are certain from
code and checkpoints: the stored target is the post-33-blocks residual state,
and the critic trunk contains 34 blocks. The label-level disagreement does
not affect any measurement, but it blocks any published claim about the
boundary's architectural type until reconciled (see open issues).

**Layer selection (exploratory).** R33 was selected on 2026-06-07 from
matched 20k-row AR probes: R33 reached validation/test teacher directional
NMSE 0.381983/0.388301 versus 0.490728/0.501399 for R34 and ~0.441/0.437 for
the mature, heavily tuned R27 line, with a roughly 2× better `source_raw`
floor (~0.07 vs. ~0.13). These probes predate the contamination fixes of
Section 12 and are scouting evidence only; the selection was not re-run
clean. No R33-over-R27 superiority claim is made — that would require a
row-matched clean R27 retrain that does not exist.

## 4. Method: AV and AR

**AV (activation verbalizer).** A full fine-tune of Nano30B that conditions
generation on `h` by *input-embedding injection*: one reserved marker-token
embedding position in a fixed prompt is replaced by `h` rescaled to L2 norm
`injection_scale = 75` (`normalize_activation` in the fork's `nla/schema.py`;
resolution in `scripts/nano_av_warmstart_smoke.py`). The model is trained
with next-token loss on teacher explanations and generates
`<explanation>…</explanation>`-delimited text. Publication-protocol
generation is greedy (`do_sample=false`, temperature 0.0), no forced prefix
(empty prefix, SHA-256 of the empty string recorded), `max_new_tokens=256`,
stop text `</explanation>`, `legacy_batch` backend, bfloat16, seed 20260709;
the full protocol is hashed into each generated record
(`generation_protocol_sha256` = `e5e3a265…d416`).

**AR (activation reconstructor / critic).** A truncated Nano30B prefix
retaining blocks 0–33 with the LM head and final norm stripped, plus a
`Linear(d, d)` value head read at the last token of a fixed critic template
(`Summary of the following text: <text>{explanation}</text> <summary>`). The
training objective is the directional loss `2(1 - cos(h_hat, h))`. The AR
also serves as the frozen critic in the (historical) RL experiments.

**Metric definitions** (fixed in `docs/methods/measurement_contract.md` and
implemented in `scripts/nano_eval_core.py::activation_reconstruction_metrics`):

```
directional_mse       = ||unit(h_hat) - unit(h)||^2  = 2 * (1 - cosine)
raw_mse               = mean over rows and features of (h_hat - h)^2
mean_predictor_raw_mse = mean of (h - train_mean)^2
centered_r2           = 1 - raw_mse / mean_predictor_raw_mse
norm_ratio_mean       = mean(||h_hat|| / ||h||)
```

`directional_mse` is dimension-independent and **ignores magnitude**; a
prediction that is an exact positive scalar multiple of the target scores
perfectly on it while failing raw metrics (this is unit-tested). Historical
project reports called this quantity "NMSE", and some older reports divided
it by d_model=2688 (values ~1e-4); we use only `directional_mse` here and
never conflate it with raw-vector MSE or magnitude recovery. Every
directional result below is accompanied by raw MSE and centered raw R²
against the train-mean predictor. **FVE-NRM** is
`1 − directional_mse(model) / directional_mse(train-mean predictor)` — a
fraction-of-error-explained in *direction* space
(`eval_nano_av_ar_roundtrip_gate.py::metric_summary`). Caution: the upstream
NLA codebase emits a field of the same name (`fve_nrm`) computed against a
different, stricter raw-variance baseline (its looser direction-space
baseline is called `fve_nrm_meannorm` there), so FVE-NRM values here are not
comparable to upstream `fve_nrm` numbers.

## 5. Dataset construction and family-disjoint split

**Rows.** 275,396 stored activation rows from 27,647 FineWeb `sample-10BT`
documents (document span 10500:38162, ten prefix token positions per
document), each row = (document, prefix position) with the R33 activation at
the last prefix token, d_model 2688, zero non-finite activations, zero
empty/malformed texts. The original teacher preparation dropped 865 teacher
parse failures and 1,015 unmatched activation rows; these exclusions remain
visible in the data lineage. Teacher explanations (3–5 short lines, forbidden
from verbatim continuation cues, teacher never sees `h`) come from
`nemotron-3-super-v3`; the May-28 teacher root parquet is byte-bound by
SHA-256 `76b78d2c…825e`.

**Deterministic extraction.** After the fidelity failures of Section 12, all
activations were re-extracted under a frozen deterministic profile (PyTorch
deterministic algorithms, TF32 off, cuDNN benchmark off, float32 matmul
`highest`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, seed 20260709, source commit
`0dabaad…`). An independent full eight-shard replay reproduced the merged
parquet byte-identically (SHA-256 `e3008a15…21ac`). This proves deterministic
replay of the extraction pipeline — **not** equality with an arbitrary fresh
forward pass (Section 11).

**Content families.** To prevent near-duplicate leakage, documents are
grouped into *content families*: NFKC-casefolded word tokenization, 5-token
shingles, and union-find over (i) exact normalized-text duplicates, (ii) a
bottom-k signature accelerator, and (iii) an exact deterministic
prefix-filter closure that evaluates every candidate pair at Jaccard ≥ 0.80
(`build_nano_content_family_manifest.py`; algorithm version
`bottomk_plus_deterministic_prefix_jaccard_union_find_v2`). A separate
exact-prefix refinement merges families sharing exact row-level content keys.
The refined manifest (SHA-256 `479cbab5…41b6`) has 5,009 families over the
27,647 documents.

**Split.** Whole families — never rows or documents — are assigned
train/validation/test with weights 0.90/0.05/0.05 at seed 20260709
(`build_nano_publication_family_split.py`), giving 4,504 / 250 / 255
families and 247,865 / 13,766 / 13,765 rows with zero cross-split family,
document, or exact-prefix content overlap. Split parquet hashes: train
`cf618cb0…fe6c`, validation `f543eb9e…e1ff`, test `86973528…63c5`. Evaluation
uses 512 rows per split selected by seeded family-stratified sampling
(selection seed 20260709).

**Boundary caveats (load-bearing).** (i) The split builder forbids
historically exposed families from *test* only; validation is not guaranteed
exposure-free by construction. (ii) The v6 selected-pair exposure audit
(Section 10) later showed that *all* 5,009 families — including the 255 test
families — were exposed somewhere in the project's history, so the test
split is family-disjoint from clean SFT training but **exploratory**, not
confirmatory. We label it "exploratory test" throughout.

## 6. Training setup and compute

All three selected runs trained for exactly 1,291 optimizer updates on the
frozen split, with validation-only checkpoint selection, offline W&B, and
hash-bound queue contracts. Trainer: Miles (pinned `radixark/miles@051cd15`)
with FSDP2 and the project's NLA patches.

| Run | GPUs | Recipe | Wall time | GPU-hours |
|---|---|---|---|---:|
| Primary AR (`nano-ar-r33-publication-deterministic-family-clean-4gpu-unfusedtorchconv-expertscan-cudablock-lr5e5-warmup25-gb192-mb48`) | 4× H100 NVL | lr 5e-5, cosine, warmup 25, gb192/mb48, padded microbatches | 3.8867 h | 15.5467 |
| Primary AV (`nano-av-r33-publication-deterministic-family-clean-8gpu-pospass-lr1e4-warmup25-gb192-mb2-dyn4096`) | 8× H100 NVL | lr 1e-4, cosine, warmup 25, gb192/mb2, dynamic packing 4096 | 13.4608 h | 107.6867 |
| Independent AR (seed 314159) | 4× H100 NVL | identical to primary AR | 3.8781 h | 15.5122 |

Total selected successful training: **138.7456 H100-NVL GPU-hours**
(hash-bound report `compute_accounting.json`, SHA-256 `7bde74be…7238`; mean
logged GPU utilization 79.8% / 61.1% / 81.3%). The accounting explicitly
excludes activation extraction, DCP→HF conversion, evaluation, historical
HPO and RL, and failed diagnostics whose timing is not exactly recoverable —
it is not full-project compute.

Two fail-closed training-path gates guard correctness: the AR critic uses
padded masked microbatches with explicit last-token value indexing because
packed THD critic training is provably non-equivalent for Nemotron-H
(Section 12); the AV run passed a live packed-vs-padded equivalence gate
(packed/padded response NLL 2.563666 / 2.565836; max abs/rel difference
0.022072 / 0.007950) before training was allowed to proceed. Selected
checkpoints (both `iter_0001291`): AR HF directory SHA-256 `5e792120…d760`;
AV DCP model fingerprint `dcp_model_sha256:43346232…386a`.

## 7. Evaluation protocol and controls

All evaluations run on the frozen 512-row family-stratified samples with the
hashed generation protocol of Section 4. Controls are computed both at
*generation* (what vector conditions the AV) and at *scoring* (what text the
AR reads):

- **av_real** (primary): AR reconstruction from text the AV generated from
  the row's true activation.
- **teacher**: AR reconstruction from the ground-truth teacher explanation —
  an upper reference for the supervised pipeline, not a raw-space bound.
- **av_shuffled**: AV text generated from a same-split activation of a
  *different content family* (seeded per row).
- **av_zero / av_mean / av_none**: AV text generated from the zero vector,
  the train-mean activation, or with no injection at all.
- **mean**: the train-mean activation vector used directly as `h_hat`
  (no model involved) — the raw-space floor every useful model must beat.

Gate requirements (report-embedded): per-control directional margin ≥ 0.1344,
rowwise win fraction ≥ 0.9, closed-generation fraction ≥ 0.95, usable
fraction ≥ 0.99, ≥ 100 independent families per split, and family-level
inference mandatory. The independent verifier
(`verify_nano_roundtrip_eval_report.py`) recomputes every control's paired
family-clustered statistics directly from the report's rowwise arrays,
enforces positive bootstrap lower bounds, sign-flip p ≤ threshold, win
fractions, checkpoint/tokenizer/dataset identity, and emits
`claim_scope: directional_av_to_ar_reconstruction` with
`raw_magnitude_claim_supported: false`.

## 8. Statistical methodology

The independent statistical unit is the **content family**, not the row.
For each control, the paired rowwise directional-loss delta
(control − candidate) is averaged within each family (equal cluster
weights); a nonparametric bootstrap over families (10,000 resamples, seed
20260709, percentile 95% interval) gives the confidence interval, and a
one-sided sign-flip permutation test over family effects (Monte-Carlo
100,000 samples, seed 20260709, add-one correction; exact enumeration below
20 clusters) gives the p-value. Rowwise win fractions and a top-5
concentration diagnostic are reported alongside. With 250/255 families the
Monte-Carlo sign-flip floor is p ≈ 1.0e-5, which is how the "~1e-5" values
below should be read. Functional comparisons (Section 10.2) use the same
family-clustered bootstrap on per-row metric deltas.

## 9. Component results

### 9.1 AR from teacher text (512 validation rows; verifier `1e62f111…2518`, passed)

| Quantity | Value |
|---|---:|
| Teacher directional MSE | 0.281703 |
| Teacher cosine | 0.859148 |
| Teacher FVE-NRM | 0.584534 |
| Teacher raw MSE | 8.537785 (train-mean predictor: 7.104776) |
| Teacher centered raw R² | **−0.201696** |
| Controls (directional MSE) | teacher_shuffled 0.968888; mean 0.678041; blank 0.756098; generic 0.781429; source_context 0.301252; source_raw 0.083248 |
| Rowwise wins (teacher vs.) | shuffled 512/512; blank 98.4%; generic 99.2%; mean 98.6% |

Teacher text recovers strong row-specific direction, but raw MSE is *worse*
than the constant train-mean predictor: no raw-magnitude claim. Two controls
bound interpretation: `source_context` (AR reading the raw source prefix
through the critic template) is statistically close to teacher text
(0.301252; teacher wins only 57.8% of rows), and `source_raw` (0.083248) is
far better than any explanation path — much of the recoverable direction is
predictable from the source text itself, and text about the *source* is
almost as good as text about the *activation*. The NLA claim therefore rests
on the round trip and controls, not on the AR component alone.

### 9.2 AV conditioning (512 validation rows; verifier `e4d2e2e9…f82e`, passed)

Mean response NLL of the teacher explanation under the AV, by conditioning
vector:

| Condition | NLL |
|---|---:|
| Real activation | **0.776775** |
| Shuffled activation (cross-family) | 1.311727 |
| Zero vector | 1.176494 |
| Train-mean vector | 1.237522 |
| No injection | 1.220974 |

Real-activation conditioning beats every control by 0.40–0.53 nats/token;
the injected activation carries row-specific information the AV uses.

## 10. End-to-end results

### 10.1 Round trip: h → AV text → AR reconstruction

Reports `948460b5…3c25` (validation) and `4802cd70…3123` (exploratory test);
verifiers `e1adedd4…4ae8` / `f3823bd3…6a71e`, both passed; gate
`passed=true` on both splits.

| Split | Rows | Families | AV-text dir. MSE | Teacher dir. MSE | Raw MSE | Centered raw R² | Norm ratio | Closed/usable |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Validation | 512 | 250 | **0.307004** | 0.304714 | 9.449079 | −0.326586 | 1.5326 | 512/512, 512/512 |
| Exploratory test | 512 | 255 | **0.319225** | 0.302637 | 9.647148 | −0.335374 | 1.5290 | 512/512, 512/512 |

AV-generated text reconstructs direction essentially at teacher-text level
(validation gap +0.0023; exploratory-test gap +0.0166; teacher text remains
slightly better rowwise, winning 54.3% of validation and 63.1% of
exploratory-test rows — parity, not superiority). All 1,024 *candidate*
(av_real) generations closed their `</explanation>` tag and passed
parse-quality screening (zero repetition loops, zero fallback-only rows);
among control generations, only the exploratory-test av_shuffled control
dipped to 510/512 closed, above the 0.95/0.99 gate thresholds.

Family-clustered control margins (control − candidate, directional MSE),
exploratory test:

| Control | Margin | 95% family CI | Rowwise wins | Sign-flip p |
|---|---:|---|---:|---:|
| mean | 0.361806 | [0.352926, 0.370762] | 99.80% | ~1e-5 |
| av_mean | 0.522112 | [0.511974, 0.532006] | 100% | ~1e-5 |
| av_none | 0.536178 | [0.524715, 0.548039] | 99.61% | ~1e-5 |
| av_shuffled | 0.645884 | [0.634535, 0.657831] | 100% | ~1e-5 |
| av_zero | 0.663983 | [0.654222, 0.674222] | 100% | ~1e-5 |

Validation margins are equivalent (mean 0.361979 [0.351920, 0.372440];
av_shuffled 0.659096 [0.646679, 0.671946]; all p ≈ 1e-5). The decisive
comparison is **av_real vs. av_shuffled/av_mean/av_none/av_zero**: identical
generation and scoring machinery, differing only in whether the *true row's*
activation conditioned the text. The ≥0.52 margins there are the direct
evidence that row-specific activation information flows through the language
bottleneck. Raw MSE, by contrast, is ~33% *worse* than the train-mean
predictor on both splits (negative centered raw R², norm ratio ~1.53): the
pipeline systematically overshoots magnitude, and no native raw-space claim
is supported.

### 10.2 Stored-snapshot counterfactual functional reinjection

Functional evaluation asks whether `h_hat` *behaves* like `h`: the
reconstruction is rescaled per-row to the stored activation's norm, injected
at the boundary position of the stored counterfactual forward pass, and the
patched next-token distribution is compared with the unpatched one. Because
strict fresh-forward identity fails (Section 11), this is **stored-snapshot
counterfactual reinjection**, not a fresh-forward claim; the verifier's
claim-scope field says exactly that. Reports `4f6116db…f9ff` /
`8cd9324f…44ab`; verifiers `efe2ab8d…4253` / `302a2f4d…0d7f`, both passed.

Exploratory test (per-variant means, 512 rows except shuffled = 88):

| Variant | KL(orig‖patched) | JS | Logit Pearson | Top-10 overlap | Top-50 overlap |
|---|---:|---:|---:|---:|---:|
| Stored gold (upper bound) | 0.0029 | 0.0007 | 0.9993 | 0.9721 | 0.9736 |
| **AV candidate** | **0.9495** | **0.1521** | **0.9078** | **0.6256** | **0.6393** |
| Teacher | 0.9701 | 0.1451 | 0.9119 | 0.6311 | 0.6453 |
| Mean | 4.1241 | 0.4887 | 0.7354 | 0.2174 | 0.2868 |
| Zero | 6.2975 | 0.5947 | 0.5769 | 0.1063 | 0.1696 |
| Shuffled | 9.5289 | 0.6563 | 0.6218 | 0.0318 | 0.0986 |

(Validation: candidate KL 1.0883, JS 0.1550, Pearson 0.9016, top-10 0.6227,
top-50 0.6379; teacher KL 1.1991.) The functional `shuffled` control here is
a *within-document rotation* of stored gold vectors — defined only for rows
whose document contributed ≥2 selected rows (hence 78/88 rows) — a
same-document control deliberately harder than, and distinct from, the
round-trip's cross-family shuffle. Candidate-vs-teacher family-clustered 95%
intervals include zero on **all** registered functional metrics on both
splits (e.g., exploratory-test KL improvement +0.0212 [−0.1032, +0.1497]);
candidate-vs-mean/zero/shuffled intervals are decisively positive.
Reinjecting the stored gold activation itself is near-exact (KL ≈ 0.003),
confirming the reinjection machinery; the ~0.95-nat gap between candidate
and gold is the information lost through the language bottleneck. Note that
per-row gold-norm rescaling means functional results assess *direction*
under an oracle norm, consistent with the directional claim scope.

### 10.3 Independent-AR replication (validation-only, cross-critic)

A second AR was initialized with a fresh seed-314159 value head and changed
router initialization (16/16 independence checks; verification file
`4639285f…d6e3`; canonical init-manifest SHA-256 `34e863f7…e941`) and trained
independently with the primary recipe (1,291 updates, validation-only
selection). It is *not* merely a data-order reseed of the primary AR.

- Component (512 validation rows, teacher text; report `368c84ca…65aa`,
  verifier passed): directional MSE 0.286169, cosine 0.856916, FVE-NRM
  0.577948 — within 0.0045 of the primary AR (0.281703 / 0.859148 /
  0.584534). Teacher beats shuffled 512/512 and mean 507/512.
- Cross-critic round trip (frozen primary-AV validation generations, cache
  SHA `80759667…42ab`, rescored by the independent AR; report
  `6f0829a6…52b`, verifier `dd3de6e1…4331`, passed): AV-text directional MSE
  **0.310963** vs. independent teacher-text 0.308533; 512/512 closed and
  usable; all five controls pass with positive family-clustered lower bounds
  (mean margin CI [0.348206, 0.368142]; av_shuffled [0.640363, 0.665274]);
  rowwise wins 99.80% (mean) and 100% (others); centered raw R² −0.399865.

This qualifies validation-only **cross-critic directional replication for
the one selected AV seed**: the signal in the generated text is not an
artifact of the specific primary AR that also served as the RL-era reward
model. It is not an independent AV-seed replication, not a test-split
result, and not a raw-magnitude result.

## 11. Activation-fidelity and magnitude analysis

**Fresh-forward identity fails; alignment is high.** A 64-row
validation-only audit under the deterministic profile (report
`5a457245…c999`; the only fidelity artifact — the companion non-mb8 run
OOM'd and produced no JSON) found: repeated fresh forwards bit-exact;
full-forward vs. extraction-forward bit-exact; but **all 64 rows fail** the
strict fresh-vs-stored tolerances (rel-L2 ≤ 0.01, 1−cos ≤ 1e-4,
max-abs ≤ 0.01) with per-row relative L2 mean 0.031329 and max 0.185983
(the report's aggregate block records mean 0.031405 / max 0.195980 — an
in-file field inconsistency disclosed in the open issues), mean cosine
0.999142 (min 0.983146), and centered raw R² 0.997255 when the stored vector
predicts the fresh one. The stored snapshot is a deterministic, replayable target that is
*directionally* almost identical to — but not numerically identical with —
the current runtime's fresh forward. All claims are therefore stored-snapshot
scoped; an earlier 32-row audit (mean 2.56%, max 7.73% rel-L2) reached the
same verdict.

**Magnitude is a global scale mismatch — post-hoc evidence only.** The
pipeline overshoots norms by ~1.53×. A one-parameter calibration
(`calibrate_nano_activation_magnitude.py`, report `facad9f4…923a`) fit a
single nonnegative origin scalar on validation *teacher* reconstructions
only (selected by raw MSE among {identity, origin-scalar, train-mean-scalar};
scalar = 0.560604) and applied it unchanged to AV predictions:

| Split | Raw MSE before → after | Centered raw R² before → after |
|---|---|---|
| Validation | 9.446685 → 3.648806 | −0.326250 → 0.487733 |
| Exploratory test | 9.647148 → 3.770353 | −0.335374 → 0.478102 |

Directional MSE is unchanged by construction. The exploratory-test
family-clustered 95% CI for the raw-MSE improvement is [5.634404, 6.123574],
candidate better in every bootstrap draw. This says much of the raw error is
one global scale factor — but the test boundary is exposed, the fit is
post-hoc, and native (uncalibrated) magnitude recovery remains unsupported.
(Validation "before" values here come from the July-16 immutable-cache
rescore, which differs from the July-15 report in the fourth decimal —
0.306969 vs. 0.307004 directional — a documented runtime-replay drift; both
reports are preserved and hashed.)

**Subgroup robustness (post-hoc).** Quartile edges fit on validation and
applied unchanged to the exploratory test across source-token length,
teacher word count, target activation norm, and family frequency (report
`fd24360b…72c`) yield 13 realized groups per split (the family-frequency
dimension collapses to one bin; project docs describing "16 bins" count the
requested, not realized, binning). Every group keeps every control's
family-clustered lower bound positive; the weakest exploratory-test group is
the lowest activation-norm quartile (directional MSE 0.370077; calibrated
centered raw R² 0.415606; minimum control lower bound 0.319044).

**Exposure audit: no in-corpus confirmatory boundary.** The fail-closed v6
selected-pair exposure audit (report `373e2988…088b`, inventory
`c193f2ef…9aabb`, joint manifest `9d68a894…bc20`) enumerated 136 exact
sources (selected-pair train/validation/test plus every historical
evaluation artifact), mapped all 28,665 unique historical documents with
**zero unmapped**, and found **all 5,009 families exposed**; split assignment
under a zero-exposure constraint is mathematically infeasible
(`passed=false` by design). An inventory of existing teacher tables (63
candidates, 53 usable) found no teacher-backed document ID outside the
exposed 0..38161 range. Under the standing rule not to generate new teacher
text mid-study, any *confirmatory* generalization claim requires a genuinely
external teacher-backed boundary. Repartitioning the same 275,396 rows
cannot fix this.

## 12. Failure history and correctness remediation

The protocol above exists because of documented failures. We list them
because they changed the publication standard, and because several would
have silently survived a conventional reporting pipeline.

1. **Packed-boundary contamination (found 2026-06-10).** Vendored
   Nemotron-H remote code passed `seq_idx=None`, letting Mamba state leak
   across samples packed into one THD microbatch; even after patching,
   packed critic training was not equivalent to the padded reward/eval path
   (reward/train MSE divergence 17.9–22% at step 0). Fix: padded masked
   critic microbatches with explicit last-token value indexing (live
   equivalence proof `max|ratio−1| = 0.0000`). All pre-fix AR/AV checkpoints
   were reclassified as contaminated scouting artifacts.
2. **LR-schedule bug.** The Miles actor unconditionally rewrote fresh SFT
   runs to constant LR; "cosine" labels on pre-fix runs describe requested,
   not executed, schedules. Caught by an LR-decay canary; fixed with a live
   scheduler patch, and a second resume-path variant was fixed later
   (patch 0018).
3. **Packed-position AV bug (found 2026-07-09, decisive).**
   `NemotronHForCausalLM.forward` accepted packed `position_ids` but dropped
   them before the backbone, so packed AV streams ran as one long sequence
   (packed-vs-padded response-NLL max abs difference 0.947 before the fix,
   0.0163–0.0221 after; commit `6abfe18`). Every packed-trained AV/RL actor
   checkpoint before the fix was invalidated for publication; a fail-closed
   packed-vs-padded gate now precedes any packed AV run.
4. **The July-8 RL headline and its invalidation.** An RL-tuned AV
   ("corrected-K3 hero", 342 GRPO updates against the frozen primary AR)
   passed a predeclared 512/512 gate with a 30.97%/32.34% relative
   directional improvement over SFT and was promoted. An independent
   publication audit (2026-07-08, preserved verbatim in
   `docs/reviews/2026-07-08-r33-rl-hero-publication-audit.md`) invalidated
   the headline: (a) half the SFT baseline's generations used a different
   generation prefix, inflating the effect from a fair ~23.6%/22.5% to
   30.97%/32.34%; (b) the result was same-critic (reward model = eval
   scorer) with cross-critic transfer verified only at update 16; (c) the
   metric was direction-only while raw-space centered R² was ≈ 0 — reported
   as "NMSE" without that qualification; (d) stop-guards were relaxed
   across retries until a run survived; (e) the 512-row eval covered ~13
   effective near-duplicate content families per split, making document-level
   CIs anti-conservative, with additional train/eval leakage through the
   critic. A protocol-matched salvage rescore (2026-07-09) of the retained
   RL actor against a correctly generated SFT baseline still showed a
   +20–23% relative directional gain under both the primary and a reseeded
   critic (family-clustered CIs positive; length-matched gain ~13–16%), but
   the entire lineage rests on the pre-remediation activation snapshot that
   failed identity audits and on exposed test families — it is internal
   hypothesis evidence only. **No RL improvement is claimed, and no
   publication-valid RL model currently exists.**

   A later family-clean online-RL internal hero is a separate lineage. It was
   trained for 342 online updates (approximately 43 hours and 65,664 generated
   responses) from the qualified clean SFT pair. Its retained iteration-342
   actor/critic pair reduces directional round-trip
   NMSE from `0.309055` to `0.224386` against an exact row- and
   generation-protocol-matched SFT baseline on 122 validation families
   (`27.4%` relative; raw MSE `9.5523 -> 7.2665`). It beats all required
   controls and closes 121/122 generations. This is strong validation-only
   evidence, recorded in evidence-table row 7.1; it is not included in the
   paper's publication claim until its artifacts are locally hash-bound, the
   result transfers through an independent critic and seed, and a new external
   boundary is evaluated once.
5. **Remediation.** Deterministic re-extraction with byte-identical replay
   (Section 5); exact-Jaccard content families replacing doc-id splits;
   generation-protocol hashing bound into every cached generation with
   fail-closed parity checks; a metric contract separating directional from
   raw recovery; family-clustered inference with ≥100-family minimums;
   validation-only selection with test-aware HPO removed; fail-closed
   verifiers for every report; a stale-claim documentation checker
   (`verify_docs_consistency.py`) that fails CI if the invalidated headline
   reappears as active; and registry entries for failed and invalidated
   runs, not only winners.
6. **Residual qualitative-tooling failure (2026-07-16).** The first
   source-grounded qualitative panel silently omitted the resolved source
   text column; it was invalidated and rebuilt fail-closed (corrected panel
   SHA-256 `4f5d6148…81c8`) — a small instance of the same
   silent-default class, caught by the new standards.

## 13. Limitations

1. **Directional only.** The primary metric ignores magnitude; native
   centered raw R² is negative on both splits. "Reconstruction" here always
   means direction.
2. **Stored snapshot only.** Strict fresh-forward identity fails (Section
   11); functional results are stored-snapshot counterfactual reinjection
   with per-row gold-norm rescaling.
3. **Exploratory boundary.** All 5,009 in-corpus families have historical
   exposure; the test split is family-disjoint from clean SFT training but
   not project-pristine. The split builder, moreover, only forbids exposed
   families from test, so validation carries no exposure guarantee at all.
4. **One AV seed.** Cross-critic replication covers the AR side only; AV
   training-seed variance is unmeasured.
5. **No semantic-quality claim.** Generated explanations are fluent and
   structurally clean (automatic screen: zero flags on the 100-example
   panel), but blinded human semantic review is incomplete; the historical
   audit found frequent hallucinated details in RL-era generations. The
   explanations are a lossy information channel, **not** faithful
   chain-of-thought or verified descriptions of model cognition.
6. **No layer claim.** R33-over-R27 would require a row-matched clean R27
   retrain; the layer choice rests on pre-fix scouting probes.
7. **Confounded ceiling.** Teacher text and source-context reconstruct
   comparably; the information the AR reads may substantially reflect
   source-predictable content rather than activation-specific computation.
   The av_shuffled/av_none margins bound, but do not eliminate, this
   concern.
8. **Single model, single boundary, English web text.** No cross-model,
   cross-layer (beyond scouting), or cross-domain evidence.
9. **Method divergences from [1]** (injection scale, frozen critic,
   architecture adapters) mean no upstream comparability.

## 14. Reproducibility statement

The release is organized around the fail-closed pair manifest
(`nano_nla_checkpoint_pair_manifest.v1`, release ID
`r33-clean-sft-av-ar-iter1291-20260715`, `qualified: true`, manifest SHA-256
`37166ce7…3edb`), which binds both checkpoint fingerprints, all six
verifier verdicts, the family manifest, dataset hashes, generation caches,
evaluation source archives (`3782fd11…2fd3`, `0db79a5d…dc9f`), and an
explicit limitations block. Local lightweight evidence:
`artifacts/runai_eval/r33-clean-sft-av-ar-qualified-20260715/` (archive
SHA-256 `b9043eae…24bc`), publication follow-up under
`artifacts/runai_eval/r33-clean-sft-publication-evidence-20260716/`, and
independent-AR evidence under
`artifacts/runai_eval/r33-independent-ar-publication-evidence-20260716/`.
Model payloads are preserved in internal S3 with verified object counts and
byte totals (independent AR HF: 10 files, 38,462,226,688 bytes, directory
SHA-256 `c2eea74f…1925`). Compact training curves for all 3,873 selected
optimizer steps: SHA-256 `7d9c22b9…2a6`. Verification: the release
attestation records 749 tests passed at its status date (cluster-side
environment); an independent local recheck during draft preparation ran the
repository's configured suite at 571 passed / 24 skipped (skips are
GPU/cluster-dependent) with the docs-consistency checker passing — the
count difference is environmental scope, not failures.

**Known gaps a re-implementer must respect.** (i) A no-weights public
release candidate exists (496 files, 6,859,370 bytes, archive SHA-256
`3eb8e64e…573a`, zero security findings) but **currently ships an obsolete
family-exposure report** (833 unmapped documents, 130 sources, a
4,612/356/41-family split table contradicting the frozen 4,504/250/255 split
shipped beside it — an early v2 failed attempt) instead of the authoritative
v6 audit, whose JSON is absent from the bundle while the bundle's own staged
docs cite the v6 numbers; the **current** builder config
(`configs/nano_release/r33_public_bundle_candidate.yaml`, SHA-256
`03ca4a47…fa7b`) still stages the obsolete file, so restaging without a
config fix reproduces the defect. The archive must not be described as
final or self-contained until the config is fixed, the v6 artifacts are
added, and the bundle is restaged, re-audited, and re-attested (see open
issues). (ii) The upstream NLA
import has no verifiable upstream commit pin. (iii) No `pip freeze` was
retained for the primary AV/AR runs (the independent-AR environment is
snapshotted). (iv) `weights_included=false` and
`legal_clearance_granted=false` in the machine-readable attestation; nothing
public exists yet.

## 15. Ethics, privacy, and licensing

Source text is web-crawled FineWeb (ODC-By-1.0, subject to Common Crawl
terms); generated and teacher text may reflect its biases and harms. An
automatic release-text audit (report `00e501ff…3419`) scanned all 1,024
frozen generated explanations: zero configured PII/credential/internal-path
findings and zero source-copying failures (panel maximum contiguous source
match: 5 words). Fourteen phone-like patterns occur in *source excerpts*
only; source-containing examples stay internal pending human adjudication or
redaction. This is automated triage, not proof of privacy or consent. The
repository has no top-level license; the base model is governed by the
NVIDIA Open Model License per its local model card (exact acquired-model
terms unconfirmed); teacher outputs were generated through an NVIDIA
inference API whose governing service agreement is unidentified — the
highest-risk legal blocker; upstream NLA code is Apache-2.0 (Anthropic PBC).
No weights, source, teacher text, or examples may be published before
owner/legal sign-off. Interpretability outputs that *look* like
explanations invite over-trust; Section 13.5's non-faithfulness caveat is a
safety-relevant part of the claim, not boilerplate.

## 16. Conclusion

Under a deterministic stored-snapshot protocol with content-family-disjoint
splits and fail-closed verification, a supervised Nano30B NLA pair at
boundary R33 demonstrates that AV-generated natural language carries
row-specific activation information: AR recovers activation direction from
generated text at near-teacher level (0.307/0.319 directional MSE vs.
0.305/0.303), beats every semantic and activation control with positive
family-clustered intervals, substitutes functionally for the stored
activation at teacher level in counterfactual reinjection, and replicates
through an independently initialized and trained reconstructor. Equally
firmly: raw magnitude is not natively recovered, fresh-forward identity
fails, the evaluation boundary is exploratory because the corpus is fully
exposure-saturated, one AV seed is qualified, semantic faithfulness is
unreviewed, and the historical July-8 RL gain is invalid. A separate
family-clean internal AV+AR pair has a strong matched validation-only RL result,
but has not cleared replication or an external test boundary. The evidence supports (a)
release of the checkpoint pair as a research artifact once legal and human
review complete, and (b) an exploratory technical report with exactly the
claim stated here. It does not support a confirmatory scientific
generalization claim — that requires an external teacher-backed boundary —
and it supports no publication-level RL claim. We believe the verifier-bound negative results
are as load-bearing a contribution as the positive ones.

## References

[1] K. Fraser-Taliente, S. Kantamneni, E. Ong, et al. "Natural Language
Autoencoders Produce Unsupervised Explanations of LLM Activations."
Transformer Circuits Thread, 2026.
https://transformer-circuits.pub/2026/nla/index.html

[2] natural_language_autoencoders reference implementation (Apache-2.0,
Anthropic PBC). https://github.com/kitft/natural_language_autoencoders
(vendored production fork under `external/natural_language_autoencoders/`;
upstream commit pin unverifiable — see open issues).

[3] NVIDIA. NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 model card (local copy at
the cluster model root; NVIDIA Open Model License). [TODO: confirm the
public model-card URL and any Nemotron-3/Nemotron-H technical report
citation from official sources before external use.]

[4] HuggingFaceFW/fineweb dataset card (ODC-By-1.0).
https://huggingface.co/datasets/HuggingFaceFW/fineweb [TODO: add the
FineWeb paper citation after verifying venue/authors against the official
record.]

[5] Project measurement contract and audit record:
`docs/methods/measurement_contract.md`,
`docs/reviews/2026-07-08-r33-rl-hero-publication-audit.md`,
`docs/runs/r33_clean_sft_av_ar_20260715.md`.
