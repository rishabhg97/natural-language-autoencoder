# NLA Visualization Frontier Ideas

Status: research ideation, `2026-07-16`. Companion documents:
`docs/viz/nla_visualization_ideas.md` (the tiered design proposal, cited below
as "companion §x.y") and `docs/viz/nla_roundtrip_visualiser.html` (the
pedagogical explainer). The dashboard prototype referenced elsewhere was
deleted on request; no `nla_dashboard.html` currently exists.

> [!CAUTION]
> Everything in this document is **exploratory-framing only**. Nothing here
> creates a confirmatory claim, consumes sealed rows for confirmatory
> purposes, or upgrades the qualified release scope. All claims inherit the
> boundaries of `docs/methods/measurement_contract.md` and
> `docs/runs/r33_clean_sft_av_ar_20260715.md`: directional (not
> raw-magnitude) reconstruction, stored-snapshot (not fresh-forward)
> functional semantics, family-clustered uncertainty, exploratory test split,
> single qualified AV seed, invalidated RL history. Generated descriptions are
> learned encodings of selected activations — **not** the model's literal
> thoughts, hidden chain of thought, or verified causal explanations.

Method note: per the ideation protocol, 32 candidate directions were generated
independently *before* reading the companion proposal (archived in the session
scratchpad), then deduplicated and merged against it. Where a concept below
extends or unifies a companion idea, its lineage is tagged; concepts tagged
`[new]` do not appear in the companion proposal.

Numbers quoted below come from the qualified run record and current-state docs
unless tagged `[exploratory session finding, companion doc]`, which marks
offline cache analyses from the companion proposal that have not passed a
hash-bound verifier.

---

## 1. First-Principles Opportunity

An NLA gives an activation vector four properties no other interpretability
object has simultaneously. Each property converts a class of previously
awkward questions into direct experiments:

**Readable.** The latent code is English. It can be diffed, searched,
sectioned, truncated, translated, and judged by a person with no linear
algebra. Error analysis becomes *literary*: a failed reconstruction is not a
residual norm, it is a sentence you can read and disagree with. The unit of
analysis drops below the vector — to the clause, the word, the token — and
standard NLP machinery (edit distance, paraphrase, occlusion) becomes
interpretability machinery.

**Editable.** The latent is writable with a keyboard. Probes read; SAEs
decompose; steering vectors push along fixed directions someone else derived.
Only a readable-writable code lets a user *author* a counterfactual internal
state in the same language they think in, without knowing what a basis is.
This is the first natural-language API to activation space, and it changes who
can run interventions.

**Reconstructable.** AV→AR closes a measurable channel. That makes the code
subject to the full apparatus of communication theory: rate–distortion (how
much direction do k words buy?), redundancy (does the code survive typos?),
identifiability (how many activations can 512 descriptions distinguish?), and
invariance (does meaning-preserving rewording preserve the decoded vector?).
Crucially the channel has *two independent decoders* (primary and
seed-`314159` AR), so "what the text carries" can be separated — partially —
from "what one reader extracts."

**Reinjectable.** Reconstruction quality becomes something the model *does*.
Under the frozen stored-snapshot protocol, a reconstructed vector patched at
boundary R33 yields next-token distributions that can be compared against the
original (test KL `0.9495` for the AV candidate versus `4.12` mean / `6.30`
zero / `9.53` shuffled). Claims graduate from "correlates with" to "suffices
to approximately restore, under this protocol" — and edits graduate from
descriptive to causal, with dose–response and mediation designs available.

### The deepest unanswered questions this interface can attack

1. **Where do the bits die?** The reconstruction ladder (source_raw `≈0.083`
   → teacher/AV text `≈0.30` → mean control `0.678` → shuffled `0.969`,
   directional MSE) says most row-specific information survives text but a
   large residual does not. Is the residual *systematic* (a nameable subspace
   English never carries) or *idiosyncratic* (per-row tail variance)? Is the
   loss committed at vector→text or text→vector? The twin-critic agreement
   observation (`cos≈0.993` between the two ARs' reconstructions
   [exploratory session finding, companion doc]) points to vector→text, but
   shared teacher data and shared base-model geometry could correlate the two
   readers' errors — this needs a designed experiment, not a scalar.
2. **Is the code semantic, or a private cipher?** The entire scientific value
   of a *natural-language* autoencoder rests on the words meaning what they
   say. Paraphrase invariance, human-vs-AR usability decomposition, and
   cross-reader agreement can measure semanticity — and must, before any RL
   resumes, because a near-deterministic frozen critic is a near-perfectly
   exploitable reward (the July-8 lesson).
3. **Is language a chart or a lattice?** Does the code behave like a
   continuous coordinate chart on the activation manifold (smooth text change
   under smooth activation change) or like a symbol lattice that quantizes
   activation space into discrete describable cells with snap boundaries?
   Interpolation, iteration fixed points, and contraction structure decide.
4. **What is the writable set?** AR maps *all of text space* into activation
   space. What is the image? Which regions of the activation manifold are
   language-addressable, and what do unreachable holes contain? Interventions
   can only go where language can point.
5. **Do described differences cause the described behaviors?** If the code
   says "formal legal register" and we edit it to "casual food blog," does
   behavior move in the *named* direction, at a dose that scales, with
   paraphrase placebos flat? This is the falsifiable core of "readable
   control surface."
6. **What are the timescales of internal state?** Reading the code at every
   position turns R33 into a signal with named channels — which clauses are
   slow variables (topic, genre) and which are fast (local syntax, next-token
   machinery)? No other method yields a *diffable* per-token state readout.
7. **When must the instrument not be trusted?** Confabulation from empty
   signal, prior collapse off-distribution, fresh-vs-stored drift, magnitude
   blindness, and critic exploitation are all known or suspected. A mature
   instrument displays its own failure conditions as prominently as its
   readings.

---

## 2. Frontier Concepts

Thirty-two concepts, grouped in eight families. Shorthand used throughout:

**Data assets.** `CACHE` = frozen prediction caches
(`*_roundtrip_predictions.npz`: 2,688-d targets plus all reconstruction
variants, both splits, both critics). `TEXT` = 1,024 frozen generations with
per-control generations, teacher explanations, and the 100-row
source-resolved qualitative panel. `FUNC` = per-row functional-recovery
reports (KL/JS/logit-Pearson/top-k). `PARQ` = 275,396-row activation parquets
with `token_text`/`token_position` provenance (cluster). `AR` = frozen AR HF
checkpoint (~38 GB). `IND-AR` = independent seed-`314159` AR. `AV` = frozen
AV checkpoint. `BASE` = Nano30B target model. `SAE` = a hypothetical R33
sparse autoencoder (not yet trained).

**Compute levels.** `L0` = laptop, existing artifacts only. `L1` = one H100
(AR-only forwards). `L2` = two GPUs (AV generation and/or BASE+AR
reinjection, phased). `L3` = ~158 GB resident (BASE+AV+AR) live service
(2×H200 or 4×H100-NVL). `L4` = new training or extraction (SAE, new
boundaries, external corpora).

**Standing rules for every concept.** Fit nothing on test (validation-fitted
thresholds applied unchanged, as in the subgroup audit); show family-clustered
uncertainty; overlay the mean-control ghost wherever a reconstruction quality
is shown; badge every panel `frozen / precomputed / live-exploratory`; sealed
rows never enter public demos.

---

### Family A — Geometry of the language channel

#### C1. Residual Anatomy Atlas `[unifies companion §0.3 + §C.14]`
The systematic study of `h − ĥ`: what English drops, and whether it is nameable.
- **Research question.** Is round-trip error concentrated in a stable
  low-dimensional subspace (a "language-blind" subspace of R33), and can that
  subspace be characterized in words, token statistics, or SAE features?
- **User experience.** A spectral strip shows per-PC transmitted vs residual
  energy (companion §0.3). New: a *residual PCA* pane — principal directions
  of the error vectors themselves — where clicking a residual direction opens
  (a) the rows/passages that load on it most, (b) their explanations with the
  shared missing content highlighted by contrast mining, and (c) a
  "speak-the-blind-spot" panel: AV verbalizations of `mean + α·residual-PC`
  (companion §C.14) so the model narrates the direction it fails to transmit.
- **Visual encoding.** Transmission-coefficient strip (astronomy-style
  absorption spectrum); residual scree with stability bands across
  split/critic; small-multiple passage cards; α-slider text morph.
- **Interaction model.** Select PC → linked rows; toggle
  primary/independent critic; toggle validation/test; α-slider for blind-spot
  verbalization; export direction as a hypothesis card.
- **Models & data.** CACHE, TEXT for L0 core; AV at L2 for blind-spot
  verbalization batch.
- **Mode.** Static core; blind-spot verbalizations precomputed.
- **Compute.** L0 core, L2 for one verbalization batch.
- **Controls.** Residual PCA of the *teacher-text* reconstruction (is the
  blind spot the AV's or the format's?); shuffled-control residuals (should
  be isotropic around the prior); split/critic stability of the top residual
  directions (cross-validated subspace angles).
- **Measurements.** Residual energy fraction per target-PC band; subspace
  principal angles across critics/splits; fraction of residual variance in
  top-k residual PCs; blind-spot verbalization consistency across α.
- **Success means.** A stable, cross-critic, cross-split residual subspace
  with a coherent verbal/token-statistic signature — a concrete "what English
  cannot say about R33" object, and the natural target for format changes
  (lexical anchors, teacher v2).
- **Failure means.** Residual is isotropic/unstable → the loss is
  rate-limited tail variance, not a nameable subspace; format engineering
  should expect diffuse, not targeted, gains. That is a publishable negative.
- **Confounds.** PCA basis instability at 512 rows; residuals contaminated by
  the global magnitude mismatch (analyze direction-only residuals and
  calibrated residuals separately); teacher-format ceiling vs AV limitation.
- **Misleading-risk.** Medium: users may read blind-spot verbalizations as
  the model "describing its blind spot" — they are AV outputs on synthetic
  off-manifold inputs and must be badged as such.
- **Novelty.** SAE/probe work studies what *is* represented; a systematic,
  speakable census of what a readable channel *drops* has no analogue.
- **Difficulty.** Days (core); +1 GPU session for verbalizations.
- **Demo appeal.** Medium-high; the spectrum strip is a strong figure.

#### C2. Two Charts, One Manifold `[new; extends companion §C.12]`
Dual-path interpolation: compare the path a straight line takes in activation
space with the path a straight edit takes in language space.
- **Research question.** Does the language code induce a metric on activation
  space compatible with the ambient one? Where does the chart tear (smooth
  slerp → discontinuous text) or fold (smooth text edit → discontinuous
  reconstruction)?
- **User experience.** Pick rows A and B. Top rail: slerp
  `h_A→h_B` in activation space, verbalize waypoints (companion §C.12's Alley).
  Bottom rail: interpolate *in text* — a sequence of minimal edits morphing
  `z_A` into `z_B` (LLM-authored, human-checkable) — AR-encode each waypoint.
  Between rails, a "tear gauge": per-step text edit distance (top) and per-step
  activation displacement (bottom), with snap points flagged.
- **Visual encoding.** Two parallel rails of waypoints on the cached PC map
  connected by correspondence lines; step-size bar charts per rail; text
  diffs at snap points.
- **Interaction model.** Choose within-family vs cross-family pairs; scrub the
  interpolation; click a snap to read the text transition; swap AR critic.
- **Models & data.** CACHE for endpoints; AV (verbalize waypoints) + AR
  (encode text waypoints); an external LLM or template rules for text
  interpolation sequences.
- **Mode.** Precomputed batches (dozens of pairs); optional live pair picker.
- **Compute.** L2 precompute; L3 if live.
- **Controls.** Linear vs spherical interpolation (magnitude confound);
  token-level random edit paths of matched edit distance (does *any* text
  path move ĥ smoothly, or only meaning-changing ones?); within- vs
  cross-family pairs (expected smooth vs snap).
- **Measurements.** Lipschitz-style ratios (Δtext vs Δĥ, Δh vs Δz) per step;
  snap-point counts vs pair distance; agreement between the two paths'
  midpoints (cos of mid-activation vs encoded mid-text).
- **Success means.** Within-family pairs interpolate smoothly on both rails
  and the rails agree → language behaves as a local chart; snap structure on
  cross-family pairs maps the chart's cell boundaries — the "symbol lattice"
  granularity becomes measurable.
- **Failure means.** Ubiquitous snapping even within families → the code is a
  classifier-like quantizer; continuous steering via text should not be
  expected, and surgery (C15) results must be interpreted as cell jumps.
- **Confounds.** Off-manifold interpolants (slerp between real activations
  need not be a realizable activation — verbalizations there are AV
  extrapolations); text-interpolation authorship bias.
- **Misleading-risk.** Medium: interpolated waypoints look like "states the
  model passed through"; they are synthetic probes and must be labeled.
- **Novelty.** Latent interpolation is standard in VAEs/GANs; *comparing an
  induced language-metric against the ambient activation metric* is new to
  interpretability.
- **Difficulty.** ~1 week including the text-interpolation harness.
- **Demo appeal.** High — morphing text next to a bending path is striking.

#### C3. Semantic Telephone Observatory `[extends companion §C.13]`
Iterated `h→z→ĥ→z′→…` as a dynamical system: attractors, basins, contraction.
- **Research question.** Is the round trip a contraction toward a small set of
  fixed-point descriptions ("eigen-descriptions")? What do the attractors say,
  and how fast does row-specific detail evaporate?
- **User experience.** Launch a cohort of rows through k iterations. A basin
  map shows trajectories spiraling on the PC plane; a text panel plays each
  row's description history as a diff filmstrip — watch which clauses
  survive iteration and which sublimate. An attractor census table lists
  discovered fixed texts with basin sizes.
- **Visual encoding.** Trajectory streamlines with iteration-order color;
  per-iteration cos-to-original decay curves with family-clustered bands;
  attractor cards (text + basin size + mean escape distance).
- **Interaction model.** Seed from validation rows, controls, OOD points
  (C5), or hand-written text; step/play; pin two trajectories to compare.
- **Models & data.** AV + AR; CACHE for context map.
- **Mode.** Precomputed cohorts; live seeding in playground mode.
- **Compute.** L2–L3 (each iteration is one AV generation + one AR forward).
- **Controls.** Teacher-text seeds (does iteration from human-style text find
  the same attractors?); temperature-0 vs sampled AV (stochastic vs
  deterministic dynamics); mean/zero-vector seeds (prior attractor location).
- **Measurements.** Per-step contraction ratio distribution; number and
  entropy of distinct attractors per 512 seeds; steps-to-ε-stability;
  attractor overlap between critics.
- **Success means.** A small, interpretable attractor set with measured
  contraction — a compact statement of "what the channel preserves under
  self-communication," and a prior-collapse map for free.
- **Failure means.** Wandering non-convergent trajectories → the loop is
  noise-dominated; single-pass metrics overstate stability of the code.
- **Confounds.** Sampling temperature; parse-failure fallbacks silently
  injecting empty text (must surface parse state per step).
- **Misleading-risk.** Low-medium with iteration badges; avoid "the model's
  self-image converges" phrasing.
- **Novelty.** Fixed-point analysis of a *readable* autoencoding loop — the
  attractors are sentences, which no vector autoencoder can offer.
- **Difficulty.** Days on top of the serving harness.
- **Demo appeal.** High — "activation telephone" is instantly graspable.

#### C4. Lump & Split Atlas `[new]`
Partition comparison: neighborhoods induced by text similarity vs activation
geometry.
- **Research question.** Where does language *lump* geometry (distinct
  activation clusters share indistinguishable descriptions) and where does it
  *split* it (distinct descriptions for geometrically identical states —
  distinctions without a difference)?
- **User experience.** Two clusterings of the same 1,024 rows — one on
  activation cosine, one on explanation-text embedding — joined by an
  alluvial diagram. Thick crossing ribbons are the interesting anomalies;
  clicking a "lump" ribbon shows geometrically-far rows whose explanations
  read the same (with the texts side by side); a "split" ribbon shows
  near-identical activations described differently.
- **Visual encoding.** Alluvial/contingency ribbon; per-ribbon exemplar
  cards; a scatter of within-cluster text-similarity vs activation-similarity
  with lump/split quadrants.
- **Interaction model.** Adjust cluster granularity (co-clustering tree cut);
  select ribbon → row pairs; send any pair to C2 (interpolate) or C15
  (surgery) for follow-up.
- **Models & data.** CACHE + TEXT + an off-the-shelf text embedder (local);
  no model calls.
- **Mode.** Static.
- **Compute.** L0.
- **Controls.** Teacher-text partition as a third column (is lumping the
  AV's or the format's?); random-partition baseline for ribbon-crossing
  rates; family structure overlay (families should co-cluster in both).
- **Measurements.** Adjusted mutual information between partitions;
  lump/split rates vs granularity; retrieval-confusion overlap (are C10's
  confusions exactly the lumps?).
- **Success means.** Lumps localize the retrieval failures and identify
  description-vocabulary gaps (candidate teacher-v2 targets); splits, if
  present, flag confabulated distinctions — a concrete faithfulness warning.
- **Failure means.** Partitions agree almost everywhere → text similarity is
  a faithful proxy for R33 geometry at this granularity; that itself is a
  strong (and citable) positive.
- **Confounds.** Text-embedder bias defines "same description"; mitigate with
  two embedders + a token-overlap metric.
- **Misleading-risk.** Low; the view is explicitly comparative.
- **Novelty.** Cluster-agreement analysis is standard; running it between a
  model's *own generated language* and its geometry is new, and uniquely
  possible here.
- **Difficulty.** Days.
- **Demo appeal.** Medium; the anomaly cards are the hook.

#### C5. The Writable Set `[new]`
Map the image of language in activation space: where can English point?
- **Research question.** What subset of R33 space is reachable by AR from
  *any* text? Do reconstructions from diverse text cover the activation
  manifold, or collapse onto a low-dimensional "describable shell" with holes?
- **User experience.** A coverage atlas: the activation manifold (sampled
  from PARQ) rendered as a density field; overlaid, the AR-image cloud from a
  broad text battery — frozen explanations, teacher texts, random FineWeb
  sentences, template permutations, other languages, keyboard mash,
  adversarial suffixes. Terra-incognita regions (activation density with no
  language cover) glow; clicking one shows its nearest real rows and their
  (failing) descriptions.
- **Visual encoding.** Cartographic density map with hatched unreachable
  regions; per-text-battery color layers with toggles; reachability histogram
  (distance from each real activation to nearest AR image point).
- **Interaction model.** Toggle text batteries; brush a region → nearest
  texts that land there; "try to reach" mode: type text live and watch where
  it lands (L1).
- **Models & data.** AR only + PARQ sample + text batteries.
- **Mode.** Precomputed; optional live try-to-reach.
- **Compute.** L1 (thousands of AR forwards).
- **Controls.** Norm handling declared (AR images vs unit-normalized);
  matched-size random subspace baseline for coverage statistics; battery
  ablations (does coverage come only from in-format explanations?).
- **Measurements.** Coverage fraction at radius r (directional); intrinsic
  dimension of AR image vs activation sample; hole census with per-hole
  nearest-row profiles; battery-specific coverage gains.
- **Success means.** A quantitative writability map — surgery and steering
  claims (C15–C18) can be scoped to the reachable set, and holes become
  hypotheses ("R33 content English cannot address": candidate token-identity
  or positional machinery).
- **Failure means.** If AR images cover almost everything, the value head is
  expressive enough that "reachability" never binds — good news for
  intervention work, and the atlas becomes a null result worth stating.
- **Confounds.** 2-D projection artifacts (report coverage in full dimension,
  project only for display); AR trained only on explanation-format text (OOD
  text may map degenerately — that *is* the finding, but separate format-OOD
  from content-OOD).
- **Misleading-risk.** Medium: "holes" in a projection can be projection
  lies; all hole claims must be verified in native dimension.
- **Novelty.** Reachable-set analysis exists in control theory; nobody has
  drawn the reachable set of *natural language* over a model's state space.
- **Difficulty.** ~1 week.
- **Demo appeal.** High — "the map of what words can touch."

#### C6. Norm Blind-Spot Explorer `[unifies companion §0.8c + §B.7 + §C.10]`
One instrument for the entire magnitude caveat: shrinkage geometry,
calibration, and behavioral consequence.
- **Research question.** Is the raw-magnitude failure fully explained by
  angular-uncertainty shrinkage (the validation-fitted `0.560604` scalar),
  and does magnitude matter *behaviorally* at reinjection?
- **User experience.** Three linked panes. (1) Shrinkage cone: per-row
  optimal scalar vs cos, with the `s* ≈ cos·(‖h‖/‖ĥ‖)` construction drawn.
  (2) Calibration slider: apply a global scalar and watch raw MSE / centered
  R2 respond (validation-fitted value marked; test values displayed as
  exploratory). (3) Norm dial (companion §B.7): behavioral metrics vs
  injected norm of `unit(ĥ)` swept 0→2× true norm, with oracle-norm, raw-AR,
  and calibrated positions flagged; plus the injection-scale sweep for the AV
  channel (companion §C.10, addressing the never-re-derived scale `75`).
- **Visual encoding.** Cone scatter with theoretical curve; slider-coupled
  metric gauges; dose–response curves with family-clustered bands.
- **Interaction model.** Slider, per-row inspection, sweep playback; toggle
  split and critic.
- **Models & data.** CACHE (panes 1–2); BASE+AR reinjection sweeps (pane 3);
  AV for the scale-75 sweep.
- **Mode.** Panes 1–2 static; pane 3 precomputed sweeps.
- **Compute.** L0 + L2 sweep batches.
- **Controls.** Teacher-text reconstructions under the same scalar; per-norm-
  band subgroup views (the weakest qualified bin is the lowest-norm quartile,
  dMSE `0.370077`); random-direction norm sweeps (is behavior sensitive to
  norm per se or to direction×norm?).
- **Measurements.** Scalar-fit residual structure; behavioral basin width vs
  norm; KL at oracle vs calibrated vs raw norm.
- **Success means.** Either magnitude is behaviorally forgiving (basin wide →
  the directional-only claim is functionally sufficient) or it is not (basin
  narrow → calibrated norms must ship with any intervention tool). Both
  outcomes retire an open caveat with a picture.
- **Failure means.** Non-monotone or row-inconsistent norm response →
  magnitude interacts with content; the global-scalar story is incomplete.
- **Confounds.** Stored-snapshot scope; norm effects entangled with
  injection-scale effects in the AV pane (sweep them separately).
- **Misleading-risk.** Low — this panel *is* the honesty device.
- **Novelty.** Turning a metric caveat into a behavioral dose–response
  instrument is unusual; most tools bury normalization choices.
- **Difficulty.** Days + one GPU session.
- **Demo appeal.** Medium; scientifically load-bearing.

---

### Family B — Information theory of the code

#### C7. Words-Buy-Direction `[extends companion §A.3]`
Per-row rate–distortion where the rate axis is literally words spent.
- **Research question.** How is reconstruction distributed across the
  explanation's length? Do AV and teacher spend words with different marginal
  efficiency, and is there a knee where additional words buy nothing?
- **User experience.** Scrub a slider "words revealed k=0…N": the row's ĥ
  marker glides toward h on the PC map while the text unmasks word by word;
  a per-word marginal-gain bar sits under each token. Aggregate pane:
  median rate–distortion curves (AV vs teacher vs section-shuffled text) with
  family bands; a "knee atlas" scatter of per-row knee positions.
- **Visual encoding.** Typographic unmasking; token-level bar chart; monotone
  ribbon curves; knee scatter colored by content family frequency.
- **Interaction model.** Slider; click a token to see its marginal Δcos;
  sort rows by knee position; toggle prefix-truncation vs random-word-subset
  ordering.
- **Models & data.** AR + TEXT (~N forwards per row; 100–512 rows).
- **Mode.** Precomputed.
- **Compute.** L1.
- **Controls.** Random word order (is the gain front-loaded by syntax or by
  position?); teacher curves; length-matched controls from the gate's
  existing `length_control` analysis (guards the reported length–dMSE
  correlation, `−0.108` under SFT [exploratory session finding, companion
  doc]).
- **Measurements.** Marginal Δcos per token; knee position distribution;
  area-under-RD-curve per condition; AV-vs-teacher word-efficiency ratio.
- **Success means.** A defensible "direction per word" accounting — the
  format-budget argument (3–5 lines) becomes quantitative, directly informing
  teacher-v2 / lexical-anchor decisions (research memo NT-6/MT-3).
- **Failure means.** Flat-then-cliff curves ending at the final tokens would
  suggest suffix-anchored extraction artifacts rather than distributed
  content — an evaluator finding, not a channel finding.
- **Confounds.** Truncated text is OOD for the AR (mid-sentence cuts);
  mitigate with clause-boundary truncation variant.
- **Misleading-risk.** Low-medium; label "AR-usable information," not
  "information in the text" (a stronger decoder might read more).
- **Novelty.** Rate–distortion curves whose rate axis is human-readable words
  — and per-token marginal value typography — have no interp analogue.
- **Difficulty.** Days.
- **Demo appeal.** High; the unmasking animation explains the project in 10 s.

#### C8. Code Attribution Lens `[unifies companion §A.1 + §A.2; display from independent idea "clause→subspace ribbons"]`
Occlusion saliency and exact section Shapley on the code, mapped to subspaces.
- **Research question.** Which words and which of the four explanation
  sections carry which subspace of the reconstruction — and is the mapping
  stable across rows, critics, and splits?
- **User experience.** Every explanation renders with per-word deletion tint
  (Token Bounty). A section pane shows exact Shapley values over the 2⁴
  section subsets. New display: a bipartite ribbon — sections on the left,
  target-PC bands on the right, ribbon width = Shapley-attributed recovery of
  that band — the "which sentence carries which part of the state" figure.
- **Visual encoding.** Text tinting; Shapley waterfall per row; aggregate
  bipartite ribbons with stability whiskers.
- **Interaction model.** Click word/section → recomputed ĥ displacement on
  the map; pin two rows; aggregate/per-row toggle; critic toggle.
- **Models & data.** AR + TEXT (16 subset forwards per row for Shapley;
  ~60–150 for word occlusion).
- **Mode.** Precomputed (100–512 rows).
- **Compute.** L1.
- **Controls.** Filler-word deletions as saliency floor; C21's null-text
  log-odds as the interpretive baseline; section-*shuffled* (order) variant
  to separate content from position-in-template.
- **Measurements.** Shapley per section per PC band; word-saliency
  distributions by POS/content-class; cross-critic saliency rank correlation.
- **Success means.** A reproducible section→subspace dictionary; edits in C15
  can then be *targeted* ("to move the syntax subspace, edit section 3").
- **Failure means.** Saliency concentrated on template scaffold or
  punctuation → private-code alarm (feeds C20/F4 directly).
- **Confounds.** Deletion OOD-ness; section interactions beyond Shapley's
  additive summary (report interaction terms for the 2-way pairs).
- **Misleading-risk.** Medium: occlusion saliency measures the AR's reading,
  not the AV's intent; label accordingly.
- **Novelty.** Exact Shapley over a *generated explanation's structure*
  against a vector target is NLA-native; no probe/SAE equivalent exists.
- **Difficulty.** Days.
- **Demo appeal.** High (tinted text is immediately legible).

#### C9. Consensus Constellation `[extends companion §2.4]`
Multiple tellings of one activation: sampling noise vs channel ceiling.
- **Research question.** Is the residual error absent from *every* telling
  (channel ceiling) or scattered across tellings (recoverable by ensembling)?
- **User experience.** For one h, k sampled explanations fan out as a
  constellation of ĥ points around the target; a consensus marker shows the
  average-vector reconstruction; a live button adds more samples and updates
  the ensemble curve (cos vs k).
- **Visual encoding.** Constellation on the PC map with target crosshair;
  ensemble-gain curve; per-telling text cards ranked by cos.
- **Interaction model.** Sample-more button; per-telling inspection; row
  gallery sorted by fan geometry (tight-displaced vs straddling).
- **Models & data.** AV + AR; validation rows.
- **Mode.** Precomputed cohort + live single-row mode.
- **Compute.** L2–L3.
- **Controls.** Temperature sweep; teacher-text position in the fan;
  averaging in text space (an LLM merges the k tellings into one text, then
  AR) vs vector space — separates linguistic from geometric ensembling.
- **Measurements.** Ensemble cos vs k with family bands; fan anisotropy vs
  residual-PC alignment (does sampling variance point along C1's blind
  subspace?); fraction of rows where best-of-k beats teacher.
- **Success means.** Clean decision evidence: straddling fans ⇒ RAFT/best-of-k
  (research memo MT-1/MT-2) has headroom; tight displaced fans ⇒ the ceiling
  is informational and RL would chase noise.
- **Failure means.** High parse-failure rates at useful temperatures —
  a generation-health finding that bounds ensembling in practice.
- **Confounds.** Temperature choice; vector averaging inflating cos by
  shrinking toward mean (report cos-to-mean alongside).
- **Misleading-risk.** Low with the mean-ghost overlay.
- **Novelty.** "Ensemble the *tellings*, not the models" — and the
  text-space-merge control — are new.
- **Difficulty.** Days on serving harness.
- **Demo appeal.** High; "one thought, many tellings" reads instantly.

#### C10. Capacity Ladder `[extends companion §0.2; adds stratification + explicit assumptions]`
Identification information of descriptions, with the assumptions on the wall.
- **Research question.** How many activations can descriptions distinguish,
  under what prior, and where does identifiability degrade (domain, position,
  norm band, family frequency)?
- **User experience.** The retrieval curve (top-1 accuracy vs gallery size N,
  log₂N bits axis; companion reports `94.3%` top-1 at N=512 [exploratory
  session finding, companion doc]) with an always-visible assumptions card:
  *"Fano-style lower bound on identification information about which corpus
  row, uniform prior over the gallery; NOT the channel capacity of
  explanation-space."* New: stratified small multiples per subgroup-audit
  bins; confusion graph where every edge opens the two confused explanations.
- **Visual encoding.** Monotone curves + bands; bits axis; stratified small
  multiples; confusion graph colored same-doc/same-family/stranger.
- **Interaction model.** Adjust N; brush strata; click confusion edge →
  side-by-side texts with the undistinguished content highlighted.
- **Models & data.** CACHE (all-pairs cos suffices); TEXT for readouts.
- **Mode.** Static.
- **Compute.** L0.
- **Controls.** Teacher-text curve; control curves (flat at chance); IND-AR
  replication of the curve; within-family-only galleries (hard mode).
- **Measurements.** Top-1/median-rank vs N; per-stratum identification bits;
  confusion-edge family structure.
- **Success means.** A defensible, correctly-hedged headline statistic and a
  map of *where* identifiability is weak (candidate hard-case corpus for
  format work).
- **Failure means.** If hard-mode (within-family) identification collapses,
  the code individuates topics, not states — sharpening what "row-specific"
  means in the qualified claim.
- **Confounds.** Gallery composition (family structure inflates easy
  distinctions — hence hard mode); retrieval-vs-capacity conflation (hence
  the assumptions card).
- **Misleading-risk.** Low *because* the assumptions card is part of the
  design; without it, high.
- **Novelty.** Standard retrieval analysis; the honesty furniture and
  stratification are the contribution.
- **Difficulty.** 1–2 days.
- **Demo appeal.** Medium-high ("≥9 bits per paragraph" is quotable — with
  its assumptions).

#### C11. Redundancy & Error-Correction Probe `[new]`
Treat the code as a noisy channel: how much corruption does it survive?
- **Research question.** Is the explanation redundant (graceful degradation
  under typos/word drops — an error-correcting code) or brittle (cliff
  behavior — consistent with high-entropy private encoding)?
- **User experience.** A corruption console: choose noise type (character
  typos, word deletion, word-order swaps, clause deletion, synonym
  substitution) and rate; degradation curves render per noise type; a text
  pane shows a corrupted explanation with its surviving reconstruction cos.
- **Visual encoding.** Degradation curves (cos vs corruption rate) with
  family bands, one per noise type; per-row cliff-point rug plot.
- **Interaction model.** Noise sliders; per-row inspection; compare AV vs
  teacher text robustness; critic toggle.
- **Models & data.** AR + TEXT.
- **Mode.** Precomputed grid (5 noise types × ~6 rates × 100–512 rows).
- **Compute.** L1.
- **Controls.** Matched corruption of *control* texts (floor); paraphrase
  (C20) as the "semantic noise" endpoint; scaffold-only corruption vs
  payload-only corruption (ties to C8's scaffold/payload split).
- **Measurements.** Area under degradation curve per noise type; cliff-rate;
  robustness ratio AV/teacher; interaction with explanation length.
- **Success means.** Graceful curves → the code carries distributed,
  redundant semantics (supports the "honest English" reading and predicts
  paraphrase robustness); the noise tolerance number becomes a channel spec.
- **Failure means.** Cliffs on tiny perturbations → information is carried
  by fragile surface features; combined with C20, this is the strongest
  cheap cipher evidence available.
- **Confounds.** Tokenizer artifacts (typos change token boundaries — report
  token-level and character-level noise separately).
- **Misleading-risk.** Low.
- **Novelty.** Channel-coding analysis of a generated explanation is new;
  SAEs have no analogous perturbable code.
- **Difficulty.** 2–3 days.
- **Demo appeal.** Medium.

---

### Family C — Dynamics

#### C12. Residual-Stream Cinema `[extends companion §C.11 with half-life barcodes]`
The per-token state readout as a genome-browser / score view.
- **Research question.** What are the timescales of R33 content as read
  through its own code — which description channels are slow state, which
  are fast, and where are the change points?
- **User experience.** A document scrolls horizontally as the reference
  track. Below it, one track per explanation section (genre/topic, discourse,
  register/syntax, final-token machinery): each track renders that section's
  text at each sampled position, with persistent spans drawn as continuous
  bars (unchanged text = one long bar) and change points as breaks. A
  barcode lane shows per-section "half-life" statistics; a heat lane shows
  word-level diff magnitude between consecutive positions.
- **Visual encoding.** Genome-browser tracks; persistence bars; change-point
  glyphs aligned to source tokens (clause boundaries, sentence starts);
  half-life distribution barcodes (borrowing persistence diagrams from TDA).
- **Interaction model.** Scrub positions; click a change point → side-by-side
  diff of the two descriptions + the source tokens between them; select a
  section → its half-life distribution across documents; jump to C13 to
  test a change point causally.
- **Models & data.** BASE (extraction at many positions) + AV; 5–20
  documents precomputed; PARQ provenance for alignment.
- **Mode.** Precomputed filmstrips; live single-document mode later.
- **Compute.** L2 precompute; L3 live.
- **Controls.** Shuffled-position control (descriptions of random other
  positions — do apparent persistences exceed topic-level chance?); n-gram
  novelty baseline (do change points merely track source-surface novelty?);
  min-position ≥ 50 rule respected and displayed.
- **Measurements.** Per-section persistence distributions; change-point
  alignment rates with linguistic boundaries vs baseline; cross-document
  consistency of section timescales.
- **Success means.** A reproducible timescale spectrum of R33 content stated
  in the code's own vocabulary — e.g., *"genre persists ~sentences; final-
  token machinery churns every step"* — a genuinely new kind of dynamical
  observable for a hybrid Mamba/attention model.
- **Failure means.** Description churn uncorrelated with source structure →
  either AV sampling noise dominates (fix with greedy decoding) or the code
  is not a stable state readout at this cadence — both essential to know
  before any "dynamics" claims.
- **Confounds.** Per-position activations here are *fresh* extractions, not
  the frozen snapshots — the stored-vs-fresh drift caveat applies and each
  filmstrip must carry a fresh-forward badge; sampling stochasticity
  (decode greedily; show a temperature-jitter band from repeats).
- **Misleading-risk.** Medium-high: a subtitle track under text is the most
  anthropomorphizable view in the suite ("the model's inner monologue").
  Mandatory framing: *a per-position learned encoding of one boundary's
  activation*, plus the C21 null-text always one keypress away.
- **Novelty.** Logit lens gives per-position token distributions; SAE gives
  per-position feature firings. A *diffable prose* state track with
  persistence structure exists only here.
- **Difficulty.** ~1–2 weeks including alignment plumbing.
- **Demo appeal.** Very high — this is the film-worthy artifact.

#### C13. Causal Wake Profiler `[extends companion §B.9; adds dose dimension]`
The spatiotemporal footprint of a reinjection.
- **Research question.** How far into the future does a patched
  reconstruction change behavior — and how does the wake scale with dose and
  with condition (AV / teacher / controls / edited codes from C15)?
- **User experience.** After a patch at position t, a wake plot shows
  teacher-forced per-position KL at t+1…t+k as a decaying wave (fluid-
  dynamics streakline aesthetic); conditions overlay; a dose slider (α along
  `h→ĥ` or `ĥ→ĥ_edit`) extrudes the wave into a wake-surface.
- **Visual encoding.** Decay curves / ridgeline stack by condition;
  amplitude×half-life summary scatter; dose-extruded surface with contour
  lines.
- **Interaction model.** Pick row/position; toggle conditions; scrub dose;
  click a downstream position → token-level distribution comparison.
- **Models & data.** BASE + AR (+edited codes from C15); FUNC for the
  patch-position-only baseline already measured.
- **Mode.** Precomputed cohorts.
- **Compute.** L2 (k extra teacher-forced positions per row×condition×dose).
- **Controls.** Zero/mean/shuffled wakes (expected large, per FUNC:
  patch-position KL `4.1–9.5`); paraphrase-placebo edits (wake should match
  unedited); position-matched no-op patches (h reinjected — measures harness
  floor and stored-drift wake).
- **Measurements.** Wake amplitude (KL at t+1), half-life (positions to
  ε), integrated wake energy; dose-monotonicity index.
- **Success means.** AV wakes ≈ teacher wakes ≫ control wakes with clean
  dose scaling — the temporal extension of the qualified functional claim,
  and the natural outcome measure for all surgery experiments.
- **Failure means.** AV wakes indistinguishable from mean-control wakes
  beyond t+1 → functional recovery is shallower than the patch-position
  metrics suggest; the functional claim's scope note gains teeth.
- **Confounds.** Teacher-forcing masks compounding divergence (complement
  with free-running divergence trees on a subsample); stored-snapshot scope.
- **Misleading-risk.** Low-medium; "wake" language is mechanistically neutral.
- **Novelty.** Activation-patching papers report effect-at-patch or
  end-task deltas; a dose-resolved temporal decay profile of a *language-
  reconstructed* patch is new.
- **Difficulty.** ~1 week.
- **Demo appeal.** High (the wake surface is a beautiful figure).

#### C14. State-Transition Grammar `[new]`
A finite-state summary of R33 dynamics mined from the code stream.
- **Research question.** Do per-position descriptions move through a small
  set of recurring states with structured transitions (a grammar of internal
  state), or is the sequence memoryless given the source?
- **User experience.** From C12's corpus of per-position descriptions,
  cluster description states (per section), then render the transition graph:
  nodes = description states (labeled by their medoid text), edges =
  transition probabilities; a document trace lights up its path through the
  graph like a subway map.
- **Visual encoding.** Force-directed or layered transition graph with
  self-loop thickness = persistence; document-trace animation; per-section
  sub-graphs.
- **Interaction model.** Select node → example positions/passages; select
  edge → change-point examples; compare graphs across document genres.
- **Models & data.** C12's precomputed description corpus (no new GPU work).
- **Mode.** Static atop C12.
- **Compute.** L0 (given C12).
- **Controls.** The critical one: a *source-only* baseline grammar (same
  clustering applied to teacher-style descriptions generated from source
  prefixes without h, or to topic-model states of the source) — the NLA
  grammar is interesting only where it diverges from source-surface
  dynamics; order-shuffled transition matrix for chance structure.
- **Measurements.** State count vs description-variance explained; transition
  entropy; divergence between NLA grammar and source-baseline grammar;
  cross-document state reuse.
- **Success means.** Recurring internal-state motifs not reducible to source
  surface statistics — candidate objects for targeted causal tests (patch at
  state-X positions, measure state-Y induction).
- **Failure means.** Grammar ≈ source baseline → at R33, the verbalized
  state tracks input surface structure; a clean negative that bounds
  "internal dynamics" claims.
- **Confounds.** Clustering granularity (sweep and report); AV sampling
  noise (greedy decode).
- **Misleading-risk.** Medium: state graphs invite "the model plans"
  narratives; label states as *description clusters*.
- **Novelty.** HMM-style summaries of hidden states exist; deriving the
  state alphabet from the model's own generated language is new.
- **Difficulty.** Days atop C12.
- **Demo appeal.** Medium-high.

---

### Family D — Causal intervention

#### C15. Explanation Surgery Bench `[extends companion §2.1 with a controlled-trial design]`
The keyboard-writable latent, hardened into an experiment.
- **Research question.** Do targeted edits to the code cause behavioral
  changes *in the named direction*, beyond paraphrase placebo and
  matched-edit-distance random controls?
- **User experience.** Pick a row; its explanation appears as editable text
  with C8 saliency tint as guidance. Make an edit (free-form or preset chips:
  flip register / swap topic noun / negate final-token constraint). The bench
  runs the AR encode + stored-snapshot reinjection and renders: next-token
  distribution morph, Δcos compass on the PC map, wake profile (C13), and an
  *effect ledger* row recording {edit intent, edit distance, Δĥ, behavioral
  delta, control outcomes}.
- **Visual encoding.** Side-by-side token distributions with movement
  arrows; compass glyph; ledger table with effect-size dots and CI whiskers.
- **Interaction model.** Edit → run → compare; every free-form edit
  auto-spawns its two controls (paraphrase of the same clause; random token
  edit of equal edit distance) and displays all three outcomes together;
  batch mode replays an edit template across many rows.
- **Models & data.** AR + BASE (2 forwards per variant); TEXT; C8 saliency.
- **Mode.** Live (the point), with precomputed preset-chip galleries.
- **Compute.** L2 minimum (AR+BASE); L1 "surgery-lite" shows Δcos only.
- **Controls.** Paraphrase placebo (should do nothing); random-edit control
  (how much does *any* perturbation of this size do?); dose ramp (C16);
  magnitude renormalization on/off; teacher-text surgery (is the writable
  interface AV-specific?).
- **Measurements.** Named-direction hit rate (did the top-k distribution move
  toward tokens consistent with the edit intent, scored blind); effect size
  vs edit distance; placebo rates; per-edit-type reliability across rows.
- **Success means.** Edits beat both controls with intent-consistent
  direction at reasonable rates → the first demonstration that a
  natural-language latent is a *usable causal control surface*, with effect
  sizes and failure cases catalogued.
- **Failure means.** Edits ≈ random-edit control → the AR reads holistic
  surface patterns, not compositional meaning; steering claims die here
  (cheaply), and C20 becomes the priority.
- **Confounds.** Edited text is OOD for the AR (paraphrase placebo partially
  controls this); intent scoring subjectivity (blind scoring protocol, §5);
  stored-snapshot scope; norm handling (C6 policy applied and displayed).
- **Misleading-risk.** High if shipped without controls — a demo that only
  shows successes would manufacture belief in language-steering. The
  bench's ledger design (all runs recorded, controls always visible) is the
  mitigation.
- **Novelty.** Steering today means adding vectors someone derived offline.
  Editing a sentence and watching the model change is categorically new, and
  the paraphrase-placebo design is what makes it science, not theater.
- **Difficulty.** ~1–2 weeks on serving harness (companion's §2.1 is the
  core loop; controls and ledger are the additions).
- **Demo appeal.** Very high — flagship material (F2).

#### C16. Semantic Dose–Response `[new]`
Interpolate the *edit*, not just the injection norm.
- **Research question.** Is the behavioral effect of a code edit graded and
  monotone in intervention strength, and where are thresholds/saturation?
- **User experience.** After any C15 edit, a dial sweeps
  `h(α) = (1−α)·ĥ_orig + α·ĥ_edit` (and the slerp variant); behavioral
  metrics plot against α with sampled continuations pinned at α = 0, ¼, ½,
  ¾, 1; an isobologram-style pane compares text-space dose (edit k of n
  clauses) against vector-space dose.
- **Visual encoding.** Dose–response curves with bands; pinned continuation
  cards along the dial; text-dose vs vector-dose surface.
- **Interaction model.** Scrub α; switch lerp/slerp; select response metric
  (KL, target-token logit, wake energy).
- **Models & data.** AR + BASE; C15 edits.
- **Mode.** Precomputed for preset edits; live per-edit on the bench.
- **Compute.** L2.
- **Controls.** Paraphrase-placebo dose curves (should stay flat);
  random-edit dose curves; norm-only dose (C6) as reference shape.
- **Measurements.** Monotonicity index; EC50-style half-effect dose;
  saturation level; text-dose vs vector-dose agreement.
- **Success means.** Graded, monotone response strengthens the causal
  reading of C15 enormously (dose–response is a classic causality
  criterion) and gives intervention design a dosing rule.
- **Failure means.** Step-function responses locate the symbol-lattice
  boundaries of C2 behaviorally; non-monotonicity flags off-manifold
  interpolant artifacts.
- **Confounds.** Interpolated vectors may leave the activation manifold
  (compare against C5's reachable set); stored-snapshot scope.
- **Misleading-risk.** Medium; curves imply mechanism smoothness that only
  holds along this one segment.
- **Novelty.** Dose–response for *semantic text edits* mapped through a
  reconstructor has no precedent; activation steering has scale sweeps, but
  the dose here is authored meaning.
- **Difficulty.** Days atop C15.
- **Demo appeal.** High.

#### C17. Wrong-Brain Transplant Matrix `[extends companion §B.6]`
Cross-context reinjection as a behavioral distance microscope.
- **Research question.** How context-portable is a reconstructed state? Does
  transplanting ĥ_j into context i produce donor-like behavior graded by
  semantic proximity?
- **User experience.** A donor×host matrix (12×12 to start) colored by
  patched-KL; rows/columns sorted by family distance; clicking a cell shows
  host continuation vs donor-influenced continuation with token-level
  highlights; marginal profiles show which donors are "loud" and which hosts
  are "receptive."
- **Visual encoding.** Heatmap with distance-graded expectation bands;
  cell-level continuation diff cards; loudness/receptivity marginals.
- **Interaction model.** Re-sort by norm/position/family; select cell →
  full comparison; escalate any cell to C13 wake view.
- **Models & data.** BASE + CACHE (reconstructions already exist); FUNC
  machinery.
- **Mode.** Precomputed.
- **Compute.** L2 (n² patched forwards).
- **Controls.** Own-row diagonal (floor); shuffled control column (ceiling);
  position-matched and norm-matched off-diagonal subsets (portability vs
  mere position/norm mismatch); donor *teacher-text* reconstructions.
- **Measurements.** KL vs family-distance regression; donor-loudness
  variance; asymmetry (i→j vs j→i).
- **Success means.** Graded structure (own ≪ same family ≪ stranger ≈
  shuffled) — behavioral confirmation that reconstructions carry
  row-specific, context-transportable content; loud donors are natural
  steering candidates.
- **Failure means.** Flat off-diagonal at shuffled level → reconstructed
  states do not transplant; "steering with descriptions" inherits a strong
  scope limit (patch-in-place only).
- **Confounds.** Position mismatch dominates unless controlled;
  boundary-state discontinuity for the Mamba side of the hybrid stack (the
  patch replaces the residual, not the SSM state — a Nano-specific caveat
  worth displaying).
- **Misleading-risk.** Medium ("transplant" language invites overclaim;
  keep the stored-snapshot badge on every cell).
- **Novelty.** Cross-*context* patching of language-reconstructed states —
  patching literature stays within-prompt; this measures portability.
- **Difficulty.** Days.
- **Demo appeal.** High (the matrix reads like a plate-well assay).

#### C18. Write-Only Programming `[new; distinct from companion §2.2]`
Author a state from scratch: no target vector, only intent.
- **Research question.** Can a human *program* a behavioral disposition by
  writing a description de novo — and does success depend on staying inside
  the code's learned format and the writable set (C5)?
- **User experience.** The user writes an explanation from a blank template
  ("genre: recipe; register: imperative; final token: a number…"), AR
  encodes it, the vector is injected into a neutral host prefix, and the
  continuation renders beside the unpatched continuation. A target picker
  defines the intended effect *before* running (e.g., "continuation should
  contain cooking vocabulary"), and hit/miss is scored automatically against
  the pre-registered target.
- **Visual encoding.** Intent card → outcome card with hit/miss stamp;
  cumulative hit-rate curves per intent category; landing-point display on
  the C5 atlas (did the authored text land on-manifold?).
- **Interaction model.** Write → predict → run → score; a gallery of
  successful and failed programs with their landing points; remix any
  program.
- **Models & data.** AR + BASE; C5 atlas for landing context.
- **Mode.** Live; curated gallery precomputed.
- **Compute.** L2.
- **Controls.** Scrambled versions of each authored text (same words,
  shuffled); template-only skeleton (no content); dose ramp on authored
  vectors; intent-swapped scoring (does the "cooking" program also trigger
  "finance" scorers? specificity check).
- **Measurements.** Pre-registered hit rate per intent class; specificity
  matrix (intent × triggered-effect); on-manifold landing fraction vs hit
  rate correlation.
- **Success means.** Above-control hit rates for naive-authored programs →
  the AR generalizes beyond AV-generated text and the interface is genuinely
  writable by humans — the strongest possible "control surface" evidence
  short of RL.
- **Failure means.** Only format-perfect, AV-like texts land on-manifold and
  work → the writable interface is real but narrow (a dialect, not English);
  quantifying that narrowness is itself the finding.
- **Confounds.** Automatic effect scorers embed their own bias (use held-out
  scorer prompts + human spot checks); prompt-format leakage (host prefix
  choice matters — vary it).
- **Misleading-risk.** High in public settings ("I mind-controlled the AI");
  requires the F5 framing kit and pre-registered scoring.
- **Novelty.** De-novo authored latents with pre-registered behavioral
  targets: no interpretability tool currently offers this interaction at all.
- **Difficulty.** ~1 week atop the C15 harness.
- **Demo appeal.** Very high.

#### C19. Mediation Cascade `[new]`
Where along source→h→z→ĥ→behavior does a causal signal survive?
- **Research question.** For a minimal edit to the *source prefix* (one word
  swap), how much of the induced activation change survives verbalization,
  reconstruction, and reinjection — a mediation analysis through the
  language channel?
- **User experience.** A five-column cascade: source diff → Δh (norm + top
  PC loadings) → z diff (highlighted words that changed) → Δĥ → behavioral
  delta. A Sankey band shows effect magnitude surviving each stage; a corpus
  pane aggregates over many minimal pairs by edit type (entity swap, negation,
  register flip, number change).
- **Visual encoding.** Cascade columns with diff highlighting; Sankey
  attenuation bands; edit-type aggregate bars with family-clustered CIs.
- **Interaction model.** Author a minimal pair or pick from a bank; inspect
  any stage; flip mediation direction (patch ĥ_edited into original context
  vs original ĥ into edited context).
- **Models & data.** BASE (two fresh extractions per pair) + AV + AR.
- **Mode.** Precomputed minimal-pair bank (~hundreds); live authoring later.
- **Compute.** L2–L3.
- **Controls.** Null pairs (whitespace/punctuation edits); paraphrase pairs
  (meaning-preserving source edits — does z stay put?); the fresh-forward
  drift floor (C30) as the Δh noise reference.
- **Measurements.** Stage-wise retention ratios (‖Δĥ‖/‖Δh‖ directional;
  z-diff detectability AUC; behavioral-delta recovery fraction); edit-type
  profiles.
- **Success means.** A quantitative answer to "does the code *notice* what
  changed?" — edit types the channel transmits vs absorbs, which is exactly
  the information needed to trust surgery results and to target teacher-v2.
- **Failure means.** z rarely reflects controlled Δh → descriptions are
  insensitive to precisely-localized state changes; a sharp, honest bound on
  reading fidelity (and a caution for all diff-based views, C12 included).
- **Confounds.** Fresh-extraction drift (both sides fresh, so paired design
  absorbs the stable component); minimal pairs induce non-minimal Δh
  (tokenization ripple — display token alignment).
- **Misleading-risk.** Low-medium; the cascade format itself teaches where
  claims stop.
- **Novelty.** Causal-mediation formalism applied with a *readable mediator*
  — the mediator can be inspected as text, which mediation analysis has
  never had.
- **Difficulty.** ~1–2 weeks.
- **Demo appeal.** High for expert audiences.

---

### Family E — Semantics & integrity

#### C20. Paraphrase Microscope / Cipher Detector `[elevates companion §A.4; adds human-legibility decomposition]`
The semanticity instrument: does the code mean what it says?
- **Research question.** Is reconstruction invariant to meaning-preserving
  rewrites and sensitive to meaning-changing ones — and is the information
  carried where humans read content, or in surface features (a private
  AV↔AR cipher)?
- **User experience.** For each row, two clouds on the PC map: ĥ under k
  paraphrases (synonym swap, clause reorder, register rewrite,
  back-translation) vs ĥ under k matched *semantic* edits. A per-row
  semanticity index (paraphrase dispersion ÷ semantic-edit dispersion)
  sorts a gallery from "honest English" to "cipher-suspect." A second pane
  masks words by human-rated contentfulness: reconstruction from
  content-words-only vs filler-only (extends companion §A.5) — if filler
  carries signal, the cipher lamp lights.
- **Visual encoding.** Paired dispersion clouds; semanticity-index strip
  with row rugs; content/filler dual bars per token; a court-style verdict
  chip per row (honest / mixed / suspect) with the evidence behind it.
- **Interaction model.** Click any paraphrase → its text diff + landing
  point; adjust paraphrase source (human vs LLM); toggle critic (IND-AR
  replication of the index); export suspect rows to F4.
- **Models & data.** AR + TEXT + paraphrase generators (LLM + human-authored
  subset from §5); human contentfulness ratings (§5).
- **Mode.** Precomputed core; grows with §5 data.
- **Compute.** L1 (~6–20 forwards/row × 100–512 rows).
- **Controls.** Paraphrase-magnitude calibration (embed-distance-matched
  pairs so "paraphrase" and "semantic edit" differ in meaning, not size);
  teacher-text microscopy (is the cipher AV-specific?); cross-critic index
  agreement (a cipher would have to be *shared* by independently trained
  readers — see confounds).
- **Measurements.** Semanticity index distribution; filler-carried cos;
  paraphrase-cliff rate (C11 tie-in); index correlation across critics.
- **Success means.** High, cross-critic-stable semanticity with content-word
  concentration → the central metaphor of the method survives its hardest
  cheap test; the SFT-era baseline index becomes the Goodhart reference
  band (C31).
- **Failure means.** Paraphrase cliffs or filler-borne signal → the code is
  partly non-semantic *now, before any RL* — reframing every readable-latent
  claim in this repo and making the Sentinel non-optional. A crucial,
  publishable failure.
- **Confounds.** Both critics share teacher data and base geometry, so
  cross-critic agreement bounds but does not eliminate shared-cipher risk
  (per the measurement contract, shared critic errors must not be read as
  definitive AV-side proof — the human-legibility pane exists precisely to
  break this tie); paraphrase generators can leak meaning changes (human
  verification sample).
- **Misleading-risk.** Low — this is the instrument that *reduces* others'
  risk.
- **Novelty.** Steganography checks exist for CoT faithfulness; a
  quantitative per-row semanticity index over a reconstruction channel with
  two decoders and human masking is new.
- **Difficulty.** ~1 week (core), plus §5 pipelines.
- **Demo appeal.** Medium as a chart; very high as F4's centerpiece.

#### C21. Null-Text Almanac `[companion §0.7, kept as the universal baseline]`
The AV's prior: what it says when it knows nothing.
- **Research question.** What is the AV's unconditional "explanation-speak"
  distribution (from zero/mean/shuffled injections), and which tokens in any
  real description exceed it?
- **User experience.** A browsable almanac of control-condition text:
  template collapse rates, hedging lexicon, per-token log-odds
  (real-vs-zero) dictionary. Every other view's text rendering can toggle
  "almanac tint" — boilerplate fades, signal-bearing words glow.
- **Visual encoding.** Token log-odds glow; control-text mode collapse
  dendrogram; hedge-rate meters.
- **Interaction model.** Search the almanac; toggle tint anywhere; click a
  boilerplate phrase → its frequency across conditions.
- **Models & data.** TEXT (1,024 × 5 control generations) only.
- **Mode.** Static.
- **Compute.** L0.
- **Controls.** It *is* the control; verify stability across splits.
- **Measurements.** Log-odds dictionary; collapse entropy per condition;
  fraction of real-text tokens above almanac threshold.
- **Success means.** Every saliency/diff view in the suite becomes
  interpretable relative to a measured null — the single highest-leverage
  honesty feature per unit effort.
- **Failure means.** If real text barely exceeds the null on most tokens,
  the per-token views (C7, C8) are mostly reading prior — vital to know.
- **Confounds.** Zero/mean vectors are off-manifold inputs; the "null" is an
  off-distribution behavior, label as such.
- **Misleading-risk.** Low.
- **Novelty.** A quantified confabulation prior as reusable UI furniture.
- **Difficulty.** 1–2 days.
- **Demo appeal.** Medium (the blind-lineup game hook is fun).

#### C22. Upset Forensics `[companion §0.6, kept]`
Where the machine's code beats the human-style summary, and why.
- **Research question.** On which rows and content types does AV text
  reconstruct better than teacher text, and what does the AV include that
  the summary omits? (Companion reports 234/512 upsets [exploratory session
  finding, companion doc].)
- **User experience.** Teacher-vs-AV per-row scatter with the y=x diagonal;
  upset rows open side-by-side texts with AV-only content flagged
  (candidates: final-token/register machinery a faithful summary omits).
- **Visual encoding.** Diagonal scatter; margin-sorted gallery; token-class
  attribution of upset margins (via C8 saliency deltas).
- **Interaction model.** Sort by margin/family/length; export upset clusters
  as format-hypothesis cards for teacher-v2.
- **Models & data.** CACHE + TEXT.
- **Mode.** Static.
- **Compute.** L0.
- **Controls.** Length matching (AV text may simply be longer/shorter);
  IND-AR agreement on upset identity.
- **Measurements.** Upset rate by stratum; margin distribution; token-class
  enrichment in upsets.
- **Success means.** A concrete inventory of what a purpose-built code
  carries that summaries drop — the empirical brief for lexical anchors
  (NT-6).
- **Failure means.** Upsets random / length-explained → AV≈teacher parity is
  parity all the way down; format work should look elsewhere.
- **Confounds.** Teacher never saw h (upsets partly measure the teacher's
  prompt constraints, not AV skill).
- **Misleading-risk.** Low.
- **Novelty.** Modest — a well-aimed drill-down.
- **Difficulty.** 1–2 days.
- **Demo appeal.** Medium.

#### C23. Provenance Typography `[new]`
A rendering standard that makes every description carry its own epistemics.
- **Research question.** (Design research.) Can typography alone keep users
  calibrated about generated-text trustworthiness — and can we measure that?
- **User experience.** Everywhere a description appears: per-token opacity =
  AV generation confidence (logprob), underline weight = AR-usability
  (C8 saliency), almanac tint (C21) on demand; a provenance chip row under
  every text: {stored-snapshot|fresh-forward, protocol hash, critic,
  split, exploratory badge}. Descriptions are never rendered as bare
  authoritative prose anywhere in the suite.
- **Visual encoding.** Typographic channels (opacity/weight/tint), chip row;
  a legend card teaching the encoding once.
- **Interaction model.** Hover token → its three scores; global toggle;
  copy-as-plain-text intentionally adds a caveat footer.
- **Models & data.** AV logprobs (recorded at generation), C8 batch, C21.
- **Mode.** Static infrastructure.
- **Compute.** L0 (given C8/C21 batches).
- **Controls.** A/B user study (§5): calibration questions with vs without
  the typography.
- **Measurements.** User trust-calibration deltas; over-trust rate on
  control-condition texts with/without typography.
- **Success means.** A reusable, evaluated defense against the field's most
  chronic failure (fluent text read as truth) — publishable as an
  uncertainty-visualization contribution on its own.
- **Failure means.** No calibration effect → typography is not enough;
  stronger interventions (forced null-comparison interactions) needed.
- **Confounds.** AV confidence ≠ correctness (display is "generation
  confidence," never "truth").
- **Misleading-risk.** It exists to reduce it; residual risk is users
  reading opacity as importance.
- **Novelty.** Uncertainty typography exists; wiring it to a
  reconstruction-channel's *measured* per-token usability is new.
- **Difficulty.** Days.
- **Demo appeal.** Subtle but distinctive in every screenshot.

---

### Family F — Comparative interpretability

#### C24. Rosetta Panel `[new]`
One activation, four instruments: NLA text, SAE features, logit lens, probes.
- **Research question.** What does the NLA say that sparse features, the
  logit lens, and supervised probes do not (and vice versa)? Are
  disagreements systematic?
- **User experience.** For a selected (row, position): the AV description
  (with typography), top SAE latents with autointerp labels, logit-lens
  top-k tokens at R33, and available probe outputs, arranged as four
  "translations" of one artifact. An agreement graph links elements that
  corroborate (SAE "legal-document" latent ↔ description clause "formal
  legal register"); unmatched elements stand out.
- **Visual encoding.** Four-column rosetta card; corroboration edges;
  per-element uniqueness badges; corpus-level agreement matrix.
- **Interaction model.** Click an SAE latent → its max-activating examples
  vs the rows whose descriptions contain its matched clause; click a
  description clause → which latents/probes co-vary with its presence.
- **Models & data.** CACHE + TEXT + **SAE at R33 (must be trained — L4)** +
  logit-lens projections (BASE unembedding applied at R33; cheap) + any
  probes.
- **Mode.** Precomputed corpus panel.
- **Compute.** L4 once (SAE training), then L1.
- **Controls.** Matching by embedding must beat permuted matching (edge
  significance); autointerp labels are themselves generated text — carry
  their own provenance chips; logit lens at a pre-attention boundary of a
  hybrid stack may be ill-calibrated (tuned-lens variant as check).
- **Measurements.** Clause↔latent mutual information; coverage (fraction of
  description content matchable to any latent, and vice versa);
  disagreement taxonomy counts.
- **Success means.** A concrete uniqueness statement for NLA ("relational/
  discourse content appears in descriptions but no latent") and a
  cross-validation harness that catches confabulation (description clause
  with zero co-varying latent = suspect).
- **Failure means.** Descriptions reduce to a bag of SAE latents → NLA is a
  readable *rendering* of known structure; still useful, differently framed.
- **Confounds.** SAE dictionary quality; autointerp circularity (LLM labels
  matched to LLM descriptions — use lexical-overlap-free matching).
- **Misleading-risk.** Medium (corroboration edges could be read as ground
  truth; they are co-variation).
- **Novelty.** First side-by-side of a *generative-language* readout against
  the standard toolkit on the same activations; the mutual-validation
  design is the point.
- **Difficulty.** Weeks (dominated by SAE training/quality).
- **Demo appeal.** High for the interp community specifically.

#### C25. Steering-Vector Cross-Validation `[new]`
Do descriptions and steering directions agree about what a direction means?
- **Research question.** When AV verbalizes a known steering vector (e.g.,
  diff-of-means for a concept), does the description name the concept? When
  AR encodes a concept phrase, does the vector align with the steering
  direction?
- **User experience.** A concept library (register, sentiment, language,
  topic classes constructible from PARQ metadata without new labels); for
  each: the steering direction's AV description at several injection norms,
  and the AR encoding of canonical concept phrases; an agreement matrix
  (concept × concept: cos between AR-encoded phrase and steering direction).
- **Visual encoding.** Agreement matrix with diagonal expectation; per-cell
  drill-down cards; norm-sweep description strips (does the verbalization
  sharpen with dose?).
- **Interaction model.** Add a concept by selecting two row sets (contrast);
  read its verbalization; test its phrase-encoding alignment.
- **Models & data.** PARQ (for contrast sets) + AV + AR.
- **Mode.** Precomputed library; extendable.
- **Compute.** L2.
- **Controls.** Random-direction verbalizations (what does AV say about
  noise? ties to C1 blind-spot behavior); shuffled-label contrasts;
  phrase-set robustness (k phrasings per concept).
- **Measurements.** Diagonal-vs-off-diagonal agreement gap; verbalization
  hit rate (blind raters match description→concept); dose sharpening slope.
- **Success means.** NLA validates/labels steering vectors (useful to the
  steering community) and inherits their independent evidence in return —
  bidirectional trust transfer between toolkits.
- **Failure means.** Descriptions of steering directions are prior-collapse
  text (they are off-manifold single directions — plausible!) → documents a
  hard boundary of AV's input domain; the atlas (C5) gains an annotation.
- **Confounds.** Steering directions are not activations (norm/manifold
  mismatch — sweep norms and say so); concept leakage via contrast-set
  construction.
- **Misleading-risk.** Medium.
- **Novelty.** First systematic NLA↔steering translation table.
- **Difficulty.** ~1 week.
- **Demo appeal.** Medium-high.

#### C26. Boundary Babel `[new as visualization; NT-8's question, read through code]`
The same positions verbalized at different boundary types.
- **Research question.** Do post-Mamba (R33-type) and post-attention
  (R27/R34-type) boundaries verbalize different *kinds* of content —
  e.g., local-composition vs retrieved-context — in a hybrid stack?
- **User experience.** For matched positions, parallel description columns
  per boundary; a differential lexicon (words systematically present at one
  boundary and absent at the other); section-level divergence tracks over
  document position (C12-style, per boundary).
- **Visual encoding.** Parallel-text columns with cross-boundary diff
  highlighting; differential word clouds disciplined by log-odds (not
  raw frequency); per-section divergence curves.
- **Interaction model.** Scrub positions; select a differential word → the
  positions where the boundaries disagree; jump to reinjection checks.
- **Models & data.** Requires AV/AR pairs at a second boundary — **new
  training (L4)** — or, cheaply, AR-only probes at R27 using existing R27
  assets (scouting-grade) with descriptions from the R33 AV only (weaker:
  same text, different reconstruction target).
- **Mode.** Precomputed.
- **Compute.** L4 (full) / L2 (scouting variant).
- **Controls.** Within-boundary seed variance (is cross-boundary difference
  bigger than same-boundary retrain difference? — cannot claim otherwise);
  matched dMSE operating points (a worse pair verbalizes differently for
  quality reasons).
- **Measurements.** Cross-boundary description divergence vs within-boundary
  baseline; differential-lexicon stability; per-section divergence profiles.
- **Success means.** A vocabulary-level statement about hybrid-architecture
  information routing ("post-Mamba boundaries verbalize more local-
  composition machinery") — a genuinely novel architecture finding told in
  the model's own words.
- **Failure means.** Divergence ≈ seed variance → descriptions are
  boundary-insensitive at this granularity; fold into the layer-decision
  record and stop.
- **Confounds.** Seed variance (single qualified AV seed today — this
  concept is gated on affording comparison pairs); teacher format identical
  across boundaries may mask differences.
- **Misleading-risk.** Medium.
- **Novelty.** Layer comparisons exist for probes/SAEs; a *lexical*
  differential between boundary types is new.
- **Difficulty.** Weeks-months (training-gated).
- **Demo appeal.** Medium; high for architecture researchers.

---

### Family G — Human participation

#### C27. Activation Pictionary `[new; subsumes companion §0.7 game + §B.8]`
Blind identification psychophysics: how much do humans extract from the code?
- **Research question.** What is human identification performance
  (description → which of k source passages?) vs the AR's, per condition —
  and where do humans and the AR disagree about distinguishability?
- **User experience.** A round: read one description, pick the matching
  passage among k (distractors sampled same-family or cross-family for
  difficulty control); streak scoring; a live d′ meter per player;
  aggregate boards comparing humans / AR / chance per condition (real,
  shuffled, teacher).
- **Visual encoding.** Game cards; per-condition accuracy forest plot
  (humans vs AR with CIs); confusion overlap Venn (human errors vs AR
  errors vs C4 lumps).
- **Interaction model.** Play; review your errors with the answer's
  explanation; contribute to the public aggregate (opt-in, §5).
- **Models & data.** TEXT + qualitative panel (source-resolved rows only:
  the 100-row panel + any §5-cleared extensions); AR scores from CACHE.
- **Mode.** Static content; live scoring.
- **Compute.** L0.
- **Controls.** Teacher-text rounds (is AV code harder for humans than
  summaries — machine-dialect evidence?); shuffled rounds (should be at
  chance — a manipulation check on the game itself); passage-length and
  family-difficulty stratification.
- **Measurements.** Human d′ per condition/stratum; human–AR error overlap;
  learning curves across rounds.
- **Success means.** A human-legibility number to place beside dMSE `0.307`
  — and the human/AR disagreement set is exactly where cipher suspicion
  (C20) should focus.
- **Failure means.** Humans near chance while AR near ceiling → the code is
  substantially machine-dialect *already*; major reframing of "readable."
- **Confounds.** Distractor difficulty (fix by design + report); player
  population skew (§5); source passages must be privacy-cleared (the 14
  phone-like source patterns stay internal per the release-text triage).
- **Misleading-risk.** Low; game framing is honest about being a test.
- **Novelty.** Turns the headline retrieval statistic into a human
  psychophysics experiment with reusable data.
- **Difficulty.** Days (content exists).
- **Demo appeal.** Very high; core of F5.

#### C28. Intervention Prediction Market `[new]`
Humans forecast the effect of an edit before it runs.
- **Research question.** Do NLA descriptions support *accurate mental
  models* — can people predict the behavioral effect of a code edit better
  than base rates, and does accuracy improve with exposure?
- **User experience.** A queued C15 edit is shown (original code, edit,
  host context); the user forecasts among structured outcomes ("top token
  becomes food-related", "no change", "distribution flattens") with a
  confidence slider; the run executes; calibration and Brier scores update;
  a market view shows crowd distributions vs outcomes.
- **Visual encoding.** Forecast cards; reliability diagrams per user and
  crowd; edit-type difficulty ladder.
- **Interaction model.** Predict → reveal → review; spectate mode replays
  resolved markets.
- **Models & data.** C15 harness + logged ledger; outcome scoring rules
  (pre-registered).
- **Mode.** Live (piggybacks on C15 runs); replayable archive.
- **Compute.** Marginal on C15's L2.
- **Controls.** Placebo-edit questions mixed in (paraphrase edits — correct
  answer "no change"; measures sycophantic yes-bias); description-hidden
  rounds (predict from the raw edit vector norm only — does the *text*
  add predictive power? the key comparison).
- **Measurements.** Brier scores vs base-rate and vs description-hidden
  rounds; calibration slopes; learning curves.
- **Success means.** Text-visible forecasts beat text-hidden ones → the
  readable code demonstrably improves human causal understanding — the
  clearest possible evidence that this interpretability method serves its
  actual purpose.
- **Failure means.** No forecast advantage from reading the code →
  descriptions feel explanatory without being predictive: a first-class,
  quantified confabulation warning about the whole interface.
- **Confounds.** Outcome-scoring granularity; forecaster expertise mix
  (collect self-reported expertise, §5).
- **Misleading-risk.** Low; the design measures misleadingness itself.
- **Novelty.** Prediction-market epistemics applied to interpretability
  claims — not present in any current interp tool.
- **Difficulty.** ~1 week atop C15.
- **Demo appeal.** High, especially live-audience.

#### C29. Beat the Autoencoder `[companion §2.2, kept]`
Humans write the code; the frozen critic scores it.
- **Research question.** How close can motivated humans get to AV/teacher
  reconstruction quality, and what do winning human strategies reveal about
  what the AR reads?
- **User experience.** Read a passage (and position marker), write an
  explanation in the four-section template, get scored by AR against the
  stored h; leaderboard vs AV (`0.307`) and teacher (`0.305`); per-section
  ablation feedback on your submission (C8 machinery).
- **Visual encoding.** Score gauge with AV/teacher/mean-control reference
  lines; section-attribution bars on the submission; strategy gallery of
  top human entries.
- **Interaction model.** Write → score → revise; opt-in archive of
  submissions (§5).
- **Models & data.** AR + qualitative-panel rows (source-cleared).
- **Mode.** Live (single 38 GB model — the cheapest live loop).
- **Compute.** L1.
- **Controls.** Template-only and passage-copy submissions as reference
  scores; time-boxed vs unlimited conditions.
- **Measurements.** Human score distribution vs AV/teacher; revision gain
  curves; lexical analysis of top-scoring human text (do winners converge on
  AV-dialect?).
- **Success means.** A human-baseline band for the round trip plus an
  organically-grown corpus of human-authored codes (fuel for C20's
  human-legibility pane and AR-OOD analysis).
- **Failure means.** Humans plateau far above (worse than) teacher →
  writing the code requires machine knowledge humans lack; interesting for
  the dialect question either way.
- **Confounds.** Practice effects; players reverse-engineering the critic
  (that *is* data — flag late-stage entries).
- **Misleading-risk.** Low.
- **Novelty.** Human-in-the-loop scoring against a frozen reconstruction
  critic; no analogue exists.
- **Difficulty.** Days.
- **Demo appeal.** Very high; F5 act two.

---

### Family H — Trust, drift, monitoring

#### C30. Drift Seismograph `[new]`
The fresh-vs-stored gap as a first-class display, with semantic consequences.
- **Research question.** Is the stable runtime drift (mean cos `0.999142`,
  min `0.983146`, 64-row audit) semantically visible — do descriptions of
  fresh vs stored vectors differ, and do reinjection outcomes care?
- **User experience.** A seismograph strip per audited row: drift magnitude
  trace, description-diff magnitude (AV on fresh vs stored h), and
  functional-delta tick; an aggregate scatter (drift vs description change)
  with the "description noise floor" band (AV resampling variance) drawn.
- **Visual encoding.** Strip-chart seismogram aesthetic; scatter with floor
  band; worst-row drill-down cards.
- **Interaction model.** Sort by drift; inspect min-cos rows; toggle
  reinjection deltas.
- **Models & data.** The 64-row fresh-vs-stored audit vectors + AV + AR.
- **Mode.** Precomputed.
- **Compute.** L2 (small batch).
- **Controls.** AV resampling variance on identical vectors (the noise
  floor — without it the panel is uninterpretable); paired stored/fresh
  reinjection.
- **Measurements.** Description-change rate above noise floor vs drift;
  functional-delta vs drift slope.
- **Success means.** Either drift is semantically invisible (descriptions
  and behavior indifferent → the stored-snapshot caveat is metrologically
  real but semantically benign — a scope *simplification*) or visible
  (→ the caveat is load-bearing and every fresh-forward view (C12, C19)
  needs its badge burned in). Both are wins for claim hygiene.
- **Failure means.** Noise floor swamps everything → AV stochasticity, not
  drift, dominates; publish the floor.
- **Confounds.** 64 rows is small (family clustering limits inference —
  display counts).
- **Misleading-risk.** Low.
- **Novelty.** Nobody visualizes their reproducibility caveats; doing so is
  both novel and disarming.
- **Difficulty.** 2–3 days.
- **Demo appeal.** Low-medium (high for reviewers).

#### C31. Goodhart Sentinel `[companion Tier-2 instrumentation, kept and extended]`
The RL integrity console, pre-armed with SFT reference bands.
- **Research question.** During any future RL, is reward improvement
  accompanied by independent-critic agreement, functional recovery, text
  health, semanticity (C20 index), and human-legibility — or diverging from
  them (reward hacking)?
- **User experience.** A cockpit of aligned needles: primary-critic reward;
  IND-AR dMSE; reinjection KL; parse/repetition/length-drift panel (SFT
  length–dMSE corr `−0.108` as reference; the invalidated RL run reportedly
  reached `0.34–0.49` [exploratory session finding, companion doc]);
  semanticity index sampled per checkpoint; periodic Pictionary probes.
  Divergence wedges shade toward a labeled hacking region; every needle
  carries its SFT band.
- **Visual encoding.** Aligned needle gauges with reference bands; wedge
  divergence chart over training steps; checkpoint small-multiples.
- **Interaction model.** Hover a checkpoint → sampled generations with
  typography; alarm thresholds are validation-fitted and pre-registered.
- **Models & data.** RL telemetry + IND-AR + C20 batch per checkpoint +
  BASE reinjection samples.
- **Mode.** Live-during-training instrument.
- **Compute.** L2–L3 alongside training.
- **Controls.** The SFT operating point is the control; sentinel metrics
  must be excluded from the reward (else the sentinel is Goodharted too —
  state this in the preregistration).
- **Measurements.** Reward-vs-sentinel divergence rates; earliest-detection
  lead time (replayable on July-8 artifacts as a retrodiction exercise,
  labeled internal evidence only).
- **Success means.** The July-8 failure class becomes detectable-by-design;
  RL can be attempted responsibly.
- **Failure means.** Sentinel metrics all co-move with a hacked reward →
  our independence assumptions are wrong; RL stays blocked and the
  measurement contract gains a section.
- **Confounds.** Shared-teacher critic correlation (again); probe cadence
  vs training speed.
- **Misleading-risk.** Low; it is the anti-misleading device.
- **Novelty.** Instrumenting *interpretability-method training* against
  reward hacking, with human probes in the loop, is new practice.
- **Difficulty.** ~1 week given C20; integration with RL infra extra.
- **Demo appeal.** Low public; very high internal/reviewer value.

#### C32. OOD Compass `[new]`
A trust gauge fused from cheap indicators, mounted on every live panel.
- **Research question.** Can we predict, before showing a description, that
  the NLA is off its supported distribution for this input?
- **User experience.** A small compass widget beside every live input:
  needles for activation-norm band (vs training p05–p95), nearest-family
  cosine, AV NLL percentile, retrieval self-rank, and parse health; fused
  into a green/amber/red trust state with the *reasons* listed. Red states
  overlay the output with the C21 null-text comparison automatically.
- **Visual encoding.** Compass/radar glyph; reason chips; state-colored
  output frame.
- **Interaction model.** Click for indicator detail; calibration page shows
  how the fusion was validation-fitted.
- **Models & data.** PARQ statistics (norm bands, family index) + live AV/AR
  outputs.
- **Mode.** Live infrastructure.
- **Compute.** Marginal on serving.
- **Controls.** Fusion weights fitted on validation-only distortion
  correlations, applied unchanged elsewhere; red-team inputs (nonsense,
  other languages, adversarial suffixes) as acceptance tests.
- **Measurements.** Indicator-vs-dMSE correlations on held rows; red-state
  precision/recall for known-bad inputs.
- **Success means.** Live demos fail *loudly* instead of confabulating
  quietly — the precondition for any public deployment.
- **Failure means.** Indicators don't predict distortion → we cannot
  currently detect OOD inputs; live modes should restrict to curated
  corpora, and that finding gates F5's design.
- **Confounds.** Indicator correlations may themselves be family-structured
  (cluster the fit).
- **Misleading-risk.** Medium if over-trusted (a green light is not a
  guarantee — the legend must say "no red flags detected," not "trusted").
- **Novelty.** Per-interaction OOD gauges are rare anywhere in ML demos;
  absent in interp tools.
- **Difficulty.** Days.
- **Demo appeal.** Quietly high — reviewers notice it.

---

## 3. Five Flagship Experiences

### F1. The Bottleneck Spectrometer — *rigorous scientific instrument*
**(unifies C1 + C7 + C10 + C11; the instrument for "where do the bits die?")**

One instrument, three linked spectra of the same channel:

- **Spectral pane (C1):** per-PC transmission strip with residual-PCA
  drill-down and speak-the-blind-spot verbalizations.
- **Lexical pane (C7 + C11):** words-buy-direction curves, per-token marginal
  value, and corruption-robustness curves — the code's rate and redundancy.
- **Identification pane (C10):** the capacity ladder with its assumptions
  card, stratified small multiples, and readable confusion graph.

**Why it is the instrument.** Each pane measures the same loss (`≈0.30`
directional MSE) in a different basis — geometric, lexical, and
combinatorial. The scientific payoff is in the *cross-links*: select a
residual PC in the spectral pane and the lexical pane re-renders marginal
word values *restricted to that subspace* (which words, if any, buy the
blind direction?); the identification pane highlights confusions whose
differences live in that subspace. A hypothesis born in one basis is
immediately testable in the other two.

**Workflow walkthrough.** A researcher notices residual-PC-3 is stable
across critics and splits. Clicking it: the top-loading rows are all
mid-list positions in enumerations; the lexical pane shows *no* word in any
explanation buys PC-3 (flat marginal curves in-subspace); the confusion
graph shows exactly these rows confusing with their own list-neighbors.
Hypothesis card auto-drafts: *"Enumeration-index content at R33 is not
addressed by the current format"* — with an attached NT-6-style experiment
suggestion (an ordinal anchor line) and the evidence bundle hashed.

**Requirements.** CACHE + TEXT for everything except blind-spot
verbalizations (one L2 batch) and corruption/truncation batches (L1,
~50k AR forwards total — hours). All panes ship as static JSON afterward.

**Controls & standards.** Teacher-text twins of every curve; IND-AR overlay
on every statistic; family-clustered bands; assumptions card permanently
visible in the identification pane; all subspace claims verified in native
2,688-d, never from the projection.

**Prototype plan.** Week 1: spectral + identification panes offline. Week 2:
AR batches for lexical pane; cross-linking. The hypothesis-card exporter is
a text template — cheap and disproportionately valuable.

### F2. The Surgery Theater — *causal intervention interface*
**(C15 + C16 + C6 policy + C28 hooks)**

A controlled operating room for editing internal state via language, built
so that every seductive success is displayed beside its placebo.

**The stage.** Left: the patient — source prefix, position marker,
description with saliency tint. Center: the incision — an editable code
pane; every edit spawns its paraphrase-placebo and matched-random-edit
siblings automatically, shown as three parallel operating lanes. Right: the
outcome wall — next-token morphs, Δcos compass, dose–response strip (C16),
wake profile (C13), and the effect ledger accumulating every run with
family-clustered summaries per edit type.

**The discipline.** (1) *No unlogged runs*: the ledger is append-only and
hash-chained (same discipline as the eval verifiers). (2) *Blind outcome
calls*: when an intent is pre-registered ("continuation becomes culinary"),
the hit/miss is scored by a rubric evaluated without seeing which lane was
real — the paraphrase lane is the sham surgery. (3) *Dose or it didn't
happen*: single-α successes are labeled anecdotes; the α-strip is one click.
(4) *Norm policy pinned*: C6's chosen calibration is displayed on every run.

**What science it yields.** Per-edit-type causal effect sizes with placebo
subtraction; the first quantitative map of *which described properties are
steerable* (register? topic? next-token constraints?) and at what dose; and
via C28's forecast hooks, whether the code teaches humans a predictive
causal model.

**Requirements.** L2 (AR + BASE resident, phased loading works); the C15
serving loop is ~2 forwards per lane. Precomputed preset-chip galleries let
the theater demo offline.

**Failure is informative by design.** If placebo≈edit everywhere, the
readable control surface is refuted cheaply and publicly — that result,
with its ledger, is a paper section, not an embarrassment.

**Prototype plan.** The companion's §2.1 loop first (its cheapest live
loop), then lanes + ledger (days), dose strip (days), blind scoring (§5
rubric work).

### F3. Residual-Stream Cinema — *token-by-token internal dynamics*
**(C12 + C14 + C13 overlay)**

A genome-browser for the residual stream, where the annotation tracks are
written by the model's own verbalizer.

**The experience.** A document runs left-to-right. Track 1: source tokens.
Tracks 2–5: the four description sections, drawn as *persistence bars* —
text that survives across positions renders as one continuous labeled span;
churn renders as fragmentation. Track 6: diff-magnitude heat. Track 7
(optional): wake overlays from C13 at positions where patches were tested —
observed dynamics and causal dynamics on one timeline. A side pane mines
the corpus-level state-transition grammar (C14) and lets you jump to any
document that exercises a chosen transition.

**The scientific object** is the *timescale spectrum by content type*:
half-life distributions per section, change-point alignment against
linguistic boundaries, and their stability across documents — a dynamical
characterization of a post-Mamba boundary no probe-based track can produce,
because only this readout is diffable prose.

**Honesty engineering.** Every filmstrip carries a fresh-forward badge (this
is outside the stored-snapshot claim); a shuffled-position control strip is
one keypress away (does apparent persistence exceed topic-chance?); greedy
decoding with a resample-jitter band distinguishes churn from sampling
noise; the C21 tint marks which changed words exceed the null text.

**Requirements.** L2 precompute: 5–20 documents × ~100 positions × (1
extraction + 1 AV generation) — a few GPU-hours; then fully offline.
Live single-document mode needs L3.

**Prototype plan.** Week 1: extraction+verbalization batch, track renderer,
diff engine. Week 2: persistence/half-life statistics, controls, grammar
miner (C14) atop the same corpus.

### F4. The Cipher Court — *the failure-mode flagship*
**(C20 + C21 + C11 + C23 + C32; the instrument that decides when NOT to trust the NLA)**

An adversarial proceeding against the code's semanticity, rendered as a
courtroom where every row can be put on trial — and the verdicts aggregate
into the project's trust dashboard.

**The trial.** For a selected row, evidence is heard in fixed order:
(1) *Paraphrase testimony* — the dispersion clouds; does meaning-preserving
rewording preserve the vector? (2) *Redundancy testimony* (C11) — graceful
degradation or cliffs? (3) *Legibility testimony* — human-rated content
words vs filler, masked reconstructions; who can read this code?
(4) *Cross-reader testimony* — IND-AR agreement, with the shared-teacher
confound read into the record verbatim (per the measurement contract,
shared errors are *not* proof of AV-side loss). (5) *Null comparison* —
how much of this text is almanac boilerplate? The verdict chip (honest /
mixed / suspect) is computed from validation-fitted thresholds and displayed
with all five evidence values, never alone.

**The docket view** sorts the corpus by verdict and lets suspicion be
studied as a population: is "suspect" enriched in short texts, low-norm
rows (the weakest qualified subgroup), particular families? Docket
statistics feed C31's sentinel bands directly: RL must not raise the
suspect rate.

**Why a failure-mode flagship.** The July-8 invalidation proved this
project's biggest risk is not low fidelity but *illegitimate* fidelity.
The Court makes the integrity case visible, quantitative, and repeatable —
and if the SFT-era code is already partly cipher, this is the room where
that surfaces first, before RL makes it worse.

**Requirements.** L1 batches (paraphrase/corruption/masking grids); §5
pipelines for human ratings and paraphrases; no live models needed for the
core.

**Prototype plan.** Week 1: paraphrase + corruption batches, verdict logic
(validation-fitted), docket UI. Week 2: human-rating pipeline integration,
IND-AR replication overlays, sentinel band export.

### F5. The Glass Box Booth — *the unforgettable public demonstration*
**(C27 + C29 + C15-lite + C18 + C23 + C32, staged as a three-act experience)**

A walk-up installation (conference booth or web) where a visitor reads,
tests, and writes a model's internal code in under five minutes — with the
epistemics built into the theater rather than a disclaimer slide.

- **Act I — Read.** The visitor types a sentence (length-capped, screened).
  A cinema-lite strip (C12, three positions) shows the code at each
  position with provenance typography (C23) and the OOD compass (C32)
  glowing green/amber/red. If red: the booth *shows the confabulation
  live* — the same input beside the null-text almanac, teaching the failure
  mode as content.
- **Act II — Test.** Three rounds of Activation Pictionary (C27) against
  the AR's retrieval: the visitor experiences how identifying the code is —
  and sees their d′ land on the human-distribution strip next to the AR's.
  One round is always a shuffled-control ("this one was a lie — could you
  tell?").
- **Act III — Write.** Beat-the-Autoencoder (C29) or, for the finale,
  Write-Only Programming (C18): author a code from the template, watch the
  continuation shift — with the paraphrase-placebo lane running visibly in
  parallel, so the audience sees what a null result looks like on stage.

**Framing kit (non-negotiable).** Fixed vocabulary on every surface: "a
learned description of one internal snapshot," never "the model's
thoughts"; the mean-control ghost visible in every score; a persistent
"exploratory demo — stored-snapshot protocol" banner; the failure-mode act
(the red-compass path and the shuffled round) is *scripted into* the
experience, not hidden. Sealed-test rows never appear; all passages come
from the privacy-cleared panel.

**Data it generates (with consent, §5).** Human identification d′,
human-authored codes, paraphrase judgments, forecast accuracy — the booth
is the experiment generator wearing a costume.

**Requirements.** Acts II run at L0–L1; Act I and III need L2–L3 (the
companion's serving architecture: FastAPI over the existing repo
primitives, protocol-hashed, audit-logged). A degraded all-offline mode
(precomputed Act I examples, Acts II fully) must exist for venue-network
reality.

**Prototype plan.** Build Act II first (days, content exists), Act III
AR-only next (C29), Act I last (needs AV serving). Rehearse the red-compass
path deliberately — the demo's credibility *is* the failure it shows.

---

## 4. Unified Observatory

**Name.** The R33 Activation Observatory. One coherent instrument, not a
chart gallery: every view is a station along a single investigative
workflow, sharing one selection state, one interaction grammar, and one
provenance system.

### Core workflow: five altitudes

```
ATLAS (corpus)  →  SPECTRUM (channel)  →  REEL (dynamics)  →  BENCH (interventions)  →  COURT (integrity)
```

1. **Atlas** — where am I? Corpus overview: family map, subgroup strata,
   lump/split anomalies (C4), writable-set coverage (C5), capacity ladder
   (C10). Answers "which rows/regions deserve attention?"
2. **Spectrum** — what does the channel carry here? F1 panes scoped to the
   current selection (per-PC transmission, word value, redundancy).
3. **Reel** — how does it move? F3 filmstrips for documents intersecting
   the selection; grammar view (C14); drift seismograph (C30) for audited
   rows.
4. **Bench** — what happens if I change it? F2 surgery theater, dose
   strips (C16), transplants (C17), programming console (C18), mediation
   cascades (C19).
5. **Court** — should I believe any of this? F4 verdicts for the selected
   rows; almanac (C21); sentinel (C31) when training is live.

The altitudes are ordered by epistemic escalation: description → mechanism
→ intervention → integrity. Moving right requires carrying evidence, not
vibes: the Bench's row picker displays the Court verdict and Spectrum
profile of whatever you are about to operate on.

### The shared cursor

One global selection tuple — `(rows | family | stratum, position, section |
token-span, subspace, condition, critic, split)` — and every panel is a
projection of it. Brushing the Atlas re-scopes the Spectrum; picking a
change point in the Reel pre-loads the Bench at that position; every Bench
run files into the Court's docket. Nothing renders that cannot say which
cursor it is showing.

### Interaction grammar: five verbs

Every station implements the same verbs, so learning one panel teaches all:

- **select** — brush/click to move the cursor;
- **compare** — pin up to three cursor states side-by-side (the only way
  two conditions ever appear together: no unpinned overlays, so every
  comparison is deliberate and labeled);
- **perturb** — the verb that spawns model calls (truncate, corrupt, edit,
  dose, transplant); *always* spawns its registered controls with it;
- **trace** — follow one row/direction across stations (breadcrumb rail
  shows the row's journey: atlas position → spectrum profile → filmstrip →
  ledger entries → verdict);
- **audit** — flip any number/mark to its provenance: report hash, verifier
  hash, protocol hash, split, family count, fitted-on-what.

### Static vs live modes

Three badges, colored and non-suppressible:

- **FROZEN** (grey): computed from hash-bound artifacts (CACHE, TEXT,
  FUNC); identical for every user; citable with its verifier hash.
- **PRECOMPUTED** (blue): one-off GPU batches (Shapley grids, wakes,
  filmstrips) with archived configs + hashes; re-runnable, exploratory.
- **LIVE** (amber): model calls in this session; stamped
  `exploratory — not evidence` in the export; logged to the audit JSONL.

The Observatory boots fully functional in FROZEN+PRECOMPUTED mode with no
GPUs (the archive is the product; the service is an upgrade), matching the
companion's serving plan (AR-only L1 → full L3).

### Aggregate → individual movement

"Every mark is a door": any aggregate glyph (a CI whisker, a heatmap cell,
a curve) opens the row list behind it, and every row card offers its
passage, code (with typography), reconstruction profile, and journey rail.
The reverse door exists too: from any row, "show me the population" jumps
to its family/stratum aggregates — protecting users from generalizing an
anecdote. Sealed/test rows render with a watermark and are excluded from
perturb verbs in public deployments.

### Uncertainty and provenance displays

- Family-clustered 95% bands on every aggregate; row counts *and* family
  counts printed on every panel (the contract's statistical unit).
- The **mean-control ghost**: every reconstruction-quality display draws
  the train-mean predictor's achievement as a grey ghost layer, so shared-
  component credit is never mistaken for row-specific fidelity.
- Hash chips everywhere (report/verifier SHA-256 prefixes, click to copy);
  fitted-on-validation markers on every threshold; protocol chips on every
  generation.
- Per-token typography (C23) is the default text rendering everywhere.

### Safeguards against anthropomorphic interpretation

1. **Controlled vocabulary**, enforced in UI copy review: "description /
   code / encoding of a stored activation," never "thought," "belief,"
   "the model says about itself." The one allowed metaphor is channel/code
   language, which is accurate.
2. **Null adjacency**: the C21 almanac comparison is one keypress from any
   description, and red OOD-compass states auto-surface it.
3. **Failure as furniture**: the Court is a top-level station, not an
   appendix; docket statistics appear on the Atlas landing view.
4. **No bare text**: descriptions render only with typography + provenance
   chips; screenshots therefore carry their own caveats.
5. **Scope banners**: stored-snapshot vs fresh-forward badges are part of
   the data model, not the CSS.

---

## 5. Experiment Generator

Interactions that produce research-grade evidence while users explore —
with the governance that keeps a public demo from becoming an uncontrolled
benchmark.

### Instruments and the data they yield

| Interaction | Data generated | Feeds |
|---|---|---|
| Beat the Autoencoder (C29) | human-authored codes + AR scores + revision trajectories | human baseline band; AR-OOD generalization; C20 legibility pane |
| Activation Pictionary (C27) | human identification choices, RTs, confidence | human d′ vs AR; human–AR error overlap; C4 lump validation |
| Paraphrase tasks (C20) | human paraphrases + "same meaning?" judgments | semanticity index's human leg; paraphrase-generator audit |
| Surgery ledger (F2) | edit intents, doses, outcomes, placebo outcomes | steerability effect sizes per edit type |
| Prediction market (C28) | forecasts + calibration vs text-hidden control | does the code improve human causal models? |
| Blind condition ID (C27 shuffled rounds, B.8 lineup) | human real-vs-control discrimination | manipulation checks; cipher evidence |
| Typography A/B (C23) | trust-calibration responses | uncertainty-communication evaluation |

Design detail that makes these *experiments* rather than telemetry: every
instrument embeds its manipulation checks (shuffled rounds at chance,
placebo edits, text-hidden forecast arms), so each dataset carries its own
validity evidence.

### Consent and privacy

- Explicit opt-in per instrument, separate from playing: play without
  contributing is always possible; the toggle states exactly what is stored
  (choices, timings, authored text) and what never is (IP, identity unless
  a pseudonymous handle is chosen).
- User-authored text passes the existing release-text triage patterns
  (PII/credential scan) before storage; flagged items are dropped, not
  quarantined.
- Corpus passages shown publicly come only from the privacy-cleared panel;
  rows with source-side phone-like patterns stay internal per the standing
  triage finding. Sealed-test rows are never served.
- If results are to be published as human-subjects findings, obtain the
  appropriate internal review *before* collection, not retroactively.

### Sampling bias, stated and bounded

Booth/web players are self-selected, likely ML-adjacent, and improve with
play. Therefore: collect coarse self-reported expertise and prior exposure;
report all human numbers as "convenience-sample performance" with learning
curves shown, never as "human performance"; freeze a first-exposure subset
(round 1 only) for the least-contaminated estimates.

### Keeping the demo from becoming an uncontrolled benchmark

- **Pool separation.** Public instruments draw from a designated
  demo pool (validation-side, privacy-cleared). Test-split rows are never
  exposed; validation aggregates from public play are labeled exploratory.
- **Preregistration for confirmatory use.** Any human-data claim intended
  as more than exploratory gets a preregistered analysis plan (hypothesis,
  sample size, exclusion rules, metrics) *before* the collection window —
  the project already has the preregistration machinery (queue contracts,
  hash-bound plans) to do this properly.
- **Version locking.** Every datum records protocol hash, checkpoint
  fingerprints, UI version, and instrument version; cross-version pooling
  is an explicit, logged analysis decision.
- **Anti-gaming.** Rate limits; late-stage Beat-the-Autoencoder entries
  flagged for critic reverse-engineering analysis (that behavior is itself
  data about exploitability — route it to C31, don't discard it).
- **Telemetry firewall.** Product analytics (feature usage) and scientific
  data (instrument responses) are separate stores with separate retention;
  scientific analyses never quietly ingest telemetry.

---

## 6. Prioritized Roadmap

Ratings: **Sci** scientific value, **Orig** originality, **Vis** visual
impact, **Cost** implementation cost, **GPU** compute cost, **Risk**
(of producing misleading or unusable output). H/M/L; for Cost/GPU/Risk,
L is good.

| # | Concept | Sci | Orig | Vis | Cost | GPU | Risk |
|---|---|---|---|---|---|---|---|
| C1 | Residual Anatomy Atlas | H | H | M | M | L–M | M |
| C2 | Two Charts, One Manifold | H | H | H | M | M | M |
| C3 | Semantic Telephone Observatory | M | M | H | M | M | L |
| C4 | Lump & Split Atlas | M | M | M | L | L | L |
| C5 | The Writable Set | H | H | H | M | L | M |
| C6 | Norm Blind-Spot Explorer | H | M | M | L | M | L |
| C7 | Words-Buy-Direction | H | H | H | L | L | L |
| C8 | Code Attribution Lens | H | H | H | M | L | M |
| C9 | Consensus Constellation | H | M | H | M | M | L |
| C10 | Capacity Ladder | M | M | M | L | L | L |
| C11 | Redundancy Probe | M | M | M | L | L | L |
| C12 | Residual-Stream Cinema | H | H | H | H | M | M–H |
| C13 | Causal Wake Profiler | H | M | H | M | M | L |
| C14 | State-Transition Grammar | M | H | M | M | L | M |
| C15 | Explanation Surgery Bench | H | H | H | H | M | H* |
| C16 | Semantic Dose–Response | H | H | M | L | M | M |
| C17 | Transplant Matrix | M | M | H | L | M | M |
| C18 | Write-Only Programming | H | H | H | M | M | H* |
| C19 | Mediation Cascade | H | H | M | H | M | L |
| C20 | Paraphrase Microscope | H | H | M | M | L | L |
| C21 | Null-Text Almanac | M | M | L | L | L | L |
| C22 | Upset Forensics | M | L | M | L | L | L |
| C23 | Provenance Typography | M | M | M | L | L | L |
| C24 | Rosetta Panel | H | H | M | H | H (L4) | M |
| C25 | Steering Cross-Validation | M | H | M | M | M | M |
| C26 | Boundary Babel | H | M | M | H | H (L4) | M |
| C27 | Activation Pictionary | H | M | H | L | L | L |
| C28 | Prediction Market | H | H | M | M | M | L |
| C29 | Beat the Autoencoder | M | M | H | L | L | L |
| C30 | Drift Seismograph | M | M | L | L | M | L |
| C31 | Goodhart Sentinel | H | M | L | M | M | L |
| C32 | OOD Compass | M | M | L | L | L | M |

\* Risk is high *without* the built-in controls; the designs above exist to
buy that risk down. Ship them only as designed (placebo lanes, ledgers).

**Priority logic.** Highest-leverage cluster = {C20, C7, C8, C1, C27}:
low-to-moderate cost, mostly L0–L1, and they answer the two questions
everything else depends on (is the code semantic? where is the
information?). The causal cluster {C15, C16, C13} is the biggest scientific
upside and the biggest care requirement. The L4-gated items (C24, C26) are
valuable but should wait for an SAE/second-boundary decision made on
research grounds, not visualization grounds.

### One-week prototype (1 person, ≤1 short GPU session)

*Goal: the frozen-artifact core of F1 plus the first integrity signal.*

1. Days 1–2: **Capacity Ladder (C10)** + **Lump & Split (C4)** + **Upset
   Forensics (C22)** from CACHE/TEXT — pure offline; establishes the
   linked-view skeleton and hash-chip plumbing.
2. Days 3–4: one L1 session: word-truncation + occlusion + corruption grids
   → **Words-Buy-Direction (C7)**, **Code Attribution Lens (C8, word pane)**,
   **Redundancy Probe (C11)**; plus the **Null-Text Almanac (C21)** overnight
   (CPU).
3. Day 5: **paraphrase grid v0 (C20)** with LLM paraphrases (human leg
   deferred) — first semanticity index, validation-fitted verdict chips.

Deliverable: a static observatory slice (Atlas-lite + Spectrum + Court-v0)
that already answers "which words carry it, how robust is it, and is it
plausibly semantic," entirely from frozen artifacts + one GPU day.

### One-month research dashboard (1–2 people, recurring L1–L2 sessions)

Add, in order: **C1** residual atlas (+ blind-spot batch), **C6** norm
explorer with reinjection sweeps, **C13** wakes + **C17** transplant matrix
(shared reinjection harness), **C12** cinema precompute for ~10 documents,
**C30** drift seismograph, **C9** constellation batch, full **F4 Cipher
Court** with human-rating pipeline v0, **C23** typography and **C32**
compass as infrastructure, and the AR-only live loop (**C29** + C15-lite
surgery showing Δcos only). This is the complete FROZEN+PRECOMPUTED
observatory plus one modest live capability — the internal research tool
the publication follow-ups can cite for exploratory analyses.

### Ambitious flagship system (team, quarter-scale, L3 resident service)

The full Observatory: all five altitudes linked by the shared cursor; **F2
Surgery Theater** with ledger, doses, blind scoring, and **C28** prediction
markets; **F3 Cinema** with grammar mining and live single-document mode;
**F5 Glass Box Booth** as the public face with the §5 experiment generator
and its governance; **C31 Sentinel** armed with F4's bands, standing ready
for any future preregistered RL. Stretch (research-gated): **C24 Rosetta**
after an R33 SAE exists; **C2/C5** atlases as the geometry deep-dive;
**C26 Boundary Babel** if a second boundary pair is ever trained.

---

## Standards recap (applies to every concept above)

- Descriptions are learned encodings of selected stored activations —
  never presented as the model's literal thought or verified reasoning.
- Directional claims only; magnitude always displayed with the C6 caveat;
  stored-snapshot scope badged; fresh-forward views labeled exploratory.
- No causal language without intervention + controls; no capacity language
  without the assumptions card; shared-critic agreement is evidence, not
  proof, of AV-side loss.
- All thresholds validation-fitted, applied unchanged; sealed test never
  consumed for new confirmatory purposes; exploratory findings labeled at
  the point of display, not in a footnote.
- Negative and ambiguous outcomes are first-class: several concepts above
  are *designed* so their failure is the publishable result.
