# The NLA Offline Observatory — Reduced-Scope Design (v2)

Status: design specification, `2026-07-16`, revision 2 (scope reduced on
request; the v1 full-scope inventory survives in
`docs/viz/nla_visualization_frontier_ideas.md`, cited as `C#`/`F#`, and in
the descoped register, §9). Companion: `docs/viz/nla_visualization_ideas.md`
("companion §x.y").

One dashboard. Static files, CPU/browser only, **no GPU and no model at
runtime**. Everything rendered comes from one versioned, content-addressed,
precomputed bundle.

> [!CAUTION]
> Exploratory-framing only. The exhibit inherits every boundary of
> `docs/methods/measurement_contract.md` and
> `docs/runs/r33_clean_sft_av_ar_20260715.md`: directional (not
> raw-magnitude) reconstruction, stored-snapshot functional scope for
> qualified rows, fresh-forward exploratory scope for new extractions,
> family-clustered uncertainty, exploratory test split. Descriptions are
> learned encodings of selected activations, never the model's literal
> thoughts.

---

## 1. What Survived, and Why

Selection criteria: scientific value per unit of build effort; experiences
only an NLA can offer (readable, editable, reconstructable, reinjectable
code); zero runtime compute; minimal one-time GPU budget.

The reduced exhibit tells **one story in four rooms**:

```text
CHANNEL   what survives the trip through English          (the evidence)
TRACE     watch the code move through a document          (the dynamics)
BENCH     edit the code, see the consequences             (the intervention)
AUDIT     know exactly when not to trust any of it        (the integrity)
```

Everything that did not directly serve this spine was cut (§9). Headline
consequences of the reduction:

- **External Prompt Atlas dropped** — no dataset importers, no license
  review (BLiMP/GSM8K/HumanEval/FLORES), no carrier-prefix machinery, no
  OOD-compass fitting. The exhibit shows *the qualified result, explorable*.
- **Five stations → four** — COMPOSE and CONSEQUENCE merge into BENCH
  (choose a precomputed edit and see its outcome are one act, not two).
- **Two-tier bundle → one bundle** (~100–150 MB, §6). At this size DuckDB-
  WASM is unnecessary; plain Arrow + typed-array workers suffice.
- **GPU cost: roughly half an H200-day** (planning estimate, §7) instead
  of 1–3 days — one afternoon, then the workspace is deleted.

---

## 2. Corpus: Two Populations (+ a Ten-Document Film Set)

| Population | Badge | Contents | Claim scope |
|---|---|---|---|
| **QUALIFIED** | blue | The 512 evaluated R33 validation rows | Authoritative; family-clustered stats; stored-snapshot functional semantics |
| **FROZEN-EXPLORATORY** | amber | The 512 evaluated test rows | Frozen reporting of already-published exploratory numbers only; **no new lattice work**; never pooled into authoritative aggregates |

New text enters the corpus in exactly one place: the **film set** for
TRACE — 8–12 documents, drawn from the privacy-cleared panel's validation
rows where the resolved source text is long enough (≥ ~80 tokens beyond the
`_MIN_POSITION = 50` floor), topped up if needed with hand-curated
public-domain passages (attribution recorded per passage; no dataset
importers, no license table).

Text display policy (unchanged from v1): source text renders only for the
100-row privacy-cleared panel and the film set; all 1,024 frozen generated
explanations render (they passed the release-text triage); the bundle
verifier re-runs the PII/credential scan over every shipped text field.

Deep-dive (lattice) work is restricted to the **50 cleared validation
rows**. The test half of the panel ships display-only.

---

## 3. One Dashboard, Four Stations

Shell furniture on every station: population badges; the selection tuple σ
rendered as removable chips; provenance typography (C23: per-token AV
logprob opacity, occlusion-saliency underline, optional null-text tint);
the **mean-control ghost** on every reconstruction-quality mark; hash chips
linking numbers to their report SHA-256s.

### 3.1 CHANNEL — what survives the trip through English

Data: population pack (mostly existing caches) + deep-dive lattice.

| View | Lineage | One line |
|---|---|---|
| Information waterfall | companion §0.4 | The ladder: random → controls (`0.68–0.97`) → the language shelf (`≈0.30` dMSE) → source_raw floor (`≈0.083`), each rung with example texts |
| Capacity ladder | C10, companion §0.2 | Retrieval vs gallery size with bits axis, permanent assumptions card, readable confusion graph |
| Twin critics | companion §0.1 | Primary vs independent AR agreement and error-vector cosines, with the shared-teacher confound stated on-panel |
| Spectral strip | C1-lite, companion §0.3 | Per-PC transmitted vs residual energy; residual scree with cross-critic stability bands (blind-spot *verbalizations* descoped) |
| Words-buy-direction | C7, companion §A.3 | Truncation rate–distortion where the rate axis is words; per-token marginal-value typography; AV vs teacher word-efficiency |
| Code attribution lens | C8, companion §A.1–A.2 | Word-occlusion tints + exact section Shapley + section→PC-band ribbons |
| Tellings fan | C9-lite, companion §2.4 | k=8 sampled tellings per row: fan on the PC map, ensemble-vs-k curve — the sampling-noise-vs-channel-ceiling evidence that prices RAFT/RL headroom |

Every aggregate mark opens its row list; every row card shows its
explanation with typography and its journey across stations.

### 3.2 TRACE — watch the code move

Data: temporal pack (film set). Fresh-forward badge on every panel (these
are new extractions, outside the stored-snapshot claim).

| View | Lineage | One line |
|---|---|---|
| Cinema tracks | C12 / F3, companion §C.11 | Source-token reference track; one persistence-bar track per explanation section (unchanged text = one continuous bar); word-diff heat lane; change-point glyphs |
| Half-life barcodes | C12 | Per-section persistence distributions across the film set — the slow-vs-fast feature spectrum of R33 in its own vocabulary |
| Planning lens | Anthropic Planning in Poetry case study | A poetry reel scrubs through the first line token by token, highlights AV references to the future rhyme before it appears, and compares the eventual continuation with a precomputed NLA-edit steering result |

Shipped controls: shuffled-position description track (does persistence
beat topic-chance?) and a greedy-vs-resample jitter band (sampling-noise
floor). Interaction: scrub, diff any change point, and — where the change
point's row is in the lattice — jump to BENCH.

The planning lens is intentionally exploratory. Its offline pack contains
curated and templated rhyming couplets, plus non-rhyming prose and
shuffled-rhyme controls. At each source position it stores several AV
tellings, a phonetic rhyme-family score for the held-out line-ending, and
the target model's continuation. A small causal lane edits the planned
rhyme family in the newline-token explanation, reconstructs the original
and edited activations with AR, patches their difference at R33, and shows
whether completion mass moves toward the edited rhyme family. The UI calls
this a rough planning signal, not proof of a unique internal plan.

Simple hardening is built into the poetry pack without turning the exhibit
into a confirmatory study:

- **Causal-prefix seal:** future target words may occur in the held-out second
  line but are rejected if they appear anywhere in the analyzed prefix.
- **Family scoring:** exact future-word mentions and manually specified rhyme
  families are scored separately, avoiding brittle tokenizer-string matches.
- **Emergence curve:** several positions before the newline are sampled so a
  plan should strengthen toward the line break instead of appearing uniformly.
- **Tellings and nulls:** four real AV samples and two same-position,
  wrong-poem activation controls estimate sampling jitter and topic-level
  false positives.
- **Behavior agreement:** the unpatched continuation must be shown beside the
  latent readout; disagreement remains visible rather than being filtered out.
- **Causal edit:** an explicit rhyme-family edit is reconstructed by AR and
  patched at the newline at several doses, next to a norm-matched random
  direction control.

### 3.3 BENCH — edit the code, see the consequences

Data: deep-dive lattice (§5) + behavior tables. The **Counterfactual
Composer**: the writable-latent experience without a live model.

Compose side:

- **Move picker** over the lattice's families: preset clause swaps,
  section ablation, truncation, paraphrase, alternate tellings.
- **Control rack**: selecting a clause swap auto-loads its
  paraphrase-placebo and matched-random-edit partners plus the identity
  cell; the lanes travel together and cannot be dismissed individually
  (control groups are a property of the data layout, not UI discipline).
- **Coverage map**: exactly which cells exist at which depth — absence is
  visible, never silent. A fixed line states *"you are choosing among N
  precomputed experiments (grid spec `<hash>`), not editing live."*

Consequence side:

| Pane | Lineage | One line |
|---|---|---|
| Reconstruction outcome | C15-lite | Δcos compass on the PC map; directional metrics for both critics; norm ratio with the calibration caveat |
| Next-token lineup | companion §B.8 | Original vs patched vs control-lane top-k distributions with movement arrows |
| Divergence gauges | FUNC schema | KL / JS / top-10/50 overlap with the qualified reference rails (e.g. test AV KL `0.9495` vs mean `4.12` / zero `6.30` / shuffled `9.53`) |
| Causal wake | C13, companion §B.9 | Teacher-forced per-position KL `t+1…t+16` as a decaying wave; per-lane overlay; behavioral half-life stat |
| Dose strip | C16-lite | Metrics at α ∈ {0, ½, 1} for clause swaps — is the effect graded? |
| Session ledger | F2 | Append-only local list of every cell opened, with its control lanes; exportable as JSON (cell IDs + σ tuples only) |

METRIC-depth cells light the first pane only; BEHAVIOR-depth cells light
everything. Scope badge on all behavior panes: stored-snapshot protocol.

### 3.4 AUDIT — when not to trust it

| View | Lineage | One line |
|---|---|---|
| Claim ledger | model card | The supported / not-supported lists as first-class UI; every number elsewhere deep-links to its claim card |
| Cipher Court docket | C20 / F4 | Per-row semanticity verdicts (honest / mixed / suspect) from three evidence legs: paraphrase dispersion, corruption cliffs, cross-critic agreement (with its shared-teacher confound note); docket-level enrichment stats |
| Null-text almanac | C21, companion §0.7 | The AV's control-condition prior: boilerplate lexicon and per-token real-vs-zero log-odds (drives the shell tint); blind-lineup examples |
| Magnitude card | C6-lite, companion §0.8c | Shrinkage cone + the validation-fitted `0.560604` scalar story — why the headline metric is directional (behavioral norm sweeps descoped) |
| Drift card | C30 | The 64-row fresh-vs-stored audit (mean cos `0.999142`, min `0.983146`) — why TRACE wears a fresh-forward badge |
| Provenance browser | §7 | Manifest tree: shard hashes, checkpoint fingerprints, protocol hashes, grid spec, verifier report |

### 3.5 Challenge drawer

Local-only scoring (static hosting has no backend; localStorage streaks;
an explicit export-my-session file is the only way data leaves).

- **Activation Pictionary** (C27): read a description, pick the source
  passage among k — privacy-cleared rows only; shuffled rounds embedded as
  the at-chance manipulation check.
- **Forecast-before-reveal** (C28 offline variant, stretch): predict a
  BENCH cell's outcome before the panes unblur; paraphrase-placebo cells
  mixed in; scored locally.

---

## 4. Selection Tuple and Permalinks

```text
σ = (bundle_id, population, row_id | doc_id, position,
     variant | cell_id, dose, critic, comparison_pins[≤3], station, view)
```

σ serializes into the URL fragment; **permalink = citation** — any figure
reopens bit-identically against the same bundle, and a bundle-hash mismatch
renders a visible version warning. Pinning (≤3 σ states side-by-side) is
the only comparison mechanism, so every comparison is deliberate and
labeled.

---

## 5. The Reduced Lattice

Rows: the **50 cleared validation rows**. Per row:

| Move family | Cells | Depth |
|---|---|---|
| Preset clause swaps | 6 chips × 3 lanes (edit / paraphrase-placebo / random-edit) | BEHAVIOR on 24 designated rows (with dose α ∈ {½, 1}); METRIC elsewhere |
| Section ablation | 16 (all 2⁴ subsets → exact Shapley) | METRIC |
| Word occlusion | ~60–100 (one per word) | METRIC |
| Truncation | ~10 prefixes | METRIC |
| Paraphrase grid | 6 types | METRIC (Court evidence; IND-AR scored) |
| Corruption grid | 2 types × 3 rates | METRIC (Court evidence; IND-AR scored) |
| Alternate tellings | k=8 | METRIC |

Cell contents: exact intervention spec, edited/generated text, ĥ (fp16),
directional metrics (primary; independent AR on Court-relevant cells), and
for BEHAVIOR cells: top-k next-token ids+probs, KL/JS/overlap, wake trace
`t+1…t+16`, greedy continuation (≤32 tokens).

Budget (planning): ≈ 200 AR-scored cells/row × 50 rows ≈ **10k AR
forwards**; BEHAVIOR ≈ 24 rows × 6 chips × 3 lanes × 2 doses + baselines ≈
**~1k patched BASE forwards**; tellings + film set ≈ **~1k AV
generations**. The lattice spec is itself a hashed config (`grid spec`),
shown in BENCH, so coverage is a citable object.

---

## 6. Data Bundle (one tier)

```text
observatory_manifest.json    shard index: path, sha256, bytes, schema_version
provenance.json              fingerprints, protocol hashes, report bindings
aggregates.json              headline stats + family-clustered CIs (verifier re-derives)
rows.parquet                 registry: population, ids, family, subgroup bins,
                             retrieval rank, neighbors, drift metrics, text-visibility
metrics.parquet              long format: (row|cell, variant, critic, metric, value)
explanations.parquet         (ref, kind, text, parse_state, token_logprobs, protocol_sha)
interventions.parquet        lattice registry incl. control_group_id and depth
behavior.parquet             BEHAVIOR outcomes incl. wake arrays and continuations
token_trajectories.parquet   film set: per-position text refs, diff/persistence spans
geometry.parquet             projection coords; pointers into vector shards
vectors/*.f16.bin            full 2,688-d fp16 vectors (targets, variants, cells)
assets/                      legend cards, claim-ledger text, almanac dictionary
```

Estimated total **~100–150 MB** (fp16 vectors ≈ 5.4 KB each: ~10k
population/variant + ~10k cell + ~0.4k temporal ≈ 110 MB; parquet/text
~20 MB). One bundle, lazy-loaded by shard, service-worker cached.
`bundle_id` = SHA-256 of the manifest.

Runtime stack: React/TypeScript + Vite static build; Arrow files read into
typed arrays; Web Workers for fp16 cosine/subspace math (subspace claims
verified in native dimension, never from projections); D3/Observable Plot;
Service Worker cache. No DuckDB, no backend, no model call — ever.

---

## 7. One-Time Builder

```text
configs/nano_viz/offline_observatory.yaml    corpus, grid spec, budgets
observatory/build_corpus.py                  P0 (CPU)
observatory/run_model_batches.py             P1–P3 (GPU, phase-aware, resumable)
observatory/compute_geometry.py              P4a (CPU)
observatory/compute_interventions.py         P4b (CPU: Shapley, verdicts, aggregates)
observatory/build_bundle.py                  P5a (CPU)
observatory/verify_bundle.py                 P5b (CPU, fail-closed)
```

The builder is a dedicated Python package. Run components as
`python -m observatory.<module>`; the old `scripts/nano_viz_*.py` paths are
compatibility launchers only. Tests live under `tests/observatory/`, while
generated evidence remains outside Git under the configured evidence root.

| Phase | Resident (1× H200) | Work |
|---|---|---|
| P0 | — | Row selection; film-set assembly (+ min-context check); text-variant generation (swaps/paraphrases/corruptions/truncations, CPU); PII pre-scan |
| P1 | AV (+BASE for film-set extraction first) | Film-set extraction (one forward per document), per-position descriptions with jitter repeats, k=8 tellings — ~1k generations |
| P2 | AR, then IND-AR | Encode all lattice texts (~10k forwards); independent-AR pass on Court cells |
| P3 | BASE + AR (fit together: ~98 GB) | Identity pass at the functional evaluator's tolerances, then BEHAVIOR cells: patched forwards, wake traces (one teacher-forced forward each), greedy continuations |
| P4 | CPU | PCA bases (validation-fitted), projections, neighbors, retrieval, Court verdicts (validation-fitted thresholds), family-bootstrap aggregates |
| P5 | CPU | Bundle assembly; verifier |

Serial budget: **≈ half an H200-day** (planning estimate; dominated by ~1k
AV generations and ~10k short AR forwards). Then copy the bundle
local + S3, record the manifest hash, delete the GPU workspace.

Provenance binding on every output (unchanged from v1): dataset/row
identity; the qualified checkpoint fingerprints (AV DCP sha, AR HF sha,
IND-AR sha, BASE sha); generation-protocol hash; exact intervention spec;
fitted-on split for every threshold; source report SHA-256s for inherited
numbers; claim-scope enum (`stored_snapshot | fresh_forward_exploratory`);
code revision + runtime fingerprint.

Verifier (fail-closed): schema + manifest hash completeness; metric
recomputation spot checks from shipped vectors; aggregates re-derived and
diffed; control-group completeness (no semantic cell without its placebo
partners); population rules (no FROZEN rows in authoritative aggregates or
lattice); PII scan over all shipped text; source text only for cleared
rows + film set; σ-permalink round-trip test.

---

## 8. Milestones

**Milestone A — zero GPU (existing caches only).** Population pack;
CHANNEL: waterfall, capacity ladder, twin critics, spectral strip; AUDIT:
claim ledger, almanac, magnitude card, drift card, provenance browser;
Pictionary. BENCH renders the five FUNC variants as selectable "cells"
with patch-position gauges; TRACE is an explained placeholder. Already a
publishable exhibit of the qualified result.

**Milestone B — one H200 afternoon.** P0–P5 fill the lattice, film set,
tellings, Court evidence, and behavior tables. TRACE and BENCH light up
fully; the Court gets its paraphrase/corruption legs; forecast-before-
reveal becomes possible.

Logistics: the implementation agent gets exclusive new paths
(`configs/nano_viz/`, `scripts/nano_viz_*`, `apps/observatory/`) to avoid
the dirty worktree, and builds frontend + compiler + verifier against a
fixture bundle so Milestone A never waits on GPU time.

---

## 9. Descoped Register (v1 → v2)

Dropped, with the one-line reason; all remain specified in
`nla_visualization_frontier_ideas.md` and v1 of this document (git
history) for future bundle versions:

| Item | Reason |
|---|---|
| External Prompt Atlas (BLiMP/GSM8K/HumanEval/FLORES) + OOD compass + Babel + Mediation Cascade (C19) | Largest cost center: importers, licensing (CC-BY-SA), carrier prefixes, compass fitting — none needed to exhibit the qualified result |
| Semantic Telephone (C3), Interpolation Alley (C2) | Striking but moderate science-per-effort; each needs its own AV/AR loop batch and views |
| Transplant matrix (C17), behavioral norm dial (C6 pane 3) | Causal story already carried by BENCH lanes + wake; sweeps cut |
| Speak-the-blind-spot verbalizations (C1) | Needs an AV batch on synthetic vectors; the L0 spectral strip stays |
| Lump & split (C4), seven ghosts (companion §0.5), upsets (C22), redundancy as a CHANNEL view (C11) | Secondary corpus views; corruption data survives inside the Court |
| Rosetta/logit-lens (C24), Writable Set (C5), state-transition grammar (C14) | Optional shards for a future bundle version |
| Next-token-lineup game, human-masking Court leg (C20), curated write-only gallery (C18) | Depend on extra content or human data not yet collected |
| Deep-dive coverage of the test-half panel | FROZEN population is display-only; keeps the exploratory boundary simple |

The bundle is versioned precisely so any of these can return as an
additive shard through the same builder + verifier, without touching the
app's core.
