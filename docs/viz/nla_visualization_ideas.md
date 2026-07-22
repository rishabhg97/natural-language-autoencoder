# NLA Visualization Ideas — What Only a Language-Coded Autoencoder Can Show

Status: design proposal, `2026-07-16`. Companion to the publication drafts.
(An earlier single-file dashboard prototype was built and then deleted on
request the same day; references below to "the existing dashboard" describe
panels from that prototype that any future implementation can re-create from
the same artifacts.) Everything here is
**exploratory-framing only** — nothing below creates a confirmatory claim, and
nothing may consume the sealed corpus for new confirmatory purposes.

## Why NLA visualization is its own genre

Three properties distinguish an NLA from every other interpretability object,
and every good visualization below exploits at least one:

1. **The code is readable.** The latent is English — it can be diffed,
   sectioned, searched, and judged by a human.
2. **The code is writable.** You can *edit* the latent with a keyboard and
   push it back through the decoder — no SAE, probe, or steering vector
   offers a human-writable interface to activation space.
3. **The loop closes in behavior.** Reinjection turns reconstruction quality
   into next-token distributions — fidelity becomes something the model
   *does*, not a number.

## Fresh empirical anchors (computed this session, offline, from the local vector caches)

The frozen prediction caches contain full 2,688-d vectors (targets, all seven
reconstruction variants, both splits, both critics). Five one-pass findings
that several visualizations below are built around — each reproducible in
seconds from `PUB/*_roundtrip_predictions.npz`:

| Finding | Value | Why it matters |
|---|---|---|
| **Retrieval identifiability** | AV text retrieves its own `h` among 512 at **94.3% top-1** (teacher text 92.4%; controls ~0.3%; median rank 1) | An intuitive, headline-able restatement of dMSE 0.307 — the explanation *individuates* the activation, and machine-written code slightly beats the human-style summary |
| **Twin-critic determinism** | cos(ĥ_primary, ĥ_independent) = **0.993** mean; both sit at ~0.847 from truth; critics agree more with each other than with truth on **100% of rows** | Text→vector decoding is nearly a function ⇒ **the loss is committed at the vector→text step**. The strongest "loss is in the language" argument available |
| **Spectral low-pass** | Residual energy in top-10 target PCs: 8.8% vs 17.4% of target variance there | Language preserves the dominant axes of variation and drops the idiosyncratic tail — a channel allocating bits by variance |
| **Round-trip contraction** | Pairwise Shepard: corr 0.53, but dissimilar pairs (cos<0.3) reconstruct at cos ≈ 0.56 (**+0.28 bias**) | The bottleneck drags everything toward a shared "explanation-speak" direction — semantic quantization toward the language prior |
| **Student beats teacher** | AV text reconstructs better than teacher text on **234/512 rows** (margins up to 0.56) | The AV writes a purpose-built *code*, not a summary — upset rows are readable |

Supporting anchors: within-family target cos 0.53 vs cross-family 0.44;
explanation length vs dMSE corr −0.108 (0.321→0.282 across word-count
quartiles — mild under SFT, exploded to 0.34–0.49 in the invalidated RL run);
fitted magnitude scalar 0.5606 ≈ mean-cos/mean-overshoot (0.847/1.533 = 0.552,
within ~1.5%) — the "magnitude failure" is approximately the Stein-optimal
shrinkage forced by angular uncertainty, i.e. English is a directional code
*by necessity*, not by accident.

---

## Tier 0 — pure offline, buildable today into the existing dashboard

No GPU, no new model calls; all data already on this machine. Ordered by
insight-per-effort.

### 0.1 Twin Critics, One Dialect *(the replication visual that doubles as a mechanism claim)*
Per-row triangle glyphs of (h, ĥ_primary, ĥ_indep) — 512 needle-thin isoceles
triangles that all look identical, which *is* the finding. Upgrade beyond the
agreement scalar: plot **cos of the two critics' error vectors**
(e_A = ĥ_A − h vs e_B = ĥ_B − h). Shared errors ⇒ the mistake is in the text,
not the reader. Click any glyph → read the sentence both critics decoded
identically. *(geometry #1, comparative #4)*

### 0.2 How Many Thoughts Can a Paragraph Hold? *(retrieval capacity curve + legible confusions)*
Top-1 retrieval vs gallery size N = 2…512 (bootstrap bands; teacher overlay;
controls flat at chance) with a bits axis (log₂N) — a Fano-style lower bound
on the bits an explanation carries (≥9 bits already banked at 94.3%/512).
Then the ~6% failures as a confusion graph colored same-doc / same-family /
stranger — **every confusion is readable**: show the two explanations and
highlight what English failed to distinguish. *(geometry #2, #6; comparative #6)*

### 0.3 English Is a Low-Pass Filter *(the transfer spectrum)*
Per-PC of the target distribution: target variance, transmitted variance,
residual energy → a transmission-coefficient strip. Hover a PC → the 5
highest-loading rows' passages + explanations so the viewer can *name the
axis in words*. Toggle primary/independent critic (curves should overlay —
echoes 0.1). *(geometry #3)*

### 0.4 The Information-Ceiling Waterfall *(“where the bits die”)*
2.0 (random) → controls 0.67–0.98 → **the language shelf** (source-context
0.301 / teacher 0.305 / AV 0.307 within a hair) → source_raw 0.083. Each rung
annotated with what changed; hover → example texts. Optional unit toggle to
retrieval accuracy (94.3% → 0.3%) showing the ordering is metric-independent.
The one chart that makes "the loss is the price of English, not an AV/AR
failure" self-evident. *(comparative #1)*

### 0.5 The Seven Ghosts *(prior-escape scatter)*
All seven variants decomposed into (alignment with train-mean direction) ×
(mean-orthogonal alignment with the row's truth). Controls pile on the prior
axis — the geometry of the critic saying "I don't know" — while AV/teacher
clouds escape orthogonally. Click a control point → read the AV fluently
confabulating from a zero vector. *(geometry #5)*

### 0.6 The Upset Bracket *(where the machine code beats the human summary)*
Teacher-vs-AV per-row scatter with the y=x diagonal; 234/512 upsets
highlighted; click → side-by-side texts with the AV-only content flagged
(typically final-token/register machinery a faithful summary omits).
*(comparative #2)*

### 0.7 The Confabulation Prior *(population-level control-text mining)*
From 1,024 × 5 control explanations: the AV's *null text* (what it writes
with no signal), per-token log-odds real-vs-zero (informative words glow,
boilerplate fades), hedging rates, and the near-template collapse of
zero/mean/shuffled texts. Doubles as the null model that makes saliency maps
(Tier 1) interpretable. Blind-lineup game mode: guess which of six texts had
the real vector. *(language #4, comparative #3)*

### 0.8 Quick wins
- **Survival of the Top Token**: Kaplan-Meier-style curves of P(original
  top-1 survives at rank ≤ k) per condition from the cached per-row
  `original_top1_rank` — a decision-level fidelity statistic. *(causal #9)*
- **Shepard contraction plot**: pairwise cos before vs after the round trip;
  the +0.28 contraction of dissimilar pairs made visible. *(new, this session)*
- **The Shrinkage Cone**: per-row optimal scalar vs cos with the s* ≈
  cos·(‖h‖/‖ĥ‖) construction — magnitude failure as a theorem-shaped picture.
  *(geometry #7, comparative #9)*

---

## Tier 1 — one-off GPU precompute batches (run once on RunAI, ship as static JSON)

Each batch is minutes-to-an-hour on 1–2 GPUs; results become permanent
offline dashboard panels. Grouped by shared harness.

### Batch A — AR-only (1× GPU, 38 GB; hundreds–thousands of sub-second forwards)
1. **Section Shapley / Section Scalpel** — ablate all 2⁴ subsets of the four
   explanation sections, exact Shapley per section for dMSE (and per-PC-band
   recovery): *which sentence carries which subspace of the thought*. The
   most NLA-native experiment proposed by four of five lenses independently.
   ~16 forwards/row × 100–512 rows. *(language #2, causal #5, comparative #7, geometry #8)*
2. **Token Bounty** — leave-one-word-out occlusion saliency *on the code
   itself*: each explanation word tinted by Δcos when deleted. ~60–150
   forwards/row; precompute the 100 passage-resolved rows. *(language #1)*
3. **How Few Words** — truncate explanations at k words → rate–distortion
   curves where the rate axis is literally *words spent*; compare AV vs
   teacher word-efficiency. *(language #3)*
4. **Paraphrase Stress Test / Steganography Detector** — re-encode
   synonym-swapped / reordered / register-rewritten explanations: does cos
   degrade smoothly with semantic distance (honest English channel) or cliff
   on trivial rewording (private AV↔AR cipher)? **The single most important
   pre-RL integrity check**, and a direct test of the paper's central
   metaphor. ~6 perturbations × 50–100 rows. *(language #6, comparative #5)*
5. **Scaffold vs Payload** — encode template-skeleton-only vs
   content-words-only: the effective code length, and whether the scaffold
   carries the top-PC mass while payload carries the tail. *(language #7)*

### Batch B — reinjection harness (base model + AR, 2× GPU; ~1–2k patched forwards)
6. **Wrong-Brain Transplant** — cross-patch ĥ_j into row i's forward: a
   12×12 KL heatmap graded by semantic proximity (own row ≪ same family ≪
   stranger ≈ shuffled control). The behavioral analogue of retrieval.
   *(causal #2)*
7. **The Norm Dial** — sweep injection norm of unit(ĥ) 0→2×true: does the
   behavioral basin sit at the oracle norm, the raw 1.53× overshoot, or the
   0.5606-corrected norm? **Turns the oracle-norm caveat (open issue A.12)
   into the dashboard's most honest chart.** *(causal #4)*
8. **Next-Token Lineup** — six anonymized top-10 token columns (truth, AV,
   teacher, mean, zero, shuffled); the viewer guesses which is the real
   activation; running scoreboard. Fidelity as a participatory experience.
   *(causal #6)*
9. **Two Futures + Where the Patch Bites** — greedy continuation divergence
   trees per condition, plus teacher-forced per-position KL along the true
   continuation → a **behavioral half-life** statistic per condition (AV
   cools within N tokens; controls stay hot). *(causal #3, #7)*

### Batch C — AV-required showpieces (base + AV + AR; minutes per item)
10. **The Volume Knob** — injection-scale sweep 0→300 with the faithful
    window highlighted. **Directly addresses open issue A.10 (scale 75 never
    re-derived for R33)** — a methods gap closed by a visualization.
    *(live #4)*
11. **Residual-Stream Cinema / Explanation Flipbook** — extract h at 10–15
    positions of a document, verbalize each, render as a persistent-text
    diff filmstrip per section: Genre freezes early (slow feature),
    Final-token churns every step (fast), Discourse updates at clause
    boundaries — **the timescale structure of R33 read through its own code's
    edit history**. Precompute 5–10 documents. *(live #6, language #8)*
12. **Interpolation Alley** — slerp h₁→h₂ (within/cross family), verbalize
    every waypoint: does the text morph smoothly or snap at a boundary
    (chart vs symbol lattice)? Displacement arrows show the code acting as a
    denoiser with attractors. *(geometry #9, live #8-adjacent)*
13. **Semantic Telephone** — iterate h→z→ĥ→z′→…: contraction rate,
    fixed-point texts ("stable self-descriptions"), drift toward the prior
    axis. Watch detail evaporate from the text step by step — the visible
    sublimation of the tail spectrum. *(live #3, geometry #10)*
14. **Speaking the Blind Spot** — PCA the residuals, then inject
    (mean + α·residual-PC) into the AV and read what the model says about
    the directions language systematically fails to transmit. *(geometry #4)*

---

## Tier 2 — the live "NLA Playground" (GPU-resident, genuinely interactive)

The interactions that cannot be precomputed, ranked by (uniqueness ×
snappiness). All carry a fixed "exploratory demo — not a confirmatory
result" banner.

### 2.1 Explanation Surgery — *edit the English, steer the model* ⭐ flagship
Pick a row (or output of 2.3); its explanation appears as editable text.
Change one clause — "formal legal filing" → "casual food blog" — and in
~1–2 s: AR encodes the edit, the vector is reinjected at the capture
position, and the next-token distribution morphs beside the original, with
a Δcos compass showing the vector slide across the cached PC map. Preset
edit chips (flip register / change topic noun / negate final-token
constraint) for guided use. **This is the property no other interpretability
method has — a keyboard-writable latent — and it is also the cheapest live
loop (2 forwards, no AV needed).** *(live #2, language #5, causal #1)*

### 2.2 Beat the Autoencoder — *human vs AV, scored by the frozen critic*
The viewer reads a passage and writes their own explanation into the
four-section template; one AR forward scores it against h on a leaderboard
vs AV (0.307) and teacher (0.305), with per-section ablation showing which
of their sentences carried the score. Players learn *what the critic
actually reads* by playing. Single 38 GB model, <3 s per submission —
deployable on one GPU alone. *(live #7)*

### 2.3 Explain My Prompt / Read My Mind — *the full loop on your text*
Type anything → animated rail: text → h chip (norm + nearest cached-family
neighbor) → explanation typing out → ĥ chip with live cos → twin
continuation panes (truth-state vs reconstruction-state) with divergence
highlighting. Optional "second opinion" runs the independent critic and
shows cross-critic cos live. ~10–25 s/interaction; the demo people
screenshot. *(live #1, causal #8)*

### 2.4 One Vector, Many Tellings — *temperature fan*
k sampled explanations from one h; AR-encode all; show the fan of ĥ's on
the cached PC map. Tight-but-displaced fan (predicted by twin-critic
geometry) ⇒ the gap is information genuinely absent from *any* English
telling; a straddling fan ⇒ averaging tellings should beat 0.307 — testable
live with one button. A real experiment disguised as a demo. *(live #5)*

### 2.5 Rosetta Bottleneck & Vector Algebra *(stretch panels)*
Cross-lingual invariance (h from a passage vs its translation, against the
0.53/0.44 family bands; does the English explanation track *French* syntax
in its final-token section?) and h(A)−h(B)+h(C) analogies verbalized.
Spectacular if they work; failures are informative and must be shown too.
*(live #8, #9)*

### Also live: the **Goodhart Sentinel** *(RL instrumentation, not a demo)*
Before any future RL run: four aligned needles — primary-critic reward,
independent-critic dMSE, reinjection KL, text-health (parse rate,
section-length entropy, repetition, length drift vs the SFT corr −0.108
baseline) — with the SFT operating point as a reference band and the
divergence wedge labeled as the hacking region. The twin-critic result
(0.993) is a loaded gun for RL: a near-deterministic critic is a
near-perfectly exploitable reward. This panel operationalizes the July-8
invalidation as instrumentation. *(comparative #8)*

---

## Serving architecture (RunAI)

**Placement.** Three checkpoints: base Nano30B (extraction + reinjection,
~60 GB bf16), AV (~60 GB), AR (~38 GB) ≈ 158 GB total.
- Comfortable: 2× H200 (141 GB each) or 4× H100-NVL.
- Minimum: 1× H100 (AR only) already serves 2.1-lite (surgery without
  reinjection → Δcos only), 2.2, and all of Batch A.
- The AV is only needed for 2.3/2.4/2.5 and Batch C.

**Service.** One FastAPI process wrapping the existing repo primitives (the
same code paths the frozen evals used — no new modeling code):
`/extract {text, position} → h`, `/verbalize {vector} → z` (greedy, scale 75,
protocol-hashed), `/encode {text} → ĥ`, `/reinject {vector, row|prefix} →
top-k logits`, `/ablate {text, spans[]} → Δcos per span` (batched). Simple
FIFO queue, one request per model at a time; every request logged to a JSONL
audit file. The dashboard gains a "Live" tab that talks to
`localhost:<port>` via `runai port-forward` — the static file keeps working
unchanged when the service is down (feature-detect and grey out).

**Guardrails.** Frozen generation protocol only; length caps on user text;
the sealed-corpus rows never used as new confirmatory evidence; every live
panel carries the exploratory banner; no user text persisted beyond the
audit log; offline W&B stays offline.

**Suggested build order.**
1. Tier 0 (0.1–0.8) into the static dashboard — a few days, zero GPU.
2. One RunAI session for Batch A + B precomputes (~2–3 h GPU-time total) —
   ships Section Shapley, Token Bounty, Norm Dial, Lineup, Transplant,
   Paraphrase grid as permanent offline panels.
3. Live service with AR-only (2.2 + 2.1-lite) — one GPU, lowest risk.
4. Full playground (2.1 + 2.3) on 2× H200 — the flagship.
5. Batch C showpieces (Volume Knob, Cinema, Alley, Telephone) as idle-time
   precomputes; 2.4/2.5 as stretch.

## What each layer buys scientifically

| Question the project currently can't answer | Visualization that answers it |
|---|---|
| Is the code *semantic* or a private AV↔AR cipher? | Paraphrase Stress Test (A.4) |
| Which sentence of the code carries which subspace? | Section Shapley (A.1) + Token Bounty (A.2) |
| Is the oracle-norm caveat behaviorally load-bearing? | Norm Dial (B.7) |
| Was injection scale 75 right for R33? (open issue A.10) | Volume Knob (C.10) |
| How many bits does an explanation carry? | Capacity curve (0.2) |
| Where in the spectrum does 0.307 live? | Low-pass transfer (0.3) + Blind Spot (C.14) |
| Which arrow of the round trip is lossy? | Twin Critics (0.1) |
| Is the latent writable/steerable in practice? | Explanation Surgery (2.1) |
| What are R33's fast vs slow features? | Flipbook/Cinema (C.11) |
| How would we catch RL reward-hacking early? | Goodhart Sentinel |
