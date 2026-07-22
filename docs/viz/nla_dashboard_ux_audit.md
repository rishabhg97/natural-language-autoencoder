# NLA Observatory Dashboard — UX & Information-Design Audit

Status: audit report, `2026-07-18`. Auditor stance: skeptical first-time reader
plus information-design review. Subject: `dashboard/nla-observatory` served at
`http://localhost:5199` (commit state of this working tree).

**Method.** Read `README.md`, `docs/viz/nla_offline_observatory_design.md`,
`docs/viz/nla_visualization_frontier_ideas.md`, `NlaPrimer.tsx`,
`panelGuides.ts`, and every station component; then drove the live app with
headless Chromium (the repo's own Playwright toolchain) at the project's three
e2e viewports — desktop 1440×900, tablet 1024×768, mobile 390×844 — in light
and dark modes. Every station was exercised interactively: the example-browser
dialog, critic switching, rewrite families and chips, TRACE token selection in
both film-set and poetry modes, all eight poetry cases, BENCH chips/doses/lane
focus/teacher view/metric-only rows, and AUDIT deep links, docket, and
provenance. Claims below that depend on data (not pixels) were verified
directly against the shipped shards in `public/data/`. Where a finding
references an image, it is in
[`nla_dashboard_ux_audit_assets/`](nla_dashboard_ux_audit_assets/). No code
was changed by this audit.

Severity scale: **critical** (a reader is actively misled about the science) ·
**high** (the main comparison or core linkage fails for a first-time reader) ·
**medium** (slows or muddies understanding; workaround exists) · **low**
(polish). Each finding states whether the proposed change affects scientific
meaning (i.e., changes what the dashboard claims or evidences) or is
presentation-only.

---

## 1. Executive verdict

This is one of the most scientifically honest dashboards I have reviewed. The
evidence-scope system (qualified / exploratory / negative / unavailable), the
controls-as-furniture discipline, the first-class negative results, and the
provenance depth are all real and consistently executed — the design doc's
promises about boundaries are kept in the UI, not just stated. The TRACE
token-linked readers are the strongest exhibits: original text, selected
token, decoded output, and controls form one physically connected chain.

The dashboard's weaknesses are almost all *translation* failures, not honesty
failures. It speaks builder-language at readers: raw variant codes
(`av_zero`, `mean` vs `av_mean`, `usable_closed`, report ids `e0`/`e3`/`p2`),
unexplained metrics ("anchor lift", "expected cos", "identity_cosine"), and
verdict framings (`VERDICT: TRUE` in green on a corruption cell) that a
first-time reader will decode incorrectly about half the time. There is one
genuine evidence-display contradiction (poetry term underlining vs. shard
score flags, F1) that a skeptic will find within minutes and that undermines
trust in the flagship exploratory exhibit; it should be fixed before anyone
external sees this. There is also one chart whose visual message actively
fights its own data table (the BENCH reconstruction compass, F4).

Nothing found rises to "the dashboard overclaims." The residual risks run the
other way: over-hedged, over-dense panels in which the load-bearing comparison
is present but not visually dominant. Roughly ten targeted fixes — half of
them copy — would move this from "defensible" to "genuinely readable by a
first-time skeptic in fifteen minutes."

Verdict: **strong scientific skeleton, publishable after a copy-and-encoding
pass**; two findings (F1, F4) should be treated as blockers for external
sharing, and neither requires new GPU work.

---

## 2. What the dashboard communicates successfully

- **What an NLA is, in five seconds.** The primer's three-node flow
  (`Activation h → AV reads → Learned description z → AR reconstructs →
  h_hat`) sits above the fold on every station, with the single most
  important boundary sentence — "an encoding to test, not a transcript of
  private chain-of-thought" — directly under the heading
  ([01-desktop-fold.png](nla_dashboard_ux_audit_assets/01-desktop-fold.png)).
- **Bounded answers, not vibes.** Every station opens with a guiding question,
  a one-paragraph bounded answer, a scope badge, and three headline metrics
  with "lower/higher is better" hints. The CHANNEL brief explicitly lists what
  the result does *not* establish (transcripts, fresh-forward fidelity, causal
  control).
- **Evidence-status vocabulary that actually binds.** qualified / exploratory
  / negative / unavailable badges appear where they matter and are defined in
  a legend on AUDIT. TRACE wears fresh-forward exploratory badges on every
  panel; BENCH separates its stored-snapshot metrics from validation-only
  functional readouts.
- **Controls cannot be dismissed.** The BENCH rack renders edit + paraphrase
  placebo + random edit + identity as one unit; the poetry lens ships
  shuffled-activation samples beside real ones and a norm-matched random
  direction beside every steering dose; dose 0 is present as a
  patch-fidelity control. This matches the design doc's "control groups are a
  property of the data layout, not UI discipline."
- **Negative results are first-class.** "Negative & weak results" sits beside
  the claim ledger; the poetry all-cases panel keeps zero lifts, missing
  onsets, baseline misses, and the failed causal edit on the board, with an
  aggregate footer that says "SIGNAL: WEAK" and "no causal edit produced an
  alternate-family rhyme at any dose."
- **Absence is visible.** Metric-only BENCH rows render explicit unavailable
  boxes per consequence pane; the unaligned TRACE document refuses to draw
  guessed highlights and says why; missing identity metrics and missing
  aggregates each produce a labeled unavailable state.
- **Source→output linkage in TRACE is exemplary.** Tokens are selected inside
  the full original document; the right pane names the token, its position,
  parse state, and the caveat line; the poetry variant adds future-token
  dimming and a held-out-line box labeled "never shown at any analyzed
  position"
  ([09-trace-reader-good-pattern.png](nla_dashboard_ux_audit_assets/09-trace-reader-good-pattern.png)).
- **Provenance is deep and honest.** Click-to-copy hashes, report bindings
  that open the exact shipped JSON, runtime fingerprints, excluded files with
  reasons, and a privacy card that displays "HUMAN REVIEW REQUIRED" and states
  what the automatic triage cannot establish.
- **Charts have text alternatives.** All chart SVGs carry `role="img"` with
  data-bearing `aria-label`s, and nearly every chart has a table twin
  (collapsed `details`). Focus outlines, `prefers-reduced-motion` handling,
  and `:focus-visible` styles exist. The repo ships axe checks and viewport
  screenshots.
- **The console is clean** across all four stations (no errors or warnings
  beyond Vite/devtools info), and fail-closed error boxes replace silent
  substitution when a shard is absent.

---

## 3. Ten highest-priority problems (ordered by user impact)

**F1 — Poetry term-underlining contradicts the plotted score. — HIGH
(evidence-display integrity)**
Station/panel: TRACE · Token-linked poetry NLA reader + Planning-onset curve.
Evidence: for `carrot-rabbit` at the anchor (offset 0), real AV sample 1's
text contains "rabbit" and the UI underlines it; the footnote says
"Underlined words triggered the target or alternate rhyme-family score." But
the shard has `target_family=false` for all four anchor samples (verified in
`poetry.json`), so the onset curve shows rate 0.00 at offset 0 — and the case
card shows "anchor lift 0.000"
([04-poetry-anchor-underline-vs-score.png](nla_dashboard_ux_audit_assets/04-poetry-anchor-underline-vs-score.png),
[05-onset-curve-control-occlusion.png](nla_dashboard_ux_audit_assets/05-onset-curve-control-occlusion.png)).
Why it impedes understanding: the reader's eyes see a target-term hit exactly
where the chart says there is none, on the flagship case; a skeptic concludes
either the plot or the highlighting is wrong, and trust in the whole poetry
exhibit drops. (`HighlightTerms` does display-side substring/segment matching
via `segmentByTerms`, while chips/curves follow the shard's precomputed
flags; the two matchers disagree, likely over possessive "rabbit's" or a
scoring seal/section rule.)
Proposed change: render underlines from shard-computed hit spans (ship
`matched_spans` per sample, or at minimum per-sample boolean-consistent
highlighting), and reword the footnote to state the authoritative source of
the score. If display matching is kept, label it "display-level string match —
chips carry the scored result" and make the chips render on every sample
(including "no scored hit").
Affects scientific meaning: **yes** — it corrects what the exhibit presents as
scoring evidence.

**F2 — CHANNEL's row reader never shows which token the activation came
from. — HIGH (broken source→signal anchor)**
Station/panel: CHANNEL · "Selected row: source → NLA description".
Evidence: pane 1 shows the full source; "activation position 126 of 127 raw
tokens" appears only as metadata below; no token is highlighted in the text
([02-row-reader-no-token-anchor.png](nla_dashboard_ux_audit_assets/02-row-reader-no-token-anchor.png)).
The learned description then talks about 'the phrasal verb "sits down"' and
the reader has to infer that the activation sits at the final token.
Why: this is the dashboard's first and most-used exhibit; the design's own
"visual chain from original text to highlighted token" exists in TRACE but
not here, so the single most important referent — *which activation?* — is
implicit exactly where first-time comprehension is formed.
Proposed change: highlight the activation token in the source text (same
inline-token treatment as TRACE, non-interactive), and move "position 126 of
127" into a caption tied to that highlight. `rows.json` already carries
`token_position`/`n_raw_tokens`; if character offsets are missing for exact
alignment, ship them from the builder as TRACE does.
Affects scientific meaning: no (presentation; strengthens correct reading).

**F3 — Raw builder vocabulary is load-bearing all over the UI. — HIGH
(terminology)**
Station/panel: cross-station; worst on CHANNEL waterfall and real-vs-control.
Evidence: bar labels `av_zero, av_shuffled, av_none, av_mean, mean, av_real,
teacher` with no on-panel meanings — nothing explains that `av_none` means
"description generated with no activation input" or how `mean` differs from
`av_mean`
([03-waterfall-raw-variant-names.png](nla_dashboard_ux_audit_assets/03-waterfall-raw-variant-names.png));
parse chips read `usable_closed`; twin-critics summary rows are `e3`/`p2`
with cell counts (1118/104/7434/650) and no note on why they differ; the
rewrite panel prints `spec: {"kind":"bullet_sections",…}` inline; Shapley is
annotated `utility = one_minus_directional_mse`.
Why: the guides teach *how to read the bars* but the reader still cannot name
*what each bar is*; the key comparison ("real description vs. deliberately
uninformative inputs") stays locked behind a naming scheme only the builder
knows.
Proposed change: one-line plain-language sublabels per variant (waterfall
rows and the real-vs-control table can take a second muted line each:
"av_zero — description generated from a zeroed activation"), humanize parse
chips ("parsed cleanly"), caption the e3/p2 rows ("e3 = rewrite-grid report;
p2 = full lattice report — cell counts differ by design"), and demote spec
JSON/cell ids to a `details`/hash-chip.
Affects scientific meaning: no (labels only; keep raw keys visible as
secondary mono text for provenance).

**F4 — The reconstruction compass visually contradicts its own metrics
table. — HIGH (chart integrity)**
Station/panel: BENCH · Reconstruction compass.
Evidence: for the default behavior row, the stored-target crosshair sits at
PC1 ≈ −25 while all four lane points cluster at PC1 ≈ +28; the picture reads
"reconstructions nowhere near target" while the table beneath reports cosine
≈ 0.85 and the panel's own How-to-read says to trust the native metrics
([06-compass-target-far-from-cluster.png](nla_dashboard_ux_audit_assets/06-compass-target-far-from-cluster.png)).
Why: a caveat cannot beat a picture this strong. First-time readers take away
"the reconstruction fails" on the very station whose point is calibrated
comparison; the projection likely mixes magnitude (norm ratio ≈ 1.6) into a
display that the qualified claim explicitly excludes.
Proposed change: decide what the compass is *for*. If it is directional
geometry, project L2-normalized vectors (or plot angle-preserving
coordinates) so display distance tracks the directional claim; if raw
geometry must stay, draw the norm-scaled ghost of the target and label the
axis "unnormalized PCA — includes magnitude error." Verify in the builder
which vectors feed the PCA before relabeling.
Affects scientific meaning: **yes** (the display currently implies a finding
— gross reconstruction failure — that the qualified metrics do not support;
fixing it aligns display with evidence).

**F5 — Two of the four BENCH lanes are the same color. — HIGH (encoding)**
Station/panel: BENCH · rack keys, divergence gauges, causal wake, dose
comparison, compass, top-k strip.
Evidence: `random_edit` uses `--series-4` `#526a83`, identity uses `--ink-3`
`#596875`. The dataviz palette validator fails the set: worst adjacent pair
ΔE 7.3 (protan) / 7.6 (normal) and both fail the chroma floor ("reads
gray"); on the gauges the third and fourth bars are indistinguishable
([07-gauges-lane-color-collision.png](nla_dashboard_ux_audit_assets/07-gauges-lane-color-collision.png)).
Why: the entire BENCH argument is "edit must separate from *three* controls";
when two controls merge visually, every cross-lane chart quietly becomes a
three-lane chart and the reader cannot attribute the gray line/bars.
Proposed change: give `random_edit` a distinct hue with adequate chroma
(validate the four-lane palette with the six checks; e.g., keep edit blue,
placebo purple, move random to the orange `--series-3` family, keep identity
as the gray *reference*), and where identity is conceptually a floor, prefer
drawing it as a reference line/band rather than a fourth bar. Wake/dose line
charts already have end labels — keep those; add them to gauges rows.
Affects scientific meaning: no (encoding only).

**F6 — Verdict badges read as pass/fail while the healthy outcome flips per
family. — HIGH (scientific communication)**
Station/panel: CHANNEL · Rewrite explorer (Semanticity court) and AUDIT ·
Cipher court docket.
Evidence: a `delete 10%` corruption cell shows a green `VERDICT: TRUE`
([08-corruption-verdict-framing.png](nla_dashboard_ux_audit_assets/08-corruption-verdict-framing.png));
in the docket, heavy-corruption cells show orange ✕ glyphs. Green-true means
"code survived," which for corruption ranges from ambiguous to *bad* (a
cipher-like code also survives); orange-✕ ("code broken") is the *expected
healthy* outcome for the negative-calibration cells. The legend explains
mechanics, not expectation.
Why: readers reflexively map green/✓ to "good." Half the docket's glyphs
invert that mapping, so the panel's population story (paraphrases preserve,
heavy corruption breaks) is invisible unless the reader re-derives the
calibration logic.
Proposed change: reframe per-cell outcomes as "code preserved / code broken"
in neutral colors, and add an expected-vs-observed treatment (e.g., a subtle
"as expected" tick when a paraphrase preserves or a heavy corruption breaks,
and a highlighted "unexpected" state for the reverse). Keep raw verdict
booleans in tooltips for fidelity.
Affects scientific meaning: **no change to values**, but materially changes
(corrects) the reading of the court's evidence; review wording with the
measurement contract.

**F7 — Anchors scroll under the sticky header(s). — MEDIUM-HIGH
(navigation)**
Station/panel: shell; all deep links (`AuditLink` chips, TRACE "claim in
AUDIT," CHANNEL `view=` anchors, AUDIT claim flash).
Evidence: panels have `scroll-margin-top: 0px` under a 61px sticky header
(measured live); on BENCH a second sticky banner stacks below it. Every
`scrollIntoView({block:"start"})` puts the panel title and badges underneath
the chrome — e.g., landing on the flashed claim card with the ledger header
hidden.
Why: deep links are the dashboard's citation mechanism ("permalink =
citation"); each jump that hides its own title costs the reader orientation
right at the hand-off moment.
Proposed change: `scroll-margin-top: calc(var(--header-h) + [banner] + 12px)`
on panel/claim anchors (the BENCH `--bench-header-h` var already exists);
verify the AUDIT flash target is fully visible after scroll.
Affects scientific meaning: no.

**F8 — The tellings "family mean" line is a population aggregate that reads
as this row's mean. — MEDIUM-HIGH (statistical ambiguity)**
Station/panel: CHANNEL · Alternate-tellings fan.
Evidence: for `v-248030`, the eight sampled tellings span dMSE 0.151–0.277
(mean ≈ 0.20) while the orange "family aggregate mean 0.286" line sits to the
right of *all but one* dot — because it is the panel-wide
`primary.alternate_telling.directional_mse` mean across all 50 rows, not the
mean of the plotted dots.
Why: a line labeled "family … mean" positioned inside a strip of eight dots
will be read as their mean; since it visibly isn't, careful readers suspect a
bug and casual readers mis-anchor the row's quality.
Proposed change: relabel to "all-rows mean (alternate tellings, 50-row
panel)" and/or also draw this row's own mean tick; consider moving the
population reference into the aggregate strip below the chart.
Affects scientific meaning: no values change; the correction prevents a
mis-read statistic.

**F9 — "AV loss" vs "AR loss": one panel says both. — MEDIUM-HIGH
(terminology / metric identity)**
Station/panel: CHANNEL · Real-vs-control AV loss.
Evidence: the title and the guide's "What this shows" say the *AV model*
predicts explanation tokens more easily given the correct activation; the
evidence note and the x-axis say "AR loss conditioned on the AV text." The
dashboard glossary defines AR as the reconstruction critic, which makes the
axis reading nonsensical (the critic outputs vectors, not token NLL).
Why: the panel is the fold's second headline metric ("real AV loss 0.796");
a reader who follows definitions cannot determine which model produced the
number.
Proposed change: verify the metric's producer in the source report, then use
one name everywhere on the panel (if it is the AV's autoregressive NLL under
different conditioning, say "AV token loss"; never bare "AR loss" given the
glossary's AR = critic). Add the term to the panel's terms list.
Affects scientific meaning: **yes** (metric attribution must be stated
correctly).

**F10 — Mobile TRACE overflows horizontally; several SVG labels collide or
clip at every width. — MEDIUM-HIGH (responsive/visual defects)**
Station/panel: TRACE shell (mobile); CHANNEL waterfall & real-vs-control;
BENCH dose charts; AUDIT null-text and claim cards (tablet).
Evidence (measured): at 390px the TRACE document-selector `controls-row`
forces `.trc-station` to 380px > 370px viewport content width — the whole
page pans sideways; the x-axis title "directional MSE (dMSE)" overprints the
"0.60" tick at all widths; dose-chart y-labels "KL(orig → patched)" clip at
1024/390; null-text word labels ("documentation", "administrative") exceed
their 96px label column at tablet; the "functional interventions" claim-card
h4 overflows its card at tablet.
Why: horizontal page pan on mobile breaks the reading flow entirely;
label-on-tick collisions look broken and cost trust in otherwise careful
charts.
Proposed change: allow the doc-select row to wrap (`min-width: 0`, ellipsize
the mono doc id), move x-axis titles below the tick row (increase BOTTOM),
rotate-or-truncate long y-axis titles with `title` fallback, widen the
almanac label column or middle-ellipsize, and let claim-card headers wrap.
Affects scientific meaning: no.

---

## 4. Station-by-station audit

Format per panel: *first-read* (what a first-time reader will think it
shows), *actual* (what it computes/shows), then deltas: mismatches,
unexplained terms, whether the main comparison is visually obvious, and
whether controls/caveats clarify or overwhelm. Severities in brackets refer
to the findings register (§3 ids where applicable; otherwise inline).

### 4.0 Shell, primer, evidence rail

- *First-read:* a lab instrument: brand, four tabs, a status strip, a
  pipeline diagram, a glossary.
- *Actual:* matches. The rail states population/split, counts, "static
  snapshot · no model runs at view time," and a provenance flyout.
- Station tabs communicate only their names; their one-line questions live in
  `title` tooltips, which are undiscoverable on touch and never seen by most
  readers. **[medium]** Add a compact orientation strip (four rooms, four
  questions) under the primer on first load (see §10).
- The primer repeats in full on all four stations (~270px). **[medium,
  density]** Collapse it to one line after the first station visit
  (localStorage), with the flow diagram behind a toggle.
- The evidence rail says "50 rows · 7,434 precomputed interventions," while
  the first two CHANNEL panels report 512-row report evidence; the population
  relationship (512-row e0/e2 reports vs. the 50-row deep-dive panel vs. 24
  behavior rows) is never stated in one place. **[medium, population
  clarity]** Add a consistent population pill per panel ("512-row report" /
  "50-row panel" / "24 behavior rows").
- Theme toggle precedes navigation in the mobile stacking order. **[low]**

### 4.1 CHANNEL

**Selected row: source → NLA description.** *First-read:* "the source text
and what the model wrote about it." *Actual:* full source, AV description of
one stored activation, per-critic reconstruction metrics, teacher behind a
disclosure. Missing activation-token highlight is F2 (**high**). The
metrics' norm ratio (1.280 here, up to ~1.7 on BENCH) shows with "1.0
matches magnitude" but no pointer to the magnitude story on AUDIT.
**[low]** Link "why direction-only" to the magnitude card. The guide's
critic note ("does not regenerate the displayed AV text") is exactly the
right disambiguation and should stay.

**Information waterfall.** *First-read:* "shorter bar = better; blue beats
gray; teacher ties." *Actual:* per-variant directional MSE with
family-clustered CIs from the 512-row e0 report. Raw variant names are F3
(**high**); the teacher bar is colored as "other report-e0 variants," visually
demoting the panel's stated key comparison (learned vs teacher) **[medium]**
— give teacher its own labeled treatment (e.g., outlined bar or a "reference"
tag). The design doc's ladder floor `source_raw ≈ 0.083` is absent from the
shard (verified: variants are av_mean/av_none/av_real/av_shuffled/av_zero/
mean/teacher), so the exhibit cannot answer "how close is AV text to the
best possible re-encoding?" **[medium, affects evidence completeness]** —
either ship the rung from the run record or state its absence on-panel.
"512 rows · 250 fam" repeats under every label **[low]** — hoist to one
caption. Axis-title/tick collision is part of F10. Dead vertical space below
the chart (grid stretch next to the taller sibling) **[low]**.

**Real-vs-control AV loss.** *First-read:* "real text is easier for the
model; every control is worse; wins 100%." *Actual:* AV-conditioned loss on
explanation text under real vs mean/none/shuffled/zero conditioning, from an
8-row canary and the 512-row paired e2 report. Metric-identity contradiction
is F9 (**medium-high**). "Canary" is unexplained **[low]**. The 8-row canary
bars mostly duplicate the e2 story **[low, density]** — table-only would do.
The paired-check line ("beats each control on **100.0%** of 512 rows") is the
panel's best sentence; consider promoting it above the chart.

**Twin critics.** *First-read:* "two graders agree." *Actual:* per-row
identity dMSE under both critics with the shared-teacher confound stated
twice. Good. The summary table's `e3`/`p2` report codes and wildly different
cell counts are part of F3 **[medium]**. Fifty focusable dots are keyboard
users' only path besides the per-row table; acceptable, but the table should
be advertised as the faster route ("per-row table (50 rows; select via
keyboard)" already does — good).

**Capacity ladder.** *First-read:* "accuracy falls slowly as the gallery
grows." *Actual:* identity retrieval vs gallery size 2→256 with a Fano-style
lower-bound framing and verbatim assumptions. The bits axis (1–8) never
shows the 2→256 sizes a reader was promised **[medium]** — dual-label ticks
("2 · 4 · … · 256") or relabel "gallery size (log₂)". The y-axis starts at
~94%, which makes a 4.3-point drop look like a plunge directly under copy
that says "slowly falling" **[medium]** — annotate the truncated axis or
start at a round anchor with a break marker. Top-confusion rows are
hash-to-hash pairs (`25bba68943… → 8a5fb4a8f5…`) **[medium]** — resolve to
row previews (both families have panel rows) or demote behind a details.
`fano_scope` text appears twice (evidence note + assumptions card) **[low]**.

**Retrieval.** *First-read and actual match:* hit-rate tiles plus a
kept-visible misses table. Best-practice negative-result handling. "expected
cos" is unexplained (why *expected*?) **[low]** — one clause in the guide.

**Words buy direction (rate–distortion).** *First-read:* "more words, better
reconstruction, plateaus around 45 words." *Actual:* prefix truncation **of
the teacher explanation**, scored by the primary critic; the guide's "What
this shows" says "truncates learned descriptions" — wrong text named
**[medium, copy]** (the evidence note is correct; align the guide and title
copy). OOD caveat for truncated text is present — good. Context lines (50
gray rows) have no hover identity **[low]**.

**Code attribution.** *First-read:* "which words mattered." *Actual:* word
occlusion (AR-critic deletion sensitivity) over the *teacher* text + exact
section Shapley. Nearly every word carries a tint/underline, so high-salience
words barely pop and `d_dmse ≤ 0` dotted marks are hard to find **[medium]**
— tint only top-K positives + all negatives, leave the rest plain. The note
calls the teacher explanation "a learned description of the stored
activation," blurring the AV-vs-teacher distinction the row reader carefully
draws **[medium, terminology]**. `utility = one_minus_directional_mse` and
`efficiency_error = 1.1e-16` are builder-speak **[low]** — "utility = 1 −
dMSE" and move the check into a tooltip.

**Alternate-tellings fan.** F8 (**medium-high**) on the family-mean line.
Also: every telling's preview starts with the same boilerplate
("Syntax/continuation feature: …"), hiding the differences the panel is
about **[low]** — start previews at the first differing clause or diff-mark
them.

**Rewrite explorer.** *First-read:* "original vs rewrite; does the code
survive?" *Actual:* format-transform ("paraphrase") and corruption families
scored by both critics, plus court verdicts. The "paraphrase" chips are all
*layout* transforms (bullets, spacing, rotation) — the family name promises
semantic rephrasing that never appears; the excellent caveat ("names the
intent … not a verified property") sits at the very bottom **[medium]** —
rename to "format paraphrase" and move the caveat beside the family toggle.
Verdict framing is F6 (**high**). Δ-cells use ▲/▼ for *sign*, not for
better/worse, so "cosine ▲ +0.000" and "raw MSE ▲ +0.088" wear the same
glyph with opposite valences and ±0.000 gets an arrow at all **[medium]** —
either encode improvement (▲=better per metric direction) or drop arrows for
signed numbers; suppress arrows at |Δ| < 0.0005. `identity_cosine` (the
court's decision variable) is undefined on-panel **[medium]** — one-line
definition + add to terms. Spec JSON inline **[low]** (F3).

### 4.2 TRACE

**Document token-linked NLA reader.** The strongest panel in the app
(§2). Deltas: sampled-token chips are thin underlines that vanish on
skimming, and single-character tokens (".", "5") are ~6px hit targets
**[medium]** — add padding/larger hit area and a slightly stronger
available-token treatment. Parse chips leak `usable_closed` and duplicate
"usable output" **[low]**. The stepper is discoverable and the
document-not-in-rows / unaligned-document absence states are excellent.

**Persistence of adjacent descriptions.** *First-read:* "how similar is each
description to the next; bars around 0.2–0.4." *Actual:* client-side Jaccard
word overlap between adjacent descriptions. The metric is never named
on-panel **[medium]**; there is no anchor for "high" (the shipped bundle has
no shuffled-position trace control — the absence is honestly boxed, but the
similarity axis still floats without a chance line) **[medium, statistical
ambiguity]** — name the metric and add a static reference ("random
description pairs from this film set average ≈ X" can be computed offline
and shipped as a constant with provenance, or explicitly say no baseline
exists). The 39-bar chart towers over its content (≈1,300px) with only every
~6th transition labeled **[medium, density]** — compact to a strip/heatmap
with the selected transition enlarged. Diff headers say "position 43/56"
without the tokens at those positions **[low]** — append the token texts.

**Real-vs-shuffled control / Fresh-forward drift / Cross-station: BENCH
(bottom row).** Honest and compact. The drift panel's title ("Fresh-forward
drift") does not match its guide rule ("fresh-vs-stored drift"), so it is
the one panel in the app with no How-to-read column (confirmed live)
**[medium, guide plumbing]** — align the rule/title and add a "zero would
mean exact reproduction" line. Drift metric labels (1−cos, relative L2, RMS
ratio, max |Δ|) are undefined here **[low]** — reuse AUDIT's terms.

**Poetry planning lens — reader.** Future-token dimming, the held-out-line
box, and the real/control sample lanes are excellent. Deltas: F1
(**high**). Sub-word tokens (selected token "t" of "couplet") confuse
non-tokenizer readers **[medium]** — one hint line ("tokens are model
tokens; words may split"). "offset −12" appears before any explanation of
offsets; the guide that defines offsets ("prefix and analysis window") exists
in `panelGuides.ts` but no panel title matches it — dead rule **[medium]** —
surface it (the kchips row is a natural home). Real samples that verbalize
*bananas* for a carrot poem are scientifically interesting decoder drift but
read like a data bug with no framing **[medium]** — one line in the lane
header ("samples vary; content words can drift — that is why the shuffled
control exists"). On mobile the six full samples produce ~5,000px of scroll
before the onset curve **[medium, mobile density]** — collapse samples 2+
behind "show all samples" on narrow widths.

**Planning-onset curve.** *First-read:* "blue spikes early, gray flat, onset
at −6." *Actual:* per-offset target-family rate over 4 samples vs a
shuffled-prefix control, with a shard-supplied onset gate. Blue overplots
gray wherever both are equal (most offsets are 0/0), so the control
disappears exactly where it matters
([05-onset-curve-control-occlusion.png](nla_dashboard_ux_audit_assets/05-onset-curve-control-occlusion.png))
**[medium]** — halo/ring the control series or offset it by a pixel;
verified the shard has shuffled scores at all 13 offsets, so this is purely
overplotting. Rates are quantized to 0/.25/.5/.75/1 by n=4 but the chart
never says so **[medium, statistical ambiguity]** — subtitle "each point =
share of 4 samples (gate 0.25 = ≥1 sample)". Column-click selection is
invisible until hover **[low]**. Axis-title collision at −6 tick **[low,
F10 family]**.

**Baseline continuation.** Clear input→output flow with the target-family
verdict badge. Good. The observed baselines (meta-instructions like "We need
to fill in the blank…") make the 1/8 baseline-hit stat concrete — no change
needed.

**Causal edit bench.** Step 1/2/3 scaffolding, always-visible failed-steering
verdict, and full outputs at every dose for both directions: exemplary.
Deltas: "edit Δ norm 49.80" has no scale anchor **[medium-low]** — express as
a fraction of the original activation norm; original cosine 0.782 (notably
below identity-cell quality) passes without comment **[low]** — one
sentence. Dose-0 texts duplicate across lanes by construction **[low]** —
say "identical by construction at dose 0."

**All cases summary.** Every case on the board with originals, references,
outputs, and per-case stats; the honest aggregate footer. Deltas: "anchor
lift" is never defined at point of use, and `carrot-rabbit`'s "onset −6 ·
lift 0.000" pairing is baffling without knowing lift is anchor-position-only
**[medium]** — add "(real − shuffled rate *at the anchor*)" to the label or
kv list. Outputs end mid-word with no truncation mark (32-token cap)
**[low]** — append "…" + "(32-token cap)".

### 4.3 BENCH

**Banner + brief.** The sticky "Precomputed evidence only · viewer, not a
live model editor" banner with the grid-spec hash is exactly the right
standing disclosure. With the header it forms a double sticky stack that
worsens F7 and eats ~110px **[medium-low]** — consider collapsing the banner
to a slim line after first scroll.

**Row + move picker.** *First-read:* "pick row/chip/dose; see what exists."
*Actual:* same, plus a coverage map with variant counts by depth. The
coverage table's "variants" preview column truncates at 4 with "+N" and a
title tooltip — fine. Rows are listed as `v-248348 — BEHAVIOR` / "—
metric-only" — good sorting (behavior first). No row *content* preview
(what text is this row?) before selection **[medium, selector context]** —
add the source-text preview to options or a preview line under the select
(the row reader dialog pattern from CHANNEL would port well).

**Control rack.** Lanes-travel-together works, and the quick-read strip
("movement, not success") is the right lens. Deltas: F5 colors; placebo and
identity metrics are byte-identical here (0.245/0.878/1.695 on the default
selection) with no note — if the placebo cell genuinely re-encodes to the
identity vector, say so; if not, this needs a data check upstream
**[medium, data-integrity question]**. Independent-critic rows render six
em-dashes whose explanation lives only in tooltips **[medium-low]** — one
visible line ("independent critic scored identity cells only"). Duplicate
per-lane footers (8 buttons) **[low, density]** — one shared footer row.

**Reconstruction compass.** F4 (**high**). Also the only panel where two
badges (stored-snapshot metrics vs validation-fitted PCA) must be parsed
together to know what is qualified; the split is correct — keep it, but the
axis label should carry the same "display only" tag as the sub.

**Next-token top-k movement.** *First-read:* "tokens reshuffle when
patched." *Actual:* original vs patched top-10 for the focused lane with
rank markers + a cross-lane top-1 strip. The decisive evidence — all four
lanes produce the same top-1 (`osi`, p≈0.99) and identical 0.30 overlap — is
present but un-summarized **[medium]** — add a one-line computed readout
("edit does not separate from controls on this cell: identical top-1 across
lanes") or a Δ-vs-identity column. Sub-word tokens unexplained (same fix as
TRACE) **[low]**. The teacher-chip note ("even the unedited teacher
re-encoding moves the top-10 — that movement is the harness floor") is
excellent and should be echoed on edit chips.

**Divergence gauges.** Clear metric-direction sublabels ("higher = bigger
shift"). Deltas: F5 (bar colors); no explicit edit-vs-controls delta
anywhere **[medium]** (same fix as top-k); placebo==identity values repeat
here (5.377 KL both) — same data question as the rack.

**Causal wake.** Honest teacher-forcing caveat. The guide says "a spike at
offset 0 is the immediate intervention," but the axis starts at offset 1
(wake is t+1…t+16) — readers hunt for a missing point **[medium, copy]** —
reword to "at the first offset." End-of-line labels stack at the right edge
and can collide/clip with four lanes at near-identical values **[low]** —
the nudge exists; add ellipsis room or drop to two labels when values tie.

**Continuations.** When outputs share almost no words, the full-diff marks
~every word and the panel becomes stripes **[medium]** — threshold the diff
(if shared-word ratio < ~15%, drop per-word marks and say "outputs are
substantially different"). Stored text begins mid-word ("oponnesian War")
with no ellipsis **[low]**. Generation-protocol kv is good provenance;
collapse by default **[low]**.

**Dose comparison.** Right chart form (slope), selective labels, honest
sub. With doses {0.5, 1} it is two points per lane — fine, but say "two
doses only" so nobody reads a trend **[low]**. Y-label clipping at small
widths is in F10.

**Session ledger.** Clear, honest, local-only wording. No issues.

**Metric-only rows.** Six absence boxes repeat the same sentence with the
same link **[medium-low, density]** — keep per-pane boxes (absence must stay
visible) but shorten five of them to one line referencing the first.

### 4.4 AUDIT

**Claim ledger.** The four claim cards + legend + verbatim limitations are
the dashboard's spine and read well. The status string is shown twice per
card (badge + verbatim mono) — intentional, but the mono duplicate could sit
in a tooltip **[low]**. Deep-link flash works (F7 scroll offset applies).

**Negative & weak results.** Verbatim statements with sources; exactly
right. No changes.

**Parse health & control completeness.** Tables verified not clipped;
`humanize()` on parse states helps. The kinds are still mono keys
(`qualified_av`, `trace_description`) **[low]** — sublabels.

**Cipher court docket.** Threshold cards, confound badge, worst-first
sorting, and the details-gated 50-row docket are strong. F6 applies to the
✓/✕ glyph semantics; the legend explains color=calibration but not
*expectation* **[medium-high, F6]**. `para min cos` / `para mean cos` column
heads are tight but fine; `identity_cosine` again undefined **[medium,
F3/F6 family]**.

**Fresh-vs-stored drift.** Generic-but-honest card rendering, worst-first E5
table, publication-ready=false surfaced. Good.

**Null-text almanac.** Backfill-pending caveat is prominent; the two word
lists are readable; the loss table mirrors CHANNEL's (fine as the almanac's
evidence leg). Label clipping at tablet is in F10.

**Magnitude card.** Correctly explains why the claim is directional;
candidate table with a "selected" badge. `centered_r2` etc. remain raw keys
**[low]**.

**Poetry pipeline status.** "Completion ≠ success" framing present — good.

**Provenance browser.** Deep and honest. It prints absolute builder-machine
paths (`/Users/rigarg/…/tokenizer.json`, `/workspace/nla-viz/evidence/…`)
**[medium, publication hygiene]** — relativize to repo/evidence roots or
label "builder-machine path (not resolvable here)". Everything else —
excluded files with reasons, source caches, human-review badge — is
best-in-class.

---

## 5. First-time-reader journey (as observed)

**0–30 seconds (desktop).** Fold shows: brand, four tabs, evidence rail,
primer with flow diagram and the not-a-transcript sentence, CHANNEL's
question, bounded answer, scope badge, three metrics. A newcomer *can* answer
"what is an NLA?" and "what is being claimed?" — genuinely rare. Two of the
three metrics, though, are not yet parseable ("real AV loss 0.796 / best
non-real control 1.181" presumes F9's metric identity; "94% identity
retrieval @1 · row × critic pairs" presumes retrieval). The station tabs
give no hint of the four-room story (tooltips only).

**30 seconds–3 minutes.** The row reader delivers the core experience
(source → description → scores), minus the token anchor (F2). Scrolling into
the waterfall, the reader meets `av_zero`…`mean` (F3) and must open "Terms in
this panel" or hover to proceed; the key comparison (real ≈ teacher, both ≪
controls) *is* visually present. Twin critics and capacity read fast. The
collapsed "02 · Explore what the descriptions encode" section is a good
altitude gate — but the reader has no cue that the *rewrite explorer* (the
most persuasive interactive evidence) is inside it.

**3–15 minutes.** TRACE film set is self-explanatory and pleasant; the
poetry lens tells its weak-signal/failed-steering story honestly — until the
reader notices F1's underline-vs-curve contradiction or the banana samples
and stalls. BENCH's rack/gauges/wake are individually readable, but the
reader must aggregate "no separation from controls" themselves (top-k strip,
identical gauges, wake overlay) — the stations *show* the null result but
never *say* it at panel level (the station brief does say it up top).
AUDIT rewards the visit; the claim ledger and negative results are where the
skeptic finally relaxes. Deep links land under the sticky header (F7).

**Mobile.** Everything stacks competently, but the primer+rail consume the
first two screens; TRACE pans horizontally (F10); the poetry samples column
is a 5,000px scroll. Usable for reference, not for a first visit.

---

## 6. Terminology and copy problems

| # | Where | Problem | Fix | Sci-meaning? |
|---|---|---|---|---|
| T1 | Waterfall, real-vs-control, parse tables | Raw variant/enum keys (`av_none`, `mean` vs `av_mean`, `usable_closed`, `qualified_av`) | Plain sublabels; keep keys as mono secondary (F3) | no |
| T2 | Real-vs-control | "AV loss" (title/guide) vs "AR loss" (note/axis) | Resolve to the true producer; one name (F9) | **yes** |
| T3 | Words-buy-direction | Guide says "truncates learned descriptions"; data truncates *teacher* text | Correct the guide/title copy | **yes** (names the evidence text) |
| T4 | Attribution note | Calls teacher explanation "a learned description," colliding with the AV-text term used one panel up | Reserve "learned description" for AV output; call teacher "teacher reference" | **yes** (term hygiene) |
| T5 | Court (both stations) | `VERDICT: TRUE/FALSE`, green/red, calibration `context` unexplained; expectation flips by family | Outcome language ("code preserved/broken") + expected-vs-observed (F6) | reading, not values |
| T6 | Poetry cases | "anchor lift" undefined at use; anchor-only scope unstated | Define inline: "real − shuffled rate at the anchor" | no |
| T7 | Wake guide | "spike at offset 0" but axis starts at 1 | "at the first offset after the patch" | no |
| T8 | Twin critics table | `e3`/`p2` + unexplained cell counts | Caption the two reports | no |
| T9 | Capacity | "bits" axis vs "2 to 256" copy; "canary" (real-vs-control) unexplained | Dual-label ticks; gloss canary | no |
| T10 | Rewrite | Family "paraphrase" = format-only transforms; caveat buried at panel bottom | Rename "format paraphrase"; caveat beside toggle | **yes** (prevents over-reading invariance) |
| T11 | Rewrite/court | `identity_cosine` undefined; missing from terms list | Define; add to `TERM_DEFINITIONS` | no |
| T12 | Tellings | "family aggregate mean" ambiguous (population vs row) | "all-rows mean (50-row panel)" (F8) | no |
| T13 | TRACE drift panel | Title "Fresh-forward drift" misses the "fresh-vs-stored drift" guide rule → no How-to-read renders; `panelGuides` also contains dead rules ("prefix and analysis window", "av samples", "learned description") that no panel title matches | Match titles/rules; surface the offset explanation in the poetry lens | no |
| T14 | TRACE/poetry, top-k | Sub-word tokens ("t", "osi") shown without a tokenization hint | One hint line per reader/panel | no |
| T15 | Persistence | Similarity metric unnamed (Jaccard over words, client-side) | Name it on-panel | **yes** (metric identity) |
| T16 | Shell/BENCH briefs | "behavior cells", "BEHAVIOR/METRIC depth" before definition | Gloss at first use per station | no |
| T17 | Rewrite Δ cells | ▲/▼ = sign, colliding with better/worse intuition; arrows on ±0.000 | Direction-of-goodness encoding or plain signed numbers | no |

## 7. Visual / sizing / responsive defects

| # | Severity | Where | Defect (evidence) | Fix |
|---|---|---|---|---|
| V1 | medium-high | Mobile TRACE | Page overflows horizontally (`.trc-station` 380 > 370px content; measured) — whole page pans | Wrap/ellipsize doc-select row; `min-width:0` |
| V2 | medium-high | Shell | `scroll-margin-top: 0` under 61px sticky header (+ BENCH banner) hides anchored panel titles on every deep link | Scroll margins per F7 |
| V3 | medium | CHANNEL waterfall, real-vs-control; onset curve | X-axis title overprints tick labels ("directional MSE (dMSE)" over "0.60") at all widths | Move titles below tick row; raise BOTTOM |
| V4 | medium | BENCH dose, compass; AUDIT almanac (tablet) | Y-axis titles clip ("KL(orig → patched)" 53<141px); "PC2" clipped; almanac word labels exceed 96px column | Rotate/shorten labels; widen columns |
| V5 | medium | BENCH lanes | Control-lane color collision (validator FAIL: ΔE 7.3; chroma floor) | New lane palette (F5) |
| V6 | medium | Onset curve | Coincident series: blue occludes gray control at equal values (shard has all 13 offsets; verified) | Halo/offset control marks |
| V7 | medium | Attribution | Universal underlining compresses salience; negative marks hard to find | Mark top-K + negatives only |
| V8 | medium | Persistence | ~1,300px bar stack for a secondary readout; sparse labels | Compact strip/heatmap |
| V9 | low-medium | Waterfall, capacity, tellings | Large dead space under charts from grid stretch | `align-items: start` or auto rows |
| V10 | low | Wake | End labels stack/collide at right edge when lanes tie; "edit" label can sit outside comfortable margin | Wider right margin; label dedup |
| V11 | low | Tablet AUDIT | Claim-card `h4` overflows card ("functional interventions…" 202<211px) | Allow wrap |
| V12 | low | Continuations | Near-total word-diff stripes; mid-word stored text without ellipsis | Diff threshold; "…" affordances |
| V13 | low | Mobile shell | Theme toggle above nav; primer+rail consume first two screens | Reorder; collapse primer (§10) |
| V14 | low | Evidence-rail provenance flyout | Chips wider than the flyout at narrow widths (76<162px probe; verify open state on mobile) | Wrap chips |

## 8. Scientific-communication risks

1. **Underline vs score (F1)** — the one place the dashboard *shows* evidence
   that its own scoring disagrees with. Fix before external sharing.
2. **Compass distance illusion (F4)** — a display-only projection that
   visually asserts "reconstruction failed"; the strongest picture on BENCH
   currently argues against the qualified claim it decorates.
3. **Verdict/expectation inversion (F6)** — green TRUE on corruption cells
   invites "the code robustly survives corruption = good," which is
   backwards for the cipher-court logic.
4. **Metric misattribution (F9, T3, T4, T15)** — AV vs AR loss; teacher vs
   learned description; unnamed Jaccard. Each is a place where a careful
   reader cannot reconstruct what was measured from the screen.
5. **Un-anchored statistics** — onset rates over n=4 without an n statement;
   persistence similarity without a chance anchor; "anchor lift" without its
   anchor-only scope; edit Δ norm without a scale. None overclaim, but all
   invite over- or under-reading.
6. **Population blending** — 512-row report aggregates, the 50-row cleared
   panel, and 24 behavior rows interleave without a consistent population
   pill; a skeptic tallying Ns will stumble (rail says 50; first two panels
   say 512).
7. **Identical control values without comment** — placebo == identity to the
   fourth decimal across rack/gauges (and edit == random on JS) on the
   default cell. If real, annotate "coincides by construction"; if not, audit
   the cell resolution upstream. Either way, silence here reads as a bug.
8. **Mind-reading language: clean.** No instance found of "thoughts,"
   "believes," "knows," or hidden-CoT framing outside explicit negations; the
   "not a transcript" line recurs at every text exhibit. The banana-drift
   samples (real activations verbalized as the wrong fruit) would still
   benefit from one sentence normalizing decoder drift so readers don't
   mistake honesty for misalignment.
9. **Local path leakage in provenance** — publication hygiene rather than
   science, but external readers will screenshot it.

## 9. Accessibility findings

Positives (verified live): logical heading outline (h1 → h2 brief → h3
panels → h4 subsections); visible 2px focus outlines on every stop sampled;
`:focus-visible` and `prefers-reduced-motion` rules present; all sizable
SVGs carry `aria-label`s with data summaries (7/7 on CHANNEL); nearly every
chart has a table twin; scrollable text regions are focusable; the
example-browser is a native `<dialog>` with `aria-labelledby`, backdrop
mousedown-to-close, and Escape support; axe checks run in the unit suite.

Defects:

| # | Severity | Finding | Fix |
|---|---|---|---|
| A1 | medium | Station tabs use `role=tab`/`tablist` but arrow keys do not move focus (verified) and no `aria-controls`/`tabpanel` exist | Implement roving tabindex + arrow nav, or drop to `nav` + `aria-current` buttons |
| A2 | medium | Lane identity is color-alone in several charts (gauges bars, dose lines mid-chart, onset real-vs-shuffled) and F5 makes two lanes identical even for full color vision | Palette fix + end labels (wake has them; add to gauges/dose), tooltips already carry names |
| A3 | medium | Tiny hit targets: single-character token chips (TRACE), ✓/✕ court cells (~14px), tellings dot targets | Min 24px hit areas via padding |
| A4 | low-medium | Twin-critics: 50 tabbable circles precede the table alternative in DOM order | Offer "skip to per-row table" link before the chart, or make dots one composite stop with arrow nav |
| A5 | low | No skip-to-content link past header+rail+primer | Add skip link |
| A6 | low | Search input consumes first Escape (clears text) before dialog closes — native but surprising; ensure opener regains focus on close (native behavior, verify in tests) | Document/accept; add focus-return assertion to axe/e2e |
| A7 | low | `aria-live` only on dialog result count; critic switch and lane focus changes are not announced | `aria-live=polite` status line for global selector changes |

## 10. Recommended information architecture

Keep the four stations and their order — the "one story in four rooms" spine
is right, and CHANNEL-first matches evidence strength. Change the connective
tissue:

1. **Orientation strip (new, one-time).** Under the primer on first visit: 
   four cards — CHANNEL "what survives language" · TRACE "what changes across
   tokens" · BENCH "edit the code, see consequences" · AUDIT "what is safe to
   claim" — each with its scope badge and a one-line current answer
   (qualified positive / exploratory / weak+negative / boundaries). This is
   the design doc's own table, currently living only in tab tooltips.
2. **Primer collapse.** After first station visit (localStorage), reduce the
   primer to its heading + one-line boundary sentence with a "show diagram"
   toggle; keep the glossary access point fixed.
3. **Population pills.** A single component ("512-row report e0" / "50-row
   cleared panel" / "24 behavior rows" / "8 poetry cases · 4 samples/pt")
   rendered in the badge row of every evidence panel; same position, same
   phrasing everywhere.
4. **Selected-example continuity.** The row selection already flows across
   panels and stations; make it visible: a slim "specimen bar" (example n/50,
   first 60 chars of source, position highlight) pinned at the top of
   CHANNEL section 02 and BENCH, with "change example" opening the existing
   dialog. This also gives BENCH's picker the missing row-content preview.
5. **Verdict lines on consequence panels.** BENCH panels each render
   evidence but defer judgment to the reader; add a computed one-liner per
   panel ("edit does not separate from controls on this cell") sourced from
   the same numbers already displayed. Population-level verdicts stay in the
   briefs; cell-level verdicts belong on the panels.
6. **Reading-order affordance.** "Continue to TRACE →" footer link per
   station; cheap, and makes the intended order explicit.
7. **Guides: keep, but make dismissible.** The What/How guides are the right
   default for first-time readers; add a per-session "hide reading guides"
   toggle in the header for returners (they currently cost ~200px × ~30
   panels).

## 11. Concrete panel-level changes

Ordered within each station by impact; ids reference §3/§6/§7.

**Shell**
1. Scroll margins for all anchor targets (F7/V2).
2. Orientation strip + primer collapse + population pills (§10.1–3).
3. Tabs: arrow-key nav or role downgrade (A1); skip link (A5).
4. Mobile: nav above theme toggle; wrap provenance chips (V13/V14).

**CHANNEL**
5. Row reader: inline activation-token highlight + caption (F2); link norm
   ratio to the AUDIT magnitude card.
6. Waterfall: variant sublabels (F3); distinct teacher treatment; add
   `source_raw` rung to the shard or an explicit absence note; single
   rows/families caption; axis-title spacing (V3); grid `align-items:start`
   (V9).
7. Real-vs-control: resolve AV/AR naming (F9); gloss "canary"; consider
   table-only e1 lane.
8. Twin critics: caption e3/p2; keep.
9. Capacity: dual-label bits axis; annotate truncated y-axis; humanize or
   demote confusion pairs.
10. Truncation: fix guide copy (T3).
11. Attribution: top-K+negative marking only (V7); "utility = 1 − dMSE";
    teacher-vs-learned wording (T4).
12. Tellings: relabel population mean, add row mean (F8); differentiate
    preview text.
13. Rewrite: rename family; move caveat up; verdict reframe (F6); Δ-glyph
    policy (T17); define `identity_cosine`; demote spec JSON.

**TRACE**
14. Poetry reader/samples: underline-from-flags (F1) — builder change to ship
    matched spans preferred; tokenization hint (T14); offsets explanation
    surfaced (T13); drift-panel guide rule fix; sample collapse on mobile.
15. Onset curve: control halo (V6); "n=4 per point; gate = ≥1 sample"
    subtitle; axis spacing.
16. Persistence: name Jaccard (T15); chance anchor or explicit "no baseline"
    line; compact chart (V8); token texts in diff headers.
17. Film set: token hit targets (A3); `usable_closed` chip copy; doc-select
    wrap (V1).
18. Cases: define anchor lift inline (T6); truncation marks on outputs.
19. Causal bench: Δ-norm as fraction of ‖h‖; note on original-reconstruction
    quality; "identical by construction" at dose 0.

**BENCH**
20. Lane palette + identity-as-reference-line option (F5/A2); end labels on
    gauges.
21. Compass: normalized projection or explicit magnitude framing (F4).
22. Top-k/gauges: computed separation line vs controls (§10.5); annotate
    coincident lane values or audit upstream (§8.7).
23. Rack: visible independent-critic note; shared footer; cell-id as hash
    chip.
24. Picker: row source-text preview (§10.4).
25. Wake: guide copy "first offset" (T7); label margin (V10).
26. Continuations: diff threshold + ellipses (V12).
27. Metric-only: compact absence boxes after the first.
28. Banner: slim after first scroll.

**AUDIT**
29. Court: expectation-aware cell/verdict framing (F6); define
    `identity_cosine`.
30. Provenance: relativize or label builder paths (§8.9).
31. Almanac label column width (V4); parse-kind sublabels.

## 12. Suggested implementation sequence

1. **Same-day copy pass (no layout risk):** T2/T3/T4/T6/T7/T8/T9/T11/T12/
   T16; wake guide; dead guide rules (T13); "canary"; anchor-lift
   definition. Includes the two metric-identity items — small diffs, big
   trust wins.
2. **Same-day CSS pass:** scroll margins (V2); axis-title spacing (V3);
   y-label clipping (V4); grid stretch (V9); claim-card wrap (V11); mobile
   TRACE overflow (V1); provenance chip wrap (V14).
3. **Encoding pass (≤1 day):** lane palette + gauge labels (F5, validate
   with the palette script); onset-curve control halo (V6); attribution
   marking (V7); Δ-glyph policy (T17); court/rewrite verdict reframe (F6 —
   review wording against the measurement contract).
4. **Linkage pass (1–2 days):** row-reader token highlight (F2); BENCH
   separation one-liners and picker previews; specimen bar; persistence
   compaction; continuations diff threshold; metric-only compaction.
5. **Data/builder pass (no GPU):** ship matched term spans for poetry
   samples (F1) and re-emit `poetry.json`; add `source_raw` waterfall rung
   from the run record (F7 in §3 list — waterfall completeness); decide and
   implement compass projection semantics (F4); investigate placebo==identity
   cell resolution (§8.7); relativize provenance paths. Each is
   builder-side, hash-bound, and re-verifiable with `data:verify`.
6. **IA & a11y finishers:** orientation strip, primer collapse, population
   pills, next-station links, guide-hide toggle; tab arrow-nav, skip link,
   hit targets, aria-live statuses.
7. **Regression:** extend the axe/e2e suites — assert every panel title
   resolves a guide (kills the T13 class), assert scroll-margin on anchored
   ids, snapshot the new palette in both modes, and add a fixture test that
   every displayed term-hit underline agrees with its sample's scored flags
   (locks in F1's fix).

---

## Ideal reader journey (target state)

**After 30 seconds** a reader should be able to say: *"A second model writes
a short English description of one hidden activation inside Nano30B, and a
third model can rebuild most of the activation's direction from that text
alone — beating every fake-input control. The text is a learned code, not
the model's inner monologue, and the claim is scoped to stored snapshots on
a validation panel."* (Today: achievable except the last clause; the
orientation strip and population pills close it.)

**After 3 minutes** they should add: *"I can read a real example — this
source text, the activation at this highlighted token, the description it
produced, and how well two independent critics rebuilt it. Descriptions
survive re-wording and word budgets to a measurable degree, retrieval picks
the right row 94% of the time at gallery size 50, and everything links to a
claim card that says exactly how far this evidence goes."* (Today: blocked
mainly by F2, F3, F9.)

**After 15 minutes** they should conclude: *"Across tokens the code shifts
plausibly and the poetry 'planning' lens shows a small, honest,
weak-and-uncaused signal — five of eight cases show early mentions, the
anchor-position lift is near zero, and every attempted causal edit failed
against matched controls. On BENCH, edits move the model but never separate
from placebo and random controls under this stored-snapshot protocol. The
one qualified claim is bounded direction recovery; magnitude, fresh-forward,
functional causality, and the test set are explicitly not claimed — and the
dashboard's audit trail (controls, negative results, provenance hashes,
privacy triage) lets me verify each of those sentences myself."* (Today: the
same conclusion is reachable, but only by a persistent reader; F1/F4/F6 are
the three places the current pixels argue against the correct reading.)

---

## Appendix: implementation status (`2026-07-18`, same day)

The frontend remediation pass was applied to `dashboard/nla-observatory`
after this audit (typecheck clean; all 98 vitest tests incl. axe pass; live
re-verification at all three viewports; lane palette re-validated —
chromatic lanes pass, edit↔placebo sits in the labels-required band and all
lane charts now carry direct labels).

**Implemented (frontend):**
- F1 *(mitigation)*: scored chips are labeled "scored: …" and are the stated
  authority; a caveat chip "term visible · not scored" appears exactly where
  a display-level term match lacks a scored flag (verified live on the
  carrot-rabbit anchor); footnote rewritten. Full fix (shard-computed match
  spans) remains a builder task.
- F2 *(mitigation)*: activation-position caption ("token 126 of 127 — the
  final token…") + a position-marker track under the row reader's source
  text. Exact inline token highlight needs char offsets from the builder.
- F3: waterfall variant meanings key + tooltips; teacher recolored as the
  key comparison; single rows/families caption; parse-state and
  explanation-kind labels humanized; `e1_av` canary glossed; Shapley
  "utility = 1 − dMSE"; spec JSON/cell ids demoted to disclosures.
- F4 *(mitigation)*: compass axis relabeled "unnormalized … display only" +
  an on-panel caution that distances include magnitude error and must not be
  read against the directional claim. Direction-only projection remains a
  builder decision.
- F5: `random_edit` lane moved to the orange series; gauges gained per-bar
  lane labels; palette re-validated.
- F6: court/rewrite verdicts now read "code preserved / code broken" in
  neutral chips with per-class expectations; unexpected outcomes (preserved
  heavy corruption, broken paraphrase) get a caveat ring; docket legend
  states the expectation rule.
- F7: `scroll-margin-top` from published `--header-h` (+ BENCH banner
  height) on all anchor targets — measured 77px/118px live.
- F8: tellings line relabeled "all-rows mean (whole panel)" plus a new
  dashed "this row's mean" tick.
- F9: metric renamed **AV token loss** everywhere on the panel (verified
  against report phases `e1_canary_av` / `e2_token_logprobs`).
- F10: mobile TRACE page-level overflow fixed (controls row wraps); axis
  titles moved clear of tick rows (waterfall, real-vs-control, onset,
  compass, wake); almanac label column widened; gauges hint shortened.
- §10 IA: four-room orientation strip under the primer (bounded answers +
  status badges, click-to-navigate); primer pipeline collapsible with
  localStorage persistence; BENCH picker shows the selected row's source
  preview; computed separation/coincidence readouts on gauges and top-k;
  metric-only absence boxes compacted after the first; rack footer
  deduplicated; independent-critic "—" explained in text; continuations
  suppress the word-diff when outputs share <15% of words; dose panel warns
  when only two doses exist; onset curve draws the shuffled control hollow
  and dashed with an n-per-point note; persistence chart names Jaccard, is
  width-capped, and diff headers carry token texts; poetry lens explains
  offsets and sub-word tokens; samples collapse behind a toggle on mobile.
- §9 a11y: skip-to-content link; tablist roving tabindex + arrow-key
  navigation (verified live); `Segmented` controls announce changes via a
  polite live region; token-chip hit targets enlarged; court cells carry
  outcome+expectation aria-labels.
- §8.9: provenance paths are prefixed "builder-machine path ·" via CSS.

**Deferred to the builder pass (data regeneration, no GPU):** shard-computed
term-match spans for poetry samples (completes F1); `source_raw` waterfall
rung (its absence is now stated on-panel); compass projection semantics
(F4); investigation of the byte-identical placebo/identity cells (now
annotated in the UI wherever they coincide); relative scale for the poetry
edit Δ norm; relativized provenance paths at emit time.
