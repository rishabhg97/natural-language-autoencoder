# NLA Offline Observatory Dashboard Build Plan

Date: 2026-07-18

## 1. Objective

Build one polished, local-first dashboard for exploring the completed Nano30B
R33 NLA evidence. The application must help a technical reader inspect what
the NLA reads, how its textual channel behaves, how representations change
across tokens, what precomputed edits do to the target model, and where the
evidence is weak or unqualified.

This is a static evidence viewer. It must not load model checkpoints, call a
model server, require RunAI, or perform GPU inference at runtime. All model
work is complete and the browser should read only precomputed artifacts.

The application should feel like a research instrument, not a marketing
site. The first screen is the usable observatory.

## 2. Required Reading

Read these before implementation:

- `AGENTS.md`
- `docs/viz/nla_offline_observatory_design.md`
- `docs/viz/nla_visualization_frontier_ideas.md`
- `docs/viz/nla_visualization_ideas.md`
- `observatory/README.md`
- `observatory/build_bundle.py`
- `observatory/verify_bundle.py`
- `observatory/bundle_common.py`
- `observatory/poetry_planning.py`
- `configs/nano_viz/offline_observatory.yaml`
- `configs/nano_viz/offline_observatory_poetry.yaml`

Use `docs/viz/nla_roundtrip_visualiser.html` only as explanatory background;
do not copy its architecture or styling.

## 3. Runtime Boundary

The dashboard runs entirely on a local CPU in a browser-backed development
server. It may perform lightweight filtering, sorting, aggregation, and chart
layout in Web Workers. It must not:

- load Nano30B, AV, or AR weights;
- call remote inference or grading APIs;
- alter or regenerate evidence;
- open sealed test data;
- silently substitute mock values for missing evidence;
- present a pipeline `passed` flag as evidence that a scientific hypothesis
  passed.

## 4. Source Artifacts

The artifacts are currently on the `nla-viz` RunAI workspace. Copy only these
lightweight outputs to the ignored local `artifacts/observatory/` directory.
Do not copy checkpoints, activation datasets, model caches, or optimizer
states.

### 4.1 Core Observatory Bundle

Remote source:

```text
/workspace/models/nano30b-nla-pilot/observatory/r33_offline_v1/bundle
```

Local destination:

```text
artifacts/observatory/r33_offline_v1/bundle
```

Expected core files and row counts:

| Artifact | Rows | Purpose |
|---|---:|---|
| `rows.parquet` | 50 | Qualified validation panel and identity |
| `metrics.parquet` | bundle-defined | Channel and reconstruction measurements |
| `geometry.parquet` | bundle-defined | PCA and geometry views |
| `retrieval.parquet` | bundle-defined | Capacity/retrieval ladder |
| `explanations.parquet` | 900 | Teacher, AV, alternate, and trace text references |
| `token_trajectories.parquet` | 400 | Ten-document token traces |
| `interventions.parquet` | 7,434 | Immutable intervention text registry |
| `behavior.parquet` | 888 | Precomputed functional outcomes |
| `court.parquet` | 1,200 | Paraphrase/corruption and critic evidence |
| `shapley.parquet` | bundle-defined | Section attribution evidence |
| `aggregates.json` | 1 document | Qualified summary metrics |
| `observatory_manifest.json` | 1 document | File hashes and bundle identity |
| `provenance.json` | 1 document | Claims and source provenance |

The intervention registry includes 300 paraphrases, 300 corruptions, 400
alternate tellings, 3,752 word occlusions, 800 section ablations, 500
truncations, and 1,332 clause-edit/control cells. This is sufficient for the
rewrite explorer; do not run another transformation evaluation for v1.

### 4.2 Poetry Planning Pack

Remote source:

```text
/workspace/models/nano30b-nla-pilot/observatory/r33_poetry_planning_v1
```

Local destination:

```text
artifacts/observatory/r33_poetry_planning_v1
```

Required lightweight inputs:

- `poetry_corpus/cases.jsonl`
- `poetry_extract/continuations.jsonl`
- `poetry_extract/trajectories.parquet`
- `poetry_describe/descriptions.jsonl`
- `poetry_score/sample_scores.jsonl`
- `poetry_score/position_scores.jsonl`
- `poetry_score/case_scores.jsonl`
- `poetry_reconstruct/reconstructions.jsonl`
- `poetry_intervene/interventions.jsonl`
- `reports/*.json`
- `queue_state.json`

Do not copy `poetry_extract/shards`, `poetry_describe/shards`, activation
vectors, or `reconstructions.npz` into the web bundle. The UI does not need
them.

Expected evidence:

- 8 poetry cases;
- 104 token positions;
- 624 sampled AV explanations across real and shuffled conditions;
- 5 cases with exploratory planning onset;
- 5 editable/reconstructed cases;
- 30 causal intervention outcomes;
- weak aggregate signal: mean anchor lift `0.03125`, intended baseline rhyme
  in `1/8`, and no alternate-rhyme hits under the first causal edit sweep.

These weak and negative outcomes must remain visible.

## 5. Repository Layout

Create the frontend as an isolated application:

```text
dashboard/nla-observatory/
  package.json
  vite.config.ts
  tsconfig.json
  index.html
  src/
    app/
    components/
    data/
    stations/
      channel/
      trace/
      bench/
      audit/
    workers/
    styles/
    main.tsx
  scripts/
    build_static_data.py
    verify_static_data.py
  public/
    data/                 # generated and gitignored
  tests/
```

Recommended stack:

- Vite, React, and TypeScript;
- Apache Arrow JS or compact JSON shards generated at build time;
- Observable Plot or D3 for quantitative charts;
- Lucide React for interface icons;
- Vitest and Testing Library for logic/components;
- Playwright for browser and screenshot verification.

Keep the generated data and `dist/` out of Git. Commit source, tests, schemas,
and a tiny synthetic fixture only.

## 6. Data Build Contract

`scripts/build_static_data.py` is the only layer allowed to read the raw
Parquet/JSONL evidence. It must:

1. Verify the core manifest hashes before reading tables.
2. Verify all poetry reports and the poetry config hash agree.
3. Validate expected row counts and referential joins.
4. Strip activation vectors and unneeded token arrays.
5. Emit deterministic, versioned, lazy-loadable browser shards.
6. Emit a top-level `dashboard_manifest.json` containing source hashes,
   generated file hashes, schema versions, counts, and build timestamp.
7. Fail closed on missing rows, duplicate IDs, unknown variants, broken
   references, or non-finite metrics.

Suggested browser shards:

```text
data/manifest.json
data/channel.json
data/rewrites.arrow
data/trace.arrow
data/poetry.json
data/bench.arrow
data/audit.json
data/rows/<row_id>.json       # optional lazy detail shards
```

All displayed aggregate numbers must be recomputed from or explicitly linked
to the bundled source values. Avoid hardcoded metrics in React components.

## 7. Application Structure

Use one persistent application shell with a compact evidence-build rail and
four stations. Preserve the selected bundle, row, position, intervention,
critic, and station in the URL fragment so views are reproducible.

### 7.1 CHANNEL: What Survives Language

Required views:

- information waterfall from controls to AV/teacher reconstruction;
- real-vs-control AV loss comparison;
- words-buy-direction/rate-distortion view;
- twin-critic comparison;
- alternate-tellings fan;
- rewrite explorer modeled after the supplied Anthropic screenshots.

The rewrite explorer must show original and transformed explanations side by
side. Transform choices come from the existing six paraphrase variants and
shuffle/delete corruption doses. Show directional MSE, raw MSE, cosine, norm
ratio, and a ratio/delta against the identity explanation. Never imply that
format-preserving transformations are semantically identical merely because
their labels say `paraphrase`.

### 7.2 TRACE: What Changes Across Tokens

Required views:

- token strip and scrubber;
- synchronized AV explanation at the selected position;
- persistence bars and description diffs;
- real-vs-shuffled trace control;
- fresh-forward/exploratory badge;
- poetry planning lens.

The poetry lens shows the first-line prefix, highlighted token, multiple AV
samples, target/alternate rhyme mentions, onset curve, baseline continuation,
and causal edit outcomes. Provide case navigation and keyboard token stepping.
Show failed planning and failed steering examples alongside positive onsets.

### 7.3 BENCH: Precomputed Counterfactuals

Required views:

- intervention family and variant picker;
- mandatory identity, placebo, and random-control lanes;
- directional reconstruction compass;
- next-token top-k movement;
- baseline versus patched continuation;
- KL/JS/top-k-overlap gauges;
- causal wake across future positions;
- dose comparison where available;
- append-only, local session ledger.

State clearly that users are choosing among precomputed experiments, not
editing a live model.

### 7.4 AUDIT: What Is Safe to Claim

Required views:

- supported/not-supported claim ledger;
- Cipher Court rows and semanticity verdicts;
- parse health and control completeness;
- fresh-vs-stored drift warning;
- null-text/control behavior;
- an exploratory Domain Canary casebook with matched SFT/RL explanations, full prompts,
  physical token-position links, real/shuffled/no-injection lanes, explanation length,
  behavior context, and blinded human-review status;
- bundle provenance and source hashes;
- explicit distinction among `qualified`, `exploratory`, `negative`, and
  `unavailable` evidence.

Every headline metric elsewhere should open or link to its audit/provenance
record.

The Domain Canary must display its limitations beside the result: its keyword scorer is
not a held-out classifier, RL explanations are `1.81x` longer than SFT on the current
panel, and separately regenerated behavior is checkpoint-independent. Until the blinded
review is complete, semantic-quality fields should read `pending`, never inferred from
parser success.

## 8. Interaction and Visual Design

- Build the actual instrument as the first viewport; no landing page.
- Use a quiet, information-dense research-tool layout.
- Use light mode by default with a system-aware dark mode.
- Use blue for selected evidence, green for qualified evidence, amber for
  exploratory evidence, and red only for failures or invalid states.
- Avoid decorative gradients, oversized hero text, nested cards, and
  marketing copy.
- Keep cards at 8 px radius or less.
- Use icons for familiar commands and tooltips for unfamiliar controls.
- Keep charts linked: hovering a row or token should highlight the same entity
  across visible panes.
- Support keyboard navigation, visible focus, reduced motion, and screen-reader
  labels.
- Make all controls and text fit at 390x844, 1024x768, and 1440x900.

## 9. Implementation Sequence

1. Create the isolated Vite/React application and local data directories.
2. Copy and verify the compact evidence locally.
3. Implement the deterministic static-data builder and verifier.
4. Build the shell, routing, URL state, evidence badges, and loading states.
5. Build CHANNEL, including the rewrite explorer.
6. Build TRACE, including the poetry planning lens.
7. Build BENCH with compulsory control lanes.
8. Build AUDIT and provenance deep links.
9. Add error, empty, unavailable, and malformed-bundle states.
10. Run unit, integration, accessibility, and Playwright screenshot tests.
11. Start the local development server and provide its URL.

The sequence is for implementation control only. Deliver the complete app in
one pass rather than stopping after a mock or intermediate milestone.

## 10. Verification Gates

The dashboard is complete only when:

- raw source hashes and generated shard hashes verify;
- all expected row counts match;
- no test rows or checkpoint/dataset artifacts enter the app bundle;
- every displayed number has a source path and bundle/config hash;
- poetry pipeline completion is not mislabeled as scientific success;
- rewrite and poetry controls are visible and cannot be hidden accidentally;
- URL permalinks restore the same station and selected evidence;
- no runtime network call is required beyond loading local static files;
- first meaningful station content appears without loading the full bundle;
- all tests pass;
- Playwright screenshots pass visual inspection at desktop, tablet, and mobile;
- no text overlap, clipped controls, blank charts, console errors, or broken
  keyboard interactions remain.

## 11. Out of Scope for Version 1

- live prompt entry or live model inference;
- new GPU evaluations;
- evaluation-awareness experiments;
- French translation or new LLM-generated transformations;
- opening sealed test data;
- model training, checkpoint conversion, or W&B integration;
- a public deployment before the local app and evidence verifier pass.

## 12. Deliverables

- complete source under `dashboard/nla-observatory/`;
- deterministic data builder and verifier;
- tiny committed fixture for tests;
- generated local static data from the real evidence;
- unit, integration, accessibility, and Playwright tests;
- screenshots for the three required viewport classes;
- a short dashboard README with local build/run commands;
- a running local development URL.
