# Nano30B NLA Execution Log

<!-- R33-HERO-BASELINE-PROTOCOL-INVALIDATED -->

> [!CAUTION]
> Publication status (`2026-07-16`): the archived `30.97% / 32.34%` hero
> comparison is not publication-valid because the 512-row SFT baseline mixed
> two generation protocols. Figures below remain historical internal-gate
> evidence only. The active family-clean SFT pair is qualified only for
> directional stored-snapshot recovery; the full selected-pair exposure audit
> found no unused in-corpus family. See `docs/current_state.md`.

This is the active tracker for what changed after the initial plan. Keep this
file operational: phases, run outcomes, blockers, and explicit additions or
subtractions from scope. The scientific plan lives in
[nano30b-nla-core-plan.md](nano30b-nla-core-plan.md), and runnable commands live
in [cluster_runbook.md](cluster_runbook.md).

## Document Map

| File | Role |
|---|---|
| [../README.md](../README.md) | Entry point and current doc map |
| [nano30b-nla-core-plan.md](nano30b-nla-core-plan.md) | Canonical scientific and engineering plan |
| [cluster_runbook.md](cluster_runbook.md) | Exact cluster commands and expected outputs |
| [execution_log.md](execution_log.md) | Phase tracker, run history, additions/subtractions |
| [issues_iter1.md](issues_iter1.md) | Archived detailed rationale and discarded alternatives |

## Initial Brief

The initial deliverable was a small, reproducible Nano introspection and
extraction-identity harness. It explicitly excluded training, PEFT/LoRA,
serving, RL, large data generation, Track C-as-serving, and treating teacher
summaries as ground truth.

Scientific contract:

```text
frozen target Nano: x, tau -> h_b = R_b^target(x)_tau
AV: h_b -> z
AR: z -> h_hat_b
loss/eval: h_hat_b reconstructs h_b through text
```

AR reconstructs from explanation text `z`, not from source context `x`.

## Current Phase

Phase: **qualified family-clean R33 SFT AV+AR pair; confirmatory RL not launched**.

Current objective:

```text
deterministic R33 extraction and exact replay
-> family-disjoint clean AR SFT and validation gate
-> corrected packed clean AV SFT optimizer completion
-> bounded AV validation eval
-> protocol-matched clean SFT round-trip validation and one-time test
-> stored-snapshot functional recovery and qualified pair manifest
-> independent critic/AR and preregistered confirmatory RL
```

The deterministic 275,396-row data gate and clean AR/AV component gates pass.
The protocol-matched round trip passes on 512 validation and 512 test rows at
directional MSE `0.307004 / 0.319225`, with `100%` usable generations and all
family-clustered controls passed. Stored-snapshot functional recovery passes,
and the exact checkpoint-pair manifest reports `qualified: true`. The July 8
actor `iter_0000342` and its `30.97% / 32.34%` comparison remain historical
internal evidence because the baseline protocol was mixed and the stored
activations failed the later publication identity audit.

## Phase Ledger

| Phase | Status | Outcome |
|---|---|---|
| Environment and pinned model load | done | Cluster env uses pinned Nano/model tokenizer revisions and one visible GPU. |
| Introspection harness | done | Nano wrapper assumptions were confirmed: `.backbone`, `.backbone.layers`, `.backbone.norm_f`, `.backbone.embeddings`. |
| Extraction identity harness | done | `R_34` and `R_27` are the required boundaries to verify before training. |
| Track A / Track C probes | started | Keep Track A paper-faithful; use Track C as HF oracle/debug, not serving. |
| Synthetic frozen AR smoke | done as regression only | Loss can decrease, but controls did not establish scientific signal. Do not scale synthetic data. |
| Real-data Stage 0/1 smoke | done for first tiny R_34 slice | `64` FineWeb docs x `2` positions/doc produced `128` base rows, then AR-only split. |
| Stage 2 teacher explanations | done for diagnostics | Haiku and Kimi/NVIDIA API paths run through `OpenAIChatCompletionsProvider`; parser now handles Kimi `reasoning_content`. Teacher signal still fails controls. |
| Full Nano Stage 3 training data | done for exact R_34 AR slices | Reference parquet columns and token sidecars are preserved; large-scale generation remains blocked on teacher/prompt signal. |
| Qwen released NLA inference QC | done | Positive control passed: correct AR mean MSE `0.1628`, shuffled `0.9284`, random `1.0225`, median cosine `0.9247`. |
| Nano R_34 source replay probe | done | 256/256 exact token-count rows; correct cosine `0.99909`, normalized MSE `0.00182`; no blockers. |
| Nano R_34 AR capacity probe | first run done | Job `28042647`; train normalized MSE improved `1.3703 -> 0.7891`, train cosine `0.3149 -> 0.6055`; train correct beat controls, but heldout correct did not beat train-mean/shuffle controls. |
| Nano R_34 AR capacity ablations | done | Job `28043387`; head-only, longer tail-1, row-random, and tail-2 all passed mechanically but all failed heldout scientific controls. |
| Nano R_34 prompt signal gate, old parquet | done | Job `28044029`; source_raw oracle passed, but old Haiku teacher failed controls and parquet lacked `token_ids_prefix`. |
| Nano R_34 exact Haiku prompt gate | done | Jobs `28044572`, `28044573`, `28044574`; exact prefix fraction `1.0`, source_raw NMSE `0.0030`, teacher NMSE `1.1089` vs mean `0.8595`; scientific pass false. |
| Nano R_34 exact Kimi prompt gate | done | Jobs `28044907`, `28044908`; Kimi reasoning parser path works, exact prefix fraction `1.0`, source_raw NMSE `0.00174`, teacher NMSE `1.0729` vs mean `0.9595`; scientific pass false. |
| Qwen-faithful Nano AV warm-start smoke | first run done | Jobs `28115623` and `28119255`; real `h` beats shuffled/zero/mean/no-injection controls on heldout teacher-forced NLL after a tiny 50-step `lm_head` warm-start, but decoded generations collapse to `<` tokens. |
| Historical component-full AV/AR SFT | internal evidence only | R33 AR `iter_0001289` and AV `iter_0001291` passed their old gates, but predate deterministic extraction and the packed `position_ids` fix. |
| Historical corrected-K3 RL | publication-invalidated | R33 actor `iter_0000342` passed the old internal gate; its mixed-protocol headline is not an external result. |
| Deterministic family-clean R33 extraction | done | `275,396` rows, exact full replay, `d_model=2688`, zero nonfinite activations, zero empty explanations, and zero family/content overlap. |
| Publication-clean R33 AR SFT | done; validation passed | 1,291 updates; validation teacher directional MSE `0.281703`; selected pair checkpoint preserved. |
| Publication-clean R33 AV SFT | done; validation passed | 1,291 finite updates; validation real NLL `0.776775` beats every activation control. |
| Family-clean R33 SFT AV+AR pair | qualified | Validation/test directional MSE `0.307004 / 0.319225`; `100%` usable; all controls and stored-snapshot functional verifiers pass; exact pair manifest qualified. |

## Additions Since Initial Plan

- Added cluster execution path through the local `nano30b-nla` tmux session.
- Added `scripts/cluster_nano_env.sh` for pinned Nano environment variables and
  the cluster `libstdc++` preload needed by `selective_scan_cuda`.
- Added Nano real-corpus Stage 0 extraction:
  `scripts/nano_realdata_stage0_extract.py`.
- Added Nano AR smoke parquet builder:
  `scripts/nano_realdata_ar_build.py`.
- Added Nano reference-compatible Stage 3 builder:
  `scripts/nano_realdata_stage3_build.py`.
- Added real-data AR smoke commands to the cluster runbook.
- Added reference-aligned FineWeb `sample-10BT` AV/AR/RL dry-run commands to
  the cluster runbook.
- Added the requirement to keep the reference NLA CJK single-token marker
  contract for AV/Track A dry runs. Common English markers are disallowed.
- Added a Nano tokenizer-compatible CJK-symbol fallback for cases where the
  upstream enclosed-CJK marker block has no single-token entries.
- Added a reference-datagen alignment rule: reuse Stage 1/2/3 contracts where
  possible and swap only Nano Stage 0 extraction.
- Added Qwen released-checkpoint inference-only QC as a positive control.
- Added `scripts/nano_source_replay_probe.py` to test whether stored Nano
  activations can be recovered by replaying the original source prefix.
- Added exact token provenance for future Nano extracts:
  `token_position`, `token_id`, `token_text`, and `token_ids_prefix`.
- Added `scripts/nano_ar_capacity_probe.py`, a bounded reference-style AR
  critic capacity probe that freezes lower Nano prefix blocks, trains a small
  tail plus a value head, and evaluates correct/shuffled/random/mean controls.
- Added `scripts/nano_ar_signal_gate.py`, a pre-scaling prompt diagnostic that
  compares teacher, shuffled teacher, blank, generic, source-context,
  source-raw, and train-mean controls.
- Patched the OpenAI-compatible Stage 2 provider for NVIDIA-hosted reasoning
  models: Kimi can return usable text in `message.reasoning_content` with
  `message.content = null`, and Stage 2 now extracts the last complete
  `<analysis>...</analysis>` block from verbose reasoning traces.
- Added `scripts/nano_av_warmstart_smoke.py`, a no-Miles HF-native AV
  warm-start smoke that keeps the reference NLA Stage 3/injection contract:
  `activation_vector h`, marker substitution, sidecar metadata, and target text
  `<explanation>{z}</explanation>`.
- Added AV smoke harness tests covering control construction, config loading,
  response-only SFT masking, and trainable-parameter selection.

## Historical Subtractions And Deferrals (Initial Phase)

- Synthetic AR data is no longer evidence of NLA signal; keep it only as a
  regression harness.
- Do not generate a large dataset until real-data smoke gates pass.
- Do not start PEFT/LoRA until sidecar-complete Nano Stage 3 data exists and
  frozen AR/probe controls show signal.
- Do not start serving work.
- Do not start RL.
- Do not treat Track C as serving.
- Do not treat teacher summaries as ground truth; they are warm-start labels
  only.
- Do not require Miles for the current AV smoke. Miles remains training
  infrastructure for exact upstream parity and later scale/RL work, not a
  blocker for preserving the NLA data and injection contract now.
- Do not use common English tokens such as `a`, `an`, or `the` as AV injection
  markers.

## Current Validation Status

| Item | Status | Evidence or next requirement |
|---|---|---|
| Clean AV validation | complete | Hash-bound 512-row verifier passes: real NLL `0.776775` beats shuffled, zero, mean, and no-injection controls |
| Protocol-matched clean SFT round trip | complete | Validation/test directional MSE `0.307004 / 0.319225`, `100%` usable, all family-clustered controls pass |
| Stored-snapshot functional recovery | complete | Validation/test verifier passes; candidate is teacher-level within family uncertainty and beats mean, zero, and shuffled controls |
| Qualified checkpoint-pair manifest | complete | Release `r33-clean-sft-av-ar-iter1291-20260715` binds exact AV/AR fingerprints and all six passing verifiers |
| Full selected-pair exposure audit | complete, negative boundary | All `5,009` canonical families are exposed across selected-pair train/eval and historical eval sources; external teacher-backed data is required for confirmatory replication |
| Fresh-forward fidelity | exact identity failed | 64-row validation audit is repeatable and highly aligned (mean cosine `0.999142`) but all strict stored-vs-fresh identity checks fail |
| Magnitude calibration | complete, exploratory | Validation teacher fit selected origin scalar `0.560604`; exploratory-test AV centered raw R2 becomes `0.478102`, without changing directional MSE |
| Independent critic/AR | The retained reseed critic is not an independent publication critic | Rebuilt seed-`314159` init, exact manifest hash, independently trained clean AR checkpoint, and validation report |
| RL stability and preregistration | Guard policy and finite validation-only grid are not yet finalized | Complete the four-probe grid, power check, immutable guard/config hashes, and registered analysis plan |
| Pristine external test and replication | The family-disjoint test was consumed once but is not guaranteed untouched by all historical project work | Freeze a new external boundary before a stronger publication-generalization or RL claim |
| Fresh row-matched R27 comparison | Required only for an external R33-over-R27 claim | Protocol-matched R27 data, actor/critic, rows, controls, parser, and clustered statistics |
| Qualitative/task validation | Reconstruction metrics do not establish explanation truthfulness or usefulness | Frozen-output factuality checks, blinded review, and downstream task evidence |

Canonical live state: `docs/current_state.md`; the qualified SFT pair is
`docs/runs/r33_clean_sft_av_ar_20260715.md`. The future confirmatory RL protocol
is `docs/runs/r33_publication_preregistration.md`; the July 8 hero document is
historical only.

## Historical Blockers (2026-05 Snapshot)

| Blocker | Impact | Next action |
|---|---|---|
| R_34 AR does not show heldout explanation signal | Head-only, tail-1, tail-2, row-random, longer-tail, old-parquet prompt gate, exact Haiku gate, and exact Kimi gate all fail heldout train-mean controls | Do not launch 10%+ AR SFT yet; debug the explanation-to-residual channel before scaling rows. |
| Teacher explanations fail blank/generic/mean controls | Haiku exact teacher NMSE `1.1089` vs mean `0.8595`; Kimi exact teacher NMSE `1.0729` vs mean `0.9595` on a smaller diagnostic slice | Inspect whether the critic prompt should condition on actual extracted explanation text differently, whether activations should be normalized/projected before the AR head, and whether the target boundary should move to `R_27`. |
| Exact provenance is now confirmed for regenerated rows | Exact Haiku and Kimi gates both report `exact_token_prefix_fraction = 1.0`; source_raw oracle remains strong | Treat missing `token_ids_prefix` as resolved for new data, but keep the gate requirement for all scale runs. |
| AV free generation collapses after tiny warm-start | Teacher-forced NLL shows row-specific signal, but one sampled generation produced only `<` tokens for real/shuffled/zero/no-injection controls | Next smoke should use 16-64 rows and slightly more actor capacity, for example `lm_head+embeddings` or a small adapter/tail subset, while preserving the same controls. |
| AV heldout evidence is still tiny | The 50-step run used 8 rows with 7 train and 1 heldout | Repeat on 16-64 rows with multiple heldout rows before treating the AV signal as robust. |

## Historical Near-Term Work Queue (2026-05 Snapshot)

1. Do not start 10% -> 30% -> 50% -> 75% -> 100% AR SFT scaling yet. The
   current AR teacher/prompt channel fails the required heldout controls even
   with exact provenance and Kimi reasoning enabled.
2. Keep the paper-faithful loop centered on `x -> Nano -> h -> AV -> z -> AR`.
   Teacher `z` from Super/Haiku is a supervised AV warm-start label, not final
   proof of NLA success.
3. Run the next AV warm-start on 16-64 rows with the same real/shuffled/zero/
   mean/no-injection controls and a small capacity increase beyond `lm_head`
   only.
4. Treat the Nano hybrid geometry as handled for AV by full-model injection:
   inject `h` into the embedding stream at the marker, then let the full Nano
   forward handle Mamba/MoE/attention internals. Do not introduce a truncated
   Nano AR critic until AV decoding is cleaner.
5. Resume AR closure only after AV-generated explanations become non-degenerate
   and row-specific under heldout controls.

## Recent Run: R_34 Trainable-Tail Capacity Probe

Cluster run:

```text
runs/introspection/ar-capacity-r34-tail1-haiku-singlenode-20260519T204033Z/
job 28042647, interactive_singlenode, 1x A100 80GB, 32G CPU memory
```

Configuration:

```text
boundary: R_34
records: 256 total, 192 train, 64 heldout
split: doc_random, 0 doc overlap
trainable Nano layers: .backbone.layers.33 only
trainable Nano params: 23,399,040
trainable value-head params: 7,225,344
steps: 80
batch size: 4
head_lr: 1e-3
tail_lr: 1e-5
```

Outcome:

```text
passed: true
scientific_passed: false
blockers: []

train correct normalized MSE: 1.3703 -> 0.7891
train correct cosine: 0.3149 -> 0.6055
train shuffled normalized MSE: 0.9476
train mean-target normalized MSE: 0.8938

heldout correct normalized MSE: 0.9674
heldout correct cosine: 0.5163
heldout shuffled normalized MSE: 0.9747
heldout shuffled cosine: 0.5127
heldout mean-target normalized MSE: 0.8919
heldout mean-target cosine: 0.5540
```

Interpretation:

- The implementation path is trainable: normalized train loss decreases
  monotonically and train correct beats shuffled/random/mean controls.
- The heldout split does not show explanation-conditioned reconstruction yet:
  correct is only slightly better than shuffled and worse than the train-mean
  baseline under normalized MSE/cosine.
- Raw MSE is not the primary reference objective because the NLA loss normalizes
  vectors to `sqrt(d)`, but the large raw-MSE gap should still be tracked as a
  norm-calibration diagnostic.

## Recent Run: R_34 Capacity Ablation Bundle

Cluster run:

```text
runs/introspection/ar-capacity-r34-diagnostics-haiku-20260519T210032Z/
job 28043387, interactive_singlenode, 1x A100 80GB, 32G CPU memory
elapsed 00:08:56, MaxRSS 1,936,928K
```

All four runs exited with code `0` and wrote result JSON:

| Run | Split | Tail blocks | Steps | Train NMSE | Heldout correct NMSE | Heldout shuffled NMSE | Heldout mean NMSE | Scientific pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `r34-head0-docrandom-80` | doc_random | 0 | 80 | `0.7992` | `0.9708` | `0.9779` | `0.8919` | false |
| `r34-tail1-docrandom-400` | doc_random | 1 | 400 | `0.2103` | `1.1277` | `1.1653` | `0.8919` | false |
| `r34-tail1-rowrandom-80` | random | 1 | 80 | `0.7859` | `0.9484` | `0.9818` | `0.8904` | false |
| `r34-tail2-docrandom-80` | doc_random | 2 | 80 | `0.7621` | `0.9603` | `0.9697` | `0.8919` | false |

Interpretation:

- The value head alone explains nearly all of the 80-step train improvement:
  head-only train NMSE `0.7992` is close to tail-1 `0.7891`/`0.7859`.
- Longer training overfits: tail-1 400-step train NMSE reaches `0.2103`, but
  heldout correct worsens to `1.1277`.
- Row-random splitting with heavy doc overlap improves heldout slightly over
  doc-heldout, but still loses to the train-mean baseline.
- Tail-2 improves train fit slightly, but does not create heldout
  explanation-conditioned separation.
- Scaling should therefore be gated on data/teacher signal, not on adding AR
  capacity. The next scale run should be a 10% data-generation and signal gate
  only if it includes exact provenance and control prompts.

## Recent Run: R_34 Prompt Signal Gates

Old-parquet Haiku gate:

```text
runs/introspection/ar-signal-gate-r34-haiku-20260519T213303Z/
job 28044029, interactive_singlenode, 1x A100 80GB, 32G CPU memory
elapsed 00:08:31, MaxRSS 1,700,620K
```

Key result: `source_raw` oracle passed with feature normalized MSE `0.002116`
and cosine `0.99894`, but Haiku teacher explanations failed controls:
teacher NMSE `1.1437` vs mean `0.8919`, blank `0.8916`, generic `0.8913`.
This old parquet had no `token_ids_prefix`, so source_raw used text
re-tokenization and exact-provenance regeneration was required.

Exact-provenance Haiku gate:

```text
runs/introspection/exact-r34-haiku-signal-20260519T215409Z/
stage0 job 28044572, cpu/API job 28044573, signal job 28044574
stage0 rows: 256/256 with token_ids_prefix
stage2/stage3 rows: 254/256 kept, 254/254 token_ids_prefix and api_explanation
signal gate elapsed: 00:07:47, MaxRSS 1,920,552K
```

Outcome:

```text
passed: true
scientific_passed: false
exact_token_prefix_fraction: 1.0
warnings: []
blockers: []

teacher heldout normalized MSE: 1.1089
teacher heldout cosine: 0.4456
teacher shuffled normalized MSE: 1.2060
blank normalized MSE: 0.8599
generic normalized MSE: 0.8597
source context normalized MSE: 1.0743
train-mean normalized MSE: 0.8595
source raw feature normalized MSE: 0.003019
source raw feature cosine: 0.99849
```

Exact-provenance Kimi reasoning gate:

```text
runs/introspection/kimi-exact-r34-32-signal-20260519T221433Z/
cpu/API job 28044907, signal job 28044908
source rows: first 32 exact rows from the Haiku exact run
stage2/stage3 rows: 31/32 kept, 31/31 token_ids_prefix and api_explanation
Kimi provider: nvidia/moonshotai/kimi-k2.6, max_tokens=8192,
extra_body.chat_template_kwargs.thinking=true
signal gate elapsed: 00:01:46, MaxRSS 1,480,576K
```

Outcome:

```text
passed: true
scientific_passed: false
exact_token_prefix_fraction: 1.0
warnings: []
blockers: []

teacher heldout normalized MSE: 1.0729
teacher heldout cosine: 0.4636
teacher shuffled normalized MSE: 1.1460
blank normalized MSE: 0.9578
generic normalized MSE: 0.9570
source context normalized MSE: 1.0701
train-mean normalized MSE: 0.9595
source raw feature normalized MSE: 0.001743
source raw feature cosine: 0.99913
```

Interpretation:

- The exact-provenance path is now mechanically clean for regenerated data.
- Kimi reasoning API parsing is fixed enough for Stage 2: the 32-row diagnostic
  kept `31/32` rows, compared with the earlier broken Kimi run where responses
  lived in `reasoning_content` and `content` was null.
- Both Haiku and Kimi teacher prompts beat shuffled-teacher controls but lose to
  blank/generic and train-mean controls. This is not a scale problem.
- Raw source-prefix replay remains a near-perfect oracle, so the residual target
  and source-token provenance are not the immediate blockers.
- The next diagnostic should test `R_27` and audit the AR target/head/template
  math before launching any 10%+ AR SFT run.

## Recent Run: Qwen-Faithful Nano AV Warm-Start Smoke

Scientific pivot:

```text
old AR-only proxy: x -> teacher -> z -> AR -> h_hat
paper loop:        x -> Nano -> h -> AV -> z -> AR -> h_hat
current smoke:     h -> AV -> z
```

Teacher explanations from Super/Haiku are used here only as supervised
warm-start labels for AV. They are not treated as final proof of NLA success.

Implementation:

```text
script: scripts/nano_av_warmstart_smoke.py
tests: tests/test_nano_harness.py
reference path reused: nla.injection.inject_at_marked_positions plus Stage 3
target format: <explanation>{z}</explanation>
marker: 々
marker token id: 42019
injection scale: 150
Miles: intentionally not required for this smoke
AR: not used
```

The script builds/reuses AV-SFT parquet rows with `activation_vector h` and
teacher explanation `z`, preserves NLA sidecar metadata, substitutes the marker
with injected embeddings, masks prompt tokens, and trains only the response
tokens with a HF-native loop. This is an AV-only warm-start; it does not use a
truncated Nano AR critic and does not launch RL, LoRA/PEFT, serving, or large
dataset generation.

Cluster base-likelihood run:

```text
run: runs/introspection/av-warmstart-r27-super-hf-20260521T232100Z/
sbatch: runs/introspection/av-warmstart-r27-super-hf-20260521T232100Z/av_warmstart_hf_eval.sbatch
job: 28115623
state: COMPLETED
exit: 0:0
elapsed: 00:10:10
input: runs/introspection/exact-r27-super-thinkingfalse-signal-20260520T210327Z/ar_sft_super_explained.parquet
rows: 209 input, 209 usable, 16 selected, 0 dropped
split: 12 train, 4 heldout
```

Base-model teacher-forced NLL before AV training:

| Split | Real h | Shuffled h | Zero h | Mean h | No injection |
|---|---:|---:|---:|---:|---:|
| train | `2.5555` | `2.6335` | `2.6288` | `2.6229` | `2.7105` |
| heldout | `2.8191` | `2.9051` | `2.8787` | `2.8966` | `2.9548` |

Interpretation: even without training, real `h` is a weak but consistent
likelihood control winner over shuffled/zero/mean/no-injection.

One-step no-Miles training smoke:

```text
run: runs/introspection/av-warmstart-r27-super-hf-train1-20260522T000300Z/
sbatch: runs/introspection/av-warmstart-r27-super-hf-train1-20260522T000300Z/av_warmstart_hf_train1.sbatch
job: 28116548
state: COMPLETED
exit: 0:0
elapsed: 00:02:21
rows: 8 selected, 0 dropped, 6 train, 2 heldout
trainable subset: lm_head
trainable params: 352,321,536 / 31,577,937,344
```

Heldout teacher-forced NLL after one update:

| Real h | Zero h | Mean h | Shuffled h | No injection |
|---:|---:|---:|---:|---:|
| `3.1278` | `3.2872` | `3.3177` | `3.3536` | `3.5550` |

Interpretation: the constrained HF-native training path executes, and real `h`
still beats all controls after an optimizer step.

Fifty-step no-Miles `lm_head` overfit smoke:

```text
run: runs/introspection/av-warmstart-r27-super-lmhead50-genshort-20260522T014000Z/
sbatch: runs/introspection/av-warmstart-r27-super-lmhead50-genshort-20260522T014000Z/av_warmstart_hf_lmhead50_shortgen.sbatch
job: 28119255
state: COMPLETED
exit: 0:0
elapsed: 00:06:47
rows: 209 input, 209 usable, 8 selected, 0 dropped
split: 7 train, 1 heldout
train steps: 50
batch size: 1
lr: 1e-4
trainable subset: lm_head
trainable params: 352,321,536 / 31,577,937,344
```

Training history:

| Step | Batch row | Loss | Grad norm |
|---:|---:|---:|---:|
| 1 | 1 | `2.9589` | `15.1875` |
| 10 | 6 | `2.9805` | `17.2500` |
| 20 | 0 | `2.7972` | `14.6250` |
| 30 | 1 | `1.5263` | `11.0625` |
| 40 | 3 | `0.8212` | `7.6250` |
| 50 | 5 | `0.9765` | `9.8750` |

Teacher-forced NLL after 50 steps:

| Control | Train | Heldout | All |
|---|---:|---:|---:|
| real h | `0.9833` | `2.6669` | `1.1937` |
| zero h | `1.0722` | `2.8673` | `1.2965` |
| shuffled h | `1.1219` | `2.9215` | `1.3469` |
| mean h | `1.0740` | `2.9260` | `1.3055` |
| no injection | `1.3079` | `3.1209` | `1.5345` |

Heldout real-vs-control gaps:

```text
real vs zero:       0.2003 NLL
real vs shuffled:   0.2546 NLL
real vs mean:       0.2591 NLL
real vs none:       0.4540 NLL
```

Qualitative generation sample:

```text
row: 0
doc_id: HuggingFaceFW/fineweb:train:0
target excerpt: The snippet begins and ends with pipe-delimited metadata lines...

real:        <<<<<<<<<<<<<<<<<<<<<<<<
shuffled:    <<<<<<<<<<<<<<<<<<<<<<<<
zero:        <<<<<<<<<<<<<<<<<<<<<<<<
no injection:<<<<<<<<<<<<<<<<<<<<<<<<
content F1:  0.0 for all controls
```

Interpretation:

- AV warm-start is mechanically viable without Miles when preserving the NLA
  injection/data contract.
- Teacher-forced loss gives the first positive AV row-specific signal: real
  `h` is best on heldout versus shuffled, zero, mean, and no-injection controls.
- Free generation is not yet viable; the sampled decoding collapses to repeated
  `<` tokens for all controls.
- The next blocker is actor decoding/capacity under a small AV warm-start, not
  cluster execution, Qwen artifact production, Miles installation, or Nano
  hybrid geometry.
- The Nano hybrid geometry is avoided for this AV smoke by injecting into the
  full model embedding stream and letting full Nano forward handle the hybrid
  Mamba/MoE/attention internals.

## 2026-05-26 RunAI 100k AV rsLoRA Artifact

Completed artifact run:

```text
run: /workspace/interp/outputs/nano30b-nla-pilot/av-r27-100k-rslora-artifact-v1/av-r27-99570-rslora-r192-broad-scale75-lr1e5-s800-save-gen8-2gpu-offline-20260526T1645Z
input: /workspace/interp/artifacts/nano30b-nla-pilot/super-teacher-r27-100k-thinking-merged-20260525T2150Z/base_R27_super_thinking_99570_explained.parquet
rows: 99,570
split: doc, 89,604 train / 9,966 heldout / 0 doc overlap
config: rsLoRA r192 alpha384, q/k/v/o/in/out/up/down, scale75, lr1e-5, 800 steps, batch1
trainable: 5,303,242,752 params, 14.38%
trainable_state: 20G
wandb: https://wandb.ai/rishabhga97/nano30b-nla-pilot/runs/ggrd168y
blockers: []
```

Heldout teacher-forced NLL:

| Control | Loss |
|---|---:|
| real h | `1.1707` |
| zero h | `1.4172` |
| mean h | `1.4671` |
| no injection | `1.4797` |
| shuffled h | `1.5454` |

Heldout real-vs-control gaps:

```text
real vs zero:       0.2465 NLL
real vs mean:       0.2964 NLL
real vs none:       0.3090 NLL
real vs shuffled:   0.3747 NLL
```

Subagent analysis of W&B/results:

- W&B adds useful provenance and train-curve traces, but the JSON report remains
  the authoritative source for final eval/control metrics.
- The train curve dropped quickly from about `2.3964` at step 1, reached a
  minimum around `0.8475` near step 375, then oscillated to `1.2656` at step
  800. The positive heldout gaps do not look like simple train-doc memorization.
- Generation remains a separate blocker, but it is no longer the earlier
  repeated-`<` collapse: real-control generations closed `</explanation>` in
  `7/8` examples with mean content F1 `0.4466`, versus shuffled `0.3288`,
  zero `0.2842`, and no-injection `0.2770`. No-injection closed `0/8`.
- Repeatability should standardize control NLLs/gaps, generation parse/content
  metrics, data integrity counts, train-curve summaries, parameter counts, and
  exact W&B/run artifact provenance.

Repeatability additions:

```text
scripts/run_nano_av_100k_rslora_runai.sh
scripts/summarize_nano_av_run.py
scripts/fetch_runai_wandb_offline.sh
docs/runai_av_100k_repeatability.md
```

Follow-up note: an initial `lr=2e-5` HPO launch at
`av-r27-99570-rslora-r192-broad-scale75-lr2e5-s800-save-gen8-2gpu-offline-20260526T1904Z`
accidentally inherited the AV smoke default `--row-limit 32`; it is useful only
as a launcher smoke and must not be compared as a 100k run. The repeatable
launcher now pins `ROW_LIMIT=99570`, and the corrected full-data `lr=2e-5` run
is:

```text
run: /workspace/interp/outputs/nano30b-nla-pilot/av-r27-100k-rslora-hpo-v2/av-r27-99570-rslora-r192-broad-scale75-lr2e5-s800-save-gen8-2gpu-offline-full-20260526T2006Z
pid: 68317
row_limit: 99,570
```

## Handoff Commands

Start on the cluster:

```bash
tmux attach -t nano30b-nla
cd /lustre/fsw/portfolios/llmservice/users/rigarg/mech_interp/research-projects/nano30b-nla-pilot
source scripts/cluster_nano_env.sh
export PYTHONPATH="$PWD/external/natural_language_autoencoders:${PYTHONPATH:-}"
```

Validate local docs/scripts before handoff:

```bash
python -m py_compile scripts/nano_realdata_stage0_extract.py scripts/nano_realdata_ar_build.py scripts/nano_realdata_stage3_build.py
python -m unittest discover -s tests -p 'test_nano_harness.py'
```

## 2026-07-16: Publication robustness packet and independent AR launch

- Added and tested a validation-fitted subgroup auditor over frozen round-trip
  prediction caches. All 16 validation/test bins have sufficient rows and
  families, and all registered family-clustered control intervals remain
  positive. The weakest exploratory-test slice is the lowest target-activation
  norm quartile: directional MSE `0.370077`, calibrated centered raw R2
  `0.415606`.
- Built a deterministic 50-validation/50-test panel containing source text,
  teacher explanation, and AV generation. The automatic structural screen
  reports zero flags. Blinded human semantic review remains pending.
- Rebuilt the seed-`314159` independent critic initialization. The first
  byte-level manifest gate failed only because it included a transient random
  pre-initialization head hash. The queue now supports canonical JSON hashing
  with an explicit ignored-path allowlist; all final-state comparisons remain
  mandatory. The canonical manifest SHA-256 is
  `34e863f756e0749ca19fc8c138b7bd71b5da69c907ee42ad021517542e5c8941`.
- The independent critic verifier passes all 16 checks; report SHA-256 is
  `4639285fea694f7f850c766b31d8ddea4e2b2bdd61c710f3b2a4cdd2109fb6e3`.
- Launched the clean independent AR replication with seed `314159`, four
  H100-NVL GPUs, `lr=5e-5`, warmup 25, `gb192/mb48`, cosine decay, 1,291
  updates, validation-only selection, and offline W&B. This launch is not a
  result; qualification waits for optimizer completion, frozen validation
  scoring, and verifier success.
- The independent AR subsequently completed all 1,291 updates. Final
  loss/FVE-NRM were `0.295026 / 0.472163`. Its validation-only 512-row
  component verifier passes with teacher directional MSE/cosine/FVE-NRM
  `0.286169 / 0.856916 / 0.577948`; report SHA-256 is
  `368c84cad8bb0b8a7235b1a1e96c862b69d33be72d49c1f07a563e62d0be65aa`.
  Frozen AV-text to independent-AR scoring was then launched as the remaining
  cross-critic gate.
- Local full suite after the code changes: `721 passed, 1 skipped, 162
  subtests passed`. RunAI focused suite: `21 passed`.
- Immutable source snapshot:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/source-sync/publication-work/20260716T035347Z/nano30b-nla-pilot-source.tgz`,
  SHA-256
  `700b6690ac32d8c6835891af5e7769f402c1b7047eb818777cf547522133555e`.

## 2026-07-16: Independent cross-critic qualification and release hardening

- Frozen AV-generated validation text was scored through the independently
  initialized and trained seed-`314159` AR. The 512-row, 250-family verifier
  passes: AV-text directional MSE `0.310963`, teacher directional MSE
  `0.308533`, closed/usable fractions `1.0 / 1.0`, and all five controls pass.
  Rowwise wins are `99.80%` against mean and `100%` against the other controls.
- Cross-critic report/verifier SHA-256 values are
  `6f0829a61b03ac584b109c1c7a54f689f0a14b61f7fee803afd3d5ca29bf552b`
  and
  `dd3de6e1dd10f0f64c25e23ff3152638c9cf46785133dfb6c0466c353e434331`.
  The independent centered raw R2 remains negative, so the qualified scope is
  directional reconstruction only.
- The independent HF checkpoint was fingerprinted at 10 files and
  `38,462,226,688` bytes, then mirrored to S3 at the exact object count and
  byte total. Directory SHA-256 is
  `c2eea74f5baccee97128617b05636187804c7e59aedc560d088dbf65d52f1925`.
  Manifest-first retention then removed only the redundant 36G model DCP and
  72G optimizer directory; HF, run logs, eval, split metadata, and offline W&B
  were retained.
- Fixed the qualitative evidence resolver to load
  `detokenized_text_truncated` and made the panel builder fail closed on empty
  source. The old source-empty panel was invalidated. Corrected panel and
  structural-review hashes are
  `4f5d61486330b1104dd0a256ea185d8c1c99512ee9ff4731f8135305924f81c8`
  and
  `33de2720d96bda3f663318be7fe8c10765740a3d603bcdf76682ca23661297da`.
- Added two deterministic blinded reviewer packets plus a separate hash-bound
  answer key. Human ratings remain pending; answer-key SHA-256 is
  `27585eae51d55deb9bb3821afbd1f5d1d3e7cfd0e7c4167e4111566ff06c1856`.
- Automatic release-text triage passes on all 1,024 frozen generations with
  zero configured sensitive findings and zero source-copy failures. Fourteen
  phone-like patterns occur only in source excerpts, which remain internal
  pending adjudication/redaction. Report SHA-256 is
  `00e501ff644483e614d0b60071f726f33575c95ba8b81d5c6b59c4bd79d13419`.
- Lightweight primary and independent evidence is mirrored through S3 and
  local under `artifacts/runai_eval/r33-clean-sft-publication-evidence-20260716/`
  and `artifacts/runai_eval/r33-independent-ar-publication-evidence-20260716/`.
- Added a terms/provenance inventory, draft NOTICE, and human-review handoff.
  Public redistribution remains blocked on exact teacher-service terms,
  owner/legal approval, repository-license selection, staged-bundle security
  review, and completed blinded semantic ratings.
- Added config-driven, hash-bound compute accounting. Primary AR, primary AV,
  and independent AR each contain all 1,291 optimizer steps and total
  `138.7456` H100-NVL GPU-hours. Report SHA-256 is
  `7bde74be3a874d2ae305463ca8da211c069ce0bf1001802b6bdf7ab091fd7238`;
  exclusions are explicit rather than estimated.
- Added a no-values-emitted source-bundle security scanner. The source-only
  internal snapshot has zero heavy/checkpoint files, symlinks, binaries,
  oversized files, or unallowlisted credential fixtures, but fails public
  release on 31 files with local-home, internal S3/endpoint, or cluster
  references. Audit SHA-256 is
  `bf697acdad268e219cfd729428233b1259b07aea988529e8c7e4bb01b8a62c42`.
- Built a deterministic redacted public-bundle candidate with release-relevant
  source, configs, tests, aggregate evidence, family manifests, and compact
  curves for all 3,873 selected optimizer steps. The exact 496-file tree
  passes static security review; tree SHA-256 is
  `df175c5f61cefbfc1a02451a7bd242ba69e1cb602cdd97ca4b8bd8fe9c263b77`.
- Packaged and re-read the audited tree as a normalized 6,859,370-byte archive.
  Archive SHA-256 is
  `3eb8e64ed0d9d61ed2d6b0694fbaf96b99051a63f2ce1a6c99372d93832e573a`;
  its content tree exactly matches the audit. The attestation remains
  fail-closed with `weights_included=false` and
  `legal_clearance_granted=false`.
- Final local verification after the publication-bundle tooling and current
  claim-contract updates: `749 passed`; `scripts/verify_docs_consistency.py`
  also passes.
- Preserved the candidate archive, SHA-256 sidecar, bundle manifest, security
  report, and attestation as five internal S3 objects under
  `publication/release-candidates/r33-clean-sft-av-ar-iter1291-20260716/`;
  the remote archive listing matches the expected `6,859,370` bytes.
- Reconciled all active documentation to the July 16 state. Added an exact
  human-readable release-candidate attestation and updated current state, run
  history, job tracker, gate matrix, model card, release checklist, registry,
  and README. Historical pending states remain only where explicitly labeled
  as superseded audit chronology.

## 2026-07-16 PT / 2026-07-17 UTC: RunAI Compute Released

- Confirmed there was no active Nano AV, AR, RL, evaluation, queue, or
  promotion-watcher process and that all eight H100-NVL GPUs were idle before
  teardown.
- No additional training is mandatory for the bounded claim attached to the
  qualified family-clean R33 SFT AV+AR checkpoint pair. A second AV seed,
  row-matched R27 baseline, external-data confirmation, and clean RL remain
  optional experiments required only for their respective stronger claims.
- Suspended the RunAI `train` workspace at `2026-07-17T03:47:21Z`. RunAI then
  reported phase `Stopped`, no pod, and `0.00` allocated GPU and CPU resources.
- This was a compute release, not destructive storage deletion. The
  `interp-gh200-dev-lh` and `model-store-pvc` PVC definitions remain in the
  workspace specification, and selected checkpoints/evidence remain preserved
  on PVC/S3 according to their manifests.
- Resume command, if a scoped follow-up is approved:
  `/Users/rigarg/.runai/bin/2.116.9/runai workspace resume train -p trustworthy-ai-inference`.

## 2026-07-17: Online Joint Canary Recovery And Evaluation Closure

- Resumed the accidentally suspended eight-H100 `train` workspace and completed
  two online actor and critic updates without OOM or optimizer failure.
- Fixed post-eval model/tokenizer identity propagation and stable row-key
  baseline alignment; added config-driven report-only re-gating.
- Generated and scored an exact 64-row clean-SFT comparator with identical
  dataset and generation protocol provenance.
- The online pair has directional MSE `0.291993` versus SFT `0.292173`, but the
  effect is statistically indistinguishable from zero and raw MSE regresses.
  The strict promotion report fails, so no RL checkpoint was promoted.
- Synced 71 lightweight evidence files locally and excluded actor/critic model
  shards. Local evidence is under
  `artifacts/runai_eval/r33-online-joint-canary-evidence-20260717T0951Z/`.
