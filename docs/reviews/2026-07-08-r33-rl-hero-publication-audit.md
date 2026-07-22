# R33 RL Hero Publication Audit

<!-- R33-HERO-BASELINE-PROTOCOL-INVALIDATED -->

- Audit received: `2026-07-08`
- Evidence root: `artifacts/runai_evidence/20260708T172617Z_complete_lightweight_evidence/`
- Status: external audit received; critical findings pending remediation
- Publication effect size: pending a protocol-matched SFT baseline

The text below is preserved verbatim from the independent audit output.

---

I'll audit this as requested: first scout the core docs and evidence layout inline, then fan out verification agents via a workflow over configs, code, logs, and the W&B/evidence payloads.

Still running — let me peek at the workflow's progress without interrupting it.

The workflow completed — 25 agents, 1,135 tool calls. Now let me extract the full structured results from the output file to build the final report.

The verification wave confirmed nearly everything. Now let me pull the completeness critic's findings and the remaining findings by severity.

# R33 RL Hero Audit — Final Report

**Verdict up front:** The run is real, the arithmetic is honest, and the reports reproduce bit-exactly from raw payloads — but the headline is not defensible as stated. The 30.97%/32.34% improvement is inflated by a corrupted SFT baseline (the fair number is **~23.6%/22.5%**), it is a **same-critic** result (reward model = eval scorer; cross-critic transfer was verified only at update-16, never for the promoted checkpoint), and the "normalized MSE" is a **pure direction (cosine) metric** — in raw hidden-state space the round trip reconstructs approximately nothing beyond the mean (centered R² ≈ −0.004/−0.051). Additionally, the promoted run only exists because stop-guards were relaxed after each failed attempt; replayed telemetry shows retry3 would have been aborted under every earlier guard configuration.

Method: 11 parallel auditors recomputed every statistic from raw JSONL/reports/logs, audited evaluator/trainer/dataset code in the frozen source snapshot, parsed all four attempts' train.logs and W&B payloads, and reconstructed the full lineage. Every critical/high finding was then re-derived by an independent adversarial verifier (12/13 confirmed or partially confirmed, 0 refuted), followed by a completeness pass. ~3.3M tokens, 1,135 tool calls, all against `artifacts/runai_evidence/20260708T172617Z_complete_lightweight_evidence/`.

---

## 1. Severity-ordered correctness findings

### CRITICAL

**C1. The matched SFT baseline is corrupted on half its rows, inflating the headline from ~23.6%/22.5% to 30.97%/32.34%.** *(confirmed by adversarial re-derivation)*
The hardened 512-row baseline (`validity/r33-sft/roundtrip_v512_t512_hardened_report.json`) is two populations: rows 0–255 per split byte-reuse the June v256 generations (prefix `"<explanation>\n"`), while rows 256–511 were generated 2026-06-30 under [r33_stage1_sft_v512_generate.yaml:17](artifacts/runai_evidence/20260708T172617Z_complete_lightweight_evidence/remote_source/nano30b-nla-pilot-hero-91e121d-clean/configs/nano_roundtrip/r33_stage1_sft_v512_generate.yaml) with `generation_prefix: "<explanation>"` — **missing the newline** used everywhere else. 0/512 extension rows start with the standard prefix; ~40% are degenerate (`<explanation>20</explanation>`, timestamp spam, etc.). Extension-half SFT NMSE is 31%/21% worse than the clean half although teacher NMSE proves the halves equally hard. Split by half: clean-half improvement **23.56%/22.52%** (win rates 81.6%/87.1%), corrupted-half 36.6%/40.4% (win 85.2%/90.2%). The RL hero itself was evaluated with *no* forced prefix, so the asymmetry disadvantages only the baseline. The gate direction and its predeclared 10% threshold survive; the published magnitude does not. Corollary: [docs/rl_logbook.md](docs/rl_logbook.md)'s explanation that "the first 64 rows were an unrepresentative low-error slice" is a misdiagnosis of this artifact, and the update-16 promotion diagnostic (14.53%/14.86%) inherits the same baseline.

### HIGH

**H1. The headline metric is same-critic, and no cross-critic evaluation of the promoted checkpoint exists.** *(confirmed)*
The 512/512 eval scored reconstruction with AR-SFT `iter_0001289` — the exact frozen checkpoint that supplied the RL reward (report `ar_checkpoint_dir` = queue `critic_sl_ckpt` = train.log `CRITIC_SL_CKPT`). Scoring is symmetric (baseline scored by the same critic on identical rows), so the paired comparison is internally valid, but it cannot exclude reward overfitting. A second critic existed since 2026-07-05 and the only cross-critic result in the entire lineage is for **confirm update-16**: 14.53%/14.86% (primary) vs 14.82%/14.87% (independent), transfer ratio ~1.0 — a genuine anti-reward-hacking positive. The hero's ~2× larger effect after **21× more reward-optimization pressure** was never scored by any critic it wasn't trained against, despite the retained independent critic making that a cheap rescoring job. Also: the "independent" critic is a same-recipe reseed (same init, same parquet, same split seed 42; only data-order seed 314159 and GPU layout differ), so even the update-16 transfer is weaker evidence than the name implies. The Claim Boundary sections in [docs/runs/r33_rl_hero_20260708.md](docs/runs/r33_rl_hero_20260708.md) and [docs/current_state.md](docs/current_state.md) never disclose either fact.

**H2. "Normalized MSE" is a magnitude-blind cosine metric; raw-space reconstruction is ≈0.** *(confirmed numerically)*
The evaluator L2-normalizes both prediction and target before MSE (`eval_nano_av_ar_roundtrip_gate.py:85-93`), making NMSE ≡ (2−2·cosine)/2688 exactly (verified to ±0.001 across all variants/splits). The headline 30.97% is arithmetically a rescaled **+0.053 mean-cosine gain** (0.8296 → 0.8824). In raw space, the promoted checkpoint's centered R² is **−0.004 (val) / −0.051 (test)** and raw MSE (7.097/7.268) is *worse than the constant train-mean predictor* (7.068/6.916). No claim doc discloses raw-space performance or the metric's direction-only nature. This also resolves the long-standing ~1e-4 vs ~0.3 metric-scale confusion — they are different normalizations entirely. Any downstream claim of "reconstructing h" must be read as "recovering the direction of h."

**H3. Guard relaxation was load-bearing: the promoted run survives only under the final, weakest guard set.** *(confirmed by config diff + full log replay)*
The four hero configs differ *only* in guards and resume mechanics (recipe identical). Attempt 1 died at step 63 (KL>5 ×2 rule); retry1 died at rollout 26 (relative p95 rule); retry2 died at rollout 253 (absolute p95>230 ×2); retry3 relaxed KL to ×3-consecutive and **deleted the length rule entirely** ([retry3 queue:220](configs/nano_rl/r33_component_corrected_k3_hero_342_resume228_retry3_queue_8h100.yaml): `approved_by: user-remove-length-guard-and-finish-20260707`). Replaying retry3's own telemetry: KL>5 at 10/114 steps with consecutive pairs at 326–327 and 332–333 (would have died under the original ×2 rule), and p95>230 on 25/114 rollouts, max 238 (would have died under retry2's rule at rollouts 264–265). Each relaxation had a documented engineering rationale (heavy-tailed KL, zero truncation), but the net effect is that the completion criterion was fitted to whatever the run did. Mitigation: the *promotion gate* itself was predeclared and independent of the guards.

**H4. The NO-GO → GO conversion ran through a post-hoc, evaluation-only escalation.** *(partially confirmed — facts exact, interpretation nuanced)*
The 2026-07-02/03 assessment was "statistically null vs SFT, hero blocked" ([docs/experiment_logbook.md:5063](docs/experiment_logbook.md)). The confirmation run's own configured 64/64 gates then **failed** (`gate.passed=false`, CIs crossing zero) and its 512/512 check skipped by design. GO came from a subsequently authored eval-only 512/512 diagnostic of update-16 that passed at 14.5%/14.9% — against the (weaker, half-corrupted; see C1) hardened baseline. The re-eval was row-matched, predeclared its 10% threshold, and is scientifically reasonable as a larger-sample re-test; but it was an escalation invented after the configured gate failed, and its passing margin depends on the corrupted baseline half.

**H5. Eval-set independence is overstated ~4×, and near-duplicates straddle train/heldout.** *(confirmed; severity moderated to medium-high because the paired delta survives)*
The 512 eval rows per split are the **first 512 contiguous rows** (`indices[:limit]`, evaluator lines 166–169), covering 52 doc_ids that collapse into **~13 (val) / ~12 (test)** near-duplicate content families (mod-5000 doc_id pattern; 16 groups of byte-identical RL generations across "different" documents confirm they're the same text). The document-clustered bootstrap uses `independent_unit_count=52`, so the reported CIs are anti-conservative. Worse, the component split unions duplicates only on exact sha256 of a per-doc-seeded 300-token prefix, which misses position-shifted copies: ~48% of validation eval rows belong to residue classes that also have members in the train split, and — because AR and AV used *independently computed* component splits with different content columns — **110/512 validation rows (21.5%) have their exact doc_id in the AR critic's own SFT training split**. The RL-vs-SFT paired comparison is symmetric under all of this; absolute quality claims and the stated CI widths are not.

### MEDIUM (selected; all confirmed)

- **M1. Length confound unaddressed.** RL explanations average 117.4/118.0 words vs matched SFT 72.3/68.2 and teacher 74.5/74.6; per-row improvement correlates with per-row length increase (Pearson 0.34/0.49); no length-matched control exists; and the only mechanism limiting length growth was the removed guard (H3). Mitigating: on the 20/12 rows where RL was *not* longer, it still won 80%/92%. The doc's "86.38 vs 59.30 tokens" claim mislabels unique-word counts as lengths, understating the true disparity (+58%).
- **M2. Training reward was flat across the entire promoted segment.** Updates 228–341: first-10 mean −0.2365 → last-10 −0.2278 (slope +6.1e-5/update, t≈1.1, ns); ~88% of full-lineage reward improvement happened by update ~110 of retry2. Retry3 dropped the intermediate-checkpoint comparison, so **there is no evidence the final 114 updates added value over update-228** (the −0.207 vs −0.226 "regression" I flagged earlier is a batch artifact — retry2 scored −0.206 on the same rollout index).
- **M3. Undisclosed instabilities in the promoted segment.** K3 KL spiked to **1361.65** at step 245 (pre-clip grad norm 1632; clip=1.0), 163.9 at 312, plus 8 more >5; the claim docs say "no ... failure occurred" and quote only the healthy final record. The K3 coefficient (0.001) is cosmetic: KL to the SFT reference grew monotonically to ~1.8 nats, entropy fell 35%, and logged `train/loss` is numerically ~100% KL term (pg_loss ~1e-7).
- **M4. Controls are non-discriminative and the two-gate narrative double-counts.** The SFT baseline itself beats all five controls above the 0.9 win threshold, so "beat every control" would pass for any functional AV model. The 64/64 gate rows are a strict prefix subset of the 512/512 rows (7 doc_ids), so "both post-evals passed" is one dataset evaluated twice — and the v64 improvement *is* concentrated (top-5 rows = 34%/31% of net).
- **M5. Provenance gaps around the thing that actually computed the result.** The Miles RL framework (`/workspace/interp/code/miles-051cd15` — GRPO advantages, K3 estimator, PPO loss) is outside the 177-file source fingerprint; only 2 of its files are sha-pinned, and `actor.py`'s hash changed between 2026-07-04 and the hero retries with no recorded diff. The shipped "exact" source snapshot is missing all 19 `miles_patches/` files (fingerprint not reproducible from the archive). The frozen code root was mutated post-launch (queue YAML rewritten in place; capture sha ≠ launch sha). W&B configs record `nla_local_grad_norm=false` for all retry runs, contradicting docs that the audited local-shard gradient-clipping path was active — unexplained by the captured evidence.
- **M6. HPO peeked at test.** AR/AV HPO objectives explicitly average validation+test (`nano_ar_hpo_study.py:149-150` and the AV weighted mean); winners are unchanged under val-only ordering, but "test untouched" is false.
- **M7. The verifier/dataset chain has silent-default downgrades** (the known project bug class): `build_nano_r33_rl_dataset.py` silently degraded to doc-id-only filtering for the hero RL parquet (`component_filter_applied=false`); the frozen verifier version would *fail* the same manifest; the RL parquet is passed by path, not hash, at launch; 6 of the 8 open-tag-fallback rows in the hero eval are degenerate 256-token repetition loops that "usable 100%" counts as usable.

### Notable LOW/INFO
Registry records only the winning run (no failed hero attempts, null probes, or invalidated Qwen-scale line); most of the hero effect size existed already in the invalidated 2026-06-27 Qwen-comparable run (26.8%/28.7% on the same baseline — the corrected hero adds ~5–6 points); baseline generated once with no protocol-parity check in the gate (backend/workers/prefix all differed; the gate verifies dataset hashes and row identity but not generation protocol); stored bf16 activations drift up to 7% relative L2 from live recompute (affects "h fidelity" framing, not RL-vs-SFT fairness); W&B payloads 3/4 unfinalized with phase telemetry mostly dropped; snapshot README counts off by small amounts.

## 2. What independently verified as correct

Credit where due — this project's bookkeeping is far better than most:

- **Every reported statistic reproduces bit-exactly** from the rowwise arrays: aggregates for all 7 variants × 2 splits × 2 reports, the 30.97%/32.34% relative improvements, 427/512 and 454/512 win counts, closed/usable fractions, the top-5-row concentration (6.74%/6.30%), and the document-clustered bootstrap CIs (reproduced to the last digit with `default_rng(0)`, 100k resamples).
- Dataset hashes and 512/512 row identity between hero and baseline reports verified exactly; val/test doc sets disjoint (0 overlap, no shared residue classes, max cross-split shingle Jaccard 0.018).
- Eval generation is fully greedy and deterministic; teacher text is scored through the identical, untruncated AR path; the eval ran on the HF `legacy_batch` backend, not the SGLang rollout service.
- The **Adam reset at update 228 is benign**: over the 25 overlapping updates, retry3 (fresh Adam) vs retry2 (warm Adam) differ by −0.0022 reward (paired t=−1.06, ns) with continuous KL/grad-norm medians. 342×384 = 131,328 responses verified on the selected lineage.
- **No eval-level retry-until-pass**: all 15 failed-gate reports remain in the tree, and (except one undocumented canary) all are discussed in the logbooks. The single invalidated eval has preserved cache-divergence logs justifying invalidation.
- The generated text is **genuinely fluent, on-topic natural language** — correct topics, entities, genres, following the teacher's 4-section scaffold — not gibberish or symbol-stuffing. The failure mode is detail hallucination (names, dates, numbers frequently wrong) and hedging verbosity, not degeneracy.

## 3. Methodology and reproducibility assessment

The gate *architecture* is genuinely good: predeclared thresholds, dataset-hash binding, exact row matching, clustered bootstrap, fail-closed watchers, retention manifests, runtime contracts. Three systemic weaknesses undermine it. First, **the gate certifies the wrong surface**: it binds dataset bytes and row identity but not generation protocol (C1 slipped through exactly there — the baseline report's recorded generation metadata describes the scoring rerun, not the actual generation). Second, **the fingerprint boundary excludes the physics**: the Miles tree that computes advantages/KL/loss is unpinned, patches are attested by markers rather than hashes, and the snapshot can't reproduce its own fingerprint. Third, **process degrees of freedom accumulated at the boundaries**: guards relaxed per-retry, an eval-only escalation after a failed gate, an in-place-mutated queue file, and a winner-only registry. Reproducibility today: the recipe is well documented and the evidence chain for *what was measured* is excellent; the ability to *rerun the exact training computation* is not established (unpinned Miles, no dataset hash at launch, mixed-lineage guard configs, fresh-Adam splice).

## 4. Training-dynamics analysis

The 342-update lineage is best read as three regimes. **Updates 0–~110 (retry2): all the learning.** Reward −0.414 → −0.229, entropy 1.21 → ~0.82, response length mean 119 → ~175. **Updates ~110–253: drift.** Reward flat; length keeps creeping (p95 crosses 230, killing retry2); KL to the SFT anchor grows steadily — the 0.001 K3 coefficient never binds (final contribution 0.0018 to the loss). **Updates 228–341 (retry3, promoted): flat reward, episodic instability.** Reward statistically flat (+3.7%, ns); 10 KL excursions >5 including 1361.65 at step 245 with pre-clip grad norm 1632; entropy settles at 0.79; p95 length plateaus 224–238, always under the 256 cap, zero truncation. The within-batch length-reward correlation is *negative* (−0.20) yet lengths still grew — consistent with length growth being a policy-drift byproduct rather than direct reward pressure, but the eval-time length confound (M1) stands regardless. The eval dose-response (update-16 ≈ 14.8%, update-342 ≈ 23.6% fair) versus flat late training reward suggests strongly diminishing returns; whether update-342 beats update-228 at all was never measured.

## 5. Defensible claim boundary

**What the evidence licenses:**
> "Starting from the clean R33 AV-SFT checkpoint, 342 updates of guarded GRPO against a frozen AR critic produced a checkpoint whose greedy generated explanations, scored by that same critic on exact-matched heldout rows, improve *directional* (cosine) round-trip reconstruction by roughly **23%** over the clean SFT baseline on the uncorrupted half of the comparison, winning ~82–87% of rows, with the improvement broad across documents (only 1/104 document means favor SFT). At 1/21 of the optimization budget (update-16), a comparable ~14.8% gain transferred fully to a re-seeded critic. Generated text remains fluent, on-topic natural language."

**What it does not license:** the 30.97%/32.34% figures (corrupted baseline); "improved reconstruction of h" without the direction-only qualifier (raw R² ≈ 0); critic-independence of the full-size effect (never tested at 342); "52 independent documents" or the stated CI widths (~13/12 effective families); "beat every control" as evidence (trivially true of SFT too); any explanation-*quality* claim (details frequently hallucinated; +58% length confound); anything about R27 or external superiority (correctly disclaimed already).

## 6. Unresolved validity risks

1. Reward overfitting to the frozen critic at hero scale (H1) — untested and cheap to test.
2. Length/channel-capacity confound (M1) — no length-matched control exists anywhere.
3. Anti-conservative statistics from near-duplicate families (H5) — real CIs are ~4× fewer clusters wide; the true effective n is ~13 clusters/split.
4. Critic-side leakage (21.5% of val eval docs in the critic's SFT train) — direction of bias on the *paired* delta unknown.
5. Unpinned Miles core + `nla_local_grad_norm=false` — the exact executed RL math is not fully attested.
6. Single seed, single lineage, spliced guard regimes — no replication of any kind.
7. Update-342 vs update-228 (or 16) never compared — the promoted checkpoint may not be the best one in its own run.

## 7. Ranked next experiments

1. **Rescore the existing hero generated JSONL with the independent critic** (both v512 files + the SFT baseline's). No GPU generation needed — pure scoring. Directly tests H1 at full effect size. Highest information per dollar of anything available.
2. **Regenerate the SFT baseline rows 256–511 (ideally all 512) with the hero's exact no-prefix protocol and re-run the paired gate.** Replaces the headline with a defensible number; also re-run the update-16 diagnostic against it.
3. **Length-controlled comparison**: rescore RL generations truncated to teacher/SFT length (or cap `max_new_tokens` at generation), and report per-word reconstruction gain. Resolves M1.
4. **Report raw-space metrics** (centered R², raw MSE vs mean predictor) alongside cosine in every report, and rename the metric; test a magnitude-aware AR head if raw reconstruction matters to the NLA thesis.
5. **Eval-set rebuild**: cluster by residue-family/content (not doc_id), sample randomly across the split rather than first-512, cluster the bootstrap at family level. Re-run both gates on it.
6. **Checkpoint dose-response**: evaluate update-228 (retained? if not, re-run 16→342 with saves) to establish whether the promoted tail added anything; select checkpoints by round-trip NMSE as originally planned.
7. **Independent-seed RL replication** under the retry3 guard config, declared in advance, with the fixed baseline — the project's own stated gate.
8. **Fresh row-matched R27 comparison** (already the project's declared prerequisite for external claims).
9. **True capability eval, distinct from metric optimization**: score detail-hallucination rate of RL vs SFT vs teacher explanations (names/dates/numbers against source), plus task-level probes — this is what "natural language autoencoder" ultimately has to mean.
10. **Provenance hardening**: fingerprint the Miles tree, hash-pin the RL parquet at launch, add generation-protocol fields to the gate's parity check, stop in-place queue mutation, register failed runs. (The prefix bug and the silent component-filter downgrade are new instances of the project's known silent-default bug class — add both to the audit checklist.)

One correction to my earlier interim note: the reward "getting worse" from −0.207 to −0.226 across retry3 was a prompt-batch artifact, not degradation — retry2 scored the same rollout index nearly identically. The accurate statement is that reward was *flat*, which is the more interesting finding anyway.
