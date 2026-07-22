# Nano30B NLA Research Directions Memo

Date: `2026-06-10`
Scope: research scouting for the R33 hero milestone and beyond. This is a
strategy memo, not an audit; correctness/remediation state is tracked in
`docs/architecture/audit_remediation_20260610.md` and `docs/current_state.md`.

All quantitative claims below from pre-fix runs (packed-boundary contamination,
LR-schedule uncertainty) are used as directional scouting evidence only,
consistent with the status correction in `docs/current_state.md`.

---

## 1. Executive Thesis

The five most promising directions, in priority order:

1. **Run the round trip now, on R27, before any new training.** The promotion
   gate (`h -> AV text -> AR h_hat` vs the R27 baseline) has never actually been
   executed for *any* layer — including R27 itself, whose AV
   (`av_hf_iter_0000467`) and AR (`ar_hf_iter_0000256`) model-only checkpoints
   are both backed up on S3. A zero-training R27 round-trip run de-risks
   `scripts/eval_nano_av_ar_roundtrip_gate.py` at scale, produces the baseline
   number R33 must beat, and tells us *which leg loses information* before we
   spend another GPU-hour optimizing the wrong one.

2. **Make round-trip NMSE the HPO objective, not a final gate.** Today AV is
   selected by teacher-forced real-vs-control NLL and AR by teacher-text NMSE.
   Both are proxies that can rank checkpoints differently from the metric we
   actually care about. A bounded 128-row round-trip eval per AV candidate is
   cheap (the AR critic is a ~20B truncated model) and converts the gate into a
   continuous training signal — the single highest-leverage *process* change.

3. **Attack the explanation information ceiling with programmatic lexical
   anchors.** The AR `source_raw` oracle reconstructs R33 at NMSE ~0.08; teacher
   text plateaus near 0.28. That ~0.2 NMSE gap is information that exists in the
   prefix but is not in the explanation. The AR parquets already carry
   `token_text`/`token_position` provenance, so an anchor-augmented explanation
   variant (append "final token / local continuation" lines) costs **zero
   teacher API calls** and one ~1h AR run to test. If it closes a large fraction
   of the gap, the bottleneck is format, not models — and that redirects all
   downstream effort.

4. **AR-scored rejection-sampling self-distillation for AV (RL-lite).** Pure
   next-token SFT can never make AV exceed the teacher's information content.
   Full RL is correctly blocked, but RAFT-style best-of-k filtering (generate k
   explanations per row, keep the one the frozen AR reconstructs best, SFT on
   the winners) uses exactly the round-trip machinery, is storage-light, and is
   the natural first step beyond imitation once the gate evaluator is trusted.

5. **Test the boundary-type hypothesis cheaply.** R33 is a *post-Mamba,
   pre-attention* boundary (block 32 = M, block 33 = `*`), and it beats both
   post-attention boundaries tested (R34: 0.49 teacher NMSE at 20k; R27 plateau
   0.44) by a wide margin, with a 2x better source_raw floor. R26 and R42 are
   the exact structural siblings of R33 (post-M, pre-`*`). Two 20k AR probes
   (~1-2h each at mb96) would tell us whether "pre-attention boundaries are
   systematically more text-reconstructable in hybrid Mamba/MoE models" — a
   genuinely novel finding about hybrid architectures, and insurance for the
   layer decision.

---

## 2. Current Bottleneck Hypothesis

Decompose round-trip error into three stages:

```text
(a) information in z about h     [teacher format ceiling]
(b) AV's ability to generate z'≈z given h   [imitation + injection channel]
(c) AR's ability to map z -> h geometry     [reconstruction]
```

Evidence per stage:

- **(c) is healthy.** The AR model path is near-perfect when text carries the
  information (`source_raw` 0.07-0.10 NMSE). The readout diagnostic ruled out a
  head bottleneck (ridge/closed-form did not beat the trained value head). LR
  polish and batch probes in the same basin were flat — classic signs of an
  *information*-limited, not optimization-limited, regime.
- **(a) is the prime suspect for the teacher-text plateau.** Teacher text
  (0.277) now slightly beats `source_context` (0.283-0.304) at R33 — the
  structured predictive-features format is *good*, but it is generated from the
  source prefix by a teacher that never sees `h`, is capped at 3-5 short lines,
  and is explicitly forbidden from carrying verbatim continuation cues. The
  mean-control NMSE (~0.67-0.70, i.e. cos≈0.65 to the train mean) says a large
  shared component is easy; the remaining error is the row-specific residual,
  exactly what a compressed text summary loses.
- **(b) is unmeasured.** AV real-vs-control gaps prove row-specificity of the
  *likelihood*, and the two-example generation sanity was promising, but nobody
  has measured how much reconstruction is lost between teacher z and
  AV-generated z. This is the round trip's job. Secondary (b) risks: the
  injection scale (75) was inherited from R27 and never re-derived from
  measured R33 norm statistics, and Track A (embedding injection of a layer-32
  residual through 33 lower hybrid blocks) has never been compared against a
  residual-boundary oracle at R33.

**Hypothesis:** post-fix round-trip NMSE will land in 0.35-0.55 — generated z
worse than teacher z, both far above the source_raw floor — with stage (a) and
stage (b) each contributing materially. The near-term experiments below are
chosen to *measure the split* before betting on either.

Falsification: if AV-generated z reconstructs nearly as well as teacher z
(gap < 0.03 NMSE), format/teacher work dominates and self-distillation/RL is
premature. If generated z is catastrophically worse (> 0.15 gap), AV
generation/injection dominates and format work is premature.

---

## 3. Near-Term Experiments

Ordered. NT-1/NT-2/NT-3 are the critical path to the milestone; NT-4 through
NT-8 run opportunistically alongside.

### NT-1: R27 round-trip baseline (zero training)

- **Hypothesis:** the round-trip evaluator works at 256-row scale, and the
  mature R27 pair produces a finite, interpretable baseline (expected NMSE
  0.45-0.65 from generated text vs 0.44 teacher proxy).
- **Dataset/checkpoints:** R27 99,570-row AV/AR split parquets (protected on
  PVC / S3); `s3://.../checkpoints/r27-av-ar-best/av_hf_iter_0000467/` and
  `ar_hf_iter_0000256/`.
- **Config sketch:** `scripts/eval_nano_av_ar_roundtrip_gate.py`, 256
  validation / 256 test rows, controls `real, shuffled, zero, mean, none`,
  `max_new_tokens` ≥ 200, temperature per AV sidecar; report teacher-text AR
  NMSE alongside generated-text NMSE.
- **Cost:** S3 restore (~96 GB) + ~1-2h on 2 GPUs. No training.
- **Success metric:** report completes; `av_real` generated-text NMSE beats
  shuffled/zero/mean/none controls; parse/closure rate ≥ 90%.
- **Failure interpretation:** if `av_real` does not beat controls, the R27 AV's
  generations do not transmit row-specific information through AR at all — the
  R33 gate criterion ("beat R27") would be trivially weak and the real
  comparison becomes "beat teacher-text controls". If parse rate is low, fix
  generation/stop-sequence handling before any R33 gate run.
- **If it works:** the R27 bar is set; the same command with R33 checkpoints is
  the milestone gate. Also archive per-row outputs as the qualitative panel
  seed (Section 6).
- **Caveat:** both R27 checkpoints are pre-fix artifacts — label the baseline
  as scouting-grade, exactly as the docs already label their training runs.

### NT-2: Post-fix R33 AR confirmation (already planned — gates restated)

- **Hypothesis:** the packed-boundary fix + real LR decay reproduces (or beats)
  the pre-fix scouting result, because heldout evals were honest even when
  training was noisy.
- **Dataset/checkpoint:** verified `r33_prefix_fullscan275396` parquets; R33
  critic init.
- **Config sketch:** `lr=5e-5, cosine, warmup=25, gb192/mb96, 2 GPU`; 100k
  bounded run first, then full275k. Must carry: checkpoint-local
  `modeling_nemotron_h.py.nla_patch_report.json`, `NLA_ASSERT_PACKED_EQUIV`
  preflight, LR-decay canary. Replace the silent
  `allow_packed_critic_training: true` acknowledgment with an explicit
  `packed_preflight_evidence:` field (Section 7).
- **Cost:** ~45-60 min (100k) / ~2.5-3h (275k) on 2×H200 + ~15 min eval each.
- **Success metric:** 100k teacher NMSE ≤ 0.31; 275k ≤ 0.28-0.29 with all
  controls ordered correctly.
- **Failure interpretation:** materially worse than pre-fix (> +0.03) means the
  contamination was *helping* (cross-sample leakage as data augmentation is
  unlikely but possible) or the LR schedule actually ran differently than
  labeled; re-run the 20k probe family to re-rank layers before concluding
  anything.
- **If it works:** this is the clean AR hero; freeze it as the round-trip
  scorer for NT-5/MT-1.

### NT-3: Post-fix R33 AV confirmation with a denser LR ladder

- **Hypothesis:** the pre-fix smoke ordering (1e-4 best at 32 steps) holds, but
  32-step smokes over-reward high LR; at 128+ steps the optimum likely sits in
  2e-5..1e-4.
- **Dataset/checkpoint:** `av_sft_R33_r33_prefix_fullscan275396.parquet`.
- **Config sketch:** 20k rows, 128 steps, `gb192/mb2/seq1152/dyn512`, LRs
  {2e-5, 5e-5, 1e-4}; then one 100k medium confirmation at the winner. Same
  patch/preflight/canary evidence as NT-2.
- **Cost:** a few hours per 20k probe; ~6-12h for the 100k medium (AV is the
  slow side; see Section 7 for the throughput experiment that could halve
  this).
- **Success metric:** real NLL ≤ 1.0 with shuffled/mean/none gaps ≥ 0.30, and
  — once NT-2 exists — round-trip NMSE (NT-5) used as the tiebreaker.
- **Failure interpretation:** if no LR achieves Qwen-like control gaps on clean
  code, suspect the injection channel (NT-4, MT-4) before scaling data.
- **If it works:** promote winner to full275k AV hero only after NT-5 shows it
  also wins on round trip.

### NT-4: R33 activation norm statistics + injection-scale smoke

- **Hypothesis:** `injection_scale: 75` (inherited from R27) is mismatched to
  the R33 norm distribution; the reference treats scale as norm-matched per
  site ("a round number a bit above the mean norm"), and Qwen-vs-Gemma shows
  scale mismatches are catastrophic-to-silent.
- **Dataset:** `base_R33_r33_prefix_fullscan275396.parquet` (raw vectors, no
  normalization — guaranteed by datagen invariants).
- **Config sketch:** (i) a pandas/pyarrow script computing L2-norm
  p05/p50/p95/p99 per layer for R27/R33 rows — minutes, CPU-only; (ii) if p50
  differs from 75 by > ~1.5x, two 32-step AV smokes at `scale = {p50-matched,
  2x p50}`.
- **Cost:** <10 min CPU + optionally 2 short smokes (~1h total).
- **Success metric:** norms documented in the sidecar; if a re-derived scale
  improves 32-step real NLL by > 0.05, adopt it for NT-3's medium run.
- **Failure interpretation:** if scale barely matters across a 4x range, the
  embedding channel is robust and (b)-side suspicion shifts to imitation
  quality rather than injection.
- **If it works:** record per-boundary norm stats as a required dataset-gate
  field for all future layers.

### NT-5: Round-trip-as-HPO-signal (bounded 128-row round trip per AV candidate)

- **Hypothesis:** AV checkpoints with similar real NLL can differ materially in
  round-trip NMSE (generation quality, format discipline, parse rate); the gate
  metric should select the AV hero.
- **Dataset/checkpoints:** NT-2 clean AR (frozen scorer) + each NT-3 AV
  candidate.
- **Config sketch:** extend the AV queue's post-eval step to call the
  round-trip script with `--validation-limit 128 --test-limit 128`,
  real-control only (skip shuffled/zero/mean generation controls for cadence
  runs; keep them for gate runs). Pipeline the two phases across the 2 GPUs (AV
  generates on GPU0 while AR scores on GPU1) instead of load→unload→load.
- **Cost:** ~20-40 min per candidate.
- **Success metric:** a stable ranking of AV candidates by generated-text NMSE;
  the teacher-z round trip on the same rows gives the (a)-vs-(b) decomposition
  of Section 2 *as a side effect*.
- **Failure interpretation:** if 128-row rankings are noisy run-to-run, raise
  to 256 and fix generation seed/temperature.
- **If it works:** this becomes the standard AV selection protocol; add
  round-trip fields to `runs/registry/experiments.yaml`.

### NT-6: Lexical-anchor explanation ablation (programmatic, AR-only first)

- **Hypothesis:** a compact, honest lexical anchor appended to each teacher
  explanation (e.g. `Final token: "{token_text}" at position ~{bucket};` plus
  the next 1-2 source tokens as a continuation hint) closes a large fraction of
  the teacher(0.28)→source_raw(0.08) gap, because h is known to strongly encode
  local token identity/position and the current teacher prompt explicitly
  forbids verbatim cues.
- **Dataset:** rebuild `ar_sft` (and later `av_sft`) from the existing 275k
  parquet by string-appending anchor lines from the provenance columns —
  **zero teacher API calls**. Slice to 100k for the probe.
- **Config sketch:** same as NT-2's 100k config, new dataset path, new run
  name `r33-100k-anchor-v1`. Keep an anchor-only control (explanation replaced
  by *just* the anchor lines) to separate "anchor information" from
  "anchor+semantics synergy".
- **Cost:** ~30 min CPU rebuild + 2 AR runs ≈ 2h total.
- **Success metric:** anchored teacher NMSE ≤ 0.22 (i.e. closes ≥ 25% of the
  gap). Anchor-only control quantifies how much is pure lexical replay.
- **Failure interpretation:** if anchors add < 0.02, the trained channel
  already extracts local-token info from teacher text implicitly (consistent
  with the earlier hint-kNN null) and the remaining error is genuinely
  semantic/geometric — redirect to MT-3 (teacher v2) and MT-6 (centered
  objective).
- **If it works:** decide the *interpretability trade-off* explicitly: anchors
  make explanations more reconstructive but more replay-like. The honest
  framing is a two-part explanation (semantic features + declared lexical
  anchor) — the AV then has to verbalize both, and the round trip improves
  without pretending lexical content isn't in h. Then regenerate AV-side data
  and rerun NT-3.

### NT-7: Shortcut-detection controls (eval-only)

- **Hypothesis:** part of the current teacher signal is doc-level topic rather
  than position-specific content; current global-shuffle control cannot see
  this.
- **Config sketch:** add two AR eval controls: (i) *within-doc shuffle* — swap
  explanations between positions of the same document; (ii) *anchor-stripped
  teacher* (if NT-6 adopted) — explanations with lexical lines removed.
  One-file change in `eval_nano_ar_miles_checkpoint.py` control builders.
- **Cost:** eval-time only, minutes.
- **Success metric:** within-doc shuffle NMSE materially worse than teacher
  (≥ 0.08 gap) ⇒ explanations are position-specific, not just topical.
- **Failure interpretation:** a small gap means much of the reconstruction is
  doc-topic — important for honest claims, and an argument for
  contrastive/position-discriminative teacher prompts (MT-3).
- **If it works:** promote both controls into the standard eval contract and
  the round-trip report schema.

### NT-8: Boundary-family probe — R26 and R42 (optional, decision-insurance)

- **Hypothesis:** pre-attention (post-Mamba) boundaries are systematically more
  text-reconstructable than post-attention boundaries in Nano's hybrid stack;
  R33's advantage is structural, not idiosyncratic.
- **Dataset:** same 20,416-row teacher-matched slice and prefix-extraction path
  used for the R33/R34/R51 probes (extract R26/R42 with
  `scripts/nano_prefix_activation_extract.py`).
- **Config sketch:** clone `r33_mini_probe_20k_lr2e5_cosine_128steps.yaml` for
  R26/R42; same eval contract.
- **Cost:** ~1-2h each (extraction + 128-step train + eval) on 2 GPUs.
- **Success metric:** ordering. If R26/R42 cluster near R33 (≤ 0.40 teacher
  NMSE) while post-attention boundaries sit ≥ 0.44, the boundary-type
  hypothesis is supported.
- **Failure interpretation:** if R26/R42 look like R34, R33 is special for
  another reason (depth sweet spot) — fine, the layer decision stands either
  way.
- **If it works:** publishable observation about hybrid-architecture NLAs;
  also justifies pairing AV/AR at pre-attention sites for any future layer.

---

## 4. Medium-Term Research Bets

### MT-1: AR-scored rejection-sampling self-distillation for AV (RAFT-style)

The main bet. Generate k=8 explanations per row from the post-fix AV on a
20k-row slice, score each with the frozen clean AR, keep the argmin-NMSE
winner per row, build `av_sft_selfdistill_v1.parquet`, fine-tune AV ~1 epoch at
low LR. This is the cheapest mechanism that lets AV *exceed* the teacher's
information content — it optimizes the true objective without RL infrastructure
(no GRPO, no online critic, no SGLang weight sync). Cost: ~8x20k generations
(batched, ~hours) + one short AV run. Success: round-trip NMSE (NT-5 protocol)
improves ≥ 0.03 over the SFT AV without control-gap regression and without
explanation degeneration (check length/diversity/CJK-leak diagnostics).
Failure: if best-of-8 selection barely improves reconstruction, the AV's
sampling distribution has little exploitable variance — a strong argument that
RL would also be sample-starved, and format/teacher work matters more. If it
works: iterate (RAFT round 2) or upgrade to DPO pairs (best-vs-worst per row),
and only then revisit GRPO.

### MT-2: Best-of-k + vector-averaging inference recipe

At round-trip eval time, sample k explanations, reconstruct each, and report
(i) AR-selected best-of-k and (ii) the NMSE of the *averaged* predicted vector.
(ii) is a legitimate ensemble (direction averaging) and tells us how much
generation noise costs; (i) upper-bounds RAFT gains before paying for MT-1.
Cost: eval-only, k× generation. Include both in the gate report alongside
single-sample NMSE (the headline metric stays single-sample for fairness).

### MT-3: Teacher format v2 on a 20k slice (vector-discriminative prompt)

Regenerate teacher explanations for the 20k probe slice with a revised prompt:
ask the teacher to (i) describe what distinguishes *this* position from the
same document 10 tokens earlier (contrastive framing), (ii) commit to a
predicted next token, (iii) name register/syntax constraints. Compare matched
AR probes: v1 vs v2 explanations. Cost: ~20k teacher calls (Nemotron Super
endpoint, modest bill) + two 20k AR probes. Success: ≥ 0.03 NMSE improvement at
20k scale justifies regenerating 275k. This is the controlled way to improve
labels without a full-scale teacher bill.

### MT-4: Track C boundary-oracle diagnostic for the AV injection channel

Teacher-forced NLL with the true h patched directly at the R33 boundary
sentinel position (split-forward in HF, no generation, no cache complexity) vs
the production embedding-replacement channel. If boundary patching is much
better, the embedding channel is leaving information on the table and Track B
(learned trainer-side affine adapter `W·v+b` — explicitly a trainer-only change
in the reference design) gets promoted from "deferred" to "next". If they are
comparable, close this line permanently. Cost: 1-2 days of harness work (the
split-forward machinery exists in `nano_extraction_identity.py`), eval-only
runs.

### MT-5: 1-GPU AR HPO enablement (optimizer memory)

Single-GPU AR is blocked purely by Adam state (~137 GiB floor regardless of
microbatch). Two probes: (i) 8-bit/paged Adam for the critic, (ii) half-depth
critic (17 layers) as an HPO *ranking proxy* — validate once that it preserves
the ordering of 3 known configs, then use it for cheap sweeps. Either doubles
trial throughput per 2-GPU node. Cost: each probe is a day of work + smoke
runs. Risk: 8-bit optimizer changes optimization semantics — keep hero runs on
faithful Adam.

### MT-6: Mean-centered / whitened AR objective variant

The direction-only loss rewards predicting the dominant shared component (mean
control cos ≈ 0.65). Train one AR variant on mean-centered targets (center
fit on train split, stored in sidecar) and report both centered and uncentered
metrics. Hypothesis: forces explanation capacity onto the row-specific
residual; even if headline NMSE is similar, centered-FVE will reveal whether
explanations carry distinctive content. Cost: small loss-code change + one 100k
run. Interpretation note: this changes the scientific objective — keep it as a
diagnostic twin, not a replacement, until round-trip evidence favors it.

---

## 5. Speculative / High-Upside Ideas

- **SP-1: Multi-token injection.** Project h into m > 1 token embeddings
  (fixed chunking or a learned linear) to widen the channel into the lower
  stack. Trainer-side only; pairs naturally with MT-4 evidence. High upside if
  the embedding channel is the bottleneck; meaningless otherwise — gate on
  MT-4.
- **SP-2: Structured/JSON explanations.** Emit explanations as a typed feature
  list (`{"syntax": ..., "topic": ..., "register": ..., "next_token": ...}`).
  Improves parseability, enables per-field ablations (which field carries the
  reconstruction?), and gives RL a cleaner action space later. Cost: teacher
  regeneration + format-compliance training; try only after MT-3 shows prompt
  changes move NMSE.
- **SP-3: Cross-layer multi-target AR.** One AR conditioned on a layer tag
  reconstructing {R27, R33, R34} from the same explanation. Tests whether
  explanations encode layer-stable semantics vs layer-specific geometry;
  doubles as a regularizer. The prefix-extraction path already produces
  multi-layer activations in one forward.
- **SP-4: Norm side-channel.** Direction-only loss discards ‖h‖. Add a tiny aux
  head (or a text field) predicting the norm bucket. Cheap; mostly scientific
  curiosity about what text *can* carry; could matter for downstream causal
  patching (re-injecting h_hat needs a norm).
- **SP-5: SAE-latent bridge.** Train a small SAE at R33; ask whether AV
  explanations predict active SAE latents (and vice versa). Not on the hero
  path — but it is the natural way to connect this NLA to the broader interp
  toolkit, and SAE latents could later become structured explanation targets.
  Note the known SAELens/naming friction from other projects; treat as a
  separate workstream.
- **SP-6: GRPO RL (blocked, by design).** When the round-trip gate passes vs
  R27: the reference stack (Miles GRPO + online critic + SGLang `input_embeds`)
  is already vendored and documented; the Qwen reference needed RL to reach
  FVE ≈ 0.75. RAFT/DPO results (MT-1) will predict whether RL has exploitable
  headroom and calibrate sample budgets.

---

## 6. Recommended Eval Upgrades

1. **Round-trip decomposition report** (NT-1/NT-5): for the same rows, report
   `source_raw` ≤ `anchored teacher z` ≤ `teacher z` ≤ `AV best-of-k` ≤
   `AV single-sample` ≤ controls. One table that localizes the bottleneck;
   regenerate it at every milestone.
2. **Within-doc shuffle control** (NT-7) — the strongest cheap test against
   doc-topic shortcut learning. Add to AR eval and round-trip report.
3. **Centered metrics everywhere:** report cos-after-mean-subtraction (and
   FVE-vs-mean, already present) next to raw NMSE, so improvements in the
   shared component can't masquerade as explanation quality.
4. **Generation health panel for AV:** parse/closure rate, length distribution,
   distinct-n diversity, CJK-leak counter (the reference's loudest
   injection-failure smell), and rate of explanations that are verbatim
   substrings of the source prefix (replay detector — becomes critical once
   anchors/self-distillation are in play).
5. **Per-position and per-norm-band breakdowns:** early positions (< ~50) and
   high-norm outliers are known-noisy in the reference; reporting them
   separately prevents tail rows from dominating gate decisions.
6. **Qualitative panel:** 50-100 fixed held-out rows spanning prose / code /
   math / multilingual / early-mid-late positions, regenerated at each AV
   milestone, stored as JSONL next to the eval report. The 2-example sanity in
   the logbook was informative; institutionalize it.
7. **Registry/W&B wiring:** add round-trip fields (`av_ckpt`, `ar_ckpt`,
   `parse_rate`, `rt_nmse_real`, `rt_nmse_teacher`, `baseline_r27_rt_nmse`) to
   `runs/registry/experiments.yaml` and log eval JSONs into W&B so evidence
   survives PVC incidents (this was already a stated gap; it has bitten twice
   via Longhorn).

---

## 7. Recommended Config/Training Changes

1. **Clean-run evidence as config fields, not env-var folklore:** add
   `evidence: {patch_report: required, packed_preflight: required, lr_canary:
   required}` to hero/confirmation configs, and make the runner refuse
   `complete-performance` launches without them. Replace the silent
   `allow_packed_critic_training: true` with an explicit pointer to preflight
   evidence.
2. **AV throughput probe before the AV hero:** the AV path is the wall-clock
   bottleneck (mb2, `max_tokens_per_gpu=512` with seq cap 1152 and
   `allow_oversized_dynamic_batch: true`). Measure packing occupancy; probe
   `max_tokens_per_gpu ∈ {1152, 2304}` with mb scaled on a 32-step smoke. A
   2-4x AV speedup is plausible and compounds across every remaining AV run.
   Verify the dynamic-batching path is not silently packing multiple samples
   without the boundary fix (the known audit caveat).
3. **Two-GPU pipelined round-trip eval** (NT-5): AV on GPU0, AR critic on GPU1,
   stream text — removes the load/unload serialization in the current
   two-phase script for cadence runs.
4. **Keep 100k as the standard AR proxy; 275k only for heroes.** Evidence: the
   ordering at 100k predicted the 275k result; 100k at mb96 costs ~45 min.
5. **Norm stats in the dataset gate** (NT-4): per-boundary L2-norm
   p05/p50/p95/p99 written into the sidecar at extraction time; injection scale
   must reference them.
6. **Optimizer-memory probe for 1-GPU trials** (MT-5) — only if HPO becomes
   queue-bound again; not on the hero path.
7. **Stop-sequence/closure enforcement in all AV generation paths** (eval and
   future RAFT/RL): `</explanation>` as stop string + max-token budget, so
   parse failures are budget-related, not format-related.

---

## 8. What NOT To Do Yet

- **No RL/GRPO** until the round-trip gate passes vs R27 (project rule; also
  MT-1 will tell us whether RL has headroom worth its complexity).
- **No same-basin AR LR/batch polish.** Three independent probes (low-LR
  polish, batch384, constant-LR) were flat; the regime is information-limited.
- **No broad layer sweeps** beyond NT-8's two targeted siblings. The 27-layer
  kNN sweep already showed cheap screens don't predict trained AR quality.
- **No 500k/1M-row datagen.** The 275k teacher set is not the binding
  constraint; format and round-trip alignment are. Scale data only after NT-6/
  MT-3 settle what the explanations should *say*.
- **No Track B learned injection adapters** before MT-4 oracle evidence — the
  decision rule in the core plan (A weak + C strong ⇒ B) still stands, and we
  have no C measurement at R33.
- **No R34 switch.** Its marginal AV win (~0.003 NLL) cannot offset a 0.10+ AR
  deficit; round-trip quality is AR-dominated at current operating points.
- **No SAE dependency on the hero path** (SP-5 is a parallel curiosity, not a
  gate).

---

## 9. Decision Tree to the R33 Hero Milestone

```text
0. NT-1: R27 round-trip baseline (no training)
   ├─ evaluator broken / parse rate low → fix gate harness first
   └─ baseline number recorded → continue

1. NT-2: post-fix R33 AR (100k → 275k, with evidence trio)
   ├─ ≥ pre-fix-quality (≤ ~0.28 at 275k) → freeze as clean AR hero
   └─ regression > +0.03 → rerun 20k layer probes post-fix;
        if R33 no longer leads → NT-8 siblings + R27 fallback decision

2. NT-3 + NT-4 + NT-5: post-fix R33 AV (LR ladder, scale check,
   round-trip-ranked selection)
   ├─ control gaps ≥ 0.30 and stable → pick AV by 128-row round trip
   └─ gaps weak at all LRs → injection-channel work (NT-4 deep dive, MT-4)
        before any AV hero spend

3. R33 round-trip gate (512/512, full controls, vs R27 baseline from step 0)
   ├─ R33 beats R27 AND av_real beats all controls
   │    → hero milestone CLAIMED (with parse-rate + decomposition table)
   │    → start MT-1 (RAFT) and RL planning
   ├─ teacher-z round trip strong, generated-z weak (gap > 0.10)
   │    → AV-limited: MT-1 self-distillation + MT-2 best-of-k + scale/oracle
   ├─ teacher-z and generated-z both weak (both > ~0.45)
   │    → information-limited: NT-6 anchors → MT-3 teacher v2 → re-train AR/AV
   └─ R33 loses to R27 despite better teacher proxy
        → AV side at R33 is the suspect: norm/scale audit (NT-4),
          boundary oracle (MT-4), then reconsider R34/R26 pairing for AV only

4. Post-milestone: MT-1 RAFT → DPO → GRPO, each gated on round-trip
   improvement without control-gap regression.
```

The through-line: **measure the teacher-z vs generated-z split first** (it
falls out of NT-1/NT-5 for free), because every expensive bet — anchors,
teacher regeneration, self-distillation, RL, injection adapters — is justified
by one side of that split and wasted on the other.
