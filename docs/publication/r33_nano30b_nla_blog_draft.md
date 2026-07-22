# Can You Round-Trip a 30B Model's Thoughts Through English? What We Actually Established (and What We Didn't)

<!-- R33-HERO-BASELINE-PROTOCOL-INVALIDATED -->

Status: internal blog draft (`2026-07-16`), companion to
`docs/publication/r33_nano30b_nla_paper_draft.md`. Not cleared for external
publication. Every number below traces to a hash-bound JSON artifact listed
in `docs/publication/r33_nano30b_nla_evidence_table.md`.

---

## The idea

Take one activation vector from deep inside a language model — 2,688 numbers
describing what the model is "thinking" at one token position. Ask a second
model to write a short English explanation of it. Then ask a third model to
read only that explanation and reconstruct the original vector.

If the reconstruction lands close to the original, the explanation must
carry real information about the activation — not vibes, but measurable
bits. That's the Natural Language Autoencoder (NLA) idea, introduced by
Fraser-Taliente, Kantamneni, Ong et al. in a 2026 Transformer Circuits paper.
The text in the middle is a *bottleneck*: whatever survives the round trip
had to fit through natural language.

We adapted this method (it's an adaptation, not a reproduction — different
injection scale, a frozen critic, and architecture adapters) to
NVIDIA-Nemotron-3-Nano-30B-A3B, "Nano30B", a hybrid Mamba-2/attention/MoE
model. Our pieces:

- **AV** (activation verbalizer): Nano30B fine-tuned so that one input
  embedding slot is overwritten with the activation vector (rescaled to a
  fixed norm), then generates `<explanation>…</explanation>`.
- **AR** (activation reconstructor): the first 34 blocks of Nano30B with a
  linear head that reads an explanation and emits a 2,688-dim vector.

## Why layer R33

We probed several residual boundaries with small matched AR trainings. R33 —
the residual stream after the first 33 of the model's 52 blocks —
reconstructed teacher text markedly better than R34 (directional error 0.38
vs. 0.49 at 20k rows) and better than the heavily tuned R27 line (~0.44).
Honest footnote: those probes ran before two training bugs were fixed, so
the layer choice is scouting evidence, and we make no "R33 is the best
layer" claim. Also, the project's documents carry two conflicting
conventions for what the "33" label points at: the extractor code that
built the dataset counts blocks run (making R33 sit after a Mamba block and
just before an attention block), while a later geometry audit reads the
label as a module index (block 33, an attention block). The measurements
are the same either way, but until the labels are reconciled we claim only
the boundary itself, not its architectural type.

## Why controls and round trips matter more than the headline metric

A single reconstruction number is easy to fool. Three examples of how, all
of which bit this project or its predecessors:

1. **The metric can hide what it ignores.** Our primary metric,
   `2 × (1 − cosine)`, compares *directions* and is blind to magnitude. A
   prediction 10× too large scores perfectly. So every table also reports
   raw error against a "just predict the average activation" baseline — and
   on that raw scale our system is *worse* than the baseline (more below).
   Historical reports called the directional metric "NMSE", which invited
   misreading it as raw reconstruction. We renamed it.
2. **Text about the *source* can masquerade as text about the
   *activation*.** Feeding the AR the raw source passage reconstructs
   direction almost as well as the teacher explanation does. So the
   load-bearing controls hold the entire pipeline fixed and change only the
   conditioning vector: AV text generated from the *right* activation vs.
   from a shuffled one (different content family), a zero vector, the mean
   vector, or no injection at all. Only the true row's activation should
   help — and the gap measures activation-specific information.
3. **Leaky evaluation sets flatter you.** Web corpora are full of
   near-duplicates. An earlier evaluation's "52 independent documents"
   collapsed into ~13 effective duplicate clusters. The clean protocol
   groups all 27,647 documents into 5,009 *content families* (exact ≥0.80
   Jaccard closure over 5-word shingles) and assigns whole families to
   train/validation/test. All statistics are computed over families, not
   rows.

## What the clean experiment established

Both models were retrained from scratch in July 2026 on a deterministic,
byte-replayable activation snapshot (275,396 rows) with family-disjoint
splits, then evaluated on 512 family-stratified rows per split with greedy,
protocol-hashed generation. Every report passed an independent fail-closed
verifier. The numbers:

- **The AV really uses the activation.** The teacher explanation's NLL under
  the AV is 0.777 with the true activation injected, vs. 1.18–1.31 for
  zero/mean/no-injection/shuffled conditioning.
- **The round trip works, directionally.** AR reconstruction from
  AV-generated text: directional error **0.307** (validation) and **0.319**
  (exploratory test) — essentially teacher-text level (0.305 / 0.303), where
  0 is perfect and 2 means uncorrelated directions on average. All 1,024
  generations parsed cleanly.
- **Controls lose decisively.** Cross-family shuffled conditioning scores
  ~0.97 — the true-activation text beats it on 100% of test rows. Every
  registered control margin has a positive family-clustered 95% interval
  (margins 0.36–0.66, sign-flip p ≈ 1e-5).
- **The reconstruction behaves like the activation.** Re-inject `h_hat`
  (rescaled to the original's norm) into the stored forward pass at the same
  position and compare next-token distributions: KL ≈ 0.95 on test —
  statistically indistinguishable from injecting the reconstruction of the
  *teacher's* text (0.97), and far better than mean (4.1), zero (6.3), or
  shuffled (9.5) injections. Re-injecting the stored original itself gives
  KL ≈ 0.003, so the machinery is near-lossless; the gap is what the
  language bottleneck costs.
- **It replicates across reconstructors.** A second AR, independently
  initialized (fresh head seed 314159, changed router init — 16/16
  independence checks) and independently trained, scored the *same frozen*
  AV generations at 0.311 vs. 0.309 for teacher text, all controls passed.
  So the signal isn't an artifact of one particular critic — important
  because the primary AR had also been the RL reward model historically.

Total compute for the three clean training runs: 138.7 H100 GPU-hours
(that's the selected runs only — extraction, evaluation, HPO history, and
failed runs are explicitly excluded from that number).

## The bugs that invalidated earlier results

This protocol is strict because earlier, looser versions produced results
that died under audit. In the spirit of showing the whole ledger:

- **Cross-sample state leakage.** Packing multiple training samples into one
  microbatch let Mamba state (and, via a second bug, position IDs) leak
  across sample boundaries in the vendored Nemotron-H code. Training-path
  outputs diverged from evaluation-path outputs by up to ~22% — every
  checkpoint trained through those paths was reclassified as scouting
  evidence and the paths now sit behind fail-closed equivalence gates.
- **The optimizer wasn't doing what the config said.** A trainer bug forced
  constant learning rates while configs said "cosine". Runs now carry
  LR-decay canaries.
- **The invalidated RL headline.** On July 8 an RL-tuned AV was promoted
  with "31%/32% better than SFT". An independent audit found the SFT
  baseline had been generated under two different prompt prefixes (deflating
  the baseline), the effect was scored by the same critic that provided the
  RL reward, the "NMSE" was direction-only while raw-space recovery was ≈ 0,
  guard thresholds had been progressively relaxed until a run finished, and
  the eval set had the near-duplicate problem above. A corrected rescore
  still showed a ~20–23% gain under two critics — but the whole lineage sat
  on an activation snapshot that later failed identity audits, so it
  remains internal evidence. **No RL result is claimed. There is currently
  no publication-valid RL model.** The July 8 numbers should not be cited.

A documentation checker now fails CI if the invalidated headline reappears
anywhere as an active claim.

## What it did *not* establish

- **Not magnitudes.** Raw reconstruction error is ~33% *worse* than
  predicting the training-mean vector; the pipeline overshoots norms ~1.5×.
  A single scalar (0.56, fit on validation teacher data only) fixes most of
  that after the fact — raising test centered R² from −0.34 to +0.48 — but
  that's post-hoc calibration evidence, not native magnitude recovery.
- **Not exact activations.** Stored snapshots replay byte-identically, but a
  fresh forward pass today differs from the stored vectors beyond strict
  tolerance on every audited row (cosine ≈ 0.9991 — very aligned, not
  identical). Everything is scoped to the stored snapshot.
- **Not a pristine test set.** A full exposure audit mapped every historical
  document (28,665 across 136 sources, zero unmapped) and found *all* 5,009
  content families were touched by some historical run or selection step.
  The test split is family-disjoint from clean training — but "exploratory",
  not confirmatory. No repartitioning of this corpus can fix that.
- **Not faithful explanations.** The generated text is fluent and passes
  structural screens, but nobody has blind-reviewed its semantic accuracy,
  and the RL-era audit found hallucinated details in generations. These are
  lossy information channels, not the model's chain of thought.
- **Not a layer ranking, not seed robustness.** One AV seed; layer choice
  from pre-fix probes; no clean R27 comparison.

## What comes next

The decisive missing piece is an **external teacher-backed evaluation
boundary** — new documents, provably outside the exposed corpus, with
teacher explanations generated under cleared terms — evaluated once, under
the already-frozen protocol. That's what would turn "exploratory" into
"confirmatory".

For RL, a draft preregistration already exists and is deliberately strict:
frozen guard policies (no mid-run relaxation), a four-point stability grid,
two independent seeds that must each show a positive family-clustered gain
under an independent critic, length-matched controls, and one-shot test
evaluation after selection lock. External replication on other model
families, a second AV seed, blinded human semantic review, and a
magnitude-aware AR head are the other obvious moves.

Since the original draft, a family-clean internal AV+AR pair was trained for
342 online-RL updates (about 43 hours and 65,664 generated responses), then
passed a stronger exact-matched validation eval: directional round-trip NMSE
falls from `0.309055` for SFT to `0.224386` after online RL (`27.4%`), while
raw MSE falls `23.9%`; the comparison uses 122 independent validation
families, identical 384-token generation settings, and all required controls.
That is a substantial pair-level validation signal, not a sealed result. Both
AV and AR changed, so a cross-scored component decomposition is still needed
to separate better verbalization from critic co-adaptation. The replication
and external-boundary requirements above still apply.

One operational caveat for anyone eyeing the release: the current
no-weights candidate archive accidentally ships an obsolete version of the
exposure audit (an early failed attempt reporting 833 unmapped documents)
instead of the authoritative zero-unmapped v6 report — and the bundle
builder's current config still points at the obsolete file, so a simple
rebuild would repeat the mistake. The archive should not be treated as
final until the config is fixed and it is restaged, re-audited, and
re-attested. The checkpoints themselves also remain internal pending
license and legal review.

## The takeaway

Under a deterministic, verifier-bound protocol: a 30B hybrid model's
residual activation can be compressed into a short English explanation and
recovered — in direction, at near-teacher fidelity, with functional
substitution in the stored forward pass, replicated across independently
trained reconstructors, against controls that rule out the cheap
explanations. And with equal confidence: magnitude isn't recovered, the
boundary is exploratory, and the text isn't verified faithful. The historical
July-8 RL headline is invalid; a separate family-clean actor now shows a strong
matched validation gain, but it is not yet a publication-level RL result. Both
halves are the result.
