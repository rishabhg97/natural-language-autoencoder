# Your gate needs a p-value: notes from adversarially auditing our RL-for-interpretability pipeline

*Archived draft, superseded by the July 8-10 audits. Do not post externally. 2026-07-03.*

**Epistemic status:** Historical internal analysis. Later work found mixed generation
protocols, failed fresh-forward activation fidelity, row-IID inference, and
same-critic checks described here as independent. Numerical artifacts remain
useful for debugging, but no effect in this draft is a release claim.

## TL;DR

- We're building a **natural language autoencoder** (NLA) for a 30B hybrid Mamba-attention model: one model maps a stored activation snapshot to an English explanation, another maps the explanation back to a vector. The primary metric measures directional recovery, not raw-magnitude reconstruction.
- An early RL run appeared to produce a large directional improvement, but the recipe and later comparison protocol were invalid.
- After fixing the recipe, RL passed our promotion gate again. An adversarial audit found the pass was **statistically indistinguishable from zero**: 50.4% row-win rate (p=0.95), and 5 of 256 evaluation rows accounted for ~99.6% of the net improvement.
- The gate passed because its pass condition was `mean(candidate) ≤ mean(baseline)` with margin 0.0. **A point-estimate gate certifies coin flips.**
- Three transferable lessons: gates need significance tests, not point estimates; means are hostage to tails; and if your reward model is also your evaluation model, your primary metric is structurally blind to reward hacking.

## Background: what we're building and why

The setup follows the natural-language-autoencoder recipe from the interpretability literature: take a frozen base model, tap the residual stream at a chosen boundary, and train two adapters. The **Activation Verbalizer (AV)** conditions on an injected activation vector and generates a natural-language explanation of it. The **Activation Reconstructor (AR)** reads that explanation and predicts a vector. The current round-trip metric tests recovery of activation direction on deterministic stored snapshots.

Our variant runs on a 30B-parameter hybrid Mamba/attention/MoE model, tapped at a pre-attention residual boundary (d_model 2688). Both adapters are SFT-trained. Historical `NMSE` is `2*(1-cosine)`; it is a directional metric and ignores magnitude. Raw MSE, centered R2, and norm ratio must accompany it under the current measurement contract.

After SFT plateaued, we added an RL phase: GRPO on the verbalizer, with reward = negative reconstruction error under a **frozen** AR critic. Intuitively: reward the explainer for writing text the reconstructor can actually use.

## Three results, three epistemic states

**Result 1: the seductive one.** A 32-update scouting run improved held-out round-trip NMSE by 26.8%/28.7% (validation/test) over matched SFT, winning 80–85% of rows, t≈13–15 on 512-row paired evals. It beat *gold teacher explanations* by 12–15%. It passed a 26-check composite validity gate including invariance tests, functional (downstream-logit) checks, and a human-reviewed qualitative panel.

Then a recipe audit found two defects. The configured batch of 480 wasn't divisible by the actor's 192-sample quantum, so ~20% of generated samples were silently dropped every update. Worse, the KL regularizer used a *signed* k1 estimator whose value went to −1.4: the "trust region penalty" had become a reward for drifting from the reference policy. Policy drift blew through our guard threshold from update 18 onward, and response length grew 120→179 tokens with r=0.898 correlation to drift. Textbook incentives for the textbook failure.

So: promising historical artifact, invalid recipe and invalid external comparison. Do not scale or cite it as a reconstruction result.

**Result 2: the null one.** We rebuilt the recipe — non-negative k3 KL estimator, exactly divisible batches, runtime drift guards — and ran two 8-update probes (lr 1e-5 and 2e-5). The lr 1e-5 probe **passed the promotion gate**: −1.00% NMSE on validation, −1.79% on test versus matched SFT, all parse-health and control checks at ceiling. The logbook called it "a small but real SFT improvement."

The audit recomputed everything from per-row artifacts:

| | validation (n=256) | test (n=256) |
|---|---|---|
| mean Δ (SFT − RL) | +1.10e-6 | +2.17e-6 |
| 95% bootstrap CI | [−1.24e-6, +3.41e-6] | [−0.53e-6, +5.13e-6] |
| row-win rate | 56.6% (p=0.039) | 50.4% (p=0.95) |
| effect size (dz) | 0.058 | 0.095 |
| median row Δ | ~4e-7 | ~1e-8 |

Both confidence intervals cross zero. The test-split "improvement" concentrates absurdly: the top 5 of 256 rows carry ~99.6% of the net gain; the median row moved by ~10⁻⁸ on a ~10⁻⁴ metric. The minimum detectable effect at 80% power for this eval size is ~3–3.3% relative — the probe was underpowered for the effect it claimed to detect. "Small but real" should have been "consistent with zero."

**Result 3: the invalidated one.** The lr 2e-5 probe's first full evaluation catastrophically "failed" (31% of generations unclosed, 2.3× worse NMSE). Before anyone drew conclusions, a deterministic equivalence check caught the actual cause: the KV-cache generation path diverged from full-prefix recomputation at generated token index 1 — *on the SFT model itself*. The eval was measuring a broken decode path, not the policy. The team quarantined the report, rebuilt the eval on a verified full-prefix backend, and the rerun passed. This is the pipeline working as intended, and it's worth as much as any positive result: **a fail-closed equivalence check between your fast and trusted inference paths will eventually save you from shipping a conclusion about your model that is actually a conclusion about your inference stack.**

## Lesson 1: a point-estimate gate certifies coin flips

Our gate's baseline criterion was `candidate_mean ≤ baseline_mean − margin` with `margin = 0.0`. It computed row-wise win rates — and then never thresholded them. So a candidate that wins 50.4% of rows with a tail-driven mean passes identically to one that wins 85%.

The fix costs nothing: the gate already has paired per-row data. Require the bootstrap CI of the paired delta to exclude zero, and the win rate to exceed 50% on both splits. Our historical update-32 run passes this trivially (80–85% wins); the corrected probes don't pass at all — which is the correct answer.

The general form: **any promotion gate comparing two models on the same rows should be a paired statistical test, not a comparison of two floats.** You have the row-level data. Use it.

## Lesson 2: means are hostage to tails

−1.79% sounds like a real (if small) effect. "5 rows out of 256 are the entire effect and the median row is unchanged" sounds like what it is. Report both. If your metric is a mean of a heavy-tailed per-row quantity (reconstruction errors very much are), the median, trimmed mean, and win rate aren't supplementary statistics — they're the difference between an effect and an anecdote.

## Lesson 3: reward model = eval model ⇒ your metric can't see reward hacking

Our RL reward and our promotion gate use the *same frozen AR checkpoint*. Anything the policy learns that exploits that critic's idiosyncrasies — rather than genuinely improving explanations — transfers perfectly to the gate. The primary metric is blind to the failure mode by construction.

The functional and invariance checks used in this draft are not critic-independent:
they rescore vectors produced by the same frozen AR that supplied RL reward. The
human panel is qualitatively independent but does not validate activation
recovery. A defensible cross-critic check requires a second independently
initialized and independently trained AR under the same protocol.

## Lesson 4: silent defaults are where the bodies are

Two findings that changed no headline but could have wrecked the next run: a debug flag (`--skip-grad-norm`) had disabled gradient clipping entirely *and* replaced the grad-norm telemetry with a constant 0.0 — no safety valve, and no way to notice — while the unclamped k3 KL estimator (whose gradient passes through an exponential) spiked to 600 in one probe. And the data loader's shuffle flag defaulted off, so both probes trained on the first 384 rows of a 247,700-row dataset in file order: 0.15% of the pool, clustered into ~40 documents. Neither was visible in any dashboard. Both were one config line to fix. Audits that only read results miss these; audits that read *configs and logs against code* find them.

## Where the real headroom is

The most sobering number in the project isn't an RL number. Gold teacher
explanations reach about `0.28` directional MSE, while source-derived controls
can reach about `0.08-0.10`. This is evidence of a large directional information
gap, not a calibrated raw-space information ceiling. Better explanation formats
and objectives may matter more than policy polish on the current format.

## What would change our mind

One experiment decides the RL question: a 32-update run of the *corrected* recipe, evaluated with the fixed gate (paired CI > 0 required, win rate > 50%, cross-critic audit, KL and degeneration stop conditions). If it reproduces a double-digit fraction of the historical effect under valid dynamics, RL earns its scale-up. If it comes back at another statistically-null 1–2%, then the interesting hypothesis is that the *broken* ingredients were load-bearing — that the historical gains required more policy drift than a correct trust region permits — and the right successor is off-policy (best-of-k distillation), which captures the same selection pressure without the instability.

Either outcome is informative. That's what the gate redesign buys: for ~50 GPU-hours, an experiment that cannot return an ambiguous answer.

## Takeaways

1. Promotion gates comparing models on shared rows should be paired statistical tests. Margin-zero mean comparisons certify noise.
2. Report medians and win rates next to means; heavy-tailed metrics make means unreliable narrators.
3. If reward and evaluation share a model, add an independent scorer before believing any RL gain.
4. Deterministic equivalence checks between fast and trusted inference paths are cheap and will eventually save a headline result.
5. Grep your launch commands for debug flags and your configs for defaulted-off shuffles. The bugs that survive four rounds of review are the ones that log a plausible constant.
6. Decompose your headroom before optimizing. We nearly spent 150 GPU-hours polishing the small slice.
