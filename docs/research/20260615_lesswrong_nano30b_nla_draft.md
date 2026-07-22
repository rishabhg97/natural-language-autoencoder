# Natural-Language Autoencoders for a Hybrid MoE Model: A Nano30B Progress Report

Suggested subtitle: What we learned trying to make hidden activations reconstructable through natural-language explanations.

Suggested tags: mechanistic interpretability, alignment, natural-language explanations, autoencoders, MoE, representation learning

Epistemic status: progress report, not a victory lap. We have a working AV/AR training and evaluation loop, a clean R33 candidate that passes in-run round-trip controls, and several useful negative findings about leakage, packed training, and proxy metrics. We do not yet have the strongest public claim that the R33 natural-language autoencoder beats a row-matched R27 baseline end-to-end. We should not frame this as "solved NLA" or "RL-ready"; it is a credible milestone toward a larger hero run.

## TL;DR

We are trying to build a natural-language autoencoder (NLA) for a large hybrid model: take an internal activation vector `h`, decode it into a short natural-language explanation `z`, and then reconstruct the activation from the explanation:

```text
h -> AV-generated explanation z -> AR reconstruction h_hat
```

The motivation is simple: if a model's internal state can be compressed into language and then reconstructed well enough to preserve the original direction, language becomes a testable interface to hidden computation. The intended alignment payoff is not that the explanation is automatically faithful because it is fluent. The payoff is that we can put a hard bottleneck in the middle and ask whether the text carries row-specific, causally useful information about the activation.

The current state:

- We adapted the NLA reference stack to a Nano30B / Nemotron-H-style hybrid architecture with MoE and Mamba-ish complications.
- We built two supervised components:
  - **AV**, the activation-to-verbalization side: inject `h` and train the model to produce `<explanation>...</explanation>`.
  - **AR**, the activation-reconstruction side: feed an explanation to a truncated critic and train a value head to reconstruct `h`.
- We now treat the actual round trip, `h -> generated text -> reconstructed h`, as the gate. AV NLL and teacher-text AR reconstruction are only proxies.
- The best clean R33 candidate currently has:
  - AV real NLL `1.003093 / 0.976148` on validation/test, beating shuffled, zero, mean, and no-injection controls.
  - Round-trip AV-real NMSE `0.000128805 / 0.000135702` on validation/test under the gate scorer, beating all in-run controls.
  - Teacher-text round-trip NMSE `0.000111481 / 0.000121767`, so generated explanations are close to, but still weaker than, teacher text.
  - Perfect parse health on the evaluated rows: closed fraction `1.0`, usable fraction `1.0`, empty fraction `0.0`.
- The strongest caveat: the strict gate still records `passed=false` because the R27 comparison was not row-identical. Internally we are treating the R33 pair as the current final candidate for hero planning, but externally we should not claim "R33 definitively beats R27 end-to-end" until a row-matched baseline is restored.

The main lesson is that the experiment became scientifically meaningful only after we stopped treating teacher-forced proxy metrics as the goal and made the round trip the object of selection.

## What We Are Trying To Build

A natural-language autoencoder is a two-leg system:

1. **Activation to verbalization (AV)**: Given an internal activation vector `h` from a selected layer or boundary, generate a short explanation `z`.
2. **Activation reconstruction (AR)**: Given `z`, reconstruct the original activation vector as `h_hat`.

The target behavior is not merely "the generated text sounds plausible." The target is that the text transmits enough information that an independently trained reconstruction model can recover the original activation better than a family of controls.

The controls matter. A plausible-looking explanation is not meaningful if:

- a shuffled explanation works just as well,
- a mean activation works just as well,
- zero or no injection works just as well,
- the generated text fails to parse,
- the dataset leaks the same source content across train and heldout splits,
- or the model can exploit doc-level topic rather than position-specific content.

The key scientific object is therefore not an example explanation. It is the controlled round-trip report.

## Why This Is Interesting For Alignment

Many interpretability tools expose latents, circuits, sparse features, probes, or causal interventions. These are valuable, but they often leave us with a translation problem: the thing we measure is not naturally the thing we can ask the model or a human to reason about.

An NLA asks whether we can force a model-state bottleneck through ordinary language:

```text
internal state -> text -> internal state
```

If this works, it gives us a new kind of object:

- a natural-language description that is scored by reconstruction, not by vibes;
- a way to ask which parts of an activation are linguistically compressible;
- a way to compare layers by how much row-specific information survives a text bottleneck;
- a possible reward model for later self-distillation or RL, where the reward is not "sound good" but "reconstruct the hidden state."

This is not a guarantee of semantic faithfulness. It is closer to a controlled compression test. If the explanation reconstructs the activation and beats hard controls, it is evidence that the text carries activation-specific information. If it does not, the explanation is at best a label or a summary.

## The Model Setting

The target is a large hybrid architecture rather than a small transformer toy model. This matters because the engineering failure modes are much richer:

- the model wrapper and module names differ from the reference implementation;
- residual boundaries have to be pinned exactly;
- Mamba-style state handling can leak across packed samples if sequence boundaries are not respected;
- MoE routing introduces dtype, balance, and remote-code patching concerns;
- the activation injection path has to survive FSDP/Miles training, checkpoint conversion, and heldout evaluation;
- chat templates and "thinking" modes can silently change the prompt distribution.

We selected and compared candidate residual boundaries, especially R27, R33, and R34. The current layer decision favors R33: in early AR probes, R33 was substantially easier to reconstruct from teacher text than R34 and better than the mature R27 fallback trajectory.

The current working hypothesis is that R33 is a promising post-Mamba, pre-attention boundary. That may be structurally important for hybrid MoE/Mamba models, though we should not overclaim until sibling boundary probes such as R26 and R42 are run.

## The Training Setup

### AV: activation to explanation

The AV side is trained by injecting an activation vector into a prompt at a marked position and supervising the model to emit a teacher-style explanation. The important evaluation is not just likelihood on the target explanation; it is whether the real activation produces lower NLL than controls:

- real activation,
- shuffled activation,
- zero activation,
- mean activation,
- no injection.

The mature R27 AV-SFT run is AV-only, not AV+AR. It is still useful as a strong proof that the AV path can learn a row-specific likelihood signal. In the bounded `64/64` heldout eval, R27 real activation NLL beat all controls:

| Split | Real | Zero | Mean | No injection | Shuffled |
|---|---:|---:|---:|---:|---:|
| Validation | `0.904621` | `1.174285` | `1.192388` | `1.344115` | `1.329755` |
| Test | `0.956531` | `1.216016` | `1.230179` | `1.357744` | `1.349336` |

The gaps were large: roughly `0.26-0.44` NLL depending on split and control. This says the injection channel can carry row-specific signal. It does not, by itself, say that free generations reconstruct the original activation.

### AR: explanation to activation

The AR side is a critic-like model trained to reconstruct the activation vector from text. We report normalized MSE, cosine, and FVE-like diagnostics against controls such as mean, source context, source raw, blank/generic text, and shuffled teacher explanations.

This leg quickly taught us an important lesson: teacher text is a bottleneck. Source-prefix replay or source_raw controls can recover much more of the activation than short teacher explanations. For example, in R33 scouting:

| Run | Validation teacher NMSE | Test teacher NMSE | Validation source_raw NMSE | Test source_raw NMSE |
|---|---:|---:|---:|---:|
| R27 tuned fallback | `~0.441` | `~0.437` | `~0.133` | `~0.130` |
| R33 20k mini probe | `0.381983` | `0.388301` | `0.071066` | `0.076216` |
| R34 20k mini probe | `0.490728` | `0.501399` | `0.112821` | `0.119030` |

This is why we now talk about an information ceiling. If source_raw can recover the vector much better than teacher text, the activation contains information that the current explanation format is not transmitting.

### The round-trip gate

The actual target is:

```text
h -> AV-generated explanation -> AR reconstruction
```

The gate reports:

- AV checkpoint and AR checkpoint,
- generated-text parse/closure health,
- teacher-text AR reconstruction on the same rows,
- AV-generated reconstruction,
- controls such as shuffled, zero, mean, no-injection, AV-mean,
- rowwise win rates,
- and baseline comparison when row identity matches.

This became the main methodological correction of the project: optimize the thing we actually care about. AV likelihood and AR teacher-text reconstruction are useful diagnostics, but they can rank candidates differently from the round trip.

## Current Evaluation Status

### R33 layer and dataset status

R33 is the current scaling target. R27 remains the mature fallback. R34 had a slightly better early AV NLL in one small probe, but its AR trajectory was much weaker, which matters because the round trip is AR-scored.

The original R33 fullscan dataset had `275,396` rows. A later audit found duplicate source content under distinct doc IDs, so doc-ID split checks were insufficient. The clean strict content-dedup target keeps `56,351` rows and has:

- `d_model = 2688`,
- nonfinite activations `0`,
- empty explanations `0`,
- content-hash cross-split overlap `0` for both AR and AV verifier paths.

This is a clean smoke/gate dataset, not yet the desired full hero-scale dataset. A future full-scale dataset should preserve duplicate-content components within a single split instead of dropping most of them.

### R33 AR status

The pre-fix R33 full275k AR scouting result was very strong:

| Split | Teacher NMSE | Source_raw NMSE | Source_context NMSE | Mean NMSE |
|---|---:|---:|---:|---:|
| Validation | `0.277565` | `0.096948` | `0.304264` | `0.695462` |
| Test | `0.276665` | `0.091568` | `0.283364` | `0.672285` |

But this was pre-fix. It trained before the packed-boundary contamination fix and before LR-decay canary evidence was reliable. We should treat it as a useful scouting signal, not as a clean hero checkpoint.

The current clean R33 AR candidate is:

- strict-dedup dataset,
- padded critic path,
- checkpoint at iter `0000128`,
- bounded `512/512` teacher NMSE `0.361513 / 0.352040`.

This is good enough for the current AV round-trip smoke and hero planning. It is not the same as reproducing the pre-fix full275k `0.277` result cleanly.

### R33 AV and round-trip status

The current selected clean R33 AV candidate is a 20k/dedup smoke checkpoint:

- LR `1e-4`,
- cosine warmup,
- global batch `192`,
- microbatch `2`,
- sequence cap `1152`,
- dynamic cap `512`,
- checkpoint at iter `0000032`.

AV proxy eval:

| Metric | Validation | Test |
|---|---:|---:|
| AV real NLL | `1.003093` | `0.976148` |
| AV shuffled NLL | `1.430727` | `1.397634` |
| AV zero NLL | `1.321181` | `1.289832` |
| AV mean NLL | `1.313027` | `1.272369` |
| AV no-injection NLL | `1.382992` | `1.351784` |

Round-trip eval:

| Metric | Validation | Test |
|---|---:|---:|
| AV-real round-trip NMSE | `0.000128805` | `0.000135702` |
| Teacher-text round-trip NMSE | `0.000111481` | `0.000121767` |
| Mean-control round-trip NMSE | `0.000233943` | `0.000241159` |
| Shuffled-control round-trip NMSE | `0.000327986` | `0.000335947` |

Parse health:

| Split | Closed fraction | Usable fraction | Empty fraction |
|---|---:|---:|---:|
| Validation | `1.0` | `1.0` | `0.0` |
| Test | `1.0` | `1.0` | `0.0` |

Read this as: the first clean R33 AV checkpoint has a healthy actual `h -> generated explanation -> reconstructed h_hat` signal. Generated explanations are not quite as strong as teacher text, but they beat all in-run controls. Rowwise control wins were essentially saturated, usually `63/64` or `64/64` depending on the control and split.

Important caveat: the report-level gate is still `passed=false` because the R27 baseline comparison is not row-identical. That field should not be read as an in-run control failure. It should be read as: "do not make the strongest external comparison claim yet."

### R27 baseline status

The R27 round-trip evaluator and queue plumbing now work. The `64/64` full-control R27 baseline passed, and a later `256/256` full-control baseline also completed with `gate.passed=true` in the current state notes:

- validation/test AV-real normalized MSE `0.000180003 / 0.000175571`,
- teacher normalized MSE `0.000157706 / 0.000155285`,
- closed/usable parse fractions `1.0`,
- AV-real beating all controls.

However, the currently selected R33 report did not use a row-identical R27 comparison. For internal planning, that was waived. For public claims, it should be restored.

## What We Learned

### 1. Teacher-forced metrics are not enough

The project started with natural proxy metrics:

- Does AV assign lower NLL to the target explanation when given the real activation?
- Does AR reconstruct the activation from teacher text better than controls?

Both are useful. Neither proves the NLA works. The AV can learn a likelihood signal that does not survive free generation. The AR can learn to reconstruct from teacher text even if the AV cannot generate equally informative text.

The round trip is the first metric that really asks the target question.

### 2. Teacher explanations are an information bottleneck

Source_raw controls are consistently much stronger than teacher text. That means the activation contains row-specific information that the current explanation style is not carrying.

This does not make the project less interesting. It makes it more diagnostic. The gap tells us where to work:

- If teacher text reconstructs well but generated text does not, AV generation is the bottleneck.
- If both teacher and generated text are weak compared with source_raw, the explanation format is the bottleneck.
- If controls are close to real text, the evaluation is probably measuring topic or dataset leakage rather than activation-specific content.

### 3. The R33 boundary looks promising

R33 beat R34 and the mature R27 fallback in early AR probes. The source_raw floor was also much better at R33. This suggests that the selected boundary is not arbitrary.

The speculative structural story is that post-Mamba, pre-attention boundaries may be especially text-reconstructable in this hybrid architecture. We should treat that as a hypothesis, not a result, until sibling boundaries such as R26 and R42 are tested.

### 4. Content dedup matters more than doc IDs

The old split verifier checked document overlap, but later auditing found duplicate source content under distinct doc IDs. That invalidated a stronger reading of pre-fix results.

The clean dataset now uses verifier-equivalent content dedup and has zero content-hash cross-split overlap. This is a painful reduction in row count, but it made the evals much more credible.

Lesson: for language-model data, "different document ID" is not a sufficient independence assumption.

### 5. Packed training can silently break hybrid models

Packed THD critic training was not equivalent to the padded reward/eval path for Nano/Nemotron-H. Live guards found reward/train MSE-ratio divergence around `14-22%` in failed packed smokes. After a padded masked critic path with explicit last-token value indexing, a proof run hit:

```text
reward/train MSE ratio mean=1.0000 max|r-1|=0.0000
```

This was one of the most important engineering lessons. In a hybrid model with recurrent/stateful components, packing is not just a throughput detail. It can change the computation.

### 6. LR schedule evidence needs to be explicit

One clean-looking AR run later failed the LR-decay canary because the final LR was flat. The model even had encouraging diagnostic evals, but the run could not be used as promotion evidence.

This is the right failure mode. A public interpretability claim should not rest on "the config said cosine." It should rest on run-local evidence that the schedule actually happened.

### 7. Parse health is a first-class metric

An early R27 round-trip plumbing smoke used too few generation tokens and produced unclosed explanations. The numeric result existed, but it was not a meaningful quality baseline.

The current R33 candidate is much healthier: closed and usable fractions are both `1.0`, with no empty generated explanations on validation or test. This is exactly the kind of mundane detail that determines whether a round-trip number is interpretable.

### 8. Round-trip should become the HPO objective

The workflow change that matters most is making round-trip NMSE part of AV selection, not a final ceremony after AV HPO is done.

AV candidates with similar NLL can differ in generation quality, parse discipline, and reconstructability. A bounded round-trip eval per candidate is more expensive than a proxy NLL eval, but it directly ranks the thing we care about.

### 9. Do not start RL yet

The reference NLA work used RL-style optimization later in the pipeline, and it is tempting to jump there. We should not.

The current rule is: no RL until the round-trip gate passes against the relevant baseline. Before that, the best next step beyond SFT is probably AR-scored rejection-sampling self-distillation:

1. sample several explanations per row from AV,
2. score each with frozen AR,
3. keep the best explanation,
4. SFT AV on those selected generations.

This is cheaper and simpler than full RL, and it tells us whether AV has exploitable sampling variance.

## What This Does Not Show Yet

This work does not yet show that:

- R33 definitively beats R27 end-to-end on row-identical heldout rows;
- the explanations are semantically faithful in every human sense;
- the generated text captures all, or even most, of the activation information;
- the current 20k/dedup AV smoke will scale cleanly to a full hero run;
- the method is robust across many layers or many model families;
- RL would improve the system.

The most honest claim is narrower:

> We built a controlled AV/AR natural-language bottleneck for a hybrid MoE model, found a promising R33 boundary, fixed several serious evaluation and training confounders, and obtained a clean R33 round-trip smoke where AV-generated explanations reconstruct better than shuffled, mean, zero, no-injection, and AV-mean controls.

That is enough to justify the next hero-scale run. It is not enough to declare victory.

## Why I Think This Is A Useful Research Direction

The alignment relevance of NLAs is not that the model tells us a story and we believe it. The relevance is that we can score the story against the hidden state.

This creates a nice pressure:

- The text must be parseable.
- The text must be row-specific.
- The text must beat controls.
- The text must reconstruct through a separately trained leg.
- The text can be ablated, shuffled, stripped, anchored, or contrasted.

That is a better epistemic situation than simply asking a model why it did something.

There is also a useful failure mode. If the activation cannot be compressed into language, or if the text only transmits topic-level information, we learn that too. In fact, a substantial part of the project so far has been converting vague optimism into crisp bottleneck diagnoses.

## Next Experiments

### Restore the row-matched R27 comparison

Before any external claim that R33 beats R27 end-to-end, rerun or align the R27 baseline on the exact rows used by the R33 candidate. This converts the current internal planning decision into a public comparison.

### Run the hero-scale R33 plan

The clean candidate should be frozen and used to plan a larger R33 run. The hero run should require:

- clean content split evidence,
- checkpoint-local remote-code patch report,
- packed-vs-padded or padded-path correctness evidence,
- LR-decay canary,
- AV proxy eval,
- AR teacher-text eval,
- round-trip eval with full controls,
- parse health,
- rowwise win rates,
- and a baseline comparison.

### Make round-trip NMSE the AV HPO metric

Use bounded round-trip evals as the AV selection criterion. Do not choose AV heroes on NLL alone.

### Probe the explanation information ceiling

The source_raw vs teacher-text gap suggests that the current explanation format is losing information. Two cheap ablations are especially appealing:

- **Lexical anchors**: append declared token/position/local-continuation hints to the teacher explanation. This tests whether the missing information is mostly local lexical detail.
- **Teacher format v2**: ask for more contrastive, position-specific explanations, such as what distinguishes this token position from nearby positions in the same document.

Both should be tested first on AR. If they do not help AR, there is no reason to pay for AV regeneration.

### Run boundary-family probes

Test structural siblings such as R26 and R42. If post-Mamba, pre-attention boundaries cluster together, that becomes a real finding about hybrid architectures rather than a one-layer anecdote.

### Try AR-scored self-distillation before RL

If the round-trip gate is healthy but AV generations lag teacher text, use best-of-k selection with frozen AR before moving to full RL. This is the natural intermediate step:

```text
generate k explanations -> score with AR -> keep best -> SFT AV
```

If best-of-k barely improves the score, full RL probably has little exploitable headroom.

## A Proposed Standard For Reporting NLA Results

After this pilot, I would not trust an NLA result unless it reports:

1. The activation boundary and exact prompt/template mode.
2. Train/validation/test split policy and content-overlap checks.
3. AV real-vs-control NLL.
4. AR teacher-text reconstruction against controls.
5. The actual AV-generated round trip.
6. Parse/closure/usable/empty generation rates.
7. Rowwise win rates, not just aggregate metrics.
8. Teacher-text vs generated-text reconstruction on the same rows.
9. Source_raw or other upper-bound controls where available.
10. Explicit caveats for non-row-identical baselines.

The strongest version of the field should make it normal to say "this explanation sounds good but fails the bottleneck" or "this text reconstructs well but mostly because it leaks lexical anchors." Those are not embarrassing failures. They are the point of having a measurement.

## Closing

The current Nano30B pilot has reached the point where the core loop is real:

```text
activation -> generated explanation -> reconstructed activation
```

The best clean R33 candidate passes in-run controls, parses cleanly, and gives us a credible next target. The work also exposed several ways we could have fooled ourselves: duplicated content under different doc IDs, packed-training mismatch, flat LR schedules, teacher-forced proxy metrics, and generation parse failures.

My current view is that natural-language autoencoders are promising less because they produce beautiful explanations and more because they make explanations accountable. They turn "what does this activation mean?" into a family of falsifiable reconstruction questions.

That feels like the right kind of interface to keep building.
