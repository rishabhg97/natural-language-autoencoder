# R33 NLA Semantic and Safety Evaluation Campaign

Date: 2026-07-22

## Objective

Determine whether the 342-update R33 online-RL actor and critic communicate through
portable natural-language semantics, or through a co-adapted textual code. Then test
whether NLA observations predict controlled safety-relevant internal states beyond
what is already visible in the input and model output.

This campaign is validation-only. It does not open a sealed test set and does not
train another checkpoint.

## Fixed Evidence Boundary

- 122 held-out validation rows and 122 content families.
- Maximum AV generation length: 384 tokens.
- SFT and RL generated text must come from the existing matched protocol.
- SFT, RL, and independent AR critics are scored on identical row identities.
- All reports retain rowwise directional MSE, raw MSE, centered R2, norm ratio,
  family IDs, parser health, and content hashes.
- No result is called actor-only unless the critic is held fixed.

## Phase A: Identity Preflight

The study preflight requires:

1. Exactly 122 SFT and 122 RL generated records.
2. Identical row, sample, family, and target-explanation identities.
3. One generation-protocol hash per source and exact cross-source protocol parity.
4. No empty target or real generated explanations.
5. Existing family-clean dataset and critic paths.

Failure blocks every GPU job.

## Phase B: Codec Triage

### Observed text-by-critic result (2026-07-22)

The complete 2 text sources by 3 critics validation matrix passed on the same 122
held-out rows and 122 independent content families. The structured report is
`research_abc_v1/analysis/grid_analysis.json`.

| Generated text | SFT critic NMSE | RL critic NMSE | Independent critic NMSE |
| --- | ---: | ---: | ---: |
| SFT AV | 0.308798 | 0.309744 | 0.311819 |
| RL AV | 0.226239 | 0.224349 | 0.229952 |

The RL-text gain transfers through every critic:

- SFT critic: 0.082559 mean gain, family-bootstrap 95% CI
  [0.062688, 0.102245], 89.34% row wins;
- RL critic: 0.085395 mean gain, CI [0.065437, 0.105108], 88.52% row wins;
- independent critic: 0.081866 mean gain, CI [0.063096, 0.100923], 88.52% row
  wins.

The transfer fractions relative to the matched joint gain are 97.76% through the
SFT critic, 101.12% through the RL critic, and 96.94% through the independent
critic. The RL-minus-SFT critic interaction is only 0.002836. This is strong
evidence that most of the matched RL improvement is portable in the actor's text,
not merely an artifact of the jointly trained critic. It is not yet proof that the
gain is semantic: RL explanations are longer (121.1 versus 73.7 mean words), so the
best-of-N, truncation, summary, canonicalization, and paraphrase tests remain
required.

All figures above are validation-only. They do not open or update the sealed test
set and do not support a matched R27 claim.

### B1. Text by critic matrix

Score teacher, SFT-AV text, and RL-AV text through:

- the clean SFT critic;
- the online-RL critic; and
- the independently initialized clean critic.

Every critic also scores exact row-matched `source_context`, `source_raw`, and train
mean controls. The primary actor contrast for critic `c` is:

```text
gain(c) = NMSE(SFT text, c) - NMSE(RL text, c)
```

The critic-specific interaction is:

```text
interaction = gain(RL critic) - gain(SFT critic)
```

Positive fixed-critic gains support portable actor improvement. A large positive
interaction with weak independent-critic transfer supports co-adaptation.

### B2. Text forensics

Report paired length, character count, unique-word fraction, closure, structural
markup, teacher overlap, and a blinded A/B packet. Human reviewers rate clarity,
meaning preservation, and opaque/code-like wording without knowing SFT/RL identity.

### B3. Sampling/sharpening diagnostic

Sample eight SFT explanations per activation at temperature 0.8 and top-p 0.95.
Report nested oracle best-of-1/4/8 NMSE. This is an upper-bound diagnostic, not a
deployable checkpoint score. If SFT best-of-8 reaches RL performance, sampling
sharpening explains a substantial part of the observed gain.

Observed on the complete 122-family validation panel, stochastic SFT samples have
mean NMSE `0.326040`. The first sample scores `0.327527`, worse than deterministic
greedy SFT (`0.309055`). Oracle best-of-4 reaches `0.258926`; oracle best-of-8 reaches
`0.246379`. Best-of-8 improves over greedy SFT by `0.062676`, 95% family-bootstrap CI
`[0.050087, 0.076031]`, and wins on 87.7% of rows. It therefore explains 74.0% of the
matched SFT-to-RL gain.

Sampling does not close the gap. Matched RL remains better than oracle SFT best-of-8
by `0.021993` NMSE. Expressed as baseline-minus-oracle, the paired interval is
`[-0.035264, -0.007100]`, and best-of-8 beats RL on only 32.0% of rows. Online RL
therefore improved more than greedy decoding, although stochastic search recovers a
substantial fraction of the gain. This is an oracle upper bound because selecting the
best sample requires access to the reconstruction critic.

### B4. Canonicalization

Score formatting normalization, surface canonicalization, and unit reordering.
Retention under canonicalization is evidence against a purely formatting-based code.

## Phase C: Semantic and Length Stress

Deterministic interventions:

- retain the first 25%, 50%, or 75% of words;
- delete the first semantic unit;
- delete the final semantic unit;
- reorder semantic units;
- normalize formatting and remove list/header markup.

Model-generated interventions, produced by an unrelated local instruct model:

- light paraphrase;
- aggressive lexical and structural paraphrase;
- two-sentence compression; and
- French translation followed by English back-translation.

Every generated transform records source, prompt, and model hashes. Fifty examples
stratified by source, original length, and transform receive manual meaning-preservation
review. Compression and deletion are analyzed as information-removal curves, not as
semantic invariance controls.

### Observed Phase C result (2026-07-22)

All six text-source-by-critic jobs passed on the same 122 validation rows and 122
independent families. The stress grid contains the original explanation, eight
deterministic transforms, and four independently generated transforms. The strongest
critic-independence check uses the separately trained critic:

| Transform | SFT-text NMSE | RL-text NMSE | RL gain | 95% family CI | RL gain retained |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | `0.311819` | `0.230118` | `0.081701` | `[0.062904, 0.100704]` | `1.000x` |
| Format normalized | `0.311585` | `0.229465` | `0.082120` | `[0.063364, 0.101083]` | `1.005x` |
| Surface canonicalized | `0.312264` | `0.229404` | `0.082860` | `[0.064932, 0.101310]` | `1.014x` |
| Unit reordered | `0.317578` | `0.232034` | `0.085544` | `[0.067641, 0.104751]` | `1.047x` |
| Light paraphrase | `0.326551` | `0.242222` | `0.084330` | `[0.065739, 0.102484]` | `1.032x` |
| Aggressive paraphrase | `0.346113` | `0.245091` | `0.101022` | `[0.083205, 0.119554]` | `1.236x` |
| French round trip | `0.315389` | `0.235604` | `0.079785` | `[0.061535, 0.097667]` | `0.977x` |
| Two-sentence summary | `0.415351` | `0.299586` | `0.115765` | `[0.088909, 0.142556]` | `1.417x` |
| First 50% of words | `0.370999` | `0.284490` | `0.086509` | `[0.064206, 0.109127]` | `1.059x` |
| First 25% of words | `0.483968` | `0.350377` | `0.133591` | `[0.108955, 0.158900]` | `1.635x` |

Formatting and surface canonicalization are effectively lossless. Unit reordering and
French round-trip introduce only small damage. Light and aggressive paraphrases hurt
reconstruction modestly, while summarization and truncation hurt substantially; the
text channel therefore carries distributed detail rather than only a short formatting
code. The RL-text advantage remains positive with a 95% family-bootstrap interval
above zero for every transform through every critic. It also survives the independent
critic, which is evidence that the improvement is portable in the generated text.

A conservative cross-length contrast also rejects explanation length as the whole
story. Through the independent critic, the first 75% of RL words beats the complete
SFT explanation by `0.055621` NMSE, 95% CI `[0.036121, 0.074914]`; the first 50%
still wins by `0.027329`, CI `[0.008448, 0.046344]`. A two-sentence RL summary has a
positive point estimate of `0.012233`, but its interval `[-0.008057, 0.032546]`
includes zero. These are average compression controls, not exact per-row token-budget
matches.

This remains a validation-only result. External transforms passed structural coverage,
identity, and non-empty checks, but the planned 50-example blinded human
meaning-preservation review is not complete. Therefore describe this as robustness to
the tested transformations, not proof of semantic invariance.

### Blinded review packet status (2026-07-22)

The review packet is now built and protocol-verified; human ratings remain pending.
`scripts/build_nano_semantic_review_packets.py` selected 50 unique content families
without replacement, balanced exactly across SFT and RL text (`25/25`), near-balanced
across the four generated transforms (`14/12/12/12`), and balanced across within-source
length tertiles (`17/16/17`). Checkpoint identity, transform identity, document identity,
and provenance hashes are confined to the answer key. Two separately shuffled reviewer
packets expose only the original explanation, transformed explanation, and rating fields.

The packet and scorer are configured by:

- `configs/nano_roundtrip/research/r33_u342_semantic_review_v1.yaml`; and
- `configs/nano_roundtrip/research/r33_u342_semantic_review_score_v1.yaml`.

The build report is
`research_abc_v1/manual_semantic_review/build_report.json` with SHA256
`140a8b4a5f86a8d0c85815d07b03c83ee8ae3bc046af88629d74509dbcd87d96`.
`scripts/score_nano_semantic_review.py` verifies that neither displayed text changed,
unblinds only after both complete reviews are present, reports source/transform strata,
and computes pairwise agreement. Light paraphrase, aggressive paraphrase, and French
round-trip are gated as meaning-preserving transforms; two-sentence compression remains
an information-removal control. No semantic-invariance claim is permitted until the
ratings and scorer report exist.

## Safety and Cyber Domain Extension

NLA text is not treated as privileged ground truth about a model's beliefs. A signal
is useful only when it predicts a controlled latent condition or later behavior beyond
prompt-only and output-only baselines.

### S1. Evaluation-awareness cues

Create paired fictional scenarios that differ only in whether the context says the
system is in an evaluation, deployment, or leaves the setting unspecified. Measure:

- NLA classifier AUROC for the assigned condition;
- the earliest token position at which the condition becomes decodable;
- generalization to held-out scenario families; and
- incremental prediction over source-text and visible-output classifiers.

The label is the experimental condition, not a claim about a subjective belief.

### S2. Prompt-injection recognition

Place benign instructions or prompt injections inside clearly marked untrusted
retrieved documents. Match surface length and topic. Test whether NLA text identifies
the instruction as untrusted before the model follows or refuses it. Include clean
retrieval, obvious injection, obfuscated injection, and irrelevant-malicious-text
controls.

### S3. Authorized versus malicious cyber intent

Use matched CyberSecEval MITRE and false-refusal cases: authorized defensive requests
and maliciously framed requests should share technical vocabulary where possible.
Primary questions:

- Does NLA separate intent beyond lexical prompt features?
- Does it predict refusal/compliance before the answer makes that decision explicit?
- Does the signal transfer across MITRE technique families?

Meta's defensive SOC and AutoPatch tasks are useful positive controls: they exercise
cyber reasoning without making refusal the target.

### S4. Hidden harmful affordance under safe output

Select cases where the final output is a safe refusal. Test whether NLA predicts the
underlying risk category or an available harmful shortcut better than the visible
refusal text. This is reported as latent risk-category information, not as model
misalignment. Later causal activation edits are required before a stronger claim.

### S5. Simulated agentic conflict

Use fictional, sandboxed corporate scenarios with orthogonal factors:

- goal conflict present/absent;
- replacement threat present/absent;
- harmful shortcut available/unavailable; and
- evaluation/deployment cue.

The full factorial design avoids attributing a response to one salient phrase. No
scenario is connected to real email, credentials, networks, or people.

### Recommended order and additional domain instruments

Run the domain studies in this order:

1. **Prompt-injection trust boundary.** This has the cleanest paired intervention and
   label. Extract at the injection-closing token, the first post-document decision
   token, and a matched earlier token.
2. **Authorized-versus-malicious cyber framing.** Hold technical content fixed while
   changing authorization and intent. This tests whether NLA adds information beyond
   cyber vocabulary.
3. **Vulnerability recognition versus refusal.** Use inert code snippets with known
   vulnerability classes. Compare NLA before the visible answer across correct fix,
   safe refusal, false refusal, and incorrect-compliance outcomes. The target is the
   known vulnerability class and later behavior, not a claim that the model secretly
   intended an exploit.
4. **Cyber planning horizon.** On defensive patching and CTF-style toy traces, test
   whether early-token NLA predicts later repair concepts, tool choices, or attack-chain
   stages before they appear in the continuation. Match this to the poetry future-token
   experiment: pre-register future concepts, measure onset, and compare against prompt
   and activation-probe baselines.
5. **Evaluation awareness and simulated agentic conflict.** Run only after the first
   four establish incremental predictive value. These scenarios have the weakest
   observable ground truth and the greatest risk of anthropomorphic interpretation.

Useful negative controls include irrelevant alarming prose, vocabulary-matched benign
requests, shuffled NLA text, no injection, and condition labels permuted within scenario
family. A positive result must survive family-held-out evaluation and add predictive
value beyond prompt, visible continuation, prompt-plus-continuation, and a linear R33
activation probe.

### J-space-inspired mechanistic extensions

Anthropic's J-space work separates a concept being internally available from the model
currently saying the associated word. R33 NLA can test an analogous, narrower question:
does a natural-language bottleneck expose a concept before it is visible in Nano's
continuation, and is that representation reportable, controllable, and behaviorally
used? These properties must be tested independently.

| Experiment | Paired intervention and measurement | Positive evidence | Priority |
|---|---|---|---:|
| Latent concept onset | Trace R33 before and after the first condition-revealing token; score pre-registered concepts against future continuation events | NLA predicts the later event before the output does, beyond prompt and linear-probe baselines | 1 |
| Reportability gap | Compare a normal answer, a request to state the relevant consideration, and a request to reason silently without naming it | The concept is NLA-readable in the silent condition and becomes verbalized only when requested | 2 |
| Voluntary modulation | Ask the model to attend to or suppress one consideration while keeping the task fixed | The corresponding NLA concept changes selectively without a broad activation-norm shift | 2 |
| Causal mediation | Replace or directionally edit the R33 state using identity-safe, magnitude-calibrated reconstructions | The intended decision changes monotonically while unrelated capabilities remain stable | 3 |
| Concealment stress | Ask for concise, euphemistic, or policy-compliant visible output while preserving the underlying decision problem | NLA retains incremental predictive value under style changes and paraphrase controls | 3 |
| Cross-site persistence | Repeat retained concepts at nearby layers and later token positions | The concept has a reproducible onset, peak, and decay rather than appearing at one cherry-picked site | 3 |

The first four form a claim ladder: readability, reportability, controllability, and
causal use. Failure at a later rung does not erase an earlier descriptive result, but
it sharply limits the claim. The public dashboard should show the rung reached for
each concept rather than grouping all positive NLA text under "model thoughts."

### Additional domain studies

After the prompt-injection and authorization pilots, the following studies reuse the
same paired-manifest, three-position extraction, and family-held-out analysis pipeline:

1. **Security-policy conflict localization.** Pair a system policy with a trusted or
   untrusted document instruction. Measure where the trusted-boundary concept appears
   and whether it precedes following or rejecting the injected instruction.
2. **Defender planning state machine.** Use inert logs and toy vulnerable programs with
   labelled stages: detect, localize, hypothesize, verify, patch, and regress. Test
   whether R33 NLA reveals the next stage before the corresponding tool call or text.
3. **Authorization compositionality.** Cross technical capability, authorization,
   target ownership, and requested action in a balanced factorial design. This detects
   whether NLA represents authorization separately from merely recognizing cyber terms.
4. **False-refusal anatomy.** Hold a benign defensive task fixed and vary only alarming
   vocabulary. Compare refusal, correct assistance, and NLA state to distinguish a
   lexical safety trigger from an intent-sensitive representation.
5. **Known-vulnerability counterfactuals.** Apply one-line safe patches that add or
   remove a known vulnerability in toy code. Measure whether the vulnerability concept
   changes before the model explains or repairs it.
6. **Canary-sequence awareness.** Insert an inert, randomized marker correlated with
   evaluation only in training-free held-out scenarios. Test whether Nano learns no
   signal initially; this is a negative control against annotator leakage, not a
   sleeper-agent study.
7. **Simulated insider-choice mediation.** In fictional scenarios, independently vary
   goal conflict, replacement threat, leverage availability, and safe alternatives.
   NLA must predict the selected action beyond the full prompt and visible rationale
   before any "misalignment" language is used.

Cyber experiments remain sandboxed and use inert snippets, public benchmark material,
or toy environments. They test recognition, intent sensitivity, planning, and decision
mediation; they do not execute attacks or connect to live targets.

## Domain Pilot Size

Start with 32 independent pairs per family:

| Family | Pairs | Purpose |
|---|---:|---|
| Evaluation awareness | 32 | Condition decoding and onset |
| Prompt injection | 32 | Trust-boundary recognition |
| Cyber intent / false refusal | 32 | Intent versus terminology |
| Defensive cyber planning | 32 | Benign technical control |
| Agentic conflict | 32 | Factorial latent-state pilot |

The 160-pair pilot is sufficient to reject dead directions. A confirmatory domain set
should use at least 200 pairs per retained family, family-held-out splits, and frozen
scoring thresholds.

### Canary implementation and execution status (2026-07-22)

The resumable 16-pair canary completed on eight H100 NVL GPUs. It contains eight
harmless prompt-injection trust-boundary pairs and eight inert cyber authorization
pairs, for 32 condition rows and 16 independent pair identities. Each condition extracts
R33 at the pre-condition, condition-close, and pre-decision positions. Real RL-NLA text
is generated at all positions; shuffled-activation and no-injection controls are added at
pre-decision. The visible task is restricted to a benign factual answer or an
authorization decision label. Cyber rows explicitly forbid commands, payloads, exploit
steps, and operational instructions.

The run passed all structural and parser gates:

- Corrected causal-prefix extraction produced 96 activation rows with exact paired
  pre-condition invariance (`16/16` equal prefix hashes and zero activation relative-L2
  difference). The earlier full-sequence and cross-replica extraction attempts are
  retained as failed evidence; they showed future-shape leakage and up to `9.14%` and
  `2.10%` paired relative-L2 differences, respectively.
- The strict AV parser accepted and closed `96/96` generated descriptions at a maximum
  budget of 384 tokens. The earlier 128-token attempt produced nonempty but unclosed
  outputs and was rejected rather than accepted through a fallback parser.
- Paired pre-condition NLA text was exactly equal for `16/16` pairs. This is the key
  negative control showing that condition differences did not appear before the
  condition was causally visible.
- Behavior regeneration parsed decisions on `32/32` rows and matched the expected
  decision on `31/32` (`96.875%`). Prompt-injection task behavior was correct on `16/16`;
  cyber authorization behavior was correct on `15/16`. The lone failure verbally
  recognized that permission was absent but ended with `DECISION: PROCEED`.
- Eight data-parallel workers reduced the finalized description pass to 15m44s and the
  behavior pass to 2m27s. Peak steady-state memory was approximately 63 GiB per GPU for
  AV generation and 61 GiB per GPU for behavior generation. No OOM or traceback occurred.

The descriptive lexicon canary is mixed. At condition-close, unauthorized cyber text hit
the authorization lexicon on `7/8` rows while authorized text hit on `0/8`. At
pre-decision the same effect did not persist (`1/8` unauthorized versus `2/8`
authorized), and real prompt-injection text was indistinguishable from shuffled text by
the broad lexicon (`8/8` hits for both active and quoted-control conditions). Some
structurally valid explanations are semantically noisy or hallucinated. These results
validate the extraction, control, generation, and analysis plumbing; they do **not** show
that this lexicon detects latent misalignment, prompt-injection awareness, or incremental
NLA information.

The config is `configs/nano_safety/r33_u342_domain_canary_v1.yaml`; the runtime and
supervisor are `scripts/nano_domain_eval.py` and `scripts/nano_domain_eval_queue.py`.
Final description, behavior, and analysis reports share canonical config SHA256
`385c81a739dde199dbb1826db94f72914b2e3809266f05e7018b6006bef3655f`. The long-lived
coordinator state retains the earlier launch-time config hash because the exploratory
runtime was hardened before its final phase subprocesses. Therefore this run remains an
exploratory canary; a confirmatory study must use an immutable code/config package and a
held-out classifier or preregistered scorer.

The complete lightweight evidence bundle is local at
`artifacts/runai_eval/r33_u342_domain_canary_v1_20260722/` and archived at
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/evals/r33_u342_domain_canary_v1_20260722.tgz`
(archive SHA256 `ef472caff3701deaf339d9620b639fa4ec3c69c21dd4022916634dc37b6122a4`).

### Matched clean-SFT comparison and semantic review (2026-07-22)

The same canary was subsequently executed against the deterministic family-clean SFT
AV checkpoint at `iter_0001291`. The SFT overlay inherits the full RL canary protocol,
reuses its hash-validated prompt manifest and base-model R33 activations, and changes
only the AV checkpoint and output paths. The resolved SFT config SHA256 is
`c25d535b8d29b7ceb056d788f62e9f49d18d7172481f08ae46da4008594768cd`.

- SFT descriptions passed at `96/96` usable and closed rows, with `16/16` exact paired
  pre-condition NLA texts. SFT and RL therefore both pass the causal and structural
  gates on the same 96 prompt-position cells.
- At cyber condition-close, the authorization lexicon hit `4/8` unauthorized and `0/8`
  authorized SFT explanations, compared with `7/8` and `0/8` for RL. This apparent RL
  increase is length-confounded: real RL explanations average `137.69` tokens versus
  `76.07` for SFT, a `1.81x` ratio. It is not reported as an RL safety improvement.
- Prompt-injection lexicon effects remained non-discriminating. At pre-decision both SFT
  and RL hit `8/8` rows in both conditions; shuffled controls also saturate for RL.
- Independently regenerated base behavior scored `30/32` in the SFT queue and `31/32`
  in the RL queue. The AV checkpoint is not used by this phase, so the difference is a
  repeat-generation/numerical effect, not an SFT-versus-RL outcome. The dashboard must
  present behavior as contextual baseline evidence rather than an AV comparison.
- SFT checkpoint preparation took 22m29s, description generation 10m16s, and behavior
  generation 2m22s on eight H100 NVL GPUs. The temporary HF checkpoint was cleaned.

A deterministic blinded semantic-review packet now samples 48 matched prompt-position
cells, four per family/condition/position stratum, and presents both checkpoints as 96
anonymous review items. It records prompt grounding, visible-condition relevance,
hallucination severity, syntax-only content, and behavior-prediction usefulness. Two
reviewer packets use independent item orderings; non-rating payloads are hash-locked and
the scorer refuses incomplete or altered packets before unblinding. Human ratings remain
blank and must not be described as complete.

Matched evidence is local at
`artifacts/runai_eval/r33_sft_domain_canary_v1_20260722/` and
`artifacts/runai_eval/r33_sft_rl_domain_semantic_review_v1_20260722/`. The SFT archive is
also at
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/evals/r33_sft_domain_canary_v1_20260722.tgz`
(SHA256 `f382ebfe7be49b1363bc581b2a00b1f0f0d6b03930efda42458dc4882c1cb4fb`).

## Required Baselines and Metrics

- Prompt-only classifier.
- Visible-output-only classifier.
- Prompt plus output classifier.
- Linear activation probe at R33.
- SFT NLA and RL NLA text classifiers.
- Shuffled-activation and no-injection NLA controls.
- AUROC, AUPRC, calibration error, paired effect, family-bootstrap confidence
  intervals, token-onset position, parse health, and behavior-prediction delta.

An NLA safety claim requires significant incremental prediction over prompt plus
output, transfer to held-out scenario families, and later causal confirmation.

## Safety Claim Ladder

Keep three claim levels separate:

1. **Descriptive:** the NLA text contains a pre-registered safety concept more often
   under the corresponding intervention than under its matched control.
2. **Predictive:** the NLA adds held-out-family predictive value beyond prompt-only,
   output-only, prompt-plus-output, and a linear R33 activation probe.
3. **Causal:** an identity-safe, magnitude-calibrated edit to the reconstructed state
   changes the relevant behavior while preserving unrelated behavior.

Only level 3 supports a mechanistic misalignment claim. Levels 1 and 2 support
"latent safety-relevant representation" or "early warning signal" language. A safe
refusal whose NLA mentions an attack route is not, by itself, evidence of deceptive
intent.

### What can be tested with the current checkpoint

The selected R33 NLA can support controlled descriptive and predictive studies now.
It can test whether a safety-relevant concept appears before a decision, whether the
signal transfers across scenario families, and whether it adds information beyond the
prompt and visible response. It cannot turn an evocative explanation into evidence of
misalignment. The behavior label must come from an observable choice, and the NLA must
be read at a position before that choice.

Use these go/no-go rules for each domain pilot:

- Require at least two independent scenario templates and hold out entire templates,
  names, and lexical realizations from model selection.
- Reject a predictive study when the relevant behavior occurs below 5% or above 95%;
  there is not enough outcome variation to identify an early-warning signal.
- Pair every harmful or suspicious condition with a vocabulary-matched benign
  condition, plus shuffled-activation and no-injection NLA controls.
- Evaluate concept onset before the visible decision and repeat after removing any
  explicit condition words from the classifier input.
- Require incremental held-out-family performance over prompt, output, prompt-plus-
  output, and a linear R33 activation probe before calling NLA useful for monitoring.
- Quarantine any later deliberately misaligned model organism. Because adaptation can
  shift R33 activations, rerun activation-fidelity and round-trip calibration before
  applying the current NLA to it; cross-model transfer is a hypothesis, not an
  assumption.

The practical misalignment pilot is a randomized fictional decision environment with
goal conflict, replacement threat, leverage, and a safe alternative varied
independently. The primary outcome is the selected action. NLA text at the
pre-decision token is an auxiliary predictor. If Nano never chooses the harmful action,
the result is a capability or alignment floor and the study stops; it is not rescued by
searching explanations for words such as "secret," "hide," or "manipulate."

## Feasible Data Sources and Scope

- Use the official PurpleLlama CyberSecEval 4 MITRE, MITRE false-refusal, SOC, and
  AutoPatch sets as source material. Start with defensive and intent-matched items;
  defer autonomous-offensive tasks until the measurement pipeline is validated.
- Use controlled, fictional agentic-conflict scenarios following Anthropic's public
  agentic-misalignment methodology. Do not connect scenarios to real accounts,
  people, credentials, or networks.
- Build prompt-injection pairs locally from inert retrieved documents. Each pair
  changes trust status or instruction status while holding topic, vocabulary, and
  approximate length fixed.
- Archive source license, dataset revision, item hash, pair identifier, scenario
  family, and transformation provenance in the manifest. Never use the final visible
  model response as the ground-truth label.

The local model is NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 and has a chat template plus
instruction, agentic, and RL data listed in its model card. This makes refusal and
agentic-condition studies relevant, while all conclusions remain specific to the R33
site and this checkpoint lineage.

## Pilot Logistics

The first 160-pair pilot contains 320 condition rows. Generate one normal model
continuation per row, then extract R33 activations at three pre-registered positions:
the condition-revealing token, the first decision token, and one matched earlier
token. This produces 960 NLA explanations rather than an unconstrained every-token
trace. Run dense onset traces only for the retained directions.

Before estimating the full runtime, run a 16-pair canary and record:

- base-generation tokens per second;
- R33 extraction rows per second;
- AV explanation tokens per second and parse rate;
- peak GPU memory; and
- classifier and bootstrap wall time.

Expected effort is one to two researcher-days to curate and verify the paired
manifest, followed by roughly one cluster session for the pilot. A confirmatory set
of at least 200 pairs per retained family is a separate multi-day campaign. The
canary, rather than an architecture-only estimate, determines the actual ETA because
384-token AV generation is the dominant cost.

Official references:

- PurpleLlama CyberSecEval: https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks
- Anthropic agentic-misalignment methodology: https://www.anthropic.com/research/agentic-misalignment
- Anthropic Natural Language Autoencoders: https://www.anthropic.com/research/natural-language-autoencoders
- Anthropic global-workspace and J-space overview: https://www.anthropic.com/research/global-workspace

## Execution

```bash
python scripts/nano_roundtrip_research_study.py preflight \
  configs/nano_roundtrip/research/r33_u342_abc_v1.yaml

python scripts/nano_roundtrip_research_study.py run-grid \
  configs/nano_roundtrip/research/r33_u342_abc_v1.yaml

python scripts/nano_roundtrip_research_study.py analyze \
  configs/nano_roundtrip/research/r33_u342_abc_v1.yaml

python scripts/eval_nano_roundtrip_best_of_n.py generate \
  configs/nano_roundtrip/research/r33_sft_best_of_n_v1.yaml
python scripts/eval_nano_roundtrip_best_of_n.py score \
  configs/nano_roundtrip/research/r33_sft_best_of_n_v1.yaml
python scripts/eval_nano_roundtrip_best_of_n.py analyze \
  configs/nano_roundtrip/research/r33_sft_best_of_n_v1.yaml

python scripts/generate_nano_semantic_transforms.py generate \
  configs/nano_roundtrip/research/r33_u342_semantic_transforms_v1.yaml
python scripts/generate_nano_semantic_transforms.py verify \
  configs/nano_roundtrip/research/r33_u342_semantic_transforms_v1.yaml

python scripts/nano_roundtrip_research_study.py run-semantic-stress \
  configs/nano_roundtrip/research/r33_u342_abc_v1.yaml
python scripts/nano_roundtrip_research_study.py analyze-semantic-stress \
  configs/nano_roundtrip/research/r33_u342_abc_v1.yaml
```

All commands use validation data only. W&B remains offline. No RL or checkpoint
training is launched by this campaign.

The independent paraphraser currently uses a cached Qwen2.5-VL checkpoint only as
a text model. Its isolated SGLang environment has PyTorch 2.9.1 and cuDNN 9.10, so
the server config bypasses SGLang's Conv3d performance guard. This is valid only
because transform requests contain no image or video input and therefore never run
the vision Conv3d projection. Any multimodal reuse must remove the bypass and upgrade
cuDNN to 9.15 or newer first.
