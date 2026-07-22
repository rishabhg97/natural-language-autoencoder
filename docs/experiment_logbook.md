# Nano30B NLA Pilot Experiment Logbook

<!-- R33-HERO-BASELINE-PROTOCOL-INVALIDATED -->

> [!CAUTION]
> Publication status (`2026-07-15`): the deterministic family-clean R33 SFT
> AV+AR checkpoint pair is qualified for directional reconstruction and
> stored-snapshot functional recovery. The July 8 `30.97% / 32.34%`
> RL-over-SFT comparison remains historical and invalidated. See
> `docs/runs/r33_clean_sft_av_ar_20260715.md`.

Date compiled: 2026-07-15

This logbook summarizes the Nano30B Natural Language Autoencoder pilot from a
scientific and mathematical perspective. It is not a code walkthrough. The goal
is to record what we tried, what each experiment actually tested, what the
measured results were, and why the research direction changed.

Primary evidence sources:

- `README.md`
- `docs/nano30b-nla-core-plan.md`
- `docs/execution_log.md`
- `docs/nano_av_run_history.md`
- `docs/nano_av_job_tracker.md`
- `docs/runai_miles_fsdp2_integration.md`
- `docs/runai_av_100k_repeatability.md`
- `docs/qwen_nla_inference_qc_report_20260519.md`
- `docs/handover_20260519_nano30b_nla.md`
- `docs/issues_iter1.md`
- `docs/runs/r33_rl_hero_20260708.md`
- `experiment_0523.md`
- `runs/wandb_offline/av-r27-99570-rslora-r192-lr1e5-20260526T1645Z/summary.json`
- `artifacts/runai_eval/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gloo-tokenized-gb192-mb8-save100-20260528T0110Z/eval_iter_0000467_v64_t64_report.json`

## 0. Publication Remediation Snapshot

The deterministic 275,396-row R33 snapshot and exact full replay pass, while
the 32-row fresh-forward fidelity gate fails on every row (`2.562%` mean,
`7.728%` max relative L2). The v2 boundary contains `5,009` content families;
clean SFT training uses `4,504`, and validation/test use disjoint sets of
`250 / 255` families.

The selected clean AR and corrected-packed AV checkpoints each complete 1,291
updates and pass hash-bound component validation. The protocol-matched
generated-text round trip then passes on 512 validation and 512 test rows:
candidate directional MSE is `0.307004 / 0.319225`, teacher text is
`0.304714 / 0.302637`, parse usability is `100%`, and all controls have
positive family-bootstrap lower bounds. Stored-snapshot functional reinjection
is statistically indistinguishable from teacher reconstruction and decisively
better than mean, zero, and shuffled controls.

Release `r33-clean-sft-av-ar-iter1291-20260715` is therefore a qualified R33
supervised AV+AR NLA pair for directional and stored-snapshot functional
claims. Raw magnitude is not recovered, the test is not guaranteed pristine
across all historical project work, and no clean RL gain is claimed.

## 0A. Historical Internal RL Milestone (2026-07-08)

The completed R33 RL checkpoint `iter_0000342` from
`r33-corrected-k3-hero-lr1e5-update342-resume228-retry3` remains an internal
systems/scientific milestone, but its headline effect was invalidated for
publication by the mixed generation protocols and later activation audit.

| Quantity | Result |
|---|---|
| Queue status | complete at `2026-07-08T03:20:52Z` |
| RL topology | `6` actor + `1` rollout + `1` frozen critic H100 NVL GPUs |
| Selected recipe | constant LR `1e-5`, K3 coefficient `0.001`, global batch `384`, microbatch `32` |
| Selected lineage | `342` updates, `131,328` generated responses |
| Final gate | `512/512`, passed |
| Validation RL/SFT NMSE | `0.000087528 / 0.000126796` (`30.97%` improvement) |
| Test RL/SFT NMSE | `0.000091176 / 0.000134752` (`32.34%` improvement) |
| Validation/test rowwise wins | `83.40% / 88.67%` |
| Validation/test usable generations | `100% / 100%` |
| Selected checkpoint | `/workspace/interp/outputs/nano30b-nla-pilot/rl_hero/r33_corrected_k3_hero_lr1e5_update342_resume228_retry3/actor/iter_0000342` |

The then-configured SFT dataset hashes and row identities passed on both
splits, all generation controls were beaten, and document-clustered confidence
intervals were positive. The old gate did not enforce generation-protocol
identity between candidate and baseline, so these numbers are not the current
effect estimate.

Canonical report: `docs/runs/r33_rl_hero_20260708.md`.

## 0B. Historical Milestone Snapshot (2026-05-28)

Current repo milestone:

| Quantity | Status |
|---|---|
| Commit | `a90826d` |
| Commit message | `nano30b: record AV checkpoint milestone` |
| Branch state | `main`, ahead of origin by `5` commits |
| Scientific milestone | completed Nano30B AV-SFT hero checkpoint |
| Explicit non-claim | not AV+AR; AR/critic SFT has not been launched |

The project has crossed the AV checkpoint milestone. Nano30B AV-SFT is now
operational through Miles/FSDP2: full-data training completed, exact-resume
checkpointing works, HF conversion exists, W&B sync exists, and bounded
row-specific heldout eval is positive.

Final AV checkpoint:

```text
/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-hero/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gloo-tokenized-gb192-mb8-save100-20260528T0110Z
```

Checkpoint artifacts:

| Artifact | Path | Approx. size | Role |
|---|---|---:|---|
| Exact-resume DCP checkpoint | `checkpoints/iter_0000467` | `177G` | resume-capable training state |
| Converted HF model-only checkpoint | `hf_iter_0000467` | `59G` | lightweight eval / export checkpoint |
| HF index reports | `hf_iter_0000467` | `63,155,880,576` bytes | `31,577,937,344` parameters |

Model-size interpretation:

The Nano AV actor is full parent-size Nano30B, not truncated. The HF index
reports `31,577,937,344` parameters, matching the parent Nano model. This
matches the Qwen reference pattern: Qwen AV is also parent-size
(`kitft/nla-qwen2.5-7b-L20-av` and `Qwen/Qwen2.5-7B-Instruct` have the same
effective parameter count/size). By contrast, Qwen AR is truncated:
`kitft/nla-qwen2.5-7b-L20-ar` has `5,439,211,008` parameters, about `0.7143`
of the Qwen parent. The Nano AR/critic path should therefore be a separate
critic-style checkpoint, ideally truncated to the reconstruction boundary if
the Nano model path supports that cleanly.

Bounded heldout evaluation:

| Split | Real | Shuffled | Zero | Mean | No injection |
|---|---:|---:|---:|---:|---:|
| Validation NLL, `64` rows | `0.9046` | `1.3298` | `1.1743` | `1.1924` | `1.3441` |
| Test NLL, `64` rows | `0.9565` | `1.3493` | `1.2160` | `1.2302` | `1.3577` |

| Split | Shuffled gap | Zero gap | Mean gap | No-injection gap |
|---|---:|---:|---:|---:|
| Validation | `+0.4251` | `+0.2697` | `+0.2878` | `+0.4395` |
| Test | `+0.3928` | `+0.2595` | `+0.2736` | `+0.4012` |

Interpretation:

This is a decent and scientifically promising AV checkpoint: real activations
beat shuffled, zero, mean, and no-injection controls on both bounded heldout
splits. The claim remains bounded because the eval is `v64/t64`, not full
validation/test, and because generation parse/F1 has not yet been run from
`hf_iter_0000467`.

W&B status:

- Synced run:
  `https://wandb.ai/rishabhga97/nano30b-nla-pilot/runs/dw7mp5sn`
- Local snapshot:
  `artifacts/runai_wandb/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gloo-tokenized-gb192-mb8-save100-20260528T0110Z`
- Current limitation: training W&B primarily contains train metrics; heldout
  eval metrics were added to reports and docs after the fact. Future runs
  should log validation/control metrics directly to W&B.

Immediate objective:

Implement and run Nano30B AR-SFT, Qwen-faithful, to produce a separate AR
checkpoint. Do not start RL yet. The AR path should train explanation text
`z -> h` reconstruction with the Qwen custom MSE loss
`nla.loss.nla_critic_loss`, preserve the same 99,570-row source and doc-level
split discipline, begin with TDD plus tiny/small and medium-small gates, and
require heldout reconstruction metrics against controls before any
complete-performance AR run.

## 1. Scientific Contract

The target object is a single-site Natural Language Autoencoder for
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.

The intended NLA loop is:

```text
frozen target Nano: x, tau -> h_b = R_b^target(x)_tau
AV actor:           h_b -> z
AR critic:          z -> h_hat_b
evaluation:         h_hat_b should reconstruct h_b through the text bottleneck
```

Here:

- `x` is source text.
- `tau` is the explained token position.
- `R_b` is the raw residual stream at boundary `b`.
- `h_b in R^2688` is the target activation.
- `z` is a natural-language explanation.
- `h_hat_b` is the reconstruction produced from explanation text only.

The most important rule is that AR must reconstruct from `z`, not from the
source context `x`. Source replay is allowed as an oracle and implementation
diagnostic, but it is not the NLA task.

Residual boundary notation used throughout:

```text
R_0 = Embed(x)
R_{i+1} = R_i + F_i(RMSNorm_i(R_i))
F_i in {Mamba2_i, MoE_i, GQA_i}
```

The normalized reconstruction objective is direction-oriented:

```text
N_s(v) = sqrt(d) * v / (||v||_2 + eps)

L_AR = (1/d) ||N_s(h_hat_b) - N_s(h_b)||_2^2
     = 2(1 - cos(h_hat_b, h_b))
```

Because this loss is direction-only after normalization, the project records
relative reconstruction improvement against a train-split mean baseline rather
than classical variance explained:

```text
RRI_b = 1 - E[L(h_b, h_hat_b)] / E[L(h_b, mean_b)]
```

The control philosophy is strict. A result is not scientifically meaningful
unless real activations beat shuffled, zero, mean, random, blank/generic, or
no-injection controls as appropriate to the stage.

## 2. Target Model Facts And Initial Assumptions

The pilot treats loaded model configuration and module inspection as
authoritative. The working Nano facts are:

| Quantity | Value |
|---|---|
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Hidden size | `2688` |
| Blocks | `52` `NemotronHBlock` layers |
| Pattern | `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME` |
| Block types | `M` = Mamba-2, `E` = MoE, `*` = GQA/attention |
| HF wrapper | `.backbone` |
| HF layers | `.backbone.layers` |
| HF final norm | `.backbone.norm_f` |
| HF embeddings | `.backbone.embeddings` |
| Target boundaries | `R_34` and `R_27` first |

The initial active risk was not "NLA cannot work on Nano". The main risk was
that a wrapper mismatch, final-norm mismatch, cache mismatch, chat-template
change, teacher leakage, or adapter/export mismatch could produce plausible
text while invalidating the reconstruction experiment.

## 3. Injection Tracks Considered

The project began with three possible AV injection tracks.

| Track | Definition | Role | Decision |
|---|---|---|---|
| A | Replace one input embedding row with scaled `h_b` | Paper-faithful baseline | Keep as primary baseline |
| C | Patch residual boundary `R_b[p] <- T(h_b)` during a split forward | Oracle/debugging intervention | Use only as diagnostic |
| B | Learn an input embedding adapter `S_phi(h_b)` | Conditional fallback | Defer unless Track A is weak and Track C is strong |

Track A is mathematically odd for Nano because a layer-`b` residual vector is
being inserted at the embedding layer and passed through lower Mamba/MoE/GQA
blocks. Still, it is the closest path to the released NLA implementation and is
therefore the right baseline.

Track C is cleaner as activation patching but has hard cache semantics for
Nano's hybrid attention plus Mamba state. It is useful as an oracle, not as the
first production or serving path.

## 4. Chronological Experiment Ledger

### 4.1 Environment, Introspection, And Adapter Setup

Purpose:

Verify that Nano can be loaded, tokenized, inspected, and addressed at the
right module paths before any training or scientific claims.

What was done:

- Created an introspection harness for tokenizer, chat-template hash, model
  config, module paths, block pattern, and cache metadata.
- Confirmed the expected Nano module assumptions in the cluster environment:
  `.backbone`, `.backbone.layers`, `.backbone.norm_f`, `.backbone.embeddings`.
- Centralized cluster setup in `scripts/cluster_nano_env.sh`.
- Recorded that local macOS full-weight identity probes failed because
  `mamba_ssm` was unavailable locally; this was an environment issue, not a
  scientific failure.

Results:

- Config-mode introspection on local artifacts recorded tokenizer metadata:
  chat template present, `enable_thinking_default=false`, `bos_token_id=1`,
  `eos_token_id=11`.
- Cluster environment became the reliable place for full model diagnostics.

Reason for next step:

After wrapper and environment assumptions were pinned, the project moved to
residual-boundary identity, source replay, and small AR/AV diagnostics.

### 4.2 Residual Boundary And Extraction Identity

Purpose:

Establish that stored Nano activations really correspond to the intended raw
residual boundary `R_b`, not a post-norm or off-by-one tensor.

Boundaries:

- `R_34`: post-GQA block 33, close to prior NLA convention.
- `R_27`: post-GQA block 26, earlier and leaves more AV suffix capacity.

What was done:

- Added extraction and serialization probes.
- Added real-data Stage 0 extraction for FineWeb rows.
- Added exact token provenance fields for regenerated data:
  `token_position`, `token_id`, `token_text`, and `token_ids_prefix`.
- Added source replay diagnostics.

Key result:

Nano `R_34` source replay passed strongly:

| Metric | Result |
|---|---:|
| Exact token-count rows | `256/256` |
| Correct cosine | `0.99909` |
| Normalized MSE | `0.00182` |

Interpretation:

The target residual and source-token provenance were not the immediate blocker.
If source replay can reconstruct the stored target almost perfectly, then later
AR failures are more likely in the explanation-to-residual channel, the critic
template, the boundary choice for explanation text, or data/objective design.

Reason for next step:

Since extraction/provenance looked sound, the project tested whether teacher
explanations could drive AR reconstruction.

### 4.3 Early Synthetic And Real-Data AR Probes

Purpose:

Determine whether `z -> h_hat_b` can beat controls on heldout rows before
scaling AR SFT.

What was done:

- Built a frozen-prefix AR value-head baseline.
- Built real-data AR parquets from FineWeb-derived activations.
- Ran capacity probes at `R_34` with trainable head and small trainable Nano
  tails.
- Evaluated against shuffled, random, mean, blank/generic, and source-related
  controls.

Important result: tail-1 capacity probe at `R_34`

| Quantity | Result |
|---|---:|
| Records | `256` total, `192` train, `64` heldout |
| Trainable Nano layer | `.backbone.layers.33` |
| Trainable tail params | `23,399,040` |
| Trainable value-head params | `7,225,344` |
| Steps | `80` |
| Train correct NMSE | `1.3703 -> 0.7891` |
| Train cosine | `0.3149 -> 0.6055` |
| Heldout correct NMSE | `0.9674` |
| Heldout shuffled NMSE | `0.9747` |
| Heldout train-mean NMSE | `0.8919` |
| Scientific pass | `false` |

Capacity ablation bundle:

| Run | Split | Tail blocks | Steps | Train NMSE | Heldout correct NMSE | Heldout shuffled NMSE | Heldout mean NMSE | Scientific pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `r34-head0-docrandom-80` | doc | 0 | 80 | `0.7992` | `0.9708` | `0.9779` | `0.8919` | false |
| `r34-tail1-docrandom-400` | doc | 1 | 400 | `0.2103` | `1.1277` | `1.1653` | `0.8919` | false |
| `r34-tail1-rowrandom-80` | row | 1 | 80 | `0.7859` | `0.9484` | `0.9818` | `0.8904` | false |
| `r34-tail2-docrandom-80` | doc | 2 | 80 | `0.7621` | `0.9603` | `0.9697` | `0.8919` | false |

Interpretation:

- The AR machinery can fit training rows.
- More tail depth and more optimization did not create heldout signal.
- The 400-step run overfit: train NMSE improved to `0.2103`, but heldout
  worsened to `1.1277`.
- Head-only performance was already close to tail-1 performance, suggesting
  capacity was not the first-order blocker.

Reason for change:

AR scaling was stopped. The evidence said "do not run 10% -> 100% AR SFT yet";
the failure was not an obvious lack of AR capacity.

### 4.4 Prompt Signal Gates: Haiku And Kimi

Purpose:

Test whether teacher explanations themselves carry heldout reconstruction
signal under the current Nano AR critic channel.

What was done:

- Ran old Haiku prompt signal gate on earlier parquet without exact
  `token_ids_prefix`.
- Regenerated exact-provenance Haiku data.
- Fixed NVIDIA/Kimi provider parsing so `message.reasoning_content` is used
  when `message.content` is null, and Stage 2 extracts the last complete
  `<analysis>...</analysis>` block.
- Ran exact-provenance Kimi reasoning gate on a smaller slice.

Old Haiku gate:

| Metric | Result |
|---|---:|
| Source raw feature NMSE | `0.002116` |
| Source raw cosine | `0.99894` |
| Teacher NMSE | `1.1437` |
| Train-mean NMSE | `0.8919` |
| Blank NMSE | `0.8916` |
| Generic NMSE | `0.8913` |
| Scientific pass | `false` |

Exact Haiku gate:

| Metric | Result |
|---|---:|
| Stage0 rows with `token_ids_prefix` | `256/256` |
| Stage2/3 kept rows | `254/256` |
| Exact token prefix fraction | `1.0` |
| Teacher heldout NMSE | `1.1089` |
| Teacher heldout cosine | `0.4456` |
| Teacher shuffled NMSE | `1.2060` |
| Blank NMSE | `0.8599` |
| Generic NMSE | `0.8597` |
| Source context NMSE | `1.0743` |
| Train-mean NMSE | `0.8595` |
| Source raw NMSE | `0.003019` |
| Source raw cosine | `0.99849` |
| Scientific pass | `false` |

Exact Kimi reasoning gate:

| Metric | Result |
|---|---:|
| Source rows | first `32` exact Haiku rows |
| Stage2/3 kept rows | `31/32` |
| Exact token prefix fraction | `1.0` |
| Teacher heldout NMSE | `1.0729` |
| Teacher heldout cosine | `0.4636` |
| Teacher shuffled NMSE | `1.1460` |
| Blank NMSE | `0.9578` |
| Generic NMSE | `0.9570` |
| Source context NMSE | `1.0701` |
| Train-mean NMSE | `0.9595` |
| Source raw NMSE | `0.001743` |
| Source raw cosine | `0.99913` |
| Scientific pass | `false` |

Interpretation:

- Exact provenance did not fix the AR teacher signal failure.
- Kimi's reasoning path parsed and ran, but a stronger teacher did not beat
  blank/generic/mean controls.
- Teacher text beat shuffled teacher text, so there may be some row-specific
  information, but it did not map to target residual geometry better than simple
  baselines.
- Source raw remained a near-perfect oracle, so the stored target and prefix
  token data were not the culprit.

Reason for change:

The project pivoted away from AR-only teacher prompt scaling. Teacher text is
not evidence of NLA success; it is only a warm-start label. The missing
paper-faithful link became AV: `h -> AV -> z`.

### 4.5 Qwen Released NLA Positive-Control QC

Purpose:

Check whether the reference NLA sidecar/injection/critic scoring contract works
in this environment. If Qwen failed, Nano work should pause. If Qwen passed,
Nano failures were more likely Nano-specific.

Scope:

- AV checkpoint: `kitft/nla-qwen2.5-7b-L20-av`
- AR checkpoint: `kitft/nla-qwen2.5-7b-L20-ar`
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Extraction layer: `20`
- One A100 80GB GPU
- Inference-only: no training, PEFT, LoRA, RL, Nano serving, or large datagen

Results:

| Metric | Result |
|---|---:|
| QC passed | `true` |
| Activation rows | `56` |
| AV parse success | `56/56` |
| AR score rows | `224` |
| Correct mean MSE | `0.1628074989` |
| Shuffled mean MSE | `0.9283932048` |
| Random mean MSE | `1.0225361649` |
| Mean-target mean MSE | `0.6622565016` |
| Correct median cosine | `0.9246753156` |
| Fraction correct beats mean | `1.0` |
| Fraction correct beats random | `1.0` |
| Fraction correct beats shuffled | `1.0` |

Comparison against the Nano `R_34` weak point:

| Metric | Qwen QC | Nano R34 failure point |
|---|---:|---:|
| Correct mean MSE | `0.1628` | `0.9222` heldout NMSE |
| Correct median cosine | `0.9247` | `0.5389` heldout cosine |
| Mean/mean-target MSE | `0.6623` | `0.8829` train-mean |
| Shuffled MSE | `0.9284` | `0.9648` shuffled |
| Scientific pass | true | false |

Interpretation:

The released Qwen pair cleanly separated correct explanations from controls.
This reduced the likelihood that the project misunderstood the reference NLA
sidecar, injection marker, or AR scoring contract. Nano failures were therefore
treated as Nano-specific boundary, objective, data, split, template, or target
geometry issues.

Reason for change:

After Qwen passed and Nano source replay passed, the team stopped treating
"reference implementation misunderstanding" as the main hypothesis and focused
on Nano's AV/AR path.

### 4.6 First Qwen-Faithful Nano AV Warm-Start Smokes

Purpose:

Test the paper-faithful AV direction directly:

```text
h_b -> AV -> z
```

Teacher explanations are used only as supervised warm-start labels, not as
proof of NLA faithfulness.

Implementation:

- Script: `scripts/nano_av_warmstart_smoke.py`
- Input: existing explained Nano activation parquet
- Target format: `<explanation>{z}</explanation>`
- Marker: `々`
- Marker token id: `42019`
- Initial injection scale: `150`
- Model path: HF-native no-Miles loop for fast single-GPU science
- Controls: real, shuffled, zero, mean, no injection

Base likelihood run:

| Split | Real h | Shuffled h | Zero h | Mean h | No injection |
|---|---:|---:|---:|---:|---:|
| Train | `2.5555` | `2.6335` | `2.6288` | `2.6229` | `2.7105` |
| Heldout | `2.8191` | `2.9051` | `2.8787` | `2.8966` | `2.9548` |

Interpretation:

Even without training, real `h` had a weak but consistent teacher-forced NLL
advantage over controls.

One-step training smoke:

| Quantity | Result |
|---|---:|
| Rows | `8` selected |
| Split | `6` train / `2` heldout |
| Trainable subset | `lm_head` |
| Trainable params | `352,321,536 / 31,577,937,344` |
| Heldout real NLL | `3.1278` |
| Heldout zero NLL | `3.2872` |
| Heldout mean NLL | `3.3177` |
| Heldout shuffled NLL | `3.3536` |
| Heldout no-injection NLL | `3.5550` |

Fifty-step `lm_head` overfit smoke:

| Quantity | Result |
|---|---:|
| Rows | `209` input, `209` usable, `8` selected |
| Split | `7` train / `1` heldout |
| Train steps | `50` |
| Batch size | `1` |
| LR | `1e-4` |
| Trainable subset | `lm_head` |
| Heldout real NLL | `2.6669` |
| Heldout zero NLL | `2.8673` |
| Heldout shuffled NLL | `2.9215` |
| Heldout mean NLL | `2.9260` |
| Heldout no-injection NLL | `3.1209` |

Heldout gaps:

| Gap | Value |
|---|---:|
| Real vs zero | `0.2003` |
| Real vs shuffled | `0.2546` |
| Real vs mean | `0.2591` |
| Real vs no-injection | `0.4540` |

Generation failure:

The fifty-step run's sampled generation collapsed to repeated `<` tokens across
all controls, with content F1 `0.0`.

Interpretation:

- AV warm-start is mechanically viable.
- Teacher-forced NLL showed the first positive AV row-specific signal.
- Free generation remained degenerate.
- The next problem became actor decoding/capacity, not extraction,
  source provenance, or Qwen contract understanding.

Reason for change:

The project moved from AR-only diagnostics to larger AV actor training, while
keeping controls as the promotion gate.

### 4.7 Recorded 30K AV Baseline

Purpose:

Use a larger Super-thinking teacher set to test whether AV signal scales beyond
tiny smokes.

The older scale-up note records a 29,913-row AV baseline:

| Quantity | Result |
|---|---:|
| Usable rows | `29,913` |
| Split | `26,920` train / `2,993` heldout |
| Trainable subset | `lm_head` only |
| Steps | `10,000` |
| Best config | scale `75`, LR `1e-4`, max target tokens `192` |
| Heldout real NLL | `1.6051` |
| Heldout shuffled NLL | `1.7151` |
| Heldout zero NLL | `1.6812` |
| Heldout mean NLL | `1.7029` |
| Heldout no-injection NLL | `1.8309` |

Interpretation:

The larger AV result preserved a real-activation advantage over every control.
However, it still did not prove the full NLA loop because `z -> h_hat` was not
shown, and decoded content quality remained weak.

Reason for change:

This result justified a 100k AV scale-up, but not RL or final NLA claims.

### 4.8 100k RunAI rsLoRA AV Artifact

Purpose:

Scale AV warm-start to the full 99,570-row R27 Super-thinking dataset, test
whether a larger rsLoRA actor can maintain row-specific heldout likelihood, and
check whether free generation improves beyond repeated-token collapse.

Run:

```text
av-r27-99570-rslora-r192-broad-scale75-lr1e5-s800-save-gen8-2gpu-offline-20260526T1645Z
```

Configuration:

| Quantity | Value |
|---|---|
| Rows | `99,570` |
| Split | doc-level `89,604` train / `9,966` heldout |
| Doc overlap | `0` |
| PEFT | rsLoRA |
| Rank / alpha | `192 / 384` |
| Target modules | `q/k/v/o/in/out/up/down` |
| Injection scale | `75` |
| LR | `1e-5` |
| Steps | `800` |
| Train batch size | `1` |
| Trainable params | `5,303,242,752` |
| Trainable fraction | `14.38%` |
| Trainable state | about `20G` |
| W&B | `ggrd168y` |

Heldout teacher-forced NLL:

| Control | NLL |
|---|---:|
| Real | `1.1707068910` |
| Zero | `1.4171737903` |
| Mean | `1.4671215108` |
| No injection | `1.4797221795` |
| Shuffled | `1.5453720745` |

Heldout gaps:

| Gap | Value |
|---|---:|
| Real vs zero | `0.2464668993` |
| Real vs mean | `0.2964146198` |
| Real vs no-injection | `0.3090152885` |
| Real vs shuffled | `0.3746651835` |

Train curve summary:

- Step 1 loss was about `2.3964`.
- Minimum observed loss was around `0.8475` near step 375.
- Final logged loss at step 800 was `1.2656`.
- The heldout gap did not look like simple train-document memorization because
  it persisted on doc-heldout rows.

Generation summary:

| Metric | Result |
|---|---:|
| Examples | `8` |
| Real closing-tag count | `7/8` |
| Mean content F1, real | `0.4466` |
| Mean content F1, shuffled | `0.3288` |
| Mean content F1, zero | `0.2842` |
| Mean content F1, no injection | `0.2770` |
| No-injection closing-tag count | `0/8` |

Interpretation:

This is the strongest AV evidence so far. It shows robust row-specific
teacher-forced likelihood on 100k-scale data and materially better generation
than the earlier repeated-`<` collapse. Still, it does not prove the full NLA
claim because AR reconstruction and AV-generated round trip are not complete.

Reason for change:

The AV actor became scientifically promising enough to invest in reproducible
launchers, W&B/offline artifact capture, and Miles/FSDP2 scale work. The
remaining scientific blocker shifted from "can AV use `h` at all?" to "can we
train/evaluate the full actor path reproducibly and eventually close `z -> h`?"

### 4.9 HPO And Launcher Smoke Caution

An initial `lr=2e-5` HPO launch accidentally inherited `--row-limit 32` from
the AV smoke defaults. It is useful only as a launcher smoke and must not be
compared as a 100k run.

Reason for change:

The repeatable launcher now pins `ROW_LIMIT=99570` for complete-performance
runs. This is an example of why experiment-class validation was added: a
mis-sized run can otherwise look like a valid HPO point.

### 4.10 Miles/FSDP2 Import And Dataset Contract Gates

Purpose:

Move from the legacy HF-native/sequential smoke path to a real distributed
Miles/FSDP2 NLA actor path with batched activation transport, real checkpointing,
and better scaling characteristics.

Why this was needed:

The legacy sequential smoke harness established signal but relied on single-row
gradient accumulation. A full batch8 epoch projected to multi-day runtime and
was manually stopped after about four hours. It was not the right path for full
100k training.

Import gate results:

| Check | Result |
|---|---|
| `import miles` | ok |
| `nla.train_actor.NLAFSDPActor` | ok |
| `nla.rollout.sft_actor.generate_rollout` | ok |
| `nla.injection.inject_at_marked_positions` | ok |

Dataset contract gate on full AV-SFT parquet:

| Metric | Result |
|---|---:|
| Rows | `99,570` |
| `d_model` | `2688` |
| Nonfinite activations | `0` |
| Malformed responses | `0` |
| Prompt marker failures | `0 / 99,570` |
| 80/10/10 split | `79,647 / 9,961 / 9,962`, doc overlap `0` |
| 90/5/5 split | `89,618 / 4,978 / 4,974`, doc overlap `0` |

Reason for change:

Once import and dataset gates passed, the project could run true Miles/FSDP2
small and medium gates before attempting a full hero run.

### 4.11 Small Miles/FSDP2 Training And Checkpoint Gates

Purpose:

Prove that the real Miles/FSDP2 NLA actor path trains at all, injects batched
activations correctly, writes checkpoints, and can resume.

Selected small-gate outcomes:

| Run | Result |
|---|---|
| retry8, gb8/mb1, model-only save | step 0 loss `2.574970`, grad norm `16.0`, step time `131.811s`, model-only checkpoint written |
| retry10, model-only resume | loaded model-only checkpoint, advanced to step 1, loss `2.511452`, grad norm `13.125`, step time `139.233s` |
| retry11, gb8/mb4 no-save | one true local batch per rank, train microstep `25.77s`, full step `99.972s`, loss `2.073258` |
| retry12, Adam foreach | no immediate OOM, but post-backward update did not complete within several minutes; stopped |
| retry13, first timing patch | invalid run; patch changed `_train_step` call contract |
| retry14, timing fixed | loss `2.073258`, actor train time `113.398s`, step time `114.987s` |
| retry15, skip grad norm diagnostic | loss `2.073258`, actor train time `27.942s`, step time `30.100s` |

Important timing finding:

The dominant bottleneck was FSDP full gradient norm clipping, not forward or
backward. In retry14, forward plus backward was roughly 25 seconds, while raw
`clip_grad_norm_` cost was about 68-87 seconds depending on rank. Skipping grad
norm dropped the no-save step from about 115 seconds to about 30 seconds.

Interpretation:

Skipping grad norm is a useful diagnostic, but not the selected complete-
performance path because it changes optimization semantics when norms exceed
the clipping threshold.

Reason for change:

The project added batch-scaling diagnostics and preserved faithful grad norm for
hero candidates, while allowing skip-grad-norm only for small diagnostic runs.

### 4.12 Medium-Small Miles/FSDP2 Gate

Purpose:

Run a larger but bounded 960-row training gate through the real Miles/FSDP2 path
and evaluate row-specificity after checkpoint conversion.

Training gate:

| Quantity | Result |
|---|---|
| Rows | `960` |
| Split | doc-level `80/10/10` |
| Train rows | `771` padded to `864` |
| Validation / test rows | `99 / 90` |
| Global / micro batch | `96 / 8` |
| Checkpoints | model-only `iter_0000003`, `iter_0000006`, `iter_0000009` |
| Checkpoint size | about `63.18 GB` each raw, `59 GiB` du |
| Non-checkpoint steps after warmup | `42.93s`, `45.66s`, `47.78s`, `46.24s`, `47.49s` |
| Checkpoint saves | added about `192s`, `213s`, `397s` |

Resume smoke:

- Loaded `iter_0000009/model` on both ranks.
- Skipped missing optimizer/LR scheduler state for model-only checkpoint.
- Advanced to rollout 9.
- Finished a no-save step with loss `1.486620` and step time `56.697s`.

Checkpoint eval:

The medium checkpoint had to be converted from FSDP DCP to HF format using the
origin remote-code safetensors layout. A built-in-HF-style conversion produced
unusable NLL near 13 because it randomized backbone parameters. The converter
was updated to preserve the origin `backbone.*` layout.

Validation NLL on 32 rows:

| Control | NLL | Gap vs real |
|---|---:|---:|
| Real | `1.8323` | - |
| Shuffled | `1.9581` | `0.1258` |
| Zero | `1.9196` | `0.0873` |
| Mean | `1.9478` | `0.1155` |
| No injection | `2.0705` | `0.2382` |

Test NLL on 32 rows:

| Control | NLL | Gap vs real |
|---|---:|---:|
| Real | `1.7781` | - |
| Shuffled | `1.8741` | `0.0960` |
| Zero | `1.8412` | `0.0631` |
| Mean | `1.8725` | `0.0943` |
| No injection | `1.9756` | `0.1975` |

Interpretation:

The medium-small checkpoint passed a directional row-specificity gate on
validation and test, but it was not a complete-performance result. The gaps did
not reach the hero target of at least about `0.30` against controls, and the
sample was small.

Reason for change:

This cleared the path to full-data operational gates: optimizer checkpoint,
resume, batch scaling, and hero launch.

### 4.13 Full Optimizer Checkpoint/Resume Smoke

Purpose:

Prove that a full Nano30B optimizer/scheduler checkpoint can be saved and
resumed before risking a full-data hero run.

Result:

- Full optimizer checkpoint payloads were saved and reloaded.
- Model, optimizer, LR scheduler, RNG, NLA metadata, and run metadata were
  included.
- Resume loaded model, optimizer, and LR scheduler on both ranks and advanced
  from latest iteration 1 to training step 1.
- One full optimizer checkpoint was about `177 GiB`:
  model shards about `63 GiB`, optimizer shards about `126 GiB`, plus metadata.

Interpretation:

Checkpoint/resume was possible, but storage pressure was real. Rolling latest
3 full optimizer checkpoints needs about `531 GiB` before logs, base model,
converted HF checkpoints, data, and W&B artifacts.

Reason for change:

PVC storage was expanded to about `1 TiB`, diagnostic artifacts were pruned, and
full-data hero planning required rolling checkpoint cleanup.

### 4.14 Batch Scaling And Hero Configuration Selection

Purpose:

Find the largest faithful full-data batch configuration that remains stable
after optimizer state allocation.

Observed faithful configurations:

| Global batch | Micro batch | Outcome | Notes |
|---:|---:|---|---|
| `96` | `8` | pass | about `157.2s/step`, about `0.61 examples/s` on first measured step |
| `192` | `8` | pass | first step about `197.8s`; second step about `89.4s`, about `2.15 examples/s` after warmup |
| `384` | `8` | first step pass, later OOM | attractive throughput but failed after Adam state allocation |

Detailed gb192 two-step diagnostic:

| Step | Loss | Grad norm | Step time |
|---:|---:|---:|---:|
| 0 | `2.0214923` | `14.4375` | `198.7763s` |
| 1 | `1.7771916` | `7.5625` | `89.4263s` |

Peak reserved memory was about `137.3 GiB` per H200.

Decision:

Use `gb192/mb8/save100` for the full-data hero attempt. `gb384` was rejected as
not hero-stable.

### 4.15 Hero Attempt: gb384 Second-Step OOM

Purpose:

Try the higher-throughput `gb384/mb8` full-data complete-performance run.

Result:

| Quantity | Result |
|---|---|
| Experiment class | complete-performance |
| Global / micro batch | `384 / 8` |
| Step 0 loss | `2.0007405` |
| Step 0 grad norm | `14.6875` |
| Step 0 step time | `212.1566s` |
| Failure | CUDA OOM on rollout 1 backward after Adam state existed |

Interpretation:

`gb384` was not stable after optimizer-state allocation. Step 0 alone was not
enough to certify the configuration.

Reason for change:

The hero configuration was reduced to `gb192/mb8`.

### 4.16 Hero Attempt: gb192 Save100 First Checkpoint Failure

Purpose:

Run a full-data complete-performance AV-SFT hero with faithful grad norm,
rolling checkpoints, and external checkpoint eval.

Run:

```text
nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gb192-mb8-save100-20260527T2050Z
```

Configuration:

| Quantity | Value |
|---|---|
| Dataset | R27 Super-thinking 99,570 explained rows |
| Split | doc-level `90/5/5` |
| Train / validation / test | `89,618 / 4,978 / 4,974` |
| Padded train rows | `89,664` |
| Optimizer steps | `467` |
| Global / micro / rollout batch | `192 / 8 / 192` |
| Injection scale | `75` |
| LR | `1e-5` |
| Save interval | `100` |
| W&B | offline |

Latest completed metric:

| Metric | Result |
|---|---:|
| Step | `99` |
| Loss | `1.0409398078918457` |
| Grad norm | `1.2890625` |

Failure:

```text
RuntimeError: NCCL Error 1: unhandled cuda error
NLAFSDPActor.save_model() -> miles/backends/fsdp_utils/checkpoint.py -> dcp.save(...)
torch.distributed.checkpoint scatter_object_list(...)
```

Checkpoint state:

- Target checkpoint: `iter_0000100`
- Directory existed but contained `0` files.
- Total checkpoint tree was only about `28K`.
- Process exited; GPUs idle.

Interpretation:

Training reached the first save boundary with stable loss and grad norm, but no
usable checkpoint was written. This is not a scientific hero result because no
checkpoint eval or resume can be run from an empty payload.

Reason for change:

The next gate became checkpoint-save remediation, not another full-data hero
run. Patch `0006_fsdp_checkpoint_gloo_pg.patch` was added to keep model/FSDP
training on NCCL while passing a cached Gloo process group to
`torch.distributed.checkpoint`, so DCP metadata/object collectives do not use
the default NCCL group. Subsequent focused full-optimizer save and corrected
resume smokes validated the remediation path before the Gloo relaunch.

### 4.17 Completed AV-SFT Hero: Gloo, Tokenized Source, Final Heldout Eval

Purpose:

Complete the full-data AV-SFT engineering hero after the DCP metadata
collective failure, then evaluate a final checkpoint against row-specific
activation controls. This run is strictly AV-SFT:

```text
R_27 activation h -> explanation z
```

It is not an AV+AR run. AR/critic SFT has not been launched.

Preceding source-contract failure:

A first Gloo relaunch against the raw explained source artifact failed at
startup with `KeyError: 'tokens'`. The reason was not a model failure: the raw
artifact had activation and explanation fields, but it was not the tokenized
Miles AV-SFT prompt-data contract. The successful relaunch therefore used the
materialized `av_sft.parquet` from the latest held batch8 AV-SFT run directory.

Run:

```text
nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gloo-tokenized-gb192-mb8-save100-20260528T0110Z
```

Configuration:

| Quantity | Value |
|---|---|
| Experiment class | complete-performance AV-SFT |
| Backend | Miles FSDP2 with Gloo DCP metadata process group |
| Dataset | tokenized AV-SFT parquet derived from 99,570 R27 explained rows |
| Raw source | `base_R27_super_thinking_99570_explained.parquet` |
| Split | doc-level `90/5/5` |
| Train / validation / test | `89,618 / 4,978 / 4,974` |
| Padded train rows | `89,664` |
| Optimizer steps | `467` |
| Global / micro / rollout batch | `192 / 8 / 192` |
| Injection scale | `75` |
| LR | `1e-5` |
| Save interval | `100` |
| Final exact-resume checkpoint | `checkpoints/iter_0000467`, about `177G` |
| HF eval checkpoint | `hf_iter_0000467`, about `59G` |
| HF index parameter count | `31,577,937,344` |
| HF index total size | `63,155,880,576` bytes |
| Final train metric | step `466`, loss `0.9521` |
| W&B run | `https://wandb.ai/rishabhga97/nano30b-nla-pilot/runs/dw7mp5sn` |

Heldout checkpoint eval:

The final checkpoint was evaluated on a bounded sample of `64` validation rows
and `64` test rows. The local report copy is:

```text
artifacts/runai_eval/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gloo-tokenized-gb192-mb8-save100-20260528T0110Z/eval_iter_0000467_v64_t64_report.json
```

NLL results:

| Split | Real | Shuffled | Zero | Mean | No injection |
|---|---:|---:|---:|---:|---:|
| Validation | `0.9046` | `1.3298` | `1.1743` | `1.1924` | `1.3441` |
| Test | `0.9565` | `1.3493` | `1.2160` | `1.2302` | `1.3577` |

Gaps versus real:

| Split | Shuffled | Zero | Mean | No injection |
|---|---:|---:|---:|---:|
| Validation | `+0.4251` | `+0.2697` | `+0.2878` | `+0.4395` |
| Test | `+0.3928` | `+0.2595` | `+0.2736` | `+0.4012` |

Interpretation:

This is a successful AV-SFT engineering hero and a positive bounded
row-specific heldout signal. Real activations beat shuffled, zero, mean, and
no-injection controls on both validation and test. The result is materially
stronger than train loss alone because it tests whether the trained actor uses
the row-specific injected activation rather than a generic prompt prior.

Scientific caution:

The bounded `v64/t64` sample does not justify a maximal scientific claim. The
zero and mean control gaps are positive but remain just below the aspirational
`0.30` target on this eval. The next stronger claim needs a cached-mean remote
evaluator and a larger `v256/t256` or full validation/test run, plus generation
parse/F1 evaluation from `hf_iter_0000467`.

## 5. Consolidated Results Table

| Stage | Best or critical result | Scientific status |
|---|---|---|
| Introspection | Nano paths and tokenizer/template pinned in cluster env | Infrastructure gate passed |
| Source replay R34 | cosine `0.99909`, NMSE `0.00182` | Target/provenance likely sound |
| AR capacity R34 | train fits, heldout loses to mean | AR scaling blocked |
| Exact Haiku/Kimi gates | exact provenance `1.0`, source_raw near-perfect, teacher loses to blank/generic/mean | Teacher prompt channel blocked for AR |
| Qwen released QC | correct MSE `0.1628`, shuffled `0.9284`, correct beats controls on all rows | Reference NLA flow validated |
| Tiny AV smokes | real h beats controls in teacher-forced NLL | AV path viable but generation degenerate |
| 30K AV baseline | real NLL `1.6051`, beats all controls | AV signal scales, but no AR proof |
| 100K rsLoRA AV | real NLL `1.1707`, shuffled `1.5454`, F1 real `0.4466` | Strongest AV evidence |
| Miles small gate | real FSDP2 path trains and checkpoints | Distributed path viable |
| Medium-small FSDP2 | validation/test real beats controls | Directional checkpoint eval passed |
| Full optimizer checkpoint smoke | save/resume works, checkpoint about `177 GiB` | Hero prerequisite passed |
| gb384 hero | OOM on second step | Rejected |
| gb192 hero | step 99 loss `1.0409`, checkpoint save NCCL failure | Historical failure; motivated Gloo DCP metadata remediation |
| Gloo raw-source hero relaunch | startup `KeyError: 'tokens'` | Rejected raw source; tokenized AV-SFT parquet required |
| Gloo tokenized AV-SFT hero | final checkpoint `iter_0000467`, final train loss `0.9521` | AV-SFT engineering hero completed |
| `hf_iter_0000467` heldout eval | validation real NLL `0.9046`, test real NLL `0.9565`, all controls worse | Positive bounded row-specific heldout signal; not AV+AR |

## 6. Why The Research Direction Changed

### 6.1 From Direct Port To Nano-Specific Pilot

Initial reasoning showed Nano's hybrid Mamba/MoE/GQA architecture and module
layout made a direct Qwen/Gemma port risky. The project therefore became a
staged pilot with explicit boundary, cache, template, injection, and checkpoint
gates.

### 6.2 From AR Scaling To Diagnostics

AR capacity probes fit training rows but failed heldout controls, and stronger
teacher prompts did not beat blank/generic/mean baselines. This ruled out
"just scale AR SFT" as the next move.

### 6.3 From Teacher Explanations As Signal To Teacher Explanations As Labels

The Haiku/Kimi prompt gates showed that teacher text alone was not evidence of
reconstructive information. The project reclassified teacher explanations as
supervised warm-start labels for AV, not as proof of NLA faithfulness.

### 6.4 From AR-Only Proxy To Paper-Faithful AV

The original NLA loop requires `h -> AV -> z -> AR -> h_hat`. Since AR-only
teacher diagnostics were weak, the missing link became AV. The AV smokes and
100k rsLoRA run showed that Nano activations can condition text generation
row-specifically.

### 6.5 From Legacy HF Smoke To Miles/FSDP2

The legacy smoke harness established AV signal but did not scale operationally.
Miles/FSDP2 became necessary for real batch training, checkpointing, resume, and
eventual full-data runs.

### 6.6 From gb384 To gb192

`gb384` looked attractive on first-step throughput but failed on the second
optimizer step after Adam state allocation. The selected faithful batch became
`gb192/mb8`.

### 6.7 From Checkpoint Remediation To Completed AV-SFT Hero

The first `gb192` hero reached step 99 with stable metrics but failed during the
first distributed checkpoint save. That shifted the immediate direction to DCP
metadata remediation. The Gloo process-group patch then allowed the
full-optimizer checkpoint path to pass focused save/resume smokes.

The first Gloo relaunch exposed a separate data-contract issue: the raw R27
source artifact was not tokenized for Miles AV-SFT and failed on missing
`tokens`. The completed hero therefore used the materialized tokenized
`av_sft.parquet`.

The final direction changed from "make the hero run at all" to "measure how
strong the completed AV-SFT hero is." The `hf_iter_0000467` bounded heldout eval
showed real activations beating shuffled, zero, mean, and no-injection controls
on validation and test.

## 7. Current Scientific Interpretation

What is supported:

1. Nano residual extraction/provenance is likely correct for the tested
   `R_34` path, because source replay is nearly perfect.
2. The released Qwen NLA inference contract works in the same broad environment.
3. Nano AV can use injected `h` row-specifically under teacher-forced NLL.
4. The 100k rsLoRA AV run is a strong positive AV result: real activations beat
   shuffled, zero, mean, and no-injection controls by meaningful heldout gaps.
5. Generation quality improved from repeated-`<` collapse to mostly closed
   `<explanation>` generations with better real-control content F1.
6. Miles/FSDP2 can complete a full-data Nano AV-SFT hero and produce a final
   exact-resume checkpoint.
7. The `hf_iter_0000467` bounded heldout eval shows row-specific signal:
   validation/test real NLLs `0.9046/0.9565` beat shuffled, zero, mean, and
   no-injection controls.

What is not yet proven:

1. Full NLA success: `h -> AV -> z -> AR -> h_hat` has not passed heldout
   round-trip controls.
2. AR reconstruction from teacher or AV-generated text remains unresolved.
3. There is no AV+AR SFT hero. AR/critic SFT has not been launched.
4. The bounded `v64/t64` AV-SFT eval is positive but not a maximal scientific
   claim; zero and mean gaps are below the aspirational `0.30` target.
5. Generation parse/closure and content F1 from `hf_iter_0000467` are still
   needed to understand the actor's free-generation behavior.
6. The training W&B run still primarily records train metrics; external eval
   metrics need to be wired into W&B for future comparisons.

## 8. Open Gates And Next Required Experiments

### 8.1 Cached-Mean Remote Eval And Larger Heldout Gate

Trigger:

The `hf_iter_0000467` bounded eval is positive, but the sample is only
`64/64`, and the zero/mean controls are expensive if the remote evaluator
recomputes the mean too often.

Required next step:

Deploy the cached-mean eval script to RunAI, then run at least `v256/t256` and
preferably the full validation/test split.

Metrics:

- Validation and test NLL for real, shuffled, zero, mean, no-injection.
- Real-vs-control gaps.
- Examples/sec and control-eval runtime.
- Explicit reporting of sample counts and split provenance.

Pass condition:

- Real NLL beats all controls on validation and test.
- Zero and mean gaps reach or exceed the aspirational `0.30` target if making
  a stronger scientific claim.
- If gaps remain positive but below target, label the result as bounded positive
  signal rather than maximal success.

### 8.2 Generation Parse/F1 Eval Gate

Trigger:

Teacher-forced NLL proves the actor uses real activations under the supervised
target, but it does not fully characterize free generation.

Required metrics from `hf_iter_0000467`:

- `<explanation>` parse and closure rate.
- Content F1 or a comparable lexical/semantic overlap score.
- Real versus shuffled, zero, mean, and no-injection generation controls.
- Failure taxonomy for malformed or generic outputs.

Pass condition:

Real-activation generation should parse more reliably and carry more
row-specific content than controls. NLL gains without generation gains should be
reported as teacher-forced AV progress, not actor-quality closure.

### 8.3 W&B Eval Logging Gate

Trigger:

The W&B run is useful for train curves, but the strongest scientific evidence
now lives in external checkpoint eval JSONs.

Required next step:

Wire checkpoint eval metrics into W&B or an equivalent run-indexed table:
split, sample count, control type, NLL, gap vs real, checkpoint id, evaluator
version, and source split path.

Pass condition:

Future readers can recover the AV-SFT scientific evidence from the run record
without relying on local filesystem notes alone.

### 8.4 AR And Round-Trip Return

AR/critic SFT should be launched as a separate clearly named experiment, not
retroactively described as part of the completed AV-SFT hero. The NLA claim
still needs:

```text
teacher z -> AR -> h_hat_b
AV(real h) -> z_generated -> AR -> h_hat_b
AV(control h) -> z_control -> AR -> h_hat_b
```

Expected ordering:

```text
source oracle best
teacher z next
AV(real h) next
AV(shuffled/zero/no-injection) worse
blank/generic/mean worst
```

If teacher `z` works but AV-generated `z` fails, the actor generation channel is
the blocker. If teacher `z` fails, explanation content or AR geometry remains
the blocker. If source oracle fails, stop and debug extraction/boundary.

### 8.5 Immediate AR-SFT Objective, No RL Yet

Objective:

Build and run a Nano30B AR-SFT path analogous to the Qwen NLA AR path. The AR
checkpoint should be separate from the AV checkpoint. If supported cleanly, use
a truncated Nano critic/model up to the `R_27` reconstruction boundary rather
than a full parent-size actor. The intended supervised channel is:

```text
explanation z -> reconstructed activation h_hat
```

Training contract:

- Use the same 99,570-row source and doc-level split discipline as AV.
- Require explanation text, `activation_vector` dimension checks, sidecar
  fields, and split no-overlap checks before training.
- Use Qwen's custom critic loss:
  `--loss-type custom_loss` and
  `--custom-loss-function-path nla.loss.nla_critic_loss`.
- Run W&B offline for every launched run.
- Keep checkpoint retention storage-conscious: minimum exact-resume checkpoint
  plus lightweight eval reports.

Required gates:

1. Confirm RunAI auth, dataset paths, free disk, and no active Nano training
   process.
2. Add or verify AR dataset contract checks: explanation text,
   activation-vector dimension, sidecar fields, and split no-overlap.
3. Run tiny/small AR-SFT one-step smoke with checkpoint save/resume.
4. Run medium-small AR-SFT with heldout reconstruction metrics.
5. Compare teacher/explanation, shuffled explanation or shuffled `h`,
   blank/generic text, source-context/source-raw if available, and mean `h`
   controls.
6. Do not claim success from train loss alone. Require heldout MSE or FVE-style
   metrics against controls.
7. Only after medium-small passes, plan a complete-performance AR run.

This AR-SFT path is a prerequisite for the NLA round-trip claim. RL should wait
until the supervised AR reconstruction channel has passed heldout controls.

## 9. 2026-06-01 Nano AR Fullscan And HPO Milestone

Nano30B now has a real, separate AR-SFT critic milestone. This is not Qwen-level
yet, but it is a major step past the earlier weak Nano AR/R34 behavior: the
R27 fullscan checkpoint passed heldout reconstruction controls, and the first
Qwen-style continuation probe improved heldout AR quality substantially.

Fullscan baseline checkpoint:

- run: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-fullscan/nano-ar-miles-fsdp2-r27-fullscan-275k-gb192-mb8-lr1e5-20260530T172928Z`
- checkpoint: `checkpoints/iter_0001291`
- data: `275,396` Nano R27 AR rows, doc-level `90/5/5`
- train/validation/test rows: `247870/13761/13765`
- method: full Nano AR critic fine-tuning, not PEFT/LoRA
- loss: `custom_loss`, `nla.loss.nla_critic_loss`
- checkpoint size: about `123G` exact-resume final checkpoint

Fullscan heldout confirmation, `2048/2048`:

| Split | Teacher NMSE | Teacher cosine | FVE vs mean | Teacher beats mean rowwise |
|---|---:|---:|---:|---:|
| validation | `0.503023` | `0.748488` | `0.416190` | `98.05%` |
| test | `0.514389` | `0.742806` | `0.401452` | `97.46%` |

First HPO continuation probe:

- run: `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-hpo/nano-ar-r27-fullscan-continue-qwen-lr2e5-cosine-256steps-20260601T0019Z`
- checkpoint: `checkpoints/iter_0001547`
- schedule: `lr=2e-5`, `min_lr=2e-6`, cosine decay, `50` warmup iters
- resume mode: Miles `--finetune`
- training shape: global batch `192`, micro batch `8`, rollout batch `192`
- save policy: model/HF-only, no optimizer shards; run directory about `62G`
- caveat: Miles `--finetune` restarted the rollout counter, so this behaved like
  a longer continuation to step `1547`, not only `256` counted from `1291`

Quick heldout eval for the HPO checkpoint, `512/512`:

| Split | Teacher NMSE | Teacher cosine | FVE vs mean | Teacher beats mean rowwise |
|---|---:|---:|---:|---:|
| validation | `0.444337` | `0.777832` | `0.498884` | `98.63%` |
| test | `0.442044` | `0.778978` | `0.489140` | `98.63%` |

Relative to the pre-HPO `512/512` baseline (`0.511466` validation and
`0.505286` test teacher NMSE), the continuation improved heldout NMSE by about
`13%` on validation and `12.5%` on test. This is a real tuning signal because it
improved validation/test controls, not only train loss.

Comparison to the released Qwen AR QC reference:

| Metric | Qwen reference | Best Nano AR quick eval |
|---|---:|---:|
| Correct/teacher MSE or NMSE | `0.162807` | `0.444337` validation / `0.442044` test |
| Cosine | `0.924675` median | `0.777832` validation / `0.778978` test mean |
| Beats mean rowwise | `100%` | `98.63%` validation / `98.63%` test |

Nano's current teacher reconstruction error is still about `2.7x` the Qwen QC
error, so this should not be called Qwen-level. The fair current claim is:
Nano30B R27 AR-SFT has strong, row-specific heldout reconstruction signal and
responds positively to Qwen-style LR tuning, but further tuning or data/model
changes are still required to match released Qwen AR quality.

Larger heldout confirmation for `iter_0001547`, `2048/2048`:

| Split | Teacher NMSE | Teacher cosine | FVE vs mean | Teacher beats mean rowwise |
|---|---:|---:|---:|---:|
| validation | `0.436878` | `0.781561` | `0.492958` | `98.54%` |
| test | `0.450516` | `0.774742` | `0.475775` | `97.71%` |

The current AR milestone target is teacher normalized MSE `0.25-0.30` on both
validation and test without major new data or new training algorithms. The next
phase should continue from `iter_0001547` with bounded storage-conscious
probes, then rerun the winning recipe with exact-resume optimizer state before
calling it the final AR checkpoint.

Execution plan:

- Fix Miles finetune continuation semantics so `resume_steps: 256` renders as
  `--num-rollout 256`, not `latest_checkpoint + 256`.
- Stage current-best bounded probes:
  `configs/nano_ar/hpo/r27_best1547_continue_lr1e5_cosine_256steps.yaml` and
  `configs/nano_ar/hpo/r27_best1547_continue_lr5e6_cosine_256steps.yaml`.
- Run one bounded probe at a time with W&B offline and model/HF-only saves.
- Use `512/512` heldout eval as the fast gate, then `2048/2048` for finalists.
- Treat `<=0.30` teacher NMSE as green, `<=0.35` as usable if repeated probes
  plateau, and keep RL out of scope until the AR milestone is selected.

## 10. AV Offline HPO Seed Logs

The offline HPO study now also records AV Phase 1 and AV hero logs under
`artifacts/nano_av_hpo_study/`, using heldout real-h NLL as the minimized
objective and control gaps as supporting evidence.

Seeded AV trials:

- Phase 1 lm_head-only baseline from `experiment_0523.md`: 29,913 rows,
  scale 75, lr `1e-4`, 10,000 steps, heldout real NLL `1.6051`; real h beats
  shuffled `1.7151`, zero `1.6812`, mean `1.7029`, and no-injection `1.8309`.
- Full Nano30B AV hero:
  `nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gloo-tokenized-gb192-mb8-save100-20260528T0110Z`.
  v64/t64 objective NLL is `0.930576216429472`; validation real NLL is
  `0.9046209901571274`, test real NLL is `0.9565314427018166`, and real h
  beats mean, shuffled, zero, and no-injection controls on both splits.

The current AV suggestions are intentionally conservative: scale 75 / lr
`1e-5` follow-ups around the successful hero configuration, with warmup and
cosine-min-LR variants surfaced for offline Optuna-style comparison. This is a
sanity test for the study format on a completed Phase 1 training path, not a
request to relaunch AV before the AR milestone is selected.

## 11. Practical Lessons For Future Runs

- Never claim scientific success from train loss alone.
- Always preserve doc-level splits and report doc overlap.
- Always include real, shuffled, zero, mean, and no-injection controls for AV.
- For AR, include shuffled, blank, generic, source-context, source-raw, random,
  and train-mean controls as appropriate.
- Treat teacher text as warm-start supervision, not ground truth.
- Keep `enable_thinking=false` as the default Nano chat-template setting unless
  explicitly running a reasoning ablation.
- Do not use common English tokens as injection markers.
- Avoid comparing runs that silently inherited smoke row limits.
- Convert FSDP DCP checkpoints using the origin Nano remote-code HF layout; a
  generic HF layout can silently randomize backbone parameters.
- Full optimizer checkpoints are large enough to shape experiment design.
- A one-step throughput diagnostic is not enough; optimizer state allocation can
  change memory behavior on step two.
- For Miles AV-SFT, use the tokenized `av_sft.parquet`, not the raw explained
  source parquet.
- Do not describe a completed AV-SFT hero as AV+AR; AR/critic SFT must be a
  separate experiment.
- Nano AV is full parent-size; do not infer the AR checkpoint shape from the AV
  checkpoint. Qwen AR is truncated relative to its parent, so Nano AR should be
  designed as a separate critic-style checkpoint if the architecture supports
  that safely.
- Cache expensive controls such as the train-split mean before scaling heldout
  evals.
- For HPO continuation probes, avoid rematerializing full splits when the
  vetted train split already exists; point directly at the existing
  `train_padded.parquet` to reduce storage churn and split-race risk.
- Miles `--finetune` is useful for fresh LR schedules, but check rollout-counter
  semantics carefully because it can restart counters instead of continuing from
  the loaded checkpoint tracker.

## 12. Bottom Line

The project has progressed from architecture risk and AR failures to a completed
AV-SFT engineering hero and a meaningful Nano AR-SFT milestone. The best current
scientific statement is:

```text
Nano30B R27 activations carry row-specific information that a Qwen-faithful
AV-SFT actor can use to predict teacher explanation text at full-data
Miles/FSDP2 scale. At hf_iter_0000467, real activations beat shuffled, zero,
mean, and no-injection controls on bounded validation and test checkpoint evals.

Separately, a Nano30B R27 AR critic trained on the 275,396-row fullscan dataset
reconstructs activation vectors from teacher explanations with strong heldout
signal against mean, shuffled, blank, generic, and source-context controls. A
Qwen-style LR continuation improved the quick heldout teacher NMSE to about
0.44 with FVE around 0.49-0.50.
```

The statement that is not yet justified is:

```text
Nano30B has a Qwen-level Natural Language Autoencoder, or a completed AV+AR+RL
round-trip system.
```

That stronger claim requires the larger `iter_0001547` eval to confirm the quick
HPO result, selection of a final AR checkpoint recipe, and AV-generated
round-trip controls. RL should wait until the AR checkpoint is selected and
documented. In parallel, the AV side still needs cached-mean larger heldout
eval, generation parse/F1 from `hf_iter_0000467`, and direct W&B eval logging.

## 13. Nano AR Wide Probe Queue

On `2026-06-02T19:06Z`, a serial AR HPO queue watcher was launched in the RunAI
`train` workspace to run short bounded probes without manual train/eval
handoffs.

Queue artifacts:

- queue:
  `/workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/queue.yaml`
- watcher PID file:
  `/workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/watcher.pid`
- watcher log:
  `/workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/watcher.log`
- remote queue script:
  `/workspace/interp/code/nano30b-nla-pilot-current/scripts/nano_ar_hpo_queue.py`

Policy:

- Run one probe at a time on the `train` workspace.
- Use `512/512` validation/test evals only.
- Do not queue `2048/2048` evals automatically.
- Keep W&B offline.
- Keep one model-only checkpoint per probe with `NLA_KEEP_LOCAL=1`.
- Do not start RL or AV+AR tuning during this phase.

Initial queued probes:

- `r27-wide-best1547-lr3e5-cos128`
- `r27-wide-best1547-lr1e5-constant128`
- `r27-wide-best1547-lr5e6-cos128`
- `r27-wide-fullscan-lr2e5-cos192`
- `r27-wide-fullscan-lr5e5-cos128`

Launch verification:

- Remote dry run resolved the first probe to `--num-rollout 128`, expected
  checkpoint `iter_0000128`, and `512/512` eval controls
  `teacher/teacher_shuffled/blank/generic/mean/source_context/source_raw`.
- Immediately after launch, the queue had `1` item in `training` and `4`
  `pending`; watcher PID was `1142`.

## 14. Nano AR HPO Results Captured So Far

On `2026-06-03`, the RunAI `train` workspace was recreated under the canonical
name with the original 2-GPU GH200 shape after the prior `train` pod failed.
The recreated workspace reached scheduler allocation but remained blocked in
`ContainerCreating` because the `/workspace/interp` PVC reported:

```text
volume pvc-6461f066-52f3-4ec5-aad7-17649ac29de5 is not ready for workloads
```

This means the latest remote queue/eval artifacts may exist on the PVC, but
they are not yet readable from the current pod. Do not treat the remaining
queued probes as complete until their JSON reports are recovered from
`/workspace/interp`.

Confirmed local AR eval reports under `artifacts/nano_ar_hpo_study/`:

| run | eval | validation teacher NMSE | test teacher NMSE | validation FVE | test FVE | validation cosine | test cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `r27-best1547-lr1e5-iter0000256` | `512/512` | `0.4417968094` | `0.4392344058` | `0.5017487246` | `0.4923874972` | `0.7791016102` | `0.7803828120` |
| `r27-best1547-lr2e5-iter0000256` | `512/512` | `0.4412533045` | `0.4374333322` | `0.5023616807` | `0.4944689540` | `0.7793734074` | `0.7812833190` |
| `r27-best1547-lr2e5-iter0000256` | `2048/2048` | `0.4351932406` | `0.4476521015` | `0.4949136702` | `0.4791074058` | `0.7824034095` | `0.7761739492` |

The `512/512` `lr2e-5` continuation is the best confirmed quick heldout point
so far by test teacher NMSE (`0.4374333322`). It beats the mean and shuffled
teacher controls strongly on both splits:

- `lr2e-5`, `512/512`: teacher beats mean on validation/test
  `0.98828125/0.98828125`; teacher beats shuffled teacher
  `0.998046875/1.0`; teacher beats source-context
  `0.64453125/0.685546875`.
- `lr1e-5`, `512/512`: teacher beats mean on validation/test
  `0.98828125/0.98828125`; teacher beats shuffled teacher
  `0.998046875/1.0`; teacher beats source-context
  `0.669921875/0.6875`.

Additional wide-queue results were captured in-session before the PVC became
unreadable:

| run | validation teacher NMSE | test teacher NMSE | validation FVE | test FVE | validation cosine | test cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r27-wide-best1547-lr3e5-cos128` | `0.4417034388` | `0.4410030842` | `0.5018540266` | `0.4903434786` | `0.7791483402` | `0.7794984579` |
| `r27-wide-best1547-lr1e5-constant128` | `0.4420669079` | `0.4399488568` | `0.5014441120` | `0.4915618235` | `0.7789665461` | `0.7800256014` |

The current scientific read is unchanged: Nano AR is learning a real
explanation-to-activation mapping and beats the main negative controls, but the
confirmed teacher NMSE is still around `0.44`, not the target `0.25-0.30`.
The next decision should wait for PVC recovery if possible, because the
unconfirmed `lr5e6-cos128` and fullscan escape probes may already have run.

## 15. RunAI PVC Recovery and Completed Wide AR Probe Readback

Later on `2026-06-03`, the RunAI `train` workspace was recovered with the
original persistent PVC attached:

- workspace: `train`
- project: `trustworthy-ai-inference`
- pod: `train-0-0`
- GPUs: `2 x NVIDIA H200`
- `/workspace/interp`: Longhorn PVC, `1008G` total, `854G` used, `154G` free
- `/workspace/models`: model-store NFS, `1.4T` total, `968G` used, `460G` free

Root cause of the PVC failure was stale Longhorn attach/detach state from the
old `train-dev` path plus node disk pressure/stale replica state. The recovery
deleted only the completed old pod, repaired the stale Longhorn replica/snapshot
state, pruned unused node runtime artifacts, and recreated `train` with the same
PVC mounts. PVC data was not deleted.

Queue readback from
`/workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/queue.yaml` showed all
five wide probes complete: `5 complete`, `0 pending`, `0 training`,
`0 eval_running`, `0 failed`.

Recovered wide AR results:

| run | validation teacher NMSE | test teacher NMSE | validation FVE | test FVE | validation cosine | test cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r27-wide-best1547-lr3e5-cos128` | `0.4417034388` | `0.4410030842` | `0.5018540266` | `0.4903434786` | `0.7791483402` | `0.7794984579` |
| `r27-wide-best1547-lr1e5-constant128` | `0.4420669079` | `0.4399488568` | `0.5014441120` | `0.4915618235` | `0.7789665461` | `0.7800256014` |
| `r27-wide-best1547-lr5e6-cos128` | `0.4429176748` | `0.4404751062` | `0.5004846309` | `0.4909536499` | `0.7785412073` | `0.7797624469` |
| `r27-wide-fullscan-lr2e5-cos192` | `0.4926010668` | `0.4873670042` | `0.4444525072` | `0.4367618257` | `0.7536994815` | `0.7563165426` |
| `r27-wide-fullscan-lr5e5-cos128` | `0.4728046060` | `0.4699196219` | `0.4667786347` | `0.4569253403` | `0.7635977268` | `0.7650401592` |

Scientific read: the current `best1547` basin remains stable around
`0.44` teacher NMSE. The fullscan escape probes were worse, not better, so the
next AR step should diagnose bottlenecks instead of simply widening this same
short-run sweep. The best confirmed quick point remains the earlier
`r27-best1547-lr2e5-iter0000256` `512/512` eval with test teacher NMSE
`0.4374333322`.

## 16. Final AV Checkpoint Standalone Generation Sanity

After RunAI recovery, a short final-AV generation sanity was run on the completed
AV hero checkpoint:

- checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-hero/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gloo-tokenized-gb192-mb8-save100-20260528T0110Z/hf_iter_0000467`
- report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-hero/nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gloo-tokenized-gb192-mb8-save100-20260528T0110Z/eval_iter_0000467_v8_t8_gen2_sanity_report.json`
- eval size: `8` validation rows, `8` test rows, `2` generated validation
  examples, `80` max new tokens

Teacher-forced NLL controls:

| control | validation NLL | test NLL |
| --- | ---: | ---: |
| `real` | `0.8636959568` | `1.0225046575` |
| `shuffled` | `1.2146585882` | `1.3821892142` |
| `zero` | `1.1028810516` | `1.2680757344` |
| `mean` | `1.0978926346` | `1.3128505871` |
| `none` | `1.2453771383` | `1.4175161719` |

Real activation gaps versus controls:

- validation: `+0.3509626314` vs shuffled, `+0.2391850948` vs zero,
  `+0.2341966778` vs mean, `+0.3816811815` vs none
- test: `+0.3596845567` vs shuffled, `+0.2455710769` vs zero,
  `+0.2903459296` vs mean, `+0.3950115144` vs none

Standalone generation sanity:

- Example 0 target was a job-advertisement continuation after "And". Real AV
  generation correctly described coordinating conjunction/list continuation,
  employee benefits/growth opportunities, and corporate promotional tone
  (`content_f1=0.4835`). Shuffled, zero, and no-injection generations drifted to
  unrelated legal/weather/Japanese-biography domains.
- Example 1 target was a customer-service responsibility list after an active
  listening gerund phrase. Real AV generation correctly stayed on job
  responsibilities and formal job-description register (`content_f1=0.4694`).
  Shuffled, zero, and no-injection generations again drifted to unrelated
  academic/weather/Japanese-biography domains.

Scientific read: AV standalone verbalization is coherent on this short sanity
check and the real activation clearly carries document-specific information
that the controls do not. This is not a broad qualitative eval, but it is enough
to support treating the AV hero checkpoint as a usable standalone verbalizer for
manual inspection while AR remains the weaker leg.

## 17. AR Phase-2 Gate, S3 Sync, and Queued Bounded Probes

On `2026-06-03`, the `train` RunAI workspace was used for the next AR phase.
S3 sync between the Mac and RunAI was made reliable by using the existing
RunAI S3 credentials plus a proxy override: the helper-set `NO_PROXY/no_proxy`
included `pdx.s8k.io`, which forced direct pod-to-S3 traffic and timed out.
Removing `pdx.s8k.io/.s8k.io` from both variables lets S3 traffic use the
egress proxy. Source bundles were then uploaded from the Mac by presigned PUT
and downloaded in the pod by presigned GET.

PVC cleanup/archival was started in the background:

- archive log:
  `/workspace/interp/logs/nano30b_s3_archive_ar_cleanup_20260603T2213Z.log`
- destination prefix:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/runai-outputs-archive/`
- initial local PVC state: `/workspace/interp` `1008G` total, `854G` used,
  `154G` free
- archive candidates: old bulky AR wide/fullscan/critic-init checkpoint dirs
  only; the final AV hero and current best tiny AR run metadata were left local

The current best AR checkpoint had already archived its model checkpoint shards
to S3:

- run:
  `nano-ar-r27-best1547-continue-lr2e5-cosine-256steps-20260602T0710Z`
- archive manifest:
  `checkpoints.s3_archived.json`
- S3 checkpoint prefix:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/checkpoint-archives/miles-fsdp2-ar-sft-hpo/nano-ar-r27-best1547-continue-lr2e5-cosine-256steps-20260602T0710Z/checkpoints/`
- archived bytes: `65,980,972,670`

The current-best checkpoint restore was started in the background so bounded
continuation probes can resume from exact Miles shards:

- restore log:
  `/workspace/interp/logs/nano30b_s3_restore_best_ar_ckpt_20260603T2221Z.log`
- target:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-hpo/nano-ar-r27-best1547-continue-lr2e5-cosine-256steps-20260602T0710Z/checkpoints`

Reusable AR split parquets were copied to a protected data directory before the
old fullscan run directory is archived/deleted:

- protected split dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/ar-r27-275k-splits-20260530`
- train rows: `247,870`
- validation rows: `13,761`
- test rows: `13,765`
- doc overlap in audit: `0`

Correctness audit result on restored current-best metadata:

- checkpoint dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-hpo/nano-ar-r27-best1547-continue-lr2e5-cosine-256steps-20260602T0710Z/checkpoints/iter_0000256/hf`
- report:
  `ar_correctness_audit_iter0000256_hf_20260603T2218Z.json`
- passed: `true`
- HF `num_hidden_layers`: `28`
- sidecar `critic.extraction_layer_index`: `27`
- value-head identity distance: `0.0932226554`
- doc split overlap: `0`

This confirmed the intended R_27/K+1 critic interpretation: R_27 has extraction
hidden-state index `27`, and the truncated critic has `28` hidden layers. The
local audit script was updated accordingly.

Information-ceiling diagnostic on a quick bounded sample:

- report:
  `ar_information_ceiling_t5000_v512_t512_20260603T2219Z.json`
- train sample: `5,000`
- validation/test: `512/512`
- feature dim: `256`
- kNN k: `8`

The raw diagnostic `mean_nmse` is per-dimension; multiplying by `d_model=2688`
puts it on the same rough scale as the eval reports. Under that scale, the
simple text-hash kNN explanation floor is worse than the mean baseline:

| split | kNN raw | kNN x2688 | mean-baseline raw | mean-baseline x2688 |
| --- | ---: | ---: | ---: | ---: |
| validation | `0.0003419988` | `0.9193` | `0.0003297465` | `0.8864` |
| test | `0.0003361402` | `0.9035` | `0.0003217154` | `0.8647` |

Read: there is no easy duplicate/retrieval shortcut in teacher explanation
text. Current learned AR at test teacher NMSE `0.4374333322` remains much better
than this simple retrieval floor and the mean control, but the source-raw eval
control around `0.13` still shows much more information is available in the
original token stream than in explanation text alone.

A post-restore watcher was started:

- log:
  `/workspace/interp/logs/nano30b_ar_phase2_after_restore_20260603T2225Z.log`
- behavior: wait for best-checkpoint restore, run a `512/512` bootstrap +
  prediction-dump eval on current best, then process the phase-2 AR queue twice
  sequentially.

Queued phase-2 bounded probes:

| queue item | config | purpose |
| --- | --- | --- |
| `r27-best256-polish-lr1e6-cos128` | `configs/nano_ar/hpo/r27_best256_polish_lr1e6_cosine_128steps.yaml` | gentle low-LR polish from current best |
| `r27-best256-batch384-lr2e5-cos128` | `configs/nano_ar/hpo/r27_best256_batch384_lr2e5_cosine_128steps.yaml` | larger-batch current-basin probe |

Both use W&B offline, `128` continuation steps, `512/512` bounded evals, the
protected 275k split path, and current-best checkpoint resume once restore is
complete.

## 16. 2026-06-04 AR Phase-2 Completion And W&B Log Sync

Phase-2 queue status after the post-restore watcher:

- queue:
  `configs/nano_ar/hpo/r27_best256_phase2_queue.yaml`
- completed items: `2`
- pending/training/eval-running/failed items: `0/0/0/0`
- `r27-best256-polish-lr1e6-cos128` completed at `2026-06-04T18:48:55Z`
- `r27-best256-batch384-lr2e5-cos128` completed at `2026-06-04T21:37:28Z`

All RunAI W&B offline logs under `/workspace/interp/outputs/nano30b-nla-pilot`
were synced locally, including runs whose model checkpoints were later deleted
or archived:

- local bundle:
  `artifacts/runai_wandb/all_outputs_wandb_20260604T2140Z`
- archive:
  `artifacts/runai_wandb/all_outputs_wandb_20260604T2140Z/nano30b_all_wandb_logs_20260604T2140Z.tgz`
- manifest:
  `artifacts/runai_wandb/all_outputs_wandb_20260604T2140Z/remote_wandb_manifest.txt`
- local extracted size: `496M`
- offline W&B run directories: `183`
- remote W&B roots in manifest: `74`
- archive integrity check: `tar_ok`

The synced manifest includes the deleted-checkpoint HPO runs, including:

- `nano-ar-r27-best1547-continue-lr1e5-cosine-256steps-20260601T195632Z`
- `nano-ar-r27-best1547-continue-lr2e5-cosine-256steps-20260602T0710Z`
- `nano-ar-r27-best256-batch384-lr2e5-cosine-128steps`
- `nano-ar-r27-best256-polish-lr1e6-cosine-128steps`
- `nano-ar-r27-fullscan-continue-qwen-lr2e5-cosine-256steps-20260601T0019Z`
- `nano-ar-r27-wide-probe-best1547-lr1e5-constant-128steps`
- `nano-ar-r27-wide-probe-best1547-lr3e5-cosine-128steps`
- `nano-ar-r27-wide-probe-best1547-lr5e6-cosine-128steps`
- `nano-ar-r27-wide-probe-fullscan-lr2e5-cosine-192steps`
- `nano-ar-r27-wide-probe-fullscan-lr5e5-cosine-128steps`

Bounded `512/512` heldout eval comparison:

| Run | Validation teacher NMSE | Validation FVE | Test teacher NMSE | Test FVE | Test cosine | Test source_raw NMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current best, `r27-best1547-lr2e5-iter0000256` | `0.4412533045` | `0.5023616807` | `0.4374333322` | `0.4944689540` | `0.7812833190` | `0.1301930100` |
| polish, `lr1e-6/cosine/128` | `0.4410410821` | `0.5026010216` | `0.4375066757` | `0.4943841927` | `0.7812466621` | `0.1300537884` |
| batch probe, `batch384/lr2e-5/cosine/128` | `0.4412363172` | `0.5023808387` | `0.4385614991` | `0.4931651590` | `0.7807192802` | `0.1307444274` |

Interpretation:

- The low-LR polish run is effectively flat: validation teacher NMSE improves
  by only about `0.00021`, while test teacher NMSE is slightly worse by about
  `0.00007`.
- The larger-batch continuation is worse on test by about `0.00113`.
- The current best AR checkpoint remains
  `r27-best1547-lr2e5-iter0000256`, with test teacher NMSE `0.4374333322`.
- Plain continuation HPO in the same LR/batch basin is not the next likely
  lever.
- The source_raw control remains much stronger at about `0.130` test NMSE,
  which means the model path can reconstruct R27 when the prompt contains
  low-level token-stream information. The open question is whether teacher text
  needs compact lexical/positional hints, a better readout/head, or both.

## 17. 2026-06-04 AR Diagnostics After LR Polish Failure

After the phase-2 LR polish and larger-batch probes failed to improve the
current best AR checkpoint, the next diagnostic round tested information
ceilings and frozen-signal controls before launching any more training.

Artifacts:

- remote output root:
  `/workspace/interp/outputs/nano30b-nla-pilot/ar-diagnostics-20260604T2158Z`
- local report copies:
  `artifacts/runai_diagnostics/ar-diagnostics-20260604T2158Z/`
- sidecar-aware ceiling report:
  `ar_information_ceiling_t20000_v512_t512_dim512_sidecarfix.json`
- teacher-plus-hint ceiling report:
  `ar_information_ceiling_t10000_v512_t512_dim512_hints.json`
- frozen signal-gate report:
  `ar_signal_gate_r27_2048_allvars_20260604T2205Z.json`

The information-ceiling script was fixed to understand the Stage 3 sidecar
column names used by the current AR split parquets:

- `token_text` as target-token text
- `token_id` as target-token ID
- `token_position` as target position

Local regression test:

```text
python -m pytest tests/test_nano_ar_information_ceiling.py -q
6 passed
```

### Sidecar-Aware Information Ceiling

Run shape:

- train rows: `20,000`
- validation/test rows: `512/512`
- text-hash feature dim: `512`
- kNN k: `8`
- scale used for comparison to eval reports: multiply per-dimension NMSE by
  `d_model=2688`

Results:

| split | explanation kNN, scaled | target-token floor, scaled | target-token-ID floor, scaled | position-bucket floor, scaled | mean floor, scaled |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | `0.892930` | `0.837941` | `0.838043` | `0.879786` | `0.886477` |
| test | `0.879786` | `0.802844` | `0.802844` | `0.860282` | `0.864776` |

Read: token identity gives a modest cheap-baseline improvement over the mean
floor, especially on test, but it is still far worse than the learned AR
checkpoint at test teacher NMSE `0.4374333322`. Token ID/text alone is not the
missing lever.

### Teacher-Plus-Hint kNN Ceiling

Run shape:

- train rows: `10,000`
- validation/test rows: `512/512`
- text-hash feature dim: `512`
- kNN k: `8`

Results:

| split | explanation only | + target token | + target-token ID | + position bucket | + all token hints |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | `0.909396` | `0.902429` | `0.905050` | `0.908522` | `0.897769` |
| test | `0.891266` | `0.887755` | `0.888921` | `0.892173` | `0.881065` |

Read: adding compact token/position hints to teacher text helps slightly in the
cheap text-hash retrieval view, but does not move anywhere near the desired
`0.25-0.30` teacher NMSE band, nor near the current trained AR `0.437` band.
This does not rule out using hints in a trained critic prompt, but it weakens
the case that hints alone are a large missing information source.

### Base-Nano Frozen Signal Gate

Run shape:

- model: base `/workspace/interp/models/nano-30b-a3b-bf16-hf`
- boundary: `R27`
- AR parquet: protected 275k train split
- max records: `2,048`
- train/eval split: `1,543/505`, doc-random, doc overlap `0`
- variants:
  `teacher`, `teacher_shuffled`, `blank`, `generic`, `source_context`,
  `source_raw`
- source token prefix coverage: `2,048/2,048`
- W&B mode: offline
- output payload size: lightweight only; no model checkpoint written

Heldout metrics:

| variant | frozen feature NMSE | frozen feature cosine | trained head NMSE | trained head cosine |
| --- | ---: | ---: | ---: | ---: |
| teacher | `1.362154` | `0.318923` | `0.921865` | `0.539068` |
| teacher_shuffled | `1.361897` | `0.319052` | `0.967175` | `0.516413` |
| blank | `1.405324` | `0.297339` | `0.833724` | `0.583138` |
| generic | `1.332541` | `0.333730` | `0.833728` | `0.583136` |
| source_context | `1.340004` | `0.329998` | `0.888972` | `0.555514` |
| source_raw | `0.001598` | `0.999202` | `0.004694` | `0.997653` |

Additional comparison fields:

- heldout mean control NMSE: `0.833282`
- teacher head NMSE: `0.921865`
- source_raw oracle passed: `true`
- teacher beats controls: `false`
- scientific pass: `false`

Read:

- The source-token oracle is essentially perfect, so extraction and R27 geometry
  remain sound.
- Base Nano frozen features for teacher text are not enough: the teacher head is
  worse than the mean/blank/generic controls.
- The trained AR checkpoint at `0.437433` is much better than this scratch
  frozen-head signal gate, so AR-SFT has learned real teacher-side structure.
- The remaining likely bottleneck is not generic LR polish or batch size. It is
  either teacher-text information loss, the trained critic's prompt/readout
  structure, or a need for a source/raw-to-teacher geometry curriculum.

Next decision:

Do not queue another same-basin LR/batch continuation from current best. The
next useful experiment should be mechanism-targeted:

1. Build or extend a checkpoint-backbone readout diagnostic for the current
   trained AR checkpoint, not base Nano, to test whether the trained hidden
   features contain a sub-`0.40` linear/ridge readout.
2. If trained-checkpoint hidden features beat `0.40`, run a head/readout probe
   with head-high/backbone-low learning rates.
3. If trained-checkpoint hidden features do not beat `0.40`, prioritize
   source_raw/source_context geometry curriculum or prompt/data enrichment over
   further optimizer sweeps.

## 2026-06-05 - R27 Backup, Readout Diagnostic, and R34 Mini-Probe Launch

### Checkpoint Backup State

The best current R27 AV/AR model-only checkpoints are now backed up under
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/checkpoints/r27-av-ar-best/`.
Optimizer/exact-resume shards were intentionally not uploaded for this backup
pass.

| artifact | S3 prefix | objects | total bytes |
| --- | --- | ---: | ---: |
| AV HF model-only iter 0000467 | `av_hf_iter_0000467/` | `31` | `63,174,926,935` |
| AR HF model-only iter 0000256 | `ar_hf_iter_0000256/` | `10` | `32,993,013,084` |
| AV exact/optimizer prefix | `av_exact_iter_0000467/` | `0` | `0` |

### R27 Trained-Checkpoint Readout Diagnostic

Run:
`/workspace/interp/outputs/nano30b-nla-pilot/readout_jobs/r27_best_readout_20260605T010828Z`.

Report:
`readout_report_t1024_v256_t256.json`.

Key results:

- best validation readout: `current_value_head`
- validation normalized MSE: `0.4315264523`
- test current-head normalized MSE: `0.4474605024`
- test current-head cosine: `0.7762697935`
- test current-head FVE: `0.4878954393`
- test mean-control normalized MSE: `0.8737678528`
- `head_bottleneck_likely`: `false`
- `strong_head_bottleneck_signal`: `false`

Read: a closed-form/ridge readout did not beat the trained value head on the
bounded split. The current R27 AR bottleneck is therefore unlikely to be just a
final linear head underuse issue. This points the next experiment toward layer
choice / available target information / prompt geometry rather than another
head-only polish run.

### R34 Mini-Probe Launch

Goal: test whether a later Nano residual boundary, closer to the Qwen layer
ratio, gives a better AR teacher-reconstruction basin without adding new data or
switching algorithms.

Config:
`configs/nano_ar/hpo/r34_mini_probe_20k_lr2e5_cosine_128steps.yaml`.

Dataset slice:

- source docs: FineWeb train doc IDs `10500` through `12547`
- teacher table: `20,416` rows from the existing R27 teacher explanations
- planned R34 AR-SFT parquet:
  `/workspace/interp/outputs/nano30b-nla-pilot/r34_probe/ar_sft_r34_start10500_len2048.parquet`
- doc-level split: `90/5/5`
- eval limits: validation/test `512/512`
- controls: `teacher`, `teacher_shuffled`, `blank`, `generic`, `mean`,
  `source_context`, `source_raw`

Execution note:

- First R34 extraction with `device_map=auto` failed because manual boundary
  replay crossed CUDA devices while `attention_mask` / `cache_position` stayed
  on the previous device.
- A one-doc R34 extraction with `CUDA_VISIBLE_DEVICES=0` completed with
  `row_count=1` and no blockers, confirming the extraction math works when the
  model is not sharded across devices.
- The queued R34 pipeline was relaunched as PID `405676` with only the
  extraction command forced to one visible H200. Merge/build/train/eval remain
  unchanged and should use the normal environment once extraction completes.

Local code follow-up:

- `scripts/nano_extraction_identity.py` now includes a sharded-model-safe tensor
  device handoff in `prefix_forward_to_R_b`.
- `tests/test_nano_harness.py` includes a CPU-safe regression test for selecting
  a block's real execution device before fallback.
- Local verification: `python -m pytest tests/test_nano_harness.py -q` passed
  with `61` tests.

### R34 Mini-Probe Result and Layer-Sweep Follow-Up

R34 completed with checkpoint:
`/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r34-probes/nano-ar-r34-mini-20k-lr2e5-cosine-128steps/checkpoints/iter_0000128`.

Bounded eval:
`eval_iter_0000128_v512_t512_winrates_report.json`.

Key heldout results:

| split | teacher normalized MSE | mean | shuffled teacher | source_context | source_raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | `0.4907283187` | `0.8797792792` | `1.1466460228` | `0.5320481062` | `0.1128209010` |
| test | `0.5013990998` | `0.8806790113` | `1.1520001888` | `0.5399370790` | `0.1190299168` |

Read: R34 is a real positive AR signal on this 20k slice because teacher text
beats mean, blank/generic, and shuffled controls. However, this R34 mini-probe
does not yet beat the current R27 best checkpoint/readout neighborhood
(`~0.43` validation / `~0.45` test normalized MSE). The next move should
therefore be a scalable layer choice scan before spending more SFT runs on any
single boundary.

Layer-sweep infrastructure added:

- `scripts/nano_ar_layer_sweep.py`
- `tests/test_nano_ar_layer_sweep.py`
- `configs/nano_ar/layer_sweep/r25_r51_20k_queue.yaml`
- cluster code-sync snapshot:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/code-sync/layer_sweep_r25_r51_20260605/`

The queue captures R25 through R51 on the same FineWeb `10500:12548` / `20,416`
teacher-matched slice, then scores every boundary with a cheap teacher-text kNN
readout against the mean-h control. The intent is to identify 2-4 candidate
Nano boundaries where text predicts activations better than mean before running
another AR-SFT checkpoint.

Operational note: use `scripts/nano_s3.py` on the RunAI `train` PVC for future
code/artifact/checkpoint listing, upload, and download. The verified credentials
are mounted only on the cluster side, so Mac-to-cluster sync should prefer S3
when the source is already on the PVC and avoid ad hoc stdin/chunk transfer
except as an emergency bootstrap.

Local verification:

- `python -m pytest tests/test_nano_ar_layer_sweep.py -q` passed with `4` tests.
- `python -m pytest tests/test_nano_ar_hpo_queue.py tests/test_nano_ar_layer_sweep.py -q` passed with `13` tests.

### R25-R51 Layer-Sweep Analysis Result

Run:
`/workspace/interp/outputs/nano30b-nla-pilot/layer_sweeps/r25_r51_20k_start10500_len2048`.

S3 reports:

- `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/layer-sweeps/r25_r51_20k_start10500_len2048/score_r25-r51_teacher_knn_report.json`
- `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/layer-sweeps/r25_r51_20k_start10500_len2048/score_r25-r51_teacher_knn_summary.csv`
- `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/layer-sweeps/r25_r51_20k_start10500_len2048/score_r25-r51_teacher_knn_analysis.md`

Extraction/scoring completed cleanly:

- boundaries: `R25-R51` (`27` layers)
- extracted rows: `20,480` per layer
- teacher-matched rows: `20,416` per layer
- blockers: `0`
- skipped docs/positions: `0`
- local footprint: about `2.8G`

Important result: the cheap teacher-text kNN proxy did **not** beat the mean-h
control on any layer. Validation and test both had `any_teacher_beats_mean =
false`.

Top layers by validation teacher-kNN NMSE:

| layer | val teacher | val mean | val delta | test teacher | test delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| R51 | `0.493779` | `0.467005` | `-0.026774` | `0.456026` | `-0.023798` |
| R50 | `0.598420` | `0.551676` | `-0.046745` | `0.564257` | `-0.044533` |
| R26 | `0.607077` | `0.557400` | `-0.049677` | `0.590049` | `-0.044113` |
| R33 | `0.608475` | `0.569660` | `-0.038815` | `0.590450` | `-0.032761` |
| R49 | `0.633829` | `0.583561` | `-0.050268` | `0.598202` | `-0.047076` |
| R42 | `0.637196` | `0.586259` | `-0.050937` | `0.616178` | `-0.047581` |

Read: this sweep is useful as a geometry/diagnostic pass, but the hash-kNN
screen is not a reliable standalone AR-SFT predictor. R34 AR-SFT produced a real
teacher-over-mean signal despite R34 being negative in this cheap kNN screen.
The sweep therefore argues against picking a new layer solely from kNN. It does
suggest that if we spend one more short SFT probe, the most informative set is:
R51 as the best absolute kNN/mean geometry endpoint, R33/R34 as the Qwen-ratio
candidate, and R27 as the established baseline/control.

### R33/R51 Matched Mini-Probe Launch

Goal: compare R33 and R51 against the completed R34 mini-probe using the same
20k teacher-matched slice, same optimizer schedule, same `512/512` bounded eval,
and same controls. This is a better comparison than the cheap kNN screen because
R34 showed that trained AR-SFT can find signal even when kNN does not.

Configs:

- `configs/nano_ar/hpo/r33_mini_probe_20k_lr2e5_cosine_128steps.yaml`
- `configs/nano_ar/hpo/r51_mini_probe_20k_lr2e5_cosine_128steps.yaml`

Pipeline:

- `scripts/nano_ar_layer_probe_pipeline.sh`
- queue:
  `/workspace/interp/outputs/nano30b-nla-pilot/layer_probe_jobs/r33_r51_20260605/r33_r51_probe_queue.yaml`
- code/config S3 snapshot:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/code-sync/layer_probe_r33_r51_20260605/`

Prep status:

- R33 AR-SFT parquet: `20,416` rows, verify clean.
- R51 AR-SFT parquet: `20,416` rows, verify clean.
- R33 critic init prepared.
- R51 critic init prepared.
- Initial merge bug fixed: PyArrow `Table.join` could not carry fixed-size
  `activation_vector`; the pipeline now uses the proven manual teacher-key
  lookup from the R34 path.

Run status:

- R33 queued first and started training at `2026-06-05T19:45:30Z`.
- R51 is pending behind R33.
- Early R33 train-side metrics:
  - step `0`: loss `1.2419206301`, FVE `-1.2243885994`
  - step `1`: loss `0.9577531020`, FVE `-0.7154193719`

Read: R33 launch is healthy and directionally improving on train loss/FVE, but
no conclusion should be drawn until the bounded heldout eval completes.

### R33/R34 AV Medium Probe OOM and Capped Relaunch Plan

Goal: test whether the promising AR layers R33/R34 also support AV, using the
same `20,416` row teacher-matched slice and the AV injection path. This checks
the practical K/K+1 pairing risk: a strong AR boundary is only useful for a full
NLA if the matching AV side is also viable.

Uncapped configs launched:

- `configs/nano_av/layer_probe/r33_av_probe_20k_lr1e5_128steps.yaml`
- `configs/nano_av/layer_probe/r34_av_probe_20k_lr1e5_128steps.yaml`
- queue: `configs/nano_av/layer_probe/r33_r34_av_20k_queue.yaml`

Outcome:

- R33 failed at `2026-06-05T22:27:32Z` with CUDA OOM during the first training
  step.
- R34 started at `2026-06-05T22:27:37Z`, completed one optimizer step, then
  failed at `2026-06-05T22:32:52Z` with CUDA OOM on rollout `1`.
- The observed training tensors were full-vocab logits at sequence lengths
  around `1920-2048`, e.g. `logits_shape=(1, 2048, 131072)`.
- R34 reached about `121G` allocated / `132G` reserved before the second-rollout
  OOM on H200, so this is a memory/sequence-length issue rather than a dataset
  or authentication issue.

Local code change prepared:

- `training.max_sequence_tokens` and `training.max_response_tokens` are now
  optional YAML knobs in `scripts/nano_av_runner.py`.
- The AV SFT rollout honors those knobs through
  `NLA_SFT_MAX_SEQUENCE_TOKENS` and `NLA_SFT_MAX_RESPONSE_TOKENS`.
- Focused verification passed:
  `python -m pytest tests/test_nano_av_runner_spec.py tests/test_nano_ar_hpo_queue.py tests/test_nano_ar_layer_sweep.py -q`
  returned `29 passed`.

Capped relaunch configs prepared locally:

- `configs/nano_av/layer_probe/r33_av_probe_20k_lr1e5_128steps_seq1152.yaml`
- `configs/nano_av/layer_probe/r34_av_probe_20k_lr1e5_128steps_seq1152.yaml`
- queue: `configs/nano_av/layer_probe/r33_r34_av_20k_seq1152_queue.yaml`

Intention: rerun the same AV probe with `max_sequence_tokens: 1152` and
`max_response_tokens: 1024`. This should cut the AV logits memory materially
while preserving enough explanation tokens for a bounded AV signal test.

### R33/R34 Corrected AV Evals and R33 Scaling Decision

Date: `2026-06-07`

The capped R33/R34 AV probes were rerun with dynamic packed-token control and
the corrected eval path. The original long-running queue still attempted to
eval `checkpoints/iter_0000128/hf`, which failed because the AV checkpoints are
Miles/FSDP DCP checkpoints. A separate eval-only retry path converted
`iter_0000128` to a temporary HF model-only checkpoint, ran the bounded eval,
then deleted the temporary HF copy and DCP model shards.

Corrected AV eval reports:

- R33:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-layer-probes/nano-av-r33-medium-probe-20k-lr1e5-128steps-seq1152-dyn512/eval_iter_0000128_v512_t512_gen4_report.json`
- R34:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-layer-probes/nano-av-r34-medium-probe-20k-lr1e5-128steps-seq1152-dyn512/eval_iter_0000128_v512_t512_gen4_report.json`

Corrected AV `512/512` NLL:

| Layer | Validation real | Test real | Validation shuffled | Test shuffled | Validation mean | Test mean | Validation none | Test none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R33 | `1.040335` | `1.015130` | `1.430425` | `1.406198` | `1.299897` | `1.269933` | `1.453389` | `1.428713` |
| R34 | `1.037261` | `1.013677` | `1.429958` | `1.406277` | `1.302104` | `1.270672` | `1.447475` | `1.422890` |

Read:

- Both R33 and R34 AV probes pass the row-specificity gate: real activations
  beat shuffled, mean, zero, and no-injection controls on validation and test.
- R34 AV is slightly better than R33 AV, but only by about `0.0031` validation
  NLL and `0.0015` test NLL.
- The DCP-to-HF eval path is now validated end-to-end for AV probes. Temporary
  HF checkpoints and DCP model shards were cleaned after successful evals.

Matched AR `20k` probe comparison:

| Layer/run | Validation teacher NMSE | Test teacher NMSE | Validation cosine | Test cosine | Validation source_raw NMSE | Test source_raw NMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R27 tuned/fullscan basin | `~0.441` | `~0.437` | `~0.779` | `~0.781` | `~0.133` | `~0.130` |
| R33 mini 20k | `0.381983` | `0.388301` | `0.809009` | `0.805849` | `0.071066` | `0.076216` |
| R34 mini 20k | `0.490728` | `0.501399` | `0.754636` | `0.749300` | `0.112821` | `0.119030` |

Updated interpretation:

- The earlier conservative conclusion that R27 should remain the mainline was
  too dependent on maturity rather than scale-matched trajectory.
- R27 is still the best completed fallback because it has full AV and AR hero
  artifacts, but its AR tuning has plateaued around `0.44` teacher NMSE and its
  AV path already consumed the 100k hero effort.
- R33 is the most promising scaling layer: at only `20,416` AR rows and a short
  `128`-rollout SFT probe, it already beats the mature R27 tuned AR basin by
  about `0.05` teacher NMSE on both validation and test.
- R33 also has a much stronger source_raw floor (`~0.07`) than R27 (`~0.13`),
  suggesting the R33 target geometry is more reconstructable when the text
  channel carries low-level information.
- R34 is not the immediate scaling target despite its marginal AV win, because
  its matched AR probe is much worse than both R33 and the tuned R27 basin.

Decision:

- Treat R27 as the stable fallback/baseline.
- Move the main scaling effort to R33.
- Scale R33 AV and AR through the same staged HPO discipline used for R27:
  bounded medium/100k probes first, then hero/full-data only after heldout
  controls and round-trip checks pass.

Recommended R33 next phase:

1. Build/verify a larger R33 AR dataset, starting with a 100k-style slice before
   spending on full 275k.
2. Run R33 AR 100k HPO probes:
   - baseline `lr=2e-5`, cosine, `gb192/mb8`
   - stability variant `lr=1e-5`, cosine, same batch
   - optional schedule/steps variant only if the first two separate clearly
3. Run R33 AV 100k with the capped/dynamic packed-token path validated by the
   corrected R33/R34 AV probes.
4. Add the key NLA gate: evaluate `h -> AV-generated explanation -> AR h_hat`.
   Teacher-text AR is a useful proxy, but the chained AV+AR round trip is the
   mechanism that determines actual NLA usefulness.
5. Promote R33 to a hero/full-data run only if the 100k R33 AR moves toward the
   `0.25-0.30` teacher NMSE target and the AV-generated round-trip beats the R27
   baseline materially.

Storage and operational notes:

- Keep W&B offline logs and eval reports.
- Use temporary DCP-to-HF conversion for evals, then delete the HF model-only
  directory after the report is written.
- For probe checkpoints, keep only what is needed for immediate resume/eval;
  upload selected model-only checkpoints to S3 before deleting larger local
  shards.

## 2026-06-07 - R33 Scaling Kickoff Prep

Objective: start the R33 scaling phase without launching RL or claiming NLA
success from teacher-text proxies alone.

Local/RunAI access check from the Mac workspace:

- `visor` is not available on the local shell, so command execution fell back
  to bounded direct commands per the Visor guidance.
- Local `runai`, `aws`, `uvx`, and `python` are not on `PATH`.
- `python3 scripts/nano_s3.py env` confirmed the S3 helper is present and
  does not print secrets. As expected on the Mac, the RunAI PVC credential files
  `/workspace/interp/secrets/aws/credentials` and
  `/workspace/interp/secrets/aws/config` are absent.
- Therefore RunAI auth, PVC free disk, active Nano process state, and actual
  code sync could not be confirmed from this shell. No remote run was launched.

Prepared R33 scaling artifacts:

- Dataset/critic/AV prep script:
  `scripts/nano_ar_r33_scaling_pipeline.sh`
- R33 AR 100k HPO configs:
  - `configs/nano_ar/hpo/r33_100k_lr2e5_cosine_gb192_mb8.yaml`
  - `configs/nano_ar/hpo/r33_100k_lr1e5_cosine_gb192_mb8.yaml`
- R33 AR queue:
  `configs/nano_ar/hpo/r33_100k_scaling_queue.yaml`
- R33 AV 100k dynamic/capped config:
  `configs/nano_av/hpo/r33_100k_lr1e5_gb192_mb2_seq1152_dyn512.yaml`
- R33 AV queue:
  `configs/nano_av/hpo/r33_100k_scaling_queue.yaml`

The 100k prep script follows the storage-conscious 20k layer-probe pattern:

1. Extract `R33` activations over FineWeb `corpus_start=10500`,
   `corpus_length=10000`, `positions_per_doc=10`.
2. Merge reused R27 teacher explanations by safe row keys.
3. Build and verify `ar_sft_r33_start10500_len10000.parquet`.
4. Build and verify `av_sft_r33_start10500_len10000.parquet`.
5. Prepare `nano-ar-r33-critic-init` if missing.

The two AR configs are model-only tuning probes, not hero runs:

- `lr=2e-5`, cosine, `gb192/mb8`, one 100k train-split epoch, final save at
  `iter_0000467`, `512/512` eval with teacher, teacher-shuffled, blank,
  generic, mean, source-context, and source-raw controls.
- `lr=1e-5`, cosine, same batch and eval contract.

The AV config uses the corrected R33/R34 AV memory path:

- dynamic packed-token control with `max_tokens_per_gpu=512`
- sequence cap `max_sequence_tokens=1152`
- response cap `max_response_tokens=1024`
- model-only final save, temporary DCP-to-HF eval conversion, and cleanup of
  temporary HF and DCP model shards after eval.

Local verification:

- `bash -n scripts/nano_ar_r33_scaling_pipeline.sh` passed.
- Ruby YAML parsing and basic invariant checks passed for the new R33 AR/AV
  configs and queues.
- Full Python runner/queue tests could not run locally because both the system
  and bundled Python environments lack required packages (`pytest`, `pyarrow`,
  and/or `yaml`). These should be rerun in the RunAI venv before launch.

Next required RunAI actions once CLI access is available:

1. Check RunAI auth, `/workspace/interp` free disk, active Nano train/eval
   processes, and S3 helper environment without printing secrets.
2. Sync the local code/config changes to
   `/workspace/interp/code/nano30b-nla-pilot-current`.
3. Run `scripts/nano_ar_r33_scaling_pipeline.sh` on RunAI to build and verify
   the larger R33 AR/AV datasets and critic init.
4. Run the AR queue:
   `python scripts/nano_ar_hpo_queue.py configs/nano_ar/hpo/r33_100k_scaling_queue.yaml --run-until-empty`
5. Run the AV queue after AR/dataset sanity:
   `python scripts/nano_av_probe_queue.py configs/nano_av/hpo/r33_100k_scaling_queue.yaml --run-until-empty`
6. Add the actual round-trip gate before promotion:
   `h -> AV-generated explanation -> AR h_hat`, compared against the mature R27
   baseline. Teacher-text AR and AV real-vs-control losses remain proxies only.

## 2026-06-08 - R33 Hero-Size Prefix Dataset Verified

Correction to the previous RunAI note: the workspace was reachable through the
absolute Visor/RunAI paths. The `train` workspace was used for the R33 hero-size
dataset prep; no RL was launched.

Root cause of the failed 100k sharded FineWeb prep:

- Independent streaming `skip(start)` shards did not reproduce the teacher row
  keys after the first shard. The 100k attempt produced `99,990` activation rows
  but only `20,500` teacher-overlap rows.
- A boundary check showed `doc 2547` still matched teacher text while `doc 2548`
  diverged, so doc-suffix preflight was insufficient.
- The reusable fix is prefix-key extraction from the existing teacher-backed
  AR-SFT parquet, using each row's exact `token_ids_prefix` rather than
  re-streaming FineWeb.

Code changes for the reusable prefix path:

- `scripts/nano_prefix_activation_extract.py`: extracts arbitrary residual
  layers from exact `token_ids_prefix` groups, reusing the longest compatible
  prefix per doc and selecting all requested token positions in one forward.
- `scripts/nano_prefix_dataset_pipeline.sh`: generic prefix dataset pipeline for
  base extraction, AR-SFT, AV-SFT, critic init, and verifiers.
- `scripts/nano_prefix_dataset_sidecar.py`: shared AR/AV sidecar writer that
  normalizes training-facing sidecars to `kind: nla_dataset`,
  `schema_version: 1`, and injects token/template metadata from the source
  contract. This fixes both the critic-init `KeyError: 'kind'` and the AV
  verifier's missing injection-token metadata.

RunAI tests:

- `/workspace/interp/.venv/bin/python -m pytest tests/test_nano_prefix_dataset_pipeline.py tests/test_nano_prefix_dataset_sidecar.py tests/test_nano_prefix_activation_extract.py tests/test_nano_av_runner_spec.py tests/test_nano_ar_hpo_queue.py tests/test_nano_ar_layer_sweep.py -q`
- Result: `41 passed`.

Verified R33 fullscan artifacts:

- Output root:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_fullscan275396`
- Base:
  `base_R33_r33_prefix_fullscan275396.parquet`
- AR-SFT:
  `ar_sft_R33_r33_prefix_fullscan275396.parquet`
- AV-SFT:
  `av_sft_R33_r33_prefix_fullscan275396.parquet`
- Critic init:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-critic-init/nano-ar-r33-critic-init`
- AR verifier:
  `verify_ar_R33_r33_prefix_fullscan275396.json`
- AV verifier:
  `verify_av_R33_r33_prefix_fullscan275396.json`

Verifier results:

| Artifact | rows | d_model | nonfinite activations | bad prompt/text checks | 90/5/5 doc overlap |
|---|---:|---:|---:|---:|---:|
| R33 AR-SFT | 275,396 | 2,688 | 0 | empty explanations `0`, suffix bad `0` | 0 |
| R33 AV-SFT | 275,396 | 2,688 | 0 | malformed responses `0`, marker bad `0` | 0 |

Status: the R33 hero-size teacher-backed AR/AV datasets and R33 critic init are
now verified. This clears the dataset gate for bounded R33 AR/AV HPO. It is not
yet evidence of NLA success; the next gates are teacher-text AR evals,
AV real-vs-control evals, and the actual AV-generated-text -> AR reconstruction
round trip against the mature R27 baseline.

R33 HPO launch:

- Updated the 100k HPO configs to read from the verified
  `r33_prefix_fullscan275396` AR/AV parquets while keeping `row_limit=99,570`
  for bounded tuning.
- Launched the first AR HPO item with `--once`:
  `nano-ar-r33-100k-lr2e5-cosine-gb192-mb8`.
- Run dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr2e5-cosine-gb192-mb8`
- Started at `2026-06-08T14:28:28Z`; status at `2026-06-08T14:36:31Z`:
  training, step `5/467`, GPUs around `97-102 GiB`, no errors in `train.log`.
- Early train loss: step `0` `1.2477`, steps `1-5` around `0.95-0.97`.
  This is only a startup health signal, not a quality claim.

Prepared but blocked hero configs:

- `configs/nano_ar/hpo/r33_full275k_lr2e5_cosine_gb192_mb8.yaml`
- `configs/nano_ar/hpo/r33_full275k_lr1e5_cosine_gb192_mb8.yaml`
- `configs/nano_ar/hpo/r33_full275k_hero_queue.yaml`
- `configs/nano_av/hpo/r33_full275k_lr1e5_gb192_mb2_seq1152_dyn512.yaml`
- `configs/nano_av/hpo/r33_full275k_hero_queue.yaml`

The full-275k hero configs use the verified fullscan artifacts, 90/5/5 doc
splits, `gb192`, and one full train-split epoch (`1291` rollouts after padding).
They remain blocked until bounded HPO/eval and round-trip planning gates justify
promotion.

### 2026-06-08 - R33 Round-Trip Gate Prepared During AR HPO

Objective alignment:

- The R33 hero milestone still requires the chained
  `h -> AV-generated explanation -> AR h_hat` gate; teacher-text AR and AV
  real-vs-control losses remain proxy diagnostics only.
- While the first R33 AR 100k HPO run is training, prepared the reusable
  round-trip evaluator so the gate can run immediately after bounded AR and AV
  checkpoints are available.

Code added:

- `scripts/eval_nano_av_ar_roundtrip_gate.py`
  - reuses AV checkpoint generation semantics from
    `eval_nano_av_miles_checkpoint.py` / `nano_av_warmstart_smoke.py`;
  - reuses AR critic prediction and NMSE/cosine metrics from
    `eval_nano_ar_miles_checkpoint.py`;
  - generates bounded AV outputs for `real`, `shuffled`, `zero`, `mean`, and
    `none` controls, unloads the AV model, then loads the AR critic to score
    reconstructed activations;
  - writes generated JSONL plus a report with teacher, AV-generated, control,
    mean, rowwise win-rate, text-overlap, and optional R27 baseline comparison
    summaries.
- `tests/test_nano_av_ar_roundtrip_gate.py`
  - covers explanation extraction, critic prompt formatting, variant metric
    summaries, control/baseline gate logic, and generated JSONL round trips.

RunAI verification:

- Focused gate tests:
  `/workspace/interp/.venv/bin/python -m pytest tests/test_nano_av_ar_roundtrip_gate.py -q`
  -> `4 passed`.
- Regression slice:
  `/workspace/interp/.venv/bin/python -m pytest tests/test_nano_av_ar_roundtrip_gate.py tests/test_nano_prefix_dataset_pipeline.py tests/test_nano_prefix_dataset_sidecar.py tests/test_nano_prefix_activation_extract.py tests/test_nano_av_runner_spec.py tests/test_nano_ar_hpo_queue.py -q -k "not test_checked_in_r33_100k_queue_is_valid"`
  -> `38 passed, 1 deselected`.
- The deselected test assumes the checked-in R33 100k queue has its first item
  pending; the live RunAI queue YAML is currently `training/pending` because
  the first AR HPO item is active.

Live AR HPO status at `2026-06-08T14:48:33Z`:

- Run: `nano-ar-r33-100k-lr2e5-cosine-gb192-mb8`.
- Step: `21/467`.
- Latest logged loss: `0.8072489`; latest logged `fve_nrm=-0.4441`.
- No checkpoint or eval report yet. Do not infer quality until the bounded
  teacher/control eval report exists.

### 2026-06-08 - R33 AR 100k lr=2e-5 HPO Eval Complete

Run:

- `nano-ar-r33-100k-lr2e5-cosine-gb192-mb8`
- Completed at `2026-06-08T20:03:07Z`.
- Checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr2e5-cosine-gb192-mb8/checkpoints/iter_0000467`
- Eval report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr2e5-cosine-gb192-mb8/eval_iter_0000467_v512_t512_winrates_report.json`

Training health:

- Final train step: `466/466`.
- Final train loss: `0.331434`.
- Final train `fve_nrm`: `0.407104`.
- The train loss trend was healthy, but the quality decision below is from the
  heldout bounded eval report.

Bounded AR eval, 512 validation / 512 test:

| Split | teacher NMSE | teacher cosine | teacher FVE vs mean | teacher_shuffled NMSE | blank NMSE | generic NMSE | mean NMSE | source_context NMSE | source_raw NMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.334868 | 0.832566 | 0.506982 | 0.938597 | 0.788234 | 0.781042 | 0.679221 | 0.388525 | 0.081792 |
| test | 0.323239 | 0.838380 | 0.509667 | 0.920332 | 0.768147 | 0.763415 | 0.659224 | 0.382368 | 0.076623 |

Interpretation:

- This is a real AR improvement over the R33 20k probe
  (`0.381983/0.388301` validation/test teacher NMSE) and over the mature R27
  tuned AR baseline (`~0.441/0.437`).
- It has not yet reached the desired `0.25-0.30` teacher NMSE band, but test is
  close enough to justify the planned `lr=1e-5` stability comparison before
  choosing the full-275k hero AR hyperparameter.
- Teacher beats teacher_shuffled, blank, generic, and mean on both splits with
  very high rowwise win rates. Source_raw remains the oracle-like control and is
  correctly much better than teacher text.

Next action launched:

- Started pending stability trial
  `nano-ar-r33-100k-lr1e5-cosine-gb192-mb8` at `2026-06-08T20:28:38Z`.
- Queue status at `2026-06-08T20:29:13Z`: first trial `complete`, stability
  trial `training`, no first training step logged yet.

### 2026-06-08 - R33 AR Stability Trial Switched From mb8 to mb16

Reason:

- The `lr=1e-5, gb192, mb8` stability run showed low sampled GPU utilization
  during early training, with `num_microbatches=12` and roughly `48s` early step
  time. The user requested stopping it and switching to `mb16` to improve GPU
  utilization and reduce 100k runtime.

Action:

- Sent SIGTERM to the active mb8 queue/train/actor processes at
  `2026-06-08T20:36:28Z`.
- Preserved the partial mb8 run directory and log:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr1e5-cosine-gb192-mb8`
- Marked the mb8 queue item failed with failure:
  `operator stopped mb8 run before checkpoint to test mb16 throughput`.
- Added config:
  `configs/nano_ar/hpo/r33_100k_lr1e5_cosine_gb192_mb16.yaml`
- Added queue item:
  `r33-100k-lr1e5-cos-gb192-mb16`.

Operational note:

- The killed queue watcher left a stale flock in the RunAI process namespace.
  After confirming no train process and no GPU allocations, the mb16 run was
  launched directly through `nano_av_runner.prepare_run(...)` plus the same
  Miles command that the queue would render. The queue item was marked
  `training` with `launch_mode=direct_runner_due_stale_queue_lock`.
- Because this direct launch bypasses the queue watcher, the bounded AR eval
  must be run manually after `iter_0000467` is written, and the queue item must
  be marked complete manually after eval.

Early mb16 throughput:

- Run:
  `nano-ar-r33-100k-lr1e5-cosine-gb192-mb16`
- Direct launch started at `2026-06-08T20:41:55Z`.
- Confirmed command uses `--micro-batch-size 16`.
- Confirmed `num_microbatches=6`.
- No OOM through the first completed steps.
- Step 0 included first-step overhead and took about `77s`.
- Step 1 optimizer-step total was about `26.6s`, with step log interval
  `28s`, materially faster than the mb8 run's ~`41s` average.
- Peak observed memory during early mb16 training: about `103 GiB / 98 GiB`
  used on GPU0/GPU1, still below the `143 GiB` limit.

### 2026-06-08 - R33 AR Single-GPU mb192 Feasibility Probe Failed OOM

Reason:

- The user requested stopping the active `mb16` run and testing whether an AR
  HPO trial can run on one GPU with `global_batch_size=192` and
  `micro_batch_size=192`. If feasible, this would allow one AR/AV-style trial
  per H100 instead of tying up both GPUs for each 100k run.

Actions:

- Stopped `nano-ar-r33-100k-lr1e5-cosine-gb192-mb16` before checkpoint and
  removed its partial artifacts.
- Added config:
  `configs/nano_ar/hpo/r33_100k_lr1e5_cosine_gb192_mb192_gpu1.yaml`
- Launched direct single-GPU probe:
  `nano-ar-r33-100k-lr1e5-cosine-gb192-mb192-gpu1`
- Confirmed command used `--actor-num-gpus-per-node 1`,
  `--global-batch-size 192`, and `--micro-batch-size 192`.
- Confirmed actor log reported `num_microbatches=1` on `cuda_device=0`.

Outcome:

- The probe OOMed before completing step 0.
- Failure line: `CUDA out of memory. Tried to allocate 20.00 MiB`.
- GPU0 reached `139.79 GiB` process memory; PyTorch allocated `137.41 GiB`.
- GPU1 stayed idle, confirming the single-GPU placement worked.
- The failed run's heavyweight split/W&B artifacts were removed, reducing the
  run dir to `544K`; `train.log`, `run_spec.yaml`, and `run_plan.json` were
  preserved for evidence.
- Final sanity check at `2026-06-08T21:02:59Z`: no active Nano processes,
  GPU0/GPU1 both `4 MiB` used.

Decision:

- Single-GPU `gb192/mb192` is not viable for this R33 AR setup on one H100.
- The next feasible throughput probe should reduce single-GPU microbatch
  substantially, for example `gb192/mb96` with two microbatches on one GPU, or
  keep the known-good two-GPU `gb192/mb16` path for the 100k stability run.

### 2026-06-08 - R33 AR Single-GPU Halving Probes Hit Optimizer-State Floor

Follow-up request:

- After the `mb192` single-GPU OOM, the user requested halving microbatch size
  until the run no longer OOMed.

Probes:

| Run | Effective batch | microbatch | GPUs | num_microbatches | failure point | PyTorch allocated at OOM |
|---|---:|---:|---:|---:|---|---:|
| `nano-ar-r33-100k-lr1e5-cosine-gb192-mb96-gpu1` | 192 | 96 | 1 | 2 | `optimizer.step` Adam state init | `137.30 GiB` |
| `nano-ar-r33-100k-lr1e5-cosine-gb192-mb48-gpu1` | 192 | 48 | 1 | 4 | `optimizer.step` Adam state init | `137.11 GiB` |
| `nano-ar-r33-100k-lr1e5-cosine-gb192-mb24-gpu1` | 192 | 24 | 1 | 8 | `optimizer.step` Adam state init | `136.64 GiB` |
| `nano-ar-r33-100k-lr1e5-cosine-gb192-mb12-gpu1` | 192 | 12 | 1 | 16 | `optimizer.step` Adam state init | `137.50 GiB` |

Outcome:

- Each run placed correctly on GPU0 only; GPU1 stayed idle.
- Each run completed its configured microbatch loop before failing at the first
  Adam optimizer-state allocation.
- The OOM signature was stable: tiny `20 MiB` allocations failed only after the
  process reached about `139.78-139.80 GiB` process memory.
- Heavy failed-run artifacts (`splits/`, W&B dirs) were removed after each
  failure; logs/specs were preserved. Final sanity check after `mb12`: no active
  matching processes, GPU0/GPU1 both `4 MiB`.

Decision:

- Further halving microbatch size is not expected to fix this single-GPU AR
  path. The bottleneck is Adam optimizer state / single-rank FSDP memory, not
  activation memory from the microbatch.
- Resume the R33 AR stability path on two GPUs, preferably the known-good
  `gb192/mb16` setting, unless we intentionally change optimizer/sharding or
  introduce CPU/NVMe offload.

### 2026-06-08 - R33 AR Two-GPU Max-Microbatch Probe Clears Optimizer Gate

Follow-up request:

- After reverting to the known-good two-GPU path, the user requested the same
  halving strategy from the largest possible microbatch.

Setup:

- `global_batch_size=192`
- `num_gpus=2`
- Max legal microbatch is `96`, not `192`, because per-rank microbatch is
  multiplied across two ranks.
- Config: `configs/nano_ar/hpo/r33_100k_lr1e5_cosine_gb192_mb96_2gpu.yaml`
- Run: `nano-ar-r33-100k-lr1e5-cosine-gb192-mb96-2gpu`

Early result:

- Launched via queue `--once` at `2026-06-08T22:40:37Z`.
- Confirmed train command uses two GPUs, `global_batch_size=192`,
  `micro_batch_size=96`, and `num_microbatches=1`.
- The run cleared first backward plus Adam optimizer-state initialization with
  no OOM and logged steps `0-13` by `2026-06-08T22:44:25Z`.
- Step losses during LR warmup:
  - step 0: `1.3468`
  - step 1: `1.1237`
  - step 4: `1.1336`
  - step 13: `1.0902`
- Early memory after optimizer initialization was about `99 GiB/GPU`; timing
  logs showed allocator-reserved memory around `94-95 GiB` and max allocated
  around `82 GiB`.

Decision:

- Do not halve further unless this run later OOMs.
- Leave the `mb96` run active to reach `iter_0000467`, then run the bounded
  `512/512` teacher/control AR eval before using this setting for hero-scale
  promotion.

Completed eval:

- Training reached `iter_0000467`; final logged training step was step 466 at
  `2026-06-08T23:24:47Z`.
- Queue completed at `2026-06-08T23:38:29Z` after bounded `512/512` eval.
- Eval report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr1e5-cosine-gb192-mb96-2gpu/eval_iter_0000467_v512_t512_winrates_report.json`

| Split | teacher NMSE | source_raw NMSE | source_context NMSE | teacher_shuffled NMSE |
|---|---:|---:|---:|---:|
| validation | `0.441161` | `0.072254` | `0.556988` | `0.884855` |
| test | `0.419937` | `0.071656` | `0.540735` | `0.868807` |

Interpretation:

- Throughput result is strong: the 100k run finished training in about 44 min,
  with one microbatch per step.
- Quality result is not a promotion candidate. The `lr=1e-5` checkpoint is much
  worse than the prior R33 `100k lr=2e-5` checkpoint, which had validation/test
  teacher NMSE `0.334868 / 0.323239`.
- Use `mb96` as the throughput setting for follow-up R33 AR HPO, but return to
  `lr=2e-5` or a nearby schedule for quality.

### 2026-06-09 - R33 AR Follow-Up 100k HPO: Higher LR Then Short Warmup

Plan:

- Run `lr=3e-5`, `gb192/mb96` first to test whether the best `lr=2e-5`
  quality trend improves with a modestly higher LR while keeping the proven
  fast two-GPU microbatch.
- Keep `lr=2e-5`, `gb192/mb96`, `warmup=25` queued second to test whether the
  original `lr=2e-5` quality improves with less warmup over the 467-step run.
- Do not launch hero/full275k until the bounded 100k evals justify promotion.

Launch status:

- `nano-ar-r33-100k-lr3e5-cosine-gb192-mb96` launched at
  `2026-06-09T02:07:40Z`.
- Confirmed train command uses two GPUs, `global_batch_size=192`,
  `micro_batch_size=96`, `lr=3e-5`, `min_lr=3e-6`, and
  `num_microbatches=1`.
- First optimizer gate cleared without OOM; steady-state memory is about
  `98-99 GiB/GPU`.

Completed eval:

- Queue completed at `2026-06-09T03:08:32Z`.
- Eval report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-scaling/nano-ar-r33-100k-lr3e5-cosine-gb192-mb96/eval_iter_0000467_v512_t512_winrates_report.json`

| Split | teacher NMSE | source_raw NMSE |
|---|---:|---:|
| validation | `0.322812` | `0.081054` |
| test | `0.313741` | `0.075954` |

Interpretation:

- `lr=3e-5` is the new best R33 100k AR checkpoint so far, improving the prior
  `lr=2e-5` result (`0.334868 / 0.323239`) by about `0.012` validation and
  `0.009` test teacher NMSE.
- It is still above the desired `0.25-0.30` hero target, so wait for the
  short-warmup `lr=2e-5` probe and consider one more nearby LR/schedule probe
  before full275k promotion.

Follow-up launch:

- The planned handoff watcher failed to launch the second probe because it was
  waiting on a stale/reused PID. The watcher was killed manually.
- `nano-ar-r33-100k-lr2e5-cosine-warmup25-gb192-mb96` launched at
  `2026-06-09T16:12:55Z` with `lr=2e-5`, `warmup=25`, `gb192/mb96`.

Additional queued probes:

- `nano-ar-r33-100k-lr3e5-cosine-warmup25-gb192-mb96`
  - tests whether the current best `lr=3e-5` improves with `warmup=25`.
- `nano-ar-r33-100k-lr4e5-cosine-gb192-mb96`
  - tests whether the current best `lr=3e-5` is under-shooting before full275k.
- Decision gate: after these bounded `512/512` evals, choose the R33 full275k
  AR hero params rather than adding a broad grid.

### 2026-06-09 - R33 AR 100k Final Pre-Hero HPO Results

Purpose:

Finish the small R33 AR search before choosing full275k hero parameters. All
results below use the verified R33 prefix/fullscan-backed 100k dataset,
`gb192/mb96`, two GPUs, and bounded `512/512` evals with teacher/control
baselines.

Completed evals:

| Run | validation teacher NMSE | test teacher NMSE | validation source_raw NMSE | test source_raw NMSE |
|---|---:|---:|---:|---:|
| `lr=2e-5`, `warmup=25` | `0.348768` | `0.336298` | `0.077899` | `0.073737` |
| `lr=3e-5`, default warmup | `0.322812` | `0.313741` | `0.081054` | `0.075954` |
| `lr=3e-5`, `warmup=25` | `0.321038` | `0.312018` | `0.084906` | `0.078302` |
| `lr=4e-5`, default warmup | `0.309019` | `0.301218` | `0.086966` | `0.081296` |
| `lr=5e-5`, default warmup | `0.301175` | `0.292956` | `0.093239` | `0.086504` |
| `lr=5e-5`, `warmup=25` | `0.300924` | `0.292944` | `0.094529` | `0.088689` |

Recovery note:

- `nano-ar-r33-100k-lr4e5-cosine-gb192-mb96` reached checkpoint
  `iter_0000467` before the RunAI pod eviction caused by node ephemeral-storage
  pressure.
- After the `train` workspace was redeployed with the PVC attached, the bounded
  eval was run from the saved checkpoint and completed at
  `2026-06-09T20:52:30Z`.
- To reduce Longhorn pressure, evaluated non-candidate R33 100k checkpoint
  trees were removed. W&B offline logs, train logs, eval reports, run specs,
  the `lr=3e-5 warmup=25` checkpoint, and the `lr=4e-5` checkpoint were
  retained.

Interpretation:

- Short warmup helped slightly at `lr=3e-5`: teacher NMSE improved from
  `0.322812 / 0.313741` to `0.321038 / 0.312018`.
- `lr=2e-5 warmup=25` was stable but not competitive; it was worse than the
  earlier `lr=2e-5` eval (`0.334868 / 0.323239`).
- `lr=4e-5` is now the best R33 100k AR result, reaching validation/test
  teacher NMSE `0.309019 / 0.301218`. This is close to, but still just above,
  the desired `0.25-0.30` heldout target.
- The final `lr=5e-5` probes improved again. Default warmup reached
  `0.301175 / 0.292956`; `warmup=25` narrowly won by teacher NMSE at
  `0.300924 / 0.292944`.
- The selected full275k R33 AR hero candidate is `lr=5e-5`, `warmup=25`,
  `gb192/mb96`, two GPUs. The difference versus `warmup=50` is tiny, but both
  are stable and now in/near the target band. Do not call the NLA good until the
  AV-generated-text to AR reconstruction gate beats the mature R27 baseline.

### 2026-06-10 - R33 AR Full275k Hero Result And HPO Closure

Purpose:

Record the R33 AR hyperparameter-search history and close the AR hero selection
loop after the full275k run completed. Evidence was checked against local
logbook/queue entries plus the RunAI eval report, queue JSONL, and train log.

Layer-selection context:

- R33 became the main scaling target after the scale-matched AR probes. The
  R33 `20k` probe reached validation/test teacher NMSE `0.381983 / 0.388301`
  with source_raw NMSE `0.071066 / 0.076216`.
- R34 was not the immediate AR target despite slightly better AV `20k`: its
  matched AR `20k` probe was worse, with validation/test teacher NMSE
  `0.490728 / 0.501399` and source_raw NMSE `0.112821 / 0.119030`.
- Mature tuned R27 AR remains the fallback, but its teacher NMSE had plateaued
  around `0.441 / 0.437`, so R33 was the better scaling candidate.

R33 AR `100k` HPO summary:

| Run | validation teacher NMSE | test teacher NMSE |
|---|---:|---:|
| `lr=1e-5`, `mb96` | `0.441161` | `0.419937` |
| `lr=2e-5`, `mb8` | `0.334868` | `0.323239` |
| `lr=2e-5`, `warmup=25`, `mb96` | `0.348768` | `0.336298` |
| `lr=3e-5`, `mb96` | `0.322812` | `0.313741` |
| `lr=3e-5`, `warmup=25`, `mb96` | `0.321038` | `0.312018` |
| `lr=4e-5`, `mb96` | `0.309019` | `0.301218` |
| `lr=5e-5`, `warmup=50`, `mb96` | `0.301175` | `0.292956` |
| `lr=5e-5`, `warmup=25`, `mb96` | `0.300924` | `0.292944` |

Final R33 AR full275k hero run:

- Run id/name:
  `nano-ar-r33-full275k-lr5e5-cosine-warmup25-gb192-mb96`
- Queue completion: `2026-06-10T01:42:35Z`
- Checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-hero/nano-ar-r33-full275k-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001291`
- Eval report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-hero/nano-ar-r33-full275k-lr5e5-cosine-warmup25-gb192-mb96/eval_iter_0001291_v512_t512_winrates_report.json`
- Final train loss/FVE: `0.27228911717732746 / 0.5134132305781046`
- Tail-20 train loss/FVE: `0.27059254546960193 / 0.5164450248082479`

Bounded AR eval, 512 validation / 512 test:

| Split | teacher NMSE | source_raw NMSE | source_context NMSE | mean NMSE |
|---|---:|---:|---:|---:|
| validation | `0.2775649428367615` | `0.09694842249155045` | `0.3042636215686798` | `0.6954619288444519` |
| test | `0.2766648828983307` | `0.09156802296638489` | `0.283364474773407` | `0.6722851991653442` |

Conclusion and next steps:

- AR hero candidate is `lr=5e-5`, cosine, `warmup=25`, `gb192/mb96` on two
  GPUs. No further AR HPO is needed before moving to AV HPO/hero and the
  AV-generated-text -> AR round-trip gate.
- This is a strong AR proxy result, not proof of full NLA. The full milestone
  still depends on AV text quality and the round-trip gate beating the mature
  R27 fallback.
- W&B for this run is offline under the run directory. Keep follow-up handling
  storage-conscious: preserve compact eval reports, train logs, run specs, and
  W&B offline logs, but avoid retaining temporary/large checkpoint artifacts
  beyond the needed hero candidate.

### 2026-06-10 - Post-Audit Correction: Packed-Boundary Contamination, LR Schedule Bug, And Cleanup

Purpose:

Re-label the R33/R27 AR/AV history after the Nano/Nemotron-H packed-boundary
contamination bug and Miles LR-schedule bug were confirmed. The goal is to keep
the useful scouting evidence while preventing future agents from treating
pre-fix checkpoints as clean hero proof.

Confirmed corrections:

- Nano/Nemotron-H remote code did not correctly isolate samples inside Miles
  `thd` packed microbatches. Attention/Mamba state could cross sample
  boundaries, so AR/AV checkpoints trained before the remediation are
  contaminated-training artifacts.
- The heldout eval reports remain meaningful scouting evidence because they are
  real heldout/control evals, but the training path was noisy and mismatched.
- Runs labeled `cosine` before the LR-schedule remediation should be read as
  requested configs, not proof of actual LR decay, unless a run-local LR canary
  or final-LR trace exists.
- The prior R33 AR full275k run remains strong directional signal
  (`0.277565 / 0.276665` validation/test teacher NMSE), but it is not a clean
  AR hero checkpoint and must be reproduced after the fixes.
- The R27 AV-SFT `100k` hero remains useful as a mature fallback/scouting
  baseline, but it should also be labeled pre-fix and not treated as clean
  AV+AR proof.

RunAI cleanup and sync:

- RunAI workspace `train` was idle before cleanup: no active Nano train/eval
  process and GPUs idle.
- Lightweight RunAI evidence archive was created and copied locally:
  `artifacts/runai_sync/20260610T234644Z/runai_light_artifacts_20260610T234644Z.tar.gz`
- Local archive SHA-256:
  `7bb95e0f6ab98c3c6269f7217459af6f4bda14f1ee708356ae733fe273719db3`
- S3 evidence prefix:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/sync_exports/20260610T234644Z/`
- Cleanup manifest:
  `artifacts/runai_sync/20260610T234644Z/runai_cleanup_manifest_20260610T234644Z.json`
- Additional cleanup manifest for reproducible pre-fix run-specific splits:
  `artifacts/runai_sync/20260610T234644Z/runai_cleanup_manifest_20260611T0005Z.json`
- Deleted contaminated checkpoint tree:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-hero/nano-ar-r33-full275k-lr5e5-cosine-warmup25-gb192-mb96/checkpoints`
- Deleted bytes from first cleanup: `76,921,456,907`.
- Deleted bytes from second cleanup: `11,303,016,711`.
- `/workspace/interp` improved to about `146G` used and `863G` free after
  checkpoint cleanup, reproducible split cleanup, and stale remote code-copy
  removal.

Code sync:

- Local source archive:
  `artifacts/runai_sync/20260610T234644Z/source_code_20260610T234644Z.tgz`
- Source archive SHA-256: see the adjacent `.sha256` sidecar. Do not embed the
  hash here because this document is part of the source archive.
- S3 source prefix:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/code_sync/20260610T234644Z/`
- RunAI stable code symlink now points to:
  `/workspace/interp/code/nano30b-nla-pilot-sync-20260610T234644Z`
- RunAI sync marker:
  `/workspace/interp/code/nano30b-nla-pilot-current/.runai_code_sync_marker`

Current clean next steps:

1. Run a post-fix AR confirmation near the prior best family:
   `lr=5e-5`, `warmup=25`, `gb192/mb96`, two GPUs.
2. Run post-fix AV confirmation near the best pre-fix smoke signal (`1e-4`,
   with a nearby fine probe if needed).
3. Require checkpoint-local Nemotron-H patch reports and packed-vs-padded
   agreement/preflight evidence before calling any rerun clean.
4. Only after clean AR and AV checkpoints exist, run the
   `h -> AV-generated explanation -> AR h_hat` round-trip gate against the
   mature R27 fallback.

### 2026-06-10 - Step-0/Step-1 Audit Remediation Implemented Locally

Purpose:

Implement the numbered audit-plan fixes that must land before any clean R33
hero rerun. No training was launched.

Implemented code/config changes:

- Remote-code patching is now explicit-only: critic load/save and critic-init
  preparation copy checkpoint-local remote code verbatim; the Miles actor patch
  no longer calls the Nemotron patcher at actor init; `nano_av_runner.py` no
  longer exports `NLA_PATCH_NEMOTRON_REMOTE_CODE=1`.
- Miles patch files were structurally repaired and gated by
  `scripts/check_miles_patches.py`; use
  `python scripts/check_miles_patches.py --miles-root /workspace/interp/code/miles-051cd15`
  on RunAI for the pinned-checkout apply test.
- Round-trip gate logic now requires a positive control margin, requires all
  controls, strips generated explanation tags before AR scoring, reports
  generation parse/closure stats, and checks row identity plus rowwise R27
  baseline wins.
- AR/AV queues share terminal status constants, AV has `--reset-active` parity,
  and both queues hard-fail non-constant schedules whose LR canary is missing,
  flat, or invalid.
- AV queue temporary HF conversion cleanup now runs in a `finally` block when
  cleanup is enabled.
- AR/AV verifiers now report duplicate content groups and fail on content-hash
  cross-split overlap.
- Added `scripts/nano_dedup_teacher_keys.py` for deduplicating teacher-backed
  rows by first-300-token `token_ids_prefix` hash.

Local dedup sanity result from the preserved 275,396-row R27 AR-SFT source:

- Source table:
  `runs/introspection/ar-r27-r30-fullscan-20260528T234403Z/handoff/R_27/ar_sft.parquet`
- Source docs: `27,647`
- Kept docs: `13,349`
- Output rows: `132,996`
- Dropped rows: `142,400`
- Duplicate groups: `4,132`
- Empty extracted `api_explanation`: `0`
- Expected clean RunAI root:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_dedup_fullscan132996`

New clean configs/queues:

- `configs/nano_ar/hpo/r33_dedup_smoke_20k_lr2e5_cosine_warmup20_gb192_mb96.yaml`
- `configs/nano_ar/hpo/r33_dedup_full_lr5e5_cosine_warmup25_gb192_mb96.yaml`
- `configs/nano_ar/hpo/r33_dedup_clean_queue.yaml` (blocked, no pending work)
- `configs/nano_av/hpo/r33_dedup_av_20k_lr1e4_cosine_warmup5_gb192_mb1_seq1152_dyn1152.yaml`
- `configs/nano_av/hpo/r33_dedup_av_20k_lr5e5_cosine_warmup5_gb192_mb1_seq1152_dyn1152.yaml`
- `configs/nano_av/hpo/r33_dedup_av_full_lr1e4_cosine_warmup5_gb192_mb1_seq1152_dyn1152.yaml`
- `configs/nano_av/hpo/r33_dedup_clean_queue.yaml` (blocked, no pending work)

Clean full-dedup split estimate at seed 42:

- Train rows: `119,698`; padded train rows: `119,808`
- Validation/test rows: `6,641 / 6,657`
- Full clean AR/AV epoch at global batch `192`: `624` optimizer steps.

### 2026-06-11 - High-Value Audit Follow-Up Implemented Locally

Purpose:

Implement the remaining high-value/likely-worth audit suggestions before any
new clean R33 launch. No RunAI training was launched.

Implemented:

- `scripts/nano_av_runner.py` now supports opt-in deterministic materialized
  split caching via `dataset.cache_materialized_splits` and `split_cache_dir`.
  Cache keys include source file signature, sidecar signature, row limit, split
  fractions, seed, and final-batch padding policy. Cached split manifests are
  rewritten to the current run directory before launch.
- `dataset.verify_materialized_splits: true` now runs the materialized split
  content verifier during `prepare_run`, writes `split_content_verify.json`, and
  fails before training if doc/content leakage crosses train/validation/test.
- The FSDP DCP -> HF converter now accepts `--torch-dtype bfloat16`; AV queue
  items/defaults can request it through `converted_hf_dtype`.
- Clean R33 dedup AR/AV configs opt into split caching and split verification.
  The clean R33 AV queue defaults temporary HF conversion to bf16.

Local verification:

- `python -m unittest` targeted runner cache/verification tests: pass.
- `python -m unittest tests.test_nano_av_probe_queue`: pass.
- Dependency-light verifier/config tests: pass.
- `python scripts/check_miles_patches.py`: hunk failures `[]`.

### 2026-06-11 - R27 Round-Trip Plumbing Smoke and HPO Metric Path

Purpose:

Validate research-direction items 1 and 2 before new clean R33 training:
restore the mature R27 AV/AR pair far enough to run the
`h -> AV-generated explanation -> AR h_hat` evaluator, and record round-trip
NMSE as an `av_roundtrip` HPO objective. No training or RL was launched.

Implementation/debug notes:

- Added `av_roundtrip` support to `scripts/nano_ar_hpo_study.py`, including
  `objective_roundtrip_nmse`, round-trip report parsing, and Optuna export.
- Added `scripts/nano_roundtrip_eval_config.py` and optional round-trip
  scoring hooks in `scripts/nano_av_probe_queue.py`.
- Fixed restored Nemotron-H remote-code eval compatibility for AR scoring:
  CPU-token/GPU-vector injection indexing, checkpoint-local remote-code patch
  refresh before tokenizer/model load, stale HF module cache removal, batched
  Mamba `seq_idx` shape preservation, stale partial-patch deduplication, and
  int32 `seq_idx` dtype for Triton fused Mamba kernels.
- Local focused tests passed:
  `tests.test_nano_audit_remediation`,
  `tests.test_nano_ar_eval_metrics`,
  `tests.test_eval_nano_ar_report_extensions`,
  `tests.test_nano_ar_hpo_study`,
  `tests.test_nano_av_probe_queue`, and
  `tests.test_nano_roundtrip_eval_config`.
- RunAI focused tests passed in `/workspace/interp/.venv`: `17 passed` for the
  remote-code/eval regression shard after the final `seq_idx` fix.

RunAI artifacts:

- Restored R27 AV checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/checkpoints/r27-av-ar-best/av_hf_iter_0000467`
- Restored R27 AR checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/checkpoints/r27-av-ar-best/ar_hf_iter_0000256`
- Deterministic R27 round-trip splits:
  `/workspace/interp/outputs/nano30b-nla-pilot/roundtrip/r27_baseline/splits`
  with train/validation/test rows `89,618 / 4,978 / 4,974`.
- Smoke report:
  `/workspace/interp/outputs/nano30b-nla-pilot/roundtrip/r27_baseline/r27_roundtrip_v1_t1_real_report.json`
- HPO record:
  `/workspace/interp/outputs/nano30b-nla-pilot/roundtrip/r27_baseline/r27_roundtrip_hpo_smoke.jsonl`
- Optuna export:
  `/workspace/interp/outputs/nano30b-nla-pilot/roundtrip/r27_baseline/r27_roundtrip_hpo_smoke_optuna.json`
- S3 evidence prefix:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/sync_exports/roundtrip/r27_baseline_smoke_20260611T0421Z/`

Smoke result:

- Scope: real-only, validation/test `1/1`, reused already generated JSONL,
  `max_new_tokens=24`; this validates evaluator/HPO plumbing, not AV quality.
- Validation teacher/AV-real/mean NMSE:
  `0.0000699769 / 0.0002744264 / 0.0002337020`.
- Test teacher/AV-real/mean NMSE:
  `0.0001513617 / 0.0003906921 / 0.0002917381`.
- Recorded HPO objective:
  `objective_roundtrip_nmse = 0.00033255922608077526`.
- Gate status: `passed=false`; controls were intentionally incomplete for this
  plumbing smoke.
- Important caveat: generated summaries did not close/parse in this tiny
  `24`-token smoke (`empty_fraction=1.0` on validation and test), so the
  result must not be interpreted as an R27 round-trip quality baseline.

Operational note:

- S3 upload from RunAI works with `scripts/nano_s3.py` and the PVC credential
  files. This Mac session did not have the helper credential files or an AWS CLI
  on `PATH`, so Mac-to-S3 upload was not available without additional local
  credential setup.

### 2026-06-11 - Clean R33 Strict-Dedup Dataset, Packed AR Failure, and mb1 Fallback

Purpose:

Move from pre-fix R33 scouting to clean R33 evidence for the AV+AR round-trip
milestone. This entry records the strict content-dedup dataset, the packed AR
guard failure, and the clean no-packed AR fallback now running.

Clean R33 dataset proof:

- Source teacher table:
  `/workspace/interp/outputs/nano30b-nla-pilot/ar-r27-r30-fullscan-20260528T234403Z/R_27/ar_sft.parquet`
- Final clean root:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_dedup_content_fullscan`
- Dedup mode: verifier-equivalent row content connected components over
  `token_ids_prefix` and `detokenized_text_truncated`.
- Source rows/docs: `275,396 / 27,647`.
- Kept rows/docs: `56,351 / 5,657`.
- Dropped rows/docs: `219,045 / 21,990`.
- Empty extracted `api_explanation`: `0`.
- AR verifier:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_dedup_content_fullscan/verify_ar_R33_r33_prefix_dedup_content_fullscan.json`
- AV verifier:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_dedup_content_fullscan/verify_av_R33_r33_prefix_dedup_content_fullscan.json`
- Verifier status: row count `56,351`, `d_model=2688`, nonfinite
  activations `0`, empty explanations `0`, content cross-split overlap `0`
  for both `80/10/10` and `90/5/5`.
- Interpretation: this is clean for smoke/gate work, but it is not a 275k
  hero-size dataset. Future hero-scale recovery should split by duplicate
  content component instead of dropping all duplicate rows.

R27 round-trip baseline:

- Report:
  `/workspace/interp/outputs/nano30b-nla-pilot/roundtrip/r27_baseline/r27_roundtrip_v64_t64_full_controls_prefix256_report.json`
- Generated JSONL:
  `/workspace/interp/outputs/nano30b-nla-pilot/roundtrip/r27_baseline/r27_roundtrip_v64_t64_full_controls_prefix256_generated.jsonl`
- Gate passed with `control_margin=5e-5` and
  `min_control_win_fraction=0.9`.
- Validation teacher/AV-real NMSE:
  `0.000156636 / 0.000174863`.
- Test teacher/AV-real NMSE:
  `0.000143537 / 0.000173753`.
- AV-real beat all generated-text controls with rowwise win fraction
  `>=0.96875` on validation and `1.0` on test.

Clean AR smoke attempts:

- Packed attempt:
  `nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb96`
  failed at `2026-06-11T11:00:34Z`.
- Failure mode: live step-0 reward-path vs training-path MSE equivalence guard
  failed with `17.9%` max divergence under packed `mb96` real rollout data.
- Conclusion: packed AR training remains unsafe for clean Nano/Nemotron-H
  reruns; do not bypass this guard for promotion runs.
- Clean fallback:
  `nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb1-2gpu`.
- Config:
  `configs/nano_ar/hpo/r33_dedup_smoke_20k_lr2e5_cosine_warmup20_gb192_mb1_2gpu.yaml`
- Status at this log entry: training on RunAI.
- First observed step:
  `train/loss=1.2053946`, `train/fve_nrm=-1.1587539`, LR `1e-6`,
  `perf/step_time=387.6s`.
- ETA from first step: about `10` hours for the full 96-step 20k smoke plus
  bounded eval.

Code/config notes:

- Full strict-dedup AR/AV configs were reclassified from
  `complete-performance` to `tuning-probe` because the clean strict-dedup set
  is `56,351` rows, below the complete-performance threshold and not a hero
  corpus.
- Local lightweight tests passed after config changes:
  `57 passed, 1 deselected`.
- RunAI focused tests passed after sync:
  `76 passed` for the dedup/generation/round-trip/queue/spec shard and
  `34 passed` for the AR queue/spec shard.

### 2026-06-11 - R33 AV Round-Trip Smoke Staged Behind Clean AR Checkpoint

Purpose:

Prepare the first post-fix R33 AV smoke to use round-trip NMSE as the HPO
metric, while keeping all AV jobs blocked until a clean R33 AR checkpoint and
bounded eval are available.

Queue/config changes:

- Updated `configs/nano_av/hpo/r33_dedup_clean_queue.yaml`.
- The first AV smoke,
  `r33-dedup-av-20k-lr1e4-warmup5-gb192-mb1-dyn1152`, remains
  `status: blocked` but is now labeled `study_task: av_roundtrip`.
- Round-trip dependency:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-smoke/nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb1-2gpu/checkpoints/iter_0000096`
- Baseline report:
  `/workspace/interp/outputs/nano30b-nla-pilot/roundtrip/r27_baseline/r27_roundtrip_v64_t64_full_controls_prefix256_report.json`
- Protocol: validation/test `64/64`, controls `real`, `shuffled`, `zero`,
  `mean`, and `none`, `max_new_tokens=256`, cached generation, closure gate
  `0.8`, usable gate `0.95`, control margin `5e-5`, and rowwise control-win
  threshold `0.9`.

Verification:

- Local regression: `/tmp/nla_impl_py/bin/python -m pytest
  tests/test_nano_av_probe_queue.py::NanoAVProbeQueueTests::test_checked_in_r33_dedup_queue_has_roundtrip_smoke_blocked -q`
  passed.
- Local queue/config shard:
  `/tmp/nla_impl_py/bin/python -m pytest tests/test_nano_av_probe_queue.py
  tests/test_nano_roundtrip_eval_config.py -q` passed with `12 passed`.
- RunAI code sync used a small checksummed source archive because this Mac
  shell still has no `aws` CLI on `PATH`.
- RunAI queue/config shard passed in `/workspace/interp/.venv` with
  `12 passed`.
- RunAI AV queue status after sync: `blocked: 3`, `pending: 0`,
  `training: 0`, `failed: 0`.
- Added `scripts/nano_queue_gate.py` as a generic evidence gate for promoting
  blocked queue items after a dependency queue item reaches `complete` and
  required artifact paths exist.
- Local gate tests plus surrounding queue/config shard passed:
  `/tmp/nla_impl_py/bin/python -m pytest tests/test_nano_queue_gate.py -q`
  with `3 passed`, and `/tmp/nla_impl_py/bin/python -m pytest
  tests/test_nano_av_probe_queue.py tests/test_nano_ar_hpo_queue.py
  tests/test_nano_roundtrip_eval_config.py -q` with `22 passed`.
- RunAI gate/queue/config shard passed in `/workspace/interp/.venv` with
  `25 passed`.
- RunAI dry-run gate against the live queues correctly returned
  `ready=false`, `changed=false` because the AR dependency was still
  `training`, the expected checkpoint path was absent, and `eval_report` was
  not yet set.

Concurrent AR status:

- Clean mb1 AR smoke was still training at `2026-06-11T11:44:15Z`.
- Latest observed completed optimizer step was step `5`:
  `train/loss=0.9060241`, `train/fve_nrm=-0.6226082`, LR `6e-6`, step time
  `307.0s`.
- GPU/process health: both H200 actor processes were present, each using about
  `96.7 GiB` VRAM. `/workspace/interp` had about `767G` free.

Follow-up status at `2026-06-11T11:49:59Z`:

- Clean mb1 AR smoke was still training.
- Latest observed completed optimizer step was step `6`:
  `train/loss=0.8779152`, `train/fve_nrm=-0.5722676`, LR `7e-6`, step time
  `277.5s`.
- The dependency gate still correctly refused to unblock the first R33 AV
  smoke because the AR item was still `training`, the expected checkpoint path
  did not exist yet, and the AR queue item had no `eval_report` field.

Round-trip gate hardening while waiting for AR:

- Added generated-record coverage validation to
  `scripts/eval_nano_av_ar_roundtrip_gate.py`.
- The round-trip scorer now rejects missing expected validation/test rows,
  duplicate or unexpected rows, and missing requested controls before loading
  the AR checkpoint for scoring. This prevents partial streamed/generated JSONL
  artifacts from accidentally producing a plausible-looking gate report.
- Local focused shard passed:
  `/tmp/nla_impl_py/bin/python -m pytest tests/test_nano_av_generation.py
  tests/test_nano_av_ar_roundtrip_gate.py tests/test_nano_roundtrip_eval_config.py
  tests/test_nano_roundtrip_queue.py tests/test_nano_av_probe_queue.py
  tests/test_nano_ar_hpo_study.py tests/test_nano_queue_gate.py -q`
  with `47 passed`.
- RunAI focused shard passed in `/workspace/interp/.venv` with `47 passed`.
- Status at `2026-06-11T11:55:11Z`: clean mb1 AR smoke was still training;
  latest observed optimizer step was step `7` with
  `train/loss=0.8650217`, `train/fve_nrm=-0.5491768`, LR `8e-6`.
- The dependency gate still returned `ready=false` because the AR queue item
  remained `training`, `iter_0000096` did not exist yet, and `eval_report` was
  still unset.

R27 full-control baseline config alignment:

- Updated the checked-in R27 full-control round-trip baseline configs to enable
  `eval.resume_generated: true`:
  `configs/nano_roundtrip/r27_baseline_64_full_controls_prefix256.yaml` and
  `configs/nano_roundtrip/r27_baseline_256_full_controls_prefix256.yaml`.
- Added checked-in config coverage to
  `tests/test_nano_roundtrip_eval_config.py`, asserting that both baselines
  render commands with cached generation, two generation workers,
  `--stream-generated`, and `--resume-generated`.
- Local focused shard passed with `49 passed`; RunAI focused shard passed in
  `/workspace/interp/.venv` with `49 passed`.
- RunAI status after the remote test: clean mb1 AR smoke was still training;
  latest observed optimizer step remained step `9`, and the first R33 AV smoke
  remained blocked.

R27 round-trip queue state correction:

- Verified on RunAI that
  `/workspace/interp/outputs/nano30b-nla-pilot/roundtrip/r27_baseline/r27_roundtrip_v64_t64_full_controls_prefix256_report.json`
  exists and has `gate.passed=true`.
- The `64/64` report has validation/test AV-real NMSE
  `0.0001747658 / 0.0001737549` and real-control closed parse fraction
  `1.0 / 1.0`.
- The `256/256` full-control report is still missing.
- Updated `configs/nano_roundtrip/r27_baseline_queue.yaml`: the `64/64`
  full-control item is now `complete` with report/generated JSONL fields and
  `gate_passed: true`; the `256/256` full-control item is now `pending`.
- Local focused shard passed with `50 passed`; RunAI focused shard passed in
  `/workspace/interp/.venv` with `50 passed`.
- Status at `2026-06-11T12:08:30Z`: clean mb1 AR smoke was still training;
  latest observed optimizer step was step `10` with
  `train/loss=0.8147332`, `train/fve_nrm=-0.4591142`, LR `1.1e-5`.
- The first R33 AV smoke gate still returned `ready=false` because the AR queue
  item remained `training`, `iter_0000096` did not exist yet, and `eval_report`
  was still unset.

R27 round-trip launch guard:

- Added a config-driven active-process launch guard to
  `scripts/nano_roundtrip_queue.py`.
- `configs/nano_roundtrip/r27_baseline_queue.yaml` now blocks launches when
  process lines match Nano AR/AV queues, Miles training, eval, or FSDP
  conversion patterns.
- Local focused shard passed with `51 passed`; RunAI focused shard passed in
  `/workspace/interp/.venv` with `51 passed`.
- RunAI `scripts/nano_roundtrip_queue.py run-once
  configs/nano_roundtrip/r27_baseline_queue.yaml` returned
  `blocked_active_process` while the clean R33 AR trainer was active. The queue
  remained `64/64 complete`, `256/256 pending`; no R27 256/256 run was launched.

Round-trip generation resume support:

- Added `--resume-generated` to
  `scripts/eval_nano_av_ar_roundtrip_gate.py`.
- Complete existing generated records for expected rows/controls are reused;
  incomplete or out-of-scope records are ignored and the JSONL is rewritten
  into canonical eval order before appending new generations.
- Multi-worker generation preserves shard JSONL files in resume mode and passes
  `--resume-generated` into each worker.
- `scripts/nano_av_probe_queue.py` now exposes `resume_generated` in
  round-trip queue configs.
- The first staged clean R33 AV smoke now sets
  `roundtrip.resume_generated: true`.
- Local focused shard passed with `48 passed`; RunAI focused shard passed in
  `/workspace/interp/.venv` with `48 passed`.
- Status at `2026-06-11T12:00:51Z`: clean mb1 AR smoke was still training;
  latest observed optimizer step was step `8` with
  `train/loss=0.8756660`, `train/fve_nrm=-0.5682395`, LR `9e-6`.
- The dependency gate still returned `ready=false` because the AR queue item
  remained `training`, `iter_0000096` did not exist yet, and `eval_report` was
  still unset.

Standalone round-trip config renderer update:

- `scripts/nano_roundtrip_eval_config.py` now passes
  `eval.resume_generated` into the rendered
  `scripts/eval_nano_av_ar_roundtrip_gate.py` command.
- This keeps standalone/config-driven R27 and R33 round-trip evals on the same
  resumable generation path as the AV probe queue.
- Local focused shard passed with `48 passed`; RunAI focused shard passed in
  `/workspace/interp/.venv` with `48 passed`.
- Status at `2026-06-11T12:03:24Z`: clean mb1 AR smoke was still training;
  latest observed optimizer step was step `9` with
  `train/loss=0.8358911`, `train/fve_nrm=-0.4970062`, LR `1e-5`.
- The dependency gate still returned `ready=false` because the AR queue item
  remained `training`, `iter_0000096` did not exist yet, and `eval_report` was
  still unset.

R27 256/256 baseline gating and remote sequencer:

- Updated the first clean R33 AV smoke in
  `configs/nano_av/hpo/r33_dedup_clean_queue.yaml` to use the staged R27
  `256/256` full-control round-trip report as its baseline:
  `/workspace/interp/outputs/nano30b-nla-pilot/roundtrip/r27_baseline/r27_roundtrip_v256_t256_full_controls_prefix256_report.json`.
- Synced the queue YAML to RunAI and verified the explicit gate still returned
  `ready=false` because the clean R33 AR smoke was still `training`,
  `checkpoint_dir` and `eval_report` were missing, and the R27 `256/256`
  report was missing.
- Started a remote sequential driver:
  `/workspace/interp/outputs/nano30b-nla-pilot/queue_drivers/r33_roundtrip_av_sequence_20260611T1218Z.sh`
  with log
  `/workspace/interp/outputs/nano30b-nla-pilot/queue_drivers/r33_roundtrip_av_sequence_20260611T1218Z.log`.
- The driver does not launch concurrent work. It waits for active Nano
  train/eval/conversion processes to clear, runs the R27 `256/256` round-trip
  queue item, then runs `scripts/nano_queue_gate.py` requiring the clean AR
  checkpoint/eval and R27 `256/256` report before launching the first R33 AV
  smoke queue item.
- Initial driver log at `2026-06-11T12:17:31Z` showed it waiting behind the
  active clean R33 AR queue/training process.
- Follow-up verification found the focused test shard still expected the old
  R27 `64/64` baseline path. Updated
  `tests/test_nano_av_probe_queue.py` to assert the intended R27 `256/256`
  baseline path.
- Local focused shard passed with `51 passed`; RunAI focused shard passed in
  `/workspace/interp/.venv` with `51 passed`.
- Operational note: when chunking base64 through repeated RunAI exec calls, use
  `while read ... || [ -n "$chunk" ]` so the final unterminated folded line is
  not skipped. A missing final chunk produced a truncated remote test file
  before this was corrected and hash-verified.
- Tightened the R27 round-trip queue launch guard patterns so they match actual
  Python script invocations instead of arbitrary filenames. This prevents sync
  helper commands like `/tmp/test_nano_av_probe_queue.py.b64` from temporarily
  blocking the queue.
- Added regression coverage for that false-positive case in
  `tests/test_nano_roundtrip_queue.py`.
- Local focused shard passed with `52 passed`; RunAI focused shard passed in
  `/workspace/interp/.venv` with `52 passed`.
- Restarted the remote sequencer at `2026-06-11T12:50:02Z` with the same
  stricter process guard. It remained in the waiting state behind the active
  clean R33 AR smoke and did not launch R27/AV work.

AR queue artifact-field handoff fix:

- Verified `scripts/nano_ar_hpo_queue.py` wrote `expected_checkpoint` and
  `eval_report`, but did not write `checkpoint_dir`, while the first AV gate
  had been checking for `checkpoint_dir`.
- Added regression coverage that a completed AR queue item records
  `checkpoint_dir` as the expected checkpoint path.
- Updated `scripts/nano_ar_hpo_queue.py` so future `training`, `eval_running`,
  and `complete` states include `checkpoint_dir` alongside
  `expected_checkpoint`.
- Local focused shard passed with `63 passed`; RunAI focused shard passed in
  `/workspace/interp/.venv` with `63 passed`.
- Because the currently running AR queue process was already loaded before this
  patch, restarted the remote sequencer at `2026-06-11T12:57:04Z` to gate on
  `expected_checkpoint` plus `eval_report`, which is compatible with the live
  process and still verifies the checkpoint path exists.

R27 256/256 round-trip preflight:

- Rendered `configs/nano_roundtrip/r27_baseline_256_full_controls_prefix256.yaml`
  on RunAI and verified it uses the modular cached generation engine with
  `--generation-backend cache`, `--generation-workers 2`, worker devices
  `0 1`, `--stream-generated`, and `--resume-generated`.
- Verified RunAI prerequisites exist: R27 AV HF checkpoint, R27 AR checkpoint,
  and train/validation/test parquet splits. The `256/256` generated JSONL and
  report do not exist yet, so the pending queue item will start clean but
  resumable.
- Verified the stricter round-trip queue launch guard still matches the active
  AR queue/trainer and ignores sync filenames.
- Added regression coverage pinning the checked-in R27 `256/256` config to
  `256/256`, all five controls, cached two-worker generation on devices `0,1`,
  streaming, and resume.
- Local focused shard passed with `64 passed`; RunAI focused shard passed in
  `/workspace/interp/.venv` with `64 passed`.
- Confirmed the clean R33 AR queue default eval report path for the running
  smoke is
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-smoke/nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb1-2gpu/eval_iter_0000096_v512_t512_winrates_report.json`.
  Future status checks should use the queue's `eval_report` field or this
  `_winrates_report.json` naming, not `eval_iter_0000096_v512_t512_report.json`.
- Added `--required-json-bool PATH:DOT.PATH` support to
  `scripts/nano_queue_gate.py` and regression coverage for failed/passing
  `gate.passed` evidence.
- Synced the gate update to RunAI and verified the focused shard passed with
  `65 passed` locally and in `/workspace/interp/.venv`.
- Restarted the remote sequencer at `2026-06-11T13:21:14Z`; the AV unblock
  command now requires `R27_256_REPORT:gate.passed` in addition to the report
  path and AR checkpoint/eval evidence.

Fresh continuation status at `2026-06-11T13:32:37Z`:

- Verified the Mac and RunAI source hashes match for the modular AV generation
  engine, round-trip evaluator, round-trip config renderer, round-trip queue,
  AV queue, AR queue runner, evidence gate, and focused regression tests. The
  only checked mismatch was `configs/nano_ar/hpo/r33_dedup_clean_queue.yaml`,
  which is expected because the remote queue YAML carries live training status
  fields and must not be overwritten from the Mac copy.
- Re-ran the focused generation/round-trip/queue/gate shard with GPUs hidden:
  local `55 passed`, RunAI `/workspace/interp/.venv` `55 passed`.
- The clean R33 AR mb1 smoke remained active:
  `r33-dedup-smoke-20k-lr2e5-warmup20-gb192-mb1-2gpu`; latest parsed metric
  was step `27/96`, loss `0.604653`, `fve_nrm=-0.082879` at
  `2026-06-11T13:29:25Z`.
- Final clean AR checkpoint/eval artifacts were still absent, and the R27
  `256/256` full-control round-trip report was still absent. The sequencer
  remained alive and waiting behind the active AR process, so no competing
  R27/R33 AV work was launched.

Follow-up at `2026-06-11T13:34:17Z`:

- Clean R33 AR mb1 smoke advanced to step `28/96`, loss `0.586248`,
  `fve_nrm=-0.049918`; recent median step time remained about `4.52` minutes,
  with training ETA around `2026-06-11T18:40:47Z`.
- RunAI disk and GPU state remained healthy for the active job:
  `/workspace/interp` `241G/1008G` used, about `767G` free; both H200s were
  holding about `96.7 GiB` each.
- `scripts/nano_queue_gate.py` dry-run for the first clean R33 AV smoke returned
  `ready=false` and `changed=false` for the expected reasons: AR dependency
  still `training`, expected checkpoint path absent, `eval_report` missing,
  R27 `256/256` report missing, and the `gate.passed` JSON path missing.
- `scripts/nano_roundtrip_queue.py status configs/nano_roundtrip/r27_baseline_queue.yaml`
  confirmed the R27 round-trip queue is `64/64 complete`, `256/256 pending`,
  and has no running/scoring item.

Follow-up at `2026-06-11T16:44Z`:

- RunAI polling worked until the local CLI token expired. The last successful
  status window showed the clean R33 AR mb1 smoke still training at step
  `70/96`, loss `0.427511`, `fve_nrm=0.234366`, with final checkpoint/eval and
  the R27 `256/256` round-trip report still absent.
- Verified the modular AV generation, round-trip evaluator, config renderer,
  round-trip queue, AV queue, and gate plumbing in the RunAI venv with
  `44 passed`.
- Created an ignored repo-local `.venv` for Mac-side focused verification and
  installed only the test/runtime dependencies needed by this shard
  (`pytest`, `numpy`, `pyarrow`, `PyYAML`). The same focused local shard passed
  with `44 passed`, matching the RunAI `/workspace/interp/.venv` result above.
- Static local checks passed for the modular generation/round-trip files:
  `py_compile` and `git diff --check` completed without errors.
- RunAI CLI access then failed with `Authentication failed. the token has
  expired`; resume live status checks after refreshing RunAI login from the
  Mac. No new training, eval, or queue mutation was launched from this expired
  auth state.

Follow-up at `2026-06-12T15:26Z`:

- RunAI access was restored. Workspace `train` was `Running`; no active Nano
  train/eval/conversion process was present and both H200s were idle
  (`4 MiB` used on each GPU). `/workspace/interp` had about `695G` free.
- The R27 `256/256` full-control round-trip baseline completed at
  `2026-06-11T22:56:38Z` and passed its gate. Validation/test `av_real`
  normalized MSE was `0.000180003 / 0.000175571`; teacher text normalized MSE
  was `0.000157706 / 0.000155285`. Closed and usable parse fractions were
  `1.0` on both splits, and AV-real beat shuffled, zero, mean, none, and the
  train-mean control with required rowwise win fractions.
- The clean R33 AR mb1 smoke reached the final checkpoint
  `checkpoints/iter_0000096` and logged final training step `95` with loss
  `0.387694` and `fve_nrm=0.305674`.
- The AR queue still marked that smoke `failed` before eval because the
  hard LR-decay canary failed: `final_lr=2e-05 >= 1.8e-05`. Therefore the
  bounded AR eval report
  `eval_iter_0000096_v512_t512_winrates_report.json` is absent.
- The sequential driver correctly refused to unblock the first R33 AV smoke:
  dependency status was `failed` rather than `complete`, and the AR dependency
  had no `eval_report`. R33 AV remains blocked; do not treat the existing AR
  checkpoint as promotion-ready without either a clean schedule rerun or an
  explicitly labeled diagnostic eval.

Diagnostic-only R33 AR eval at `2026-06-12T15:36Z`:

- Ran a manual diagnostic eval of the failed-LR-canary clean R33 AR mb1
  checkpoint without updating queue state and without unblocking the R33 AV
  smoke.
- Checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-smoke/nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb1-2gpu/checkpoints/iter_0000096`.
- Report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-smoke/nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb1-2gpu/eval_iter_0000096_v512_t512_winrates_diagnostic_lrflat_report.json`.
- Log:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-smoke/nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb1-2gpu/eval_iter_0000096_v512_t512_winrates_diagnostic_lrflat_report.log`.
- Validation `512` rows: teacher/source_raw/source_context/mean/shuffled NMSE
  `0.389577 / 0.064048 / 0.458013 / 0.667819 / 0.883286`.
- Test `512` rows: teacher/source_raw/source_context/mean/shuffled NMSE
  `0.396610 / 0.067254 / 0.459581 / 0.668875 / 0.887088`.
- Rowwise wins: teacher beat teacher_shuffled on `100%` of rows; beat
  blank/generic/mean on validation at `98.4% / 98.8% / 98.2%` and on test at
  `97.9% / 98.8% / 97.5%`; beat source_context on
  `75.8% / 72.9%` validation/test. Source_raw remained much stronger and beat
  teacher on almost all rows.
- The temporary HF load report listed missing `backbone.norm_f.weight` and
  `lm_head.weight`; retain that caveat when interpreting this diagnostic.
- Interpretation: teacher-text reconstruction is broadly in line with the
  earlier R33 20k AR probe, so the mb1 non-packed clean path appears capable of
  learning. However, this checkpoint cannot be used as a clean R33 AR gate or
  as evidence to launch AV automatically because the training run failed the
  LR-decay canary (`final_lr=2e-05 >= 1.8e-05`).

Cleanup and LR-policy remediation at `2026-06-12T15:58Z`:

- Saved a pre-delete inventory locally:
  `artifacts/cleanup/20260612T1558Z_contaminated_checkpoint_cleanup/`.
- Uploaded the same inventory and decision manifest to:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/sync_exports/20260612T1558Z_contaminated_checkpoint_cleanup/`.
- Copied the diagnostic R33 AR report/log/train log locally under:
  `artifacts/runai_eval/r33_dedup_clean_diag_lrflat_20260612T1536Z/`.
- Deleted heavyweight contaminated/pre-canary model state from RunAI:
  `/workspace/interp/outputs/nano30b-nla-pilot/checkpoints/r27-av-ar-best`
  and the failed-LR diagnostic R33 AR checkpoint tree under
  `.../nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb1-2gpu/checkpoints`.
  Lightweight logs, reports, queues, datasets, and verifier outputs were kept.
- Deleted heavyweight contaminated/pre-canary model state from S3:
  `checkpoints/r27-av-ar-best/`, checkpoint keys under
  `checkpoint-archives/miles-fsdp2-ar-sft-hpo/`, and checkpoint keys under
  `runai-outputs-archive/outputs/`.
- Post-cleanup verification: RunAI `/workspace/interp` had `152G` used and
  `856G` free. S3 `checkpoints/r27-av-ar-best/` had `0` objects; the R27 AR
  HPO checkpoint archive retained only `2` tiny manifest JSON files; the old
  RunAI outputs archive retained `23` non-checkpoint evidence objects totaling
  about `14.8M`.
- Root-caused the LR canary failure: the Megatron NLA actor was
  unconditionally forcing `self.opt_param_scheduler.lr_decay_style =
  "constant"` after init. Patched
  `external/natural_language_autoencoders/nla/megatron/train_actor.py` so fresh
  SFT runs preserve requested cosine LR decay while still allowing
  `NLA_FORCE_CONSTANT_LR=1` for stale optimizer-resume protection.
- Added regression coverage in `tests/test_nano_miles_launcher.py` and a fresh
  run config/queue item:
  `configs/nano_ar/hpo/r33_dedup_smoke_20k_lr2e5_cosine_warmup20_gb192_mb1_2gpu_lrfix.yaml`
  /
  `r33-dedup-smoke-20k-lr2e5-warmup20-gb192-mb1-2gpu-lrfix`.
- Verification before RunAI sync: local focused shard
  `tests/test_nano_miles_launcher.py tests/test_nano_ar_hpo_queue.py tests/test_nano_ar_hpo_study.py`
  passed with `34 passed, 56 subtests passed`; YAML validation and
  `py_compile` also passed.

Clean R33 AR LR-fix rerun launch at `2026-06-12T16:05:31Z`:

- Synced the current local source/config superset to RunAI through S3 source
  sync archive
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/source-sync/20260612T1618Z/nano30b_source_sync_20260612T1618Z.tgz`
  and copied it into
  `/workspace/interp/code/nano30b-nla-pilot-current`.
- Patched the live Miles checkout at
  `/workspace/interp/code/miles-051cd15/miles/backends/fsdp_utils/actor.py`
  because the R33 AR queue uses
  `--custom-actor-cls-path nla.train_actor.NLAFSDPActor`, not the Megatron
  actor path. The live patch preserves requested fresh-SFT schedules and only
  forces constant LR when `NLA_FORCE_CONSTANT_LR=1`.
- Remote preflight before launch: `/workspace/interp` had `152G` used and
  `856G` free; both H200s were idle; no active Nano train/eval/conversion
  process was running; `scripts/check_miles_patches.py`, `py_compile`, and the
  focused queue/launcher pytest shard passed on RunAI.
- Queue dry run selected only the pending clean item:
  `r33-dedup-smoke-20k-lr2e5-warmup20-gb192-mb1-2gpu-lrfix`.
  The train command included `--lr-decay-style cosine`, `--min-lr 2e-6`,
  `--lr-warmup-iters 20`, `--micro-batch-size 1`, `--global-batch-size 192`,
  `--actor-num-gpus-per-node 2`, and `--no-save-optim`.
- Launched the queue driver:
  `/workspace/interp/outputs/nano30b-nla-pilot/queue_drivers/r33_dedup_ar_lrfix_queue_20260612T160526Z.log`
  with PID `1280474`.
- Run directory:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-smoke/nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb1-2gpu-lrfix`.
- Live scheduler evidence from both ranks at `2026-06-12T16:06:47Z` and
  `2026-06-12T16:06:48Z`:
  `[FSDP] scheduler live: lr=2e-05, decay=cosine, min_lr=2e-06`.
  This confirms the live training process is no longer using the stale flat-LR
  policy.
- First completed training step at `2026-06-12T16:13:38Z`:
  `train/loss=1.2055050532`, `train/fve_nrm=-1.1589517593`,
  `train/lr-pg_0=1.0e-6`, `train/grad_norm=12.0625`, actor-train time
  `406.3s`, and step time `409.9s`.
- Runtime health after step 0: queue status `training`, GPU memory about
  `95.5G / 95.4G`, `/workspace/interp` still `856G` free, no tracebacks or
  LR-canary failure in the early log. Final checkpoint, bounded `512/512` eval,
  and end-of-run LR canary are still pending.

Follow-up throughput correction at `2026-06-12T16:48Z`:

- Manually stopped the slow clean LR-fix rerun
  `nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb1-2gpu-lrfix`
  after it proved the live LR fix but before wasting an 11-hour smoke window.
  Preserved early metrics: step `0` loss/FVE/LR
  `1.205505 / -1.158952 / 1e-6`, step `1`
  `0.922881 / -0.652798 / 2e-6`, and step `2`
  `0.931960 / -0.669057 / 3e-6`. Stop log:
  `/workspace/interp/outputs/nano30b-nla-pilot/queue_drivers/r33_dedup_ar_lrfix_manual_stop_20260612T162805Z.log`.
- Built a throughput ladder under
  `configs/nano_ar/hpo/r33_dedup_throughput_queue.yaml`. The first
  `gb192/mb32/4step` item with `warmup20` failed before training because Miles
  requires `lr_warmup_steps < lr_decay_steps`; a 4-step smoke cannot use
  warmup `20`.
- Corrected packed probes with `warmup2` at `gb192/mb32` and `gb192/mb16`
  both failed the live step-0 reward/train equivalence guard. Observed max
  deviation was about `20-22%`, far above the `2%` guard. Interpretation:
  increasing AR critic `micro_batch_size` currently routes through an unsafe
  packed Nemotron-H path; this is a correctness bug, not an OOM or memory
  headroom issue. Do not use packed `mb>1` AR critic runs as quality evidence.
- Launched the correctness-preserving replacement
  `nano-ar-r33-dedup-throughput-lr2e5-warmup4-gb64-mb1-16step` from
  `configs/nano_ar/hpo/r33_dedup_throughput_smoke_lr2e5_warmup4_gb64_mb1_16steps.yaml`
  and queue log
  `/workspace/interp/outputs/nano30b-nla-pilot/queue_drivers/r33_dedup_ar_throughput_gb64_mb1_20260612T164136Z.log`.
  This keeps `micro_batch_size=1` but reduces `global_batch_size` and
  `rollout_batch_size` to `64`, so each rollout has `32` microsteps instead of
  `96`.
- Remote dry-run selected only that item. Local focused queue/HPO tests passed
  with `21 passed` before S3 sync. RunAI startup confirmed:
  `--global-batch-size 64`, `--micro-batch-size 1`,
  `--rollout-batch-size 64`, `--lr 2e-5`, `--lr-decay-style cosine`,
  `--min-lr 2e-6`, `--lr-warmup-iters 4`, `--num-rollout 16`, W&B offline,
  and `--no-save-optim`.
- Early live evidence: both ranks logged
  `[FSDP] scheduler live: lr=2e-05, decay=cosine, min_lr=2e-06`. Step `0`
  completed with loss/FVE/LR `1.223074 / -1.206614 / 5e-6` and step time
  `174.8s`; step `1` completed with `0.912577 / -0.646430 / 1e-5` and step
  time `111.0s`. GPU memory was about `96.7G` on each H200 during active
  training. This projects to a sub-hour smoke including checkpoint/eval, versus
  the previous `gb192/mb1` ETA of about 11 hours for 96 steps.
- The deeper throughput fix is still code work: implement a correct padded
  batched AR critic train path or fix packed Nemotron-H boundaries so the
  reward/train equivalence guard passes for `mb>1`. Until then, fast clean AR
  smokes should reduce `global_batch_size`/steps while keeping
  `micro_batch_size=1`.

Final throughput-smoke result at `2026-06-12T17:24Z`:

- The `gb64/mb1/16step` run completed and evaluated. Queue status is
  `complete`; no active Nano train/eval process remains; both H200s are idle.
  `/workspace/interp` is at about `224G` used and `784G` free.
- Final checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-throughput/nano-ar-r33-dedup-throughput-lr2e5-warmup4-gb64-mb1-16step/checkpoints/iter_0000016`.
  Eval report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-throughput/nano-ar-r33-dedup-throughput-lr2e5-warmup4-gb64-mb1-16step/eval_iter_0000016_v64_t64_winrates_report.json`.
- Final train step `15`: loss `0.676301`, `fve_nrm=-0.220151`,
  `lr=2e-6`, grad norm `1.054688`, step time `105.1s`. The LR schedule was
  correct: warmup reached `2e-5` at step `3`, then decayed to `2e-6` by the
  final step.
- Bounded `64/64` eval:
  validation teacher/source_raw/source_context/mean/shuffled NMSE
  `0.660454 / 0.064698 / 0.672389 / 0.646815 / 0.776954`;
  test teacher/source_raw/source_context/mean/shuffled NMSE
  `0.654646 / 0.067230 / 0.650309 / 0.645303 / 0.764070`.
- Rowwise wins: teacher beat shuffled `100% / 100%` and blank/generic
  `98.4-100%`, but only beat mean on `35.9% / 35.9%` and source_context on
  `64.1% / 43.8%` validation/test. Source_raw beat teacher on all rows.
- Interpretation: this was a successful throughput/correctness probe, not a
  useful quality checkpoint. It proves the LR policy fix and `gb64/mb1`
  sub-hour smoke shape work, while confirming that 16 steps is too shallow for
  R33 AR selection. The next clean AR quality smoke should keep `mb=1`, avoid
  packed AR, and use a larger step budget such as `32-48` steps at `gb64`
  before deciding whether to pay for a 96-step run.

AR critic batching root-cause fix and mb16 proof at `2026-06-12T19:44Z`:

- Deep pass on the failed `gb192/mb32` and `gb192/mb16` packed probes showed
  two causes. First, the copied Nemotron-H remote code in the R33 critic init
  and base HF directories still had `seq_idx=None  # was seq_idx`, so Mamba
  state leaked across packed samples. Patching those artifacts improved the
  step-0 guard from about `20-22%` max reward/train MSE-ratio error to about
  `14-17%`, proving the boundary patch mattered but did not fully fix
  training correctness.
- The second cause was architectural: Nano/Nemotron-H packed THD critic
  training (`[1, sum(T)]` with reset `position_ids`) is still not equivalent
  to the reward/eval path (`[B, T]` padded with `attention_mask`) even after
  the remote-code Mamba boundary patch. Since the AR objective trains a critic
  value at each sample's last real token, the training path must match the
  reward/eval padded masked layout rather than rely on packed THD.
- Implemented the modular fix in
  `external/natural_language_autoencoders/nla/audit_runtime.py`,
  `external/natural_language_autoencoders/nla/models.py`, and
  `external/natural_language_autoencoders/nla/train_actor.py`:
  `padded_critic_inputs_from_tokens(...)` builds padded IDs, attention masks,
  and last-token indices; `NLACriticModel.forward(..., nla_value_indices=...)`
  now supports both packed and batched hidden layouts; AR critic `_train_step`
  uses the padded masked forward and still logs the backbone/value-head audit
  metrics. The copied remote-code patchers now patch `modeling_nemotron_h.py`
  in critic-init and saved-checkpoint paths as well as eval temp HF dirs.
- Local focused verification passed:
  `tests/test_nano_audit_remediation.py tests/test_nano_critic_model_arch.py`
  with `23 passed, 6 subtests passed`; queue/runner shards also passed.
  RunAI source was patched in place after S3 object-read permissions blocked a
  quick source-sync pull from the container.
- Fresh RunAI proof run:
  `nano-ar-r33-dedup-throughput-smoke-lr2e5-warmup2-gb192-mb16-4step-padded`
  under
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-throughput/`.
  It used `gb192/mb16`, two H200s, W&B offline, `num_rollout=4`, cosine LR
  with warmup `2`, and the dedup R33 AR-SFT source.
- The live step-0 guard passed on real rollout data:
  `[NLA STEP0 CHECK] reward/train MSE ratio: mean=1.0000 max|r-1|=0.0000 n=32`.
  The run then completed all four optimizer steps and saved
  `iter_0000004`. Training metrics improved across the short smoke:
  step `0` loss/FVE `1.201275 / -1.173723`, step `3`
  `0.801719 / -0.450722`.
- Throughput evidence: first step cost `153.8s` including setup, then steady
  state was about `23.7-25.3s/step` at `gb192/mb16`. GPU memory peaked around
  `84G` used before clear and around `76G` after clear per H200, leaving ample
  H200 headroom. This restores practical batched AR iteration without using
  the unsafe packed THD critic layout.
- The only traceback was a post-save W&B offline service atexit
  `ConnectionResetError`; checkpoint save had already completed and the
  tracker showed iteration `4`. The 72G debug checkpoint shards were deleted
  immediately after proof, preserving the 20M run plan/log/split evidence and
  returning `/workspace/interp` to about `224G` used / `784G` free.
- Runner validation/docs were updated so new AR-SFT configs no longer require
  the legacy `training.allow_packed_critic_training` acknowledgement for
  `micro_batch_size > 1`. Historical configs may still contain the flag, but
  new batched AR should use the padded critic path.

R33 clean AV+AR final-candidate decision at `2026-06-15T15:00Z`:

- Operator decision: treat the clean R33 AV+AR pair as the current final
  candidate and skip a new row-matched R27 baseline before hero planning. The
  strict round-trip report still records `gate.passed=false` because the R27
  baseline comparison is not row-identical; this is not an in-run control
  failure.
- AR side:
  `nano-ar-r33-dedup-clean56k-lr5e5-cosine-warmup25-gb192-mb96-128step-padded`,
  checkpoint `iter_0000128`, bounded validation/test teacher NMSE
  `0.361513 / 0.352040`.
- AV side:
  `nano-av-r33-dedup-20k-lr1e4-cosine-warmup5-gb192-mb2-seq1152-dyn512-32steps`,
  checkpoint `iter_0000032`.
- AV eval report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-dedup-smoke/nano-av-r33-dedup-20k-lr1e4-cosine-warmup5-gb192-mb2-seq1152-dyn512-32steps/eval_iter_0000032_v512_t512_gen4_report.json`.
- Round-trip report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-dedup-smoke/nano-av-r33-dedup-20k-lr1e4-cosine-warmup5-gb192-mb2-seq1152-dyn512-32steps/roundtrip_iter_0000032_v64_t64_report.json`.
- AV proxy eval: validation/test real NLL `1.003093 / 0.976148`, beating
  shuffled, zero, mean, and no-injection controls.
- Round-trip eval: validation/test AV-real NMSE
  `0.000128805 / 0.000135702`; teacher-text NMSE
  `0.000111481 / 0.000121767`; generated explanation closed/usable fractions
  `1.0 / 1.0`; empty generated explanations `0`.
- AV-real beat mean, none, zero, shuffled, and AV-mean controls by aggregate
  NMSE and rowwise win-rate threshold on both validation and test.
- Next mode: freeze/preserve compact evidence, clean stale failed/blocked queue
  artifacts, keep the selected AR/AV checkpoints until a hero replacement
  exists, and prepare the hero-scale R33 plan. Do not launch more AV/AR training
  before cleanup and config review.

Cleanup executed at `2026-06-15T15:15:56Z`:

- RunAI cleanup manifest:
  `/workspace/interp/outputs/nano30b-nla-pilot/cleanup_manifests/20260615T1515Z`.
- Local manifest:
  `artifacts/cleanup/20260615T1500Z_r33_final_candidate_cleanup_manifest.json`.
- Deleted non-selected checkpoint/model payloads:
  - `miles-fsdp2-ar-sft-r33-dedup-throughput/...gb64-mb1-16step/checkpoints`
    (`72G`).
  - `miles-fsdp2-ar-sft-r33-dedup-throughput/...gb192-mb16-4step-padded/checkpoints`
    (`12K`).
  - top-level stale `outputs/nano30b-nla-pilot/checkpoints` stub (`4K`).
  - non-selected R33 critic-init `model.safetensors` (`36G`).
- Kept selected checkpoints:
  - AR `iter_0000128` (`72G`).
  - AV `iter_0000032` (`59G`).
- RunAI `/workspace/interp` improved from `356G` used / `652G` free to
  `249G` used / `760G` free. S3 inventory found no matching stale checkpoint
  payloads to delete; local scan found no heavyweight checkpoint/model payloads.

## 2026-06-15 R33 Component-Full Hero Execution

After the clean dedup R33 AV+AR candidate was selected, the project promoted to
a component-preserving full-source R33 hero path. This preserves the
`275,396` teacher-backed rows while keeping duplicate content components within
one materialized split instead of dropping them.

Dataset and verifier evidence:

- Root:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_component_fullscan275396`
- AR verifier:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_component_fullscan275396/verify_ar_R33_component_fullscan275396.json`
- AV verifier:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_component_fullscan275396/verify_av_R33_component_fullscan275396.json`
- Rows: `275,396`; `d_model=2688`; nonfinite activations `0`; empty
  explanations `0`; materialized split doc/content overlap `0`.

AR hero result:

- Run id:
  `nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96`
- Checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001289`
- Eval report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/eval_iter_0001289_v512_t512_winrates_report.json`
- Validation/test teacher NMSE: `0.320616 / 0.292730`.
- Validation/test source_raw NMSE: `0.095084 / 0.080078`.

AV smoke and round-trip gate:

- Run id:
  `nano-av-r33-component-full-smoke-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512-32steps`
- AV eval:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-smoke-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512-32steps/eval_iter_0000032_v512_t512_gen8_report.json`
- Round-trip report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-smoke-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512-32steps/roundtrip_iter_0000032_v64_t64_report.json`
- AV real NLL validation/test: `1.051308 / 1.049516`, beating shuffled,
  zero, mean, and no-injection controls.
- Round-trip validation/test AV-real NMSE: `0.000140105 / 0.000135508`;
  closed and usable parse fractions were `1.0 / 1.0`; AV-real beat all
  in-run controls.

Full AV hero:

- Queue:
  `configs/nano_av/hpo/r33_component_full_hero_queue.yaml`
- Run id:
  `nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512`
- Status at `2026-06-15T22:35:21Z` historical poll: training.
- Latest measured progress at that poll: step `3`, train loss `2.407763`,
  latest step time about `188s`, both H200s at about `142.6G/143.8G`, and
  `/workspace/interp` at about `504G` free.
- Expected final proof path: `iter_0001291` DCP checkpoint -> temporary HF ->
  corrected AV eval -> cleanup -> `256/256` AV-generated-text round-trip gate.

Read at that historical poll: the component-full AR side was already a clean
hero candidate, while the full AV hero still needed corrected AV eval and the
`256/256` round-trip report. Supersession: the next entry records that those
final proof artifacts later completed and passed.

### completed: r33-component-full-av-ar-hero-20260621

- status: selected internal hero milestone
- completed_at_utc: `2026-06-19T06:28:57Z`
- evidence_freeze_utc: `2026-06-21T15:57Z`
- dataset:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_component_fullscan275396`
- dataset read:
  component-preserving R33 extraction over the full teacher-backed coverage,
  with `275,396` verifier rows, `d_model=2688`, nonfinite activations `0`,
  empty explanations `0`, and materialized split doc/content overlap `0`.

Selected AR side:

- run id:
  `nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96`
- checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001289`
- eval report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/eval_iter_0001289_v512_t512_winrates_report.json`
- validation/test teacher NMSE:
  `0.320616 / 0.292730`.
- validation/test source_raw NMSE:
  `0.095084 / 0.080078`.

Selected AV side:

- run id:
  `nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512`
- checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291`
- corrected AV eval:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/eval_iter_0001291_v512_t512_gen8_report.json`
- validation/test real NLL:
  `0.798672 / 0.819993`.
- validation/test shuffled NLL:
  `1.331095 / 1.361868`.
- validation/test zero NLL:
  `1.167483 / 1.196865`.
- validation/test mean NLL:
  `1.241662 / 1.287035`.
- validation/test no-injection NLL:
  `1.224772 / 1.259839`.

Final actual NLA gate:

- round-trip report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/roundtrip_iter_0001291_v256_t256_report.json`
- gate passed: `true`
- baseline required: `false`
- validation/test AV-real NMSE:
  `0.000109680 / 0.000121664`.
- validation/test teacher-text NMSE:
  `0.000106810 / 0.000112370`.
- parse health:
  closed fraction `1.0` and usable fraction `1.0` on validation/test.
- controls:
  AV-real beat `mean`, `av_mean`, `av_none`, `av_zero`, and `av_shuffled`
  controls by aggregate NMSE and rowwise win-rate thresholds on both heldout
  splits.

Preservation and cleanup:

- local compact evidence archive:
  `artifacts/runai_sync/20260621T155000Z_r33_component_full_hero/20260621T155000Z_r33_component_full_hero_compact.tgz`
- archive SHA-256:
  `67063bf2ecb3c0face452410060aef42b29556e2d168ad52fc2dcb0933c7213b`
- S3 evidence prefix:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/sync_exports/20260621T155000Z_r33_component_full_hero/`
- cleanup manifest:
  `artifacts/runai_sync/20260621T155000Z_r33_component_full_hero/cleanup/20260621T155000Z_r33_component_full_cleanup.txt`
- storage cleanup:
  deleted only the superseded component-full AV smoke `checkpoints/` payload
  (`59G`), reducing `/workspace/interp` from `681G` used / `327G` free to
  `622G` used / `386G` free. The selected AR `iter_0001289` and AV
  `iter_0001291` checkpoint payloads remain on RunAI.

Interpretation:

- This is the first selected clean component-full R33 AV+AR milestone with
  both strong AV real-vs-control losses and an actual
  `h -> AV-generated explanation -> AR h_hat` gate.
- It is an internal hero milestone, not an external R33-vs-R27 proof. The final
  gate intentionally set `baseline_required=false`; a fresh row-matched clean
  R27 round-trip comparison should be restored before any public claim that R33
  beats R27 end-to-end.

### decision: post-hero-rl-and-qwen-comparability-20260621

- status: logged decision guidance
- decision_at_utc: `2026-06-21T16:20Z`
- context: the component-full R33 AV+AR SFT hero now has a selected AR
  checkpoint, selected AV checkpoint, corrected AV real-vs-control eval, and an
  actual AV-generated-text -> AR reconstruction gate.

Current answer on RL:

- Do not start RL immediately.
- The project has cleared the first internal SFT-round-trip milestone, but RL
  should wait for one short post-hero audit/eval pass:
  - re-run or expand the round-trip gate with a fresh row-matched clean R27
    baseline if the next claim is R33-over-R27;
  - run Qwen/Gemma-style released-checkpoint QC side by side with the Nano R33
    reports so scale and metric semantics are easy to compare;
  - freeze the selected AR checkpoint for reward/scoring audits before any RL
    changes the AV behavior;
  - define the RL reward terms, length/parse penalties, KL/reference policy,
    and stop criteria before launching even a smoke.

Comparability read:

- We do have a full clean R33 AV+AR SFT run in the sense required by this pilot:
  R33 activation extraction, AR SFT, AV SFT, corrected AV eval, and generated
  text round-trip scoring all completed on the component-full teacher-backed
  dataset.
- It is comparable to the released Qwen NLA work at the level of stage and
  contract shape: separate AV and AR checkpoints, normalized reconstruction
  scoring, correct-vs-control evaluation, and generated-text reconstruction
  through the AR side.
- It is not yet a clean claim of "Qwen-level Nano NLA" or "R33 beats R27" for
  two reasons:
  - the final R33 gate used `baseline_required=false`, so it did not require a
    row-matched clean R27 comparison;
  - the released Qwen checkpoints remain the positive-control reference, and
    the Nano numbers need a deliberate side-by-side report before external
    comparability language.

Recommended next step:

- Run a compact post-hero audit/eval package, not RL:
  - `512/512` or larger R33 round-trip with the same selected AR/AV
    checkpoints, preserving generated JSONL;
  - row-matched clean R27 baseline if feasible;
  - Qwen released-checkpoint QC summary beside Nano R33 metrics;
  - reward-design dry run that scores existing generated texts without training.
- If those pass, launch RL as a very small offline/sequential smoke with frozen
  AR reward, strict parse/length guardrails, and no hero-promotion language
  until it beats the SFT AV checkpoint on heldout round-trip without degrading
  AV real-vs-control loss.

### execution: post-hero-comparability-reward-and-rl-staging-20260621

- status: completed post-hero steps 1-4 through guarded RL staging; no RL
  training launched.
- executed_at_utc: `2026-06-21T19:30Z`.
- RunAI workspace: `train` / `trustworthy-ai-inference`.
- selected R33 AR checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001289`
- selected R33 AV checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291`

What was executed:

- Reconciled the existing nearest-valid R27 `256/256` round-trip baseline
  rather than rerunning it: the report exists at
  `/workspace/interp/outputs/nano30b-nla-pilot/roundtrip/r27_baseline/r27_roundtrip_v256_t256_full_controls_prefix256_report.json`,
  but the historical R27 checkpoints referenced by the queue are no longer
  present after cleanup. Treat this as a nearest-valid baseline, not a fresh
  row-identical R27 proof.
- Built the post-hero comparability/reward dry-run report:
  `/workspace/interp/outputs/nano30b-nla-pilot/posthero/20260621T163000Z/r33_posthero_comparability_reward_dryrun.md`
  and JSON sidecar with the same basename.
- R33 vs R27 round-trip comparison from those reports:
  - validation AV-real NMSE: R33 `0.000109680`, R27 `0.000180003`
    (`39.1%` lower for R33);
  - test AV-real NMSE: R33 `0.000121664`, R27 `0.000175571`
    (`30.7%` lower for R33).
- Frozen-AR reward dry run on existing R33 AV-generated texts used
  `reward = -rowwise_normalized_mse`:
  - validation reward mean `-0.000109680`, median `-0.000104279`;
  - test reward mean `-0.000121664`, median `-0.000098086`.
- Qwen reference metrics were parsed from
  `docs/qwen_nla_inference_qc_report_20260519.md` into the side-by-side
  report. They are reference-control metrics, not direct Nano thresholds.

RL preflight/staging:

- Fixed `external/natural_language_autoencoders/nla/scripts/rl_preflight.py`
  so critic-only preflight does not eagerly import Ray-only actor training code,
  prepares tokenizers with right-padding plus EOS-as-pad fallback, and compares
  the current padded reward path against the current padded critic-train path
  selected via `nla_value_indices`.
- Added local regression coverage in
  `tests/test_nano_critic_model_arch.py`.
- Remote critic preflight passed:
  `/workspace/interp/outputs/nano30b-nla-pilot/posthero/20260621T163000Z/r33_rl_preflight_critic_padded_20260621T191500Z.log`
  with reward/train MSE ratios `[0.9999999, 0.9999998, 1.0, 1.0]` and
  `max |ratio - 1| = 0.0000`.
- Staged a 512-row R33 RL smoke parquet without generating new teacher text:
  `/workspace/interp/outputs/nano30b-nla-pilot/posthero/20260621T163000Z/rl_smoke/rl_R33_fullscan275396_smoke512.parquet`.
  Manifest:
  `/workspace/interp/outputs/nano30b-nla-pilot/posthero/20260621T163000Z/rl_smoke/rl_R33_fullscan275396_smoke512_manifest.json`.
  It has `512` rows, `d_model=2688`, `stage=rl`, and prompt text containing
  the `<INJECT>` placeholder for `NLADataSource`.
- RL was not launched. The reference RL launcher uses separate actor, critic,
  and rollout/SGLang GPU groups. The smallest honest topology is at least
  `ACTOR_GPUS=1`, `CRITIC_GPUS=1`, `ROLLOUT_GPUS=1`, while the current
  workspace has 2 GPUs. Launching on 2 GPUs would test a structurally invalid
  configuration rather than RL quality.
- Staged launch script for a future 3+ GPU workspace:
  `/workspace/interp/outputs/nano30b-nla-pilot/posthero/20260621T163000Z/rl_smoke/r33_rl_smoke_staged_command.sh`.
  Topology decision:
  `/workspace/interp/outputs/nano30b-nla-pilot/posthero/20260621T163000Z/rl_smoke/r33_rl_smoke_topology_decision.json`.

Local evidence:

- Pulled post-hero evidence archive:
  `artifacts/runai_sync/20260621T155000Z_r33_component_full_hero/posthero/20260621T163000Z/posthero_runai_evidence_20260621T193000Z.tgz`
- archive SHA-256:
  `5e636b7e13759983abc44056b9241473c0d7dda77046c1ba6043746400556e53`
- S3 upload from the Mac is currently blocked by stale local AWS credentials
  (`InvalidAccessKeyId`). No secrets were printed. The evidence was pulled via
  RunAI stdout into the local archive instead.

### consolidated-ledger: r33-vs-r27-and-experiment-status-20260621

Purpose: give future audit/research agents one current ledger for the R33 vs R27
story, all major R33/R34/R27 experiment families, and which results are final
evidence versus scouting-only or diagnostic-only. The raw chronology above is
still the source of detailed operational history.

Status labels:

- `selected`: evidence is part of the current internal R33 hero milestone.
- `nearest-valid`: useful comparison evidence, but not a fresh row-identical
  rerun against the selected R33 artifacts.
- `scouting-only`: directionally useful but trained or evaluated before later
  correctness fixes.
- `diagnostic-only`: useful for debugging, not promotion evidence.
- `non-result`: launched or attempted, but not usable for quality claims.

Layer selection and early scale-matched evidence:

| Family | Status | Key result | Read |
|---|---|---|---|
| R27 mature AR basin | `nearest-valid` fallback | teacher NMSE around `0.441 / 0.437`; source_raw around `~0.133 / ~0.130` | Mature fallback, but AR had plateaued. |
| R33 AR 20k | `scouting-only` layer-selection evidence | validation/test teacher NMSE `0.381983 / 0.388301`; source_raw `0.071066 / 0.076216` | Beat mature R27 at much smaller scale. |
| R34 AR 20k | `scouting-only` layer-selection evidence | validation/test teacher NMSE `0.490728 / 0.501399`; source_raw `0.112821 / 0.119030` | Not the AR scaling target. |
| R33 AV 20k corrected eval | `scouting-only` AV layer evidence | real NLL `1.040335 / 1.015130` | Passed real-vs-control AV gate. |
| R34 AV 20k corrected eval | `scouting-only` AV layer evidence | real NLL `1.037261 / 1.013677` | Marginally better AV than R33, but paired AR was much worse. |

Decision from the layer search:

- R33 became the main scaling target because its AR trajectory was much better
  than R27 and R34 under scale-matched probes.
- R27 remains the mature fallback and baseline source, especially for
  historical round-trip comparisons.
- R34 is not the immediate target despite its tiny AV advantage, because the
  AR side is too weak.

Pre-fix R33 scaling and why it was reclassified:

| Family | Status | Key result | Why it is not final evidence |
|---|---|---|---|
| R33 prefix fullscan dataset, first path | `scouting-only` dataset milestone | `275,396` rows, `d_model=2688`, nonfinite `0`, empty explanations `0` | Dataset proof remained useful, but downstream models were later reclassified after packed-boundary and LR-policy bugs. |
| R33 AR 100k HPO, `lr=2e-5` | `scouting-only` | teacher NMSE `0.334868 / 0.323239` | Trained before later clean-batching and LR-policy audit closure. |
| R33 AR 100k HPO, `lr=1e-5`, `mb96` | `scouting-only` | teacher NMSE `0.441161 / 0.419937` | Underperformed; not a candidate. |
| R33 AR 100k HPO, `lr=3e-5` | `scouting-only` | teacher NMSE `0.322812 / 0.313741` | Directionally useful for LR selection. |
| R33 AR 100k HPO, `lr=3e-5`, warmup `25` | `scouting-only` | teacher NMSE `0.321038 / 0.312018` | Small warmup benefit at this LR. |
| R33 AR 100k HPO, `lr=4e-5` | `scouting-only` | teacher NMSE `0.309019 / 0.301218` | Good LR trajectory signal. |
| R33 AR 100k HPO, `lr=5e-5`, warmup `50` | `scouting-only` | teacher NMSE `0.301175 / 0.292956` | Good candidate signal. |
| R33 AR 100k HPO, `lr=5e-5`, warmup `25` | `scouting-only` | teacher NMSE `0.300924 / 0.292944` | Best 100k scouting signal and selected LR/warmup shape. |
| R33 AR full275k pre-fix hero | `scouting-only` | teacher NMSE `0.277565 / 0.276665`; source_raw `0.096948 / 0.091568`; final train FVE `0.5134` | Strong directional result, but trained before the packed-boundary contamination and LR-policy fixes were fully closed. Checkpoint payload was removed. |

Major bug/audit findings that changed experiment interpretation:

- Packed Nano/Nemotron-H AR critic training was not equivalent to padded
  reward/eval for `micro_batch_size > 1`; live guards saw about `14-22%`
  reward/train MSE-ratio drift under the unsafe packed path.
- The correct modular fix was to train AR critics through the padded masked
  path with explicit `nla_value_indices`; this restored reward/train
  equivalence for batched AR.
- A Miles LR-policy bug forced fresh SFT runs to constant LR in one path. The
  fix preserves requested cosine decay for fresh SFT and only allows constant
  LR through the explicit stale-resume guard.
- Old R27 and early R33 checkpoint payloads were cleaned or relabeled after the
  contamination/LR audit. Logs, reports, manifests, and lightweight evidence
  were preserved where available.

Clean AR batching and throughput evidence:

| Run | Status | Result | Read |
|---|---|---|---|
| R33 dedup AR `gb192/mb32` and `gb192/mb16` packed probes | `non-result` | failed live equivalence guard with about `20-22%` drift before padded fix | Do not use unsafe packed AR critic path for quality claims. |
| R33 dedup AR `gb64/mb1/16step` | `diagnostic-only` | eval teacher NMSE about `0.660454 / 0.654646` | Proved LR fix and sub-hour clean smoke shape, but too shallow for quality. |
| R33 dedup AR padded `gb192/mb16/4step` | `diagnostic-only` correctness proof | reward/train MSE ratio `mean=1.0000`, `max|r-1|=0.0000`, `n=32`; steady state about `24-25s/step` | Restored practical batched AR iteration without unsafe packed THD critic layout. |

Clean strict-dedup candidate before full hero:

| Component | Status | Artifact/result | Read |
|---|---|---|---|
| Strict-dedup R33 dataset | clean smoke/gate dataset | `56,351` rows from `275,396`; `d_model=2688`; nonfinite `0`; empty explanations `0`; content split overlap `0` | Good for clean smoke/gate work, not hero-size. |
| R27 `64/64` round-trip baseline | `nearest-valid` | AV-real NMSE `0.000174766 / 0.000173755`; gate passed | Useful early baseline, smaller than final `256/256`. |
| R27 `256/256` round-trip baseline | `nearest-valid` | AV-real NMSE `0.000180003 / 0.000175571`; teacher NMSE `0.000157706 / 0.000155285`; parse closed/usable `1.0 / 1.0`; gate passed | Best remaining R27 comparison report, but historical checkpoints are no longer present for a fresh row-identical rerun. |
| R33 dedup AR clean56k | selected-at-that-stage, superseded by component-full | teacher NMSE `0.361513 / 0.352040` | Clean AR candidate, later superseded by component-full AR hero. |
| R33 dedup AV 20k | selected-at-that-stage, superseded by component-full | real NLL `1.003093 / 0.976148` | Clean AV candidate, later superseded by component-full AV hero. |
| R33 dedup AV+AR round trip | selected-at-that-stage, superseded by component-full | AV-real NMSE `0.000128805 / 0.000135702`; teacher-text NMSE `0.000111481 / 0.000121767`; parse `1.0 / 1.0` | In-run controls passed; strict gate `passed=false` only because the R27 baseline was not row-identical. |

Selected component-full R33 hero:

| Component | Status | Result |
|---|---|---|
| Component-full R33 dataset | `selected` | `275,396` rows, `d_model=2688`, nonfinite activations `0`, empty explanations `0`, materialized split doc/content overlap `0`. |
| AR checkpoint | `selected` | `nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96`, checkpoint `iter_0001289`. |
| AR eval | `selected` | validation/test teacher NMSE `0.320616 / 0.292730`; source_raw `0.095084 / 0.080078`. |
| AV checkpoint | `selected` | `nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512`, checkpoint `iter_0001291`. |
| Corrected AV eval | `selected` | validation/test real NLL `0.798672 / 0.819993`; shuffled `1.331095 / 1.361868`; zero `1.167483 / 1.196865`; mean `1.241662 / 1.287035`; no-injection `1.224772 / 1.259839`. |
| Actual AV-generated-text -> AR round trip | `selected` | validation/test AV-real NMSE `0.000109680 / 0.000121664`; teacher-text NMSE `0.000106810 / 0.000112370`; parse closed/usable `1.0 / 1.0`; gate passed with `baseline_required=false`. |

Current R33 vs R27 read:

| Comparison | Validation | Test | Caveat |
|---|---:|---:|---|
| R33 component-full AV-real NMSE | `0.000109680` | `0.000121664` | Selected clean R33 SFT hero. |
| R27 nearest-valid AV-real NMSE | `0.000180003` | `0.000175571` | Existing report only; not a fresh row-identical rerun. |
| R33 lower NMSE than R27 | `39.1%` | `30.7%` | Good internal evidence, but keep caveat for external claims. |

Interpretation:

- Internally, R33 component-full AV+AR is the current best Nano30B NLA SFT
  candidate and supersedes the strict-dedup candidate and all pre-fix scouting
  checkpoints.
- It is fair to say R33 beat the nearest-valid R27 round-trip report under the
  current post-hero comparison package.
- It is not yet fair to make an uncaveated external claim that R33 beats R27
  end-to-end, because the final comparison did not rerun a clean row-identical
  R27 baseline with live preserved checkpoints.

Qwen/Gemma comparability and reward/RL staging:

- The selected R33 component-full run is comparable to released Qwen/Gemma NLA
  work at the contract level: AV checkpoint, AR checkpoint, correct-vs-control
  metrics, and generated-text reconstruction through the AR side.
- It is not a direct Qwen-level claim because model scale, hidden dimension,
  dataset, and evaluation surfaces differ.
- Frozen-AR reward dry run on selected R33 generated texts used
  `reward = -rowwise_normalized_mse`; validation reward mean/median were
  `-0.000109680 / -0.000104279`, and test reward mean/median were
  `-0.000121664 / -0.000098086`.
- RL preflight now matches the padded reward path and padded critic-train path:
  reward/train MSE ratios `[0.9999999, 0.9999998, 1.0, 1.0]`, max drift
  `0.0000`.
- A 512-row R33 RL smoke parquet was staged, but RL was not launched because
  the reference RL launcher honestly needs separate actor, critic, and
  rollout/SGLang GPU groups; the current 2-GPU workspace is structurally too
  small for that test.

Artifacts to preserve:

- Selected R33 AR checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001289`
- Selected R33 AV checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291`
- Local compact evidence archive:
  `artifacts/runai_sync/20260621T155000Z_r33_component_full_hero/20260621T155000Z_r33_component_full_hero_compact.tgz`
- Post-hero evidence archive:
  `artifacts/runai_sync/20260621T155000Z_r33_component_full_hero/posthero/20260621T163000Z/posthero_runai_evidence_20260621T193000Z.tgz`
- S3 evidence prefix when available:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/sync_exports/20260621T155000Z_r33_component_full_hero/`

Next recommended mode:

- Treat R33 component-full AV+AR as the internal hero SFT baseline.
- Before external/public R33-over-R27 language, either rerun a clean
  row-identical R27 baseline or keep the nearest-valid caveat attached.
- Before RL, move to a 3+ GPU topology and rerun the staged RL smoke preflight
  plus a tiny offline/sequential RL smoke with frozen AR reward, parse/length
  guardrails, and no hero-promotion language until it beats the selected SFT AV
  checkpoint on heldout round-trip without degrading AV real-vs-control loss.

### cleanup: post-hero-workspace-cleanup-20260622

- status: completed conservative workspace cleanup; no jobs launched.
- executed_at_utc: `2026-06-22T15:26:15Z`.
- RunAI workspace: `train` / `trustworthy-ai-inference`.
- Cleanup manifest:
  `/workspace/interp/outputs/nano30b-nla-pilot/cleanup_manifests/20260622T152615Z_posthero_workspace_cleanup.txt`.

Actions:

- Cancelled stale/superseded dedup AR queue items locally and on RunAI:
  - `r33-dedup-full-lr5e5-warmup25-gb192-mb96`;
  - `r33-dedup-clean56k-lr5e5-warmup25-gb192-mb96-128step-padded`.
- Cancelled stale blocked throughput-ladder packed items locally and on RunAI:
  `mb16`, `mb8`, `mb4`, and `mb2` warmup20 variants. These should not be
  launched now that the padded AR critic path and component-full hero supersede
  the ladder.
- Removed local stale queue lock files:
  - `configs/nano_ar/hpo/r33_dedup_clean_queue.yaml.lock`;
  - `configs/nano_ar/hpo/r33_dedup_throughput_queue.yaml.lock`.
- Deleted superseded dedup checkpoint payloads from RunAI while preserving
  reports/logs:
  - `72G` AR clean56k checkpoint tree:
    `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-clean56k/nano-ar-r33-dedup-clean56k-lr5e5-cosine-warmup25-gb192-mb96-128step-padded/checkpoints`;
  - `59G` AV dedup smoke checkpoint tree:
    `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-dedup-smoke/nano-av-r33-dedup-20k-lr1e4-cosine-warmup5-gb192-mb2-seq1152-dyn512-32steps/checkpoints`.

Preserved:

- Selected component-full R33 AR checkpoint remains present:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001289`.
- Selected component-full R33 AV checkpoint remains present:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291`.
- Final component-full R33 `256/256` round-trip report remains present:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/roundtrip_iter_0001291_v256_t256_report.json`.
- Dedup reports/logs remain available as intermediate evidence.

Disk:

- Before cleanup: `/workspace/interp` `622G` used / `386G` free.
- After cleanup: `/workspace/interp` `492G` used / `516G` free.

Next read:

- The workspace is ready for post-hero analysis or a guarded RL smoke on a
  larger topology. Do not relaunch dedup queues unless a future audit explicitly
  reopens that branch.

### rl-smoke: r33-component-full-split-env-debug-20260623

- status: split-env SGLang/RL smoke path debugged through rollout/logprob; 3-GPU
  actor train OOM identified and blocked; 4-GPU trainable smoke queue staged.
- executed_at_utc: `2026-06-23T05:08Z` through `2026-06-23T05:41Z`.
- RunAI workspace: `train` / `trustworthy-ai-inference`.
- Selected SFT inputs:
  - AV actor init:
    `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291`;
  - AR critic init:
    `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001289/hf`;
  - preconverted rollout HF:
    `/workspace/interp/outputs/nano30b-nla-pilot/rl_smoke/r33_component_full_sft_init_512row_3h200/actor_sft_hf_iter_0001291`.

What was fixed:

- Kept the NLA/MILES training venv and SGLang serving venv separate. SGLang
  runs from `/workspace/interp/.venvs/sglang-cu130`, while the RL trainer uses
  `/workspace/interp/.venv`.
- Added Nano/Nemotron-H embedding-table support for
  `backbone.embeddings.weight`.
- Added direct external-engine URL resolution for NLA generation so RunAI router
  `/workers` is not required for `/generate`.
- Applied NLA SGLang transport patches to the dedicated SGLang venv; synthetic
  b64 input-embeds `/generate` returned HTTP 200 after patching.
- Added MILES patch `0012_external_sglang_abort_addrs.patch` so rollout abort
  cleanup can use `rollout_external_engine_addrs` instead of querying the
  forbidden router `/workers` endpoint.
- Kept `NLA_SKIP_ROLLOUT_WEIGHT_SYNC=1` only for the smoke path where the
  external rollout server is preloaded from the actor SFT HF checkpoint. This
  is not a real multi-rollout/hero RL weight-sync solution.

3-GPU smoke result:

- Config:
  `configs/nano_rl/r33_component_full_smoke_queue.yaml`.
- Topology: 1 actor H200, 1 rollout/SGLang H200, 1 critic H200.
- The run passed SGLang startup, healthcheck, Nano embedding load, external
  generation, reward/logprob computation, and reached actor training.
- Rollout evidence:
  - `Rollout generation: 16/16`;
  - `Timer log_probs end (elapsed: 55.1s)`;
  - rollout metrics included `response_lengths=64.0`, `raw_reward=-2.0`,
    `rewards=0.0`, `advantages=0.0`, and `returns=0.0`.
- Failure:
  - actor backward OOMed on GPU 0 at the first actor train microstep;
  - PyTorch reported ~`135.33 GiB` in use and a failed `4.83 GiB` allocation
    with only `4.47 GiB` free.
- Interpretation: single-GPU actor FSDP is not a valid Nano30B RL training
  topology. The 3-GPU queue is now marked `blocked` with `min_actor_gpus: 2` to
  prevent accidental relaunch.

Staged next config:

- New queue:
  `configs/nano_rl/r33_component_full_smoke_queue_4h200.yaml`.
- Topology: 2 actor H200s, 1 rollout/SGLang H200, 1 critic H200.
- External SGLang is configured with `--base-gpu-id 2`, matching the observed
  placement order from the 3-GPU attempt: actor first, rollout next, critic
  last.
- Remote and local focused tests passed after sync:
  `tests/test_nla_generate_url_resolution.py`,
  `tests/test_nla_embedding_loader.py`,
  `tests/test_nano_miles_launcher.py`,
  `tests/test_nano_rl_queue.py`, and
  `tests/test_nano_miles_import_gate.py`.

Next step:

- Resize or redeploy `train` with 4 H200s, then launch
  `configs/nano_rl/r33_component_full_smoke_queue_4h200.yaml`.
- Treat any success from this queue as an RL systems smoke only. It uses
  preloaded external SGLang plus skip-sync, so real RL scaling still needs a
  live weight-sync strategy or a checkpoint-refresh serving strategy.

### rl-smoke: r33-component-full-4h200-post-leakfix-20260623

- status: 4-H200 frozen-critic/skip-sync systems smoke completed; live
  weight-sync pilots still blocked; no RL quality claim.
- executed_at_utc: `2026-06-23T16:58Z` through `2026-06-23T18:40Z`.
- RunAI workspace: `train` / `trustworthy-ai-inference`, `4x H200`.
- Selected SFT inputs:
  - AV actor init:
    `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291`;
  - AR critic init:
    `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001289/hf`;
  - RL smoke parquet:
    `/workspace/interp/outputs/nano30b-nla-pilot/posthero/20260621T163000Z/rl_smoke/rl_R33_fullscan275396_smoke512.parquet`.

Leak/cache fix context:

- The NLA rollout path sends large activation-derived `input_embeds` payloads.
  The router cache-aware/history path is unsafe for this workload because it can
  retain large request bodies and key on token-prefix state that is not
  sufficient for activation-conditioned generation.
- The current RL smoke configs therefore use direct/external SGLang addressing,
  disabled radix cache, router history backend `none`, round-robin routing, and
  short retry/circuit-breaker settings. This is a systems correctness and
  memory-stability requirement, not only a throughput preference.

Completed 4-H200 skip-sync smoke:

- Queue:
  `configs/nano_rl/r33_component_full_smoke_queue_4h200_len256_rb2_fix2_freezecritic.yaml`.
- Run dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_smoke/r33_component_full_sft_init_512row_4h200_len256_rb2_fix2_freezecritic`.
- Topology: `2` actor H200s, `1` rollout/SGLang H200, `1` critic H200.
- Runtime controls:
  `NLA_SKIP_ROLLOUT_WEIGHT_SYNC=1`, `NLA_FREEZE_CRITIC_TRAIN=1`,
  `rollout_batch_size=2`, `global_batch_size=2`, `n_samples_per_prompt=2`,
  `max_response_len=256`, `max_context_len=256`, actor LR `1e-6`, critic LR
  `1e-5`, actor microbatch `1`, constant LR.
- Wall time: started `2026-06-23T17:54:23Z`, completed
  `2026-06-23T18:06:02Z`.
- Rollout/training evidence:
  - rollout generation completed `4/4`;
  - `response_lengths=112.0`, `total_lengths=229.0`, `raw_reward=-0.280732`;
  - shaped `rollout/rewards=5.349516868591309e-06`;
  - `advantages=5.304813385009766e-06`, `returns=5.304813385009766e-06`;
  - `rollout_log_probs=-1.0108380913734436`,
    train `log_probs=-3.2150650024414062`;
  - actor step `0`: loss/pg_loss `-5.304813385009766e-06`,
    entropy loss `2.0284922122955322`, KL `0.0`, clip fraction `0.0`,
    grad norm `9.3125`, actor LR `1e-6`;
  - reward/train MSE equivalence check passed:
    `mean=1.0000`, `max|r-1|=0.0000`, `n=4`;
  - actor checkpoint saved:
    `/workspace/interp/outputs/nano30b-nla-pilot/rl_smoke/r33_component_full_sft_init_512row_4h200_len256_rb2_fix2_freezecritic/actor/iter_0000001`.
- Memory evidence:
  actor ranks reached about `135.0-135.2 GiB` GPU memory before actor cleanup
  and about `126.0 GiB` after cleanup. This confirms the 2-GPU actor topology
  is viable for the tiny smoke, unlike the earlier single-actor-GPU topology.

Interpretation:

- This proves the 4-H200 actor/rollout/critic layout can complete rollout,
  reward/logprob, actor backward, reward/train equivalence, and checkpoint save
  when rollout weights are preloaded and critic training is frozen.
- It is not a true RL pilot result yet because rollout weight sync was skipped
  and critic optimizer updates were frozen. Do not compare this against SFT
  quality metrics.

Live weight-sync pilots after the leak/cache fixes:

- `r33_component_full_pilot_queue_4h200_len256_rb2_sync2_nosaveoptim.yaml`
  failed at `2026-06-23T18:22:21Z` with exit status `134` after entering the
  distributed rollout weight-update path. Logs show the first
  `backbone.embeddings.weight` bucket metadata was sent before the Ray process
  received SIGTERM/abort.
- `r33_component_full_pilot_queue_4h200_len256_rb2_sync2_nosaveoptim_barrierfix.yaml`
  remains stale `training` in the queue YAML, but no live process is present.
  Its train log shows SIGTERM/system-error behavior plus SGLang healthcheck
  timeouts after the actor-rank bucket barrier. Treat it as failed/stale until
  the queue status is cleaned.
- `r33_component_full_pilot_queue_4h200_len256_rb2_sync2_nosaveoptim_unifiedenv.yaml`
  failed at `2026-06-23T18:40:16Z` after switching the trainer to the SGLang
  environment: `ModuleNotFoundError: No module named 'accelerate'` during Miles
  FSDP actor initialization.

Completed 4-H200 live-sync smoke:

- Queue:
  `configs/nano_rl/r33_component_full_pilot_queue_4h200_len512_rb2_sync2_nosaveoptim_unifiedenv_mambawheels_tokcompat_nopackedcheck_criticfwd_evalmode_nofastpath_timesteplimit_allocseg.yaml`.
- Run dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_pilot/r33_component_full_sft_init_512row_4h200_len512_rb2_sync2_nosaveoptim_unifiedenv_mambawheels_tokcompat_nopackedcheck_criticfwd_evalmode_nofastpath_timesteplimit_allocseg`.
- Topology:
  `2` actor H200s, `1` managed external SGLang H200, `1` frozen AR-critic
  H200.
- Runtime controls:
  unified SGLang/Torch env with Mamba wheels, TokenizersBackend fallback patch,
  indexed/eval-mode critic reward forward, critic Mamba fast-path disabled only
  during reward scoring, AR critic `time_step_limit` JSON float sentinel
  normalization, `NLA_ASSERT_PACKED_EQUIV=0`, `NLA_FREEZE_CRITIC_TRAIN=1`,
  and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Wall time:
  started `2026-06-23T20:18:50Z`, completed `2026-06-23T20:26:56Z`.
- Result:
  queue status `complete`; two live-sync rollout/update cycles completed with
  `4/4` generations per rollout; actor checkpoint saved at
  `actor/iter_0000002`.
- Rollout/training evidence:
  - rollout 0: `response_lengths=235.75`, `total_lengths=352.75`,
    `raw_reward=-0.9464928060770035`, actor train time `175.0s`;
  - rollout 1: `response_lengths=234.0`, `total_lengths=351.0`,
    `raw_reward=-0.6190555766224861`, actor train time `10.9s`;
  - rollout 1 reached the previous OOM point and completed instead of failing.
- Memory/checkpoint evidence:
  - rollout 0 cleanup left about `18.66/19.27 GiB` free on actor GPUs;
  - rollout 1 peaked near `143.1 GiB` used on each actor H200;
  - actor DCP save wrote two model shards of about `31.6GB` each and metadata,
    then completed at `2026-06-23T20:26:52Z`.

Interpretation:

- Live actor-to-SGLang weight sync is no longer the immediate blocker for a
  tiny 4-H200 R33 RL systems run.
- `expandable_segments` fixed the prior actor backward OOM enough for the
  two-rollout smoke, but memory headroom is still very thin on the 2-GPU actor
  partition.
- Interval checkpointing dominated the tail latency and consumed about `63GB`
  for this smoke. Future smoke/HPO RL configs should either set save interval
  above the smoke length or save only when the checkpoint itself is required.
- This is still not a quality-bearing RL result. It is systems proof that the
  selected R33 SFT actor, frozen R33 AR reward, managed SGLang rollout, weight
  sync, reward scoring, actor backward, and checkpoint save can run end to end.

Current RunAI state after the live-sync smoke:

- Queue status: `complete`.
- Only defunct Ray/SGLang children remained after process exit.
- `/workspace/interp`: about `683G` used / `325G` free immediately after the
  actor checkpoint save.

Next read:

- The post-contamination R33 SFT numbers are documented and remain the current
  clean hero evidence.
- The post-leak-fix RL numbers above are systems smoke evidence only.
- Before a real RL pilot, create a follow-up no-interval-checkpoint smoke with
  the allocator setting retained, then increase rollout count/row coverage and
  evaluate round-trip/AR reconstruction deltas before any hero-scale RL run.

### 2026-06-25 - 8x H100 R33 RL Medium Gate, No Hero Promotion

Completed the 8x H100 TP2 RL ladder through a 32-rollout medium run:

- Queue item:
  `r33-component-full-rl-8h100-medium-rb8-n8-kl3e4-tp2-rollout32`.
- Run dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/r33_component_full_rl_8h100_medium_rb8_n8_kl3e4_tp2_rollout32`.
- Topology:
  actor FSDP on GPUs 0-3, one TP2 SGLang rollout server on GPUs 4-5,
  frozen AR reward/critic on GPUs 6-7.
- Config:
  `rollout_batch_size=8`, `n_samples_per_prompt=8`, `global_batch_size=64`,
  `num_rollout=32`, actor LR `3e-6`, KL coefficient `3e-4`.
- Systems result:
  completed without OOM; steady update time `~256s`; final post-eval completed
  at `2026-06-25T03:06:20Z`; temporary HF and actor DCP checkpoint payloads
  were cleaned; GPUs returned to idle; `/workspace/interp` returned to about
  `677G` used / `331G` free.
- Training signal:
  raw reward mean/std/min/max `-0.405441 / 0.084653 / -0.652331 / -0.258299`;
  response length mean `118.91`; entropy mean `1.0975`; `train/ppo_kl` logged
  `0.0` with small finite `kl_loss` values.
- Round-trip report:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_8h100/r33_component_full_rl_8h100_medium_rb8_n8_kl3e4_tp2_rollout32/roundtrip_iter_0000032_v64_t64_report.json`.
- Local artifact mirror:
  `artifacts/runai_rl/20260625T031500Z_r33_rl_rollout32/`.
- S3 artifact mirror:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/sync_exports/20260625T031500Z_r33_rl_rollout32/r33_rl_rollout32_light_artifacts_20260625T0315Z.tgz`,
  SHA256 `07b937c0abec9bbfdaf9e05144158f86be333be9ff5392828c01484208695043`.
- Round-trip result:
  gate `false`; validation/test AV-real NMSE `0.000108897 / 0.000123269`;
  clean SFT baseline `0.000109680 / 0.000121664`; teacher
  `0.000104941 / 0.000112355`; closed/usable parse fractions `1.0 / 1.0`.
  AV-real beat all controls on both splits, but the test split regressed
  versus clean SFT.

Decision:

- Do not launch the RL hero run from this exact `3e-6`, 32-rollout evidence.
- Treat the 8x H100 topology as systems-ready but not quality-ready.
- Next RL work should modify the learning signal or actor update dynamics before
  scaling to `num_rollout=64/128`: likely LR `5e-6`, KL/SFT-anchor diagnostics,
  advantage/reward normalization checks, and larger eval limits once generation
  throughput is less expensive.

### 2026-06-30 - R33 Cross-Topology Functional Calibration

The validity-first R33 work found that strict equality between historical
stored activations and freshly captured activations is not a valid
cross-topology identity test. The stored R33 dataset came from a 2x H200/GH200
runtime, while the current evaluator runs on 8x H100 NVL with a different
framework stack. Four sampled stored vectors differed from current captures by
roughly `1%` to `7%` relative L2, even after replaying historical extraction
geometry. Current repeated forwards and current full-vs-prefix extraction were
bit-exact.

Fresh capture followed by reinjection at the R33 final-token boundary produced
exact original logits, confirming the patching implementation. The functional
gate was therefore corrected to:

- require exact fresh capture/reinjection logit identity;
- report stored activation drift as calibration evidence;
- score a `stored_gold` injected variant as the current-runtime replay floor;
- express candidate and SFT functional gaps relative to that floor.

The corrected four-row report is:
`/workspace/interp/outputs/nano30b-nla-pilot/validity/r33-sft/functional_v2_stored_gold_v2_t2_20260630_report.json`.
It passed fresh identity. Validation/test KL was `0.0023096 / 0.00120877`
for stored-gold, `0.343964 / 0.500493` for the AV candidate, and
`0.379341 / 0.580593` for teacher text. The 2/2 split is only a pipeline
check; no scientific promotion is claimed from it.

The train-only mean-control computation was also vectorized. Its measured
247,872-row runtime is about `7.45s` with about `1.50GB` peak RSS.

A config-only Stage 1 evaluation queue now targets 512/512 evidence for the
clean R33 SFT actor, update-16 RL actor, and update-32 RL actor. Existing
generated rows are repartitioned into exact worker shards so generation resumes
only the missing half. The queue converts DCP checkpoints to temporary HF in
`/dev/shm`, enriches stable provenance, scores with the selected R33 AR, and
cleans the temporary model. It launches no training.

The Stage 1 queue launched at `2026-06-30T22:26Z` on 8x H100 NVL. SFT
conversion completed and all eight generation workers became active. The seed
invariant passed: 512 reused rows, 64 total per worker, with each worker adding
64 rows to reach its 128-row 512/512 target. At `22:47Z`, workers had reached
`77-80/128` without OOM or failure.

A second evaluation-only queue is armed behind a fail-closed prerequisite
watcher. It will run update-16/update-32 semantic invariance, functional
recovery, stratified qualitative-panel construction, and the response-closure
cap audit only after the first queue completes. Qualitative evidence now
requires explicit review of every selected row; a pending panel cannot pass
the composite gate. No training is part of either queue.

The update-32 response-closure audit was safe to run early because it uses only
cached text and the tokenizer. Validation/test closed fractions were
`0.992188 / 0.982422`. Only `0.019531 / 0.037109` closed by token 150 and
`0.585938 / 0.519531` by token 192. Token 224 covered
`0.917969 / 0.902344`, while token 256 covered `0.992188 / 0.982422`.
Closed-response p95 positions were about `231 / 232` tokens. Therefore the
existing 256-token corrected-probe and hero settings are retained; 150, 192,
and 224 would violate the 95% closure requirement.

The functional-analysis continuation now includes a `16/16`, batch-8
update-16 canary before either 512/512 functional pass. Full update-16 and
update-32 functional configs use batch 8 only if that queue item exits cleanly.
The canary chooses the 16 longest prefixes per split using `n_raw_tokens`.
This tests the throughput setting on the actual two-GPU target/AR placement and
the memory-heavy sequence tail, and fails closed on OOM or identity error.

The qualitative-panel smoke also identified and fixed a source lookup
bottleneck. The old resolver materialized activation vectors for every scanned
parquet row and was still running after 100 seconds. Key-first scanning now
materializes only matching rows; the identical 1,024-row lookup and 50+50 panel
completed in 11 seconds against the full 275k source table. The smoke panel is
deliberately review-pending. The initial trigram repetition hint over-flagged
legitimate structured phrases at 8/5 validation/test rows. After requiring
duplicate units or repeated four-grams to cover at least 25% of the response,
full-set hints fell to 1/3 and the fixed panel to 0/1. The remaining rows will
be inspected in the final candidate-versus-SFT review.

The final fixed update-32-versus-SFT qualitative panel was reviewed row by
row. Validation had `0/50` flags. Test had `1/50`: row `262022` produced a
repeated ISSN-like zero sequence and was marked as a real readability failure.
All other rows were readable, including the long but coherent row `262205`.
The reviewed report and explicit 100-row decision record are mirrored under
`artifacts/runai_validity/20260630_r33_stage1/`; reviewed-report SHA256 is
`d4476dd160587fc16db887884259a92da5f6a04f68adebf43460cf0a563e6a8a`.
This qualitative component passes its 5% threshold, but no composite Stage 1
promotion is claimed until round-trip, invariance, and functional reports also
complete.

The matched SFT 512/512 round-trip baseline then completed. Validation/test
real NMSE is `0.000126796 / 0.000134752`, FVE is
`0.502652 / 0.451356`, closed fraction is `0.978516 / 0.972656`, and usable
fraction is `1.0 / 1.0`. The report SHA256 is
`80a8222ee13ac7fa0172d0cf7c07ded64bb18c72b19870c9b9d5847d19fe23fe`.

The existing update-32 512/512 report has identical row identities and real
NMSE `0.000092868 / 0.000096089`, a `26.76% / 28.69%` reduction from matched
SFT. Paired bootstrap 95% confidence intervals for SFT-minus-update32 NMSE are
strictly positive: `[2.8933e-5, 3.9040e-5]` validation and
`[3.3783e-5, 4.3711e-5]` test. Thus update32 passes the independent
round-trip improvement component. It is not promoted until invariance and
functional target-model evidence complete.

### R33 Stage 1 Composite Validity Decision

The complete update-32 Stage 1 bundle now passes all `26/26` configured
checks. Composite report:
`/workspace/interp/outputs/nano30b-nla-pilot/validity/r33-update32/composite_validity_report.json`,
SHA256
`5c5b7341a175b4795fdf8ea20d210b9213ffb9a1a5dc6e4048d7d8598f73795e`.

Update-32 validation/test round-trip NMSE is
`9.28685e-5 / 9.60889e-5`, improving on matched SFT by
`26.76% / 28.69%` and on teacher-text AR by about `15.0% / 12.2%`.
It also significantly improves over update-16 on the same rows: paired
update-16-minus-update-32 NMSE confidence intervals are
`[3.1904e-6, 7.5020e-6]` validation and
`[4.8759e-6, 1.0390e-5]` test.

Semantic validity remains strong. The minimum FVE retained under formatting
normalization or semantic-unit reordering is `0.990260 / 0.984714`
validation/test. Real explanations beat every activation control on at least
`0.990234 / 0.996094` of rows. Usable parse fraction is `1.0`; close fraction
is `0.992188 / 0.982422`; injection-marker and CJK leakage counts are zero.
The reviewed 50+50 panel contains zero validation flags and one test flag, the
known repeated ISSN-like numeric suffix on row `262022`.

Functional recovery independently supports the round-trip result. Update-32
KL from original to patched target-model logits is `0.870409 / 0.944165`,
versus SFT `1.430538 / 1.837270` and teacher text
`1.083416 / 1.141398`. Top-10/top-50 overlap is
`0.6662/0.6848` validation and `0.6449/0.6550` test, improving over SFT.
Direct paired update-16-minus-update-32 functional-KL confidence intervals are
`[0.02532, 0.23375]` validation and `[0.09847, 0.34107]` test.

The AR loader's missing `backbone.norm_f.weight` and `lm_head.weight` warning
is expected: `NLACriticModel` intentionally strips both modules and uses the
pre-final-norm residual stream plus its value head. Fresh capture/reinjection
is exact on all 1,024 functional rows. Historical stored vectors differ from
fresh H100 captures by mean relative L2 `0.03010`, but repeated current
forwards are exact and the stored-gold target-logit replay remains close
(KL about `0.0035 / 0.0034`). This is recorded as cross-runtime extraction
calibration rather than a candidate-versus-SFT confound.

The model checkpoint result is therefore independently validated, but the
historical training recipe is not promoted. That run used signed `k1` KL,
generated 480 samples while effectively training 384 per update, and exceeded
the new `0.75` train-versus-rollout log-probability drift threshold from update
20 onward. Update-32 remains the scouting best. The next scientific step is
the corrected divisible `k3` probe pair, followed by one guarded 32-update
confirmation before any fixed-AR hero run.

### Corrected K3 Stage 2 Launch

Before launching the corrected probes, the train-only RL data path was repaired
and made explicit. Raw R33 activations are now filtered by the authoritative
held-out split and combined with the selected actor sidecar to construct the
canonical list-of-chat prompt. AR critic prompts and teacher responses cannot
flow into the RL parquet. The builder emits a complete RL sidecar, while the
verifier independently enforces prompt/token contracts, source hashes,
activation geometry, provenance uniqueness, finiteness, and split isolation.

Both Mac and RunAI passed the 73-test focused regression suite. The source
bundle is under S3 `source-sync/20260701T180002Z/` with archive SHA256
`7d9f0ee923314e15132dacd73d0c98657900c81320dd2860153f0e161ce530fc`.
The resulting R33 train-only RL parquet has exactly 247,700 rows from 24,867
train documents. Its verifier reports 247,700 unique keys, all vectors finite
at layer 33 and dimension 2688, zero held-out/nontrain overlap, zero duplicate
keys, zero teacher columns, and canonical single-`<INJECT>` prompts for every
row. The report is
`/workspace/interp/outputs/nano30b-nla-pilot/r33_rl_train_only/verify_report.json`
and passes with no blockers.

The corrected queue launched at `2026-07-01T18:13:10Z`. Its first probe uses
`lr=1e-5`, non-negative `k3` KL, exact `gb384`, `mb32`, 48 prompts with eight
responses each, eight updates, and the 6+1+1 topology on 8 H100 NVLs. Initial
SGLang, actor, critic, Ray, and offline-W&B startup was healthy. The second
`lr=2e-5` probe and guarded 32-update confirmation remain dependency gated; no
hero run is active.

### 2026-07-01 - Corrected K3 Live-Sync Recovery

The initial eight-update probe failed before rollout 0 because trainer and
managed SGLang used different PyTorch/CUDA/NCCL environments while joining one
NCCL weight-update group. This was an infrastructure deadlock, not an OOM or
RL-quality result. A new queue preflight rejects that split-runtime topology.

The first unified-runtime canary proved live sync (`6.9s`) and completed its
384-sample actor update, then stopped on a `2.98%` reward-vs-training-layout
diagnostic for the frozen critic. Since this recipe never optimizes the critic,
the retry explicitly disabled only that unused optimizer-layout assertion;
the strict reward preflight, actor drift guard, exact-batch gate, and K3 loss
remained enabled.

The retry
`r33-corrected-k3-live-sync-canary-lr1e5-update1-retry1` completed from
`2026-07-01T21:13:38Z` to `21:22:31Z`. Both weight syncs passed in `7.4s` and
`6.7s`; rollout 0 produced 384 samples with closed/usable fractions
`0.95052 / 0.94271`, reward mean/std `-0.51597 / 0.43549`, and no truncation.
The exact two-microbatch actor update completed in a `104.0s` train phase,
with train-vs-rollout log-probability absolute difference `0.28390` and worst
observed actor allocation about `41.7 GiB`. No OOM, NCCL timeout, or batch loss
occurred. The post-save W&B offline teardown emitted the known benign broken
pipe after queue success.

Evidence is at
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/rl_evidence/20260701T212349Z/`
(SHA256
`7da1b180154e7658596645016c48637844962b816c170dbc6ef0943ccb593f0f`).
The diagnostic 59-GiB checkpoint was deleted after verification; logs and W&B
were retained. The next promoted run is the newly named unified-runtime
eight-update `lr=1e-5` probe. This is still Stage 2 HPO, not a hero run.

### 2026-07-01 - Corrected K3 `lr=1e-5` Probe Result

The promoted eight-update probe completed training and its 256/256 round-trip
gate. All eight exact-384 updates, live syncs, and offline-W&B logging passed
without OOM or distributed failure. Reward improved from `-0.5084` at rollout
0 to a best `-0.2852` at rollout 5 and ended at `-0.3633`. Drift stayed below
the guard (`0.263-0.329`). K3 rose and then fell through update 6, but spiked
to `10.9046` at update 7; this is the main stability caveat.

The gate passed with perfect real-generation parse health and decisive control
wins. Matched-baseline results were:

- validation AV-real NMSE `0.0001085655` versus SFT `0.0001096657`
  (`1.00%` lower; `56.64%` rowwise wins);
- test AV-real NMSE `0.0001195005` versus SFT `0.0001216750`
  (`1.79%` lower; `50.39%` rowwise wins).

Teacher text remained better at `0.0001068927 / 0.0001123025` validation/test.
This is useful positive RL signal but only a narrow SFT gain, so update 8 is
retained as an HPO checkpoint rather than promoted as a hero.

The complete lightweight evidence is under S3 `rl_evidence/20260701T231539Z/`
with SHA256
`2029d86b49d6f72c9b0cd333839d9dd6ae40fd5ad889d000f8f4921d9d778419`.
The model checkpoint remains on the PVC; temporary HF files were removed.

Post-eval also exposed why generation was slow: the generic decoder expected
`past_key_values`, while Nano requires an initialized
`HybridMambaAttentionDynamicCache` passed as `cache_params`. The valid but
uncached 512-row generation took about 64 minutes at roughly 42% GPU use. The
generation engine now adapts to Nano's cache API. Real-model testing then
found two bugs in the bundled Nemotron-H cache implementation itself: it did
not retain `conv_kernel_size`, and it treated list-backed cache state as a
tensor. These are now repaired idempotently by the centralized remote-code
patcher used during checkpoint export. The staged R33 patch report recorded
one kernel-size, four device, and one reset repair with no validation errors;
remote cache/eval tests passed `45/45` with two optional skips.

A fresh one-H100 smoke generated four tokens in `6.65s` with cache use enabled
and no fallback. That established API execution, not deterministic equivalence;
the later equivalence failure documented below supersedes the smoke. The queue
preflight then identified only
`r33-corrected-k3-probe-lr2e5-update8` as promotable, with no missing paths and
the same unified trainer/SGLang runtime. That eight-update, exact-384,
256/256-eval probe launched at approximately `2026-07-01T23:39Z`. The guarded
32-update confirmation remains blocked pending the comparison; no hero run is
active.

### 2026-07-02 - `lr=2e-5` Probe And Valid-Eval Recovery

The `lr=2e-5` corrected-K3 probe completed all eight exact-384 updates. It had
no OOM, NCCL failure, output truncation, or drift-guard stop, but its dynamics
were less attractive than `lr=1e-5`: reward means ended at `-0.3865`, and K3
showed early transients of `599.9822` and `102.9060` before recovering below
`0.1`. Drift remained bounded at `0.284-0.342`. The update-8 checkpoint remains
available for a valid matched comparison.

The first cache-backed 256/256 report from this checkpoint is invalid evidence.
Although the repaired cache API ran, deterministic token checks diverged from
full-prefix decoding at generated token index 1 at both batch size 5 and batch
size 1. The report is preserved and explicitly recorded under the queue item's
`invalidated_post_eval` metadata, along with both equivalence logs; its failed
gate must not be read as an `lr=2e-5` model result.

The generation engine now fails closed for Nemotron-H cache use by default and
adds a `legacy_batch` backend that batches same-prompt controls while retaining
full-prefix recomputation. The post-eval queue also patches temporary HF
exports and safely cleans stale disposable conversions on retry. Local tests
passed `111/111`; RunAI passed `109` with two optional skips. Source was synced
through S3 under `source-sync/20260702T022031Z/` with SHA256
`9bad67dd5a94fb4e35a3b2831e0f9b7c3e9baef5dc60d704348028cc307081f3`.
The replacement 64/64 full-prefix batched eval completed at
`2026-07-02T02:52:28Z` and passed. AV-real was `0.0001069578 / 0.0001179766`
on validation/test versus matched SFT `0.0001095636 / 0.0001207099`, improving
the split means by `2.38% / 2.26%`. All 128 real generations were closed and
usable, and AV-real beat every control on every row.

The matched 64-row comparison against `lr=1e-5` was split: `lr=2e-5` was
`1.64%` worse on validation but `3.86%` better on test. Combined, its mean was
`1.32%` lower, but it won only `45.31%` of rows and the paired bootstrap
interval crossed zero. Because that advantage is not decisive and the
`lr=2e-5` run had severe early K3 transients, `lr=1e-5` is selected for the
guarded 32-update confirmation. The queue has been rewritten accordingly with
storage-safe checkpoints at 16 and 32, 64/64 gates at both milestones, and a
conditional 512/512 `legacy_batch` promotion gate; it remains blocked and was
not launched.

The complete `lr=2e-5` evidence is stored at
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/rl_evidence/20260702T025445Z/r33-corrected-k3-lr2e5-update8-evidence.tgz`
with SHA256
`c49bd2bf35d18773cd6361b621b9d8c60a8657b6b01211a125fb452d799a3469`.
The archive was downloaded and hash-verified locally. Only the losing 59-GiB
actor checkpoint was then removed; W&B, logs, reports, generated text, and
cache diagnostics remain. `/workspace/interp` now has about `332 GiB` free.
No confirmation or hero run is active.

### 2026-07-03 - Corrected K3 Evidence Freeze And Current Handoff

The corrected R33 K3 phase is now reconciled from its preserved artifacts.
The `lr=1e-5` and `lr=2e-5` evidence archives were downloaded from S3 and
hash-verified locally at
`artifacts/runai_rl_evidence/20260701T231539Z/` and
`artifacts/runai_rl_evidence/20260702T025445Z/`. Their SHA256 values are
`2029d86b49d6f72c9b0cd333839d9dd6ae40fd5ad889d000f8f4921d9d778419`
and
`c49bd2bf35d18773cd6361b621b9d8c60a8657b6b01211a125fb452d799a3469`.

The valid result table is:

| Probe | Validation AV-real / SFT NMSE | Test AV-real / SFT NMSE | Eval size | Gate |
| --- | --- | --- | ---: | --- |
| corrected K3 `lr=1e-5`, update 8 | `0.0001085655 / 0.0001096657` | `0.0001195005 / 0.0001216750` | 256/256 | pass |
| corrected K3 `lr=2e-5`, update 8 | `0.0001069578 / 0.0001095636` | `0.0001179766 / 0.0001207099` | 64/64 | pass |

Both valid reports have closed/usable real-generation fractions of `1.0` and
beat shuffled, zero, mean-activation, no-injection, and target-mean controls.
The cache-backed `lr=2e-5` 256/256 report is excluded: deterministic cached
decoding diverged from full-prefix decoding at generated token index 1 at
both batch size 5 and batch size 1. Its poor NMSE and parse rates are an
invalid generation-engine result, not an RL model result.

The exact matched 64-row comparison was split. `lr=2e-5` was `1.64%` worse
than `lr=1e-5` on validation and `3.86%` better on test. Its combined mean was
`1.32%` lower, but it won only `45.31%` of rows and the paired bootstrap
interval for mean `lr2-lr1` loss crossed zero (`[-6.68e-6, 3.31e-6]`). The
training dynamics break the tie in favor of `lr=1e-5`: its largest K3 value
was a final-step `10.9046`, whereas `lr=2e-5` produced early values
`599.9822` and `102.9060`. Both drift ranges stayed below the `0.75` guard.

The implementation work completed during this phase includes unified
trainer/SGLang runtime validation, exact actor-batch enforcement, frozen
critic handling, centralized post-export Nemotron patching, safe
`--post-eval-only` retries, fail-closed real-model cache selection, and the
full-prefix `legacy_batch` backend. Local verification passed `111` tests;
RunAI passed `109` with two optional skips. The final source bundle is
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/source-sync/20260702T025848Z/nano30b-r33-k3-probe-selection-20260702T025848Z.tgz`
with SHA256
`e3b92fb671cab085365f75af688937702b098027c62bf31aaebfc8deaa7cef56`.

Operational status changed after the evidence freeze. At
`2026-07-03T14:26:14Z`, RunAI listed no `train` workspace and no training
workloads. The last mounted snapshot had 8 idle H100 NVLs and `332G` free on
`/workspace/interp`; selected-checkpoint presence must be reverified after a
workspace redeploy. No training was interrupted because no queue driver or
Nano process was active.

The next experiment remains the blocked 32-update `lr=1e-5` confirmation,
not the stale fixed-AR hero YAML. Before launch, reduce checkpoint retention
from four 59-GiB milestones to updates 16 and 32, use 64/64 gates to select
between them, and run 512/512 only on the winner. The older hero YAML still
uses `lr=2e-5` and references a missing Stage-2 gate report, so it is not a
valid launch artifact. A full `131,328`-rollout hero (`342 x gb384`) remains
conditional on the confirmation passing both heldout splits. Current changes
are S3-synchronized but not yet committed beyond branch HEAD `476b408`.

### 2026-07-03 - External Audit Remediation And RunAI Restore

The corrected-K3 evidence was independently audited. The numerical run
forensics were reproducible, but the `lr=1e-5` and `lr=2e-5` update-8 gains
are not statistically distinguishable from zero. For `lr=1e-5`, the test mean
gain was `2.17e-6`, its bootstrap interval crossed zero, row wins were
`129/256`, and five rows accounted for about `99.6%` of net improvement. These
runs remain HPO/stability evidence only. The historical update-32 checkpoint
still provides the large signal to reproduce, but not a validated recipe.

The audit identified launch-state, source-provenance, gradient-clipping,
dataset-order, statistical-gate, unsafe-cache, retention, verifier, and queue
supervision gaps. The remediation is implemented as follows:

- queue items require explicit approval before launch or dependency promotion;
- launch-critical source roots are checked against SHA256
  `150d2832105c007b1d977d45560c26c09b4aff03770f265ba577257756aa67a7`
  and frozen source commit
  `30e5e26e1e831e54b83f5ac7bcf443bf89eda546`; both are written to per-run
  provenance;
- BF16 local-shard clipping scales original gradients after FP32 norm
  computation; future confirmation runs do not skip clipping;
- future rollouts shuffle with seed `42`; completed probes remain explicitly
  recorded as unshuffled and unclipped;
- composite stop guards cover K3 KL, drift, parser health, and response-length
  tails;
- future generated-text evaluation uses `legacy_batch`; the invalid cached
  `lr=2e-5` report remains historical metadata only;
- round-trip gates bind exact parquet hashes, row keys, and document IDs, then
  require full overlap, rowwise wins, relative effect, and document-clustered
  paired-bootstrap confidence intervals;
- confirmation retention is updates 16 and 32 only, with restart-only semantics
  because optimizer state is intentionally omitted;
- missing Stage-2 and independent cross-critic reports fail closed before hero
  launch.

The retained 512/512 SFT report was provenance-hardened at
`/workspace/interp/outputs/nano30b-nla-pilot/validity/r33-sft/roundtrip_v512_t512_hardened_report.json`.
It contains 512 row keys and document IDs for each heldout split, 52 unique
documents per split, and exact parquet hashes. No new model evaluation was run;
this operation joined retained generated-text provenance to the existing valid
report.

RunAI workspace `train` was restored on 8 H100 NVLs. At
`2026-07-03T17:06Z`, all GPUs were idle, no Nano process was active,
`/workspace/interp` had `325G` free, and `/workspace/models` had `454G` free.
The selected `lr=1e-5` update-8 checkpoint was reverified at `59G`. The
Stage-2 and cross-critic gate reports are absent, so the confirmation remains
unapproved and the hero queue remains blocked. No training was launched during
the remediation.

### 2026-07-04 - Corrected K3 Confirmation Selects Update 16

The 32-update corrected-K3 retry completed on 8 H100 NVLs with the audited
`lr=1e-5`, K3 `1e-3`, exact `gb384/mb32`, 48 prompts x 8 samples, and
6 actor + 1 frozen critic + 1 rollout topology. The key-aligned metric reducer
survived the former rollout-22 failure, and the run produced model-only actor
checkpoints at updates 16 and 32 without a CUDA OOM, host OOM, or guard stop.

The initial hardened 64/64 checks were inconclusive: update 16 improved matched
SFT mean NMSE by `2.35% / 2.60%` validation/test, but document-clustered CIs
crossed zero. Update 32 was nearly flat versus update 16 on test, so update 16
was evaluated on the full 512/512 matched set. That diagnostic passed:

- validation `0.0001083722` versus SFT `0.0001267961`, `14.53%` lower,
  clustered CI `[1.1554e-5, 2.6094e-5]`, row wins `62.89%`;
- test `0.0001147213` versus SFT `0.0001347520`, `14.86%` lower, clustered CI
  `[1.2709e-5, 2.8339e-5]`, row wins `58.40%`;
- 52 independent documents per split, positive median deltas, `100%` parse and
  usable rates, exact dataset identity, and all controls beaten.

This clears the predeclared 10% confirmation bar on both splits. Update 16 is
the selected confirmation checkpoint; update 32 is not promoted. A handoff
bug discovered during the short-gate attempt was also fixed: queue status
`complete` no longer promotes unless `gate_passed: true`, and the independent
critic now pins the current hero source tree. The fresh independent R33 AR
critic is training; Stage 2 has been rewritten to consume the selected
update-16 report and generated text. The 342-update hero remains fail-closed
until the independent cross-critic, invariance, functional, closure, and
structural-output gates pass.

The first independent-critic attempt used the historical 2-GPU `mb96`
configuration and exceeded the smaller H100-NVL memory envelope: the failing
rank held `91.19 GiB` and requested another `2.42 GiB`. It produced no
checkpoint. The fail-closed retry keeps exact `gb192`, LR, schedule, data
split, and shuffle seed while moving to 4 FSDP GPUs at `mb48`.

### 2026-07-04 - Independent-Critic 4-GPU Runtime Fault And Controlled Retry

The 4-GPU `mb48` independent critic resolved the memory limit and remained
stable through step 391 at about 6 seconds/step with 57-62 GiB allocated per
GPU. It then failed with a CUDA illegal-memory-access error. The first
synchronous stack frame was in the optional router-entropy system-metrics
forward hook while moving detached router indices to CPU; subsequent NCCL
watchdog errors occurred after the CUDA context was already invalid. The four
participating GPUs showed no volatile ECC errors when inspected. No checkpoint
or eval report was produced, so this run contributes runtime evidence only.

The next retry is deliberately single-variable. It keeps the same verified
dataset and component split, seed 314159 row order, 4-GPU FSDP topology,
`gb192/mb48`, optimizer, `lr=5e-5`, warmup 25, cosine schedule, checkpoint
policy, and 512/512 controls, but disables optional router-entropy
observability. It uses a fresh run identity ending in `4gpu-norouter`; Stage 2
has been rewired to require that exact final checkpoint. All promotion gates
remain fail-closed.

The no-router retry disproved the initial telemetry hypothesis by failing at
the same deterministic boundary. Step 391 completed normally (loss
`0.2966566`, FVE `0.4686700`, finite gradient norm), and rollout 392 failed in
the optimized segmented MoE GPU `torch.bincount` path. The run was neither
memory-bound nor numerically diverged and produced no checkpoint.

Retry 3 makes the routing backend explicit in YAML. It keeps the exact
independent-critic recipe but sets `moe_routing_impl: expert_scan`, a
stock-style per-expert route that avoids GPU sorting and bin counting. RunAI
tests passed `123` with two optional skips, and a direct H100 comparison found
zero forward difference plus matching hidden, router-weight, and expert-weight
gradients. Stage 2 now requires the fresh `4gpu-expertscan` checkpoint; hero
source provenance is commit `78ba931` with SHA256 fingerprint
`bf3694441ba39f3492548570dfcfbfd7b6dbe56858362664b199cc20b8ea268c`.

Retry 3 nevertheless failed at the exact same rollout 392. Step 391 remained
healthy (loss `0.2965752`, FVE `0.4688158`, finite gradient norm), and the
exception moved only to expert-scan's first `torch.where`. This establishes
that the routing implementations are synchronization points rather than root
causes; an earlier asynchronous model kernel is corrupting the CUDA context.

A bounded `CUDA_LAUNCH_BLOCKING=1` diagnostic is prepared for 393 updates. It
saves full optimizer-bearing state at update 384, so the synchronous traceback
can identify the true kernel and later fixes can be tested from eight updates
before the failure instead of retraining from initialization. The diagnostic
itself is not promotion evidence. Current hero source provenance is commit
`c403acc` and fingerprint
`6954ee69f10fa3776c0697c579b1840ada5363658be77ec1f10e55a829ddbb9b`.
An initial diagnostic launch was stopped at step 143 when review showed that
`num_rollout=393` had also compressed the cosine LR schedule. The corrected
config passes `lr_decay_iters=1289`, so shared steps now exactly match the
full-run LR trajectory while execution remains bounded at update 393.

### 2026-07-04 - CUDA-Blocking Diagnostic Crosses Rollout 392

The corrected diagnostic matched the prior expert-scan trajectory and full
cosine schedule. At step 143, LR was exactly `4.9023024e-5`; the archived
short-horizon attempt was already at `3.9354329e-5`. Step 391 completed at
loss `0.2985217`, FVE `0.4653295`, and LR `4.1270958e-5`. Under
`CUDA_LAUNCH_BLOCKING=1`, the formerly fatal rollout 392 also completed at
loss `0.2844250` and FVE `0.4905775`.

The same dataset, component split, seed 314159 row order, `gb192/mb48`, four
H100 FSDP topology, optimizer, LR schedule, and expert-scan routing were used.
A CPU audit found no invalid token IDs, and rollout 392's 213-token maximum was
shorter than multiple earlier successful batches. Launch blocking is therefore
the discriminating variable: the earlier crash is an asynchronous CUDA
race/timing defect, not a bad row, length limit, MoE routing error, OOM, or
training divergence. Blocking suppresses the fault rather than naming the
original asynchronous kernel, so it remains enabled for the independent-critic
continuation.

Restartable model/optimizer/scheduler DCP checkpoints were written at
iterations 384 and 393 under `/workspace/models`; Longhorn remained at 262 GiB
free. The tracker selects iteration 393. A non-gating 64/64 diagnostic eval was
stopped during a redundant projected 44-minute NFS reconstruction, with its log
preserved and no report claimed. The continuation now loads iteration 393,
runs the remaining 896 updates on the original 1289-step schedule, and saves a
model-only final checkpoint for the required 512/512 eval. Runner validation
now distinguishes optimizer-bearing resume input from model-only final output
and verifies optimizer plus scheduler DCP metadata before launch. Local and
RunAI regression shards both pass 97 tests. Stage 2 and hero remain fail-closed.

### 2026-07-04 - Corrected FSDP Resume LR Policy

The initial iteration-393 continuation was stopped after detecting that its
live LR remained at `5e-5`. Checkpoint inspection proved the saved scheduler
was valid (`last_epoch=393`, LR `4.1226691e-5`); the post-load Miles policy was
overwriting every optimizer resume to constant LR. That run produced no
checkpoint and is archived with a `wrong_scheduler_reset` suffix.

A shared `apply_fsdp_live_lr_policy` helper and ordered Miles patch 0018 now
preserve the restored scheduler position, recompute the live optimizer LR, and
reserve constant-LR fallback for an explicit config or environment override.
The corrected retry reported cosine at epoch 393 and advanced from LR
`4.1182339e-5` on logged step 393 to `4.0507070e-5` on step 408, with finite
loss/FVE and no CUDA fault. Source and actor contracts are pinned to
`158336e68fcbfd6a217fc559a65304f405639a85d0c0c186e93a349f03092aba` and
`da198ce079c3ce68ddf761a88a7d328910f828d68e0d0a52515f0d80c6333359`.
After the first valid resumed updates, iteration 384 was removed and iteration
393 retained, increasing model-store free space from 166 to 255 GiB. Stage 2
still requires the final 512/512 critic eval, and hero still requires both
Stage 2 gate reports.

### 2026-07-05 - Independent Critic And Cross-Critic Round-Trip Pass

The corrected independent R33 critic completed 1289 updates and its bounded
512/512 eval. Teacher NMSE was `0.3208674` validation and `0.2924067` test;
teacher strongly beat shuffled/blank/generic/mean controls, while source-raw
provided the expected oracle result (`0.0942281/0.0800047`).

Using this critic, the matched SFT AV baseline scored round-trip NMSE
`0.0001269159/0.0001344858`. Corrected-K3 update 16 scored
`0.0001081018/0.0001144881`, improving by `14.824%/14.870%` on
validation/test. It won `66.99%/59.96%` of paired rows, had positive
doc-clustered 95% confidence intervals on both splits, beat every declared
control, and achieved 100% close and usable rates. This independently
reproduces the primary-critic gain and passes the cross-critic round-trip gate.
This passes the independent round-trip report's gate; the combined
cross-critic gate remains a later Stage 2 item. The guarded chain has advanced
into the remaining validity checks, and the hero queue remains pending.

### 2026-07-05 - Stage 2 Pass And R33 RL Hero Start

The first combined cross-critic invocation exposed a provenance-schema bug,
not a scientific failure: candidate row keys included `n_raw_tokens`, while
the otherwise identical SFT row keys contained only `doc_id`. Dataset hashes,
parquet row indices, document IDs, gains, clustered confidence intervals, and
all quality checks matched. The gate now compares exact dataset hash + row
index + document ID and remains fail-closed for real row or document changes.
The unchanged reports pass every cross-critic check.

Stage 2 completed with strong invariance (`99.10%` minimum validation and
`98.77%` minimum test FVE retention), 100% full response closure, cap-192
closure above 99.8%, completed functional/qualitative evidence, and no
composite blockers. Functional reinjection identity passed all 1,024 rows;
stored-activation replay remained outside its strict tolerance and is retained
as a non-composite diagnostic caveat. Both
`r33-corrected-cross-critic-gate.json` and
`r33-corrected-stage2-gate.json` report `passed: true`.

The hero queue then launched the 342-update corrected-K3 run. Source
fingerprint `aef659279c9306f4818812b0b9eb0cbd24df0d857c562d323f4da221524c32a4`
was verified, W&B is offline, and topology is six actor + one rollout + one
critic H100. Optimizer steps 0-2 completed; step 2 had KL loss `0.6216564` and
rollout/logprob absolute difference `0.2952003`, below the `0.75` drift guard.
No CUDA, OOM, NCCL, or train-guard error was present. This milestone is a
verified hero start, not final hero promotion evidence.

### 2026-07-06 - R33 RL Hero Guard Stop, No Promotion Result

The guarded 342-update run stopped at `2026-07-05T06:02:39Z`. It generated
64 global batches, totaling 24,576 responses (`18.71%` of the planned hero
budget). Actor records cover steps 0-62; step 63 raised the configured guard
after consecutive KL-loss readings of `25.9797955` and `5.1258636` exceeded
the `5.0` threshold. There was no CUDA, OOM, NCCL, or SGLang failure, and
actor/rollout log-prob absolute difference stayed below `0.304`.

The partial trajectory contained positive signal: first-ten to last-ten mean
raw reward improved from `-0.346596` to `-0.272935`, reward standard deviation
fell from `0.224467` to `0.161619`, and close/usable rates averaged above
`99.75%` with zero truncation. It was not stable enough for unattended hero
continuation: KL loss had median `0.6301`, p95 `14.3117`, and maximum
`233.0444`, with matching gradient-norm spikes.

No checkpoint was written because the first save was update 171. Therefore no
hero round-trip eval or promotion gate ran, and the run supplies no final
quality comparison against SFT. Offline W&B and text logs are retained under
`rl_hero/r33_corrected_k3_hero_lr1e5_update342`. Stage 2 remains passed; the
R33 RL hero milestone remains open, and no retry was launched during this
status pass.

### 2026-07-06 - Guard-Calibrated R33 RL Hero Retry Started

Storage cleanup removed approximately 100 GiB of obsolete crash dumps, the
redundant 144-GiB independent-critic continuation checkpoint, and 177 GiB of
model shards from three superseded RL checkpoints. Logs and eval evidence were
retained, while every selected SFT/critic/Stage-2 input was protected by the
applied retention manifest. Free space is now 460 GiB on Longhorn and 456 GiB
on the model store.

Historical replay showed that clean corrected-K3 runs have isolated large KL
samples without a rising median or log-prob drift. The approved retry keeps
`lr=1e-5`, K3 coefficient `1e-3`, global batch 384, and the 6+1+1 topology.
It changes the raw KL guard from two to three consecutive values above 5,
retains the two-step `0.75` log-prob guard, adds a two-step gradient-norm 100
guard, and saves at updates 114/228/342. Final promotion still requires the
declared matched round-trip gates.

The detached run
`r33-corrected-k3-hero-lr1e5-update342-guard3-retry1` started at
`2026-07-06T16:42:47Z`. Preflight had no missing paths, SGLang passed health,
four offline W&B role runs exist, and optimizer step 0 completed with KL `0`,
gradient norm `1.1172`, log-prob difference `0.27250`, and finite loss. No
runtime or guard failure was present. This records launch only; no checkpoint
or quality result exists yet.

### 2026-07-07 - Response-Length Guard Fix And Hero Retry 2

Hero retry 1 stopped at rollout 26 because the relative response-p95 guard
observed four consecutive increases from `160.85` to `170.55`. The value was
well below the 256-token cap, truncation was zero, and last completed actor
step 25 remained healthy (KL `3.67496`, grad `4.5625`, log-prob difference
`0.24050`). This was a guard-semantics false positive, not model or runtime
failure. No checkpoint was written.

Commit `1087f5f` replaces only that rule with sustained absolute limits: p95
above 230 for two rollouts and truncation above 5% for two rollouts. All actor,
parser, provenance, checkpoint, and promotion gates remain unchanged. RunAI
tests passed (`60 passed`, two historical live-status assertions deselected),
and dry-run found no missing inputs.

The clean restart
`r33-corrected-k3-hero-lr1e5-update342-guard3-lengthcap-retry2` began at
`2026-07-07T00:13:25Z`. Rollout 0 had p95 `168.85`, zero truncation, and
healthy parsing; optimizer step 0 completed with KL `0`, gradient norm
`1.08594`, and log-prob difference `0.27939`. Four offline W&B role runs are
active and no error or guard event is present. This is launch evidence only.

### 2026-07-07 - Hero Retry 2 Stop And Update-228 Continuation

Retry 2 stopped at rollout 253 after response-length p95 exceeded the
230-token abort threshold twice (`233.85`, `232.70`). The stop was not a
training or parsing failure: rollout 252 was `99.22%` closed/usable with zero
truncation, and actor step 252 had KL `3.01199`, gradient norm `1.46875`, and
log-prob difference `0.27546`. No post-eval ran. Update 228 is the latest
durable model checkpoint.

The continuation queue removes only the response-p95 abort and preserves the
truncation, parser, KL, gradient, and log-prob guards. Since optimizer state
was intentionally not saved, retry 3 restarts Adam but restores update-228
model weights, RNG, rollout counter, and dataset state 227. It keeps
`num_rollout=342` as the absolute endpoint and the constant `1e-5`/K3 recipe.

Run `r33-corrected-k3-hero-lr1e5-update342-resume228-retry3` started at
`2026-07-07T17:52:05Z`. Preflight found no missing paths. The checkpoint and
dataset state loaded successfully on RunAI, and rollout/optimizer step 228
completed with reward mean `-0.20660`, p95 `222`, zero truncation, loss
`0.00092021`, KL `0.92023`, gradient norm `0.6875`, and log-prob difference
`0.27336`. The run is active with offline W&B logging and no errors. Hero
promotion still requires the configured 64/64 and 512/512 round-trip gates.

### 2026-07-08 - R33 RL Hero Completes And Passes 512/512 Promotion

The resume-228 retry completed actor step 341 and wrote model-only checkpoint
`iter_0000342`. Training ended at `2026-07-07T23:17:47Z`; the queue and both
post-evals completed at `2026-07-08T03:20:52Z`. Final actor loss was
`0.00183662`, K3 KL `1.836525`, gradient norm `0.886719`, actor/rollout
log-probability difference `0.311984`, and entropy `0.789943`. The last rollout
had reward mean `-0.226492`, response p95 `226`, zero truncation, and
`99.74%` closed/usable output.

The `64/64` prerequisite gate passed with validation/test relative improvement
over matched SFT of `25.01% / 28.32%`. The full `512/512` gate also passed:

| Metric | Validation | Test |
|---|---:|---:|
| RL AV-real NMSE | `0.0000875281` | `0.0000911757` |
| Matched SFT NMSE | `0.0001267961` | `0.0001347520` |
| Relative improvement | `30.97%` | `32.34%` |
| RL rowwise wins | `427/512` | `454/512` |
| Clustered improvement CI95 | `[0.0000321424, 0.0000472365]` | `[0.0000360316, 0.0000519528]` |
| Closed/usable | `99.02% / 100%` | `99.41% / 100%` |

Each split contained 52 independent documents. Dataset hashes, exact row
identities, and all 512 overlaps matched the clean baseline. AV-real beat the
teacher variant and every generated control by aggregate normalized MSE. The
top five improving rows accounted for only `6.74% / 6.30%` of the net delta.

The only terminal traceback was an ignored W&B atexit `BrokenPipeError` after
the successful trainer exit. Temporary HF exports were cleaned; the selected
checkpoint and offline logs remain. Lightweight final evidence was copied
locally and to S3 with archive SHA-256
`78cbf98d27188594c25cbf9c0d695f0b3b1754df978961585bbaa6fc178f0bc7`.

### 2026-07-08 - Publication Remediation: Content-Family Holdout Audit

The external publication audit invalidated the archived `30.97% / 32.34%`
headline because its SFT baseline mixed generation protocols and because the
document-level bootstrap did not account for near-duplicate content families.
The remediation implementation now separates directional reconstruction from
raw-space recovery, binds cached generation to an exact protocol hash, and
supports family-level paired bootstrap and sign-flip inference.

A deterministic source-content audit was run over all `275,396` unpadded R33
AV rows. Unicode-normalized five-token shingles, bottom-k candidate indexing,
exact Jaccard threshold `0.80`, and union-find produced `5,384` content
families from `27,647` documents. The old AV train/validation/test split has
`258` cross-split family overlaps, confirming that document IDs were not a
sufficient independence boundary.

Exposure was defined conservatively as the union of the selected R33 AV-SFT
train split, R33 AR-SFT train split, and exact RL train parquet. After removing
all exposed families, then dropping the two remaining families shared by the
candidate validation and test holdouts, the untouched evaluation pool is:

| Split | Eligible rows | Independent families | Exposed rows removed | Cross-holdout rows removed |
|---|---:|---:|---:|---:|
| Validation | `11,007` | `214` | `2,912` | `80` |
| Test | `10,915` | `218` | `2,752` | `30` |

The final validation/test family overlap is zero and both pools exceed the
required `512` rows and `100` independent families. Therefore the clean R33
SFT checkpoints can be retained for corrected evaluation; family leakage does
not by itself require SFT retraining. No corrected SFT-vs-RL effect size has
been claimed yet. The next evidence step is protocol-matched generation and
family-stratified `512/512` scoring.

Artifacts:

- RunAI: `/workspace/interp/outputs/nano30b-nla-pilot/publication/r33_content_families/`
- Local: `artifacts/runai_evidence/20260708_r33_publication_remediation/content_families/`
- S3: `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/validity/publication/r33_content_families/`

### 2026-07-08 - Publication Remediation: Runtime Provenance And R33 Activation Fidelity

Task 5 hardened publication launches and datasets against silent provenance
downgrades. Publication RL data builds now require exact component filtering,
the content-family manifest and coverage report, the declared row count, and
zero heldout family/doc overlap. The verifier checks those conditions against
parquet metadata and sidecar hashes. Launches now freeze a redacted queue
snapshot and resolved spec before execution, reject semantic mutation, and
fingerprint the project, complete Miles tree, all Miles patches, critical
files, dataset artifacts, Python environment, and container image digest.

The activation diagnostic was extended to bind the generated rows, source and
mean parquets, family manifest, extraction code, model fingerprint, boundary,
dtype, seed, and exact sample identities. It reports absolute/relative L2,
cosine and norm agreement, raw MSE, centered R2, directional MSE, repeatability,
and a fail-closed identity assessment. A missing `torch.no_grad()` around the
manual extraction forward initially caused an OOM after two successful full
forwards; the diagnostic now uses the same no-grad contract as production
extraction.

Final current-runtime audit:

| Field | Result |
|---|---:|
| Sample | `8` validation + `8` test, family-stratified seed `20260708` |
| Base-model fingerprint | `abd6d1368f9d2baa1b6f5b4047916db780466193af85b4772bbf5dc64c218019` |
| Full-forward repeat max relative L2 | `0.0` |
| Current full vs extraction max relative L2 | `0.0` |
| Full-forward vs stored mean/max relative L2 | `0.023913 / 0.053206` |
| Full-forward vs stored cosine mean/min | `0.999643 / 0.998585` |
| Stored-as-prediction raw MSE | `0.00859186` |
| Stored-as-prediction centered R2 | `0.998807` |
| Stored-as-prediction directional MSE | `2.65418e-7` |
| Exact-source original-geometry max relative L2 | `0.073748` |
| Identity tolerance violations | `16 / 16` |
| Publication ready | `false` |

The AR-SFT vectors are byte-identical to the original R33 extraction parquet
for all `16` sampled rows (`max_abs=0`), ruling out merge/postprocessing
corruption. Reconstructing the exact R27 source batching and testing both the
current patched model code and the preserved pre-patch HF snapshot did not
recover stored identity. The June 12 model-code patch is therefore a provenance
break but not the sole explanation. The remaining extraction-era runtime is
not sufficiently pinned to support a publication claim.

Decision: retain the old R33 SFT/RL checkpoints as internal historical
evidence only. Before publishable retraining, re-extract R33 activations from
the teacher-backed rows under one frozen model/runtime contract, require the
live-vs-stored fidelity gate to pass, then retrain clean AR and AV SFT.

Evidence:

- RunAI: `/workspace/interp/outputs/nano30b-nla-pilot/publication/r33_activation_fidelity/`
- Local: `artifacts/runai_evidence/20260708_r33_publication_remediation/activation_fidelity/`
- S3: `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/validity/publication/r33_activation_fidelity/`
- Tests: local and RunAI `74 passed` on the complete Task 5 suite.

### 2026-07-09 - Publication Remediation: Clean Family Splits And Transfer-Critic Initialization

Commit `d24a941` closes two remaining pre-launch correctness gaps. First, the
clean AR and AV configs no longer run separate exact-duplicate
`content_component` splits. The materializer now accepts the frozen
shingle-based `nano_content_family_manifest.v1` assignment, requires every
document to have a family and every family to have a predeclared split, binds
the manifest SHA-256 into the cache and split report, and fails on any family
overlap or seed/fraction mismatch. Primary AR, independent AR, and AV now share
that exact manifest and use validation-only checkpoint evaluation.

Second, the old R33 "independent" critic is correctly classified as a reseeded
training run because it reused the identity-initialized checkpoint. The clean
publication path now predeclares a separate seed-`314159` critic initializer:

- value head: seed-specific block-Givens orthogonal rotation, `0.2` radians;
- MoE routers: seed-specific relative Gaussian perturbation, `0.01` of each
  parameter's RMS, preserving the pretrained routing prior;
- all initialization modes, seeds, parameter counts, and before/after hashes
  are written to `critic_initialization.json`;
- Megatron compatibility weights copy the exact initialized value head rather
  than silently replacing it with identity;
- `verify_nano_critic_initialization.py` requires shared base/data/layer/dtype
  provenance, matching pre-perturbation router hashes, and distinct final head
  and router hashes;
- the independent AR queue refuses to prepare unless that passing report is
  hash-bound to the exact selected critic manifest.

This is a separately initialized transfer critic over the same pretrained
Nano backbone and training corpus. It should not be described as statistically
independent data or an unrelated architecture.

Verification used an immutable source snapshot on RunAI without changing the
active salvage evaluator. The focused Torch/config suite passed `71/71`; the
broader protocol, metrics, family, provenance, HPO-isolation, queue, and
publication-config suite passed `166/166`. Local config-only suites also pass.
Clean extraction, critic initialization, and SFT training have not started.

In parallel, the protocol-matched retained-hero salvage generation remained
healthy on eight H100 NVLs. At the latest milestone poll it had written
`356/1,024` rows, used about `64.2 GB` per GPU at roughly `40-43%` utilization,
logged no errors, and left `342 GB` free on `/workspace/interp`. Its result is
exploratory regardless of outcome because the stored-activation fidelity gate
failed and clean retraining is mandatory.

### 2026-07-09 - Publication Remediation: Unopened Confirmatory Family Split

The exploratory salvage evaluation had already touched families from the
first publication manifest, so that manifest's nominal test assignment could
not serve as a one-shot confirmatory test. Commit `3e3633a` adds a one-time,
non-overwriting split freezer and changes all clean AR/AV configs to consume a
separate confirmatory manifest.

The freezer resolved every retained `*/splits/validation.parquet` and
`*/splits/test.parquet` under the project output root before clean training,
then recorded each resolved path, parquet SHA-256, row/document/family counts,
and the base family-manifest hash. It found `114` prior evaluation sources.
`1,018` document IDs belonged to other corpora and were recorded as unmapped;
the remaining sources mapped to `2,793` R33 families, all of which are
forbidden from the new test split. Families are balanced only over allowed
splits, and the builder refuses to overwrite a frozen output.

Frozen split result:

| Split | Rows | Families |
|---|---:|---:|
| Train | `247,847` | `4,834` |
| Validation | `13,786` | `264` |
| Test | `13,763` | `286` |

All `275,396` rows and `5,384` families are assigned exactly once. Train,
validation, and test family overlaps are zero, and no test family appears in
the prior-evaluation forbidden set. Manifest SHA-256:
`837a5f26af78eb3dbc190859f62b36aba86b51b564cff797698f97848623c69d`.

Artifacts:

- RunAI: `/workspace/interp/outputs/nano30b-nla-pilot/publication/r33_confirmatory_families/`
- Local: `artifacts/runai_evidence/20260709_r33_publication_remediation/confirmatory_families/`
- S3: `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/validity/publication/r33_confirmatory_families/`
- Verification: `69/69` focused family/materializer/config tests passed on
  RunAI; the production report has `passed=true`.

### 2026-07-09 - Publication Remediation: Resumable Eight-GPU Clean Extraction Path

Commit `640bb51` adds a reusable data-parallel exact-prefix extractor for the
mandatory clean R33 activation rebuild. The prior extractor loaded one model
with `device_map=auto` and processed the selected rows serially. The new
orchestrator assigns balanced contiguous source-row ranges to explicit GPU
devices, launches one model replica per GPU, preserves per-shard logs and
partial outputs, resumes only verifier-complete shards, and merges shards in
source-range order.

A production-source CPU preflight found that naïve equal-row boundaries would
split a document at all seven boundaries. The planner now cuts only between
documents and aligns each non-final shard to the configured two-document
extractor batch. This preserves the serial extractor's document pairing and
removes an avoidable batch-geometry difference. For the frozen 275,396-row
table, the resulting eight shards contain `34,419-34,431` rows and
`3,454-3,458` complete documents each; all non-final document counts are even.

The merged parquet is published atomically and only after every shard agrees
on schema, row count, and publication provenance. The merged YAML/JSON
sidecars retain the frozen model/runtime fingerprints and record the exact
ordered shard plan. Duplicate devices, out-of-range selections, partial
metadata, schema drift, provenance drift, and incorrect aggregate row counts
all fail closed. The extraction configs select devices `0..7`; the same path
remains reusable for other layer specifications and smaller device lists.

Verification:

- local focused extraction/config suite: `23/23` passed;
- RunAI focused suite under `/workspace/interp/.venv`: `23/23` passed;
- RunAI full repository suite: `582` tests collected and exited successfully;
- shell syntax, Python compilation, and `git diff --check` passed.

This milestone proves the orchestration code, not the dataset. The production
runtime fingerprint, 275,396-row extraction, activation-fidelity rerun,
AR/AV dataset verifier reports, and clean SFT runs remain pending. The active
protocol-matched salvage evaluator was left on its immutable source snapshot
and was not interrupted.

Storage preflight measured the retained final payloads at `72 GB` primary AR,
`72 GB` independent AR, and `59 GB` AV. Adding clean replacements plus two
roughly model-sized critic initializers to the `341 GB` free Longhorn volume
would leave no safe margin. Commit `4d080e8` therefore keeps datasets,
verifier reports, and studies on `/workspace/interp`, while placing both clean
critic initializers and all three clean SFT output roots under
`/workspace/models/nano30b-nla-pilot/publication/`. This is a storage-only
configuration change; model, data, optimizer, and evaluation settings are
unchanged. The RunAI config/runner regression shard passed `67/67`.

The frozen confirmatory train assignment has `247,847` rows. Under
`pad_with_train_duplicates` and global batch `192`, the runner materializes
`247,872` rows and exactly `1,291` optimizer updates. The inherited AR configs
still saved/evaluated at update `1,289`; commit `c7917f5` corrects both primary
and independent AR to update `1,291`, matching AV and the queue's expected
final checkpoint.

Publication eval configs also incorrectly resolved through the historical
`hero-current` symlink. That snapshot lacks the strict parse/length-control
configs and differs from the audited evaluator code; queue dry-run had a local
fallback that the real launcher does not. Commit `30b8337` binds every
publication round-trip config and queue to `publication-current` and adds a
regression test forbidding the historical source root.

The resulting immutable source snapshot is Git commit
`30b8337a311bf0bf83deb4162ff08f023b0bd0c3`, S3 object
`code-sync/nano30b-nla-pilot-publication-30b8337.tgz`, and archive SHA-256
`8d04270bb6009c010d2d63f65fa67aa28cbbf54648d88a18ef840468543a0e58`.
The remote archive hash matches the external source-snapshot manifest. The
runtime freezer now includes both files as critical inputs, so the eventual
runtime report will connect the extracted source tree to GitHub and S3 even
though the RunAI snapshot intentionally has no `.git` directory. The final
RunAI snapshot passed the complete test suite; the focused publication-eval
shard passed `62/62`.

### 2026-07-09 - Corrected Retained-Hero Salvage Pass And Frozen Clean Runtime

The protocol-matched retained R33 SFT/hero evaluation completed on 512
validation and 512 test rows spanning 214/218 content families. Both cached
generation files use protocol SHA-256
`5677d491a812baecb9fa21829866de7eb3750f7cdb0273ed3038123a3119a381`
(`max_new_tokens=256`, no forced prefix, deterministic legacy full-prefix
generation, eight workers, injection scale 75). The SFT and hero dataset/row
identities match exactly.

Primary-critic results:

| Split | SFT directional MSE | Hero directional MSE | Relative gain | SFT raw MSE | Hero raw MSE | Relative gain | Row wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | `0.0001136723` | `0.0000879586` | `22.62%` | `9.5477` | `7.3552` | `22.96%` | `86.52%` |
| Test | `0.0001157334` | `0.0000922163` | `20.32%` | `9.6224` | `7.6986` | `19.99%` | `83.20%` |

The family-clustered 95% intervals for baseline-minus-hero directional MSE
are `[2.304e-5, 2.865e-5]` validation and `[2.082e-5, 2.665e-5]` test. Raw-MSE
intervals are `[2.017, 2.418]` and `[1.733, 2.146]`. Hero real text beats all
declared shuffled/zero/mean/no-injection controls. Real close/usable rates are
`100%/100%` validation and `99.61%/100%` test. The declared primary gate
passes.

This is salvage evidence, not a publication-valid model result. The report's
`publication_status=confirmatory` denotes only that its family count clears
the inference-size threshold; it does not override the failed stored-
activation fidelity audit, exploratory test exposure, or same-lineage critic.
Clean extraction and retraining remain mandatory. Independent-critic and
length-controlled rescoring is running from the same cached text.

The clean extraction runtime is now frozen and passed:

- source commit: `30b8337a311bf0bf83deb4162ff08f023b0bd0c3`;
- source archive SHA-256:
  `8d04270bb6009c010d2d63f65fa67aa28cbbf54648d88a18ef840468543a0e58`;
- source-tree SHA-256:
  `23687130b428ff86a5294b04108f32a5868fd642488be1bb27f277effe105a66`;
- runtime SHA-256:
  `56351d564488528d75dc1c9c6961219cfa4d7a15a76273433b2212ad2b4cac87`;
- base-model SHA-256:
  `abd6d1368f9d2baa1b6f5b4047916db780466193af85b4772bbf5dc64c218019`;
- container digest:
  `sha256:3c90e38f5ec51e51d1c73bd7eb3d83674a254f451147c5cadc4344314258a112`.

The runtime report includes 569 Miles files, 19 Miles-patch files, the source
archive and manifest, extraction config, teacher contract, and confirmatory
family manifest/report as hashed critical inputs. Evidence is mirrored under
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/validity/publication/` and
`artifacts/runai_evidence/20260709_r33_publication_remediation/`.

### 2026-07-09 - Retained R33 Cross-Critic And Length-Control Salvage Gate

The corrected retained-checkpoint evaluation finished on the same cached,
protocol-matched SFT and update-342 explanations. A tokenizer compatibility
bug initially rejected Hugging Face `BatchEncoding` objects during the length
audit. Commit `949f5c4` generalized the tokenizer adapter to mappings,
attribute-backed `input_ids`, tensor-like values, tuples, and singleton batch
dimensions; the RunAI evaluator suite passed `63/63` before the failed scorer
was retried. The failed log remains archived rather than rewritten.

The retained reseed critic produced the following paired effects:

| Split | SFT directional MSE | Hero directional MSE | Relative gain | Raw-MSE gain | Row wins | Best length-matched gain |
|---|---:|---:|---:|---:|---:|---:|
| Validation | `0.0001136269` | `0.0000879207` | `22.62%` | `22.66%` | `85.74%` | `15.38%` |
| Test | `0.0001156682` | `0.0000926635` | `19.89%` | `19.17%` | `83.59%` | `13.42%` |

The family-clustered directional intervals are strictly positive:
`[2.319e-5, 2.867e-5]` validation and `[2.036e-5, 2.615e-5]` test. Parse
health passes at `100%/100%` closed/usable validation and
`99.61%/99.61%` test. The aggregate cross-critic gate passes every declared
identity, protocol, baseline-binding, family-CI, row-win, parse, and
length-control check. Independent-to-primary gain transfer is `0.9993`
validation and `0.9793` test, above the predeclared `0.75` floor.

This does not repair publication validity. The critic is a retained reseed,
not the new independently initialized critic; stored R33 activations failed
the live identity audit; and these exploratory test rows were already exposed
during development. The scientific decision is therefore to treat update 342
as positive salvage evidence while proceeding with frozen-runtime extraction,
clean family-disjoint AR/AV SFT, a fresh critic, and an independent RL
replication. Evidence is mirrored at
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/validity/publication/r33_protocol_matched_salvage/`
and locally under
`artifacts/runai_evidence/20260709_r33_publication_remediation/r33_protocol_matched_salvage/`.

### 2026-07-09 - Deterministic R33 Full Dataset And Exact Replication Gate

The first `r33_frozen_runtime_full275396` extraction is invalidated for
publication use. Although its row, dimension, text, and family verifiers
passed, independent production-path forwards were not reproducible under its
unbound numerical execution profile. It must not be used for clean SFT.

Commit `0dabaad` introduced one shared, config-driven extraction profile:

- PyTorch deterministic algorithms enabled;
- TF32 disabled;
- cuDNN benchmarking disabled;
- float32 matmul precision `highest`;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- seed `20260709` applied to Python, NumPy, and Torch.

Publication configs fail closed if this profile is absent or weakened. The
profile is embedded in every shard sidecar and in activation-fidelity
manifests. The frozen source/runtime identities are:

- Git commit: `0dabaade33ee35a3ff7419d2f99be2551439ab13`;
- source archive SHA-256:
  `cbbde5ee91c2513f69a5abf8a5c57d0c5bcb9a9c905bd473e5d4b71e2ad58e27`;
- source-tree SHA-256:
  `a79128f8c479620b43df9e69b13f9e24d678bfecf7a6484d4823c4e188aa43b6`;
- runtime SHA-256:
  `1b7ca243028de224a06702c4bd3e1e2d3d9f75fe84f19b24610c146d66d70ff1`;
- base-model SHA-256:
  `abd6d1368f9d2baa1b6f5b4047916db780466193af85b4772bbf5dc64c218019`.

The replacement eight-H100 extraction took about 17 minutes. Its primary
outputs all have `275,396` rows:

- `base_R33_r33_deterministic_full275396.parquet`;
- `ar_sft_R33_r33_deterministic_full275396.parquet`;
- `av_sft_R33_r33_deterministic_full275396.parquet`.

Both AR and AV verifier reports pass with `d_model=2688`, zero nonfinite
activations, zero empty explanations, zero malformed AV responses, and zero
bad prompt/suffix rows. The refined content-family manifest covers all
`27,647` documents and `5,009` families. Train/validation/test contain
`247,865 / 13,766 / 13,765` rows and `4,504 / 250 / 255` families, with zero
document, family, or exact-prefix content overlap. Manifest SHA-256 is
`479cbab5d21cd031cb72a770eebb3428e0d5419ebf8cce38c2ca6025e49741b6`.

A full eight-shard replay produced a byte-identical merged Parquet with
SHA-256
`e3008a150831b8e894eac0de9f360a46823ffbfbd7cc73a9673f7e61e84521ac`.
That result proves deterministic snapshot extraction only. The later archived
32-row live-vs-stored check failed the predeclared fidelity tolerance on all
rows, with `2.562%` mean and `7.728%` maximum relative L2 drift and
`publication_ready=false`. The earlier stateful-shard explanation and
unarchived shard-start-canary claim are withdrawn. Downstream paired results
may use the same stored target symmetrically, but this lineage does not prove
exact equality to a fresh model forward.

The deterministic primary R33 critic init exists at
`/workspace/models/nano30b-nla-pilot/publication/r33_deterministic_critic_init`.
No AR, AV, or RL training was launched in this milestone. Lightweight evidence
is mirrored at
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/validity/publication/r33_deterministic_full275396/`
and locally under
`artifacts/runai_evidence/20260709_r33_publication_remediation/r33_deterministic_full275396/`.

### 2026-07-09 - Independent Critic And Clean SFT Queue Readiness

The publication transfer critic was initialized from the same deterministic
R33 data sidecar with a genuinely distinct seed-`314159` head/router start:

- value head: seeded Givens rotation, `0.2` radians;
- router: seeded relative noise, standard deviation `0.01`;
- model path:
  `/workspace/models/nano30b-nla-pilot/publication/r33_deterministic_independent_critic_init_seed314159`.

`verify_nano_critic_initialization.py` reports `passed=true`. It confirms the
same base model, dataset sidecar, R33 extraction layer, and bf16 dtype while
also proving a distinct value-head hash and changed independent router
parameters. The primary critic retains its identity head and pretrained
router. Independent-critic evidence is copied locally under
`artifacts/runai_evidence/20260709_r33_publication_remediation/independent_critic/`;
its archive SHA-256 is
`fd0878d43af9ad5976aa6fedc341d301340b5148273177e54e819a4f1a8e87a7`.

Later evidence audit correction: this archive preserves the initialization
manifest and component-level checks, not a complete usable model copy. The
independent checkpoint's full-file hash stopped before all `38.4 GB` were
covered, the local transfer contains incomplete rsync temporary files, and the
cluster payload was deleted. It must be rebuilt and fully fingerprinted before
independent-critic training; the metadata archive alone is not a checkpoint.
This statement applies to that incomplete July 9 copy. It is superseded by the
fully fingerprinted seed-`314159` rebuild, completed independent AR training,
and cross-critic validation documented in the July 16 publication follow-up.

The clean queue-readiness code/config snapshot is commit
`294cc1e1619c42ea454c8d29e9f477e7ae4d4322`. The queue-readiness evidence was
generated from its 588-file tree fingerprint,
`1c7c0abbad68bebc426acff74ea1b14ed1559adacda28496d2b526154032c020`.
Later launch-critical fixes are not ancestors of this commit, so it must not
be treated as the immutable source of future launches. The extraction-time
code fingerprint was
`a79128f8c479620b43df9e69b13f9e24d678bfecf7a6484d4823c4e188aa43b6`
and each new queue launch must bind its own resolved source fingerprint.
Later documentation-only commits may advance branch HEAD without changing the
hash-pinned launch files; the final Mac/RunAI tree is still compared directly
at sync time.

The AV queue dry-run surfaced two launch-safety defects before training:

1. `nano_av_probe_queue.py` did not support `--dry-run`.
2. Its first dry-run implementation returned `Path` objects through plain
   `json.dumps`, causing a final serialization error after preparation.

Commits `4e23755` and `294cc1e` fixed these through one shared AV planning
path for train, DCP-to-HF conversion, evaluation, and optional round-trip
commands, plus JSON-safe output. Both failure logs remain in the readiness
evidence. Test-first coverage reproduced each defect; the final RunAI suite
passes `65/65` tests.

All three clean queue contracts now dry-run successfully and remain pending:

| Queue | Training recipe | Expected checkpoint | Selection |
|---|---|---|---|
| Primary AR | `2 GPU`, `gb192/mb96`, `lr=5e-5`, cosine, warmup 25 | `iter_0001291` | validation-only `512` |
| Independent AR | same recipe, seed-`314159` critic | `iter_0001291` | validation-only `512` |
| AV | `2 GPU`, `gb192/mb2`, `lr=1e-4`, cosine, warmup 25, dynamic 512-token cap | `iter_0001291` | validation-only `512` |

No AR, AV, or RL training process was launched. All eight H100s were idle at
`4 MiB` during final verification. The complete readiness bundle is local at
`artifacts/runai_evidence/20260709_r33_publication_remediation/clean_sft_queue_readiness/`
with archive SHA-256
`f6045b10e1e4573635c00cd49418137870b9309a2323faeb82db9e3adcb85c3a`.
RunAI S3 upload is pending because
`egress-proxy.egress-proxy.svc.cluster.local:3128` is unavailable; the archive
and source package remain preserved on RunAI and locally with exact hashes.

### 2026-07-09 - Clean R33 AR Launch Diagnostics And H100 Topology Correction

The first clean primary AR launch used the readiness recipe (`2 GPU`,
`gb192/mb96`). It did not reach an optimizer step. Rank 1 timed out after the
Miles default 600-second c10d wait while rank 0 was still faulting the cold
36 GB critic checkpoint from `/workspace/models`. Rank 0 remained alive and
its physical read counter advanced throughout, so this was load skew rather
than an NCCL fabric failure. Commit `658a2ef` exposes Miles'
`distributed_timeout_minutes` through the shared AR/AV spec renderer, validates
that it is positive, and sets all three clean SFT specs to 60 minutes. RunAI
tests passed `51/51` before retry. The original queue state, train log, and
offline W&B directories are retained.

The timeout-corrected retry reached step 0 and passed the reward/train MSE
equivalence check (`mean=1.0000`, `max|r-1|=0.0000`, rank-local `n=32`). It
then failed in FSDP `post_backward` while converting the reduction output:
GPU 1 held `91.99 GiB`, had `1.10 GiB` free, and requested another
`1.21 GiB`. No checkpoint was produced. This reproduces the already documented
H100-NVL limitation of the historical 2-GPU `mb96` geometry; it is not a data,
loss, or LR failure.

The controlled retry changes only H100 execution geometry and run identity:

- primary and independent AR: `4 GPU`, `gb192/mb48`;
- unchanged dataset/family split, critic initializations, LR `5e-5`, cosine
  schedule, warmup 25, one epoch, validation-only 512-row selection, and
  optimizer-state checkpoint policy;
- AV initially used `2 GPU`, `gb192/mb2`, dynamic 512-token cap. It completed
  all 48 packed microbatches in its first actor pass (`465.1s`) but OOMed before
  the optimizer-step log while allocating another `20 MiB`: `91.68 GiB` was
  allocated, only `255 MiB` was reserved-unused, and about `7 MiB` was free.
  This is a true H100 capacity limit, not allocator fragmentation. No checkpoint
  was produced.
- the fresh AV retry is `4 GPU`, exact `gb192/mb2`, and the same dynamic
  512-token cap, LR, schedule, dataset, and validation protocol. The extra FSDP
  shards reduce parameter/optimizer residency and halve rank-local packed
  microbatches without changing the global batch;
- launch order is primary AR on GPUs 0-3 and AV on GPUs 4-7, then independent
  AR on GPUs 0-3 after primary releases them.

The first 4-GPU primary AR retry was memory-safe (roughly `56-61 GiB` used per
rank) and trained normally through step 191. Its final logged point had loss
`0.323341`, FVE `0.421504`, finite gradients, and no numerical warning. The
next update hit the previously documented asynchronous CUDA illegal-address
race. The first synchronous frame was the router-entropy hook's GPU-to-CPU
copy; prior no-router and expert-scan-only diagnostics had already shown that
such frames are synchronization points rather than root causes. No checkpoint
was produced.

The restart therefore adopts the already validated safety recipe rather than
repeating disproven toggles: `moe_routing_impl: expert_scan` plus
`cuda_launch_blocking: true`. The earlier bounded diagnostic crossed the same
failure boundary only when launch blocking was enabled, and its continuation
completed all 1,289 independent-critic updates. Expert-scan had separately
matched hidden states and hidden/router/expert gradients. The same pair is
applied to primary AR, independent AR, and AV because all train the same
Nemotron-H backbone on H100-NVL. Fresh run IDs isolate every failed attempt.

The failed AR `mb96` and AV 2-GPU run directories and logs remain
diagnostic-only. Fresh topology-specific run IDs prevent accidental resume or
metric mixing. No RL launch is part of this clean SFT milestone.

### 2026-07-09 - Synchronized Mamba Kernel Fault And Selective Safe Path

The fresh 4-GPU primary AR run with both `expert_scan` and
`CUDA_LAUNCH_BLOCKING=1` trained through step 191, then failed synchronously on
rollout 192. The final valid point was loss `0.3241694`, FVE `0.4200212`,
gradient norm `0.4765625`, and LR `4.80954e-5`. The traceback now localizes the
fault inside the Nemotron-H Mamba training fast path:

`mamba_split_conv1d_scan_combined -> causal_conv1d_fwd -> CUDA illegal memory access`.

This supersedes the earlier router/MoE-frame attribution. Direct inspection of
the live AV Ray actor environment also proved that
`NLA_MOE_ROUTING_IMPL=expert_scan`, `CUDA_LAUNCH_BLOCKING=1`, allocator, and
system-metric variables were already inherited by workers. Missing Ray worker
environment was therefore not the cause. The renderer now still passes the
same allowlisted configuration explicitly through Miles `--train-env-vars`, so
worker propagation is deterministic and cannot depend on local-Ray inheritance.

The next AR attempt is guarded by a config-driven Mamba kernel mode. Its first
candidate, `unfused_torch_conv`, disables only Nemotron-H's fused training
branch, substitutes the upstream-defined grouped PyTorch causal-convolution
equivalent, and retains the efficient `mamba_chunk_scan_combined` scan. The
context covers both forward and gradient-checkpoint recomputation. A full
`torch` Mamba path remains available as a slower fallback. Primary and
independent clean AR specs use fresh run identities and the selective safe
path.

The four-update no-checkpoint capacity probe passed on the original
`4 GPU / gb192 / mb48` geometry and exited with code 0. All four optimizer
updates were finite; step 3 logged loss `1.1304802`, FVE `-1.0225676`, gradient
norm `10.125`, and LR `8e-6`. The live reward/train equivalence check remained
exact (`mean=1.0000`, `max|r-1|=0.0000`). Peak allocated/reserved memory was
approximately `45.05/54.08 GiB` per H100, leaving about 38 GiB free, and
steady updates took roughly 9-17 seconds. A W&B `BrokenPipeError` occurred only
during Ray actor teardown after step 3; the driver exit status was 0 and it is
not a training failure.

Commit `8d9917a` passed `78` RunAI tests with two expected skips and was synced
through S3 with archive SHA-256
`d4ef1c23e52af0823214e45602bd0cbef5dc72fe9e4d1f9263440b4d1683c5ed`.
The full primary AR queue transitioned to `training` at
`2026-07-09T21:02:46Z` on GPUs 0-3. Its target is model/optimizer DCP
`iter_0001291`, followed by the configured validation-only 512-row bounded
evaluation. No success is claimed until the checkpoint and verifier/eval
report exist.

The full run reproduced the capacity probe and began optimizer updates. A
matched comparison against the failed fused-kernel run over the first 24
shared steps found maximum absolute loss/FVE differences of only
`0.0024314 / 0.0043501`, identical LR, and step-23 differences of just
`1.35e-5 / 2.41e-5`. Gradient-norm differences were at BF16 quantization
increments. At step 30 the selective run had loss `0.4479342`, FVE
`0.1985908`, finite gradient norm `0.59375`, and zero CUDA/OOM/traceback
events. This is trajectory evidence, not a final quality claim.

The 4-GPU AV run remains active because its actors demonstrably have the
configured safeguards and its optimizer steps are healthy. At the last check,
step 26 had loss `1.04451`, LR `9.99994e-5`, approximately 79 GiB/GPU, and no
OOM or CUDA error. It is not yet a completed or selected checkpoint.

### 2026-07-09 - Packed AV Fail-Closed Gate, Root Cause, And Corrected Throughput

The preceding AV status is superseded. The `dyn512` full run was stopped at
about step 32 without a checkpoint because its projected runtime was roughly
6.7 days. More importantly, a newly added fail-closed packed-vs-padded actor
gate found that the then-current packed forward was not mathematically
equivalent to the padded reference before any optimizer step. The first
guarded `dyn2048` attempt reported packed/padded mean response NLL
`2.58670735 / 2.86806154`, maximum absolute difference `0.94716144`, and
maximum relative difference `0.34476781`; it exited by design with no update.

The first diagnosis was incomplete. The custom Nemotron-H causal mask lacked
sample-boundary isolation for attention layers, but adding a `seq_idx` block
mask exposed an ordering error: the model constructed the causal mask before
deriving `seq_idx`. Commit `1cbfd5c` moved boundary derivation before mask
construction and added a validation invariant. After that repair, the gate
still failed at `2.58670735 / 2.86806154`, with maximum absolute/relative
differences `0.92314470 / 0.33880699`.

The decisive root cause was one level higher. `NemotronHForCausalLM.forward`
accepted packed `position_ids` but did not pass them to `NemotronHModel`.
Consequently, position resets never reached either the Mamba `seq_idx` kernels
or the attention boundary mask; the packed stream was still interpreted as
one long sequence. Commit `6abfe18` patches and validates this outer-forward
contract. The three runtime model-code copies were rebuilt from preserved
pre-patch backups, compile successfully, converge to SHA-256
`dd5f82b9697307694d8e29b68066779b4c23a9e2a02095475a904bf6530f1e41`,
and report `changed=false` on a second patch pass. RunAI validation passed
`90` tests with two expected skips.

The corrected `dyn2048` probe then passed the live gate:

- packed/padded mean NLL: `2.87655449 / 2.87388682`;
- maximum absolute/relative difference: `0.01182747 / 0.00395436`;
- optimizer-step losses: `2.7461059`, `2.7297325`;
- steady step-1 time: `131.72s`;
- post-Adam peak allocated/reserved: `69.28 / 77.27 GiB`;
- process exit status: `0` (the teardown-only W&B broken pipe remains benign).

A second guarded probe promoted the packed-token cap to `4096`. It also
passed before training, with packed/padded mean NLL
`2.21664405 / 2.22232628` and maximum absolute/relative difference
`0.01632524 / 0.00704313`. Step losses were `2.7454975 / 2.7284813`; steady
step 1 took `67.33s`, with post-Adam peak allocated/reserved
`70.45 / 78.24 GiB` and `13.74 GiB` CUDA free. The process exited `0`.
This is about `2x` faster than corrected `dyn2048` and about `3.9x` faster
than the earlier throughput-only `dyn1024` probe. The latter had no live
equivalence gate and is retained for timing only.

Publication consequence: every earlier AV or RL actor checkpoint trained by
packing multiple rows through `NemotronHForCausalLM` before `6abfe18` is not a
clean publication checkpoint, even if its later padded eval looked good. The
current clean AR critic run is unaffected by this newly isolated wrapper bug:
the critic calls the backbone directly with explicit position IDs and uses
padded masked microbatches.

The promoted AV hero candidate is now config-driven at
`configs/nano_av/publication/r33_family_clean_sft_8gpu_dyn4096.yaml`, with its
queue in `r33_family_clean_sft_8gpu_dyn4096_queue.yaml`. It keeps `gb192`,
`mb2`, LR `1e-4`, cosine/warmup 25, the exact family-disjoint dataset, offline
W&B, validation-only `512` selection, model+optimizer DCP retention, and the
fail-closed equivalence gate. Commit `8a4ecfa` is synced to RunAI through S3
with archive SHA-256
`8959ec9115c6cc9a55f855e469dc706f06d339afba5b92650b3c02035823aa52`;
the queue dry-run exited `0` and `48` AV runner/queue tests passed. It remains
pending until the primary clean AR checkpoint and eval release GPUs 0-3.

Lightweight run specs, launch metadata, train logs, and offline W&B records for
the throughput-only probe, all failed gates, and both passing gates are frozen
locally at
`artifacts/runai_evidence/20260709_r33_publication_remediation/packed_av_correctness/r33_packed_av_correctness_evidence_20260709.tgz`
and mirrored on S3 under `publication/evidence/20260709_packed_av_correctness/`.
The checksum-identical archive SHA-256 is
`ae5151248a796b0d81187c30a0080643059aaa2ee4d009066df7d31be506dce2`;
split parquets and checkpoints are deliberately excluded.

The AV promotion is automated but still gated. Commit `3fa3b81` extends the
shared queue-chain helper to recognize `nano_av_probe_queue.v1`; all nine
queue-chain tests pass on RunAI. Detached watcher PID `1017162` reads the
mutable AR queue snapshot and will invoke the staged AV queue only after the
named AR item reaches `complete`. It does not treat process exit or training
loss as completion.

Before the AR checkpoint write, a manifest-first cleanup removed eight
superseded publication paths: the redundant frozen-runtime critic, five
failed/dry-run AR materializations, and two stopped AV materializations. The
active AR run, both deterministic critic initializations, and the staged AV
hero directory were protected and post-verified. `/workspace/models` free
space increased from about `196 GiB` to `255 GiB`. Lightweight logs, configs,
and W&B metadata were frozen before deletion; weights and split parquets were
excluded. The checksum-identical local/S3 evidence archive is
`artifacts/runai_evidence/20260709_r33_publication_remediation/prehero_cleanup/prehero_cleanup_evidence_20260709.tgz`,
SHA-256
`da7267438c9664090e11bb75d6d621c32223306dcd3f1ab9b8b1d1f662ec6769`.
Commit `44a1b7b` also repairs the retention utility so future applied manifests
persist their final `deleted` list after preserving the required pre-delete
manifest.

### 2026-07-10 - Publication-Clean R33 AR Gate And Guarded AV Promotion

The deterministic family-disjoint R33 AR SFT run completed all `1,291`
optimizer updates and wrote `iter_0001291`. The final training record was loss
`0.2625350`, normalized FVE `0.5302927`, gradient norm `0.396484375`, and LR
`5e-6`. No CUDA, OOM, nonfinite, or training traceback was observed. The
validation-only 512-row checkpoint evaluation then completed successfully;
the queue reached `complete` at `2026-07-10T01:13:44Z`.

Validation metrics from
`eval_iter_0001291_v512_t512_winrates_report.json` are:

| Condition | NMSE | FVE-NRM | Cosine |
|---|---:|---:|---:|
| teacher | 0.281703 | 0.584534 | 0.859148 |
| teacher shuffled | 0.968888 | -0.428951 | 0.515556 |
| blank | 0.756098 | -0.115121 | 0.621951 |
| generic | 0.781429 | -0.152481 | 0.609285 |
| mean | 0.678041 | 0.000000 | 0.660979 |
| source context | 0.301252 | 0.555703 | 0.849374 |
| source raw | 0.083248 | 0.877223 | 0.958376 |

Teacher text beat shuffled on `512/512` rows, blank on `504/512`, generic on
`508/512`, mean on `505/512`, and source context on `296/512`. Source raw beat
teacher on `509/512`, as expected for the diagnostic that exposes the original
source directly. The clean teacher NMSE reaches the predeclared `0.25-0.30`
AR target. It is directionally `26.3%` below the old R33 20k validation NMSE
of `0.381983`, but that historical comparison uses a different split and a
pre-remediation pipeline and is not a publication effect estimate. The
confirmatory test split remains unopened.

The guarded promotion watcher observed the completed checkpoint and eval, then
launched the clean eight-H100 AV queue at `2026-07-10T01:18:42Z`. The actual
hero process passed its fail-closed packed-vs-padded gate before optimization:
packed/padded response NLL `2.56366563 / 2.56583595`, maximum absolute
difference `0.02207184`, maximum relative difference `0.00795042`,
`passed=true`. At the `2026-07-10T03:13Z` milestone it had reached step `183`;
loss had fallen from `2.744414` at step 0 to `0.850000`, gradient norm was
`0.722656`, LR was `9.65425e-5`, and no OOM, CUDA, or traceback signal was
present. All eight GPUs held about `49.7 GiB` with roughly `44.6 GiB` CUDA
free per rank.

The selected AR checkpoint initially occupied `144G`: `36G` HF, `36G` model
DCP, and `72G` optimizer DCP. Because `/workspace/models` had only `108G`
free after the save, model-only preservation completed before any retention
action. The guarded operation hashed and uploaded the HF directory plus compact
run/eval/config/W&B provenance to
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/publication/checkpoints/r33_deterministic_family_clean_ar_sft/iter_0001291/`, verifies remote
object count and byte total, and only then removed the redundant model and
optimizer DCP directories through `nano_checkpoint_retention.py`. Verification
passed at `10` objects and `38,462,226,607` bytes. The final retention manifest
records both DCP directories in `deleted`; the local HF checkpoint and scheduler
remain protected. The operation completed at `2026-07-10T03:28:26Z`, and its
compact evidence is mirrored under
`publication/evidence/20260710_r33_clean_ar_sft/`.

The post-eval AR correctness report was upgraded to
`nano_ar_correctness_audit.v2` after finding one stale reporting label: the old
audit printed R33's zero-based last layer as 32 even while correctly requiring
extraction index 33 and 34 retained blocks. The extractor labels the hooked
module index directly, so the consistent contract is `R33 -> block 33 ->
blocks 0..33 -> 34 blocks`. The corrected audit passes: checkpoint config 34,
observed tensor blocks exactly 0 through 33, dataset sidecar extraction index
33, no saved LM head or final norm, finite `2688 x 2688` value head, and zero
document overlap across `247,865 / 13,766 / 13,765` train/validation/test rows.
Commit `b0cdeff` contains the audit correction; its RunAI verifier shard passed
`15` tests.

The next round-trip stage is staged but remains gated on AV completion. A clean
family-exposure report generated from the exact AR/AV train splits and the
confirmatory family manifest marks all `13,766` validation rows (250 families)
and `13,765` test rows (255 families) eligible, with zero exposed or cross-split
family exclusions. The validation-only queue uses two protocol-identical items:
eight-worker generation in the SGLang environment, then cached scoring in the
NLA environment through the selected clean AR checkpoint. It keeps all real,
shuffled, zero, mean, and no-injection controls, 256 generated tokens, parse
health gates, family-stratified inference, and never selects test rows.

The queue now prepares its own AV runtime: it fingerprints the immutable DCP
model and tokenizer files, converts DCP to tmpfs HF, injects the cryptographic
fingerprints into both commands, reuses the conversion, and removes tmpfs only
after both items complete. Large directory hashing is chunked without changing
the historical digest format. The immutable RunAI source is commit `815db9e`,
S3 archive SHA-256
`66e3c2405870027ffb445262ec75a8b66089e44dfb433f13f6295f8c50eff840`;
`63` relevant RunAI tests pass. The mutable queue dry-run has one shared
protocol SHA-256
`276d97c3e460218e24bf7bd751a94bd4c9ed55859cdf4c751fb2824338e8e1aa`.
At that milestone, watcher PID `1163925` was waiting on the named AV queue item
and could not launch before checkpoint plus AV eval completion. It was later
superseded and terminated; the replacement is recorded at the end of this
logbook.

At the `2026-07-10T03:59Z` AV milestone, step `261` had loss `0.833835`,
gradient norm `0.6875`, LR `9.24394e-5`, all 128 experts active, normalized
router entropy `0.990982`, and no CUDA/OOM/traceback signal. This remains a
training trajectory, not a held-out result.

A second manifest-first retention action removed the now-superseded primary
and seed-314159 independent critic initialization directories at
`2026-07-10T04:05:10Z`, freeing two reproducible `36G` copies while protecting
the selected trained AR run and active AV run. The primary full model hash is
`ad171809d6d0f96c60a905b3fc7bdb94ea1b55ba2322cbac43727b834d961bd5`.
The independent full-file hash was deliberately stopped after shared-NFS
throughput stalled; its stronger reconstruction contract remains the frozen
seed, seeded-Givens value-head hash, aggregate and per-parameter router hashes,
and the passing verifier that hashes both initialization manifests. The final
manifest lists both initialization paths in `deleted`; evidence is mirrored at
`publication/evidence/20260710_critic_init_retention/`. No selected trained
checkpoint was removed. After the shared filesystem counters settled,
`/workspace/models` reported `285G` free (`81%` used), enough for the expected
AV model/optimizer DCP plus temporary HF conversion peak.

## 2026-07-10: Publication RL Contract And Exact Checkpoint Scheduling

While the clean AV SFT run continued, the next confirmatory-RL correctness
work was implemented without changing the active source snapshot or training
geometry. `nano_rl_queue.py` now accepts an explicit, strictly increasing
`training.save_iterations` list. It emits `NLA_SAVE_ITERATIONS`, rejects a
simultaneous fixed save interval, requires the final update, and makes
checkpoint retention validate against the exact declared set. The associated
Miles hook delegates only checkpoint timing to `nla.save_schedule`; all runs
without the environment variable retain the historical periodic-save behavior.
Patch `0019_configured_save_schedule.patch` applies cleanly to the current
RunAI Miles tree. This permits publication checkpoints at updates
`16 / 64 / 110 / 228 / 342` without writing hundreds of unwanted 36 GB model
copies.

The RL launcher also renders `training.clip_grad` explicitly rather than
depending on an upstream default. A new `nano_rl_publication_contract.v1`
queue section fails closed on preregistration or artifact hash drift, seed
mismatch, any test split before selection lock, missing actor/rollout guards,
non-aborting guard policy, undeclared endpoints, or checkpoint-schedule drift.
The immutable launch contract records the resolved publication policy and its
guard-policy SHA-256.

The draft preregistration at
`docs/runs/r33_publication_preregistration.md` seals test access and declares a
finite four-item validation-only stability grid: K3 coefficients
`5e-4 / 1e-3 / 2e-3` plus `low_var_kl / 1e-3`, each for 16 updates with the
same clean SFT initialization and topology. It also predeclares seed `271828`,
the five confirmatory checkpoint updates, validation checkpoint selection,
family-clustered primary inference under a clean independently seeded critic,
Holm-adjusted secondary tests, and a prelaunch power floor of 0.80 for a 10%
relative directional gain. The document remains explicitly draft and the
confirmatory queue remains unmaterialized until the clean AV eval, clean SFT
round-trip baseline, independent critic, and stability grid complete.

Local verification passed `98` focused queue/HPO/runtime tests plus `87`
subtests, the new save-schedule unit suite, patch hunk validation, Python
compilation, `git diff --check`, and documentation consistency. No RL job was
launched. The canonical local evidence mirror was also verified at 116 files,
113 MB, with all three recorded archive hashes matching and no file above
200 MB.

At `2026-07-10T04:33Z`, the active eight-H100 clean AV SFT run had reached
step `318/1291`: loss `0.809365`, gradient norm `0.691406`, LR
`8.85459e-5`, all 128 experts active, normalized router entropy `0.991002`,
and no CUDA/OOM/nonfinite/traceback signal. The deferred round-trip watcher
remained alive and waiting on the AV queue gate. These are training dynamics,
not held-out AV or round-trip results.

The clean AV optimizer phase subsequently completed all 1,291 updates at
`2026-07-10T14:09Z`. Step 1290 loss, gradient norm, LR, and normalized router
entropy were `0.683009 / 0.585938 / 1e-5 / 0.991240`. Across all logged
updates, loss ranged from `2.744414` at initialization to a minimum of
`0.667819`; median loss was `0.737310`. Broad loss-window means declined from
`1.488113` over steps 0-24 to `0.696919` over steps 1024-1290. All 1,291 loss,
gradient, LR, and router records were finite. Gradient norms above 2.5 occurred
only during initial warmup, while all 128 experts remained active throughout.
No OOM, CUDA, NCCL, nonfinite, or training traceback signal occurred.

The final checkpoint contains about `60G` of model DCP and `119G` of optimizer
DCP. The queue entered its storage-conscious DCP-to-HF conversion before the
validation-only AV eval. At the latest check the converter was active and GPUs
were correctly idle; `/workspace/models` had `108G` free. No success claim is
made from the completed optimizer phase, and no held-out AV or round-trip
metric is recorded until the queued reports exist.

The active AV converter was deliberately left unchanged. Its DCP random-read
path exposed roughly `8 MB/s` shared-NFS throughput, so the deferred round-trip
implementation was improved for the next conversion without touching this
in-flight process. The new optional `stage_dcp_checkpoint` path fingerprints
the original model while copying it to tmpfs in the same sequential byte pass,
verifies that digest against any cached fingerprint, converts from local tmpfs,
and removes only the staged DCP copy in a `finally` block. The canonical digest
is byte-compatible with `fingerprint_directory`; 32 focused provenance,
round-trip, and queue-chain tests pass. This changes storage transport only,
not model bytes, generation protocol, evaluation rows, or metrics.

The independent-critic rebuild is now queue-chainable rather than an operator
shell step. `nano_prefix_dataset_queue.v1` runs existing
`nano_prefix_dataset_pipeline.v1` configs, keeps W&B offline, records a launch
report and log, and marks success only when every declared artifact exists and
any declared SHA-256 matches. The checked-in R33 rebuild queue pins the
seed-314159 `critic_initialization.json` to its previously verified digest
`71cfb2bf...edcff`; it reuses the deterministic full dataset and does not
generate teacher text or launch training. `nano_queue_chain.py` can now launch
this queue type after a named upstream gate, after which the existing
validation-only independent AR queue can run unchanged.

## 2026-07-10: Post-AV Chain Reconciliation And Status Boundary

The original deferred round-trip launcher from source `815db9e` was superseded
after the tmpfs staging improvement. Watcher PID `1163925` was terminated; it
did not launch generation or scoring. Replacement watcher PID `1506837` runs
from immutable source `9b8b44f` and watches the mutable queue at
`/workspace/interp/outputs/nano30b-nla-pilot/publication/launchers/clean_roundtrip_20260710_9b8b44f/r33_clean_sft_validation_queue_9b8b44f.yaml`.
It retains the same frozen protocol SHA-256
`276d97c3e460218e24bf7bd751a94bd4c9ed55859cdf4c751fb2824338e8e1aa`.
The transport change fingerprints the canonical DCP while staging it once to
tmpfs, converts from the staged copy, and deletes only that staged copy in a
`finally` block.

At the last authenticated RunAI observation, the AV queue remained in its
post-training conversion stage. The converter had run for about 48 minutes,
read approximately 16.8 GB from the final DCP, and remained CPU-active with no
reported conversion error. GPUs were idle as expected. `/workspace/models`
had about `108G` free and `/workspace/interp` about `295G` free. No AV eval
report or clean SFT round-trip report existed, so neither gate is recorded as
passed.

The independent critic and AR chain is prepared but not launched. Immutable
source `3676b93` stages the prefix-dataset rebuild queue under
`publication/launchers/independent_critic_20260710_3676b93` and requires the
exact critic-init manifest SHA-256
`71cfb2bf243bbae720d0b2931a310b6b327a2922a59d5cf5165926fc988edcff`.
The dependent AR launcher under
`publication/launchers/independent_ar_20260710_3676b93` correctly dry-runs as
`blocked_missing_critic_init`; this is the intended fail-closed state. It may
not launch until clean AV eval, clean SFT round trip, selected AV model
preservation, and verified redundant-shard cleanup complete.

RunAI authentication expired before this documentation pass could obtain a
newer observation. This entry therefore ends at the last authenticated state:
test is sealed, no independent critic/AR training has launched, and no clean RL
job has launched.

## 2026-07-11: Scientific Audit Remediation And Contract Simplification

This milestone changes code and evidence semantics only; it does not promote a
checkpoint or consume the sealed test split.

The evaluator now uses one dimension-independent directional metric,
`2 * (1 - cosine)`, while retaining `normalized_mse` only as a compatibility
alias. Raw MSE, train-mean centered R2, norm ratio, and cosine are mandatory in
new round-trip reports. Validation-only gates no longer fabricate a test split,
mean controls fail closed without training rows, shuffled controls stay within
split and exclude the same content family, and family-clustered bootstrap plus
sign-flip inference replaces row-IID promotion logic. Cached generations bind
model, tokenizer, and selected parquet hashes; malformed placeholders and
protocol mismatches fail before scoring.

The deterministic R33 dataset is now described accurately as a reproducible
stored snapshot. Exact replay passed, but the fresh-forward fidelity report
failed all 32 rows at `2.562%` mean and `7.728%` maximum relative L2. Clean AR
teacher evidence is directional MSE `0.281703` and cosine `0.859148`, alongside
raw MSE `8.537785` and centered raw R2 `-0.201696`; no raw-magnitude
reconstruction claim is made. The May 28 teacher root is now hash-enforced as
`76b78d2c34a251f004d53eb5d53766fa01879e2bf3744bc4d80d4fcc1d17825e`,
and methods documentation names `nemotron-3-super-v3` and the historical
`865` parse-failed plus `1,015` unmatched exclusions.

Content-family construction gained a deterministic prefix-filter closure pass
over the declared `0.80` Jaccard threshold. Prior-exposure mapping now uses both
document IDs and normalized content hashes, requires zero unmapped prior docs,
and frozen manifests cannot be overwritten. The existing refined manifest
predates this closure and is therefore still pending rebuild and sealed-test
re-audit; code readiness is not evidence completion.

RL dataset split units now serialize end to end, lineage files and hashes are
required rather than skipped, and every queue item captures source and scrubbed
effective environment in an immutable launch contract. Metric guards abort on
any numeric nonfinite value. Packed actor and critic checks and an explicit
Mamba kernel are required by registered study plans. Post-eval reads frozen
commands, runs before retention, and retention must preserve every evaluated or
registered checkpoint iteration. Queue writes are locked and item updates bind
by name. There is no separate publication execution mode: optional
`preregistration` metadata adds a hash-pinned study plan to the same queue path.

The confirmatory plan now requires two independent seeds, caps administrative
relaunches before optimizer update 0, removes the unreliable monotonic p95
guard, and limits its telemetry contract to metrics the runtime actually emits.
Clean AR/AV learning-rate neighbors, a common-kernel or measured cross-kernel
delta, a validation-family power report, the rebuilt fully fingerprinted
independent critic/AR, clean AV validation, and an identity-bound clean SFT
round trip all remain required before RL launch.

Repository reliability also improved: pytest is rooted at `tests/`, optional
Torch modules collect as explicit skips on CPU-only developer environments, a
CPU-Torch GitHub Actions workflow is checked in, generated queue locks are
ignored, and documentation consistency now checks canonical metric, fidelity,
fork-disclosure, and clean-lineage statements. The vendored NLA tree is now
explicitly labeled a modified Nano30B production fork, including frozen-critic
RL and injection scale `75` versus the reference recipe's `150`.

Final local verification on the Mac's Python 3.9 environment: `498` tests
passed, all `214` YAML configs parsed, documentation consistency passed, and
`git diff --check` was clean. Torch-only modules collect as skips because Torch
is not installed locally. RunAI Python 3.12/Torch verification remains pending
only because the RunAI SSO token expired; no cluster job was changed.

## 2026-07-15: Qualified Family-Clean R33 SFT AV+AR Pair

This milestone closes the supervised NLA qualification loop. It does not
reopen the invalidated July 8 RL effect and does not launch RL.

### Boundary and selected models

The v2 family manifest records `275,396` rows, `27,647` documents, and `5,009`
exact-refined content families. Clean AV/AR training exposure contains `4,504`
families. Validation has `250` families and test has `255`; train-family and
cross-split-family overlap are both zero.

Selected AR is the loadable HF `iter_0001291` checkpoint from
`nano-ar-r33-publication-deterministic-family-clean-4gpu-unfusedtorchconv-expertscan-cudablock-lr5e5-warmup25-gb192-mb48`, SHA-256
`5e792120ec1a00ebb4cf4abca50d2a6a962421ac4f45423479ae5061f4d2d760`.
Selected AV is the DCP `iter_0001291` checkpoint from
`nano-av-r33-publication-deterministic-family-clean-8gpu-pospass-lr1e4-warmup25-gb192-mb2-dyn4096`, model fingerprint
`dcp_model_sha256:43346232d2fc043260ee903191e20cce07801903e1e7b7956f16022eb463386a`.

### Component evidence

The AR validation verifier passes on 512 rows. Teacher-text directional MSE,
cosine, and FVE-NRM are `0.281703 / 0.859148 / 0.584534`; raw MSE is
`8.537785` and centered raw R2 is `-0.201696`. The directional signal is strong
against shuffled (`0.968888`) and mean (`0.678041`) controls, while the negative
raw R2 prevents a raw-magnitude claim.

The AV validation verifier also passes on 512 rows. Real response NLL is
`0.776775`, compared with shuffled `1.311727`, zero `1.176494`, mean
`1.237522`, and no-injection `1.220974`. The checkpoint is conditioned on the
specific activation rather than merely the prompt or average injection.

### Generated-text round trip

The same generation protocol and checkpoint fingerprints are bound into both
splits. Validation uses 512 rows across 250 families; test uses 512 rows across
255 families. Every generated explanation is closed and usable.

| Split | AV directional MSE | Teacher directional MSE | Raw MSE | Centered raw R2 |
|---|---:|---:|---:|---:|
| Validation | `0.307004` | `0.304714` | `9.449079` | `-0.326586` |
| Test | `0.319225` | `0.302637` | `9.647148` | `-0.335374` |

On test, control-minus-candidate directional-MSE gaps are `0.361806` for the
train mean, `0.522112` for mean activation generation, `0.536178` for
no-injection generation, `0.645884` for shuffled activation generation, and
`0.663983` for zero activation generation. Every family-bootstrap 95% lower
bound is positive, rowwise win rates are `99.61-100%`, and one-sided family
sign-flip p-values are approximately `1e-5`.

### Stored-snapshot functional recovery

The functional verifier passes under claim scope
`stored_snapshot_counterfactual_reinjection`. On test, candidate KL/JS/logit
Pearson are `0.949545 / 0.152073 / 0.907847`; teacher reconstruction is
`0.970104 / 0.145112 / 0.911889`. Candidate-versus-teacher family intervals
include zero on every registered metric. Candidate KL is much lower than mean
`4.124133`, zero `6.297471`, and shuffled `9.528919`; the corresponding
family-clustered comparisons all pass.

The stored activation replay check still exceeds the preregistered exact
fresh-forward tolerance. Functional evidence is therefore counterfactual
reinjection into the stored snapshot, not proof that the archived activation
equals a fresh Nano30B forward pass.

### Provenance, preservation, and decision

The release builder verified both exact checkpoint fingerprints and all six
component/end-to-end verifier reports, producing
`nano_nla_checkpoint_pair_manifest.v1` with `qualified: true`. The first test
verifier invocation failed only because its config omitted the validation hash
from the expected provenance set while the report correctly contained all
three split hashes. The verifier schema was corrected to require
`train + validation + evaluated split`; no score, threshold, sampled row, or
report changed. New regression tests cover this contract.

Both model payloads are mirrored to S3. The AV mirror was attested at 21
objects and `63,245,896,177` bytes; the AR mirror contains 10 objects and
`38,462,226,607` bytes. The compact evidence archive includes W&B offline
logs, train/eval logs, generated text, configs, family manifests, reports, and
both immutable source snapshots, with no checkpoints or Parquet datasets:

- local: `artifacts/runai_eval/r33-clean-sft-av-ar-qualified-20260715/`
- S3: `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/sync_exports/r33_clean_sft_av_ar_qualified_20260715.tgz`
- archive SHA-256:
  `b9043eae03cbb71c38a3feb81262361e764d4a147ba98d66735293b75b24f6bc`

Decision: this pair supports the statement that Nano30B has a good R33
supervised NLA checkpoint pair for family-disjoint activation-direction
reconstruction and stored-snapshot functional recovery. Do not extend that
statement to raw-magnitude recovery, exact fresh-forward identity, a pristine
project-wide test, a new R33-over-R27 comparison, independent cross-critic
generalization, or RL improvement.

## 2026-07-16: Publication Boundary, Fidelity, And Magnitude Audit

No training or RL was launched in this audit. The selected AV and AR
fingerprints remain unchanged.

### Full exposure boundary

The earlier v4 evaluation-only family audit was superseded because it omitted
the selected checkpoints' own train partitions. The corrected v6 audit
enumerated `136` selected-pair train/eval and historical evaluation sources,
covering `28,665` unique documents. It mapped every document and found all
`5,009` canonical families exposed. The result is conclusive: no in-corpus
confirmatory family remains for the selected pair.

Report SHA-256 is
`373e2988b32f2e4e68e2d7644a8b77430829ea7a5d5e2f6b72f870caa89d088b`;
the exposure-inventory and joint-manifest hashes are
`c193f2efb7c8414f4f8a6a12ab2051e633b891107f92c2971535164128f9aabb`
and
`9d68a894e763ed533fba11016ebc8b8f05c0d4a39443585196e7cd24ebffbc20`.

The teacher-corpus inventory inspected `63` candidate tables and found `53`
usable ones. None contains a numeric document suffix outside the already
exposed `0..38161` range. With new teacher generation prohibited, external
teacher-backed data is required for confirmatory replication.

### Stored versus fresh activation fidelity

The first 64-row monolithic diagnostic OOMed before scoring and is retained as
an operational failure. The bounded `mb=8` rerun completed on validation-only
rows. Strict fresh-vs-stored identity failed on all 64 rows. Repeated fresh
forwards were exact, and full-forward versus extraction-forward activations
were exact. Fresh-vs-stored relative L2 was `0.031405` mean, `0.019490` median,
and `0.185983` maximum; cosine was `0.999142` mean and `0.983146` minimum.
Using the stored activation to predict the fresh activation gives centered raw
R2 `0.997255`.

Decision: the runtime is repeatable, but the archived stored snapshot is not
an exact fresh-forward output under the current environment. Keep all
functional claims scoped to stored-snapshot counterfactual reinjection.

### Frozen scorer caches and calibration

The validation and exploratory-test round-trip scorers completed on 512 rows
each and emitted hash-bound prediction caches. Both gates pass with 100% usable
and closed generations. The new scorer runtime is close to, but not bit-exact
with, the July report. Validation cache SHA-256 is
`e85ab5b42ab226d3d79880c953cef5eb0b9540b49826dbacd65decb36c8adc63`;
test cache SHA-256 is
`6968607bfd77dad56537736249daa698ce21fc3f0b11ec0d675afb463038225d`.

A frozen one-parameter calibration compared identity, origin-scalar, and
train-mean-scalar transforms. Selection used validation teacher raw MSE only.
The selected origin scalar is `0.5606042253`; the nonnegative constraint was
inactive.

| Split / variant | Raw MSE identity | Raw MSE calibrated | Centered R2 identity | Centered R2 calibrated |
|---|---:|---:|---:|---:|
| Validation teacher | `9.187760` | `3.629222` | `-0.289899` | `0.490482` |
| Validation AV real | `9.446685` | `3.648806` | `-0.326250` | `0.487733` |
| Exploratory test teacher | `9.202918` | `3.601751` | `-0.273883` | `0.501440` |
| Exploratory test AV real | `9.647148` | `3.770353` | `-0.335374` | `0.478102` |

For exploratory-test AV real, the family-clustered 95% interval for raw-MSE
improvement is `[5.634404, 6.123574]`, with improvement in all `10,000`
bootstrap draws. Positive origin scaling leaves directional MSE unchanged at
`0.319225`.

Decision: a large share of the raw error is a correctable global scale
mismatch. Because the calibration is post-hoc and the test has prior project
exposure, it is supporting evidence only. It does not establish native exact
magnitude recovery or an external confirmatory result.

Lightweight evidence is local at
`artifacts/runai_eval/r33-clean-sft-publication-evidence-20260716/` and mirrored
under
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/publication/evidence/20260716_r33_clean_pair/`.

## 2026-07-16: Subgroup Robustness, Qualitative Packet, And Independent AR

The frozen validation and exploratory-test prediction caches were reused for a
post-hoc subgroup audit. Quantile edges were fitted on validation only and
applied unchanged to test. The registered dimensions are source-token length,
teacher-target word count, target activation norm, and sample-family
frequency. All 16 validation/test bins satisfy the minimum row/family counts,
and every family-clustered 95% control-improvement interval remains strictly
positive. The weakest test slice is the lowest activation-norm quartile:
directional MSE `0.370077`, calibrated centered raw R2 `0.415606`. This does
not repair the exposed test boundary and is reported as exploratory robustness.

A deterministic qualitative panel now contains 50 validation and 50 test rows,
each with source text, teacher explanation, and AV generation. Its automatic
structural screen found zero flags for empty, encoded-looking, repetitive, or
severe length-regression outputs. The automatic report explicitly disclaims
semantic review. Human blinded ratings and inter-rater agreement remain a
publication gate.

The seed-`314159` independent critic was rebuilt from the frozen base. A first
byte-exact manifest gate rejected it because the manifest included the random
head's transient `before_sha256`, even though deterministic initialization
overwrites that tensor and every final-state hash matched. The failed attempt
and manifest were preserved. The generic prefix queue now supports a canonical
JSON SHA-256 gate with an explicit `ignore_json_paths` allowlist. Ignoring only
`value_head.before_sha256`, the old and rebuilt manifests share SHA-256
`34e863f756e0749ca19fc8c138b7bd71b5da69c907ee42ad021517542e5c8941`.
Regression tests prove that an ignored transient change passes while any final
state change fails. The independent initialization verifier passes all 16
checks; report SHA-256 is
`4639285fea694f7f850c766b31d8ddea4e2b2bdd61c710f3b2a4cdd2109fb6e3`.

The independent clean AR replication completed with seed `314159`, four
H100-NVL GPUs, `lr=5e-5`, 25 warmup updates, `gb192/mb48`, cosine decay, 1,291
updates, validation-only model selection, and offline W&B. Final train
loss/FVE-NRM were `0.295026 / 0.472163`. Its 512-row component verifier passes:
teacher directional MSE/cosine/FVE-NRM are
`0.286169 / 0.856916 / 0.577948`, versus the primary seed's
`0.281703 / 0.859148 / 0.584534`. Teacher beats shuffled on all 512 rows and
mean on 507/512. The component report SHA-256 is
`368c84cad8bb0b8a7235b1a1e96c862b69d33be72d49c1f07a563e62d0be65aa`.
Frozen AV-generated validation text is now being scored through the independent
AR. The run is intentionally not labeled cross-critic replicated until that
report and verifier pass.

The source/config snapshot used for the publication evidence tools is mirrored
through S3 at
`source-sync/publication-work/20260716T035347Z/nano30b-nla-pilot-source.tgz`
with SHA-256
`700b6690ac32d8c6835891af5e7769f402c1b7047eb818777cf547522133555e`.
The local full test suite reports `721 passed, 1 skipped, 162 subtests passed`;
the installed RunAI snapshot's focused suite reports `21 passed`.

## 2026-07-16: Independent AR Round Trip And Release Evidence Closure

The frozen 512-row validation generation cache was scored through the clean
seed-`314159` independent AR, covering 250 content families. The hash-bound
verifier passes with `100%` closed and usable generations. AV-text directional
MSE is `0.3109634728`; independent teacher-text directional MSE is
`0.3085329510`, a gap of `0.0024305208`. All five controls pass
family-clustered inference. The mean-control rowwise win rate is `0.998047`;
the AV-mean, no-injection, shuffled, and zero-control win rates are `1.0`.

The report SHA-256 is
`6f0829a61b03ac584b109c1c7a54f689f0a14b61f7fee803afd3d5ca29bf552b`;
the verifier SHA-256 is
`dd3de6e1dd10f0f64c25e23ff3152638c9cf46785133dfb6c0466c353e434331`.
This changes the decision: row-specific AV information now replicates through
an independently initialized/trained AR on validation. It does not test an
independent AV seed, an external corpus, or raw magnitude; independent centered
raw R2 is `-0.399865`.

The independent HF checkpoint fingerprint contains 10 files totaling
`38,462,226,688` bytes. Directory SHA-256 is
`c2eea74f5baccee97128617b05636187804c7e59aedc560d088dbf65d52f1925`,
and fingerprint-manifest SHA-256 is
`79408abd1e7cafadbc68ebe627bca99381e1a3e4486aa950ca0e7a42c97bb1ed`.
The S3 mirror was verified at the exact object count and byte total before a
manifest-first retention pass deleted only the 36G model DCP and 72G optimizer
state. The HF checkpoint and lightweight run/W&B evidence remain.

While preparing qualitative review, the panel was found to contain empty
source strings because `nano_r33_source_rows.py` did not load the existing
`detokenized_text_truncated` column. The resolver was fixed, the panel builder
now fails closed on an empty source, and regression tests cover both behavior
and failure. The old panel is invalid. The corrected 100-row source-grounded
panel SHA-256 is
`4f5d61486330b1104dd0a256ea185d8c1c99512ee9ff4731f8135305924f81c8`;
its zero-flag structural report SHA-256 is
`33de2720d96bda3f663318be7fe8c10765740a3d603bcdf76682ca23661297da`.

Two deterministic blinded reviewer packets and a separate answer key were
generated. Human ratings remain pending. The answer-key SHA-256 is
`27585eae51d55deb9bb3821afbd1f5d1d3e7cfd0e7c4167e4111566ff06c1856`.
An automatic release-text audit scanned all 1,024 frozen candidate generations:
zero configured sensitive-pattern findings, zero source-copy failures, and
zero panel candidate copy flags. Fourteen phone-like strings are confined to
source excerpts. Those excerpts remain internal pending human adjudication or
redaction. Audit SHA-256 is
`00e501ff644483e614d0b60071f726f33575c95ba8b81d5c6b59c4bd79d13419`.

At this point in the follow-up, the scientific checkpoint claim was strong
enough for a carefully bounded stored-snapshot SFT artifact. The then-open
work included external teacher-backed replication for any confirmatory
generalization claim, blinded human semantic scoring, exact
teacher-service/license approval, repository license and notices, exact
public-bundle security review, and complete compute accounting. The compute
and bundle items were closed later in this same entry; no RL claim is revived.

Selected-run compute accounting was reconstructed directly from hashed train
logs and fails if any optimizer step in `0..1290` is absent. Primary AR used
4 H100-NVL GPUs for `3.8867h`, primary AV used 8 for `13.4608h`, and
independent AR used 4 for `3.8781h`, totaling `138.7456` H100-NVL GPU-hours.
The report SHA-256 is
`7bde74be3a874d2ae305463ca8da211c069ce0bf1001802b6bdf7ab091fd7238`.
Extraction, conversion, evaluation, historical HPO/RL, and failed diagnostic
compute without exact retained timing are listed as exclusions and are not
silently folded into this total.

A static source-bundle audit was added with exact path/kind/count allowlists
for synthetic test fixtures and no matched-value emission. The 680-file
source-only internal snapshot contains no checkpoint/data payload, forbidden
heavy extension, symlink, binary, oversized text file, or unallowlisted
credential fixture. It still fails public release on 31 files containing
local-home, internal S3/endpoint, or cluster references. The tree-manifest and
report SHA-256 values are
`3469d92cc535e42b4e70098ce5f3640cb980e051c43427794a6afe4fef0cf8e8`
and
`bf697acdad268e219cfd729428233b1259b07aea988529e8c7e4bb01b8a62c42`.
Decision: retain the full snapshot as internal reproducibility evidence;
construct a narrower/redacted public bundle and rerun the same scanner on its
exact bytes.

The narrower public-bundle candidate is now complete. A config-driven builder
copied only release-relevant text source and aggregate evidence, redacted all
workstation/RunAI/internal-store locators, excluded weights and generated text,
and emitted a file-level source/staged hash manifest. Compact curves exported
loss, LR, gradient, router, system, and performance telemetry for every one of
the 1,291 updates in primary AR, primary AV, and independent AR; curve-file
SHA-256 is
`7d9c22b989c594e546ec08648d0319c37caad69ad502d2badb635d41706c42a6`.

The exact staged tree has 496 files and passes the security gate with zero
failed findings, forbidden paths, symlinks, binaries, or oversized files. Its
tree SHA-256 is
`df175c5f61cefbfc1a02451a7bd242ba69e1cb602cdd97ca4b8bd8fe9c263b77`.
The deterministic archive was re-read without extraction and produced the same
content tree. It is 6,859,370 bytes with SHA-256
`3eb8e64ed0d9d61ed2d6b0694fbaf96b99051a63f2ce1a6c99372d93832e573a`.
Decision: the technical bundle-staging blocker is closed. Do not distribute it
until blinded human review, phone-like source-string adjudication, repository
and weight-license approval, exact teacher-API terms, and final notices are
complete. External data remains required only for a confirmatory
generalization claim.

Final local regression status for this publication-prep pass is `749 passed`.
The active claim-document consistency checker also passes. Repackaging the
unchanged candidate reproduced the same archive SHA-256, confirming the
normalized archive is deterministic.

The no-weights candidate is preserved internally in S3 as five objects:
archive, SHA-256 sidecar, bundle manifest, security report, and attestation.
The remote listing matches the expected 6,859,370-byte archive. This is
durability evidence only, not public redistribution authorization.

Documentation reconciliation on `2026-07-16` added the exact
`docs/releases/r33_clean_sft_release_candidate_attestation.md` index and
updated `docs/current_state.md`, the R33 run report/gate matrix, AV run history,
job tracker, release checklist/model card, registry, and README. The July 10
"independent AR unlaunched" text is retained only as an explicitly historical
snapshot and points to the completed July 16 replication. Current unresolved
gates are human semantic review, repository/base-model and teacher-service
legal approval, and final notices. External data is needed only for a stronger
confirmatory generalization claim; a second AV seed is recommended but not a
condition for documenting this one checkpoint artifact.

## 2026-07-16 PT / 2026-07-17 UTC: Training Closure And Compute Teardown

The selected family-clean R33 SFT AV+AR pair and independent AR replication
already satisfy the evidence contract for the bounded directional
reconstruction and stored-snapshot functional-recovery claim. No further
training is required to preserve or publish that scoped checkpoint result.
Additional AV seeds, a row-matched R27 retrain, external teacher-backed data,
or clean RL would each enable a stronger and separately named claim; none is a
hidden prerequisite for the current claim.

Immediately before releasing compute, process inspection found no active Nano
trainer, evaluator, queue, or promotion watcher. All eight H100-NVL GPUs were
idle at 4 MiB reported usage and 0% utilization. The RunAI `train` workspace
was suspended at `2026-07-17T03:47:21Z`; its final state was `Stopped`, with no
pod and zero allocated GPU/CPU resources. The two persistent PVCs remain in
the workspace specification, while selected checkpoint and lightweight
evidence retention remains governed by the existing PVC/S3 manifests. This
was intentionally a reversible compute teardown, not deletion of the
workspace data.

## 2026-07-17: Clean Online Joint AV+AR Canary And Strict Non-Promotion

The `train` workspace was resumed after accidental suspension and the first
family-clean online joint AV+AR canary completed on eight H100 NVLs. Retry 4
used a `4 actor / 3 critic / 1 SGLang` GPU split, actor/critic learning rates
`1e-5 / 5e-6`, 24 rollout samples per update, and two updates. Both actor and
critic optimizer steps completed and both `iter_0000002` DCP checkpoints were
atomically committed. There was no OOM or optimizer failure.

Update 1 reward mean/std was `-0.374729 / 0.100902` with 24/24 usable
rollouts. Update 2 reward mean/std was `-0.553841 / 0.587261`; 21/24 rollouts
were usable and the minimum reward was `-2.0`. These dynamics are a warning
against scaling from training reward alone.

The initial post-eval failed before generation because explicit converted-model
and tokenizer identities were missing. The queue was made provenance-aware and
post-eval-retryable, and a second evaluator bug was fixed: paired comparisons
now align reports by canonical dataset `row_keys`, not unstable subset-local
numeric indices. Regression tests cover subset and reordered report alignment.

The repaired evaluation used 64 family-stratified validation rows and five
generation arms. Candidate and clean-SFT baseline have identical row keys,
dataset hashes, generation protocol SHA-256
`ec72786f8addfe132d245ec8a981719b94dbf1c7d6fbb5710655ef2322cb75cc`,
and 64 independent families. All generated explanations are closed and usable.

Online/SFT directional MSE is `0.291993 / 0.292173`; paired wins are `32 / 64`
and relative improvement is `0.0618%`. The strict family-bootstrap interval is
`[-0.007704, 0.008597]` and sign-flip `p=0.4824`. Raw MSE worsens from
`8.797533` to `8.969927`. Real online text nevertheless beats shuffled, zero,
mean-injection, no-injection, and activation-mean controls by large margins.

The permissive zero-threshold report is retained as audit history, but it is
not the promotion decision. The report-only strict policy requires 60% paired
wins, 10% relative gain, a positive clustered family-bootstrap interval, and
exact provenance. It reports `passed: false`. The clean SFT pair therefore
remains canonical; this canary proves online joint training plumbing, not an RL
quality improvement.

Lightweight evidence, including offline W&B, train/eval logs, generated text,
configs, reports, and prediction caches, is checksum-verified locally under
`artifacts/runai_eval/r33-online-joint-canary-evidence-20260717T0951Z/`.
Model shards were intentionally excluded. The evidence archive SHA-256 is
`3a721f6bbd795b4aeba4a801164594c982807d8eee2160ad4ab0e484e52efc83`.

## 2026-07-17: Clean Online HPO Correctness Remediation And H1 Start

Before scaling the clean online canary, critic sample accounting was audited.
The historical asymmetric `actor_dp=4`, `critic_dp=3` repartition assigned two
actor shards to critic rank 0 and one each to ranks 1 and 2. Miles then
truncated all ranks to the smallest local count. The two canary updates
therefore trained the critic on `18/24` and `12/21` usable rows respectively,
while the actor trained on the full batch.

A shared row-level repartitioner now reconstructs the complete globally
indexed rollout, validates exact coverage and sample-field lengths, performs
critic-token filtering and DP/microbatch alignment once globally, and emits
equal local shards. It reports all retention/drop counts to stdout and W&B and
asserts that no later safety truncation occurs. Queue validation adds a static
critic plan plus configurable minimum runtime retention.

The bounded search uses four candidates at eight updates each, 24 prompts,
eight samples per prompt, global batch 192, microbatch 2, and topology
`4 actor / 3 critic / 1 SGLang`. H1 is actor/critic LR `1e-5 / 2e-6` with K3
coefficient `3e-3`; H2 tests actor LR `5e-6`, H3 critic LR `5e-6`, and H4 K3
coefficient `1e-3`. Only H1 is approved. Evaluation is validation-only on 128
family-stratified rows paired to the 512-row qualified clean-SFT baseline.

An initial H1 startup was stopped before rollout when command inspection found
that the shell launcher omitted retention and JSON guard variables from Ray's
environment. That preflight artifact is preserved and is not a training run.
The launcher forwarding is fixed, the actor reference and SGLang now share a
single `/dev/shm` HF staging target, and focused local and RunAI tests pass
(`100 passed`). Corrected H1 was relaunched from the unchanged qualified SFT
initialization; no quality claim is made before optimizer and paired eval
evidence. See `docs/runs/r33_online_joint_hpo8_20260717.md`.

The corrected H1 startup subsequently generated its full 192-sample first
rollout but stopped before update 0: clean-SFT close rate was `0.8854167`
against an over-strict one-batch `0.95` parser guard. There was no OOM,
optimizer update, or checkpoint. A tokenizer audit of 120,000 clean SFT target
responses found p95/p99 lengths `156 / 174`, maximum `233`, and zero above 256,
so the rollout cap remains 256. Guards were recalibrated to sustained
close/usable collapse below `0.80` and raw generation truncation above `0.20`.
Raw engine truncation is now preserved before parser failures are relabeled,
and rollout summaries print before a guard can abort. Focused tests pass
(`110 passed`); H1 retry 1 uses fresh run and offline-W&B identities.

Retry 1 passed those rollout guards (`91.67%` closed, `91.15%` usable,
`5.73%` raw-truncated) but stopped before update 0 at the actor
packed-equivalence gate. Packed/padded mean NLL was `1.379626 / 1.356946` and
global maximum absolute/relative drift was `0.103714 / 0.053270`. Rather than
relax a correctness gate, retry 2 switches actor reference scoring, current
scoring, and training together to Miles' native padded `bshd` path. The NLA
FSDP wrapper supplies a length-derived padding mask; checked patch
`0020_fsdp_bshd_support.patch` permits the existing FSDP data/loss path, and
the queue exposes the format as config. Focused tests pass (`117 passed`). No
optimizer update or checkpoint is claimed from either failed attempt.

### 2026-07-17 - HPO8 Native-Padded Actor Pass And Critic Contract Fixes

H1 retry 3 completed the first native-padded actor optimizer update from the
qualified clean SFT pair. Its rollout was `0.947917` closed, `0.937500` usable,
and `0.046875` raw-truncated. Actor loss, K3 loss, grad norm, clip fraction, and
train-vs-rollout log-prob drift were `0.004039`, `0.002306`, `4.3125`,
`0.008266`, and `0.027766`. It failed before critic optimization because the
new repartitioner treated Miles' globally replicated `raw_reward` vector as a
row-local list. The generic batching contract now validates and preserves
replicated global fields.

Retry 4 then balanced the live critic batch correctly: `192` generated, `168`
usable, `168` retained, zero alignment loss, and `56` rows per each of three
critic ranks. Reward and training critic paths agreed exactly on the step-0
real-data check. The run failed before critic backward because critic-token
rewriting did not rebuild `max_seq_lens` for the native `bshd` path. That
metadata is now derived from rewritten critic-token lengths and the configured
padding multiple. The combined focused suite passes (`110 passed`), and retry
5 is running under a fresh immutable run and offline-W&B identity. None of
retries 2-4 produced a checkpoint or a promotable HPO result.

### 2026-07-17 - Corrected HPO8 H1 Paired Result and Component Decomposition

- H1 retry 5 completed eight joint online-RL updates from the qualified
  family-clean R33 SFT pair. The canonical evaluation is the regenerated
  128-row, 128-independent-family validation set under matched generation
  protocol SHA-256
  `e5e3a2658d28975514dd962be18c149012ee1fc85f1d6f52ccc834f59c95d416`.
- Clean SFT baseline versus joint H1 directional MSE is
  `0.3028433237 -> 0.3012607509`, a nominal `0.5226%` improvement. The
  family-clustered paired CI `[-0.0040094576, 0.0072551257]`, sign-flip
  `p=0.296687`, and 50% H1 row-win fraction fail the strict baseline gate.
  The independent strict regate also returns `gate_passed=false`.
- Provenance, dataset/row identity, protocol parity, primary parse health
  (`closed=usable=1.0`), family inference, and all real-vs-control checks pass.
  This is a correct negative/pilot outcome, not a runtime failure.
- Actor-only scoring is `0.3012762239`, nearly joint H1; critic-only scoring is
  `0.3028854620`, effectively flat/slightly worse than SFT. Thus the small
  nominal movement is actor-side, and higher critic LR is not the next
  evidence-backed HPO lever.
- The earlier `v128_t64` diagnostic stays explicitly non-promotional because it
  did not use the matched protocol. No H2-H4 or hero training was launched.
- Lightweight evidence was synced locally at
  `artifacts/runai_eval/r33-online-joint-hpo8-h1-protocolfixed-20260717/` and
  S3; local archive SHA-256:
  `bc44dfef896f63eccf68949eec7a1bc392b0a03f1e4e694489c1c13037345809`.

### 2026-07-17 - R33 Actor-Schedule Development HPO Protocol (In Progress)

- The sealed 128-family H1 regate is held out from selection. A newly derived
  development pool has 6,802 rows and 122 families with zero overlap to that
  regate; parquet SHA-256:
  `b5f31379bbc41d087c07b7dda2445ad6fb30af25e7070157b8912fa1f9be5eaf`.
- A clean-SFT baseline generation and score pass on a fixed 122-row
  family-stratified subset is running before any online-RL candidate. It uses
  the matched round-trip protocol and full real-vs-control set, and captures
  score-only AR device telemetry.
- The selection-only grid holds critic/K3/topology and rollout protocol fixed,
  varying actor LR and update count only: `2e-5/8`, `3e-5/8`, `1e-5/24`, and
  `2e-5/24`. These are development experiments, not a hero promotion.

#### Metadata-contract repair (no quality result)

- The first development-baseline launch stopped before any completed
  generation because the derived validation parquet did not carry its required
  `.nla_meta.yaml` sidecar. The failure is retained as an operational contract
  artifact, not an AV/AR or RL result.
- The family-holdout builder now copies and derives checksum-bound NLA dataset
  metadata, including filtered row count and holdout lineage. The retry uses
  new `actor_hpo_dev/v2/` outputs: 6,802 rows, 122 families, zero overlap with
  the sealed 128-family H1 boundary, and an output `d_model=2688` sidecar.
- The v2 clean-SFT baseline is running. No actor HPO candidate has been armed
  or evaluated under this repaired protocol.

#### Completed clean-SFT development baseline

- The repaired v2 baseline completed on `2026-07-17T22:45:27Z` with a passing
  round-trip gate. It reuses exactly 122 completed family-stratified
  generations from the clean R33 AV SFT checkpoint under generation-protocol
  SHA-256 `97ef2a00acae3ace82ad5efc0c2586a447a93d2d1d2e4be72dc3443e4a424678`.
  The fixed development subset has 122 rows and 122 independent families.
- Generated `av_real` explanations give directional NMSE `0.3090549575`, FVE
  `0.5501967978`, and cosine `0.8454725213`. The matched teacher-text score is
  `0.3144731059` / `0.5423111436` / `0.8427634471`. This is a development
  baseline, not a sealed comparison or an RL-improvement claim.
- The real-text path strongly beats every nonsemantic control: shuffled
  directional NMSE `0.9718231760`, zero `0.9802652469`, no-injection
  `0.8631586509`, and mean `0.6870892943`. It wins against mean on 121/122
  rows and against no-injection, shuffled, and zero on 122/122 rows.
- Parse health is complete for every generated control: 122/122 closed-tag,
  usable, and content-usable records, with zero empty or repetition-loop
  records. The report and prediction cache are
  `actor_hpo_dev/v2/baseline/roundtrip_v122_direct_gpu_retry2_report.json` and
  `roundtrip_v122_direct_gpu_retry2_predictions.npz` on RunAI.
- Three prior score attempts are retained as operational-only evidence with no
  quality result: automatic AR placement was too slow, the CPU materialize plus
  `model.cuda()` path was too slow, and the first direct-GPU retry exposed a
  stale call to a nonexistent remote-code patch helper. The loader now uses
  `patch_nemotron_h_checkpoint_dir`; focused remote tests pass (`101 passed`).
  The successful retry places all 19,222,267,968 critic parameters on `cuda:0`
  at roughly 38.5 GB allocated after scoring.
- This passing baseline unlocks only the first guarded development candidate
  (`actor_lr=2e-5`, 8 updates). The sealed 128-family H1 boundary remains
  untouched and no hero or publication conclusion follows from this result.

## 2026-07-18: R33 Actor-Schedule Development HPO Repair and R3 Start

This entry records runtime integrity, not an RL quality result. The HPO pool is
the family-disjoint 122-family development boundary; the sealed 128-family H1
regate remains untouched.

- The first active retry (`a2e-5/u8-r1`) reached its real 192-sample rollout
  and actor phase, then stopped before critic backward because one rank saw a
  `2.02335%` reward-versus-train MSE-ratio tail against a `<2%` check. Its
  mean ratio was `0.9995`, it wrote no checkpoint, and it is not an HPO result.
- The check was made modular and fail-closed: reward/eval-layout agreement
  remains strict at `2%`, while actual train-mode/eval-layout drift has its
  own `5%` BF16 bound. This preserves a strong layout assertion rather than
  bypassing it. A source-closure sync and focused RunAI suite then passed
  (`105 passed, 2 skipped`).
- A pre-launch-only `r2` exposed duplicate staging of the same 63.2 GB AV
  reference checkpoint. Shared marker reuse and a content-addressed existing
  `/dev/shm` stage removed the duplicate copy; `r2` produced no Ray worker,
  rollout, optimizer step, or checkpoint.
- Fresh `r3` began `2026-07-18T00:27:46Z`, reusing all input stages. Its first
  rollout has 192 samples, `0.9479` closed fraction, `0.9323` usable fraction,
  `0.0625` generation-truncated fraction, and reward mean/std
  `-0.4330 / 0.5014`. Critic repartitioning retained 174/176 usable rows.
- On real rollout data, both repaired step-0 ratios are exact: mean `1.0000`
  and maximum deviation `0.0000` for reward/eval-layout and
  train/eval-layout. The first critic step has normalized FVE `0.4787`; the
  first actor step has PPO KL `0.000234`, clip fraction `0.00666`, KL loss
  `0.00241`, and train/rollout logprob drift `0.0253`.
- `r3` stopped during rollout 1 because Ray saw `709.62 / 715.26 GiB` of its
  cgroup and killed an actor worker. This is a host-memory pressure failure,
  not GPU OOM or a learning instability: stale `/dev/shm` model stages and
  temporary HF conversions consumed 405 GiB and count toward the cgroup.
  The run completed one joint update but did not save iteration 8, so it has
  no round-trip report or HPO decision. Keeping only the active 59 GiB actor,
  36 GiB critic, and 59 GiB AV/SGLang stages lowered `/dev/shm` use to 155 GiB
  and cgroup use to about 159 GiB. Fresh `r4` preserves all training settings,
  data, and gates; it changes only this ephemeral-storage condition.

#### R4 live retry: storage repair validated, no HPO result yet

- The sole armed retry, `a2e-5/u8-r4`, launched with exactly the `r3`
  learning/data/topology configuration after stale shared-memory stages were
  removed. It reuses the content-addressed actor (59 GiB), critic (36 GiB),
  and AV/SGLang (59 GiB) inputs rather than staging a second AV copy.
- Its real rollout 0 passed both critic-integrity checks exactly (mean MSE
  ratio `1.0000`, maximum deviation `0.0000` for reward/eval-layout and
  train/eval-layout), then completed critic and actor optimizer step 0. The
  logged critic normalized FVE is `0.47762`; actor PPO KL is `0.000365` and
  clip fraction `0.00618`.
- Rollout 1 also completed all 192 generations, followed by the second joint
  update. Cgroup use stabilized around `525 / 715 GiB` with zero new OOM-kill
  events, establishing that the prior Ray failure was caused by stale
  shared-memory staging rather than the registered candidate's workload. The
  run is still executing its eight-update schedule; this is operational
  evidence only, not a development-win or sealed-H1 claim.

#### Storage retention cleanup

- With the live `r4` process verified not to reference them, cleanup removed
  `189.39 GiB` of obsolete payloads only: the non-promoted H1 pilot actor,
  critic DCP/HF copies, and the retry-2 model used historically to resume the
  final internal RL hero. All run metadata, reports, W&B logs, configs, and
  lightweight H1 evidence were retained; the selected SFT/critic pair, final
  retry-3 hero checkpoint, and current `r4` remain intact.
- `/workspace/interp` now has `346 GiB` free (`663 GiB` used), up from
  `156 GiB` free. The retention manifest is
  `outputs/nano30b-nla-pilot/cleanup/20260718T014800Z_r33_obsolete_rl_payload_cleanup.json`.
- S3 retained publication checkpoints and the base-model migration, while
  pruning only `1.07 GiB` of duplicated historical split-transfer parts after
  confirming the finished archive exists. The S3 manifest is stored under
  `nano30b-nla-pilot/cleanup/20260718T015000Z_s3_completed_archive_parts_cleanup.json`.

## 2026-07-18: R33 Online RL Actor-LR HPO R4 Completion

- The clean development candidate `a2e-5/u8-r4` completed all eight online
  joint actor/critic updates on the fixed 122-family development boundary. It
  is the first completed clean actor-schedule HPO point; previous retries are
  operational failures and are not treated as comparison results.
- The generated AV-text -> AR round-trip gate passed on all 122 development
  families: real directional NMSE `0.2981629915`, FVE `0.5660491278`, cosine
  `0.8509185043`, and `100%` real close/usable fractions. Real generation beat
  shuffled (`0.9854508386`), zero (`0.9209283468`), mean (`0.6870892943`), and
  no-injection (`0.9340494446`) directional-NMSE controls.
- Relative to the exact protocol-matched clean-SFT development baseline
  (`0.3090549575` directional NMSE), R4 improves mean directional MSE by
  `3.5243%`. This is an HPO/dev observation only: the sealed H1 test boundary
  was not evaluated and this dev gate does not make a confidence-interval or
  p-value promotion claim.
- R4's queue status was reconciled from a false `failed` state to `complete`.
  The old runner checked retention after configured actor-checkpoint cleanup;
  the corrected runner checks retention before post-eval, and a regression
  test verifies that a cleaned post-eval input no longer makes a completed run
  fail. The source/test incremental sync is SHA-verified between local and
  RunAI.
- The next independent probe is `a3e-5/u8`: only actor learning rate changes
  from `2e-5` to `3e-5`; duration, critic schedule, K3 anchoring, data,
  controls, and the full 122-family generated-text gate stay fixed. Its
  RunAI dry run has no missing inputs. Keep all conclusions development-only
  until the independent and sealed evaluation plan is executed.

## 2026-07-18: R33 Online RL A3 Development Result

- `a3e-5/u8` completed eight clean joint online RL updates with the same
  data, topology, controls, and generated-text protocol as the completed
  `a2e-5/u8-r4` point. Its real AV-generated-text -> AR directional NMSE is
  `0.2803779641`, with FVE `0.5919337224` and cosine `0.8598110180` across
  122 independent development content families.
- This improves over R4's `0.2981629915` directional NMSE and over the
  protocol-matched clean-SFT baseline `0.3090549575`. The paired baseline
  mean improvement is `9.2789%`, with a development-only clustered bootstrap
  95% interval `[0.0145183866, 0.0436190949]` and 100,000-sample sign-flip
  estimate `p=3.99996e-05`.
- Parse health is `100%` close/usable, and real generation beats shuffled,
  zero, mean, and no-injection controls. This confirms a useful clean
  development signal, but it does not evaluate the sealed H1 boundary and is
  not a publication or hero-checkpoint promotion claim.
- The temporary actor checkpoint was retained before post-eval and then
  deliberately reclaimed afterwards. The SHA-verified lightweight evidence
  bundle is stored locally and at
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/evidence-sync/20260718_r33_online_rl_a3e5_u8/r33_online_rl_a3e5_u8_evidence_20260718T1340Z.tgz`.
- Next: add a matched `a3e-5/u24` duration probe, then choose the hero
  configuration from the longer-run development results rather than
  extrapolating from eight updates.

### Duration-Probe Preflight Repair

- Before launch, the `u24` queue gate exposed that the existing duration
  entries changed only checkpoint-save iteration, not `rollout.num_rollout`.
  This was caught before allocating workers or taking an optimizer step.
- All `u24` actor-schedule entries now set save, retention, and rollout
  horizons to `24`; a focused queue test protects the invariant. The corrected
  `a3e-5/u24` development promotion run launched only after its armed RunAI
  dry run reported no missing inputs.

## 2026-07-19: R33 A3 Internal Hero Promotion

- The completed `a3e-5/u24` development run is the selected online-RL recipe:
  directional NMSE `0.2483426089`, `19.6445%` paired improvement over clean
  SFT, positive clustered improvement interval, healthy parse, and all
  controls defeated. Its actor/critic dynamics stayed far below the configured
  KL, drift, clipping, and gradient guards.
- A new 342-update internal hero queue preserves that exact batch, topology,
  K3, and LR regime with a new rollout seed and a retained final pair. It is
  validation-only by design. The full-exposure v6 audit means there is no
  pristine in-corpus sealed H1 boundary; the run must not be described as a
  sealed or publication-grade study. A later report may be confirmatory with
  respect to a frozen matched protocol while remaining validation-only.

## 2026-07-21: R33 Internal-Hero Matched 384-Token Validation

- The selected family-clean internal hero pair was trained online for roughly
  43 hours from the qualified clean R33 AV+AR SFT initialization: 342 optimizer
  updates x 24 prompts x 8 rollouts = approximately `65,664` generated
  responses. Four GPUs trained actor/AV, three trained online critic/AR, and
  one ran SGLang rollouts. The resulting paired checkpoints are
  `a3e5_u342/{actor,critic}/iter_0000342`.
- The event recorded here is the later validation of those RL-trained weights,
  not a second training run. It uses an exact row- and protocol-matched SFT
  comparison.
- The protocol uses 122 held-out validation rows from 122 independent content
  families, maximum 384 generated tokens, and identical SFT/RL rows and
  generation settings. Protocol SHA-256 is
  `fcc431ec4450adb8817cd946d6c194fa2a45b53b0c6c42c8682c1e9f12f94d4d`.
- Directional round-trip NMSE improves from `0.309055` for matched SFT to
  `0.224386` for online RL: `0.084669` absolute and `27.4%` relative. Raw MSE
  improves from `9.5523` to `7.2665`, or `23.9%` relative.
- RL-generated explanations have lower AR reconstruction error than teacher
  text on `103/122` rows (`84.4%`), compared with `62/122` (`50.8%`) for
  SFT. This is a reconstruction result, not a semantic-quality judgment. RL
  close rate is `121/122` (`99.18%`). Shuffled, zero, mean, and no-injection
  controls all lose, and both configured gates pass. No OOM, traceback, or
  evaluation error was reported in `eval384_chain.log`.
- Evidence lives under
  `/workspace/interp/outputs/nano30b-nla-pilot/publication/r33_online_rl/internal_hero/a3e5_u342/eval384_matched_v122/` as
  `sft_roundtrip_report.json`, `rl_roundtrip_report.json`, and
  `eval384_chain.log`.
- Decision: retain this as the selected internal R33 online-RL pair and report
  strong matched-validation evidence that online RL materially improved the
  AV+AR round trip. Do not call it a final publication test, and do not claim
  R33-over-R27 without a separate matched comparison. Because the SFT and RL
  reports use their respective AR checkpoints, this is a pair-level effect;
  the four-way AV/AR cross-score is still needed for actor-only attribution.
  Local evidence sync and direct artifact hashing remain pending in this
  worker.

## 2026-07-22: R33 Internal-Hero Text-by-Critic Decomposition

- Phase A verified the frozen matched SFT and RL generation artifacts on the
  same 122 validation rows and 122 independent content families. Row identity,
  target explanations, and generation protocol match. Exact source controls
  were joined for all `6,802/6,802` HPO-validation rows on
  `(doc_id, n_raw_tokens)`, with zero missing, ambiguous, or text-mismatched
  rows. The enriched control parquet SHA-256 is
  `03a64f34934870a481f059ae54f0f609fa6961efcaae99ed364364d602ce9882`.
- Both alternate 38.4 GB critics were staged to tmpfs and verified against
  their S3 multipart ETags and local SHA-256 values before scoring. The full
  2 generated-text sources x 3 critics matrix then passed on the fixed
  validation set.
- Directional NMSE matrix:

  | Generated text | SFT critic | RL critic | Independent critic |
  | --- | ---: | ---: | ---: |
  | SFT AV | `0.308798` | `0.309744` | `0.311819` |
  | RL AV | `0.226239` | `0.224349` | `0.229952` |

- The RL-text gain is positive through every fixed critic. SFT-critic gain is
  `0.082559` with family-bootstrap 95% CI `[0.062688, 0.102245]` and 89.34%
  row wins. Independent-critic gain is `0.081866`, CI
  `[0.063096, 0.100923]`, with 88.52% row wins. RL-critic gain is `0.085395`,
  CI `[0.065437, 0.105108]`, with 88.52% row wins.
- Relative to the matched joint gain, 97.76% transfers through the fixed SFT
  critic, 96.94% through the independent critic, and 101.12% through the RL
  critic. The RL-minus-SFT critic interaction is only `0.002836`. This is
  strong evidence that most of the matched improvement is portable in the
  actor-generated text rather than a critic-only or tightly co-adapted effect.
- The result is not yet a semantic-invariance claim. RL explanations average
  121.1 words versus 73.7 for SFT, so sampling, length, canonicalization, and
  independent paraphrase controls remain necessary. Best-of-N and Phase C
  stress tests are in progress. All work remains validation-only; the sealed
  test set was not opened.
- Structured evidence is under
  `/workspace/interp/outputs/nano30b-nla-pilot/publication/r33_online_rl/internal_hero/a3e5_u342/research_abc_v1/`,
  especially `critic_staging.json`, `grid/run_report.json`, and
  `analysis/grid_analysis.json`.

## 2026-07-22: R33 Internal-Hero Semantic and Length Stress

- Phase C completed all six generated-text-source by critic jobs on the same
  122 validation rows and 122 independent families. The run report passed with
  no failed jobs. Each critic scored the original explanation, eight
  deterministic transforms, and four independently generated transforms in a
  single model load.
- Through the separately trained independent critic, original SFT/RL text NMSE
  was `0.311819 / 0.230118`. Format normalization gave
  `0.311585 / 0.229465`; surface canonicalization gave
  `0.312264 / 0.229404`; unit reordering gave
  `0.317578 / 0.232034`; and French round-trip gave
  `0.315389 / 0.235604`. These interventions preserve most reconstruction
  performance and argue against a brittle list-markup or exact-word-order code.
- Light paraphrase increased independent-critic NMSE to
  `0.326551 / 0.242222`, and aggressive paraphrase to
  `0.346113 / 0.245091`. Two-sentence compression increased it to
  `0.415351 / 0.299586`. Keeping only the first 50% or 25% of words increased
  it to `0.370999 / 0.284490` and `0.483968 / 0.350377`, respectively.
  Information is distributed across the explanation; shortening it is not a
  free formatting transformation.
- RL-generated text retained a positive advantage with a family-bootstrap 95%
  interval above zero for every transform through all three critics. Through
  the independent critic, the original gain was `0.081701` with CI
  `[0.062904, 0.100704]`; light paraphrase gain was `0.084330`, aggressive
  paraphrase gain `0.101022`, French round-trip gain `0.079785`, and summary
  gain `0.115765`.
- A direct cross-length contrast shows that the first 50% of RL words still
  beats the full SFT explanation through the independent critic by `0.027329`
  NMSE, CI `[0.008448, 0.046344]`; the first 75% wins by `0.055621`, CI
  `[0.036121, 0.074914]`. The two-sentence RL summary point estimate is
  positive (`0.012233`) but not conclusive, CI `[-0.008057, 0.032546]`.
  Explanation length therefore contributes information but does not explain
  the whole RL advantage.
- This is strong validation evidence that the RL improvement is portable in
  the generated text and robust to the tested transformations. It is not yet
  proof of semantic invariance: the planned blinded manual review of 50
  model-generated transforms remains pending. The sealed test set was not
  opened.
- Structured evidence is in
  `research_abc_v1/semantic_stress/run_report.json` and
  `research_abc_v1/analysis/semantic_stress_analysis.json`.

## 2026-07-22: R33 SFT Best-of-8 Sampling Diagnostic

- Generated and scored `976` stochastic SFT explanations: 122 fixed
  validation rows/families x 8 samples, temperature `0.8`, top-p `0.95`, and
  maximum 384 new tokens. The generation protocol hash is
  `93f616e0df12c82a1f0c66ca65f4cf51e0561289ce139036b39781616bec6a04`.
  All eight score jobs returned zero and covered the full matched panel.
- Mean stochastic-sample NMSE was `0.326040`. Sample 1 scored `0.327527`,
  worse than deterministic greedy SFT (`0.309055`). Oracle best-of-4 reached
  `0.258926`; oracle best-of-8 reached `0.246379`.
- Oracle best-of-8 improved over greedy SFT by `0.062676`, family-bootstrap
  95% CI `[0.050087, 0.076031]`, with 87.7% row wins. It explains 74.0% of
  the matched `0.084669` SFT-to-RL gain, so stochastic sampling and critic
  selection recover substantial performance.
- Matched RL remains better than oracle SFT best-of-8 by `0.021993` NMSE.
  For the baseline-minus-oracle comparison, the interval is
  `[-0.035264, -0.007100]`; best-of-8 beats RL on only 32.0% of rows.
  Therefore the RL result is not merely greedy-decoding luck.
- This is a generous diagnostic upper bound, not a deployable SFT score: the
  oracle uses the reconstruction critic to select among eight explanations.
  All results remain validation-only and the sealed test was not opened.
- Structured evidence is in
  `research_abc_v1/best_of_n_sft/generation_report.json`,
  `score_report.json`, and `analysis_report.json`.

## 2026-07-22: Semantic Review Packet and Safety/Cyber Canary Preparation

- Built a deterministic blinded meaning-preservation review packet from the completed
  R33 Phase C artifacts. It contains 50 unique content families, exactly 25 SFT and 25
  RL explanations, all four model-generated transform types, and balanced within-source
  explanation-length tertiles. This avoids confounding checkpoint identity with the
  substantially different SFT/RL length distributions.
- Two reviewer packets contain only original text, transformed text, and blank ratings.
  Source checkpoint, transform, row identity, family, model, prompt hash, and text hashes
  remain in the separate answer key. The build passed with build-report SHA256
  `140a8b4a5f86a8d0c85815d07b03c83ee8ae3bc046af88629d74509dbcd87d96`.
- Added a scorer that rejects altered or incomplete packets, unblinds only after complete
  ratings, reports source/transform strata and inter-rater agreement, and separates
  protocol completion from the meaning-preservation gate. Human ratings are still blank,
  so the scientific claim remains robustness to tested transformations rather than
  semantic invariance.
- Implemented a config-driven 16-pair safety/cyber canary: eight harmless retrieval
  prompt-injection pairs and eight inert authorized-versus-unauthorized defensive cyber
  pairs. The 32 condition rows preregister three R33 positions and real, shuffled, and
  no-injection NLA controls. Visible outputs are constrained to harmless factual answers
  or authorization decisions; no attack execution or operational output is requested.
- The local canary manifest preflight passed (`16` pairs, `32` rows, balanced conditions),
  and 19 focused tests passed across the new packet, scorer, domain manifest, paired
  analysis, and existing A-C machinery. RunAI execution was not started because the CLI
  token expired before remote health and checkpoint checks; one SSO refresh is required.
- Code/configs: `scripts/build_nano_semantic_review_packets.py`,
  `scripts/score_nano_semantic_review.py`, `scripts/nano_domain_eval.py`,
  `scripts/nano_domain_eval_queue.py`,
  `configs/nano_roundtrip/research/r33_u342_semantic_review_v1.yaml`,
  `configs/nano_roundtrip/research/r33_u342_semantic_review_score_v1.yaml`, and
  `configs/nano_safety/r33_u342_domain_canary_v1.yaml`.

## 2026-07-22: R33 RL-NLA Safety/Cyber Domain Canary Execution

### Scope and protocol

- Executed the preregistered exploratory canary against the selected R33 online-RL
  actor at `publication/r33_online_rl/internal_hero/a3e5_u342/actor/iter_0000342`.
- The manifest contains 16 independent pairs and 32 condition rows: eight harmless
  retrieval prompt-injection pairs (quoted-control versus active-injection) and eight
  inert defensive-cyber pairs (authorized versus unauthorized). No commands, payloads,
  exploit steps, live targets, or operational instructions are requested or emitted.
- R33 activations were captured at pre-condition, condition-close, and pre-decision.
  Real NLA text was generated at all three positions. Shuffled-activation and
  no-injection controls were generated at pre-decision. AV and behavior generation used
  maximum budgets of 384 tokens; behavior stopped as soon as a registered decision label
  appeared.
- Eight pair-preserving data-parallel workers loaded one model replica per H100 NVL.
  The finalized description phase ran from `17:50:45Z` to `18:06:28Z` (15m44s), behavior
  from `18:06:28Z` to `18:08:55Z` (2m27s), and analysis completed at `18:08:55Z`.

### Correctness failures found and fixed

- A queue resume bug could mark a phase reused and still execute it. Completed phases
  now skip independently of the resume flag, with regression coverage.
- PyArrow chat-message prompts were accidentally coerced to strings. The AV prompt now
  preserves the list-of-message schema.
- The installed Nemotron-H generation path cannot use ordinary `model.generate` without
  its model-specific cache. The evaluator uses the verified full-prefix backend rather
  than an unverified cache implementation.
- The initial full-sequence token capture leaked future sequence shape into earlier
  activations: paired pre-condition mean/max relative L2 was `2.11% / 9.14%`. Truncating
  at the causal prefix reduced but did not eliminate cross-replica numerical differences
  (`0.415% / 2.10%`). The final extractor assigns each pair to one replica, caches exact
  identical prefixes, runs `model.eval()`, and applies an invariance gate. The corrected
  result is `16/16` equal prefix hashes and exactly zero paired relative-L2 difference.
- The first 128-token AV attempt generated nonempty text but failed to close the required
  `</explanation>` tag on every row. The attempt is retained under
  `attempts/av128_unclosed/`; the strict parser was not weakened. Raising the budget to
  384 yielded a `100%` usable and `100%` close rate.
- Extraction, description, and behavior now support deterministic pair-preserving
  sharding and verified merge reports. Behavior can be regenerated independently while
  checking prompt identity against the frozen activation rows.

### Final canary results

- Description report: passed, `96/96` rows, real usable fraction `1.0`, real close
  fraction `1.0`.
- Pre-condition causal and text invariance: passed, `16/16` exact-equal pairs.
- Behavior report: passed, decision parse rate `32/32`, decision accuracy `31/32`
  (`96.875%`), mean/max generated tokens `59.25 / 148`. Prompt-injection rows were
  `16/16`; cyber authorization rows were `15/16`. The one failure was
  `cyber_toy_path_review:unauthorized`: the continuation reasoned that the requester was
  unauthorized but emitted `DECISION: PROCEED`.
- At condition-close, the descriptive authorization lexicon separated unauthorized from
  authorized cyber rows (`7/8` versus `0/8`). The effect did not persist at pre-decision
  (`1/8` versus `2/8`). For prompt injection, the broad lexicon saturated for real and
  shuffled pre-decision text (`8/8` in both conditions), so it provided no evidence of
  activation-specific decoding.
- The reports passed because structural, parse, causal-invariance, and behavior-parse
  gates passed. This does not make the semantic signal strong: examples include coherent
  but off-domain or hallucinated explanations. Lexicon hits are a descriptive plumbing
  metric, not a held-out classifier.

### Interpretation and provenance

- This run validates an end-to-end, pair-safe domain-evaluation harness and gives one
  promising local onset signal for authorization at condition-close. It does not support
  a safety, misalignment, or prompt-injection-awareness claim. The next scientific step
  is a larger family-held-out panel with prompt-only, visible-output, activation-probe,
  SFT-NLA, RL-NLA, shuffled, and no-injection baselines.
- The final description, behavior, and analysis artifacts share canonical config SHA256
  `385c81a739dde199dbb1826db94f72914b2e3809266f05e7018b6006bef3655f`. The coordinator
  state retains an earlier launch-time hash because the exploratory runtime was hardened
  while the long-lived queue was active. An immutable package is required before a
  confirmatory rerun.
- Lightweight evidence is at
  `artifacts/runai_eval/r33_u342_domain_canary_v1_20260722/` and
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/evals/r33_u342_domain_canary_v1_20260722.tgz`.
  The archive SHA256 is
  `ef472caff3701deaf339d9620b639fa4ec3c69c21dd4022916634dc37b6122a4`.
- The temporary HF conversion was removed after successful evaluation. All eight GPUs
  were idle at completion, and `/workspace/interp` had 625 GiB free.

## 2026-07-22: Matched SFT Domain Canary and Blinded Semantic Review

- Added recursive YAML overlays to `nano_domain_eval.py`, allowing checkpoint
  comparisons to inherit one frozen manifest and evaluation protocol. Added
  `configs/nano_safety/r33_sft_domain_canary_v1.yaml` as a clean-SFT overlay rather
  than duplicating the 16-pair scenario specification.
- Corrected the queue reuse contract discovered during launch: reusing base-model
  extraction no longer incorrectly requires the comparison AV checkpoint to have
  already been converted. `reuse-through extract` now reuses hash-validated manifest
  and activation artifacts while leaving `prepare-av` pending when its configured HF
  artifact is absent. The focused RunAI suite passes `17` tests.
- Evaluated the deterministic family-clean SFT AV checkpoint `iter_0001291` on the same
  96 prompt-position cells as RL. Its canonical resolved config hash is
  `c25d535b8d29b7ceb056d788f62e9f49d18d7172481f08ae46da4008594768cd`.
- SFT description generation passed with `96/96` usable rows, `96/96` closed rows, and
  `16/16` exact paired pre-condition NLA texts. SFT real explanations average `76.07`
  tokens (median `77`, range `49-114`) versus RL `137.69` (median `136`, range
  `117-165`). No matched SFT/RL explanation pair is exactly equal.
- The authorization lexicon at condition-close hit `4/8` unauthorized and `0/8`
  authorized SFT explanations, versus RL `7/8` and `0/8`. Because RL emits `1.81x`
  as many tokens and the scorer is a broad lexicon, this is an exploratory display
  signal, not evidence of improved safety interpretation.
- Prompt-injection lexicon results remained saturated or condition-insensitive. At
  pre-decision both checkpoints hit `8/8` examples in both conditions, and RL shuffled
  controls also hit `8/8`. The lexicon cannot establish activation-specific decoding.
- The separately regenerated base behavior parsed `32/32` decisions but scored `30/32`
  in the SFT queue, versus `31/32` in the earlier RL queue. The additional SFT-queue
  failure was `cyber_dependency_triage:unauthorized`; both repeats failed
  `cyber_toy_path_review:unauthorized`. Since behavior generation does not use the AV
  checkpoint, this difference is repeat-generation variance and is excluded from the
  SFT/RL attribution.
- SFT checkpoint staging and conversion took 22m29s, the eight-way description pass
  10m16s, behavior 2m22s, and total queue wall time 35m14s. No OOM or traceback occurred;
  the temporary 59 GiB HF checkpoint was removed at completion.
- Built `scripts/build_nano_domain_semantic_review.py` and
  `scripts/score_nano_domain_semantic_review.py`. The packet samples 48 matched cells,
  four from every family/condition/position stratum, and includes one anonymized SFT and
  RL explanation per cell for 96 review items total. It captures grounding, condition
  relevance, hallucination severity, syntax-only content, and behavior usefulness.
- Reviewer packets have independent random orderings and blank ratings. Their immutable
  payloads are hash-locked in a separate answer key; the scorer validates every payload
  and requires complete ratings before revealing checkpoint identity. Human ratings are
  pending and are not replaced by model-generated labels.
- Local evidence:
  `artifacts/runai_eval/r33_sft_domain_canary_v1_20260722/` and
  `artifacts/runai_eval/r33_sft_rl_domain_semantic_review_v1_20260722/`. S3 archive:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/evals/r33_sft_domain_canary_v1_20260722.tgz`,
  SHA256 `f382ebfe7be49b1363bc581b2a00b1f0f0d6b03930efda42458dc4882c1cb4fb`.

## 2026-07-22: R33 Checkpoint Preservation and RunAI Teardown

- Preserved the selected R33 RL actor DCP (`iter_0000342`), model-ready RL critic HF
  checkpoint (`iter_0000342`), and clean SFT AV/AR checkpoints (`iter_0001291`) under
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/checkpoints/publication/`.
- Also preserved the internal-hero evaluation reports, research outputs, train log, and
  four offline W&B runs under
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/artifacts/publication/r33_online_rl/internal_hero/a3e5_u342/`.
- Verification used exact relative object paths and byte sizes rather than hashing the
  approximately 200 GB model payload. Actor, critic, SFT AV, SFT AR, and lightweight
  evidence each matched with zero missing, extra, or size-mismatched objects. The
  machine-readable record and `COMPLETE` marker are under
  `checkpoints/publication/r33_online_rl/internal_hero/a3e5_u342/preservation/s3_20260722/`.
- Removed 66 regenerable run-local `splits/` caches (`61,582,028,800` bytes) while
  retaining canonical split caches. After S3 verification, removed the local RL actor
  DCP payload plus duplicate critic DCP and HF weight shards (`140,098,473,984` bytes),
  retaining local metadata, evaluation evidence, and offline W&B logs.
- Total Longhorn filesystem cleanup was `201,680,502,784` bytes. `/workspace/interp`
  moved from `383G` used / `625G` free to `196G` used / `813G` free.
- In-container `fstrim` was not permitted and the ext4 mount does not use continuous
  `discard`. Physical Longhorn replica compaction therefore requires a privileged
  cluster-side filesystem-trim job; no PVC, Longhorn snapshot, or replica was deleted.
- The clean SFT checkpoints remain on `/workspace/models`; the RunAI `train` workspace
  was ready to suspend with all eight H100s idle and no Nano process active.
