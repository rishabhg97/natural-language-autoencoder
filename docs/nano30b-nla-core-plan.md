# Nano30B Natural Language Autoencoder Core Plan

<!-- R33-HERO-BASELINE-PROTOCOL-INVALIDATED -->

> [!CAUTION]
> Publication status (`2026-07-10`): the archived `30.97% / 32.34%` hero
> comparison is not publication-valid because the 512-row SFT baseline mixed
> two generation protocols. Figures below remain historical internal-gate
> evidence only. The active path is deterministic family-clean SFT followed by
> a protocol-matched round trip and preregistered confirmatory RL. See
> `docs/reviews/2026-07-08-r33-rl-hero-publication-audit.md`.

This is the canonical operating plan for the Nano30B NLA pilot. Use
[execution_log.md](execution_log.md) for current phase tracking and run history,
[cluster_runbook.md](cluster_runbook.md) for commands, and
[issues_iter1.md](issues_iter1.md) for archived detailed rationale, longer math,
and discarded alternatives. This file should stay short enough to drive
decisions.

## Tracking Structure

| File | Role |
|---|---|
| [../README.md](../README.md) | Entry point and current doc map |
| [execution_log.md](execution_log.md) | Phase ledger, run outcomes, blockers, additions/subtractions |
| [cluster_runbook.md](cluster_runbook.md) | Exact cluster commands and expected outputs |
| [issues_iter1.md](issues_iter1.md) | Detailed issue archive |

Current active phase:

```text
clean AV validation and protocol-matched clean SFT round-trip gating
```

Next major implementation target:

```text
independent clean critic/AR -> finite RL stability grid -> preregistered replication
```

Historical internal milestone, invalidated for publication:

- clean component-full R33 AR and AV SFT checkpoints;
- corrected-K3 R33 RL actor `iter_0000342`;
- final `512/512` generated-text round-trip gate passed with
  `30.97% / 32.34%` validation/test improvement over exact matched SFT rows.

See `docs/current_state.md` for the active lineage and
`docs/runs/r33_rl_hero_20260708.md` for the historical record. A fresh
row-matched R27 comparison is required only if the publication claim asserts
R33 superiority over R27.

## Goal

Train a single-site Natural Language Autoencoder (NLA) for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.

The scientific contract is unchanged from NLA:

```text
frozen target Nano: x, tau -> h_b = R_b^target(x)_tau
AV: h_b -> z
AR: z -> h_hat_b
loss/eval: h_hat_b should reconstruct h_b through the text bottleneck
```

The project is feasible as a staged research pilot, not as a direct port of the released Qwen/Gemma NLA stack. The likely failure mode is not "NLA cannot work on Nano"; it is a boundary, cache, template, adapter, or checkpoint mismatch that creates plausible-looking text with invalid reconstruction science.

## Source-Grounded Facts

Target facts to verify from the loaded checkpoint, not prose docs:

| Item | Working assumption |
|---|---|
| HF model | `NemotronHForCausalLM` / `nemotron_h` |
| Hidden size | `2688` |
| Blocks | `52` `NemotronHBlock` entries |
| Pattern | `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME` |
| Block types | `M` = Mamba-2, `E` = MoE, `*` = GQA/attention |
| HF wrapper | `.backbone` |
| HF blocks | `.backbone.layers` |
| HF final norm | `.backbone.norm_f` |
| HF embeddings | `.backbone.embeddings` |
| Cache | attention KV plus Mamba conv/SSM states |
| Experts/token | HF currently indicates `6`; some Bridge prose has said `5`; loaded config and weight shapes are authoritative |

Critical rule:

```text
Trust loaded checkpoint config, module tree, and equivalence tests over secondary documentation.
```

## Mathematical Contract

Use residual-boundary notation:

```text
R_0 = Embed(x)
R_{i+1} = R_i + F_i(RMSNorm_i(R_i))
F_i in {Mamba2_i, MoE_i, GQA_i}
```

For source text `x`, explained token position `tau`, and boundary `b`:

```text
h_b = R_b^target(x)_tau in R^2688
```

If extracting after zero-based block `k`:

```text
b = k + 1
post-block-33 activation = h_34 = R_34
```

Normalized reconstruction loss:

```text
N_s(v) = s * v / (||v||_2 + eps), where s = sqrt(d)

L_AR = (1/d) ||N_s(h_hat_b) - N_s(h_b)||_2^2
     = 2(1 - cos(h_hat_b, h_b))
```

Report relative reconstruction improvement, not classical variance-explained framing, under direction-only normalized MSE:

```text
RRI_b = 1 - E[L(h_b, h_hat_b)] / E[L(h_b, mean_b)]
```

`mean_b` must be fit on the training split only. Report whether `N_s(mean_b)` is used, flag near-zero `||mean_b||`, and pair `RRI_b` with centered raw-vector `R^2`.

## Core Design

Use three injection tracks, but do not treat all of them as equal implementation targets.

| Track | Role | Definition | Pilot status |
|---|---|---|---|
| A | Paper-faithful baseline | replace one input embedding row with scaled `h_b` | must run |
| C | Residual-boundary oracle | patch `R_b[p] <- T(h_b)` during HF split-forward prefill | must probe as diagnostic |
| B | Learned embedding adapter | replace input embedding with `S_phi(h_b)` | only if A weak and C strong |

Track A:

```text
E(u)[p] <- alpha * h_b / ||h_b||
```

Track C:

```text
run AV prompt u to R_b
patch sentinel position p: R_b[p] <- T(h_b)
continue through G_b..G_{L-1}
build downstream attention KV and Mamba conv/SSM states from the patched computation
```

Track B:

```text
E(u)[p] <- S_phi(h_b)
```

`S_phi` is trained through downstream AV/AR reconstruction. Prefix alignment can be used as initialization or weak regularization, but is not the main objective.

Decision rule:

```text
If A works: use A.
If A is weak and C is strong: try B.
If A and B fail but C is strong: treat C as an experimental fallback/oracle variant.
If A and C both fail: revisit boundary, prompts, AR, extraction, and data before scaling.
```

Track C is not a serving path in the pilot. It is a diagnostic for whether the embedding channel is the bottleneck.

## Pilot Minimum

The first pilot should answer only two scientific questions:

1. Can Nano's AV consume `h_b` through at least one injection channel and produce text that AR can use?
2. Can Nano's AR reconstruct `h_b` from explanation text above mean, shuffled, and random matched-norm baselines?

Minimum sequence:

1. Pin model environment, tokenizer, chat template, reasoning mode, kernel mode, and dtype.
2. Add Nano architecture adapters for `.backbone`, `.backbone.layers`, `.backbone.norm_f`, and `.backbone.embeddings`.
3. Prove extraction identity for the selected boundary.
4. Run the smallest boundary audit on `R_34` plus one earlier candidate, normally `R_27`.
5. Probe Track A input-embedding injection.
6. Probe Track C residual-boundary oracle locally in HF.
7. Train frozen-prefix AR head-only baseline.
8. Run held-out round trip on AV-generated explanations:

```text
h_b -> AV -> z -> AR -> h_hat_b
```

Everything else is staged after signal.

## Boundary Choice

Pilot must-have candidates:

| Boundary | Meaning | Why |
|---|---|---|
| `R_34` | post-GQA block 33 | middle-late, close to prior NLA convention |
| `R_27` | post-GQA block 26 | earlier, more AV suffix capacity |

Deferred ablations unless smoke results are ambiguous:

| Boundary | Use |
|---|---|
| `R_33` | pre-GQA comparison for block 33 |
| `R_35` | post-MoE immediately after `R_34` |
| `R_43` | late semantic ablation with less suffix capacity |

Minimum boundary checks:

- activation norm p05/p50/p95,
- repeated-run drift,
- batch-vs-solo drift,
- basic correct-vs-shuffled causal/logit effect.

Defer deep router-load entropy, token-position buckets, and broad causal patch grids until after the first successful SFT or if boundary results are ambiguous.

## Nano Adapter Requirements

Before any training:

- locate inner model as `.backbone`,
- locate blocks as `.backbone.layers`,
- locate final norm as `.backbone.norm_f`,
- locate embeddings as `.backbone.embeddings`,
- add `embeddings.weight` to embedding-key discovery,
- include `norm_f` in final-norm stripping,
- verify value head save/reload.

Hidden-state trap:

```text
For b < L, output_hidden_states[b] should correspond to raw R_b.
For b = L, final returned hidden state is post-norm_f, not raw R_L.
Never use last_hidden_state as raw residual unless final norm is removed or bypassed.
```

Run prefix/extraction equivalence before Track A/C probes:

```text
hooked R_b ~= output_hidden_states[b] ~= prefix_forward_to_R_b
```

Run this under the exact prompt mode used for training:

- raw completion,
- chat with `enable_thinking=false`,
- AV prompt with injection marker but no injected vector,
- AR critic prompt ending at the prediction marker.

## Chat Template Control

Default to reasoning off:

```yaml
enable_thinking: false
```

Record:

- tokenizer revision,
- remote-code revision,
- chat template text or hash,
- rendered golden prompt hash,
- `add_generation_prompt`,
- parser mode,
- prompt format.

Reasoning-on can be an ablation only after reasoning-off passes reconstruction and length-control gates.

## Teacher Labels

Teacher summaries are distillation warm start, not evidence of unsupervised NLA faithfulness.

Allowed use:

- warm-start AV text style,
- warm-start AR text-to-vector mapping,
- reduce early training instability.

Required controls:

- teacher summary with activation shuffled,
- source-context summary only,
- token identity only,
- token position only,
- held-out AV-generated round trip after AV SFT.

Do not report teacher-SFT performance as evidence that Nano activations have faithful natural-language explanations.

## Decision Gates

### Pilot Minimum

| Gate | Pass condition | If failed |
|---|---|---|
| Model/template pinning | model revision, tokenizer, remote code, template hash, `enable_thinking`, dtype, kernel mode recorded | stop |
| Nano adapter | `.backbone`, `.layers`, `.norm_f`, embeddings, value head save/reload work | fix adapter |
| Extraction identity | hook, `output_hidden_states[b]`, and prefix extraction agree for `b < 52` | fix boundary/off-by-one |
| Boundary smoke | `R_34` and `R_27` have stable extraction and nontrivial correct-vs-shuffled effect | choose another boundary |
| Track A probe | input-embedding injection has non-flat scale response and correct-vs-shuffled signal | compare Track C |
| Track C oracle probe | split-forward no-patch, self-replacement, patched-cache decode, and correct-vs-shuffled pass | do not rely on C |
| Frozen AR | head-only AR beats mean/shuffled/random baselines or establishes a clear lower bound | tune AR/prompt/boundary |
| Held-out round trip | AV-generated `z` reconstructs better than shuffled/random/teacher-leakage controls through frozen AR | stop before scale |

### Before PEFT

| Gate | Pass condition |
|---|---|
| Bridge/HF equivalence | `logits_Bridge ~= logits_HF`, `R_b_Bridge ~= R_b_HF` |
| Router equivalence | top-k cardinality, route loads, and router config match loaded checkpoint |
| Adapter reload | LoRA/value-head reload reproduces logits and `R_b` metrics |
| Sidecar completeness | model/data/extraction/normalization/injection/training/eval metadata are present |

### Before Serving

| Gate | Pass condition |
|---|---|
| Track A/B server path | raw `input_embeds` are accepted and first-token logits match local HF |
| Cache safety | prefix/radix cache disabled or injection hash included in cache key |
| Track C serving | out of pilot scope unless A/B fail and C is the only viable path |

### Before RL

| Gate | Pass condition |
|---|---|
| AV SFT | real activation beats shuffled/random/teacher-leakage controls through frozen AR |
| Reward sanity | reward is length-normalized and checked against frozen AR snapshot |
| Causal agreement | reconstructed vector imitates original `h_b` better than mean/shuffled controls on a small patch set |

### Implementation Regression QC

| Gate | Pass condition |
|---|---|
| Released Qwen/Gemma checkpoint QC | after the Nano implementation path exists, reproduce reference NLA sidecar loading, injection scaling, normalized-MSE/RRI scoring, checkpoint reload, and correct-vs-shuffled AR gap on a released Qwen or Gemma NLA checkpoint |

## Sidecar Schema

Keep the sidecar grouped and versioned. Do not enumerate every transient runtime field inline in the experiment notes.

Required blocks:

```yaml
schema_version:
model_env:
  model_id:
  checkpoint_revision:
  remote_code_revision:
  tokenizer_revision:
  bridge_container:
  bridge_commit:
  dtype:
  mamba_kernel_mode:
chat_template:
  enable_thinking:
  template_hash:
  rendered_golden_prompt_hash:
  add_generation_prompt:
data_split:
  dataset_sources:
  dataset_revisions:
  split_ids_or_hashes:
  tau_selection_policy:
extraction:
  residual_boundary_b:
  zero_based_block_index:
  hook_path:
  output_hidden_states_index:
  final_norm_removed:
normalization:
  vector_norm_target:
  epsilon:
  injection_scale:
  mse_scale:
  mean_baseline_train_split_hash:
injection:
  mode:
  sentinel_token_id:
  sentinel_position_policy:
  adapter_checkpoint_hash:
  runtime_contract:
router:
  num_experts_per_tok:
  n_routed_experts:
  n_shared_experts:
  norm_topk_prob:
  routed_scaling_factor:
  topk_cardinality_checked:
ar:
  critic_prompt_template_hash:
  truncation_boundary_b:
  value_head_hash:
  ar_checkpoint_hash:
training:
  peft_method:
  hf_target_modules:
  megatron_target_modules:
  hf_to_megatron_name_map:
  logical_expert_mapping:
eval:
  metric_version:
  leakage_baselines:
  pca_baseline:
  nearest_neighbor_baseline:
  causal_patch_validation:
cache:
  disable_radix_cache:
  disable_prefix_cache:
  cache_key_includes_injection_hash:
  attention_kv_checked:
  mamba_conv_ssm_checked:
```

## Data Plan

Do not generate large data before smoke gates pass.

| Stage | Size | Purpose |
|---|---:|---|
| Boundary/adapter audit | about `1k` prompts | extraction, template, cache, boundary checks |
| Smoke | `5k-20k` activations | Track A/C signal and frozen AR lower bound |
| First SFT | `50k-100k` activations | AR/AV warm start if smoke passes |

Use the released NLA datagen contract as the default shape for Nano data:

- Nano Stage 0 replaces only the extraction backend: corpus text -> raw `R_b` vectors in a reference-compatible `base.parquet` plus sidecar.
- Reuse reference Stage 1 document-level splitting.
- Reuse reference Stage 2 teacher explanations for AV-SFT and AR-SFT only; RL rows remain unlabeled.
- Build Nano Stage 3 outputs with the same training parquet columns and sidecar token metadata as the reference repo.

For the Stage 1 dry run of the two-level data plan, switch AV/Track A marker handling to the reference CJK single-token marker mechanism before any AV/PEFT training:

- Use `nla.datagen.injection_tokens.build_token_meta` or an equivalent Nano wrapper to pick and record a rare CJK single-token injection marker. Prefer the upstream enclosed-CJK block when available, but allow verified rare CJK-symbol fallback tokens for Nano tokenizer compatibility.
- Record `injection_char`, `injection_token_id`, left/right neighbor IDs, and critic suffix IDs in the sidecar.
- Keep `<INJECT>` as the parquet placeholder and substitute the real CJK marker only at dataset-load or prompt-render time.
- Do not use common English tokens such as `a`, `an`, or `the` as injection markers; they are natural text tokens and can create false-positive hook targets.

Scale-up targets such as `250k+`, `500k+`, or `1M+` activations should be chosen only after held-out round trip passes. RL data sizing should wait until AV SFT passes through a frozen AR.

Teacher-distilled splits should be named as such:

- `AV teacher-distilled warm start`,
- `AR teacher-distilled warm start`,
- `held-out AV-generated eval`.

## Compute Plan

Treat compute as order-of-magnitude until the smoke run calibrates throughput.

Assumptions that must be recorded before giving tighter estimates:

- sequence lengths,
- batch sizes,
- selected boundary,
- Track A/B/C path,
- PEFT target modules,
- Bridge/EP config,
- rollout samples per activation,
- local HF vs SGLang/vLLM serving path.

Current rough range for an 8x H100 pilot:

| Phase | Rough range |
|---|---:|
| environment, adapters, extraction identity | `1-4` days |
| boundary and injection smoke | `1-5` days |
| frozen AR and first SFT | `2-8` days |
| evaluation and report | `1-3` days |

Do not commit to GPU-hour estimates until the `5k-20k` smoke run measures real throughput.

## Evaluation

### Must-Have For First Pilot

- cosine similarity,
- normalized MSE,
- `RRI_b`,
- centered raw-vector `R^2`,
- mean baseline,
- shuffled-target baseline,
- random matched-norm baseline,
- held-out AV-generated round trip,
- real-vs-shuffled injection gap,
- basic cache contamination test for injected requests.

### Deferred Until After First Successful SFT

- PCA/low-rank baseline,
- nearest-neighbor retrieval baseline,
- route-load entropy and effective expert count,
- broad token-position bucket analysis,
- explanation perturbation tests,
- larger causal patch validation,
- RL reward-vs-length analysis,
- live-AR vs frozen-AR reward divergence.

Qualitative review should stay small for the first pilot: about 50-100 held-out examples across ordinary prose, code, math/reasoning, multilingual text, and early/middle/late token positions.

## Final Recommendation

Core pilot:

```text
pin environment
-> prove Nano adapter and extraction identity
-> test R_34 and R_27
-> probe Track A and Track C
-> train frozen AR baseline
-> run held-out AV-generated round trip
-> run released Qwen/Gemma NLA checkpoint QC as implementation regression
```

If this produces a clear reconstruction gap over controls, then add PEFT, larger data, Track B, and RL in that order. If it does not, debug boundary/injection/AR before scaling.

The main plan should stay a decision document. Detailed mechanics, historical issue analysis, and longer math live in [issues_iter1.md](issues_iter1.md).

## References

- Natural Language Autoencoders paper/article: <https://transformer-circuits.pub/2026/nla/>
- Anthropic NLA overview: <https://www.anthropic.com/research/natural-language-autoencoders>
- NLA reference implementation: <https://github.com/kitft/natural_language_autoencoders>
- NLA inference docs: <https://raw.githubusercontent.com/kitft/natural_language_autoencoders/main/docs/inference.md>
- NLA critic model implementation: <https://raw.githubusercontent.com/kitft/natural_language_autoencoders/main/nla/models.py>
- NLA injection helper: <https://raw.githubusercontent.com/kitft/natural_language_autoencoders/main/nla/injection.py>
- Nemotron 3 Nano HF model: <https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16>
- Nemotron 3 Nano HF config: <https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/config.json>
- Nemotron 3 Nano HF configuration code: <https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/configuration_nemotron_h.py>
- Nemotron 3 Nano HF modeling code: <https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/modeling_nemotron_h.py>
- NVIDIA Megatron Bridge Nano docs: <https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html>
- NVIDIA Megatron Bridge PEFT docs: <https://docs.nvidia.com/nemo/megatron-bridge/latest/training/peft.html>
- Megatron Bridge repository: <https://github.com/NVIDIA-NeMo/Megatron-Bridge>
- SGLang server arguments: <https://docs.sglang.io/docs/advanced_features/server_arguments>
- PyTorch/SGLang hybrid-model cache discussion: <https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/>
- vLLM Nemotron-H model docs: <https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/nemotron_h/>
- vLLM Nemotron 3 Nano recipe: <https://docs.vllm.ai/projects/recipes/en/latest/NVIDIA/Nemotron-3-Nano-30B-A3B.html>
