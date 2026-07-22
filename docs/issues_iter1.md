# Nano30B NLA Issues - Iteration 1

This is an archive of the first detailed design/risk pass. Do not use it as the
active task tracker. Current phase tracking and run history live in
[execution_log.md](execution_log.md); the canonical plan lives in
[nano30b-nla-core-plan.md](nano30b-nla-core-plan.md); exact cluster commands
live in [cluster_runbook.md](cluster_runbook.md).

This file records the first concrete implementation issues for a Nano30B NLA pilot. The current conclusion is more nuanced than the first pass: keep paper-faithful input-embedding injection as the baseline, use residual-boundary injection as an oracle/debugging intervention, and keep AR prefix equivalence as an implementation diagnostic while preserving the real NLA objective `z -> h_hat_b`.

Target model assumptions:

- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Hidden size: `2688`.
- Blocks: `52` `NemotronHBlock` entries.
- Hybrid pattern: `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME`.
- Block types: Mamba-2 (`M`), MoE (`E`), GQA/attention (`*`).
- Loaded checkpoint config is authoritative for expert count and router settings. Current HF config/model card indicate `num_experts_per_tok=6`, while some Bridge prose has said `5`.
- Nano HF wrapper uses `.backbone`, block list `.backbone.layers`, final norm `.backbone.norm_f`, and embeddings under `.backbone.embeddings`.
- Nano uses a hybrid cache with attention KV plus Mamba conv/SSM states.
- Residual convention:
  - `R_0 = Embed(x)`.
  - `R_{i+1} = R_i + F_i(Norm_i(R_i))`.
  - A post-layer-33 extraction is named `h_34 = R_34`.

## Issue 1: AV Injection Should Keep The Paper Baseline And Add A Residual-Boundary Oracle

**Problem**

The released NLA paper/repo injects the activation by replacing a special token's input embedding. For Nano30B, if the extracted activation is `h_b = R_b(x)_t`, input-embedding injection asks the lower stack `G_0..G_{b-1}` to transform a synthetic embedding into a valid layer-`b` residual state.

That is an unnecessary nonlinear inverse problem:

```text
R_0 injected embedding -> lower Mamba/MoE/GQA stack -> R_b ~= h_b
```

This is especially questionable for Nano because early Mamba-2 layers use input-dependent state updates. A layer-`b` residual vector may be out-of-distribution as an input embedding even though it is perfectly in-distribution at boundary `R_b`.

However, input-embedding injection should not be skipped outright. The original NLA AV is not trying to be faithful activation patching; it is learning a communication channel:

```text
h_b -> pseudo-token input -> explanation text
```

Keeping this baseline matters because it is paper-faithful, keeps the released repo/serving path closest to intact, and gives us the cleanest comparison against Qwen/Gemma/Llama NLA checkpoints.

**Decision**

Use a staged injection decision tree, not one default implementation.

Track A: paper-faithful raw input-embedding injection.

```text
E(u)[p] <- alpha * h_b / ||h_b||
```

Track C: residual-boundary injection oracle.

```text
Run AV prompt normally to boundary R_b.
At the sentinel position p:
R_b[p] <- T(h_b)
Continue generation through G_b..G_{L-1}.
```

Track B: learned embedding adapter, implemented only if the probes justify it.

```text
E(u)[p] <- S_phi(h_b)
```

Train `S_phi` through downstream AV/AR reconstruction. Prefix alignment can be used as initialization or weak regularization, but it is not the main objective:

```text
Prefix_b^AV(E(u)[p] <- S_phi(h_b))_p ~= h_b
```

Decision logic:

```text
Run A and C probes first.

If A works:
    train with raw embedding injection and stay close to the paper/repo.

If A is weak and C works:
    add B, the learned injection adapter, to preserve input_embeds serving while improving reconstruction.

If A and B fail but C works:
    use Track C as a Nano-specific experimental fallback/oracle variant.
    serving remains out of scope unless A/B fail and C is strong.

If A and C both fail:
    do not blame injection alone; revisit layer choice, prompts, activation quality, AR, and target-model assumptions.
```

For Track C, default transport is:

```text
T(h) = h
```

Use diagonal affine transport only if the source target model and AV residual distributions differ:

```text
T(h) = mu_av + sigma_av * (h - mu_target) / (sigma_target + eps)
```

Replacement is the default. Do not use additive steering unless the activation being injected is explicitly a direction or delta.

**Implementation Requirements**

- Implement Track A first using the released NLA-style embedding replacement path.
- Implement Track C as an HF correctness oracle before attempting SGLang/vLLM residual patching.
- Defer Track B until Track A underperforms and Track C shows the activation is usable at the residual boundary.
- Define and document the exact residual boundary:
  - post-residual-add,
  - before the next block's norm,
  - exact module path in Nano's HF `modeling_nemotron_h.py`.
- Add a boundary-patching AV path:
  - prefill prompt to `R_b`,
  - replace sentinel-token residual at position `p`,
  - continue forward from boundary `b`,
  - preserve decode cache/state after patching.
- Store in sidecar:
  - `injection.mode = residual_boundary_replacement`,
  - `injection.boundary = R_b`,
  - `injection.layer_index = b - 1`,
  - `injection.position_selector = sentinel_token`,
  - `transport.type = identity | diagonal_affine | low_rank_affine`.
- For Track A, store:
  - `injection.mode = input_embedding`,
  - `injection.scale`,
  - injection token and neighbor IDs.
- For Track B, store:
  - `injection.mode = input_embedding_adapter`,
  - adapter type,
  - adapter checkpoint,
  - whether prefix alignment was used as initialization/regularization.

**Validation Gate**

- Track A and Track C must both be probed before choosing a training path.
- For Track A, correct activation embedding must change logits/generation differently from shuffled and random matched-norm activations.
- For Track C, correct residual replacement must change logits/generation differently from shuffled and random activations.
- For Track C, a norm sweep of `T(h_b)` gives a graded response, not a saturated/bimodal one.
- For Track C, replacing `R_b[p]` with the original prompt's own `R_b[p]` is approximately identity:
  - next-token KL near zero,
  - RMS ratio near 1,
  - cosine near 1.
- Track B is justified only if raw Track A is weak while Track C is strong.
- Track B must improve held-out round-trip reconstruction over Track A and over prefix-alignment-only training.

**Feasibility**

- Track A raw embedding injection: highest feasibility and most paper-faithful.
- Track C HF residual-boundary oracle: medium feasibility, high diagnostic value, nontrivial split-forward/cache risk.
- Track B learned embedding adapter: medium feasibility, conditional on probe results.
- SFT training path: medium feasibility.
- SGLang/vLLM production rollout path:
  - good for Track A,
  - likely good for Track B if it still uses `input_embeds`,
  - low-medium for Track C because residual-boundary patching is not a standard serving API.

## Issue 2: AR Must Use Exact Graph-Prefix Equivalence, But Still Reconstruct From Explanation Text

**Problem**

The AR cannot be treated as a loosely similar truncated model. For Nano, the AR prefix must compute the exact same residual boundary as the full model would compute, under the same weights, masks, positions, dtype, and Mamba state initialization.

However, the AR task must not be redefined as extracting `R_b` from the original source context. In NLA, AR receives only explanation text:

```text
source text x -> frozen target -> h_b
AV(h_b) -> explanation z
AR(z) -> h_hat_b
```

So the correct AR is:

```text
critic_prompt(z) -> Nano prefix to R_b -> value head V -> h_hat_b
```

not:

```text
source context x -> Nano prefix to R_b -> h_b
```

The latter is only an equivalence diagnostic.

**Decision**

Implement AR as an exact Nano graph-prefix stopped at the selected residual boundary:

```text
R_b_ar(z) = Prefix_b^theta(critic_prompt(z))_final_token
h_hat_b = V_psi(R_b_ar(z))
```

The prefix-equivalence test is:

```text
Prefix_b^theta(input)_t == FullNano^theta(input).hook(R_b)_t
```

This test should be run on ordinary token inputs and on critic prompts. It verifies implementation correctness, not AR quality. It is a **pre-training invariant** for the base AR prefix, not a permanent success criterion after AR tuning.

For Nano, the released NLA AR helper needs an architecture adapter before this test is meaningful:

```text
inner model: .backbone, not only .model or .transformer
block list: .backbone.layers
final norm: .backbone.norm_f
embedding key suffix: embeddings.weight
```

Final-norm stripping must include `norm_f`. For `b < 52`, `output_hidden_states[b]` should correspond to raw `R_b`; for `b = 52`, the final returned hidden state is post-`norm_f`, not raw `R_52`.

Run prefix-equivalence in four modes:

- raw completion prompts,
- chat prompts with `enable_thinking=false`,
- AV prompts with the injection marker but no injected vector,
- AR critic prompts ending at the prediction marker.

**Training Stages**

1. Frozen-prefix baseline:
   - Freeze `G_0..G_{b-1}`.
   - Train only `V_psi: R^2688 -> R^2688`.
   - This is clean and cheap, but may underfit.

2. Tethered-prefix AR if baseline reconstruction is weak:
   - Add LoRA to selected prefix modules.
   - Add tether loss:

```text
L_tether = E_x ||Prefix_b^{theta+delta}(x) - Prefix_b^theta(x)||^2
           / (||Prefix_b^theta(x)||^2 + eps)
```

3. Broader AR tuning only if needed:
   - Keep shuffled-target and mean baselines.
   - Keep a frozen AR snapshot for RL scoring audits.

After LoRA or broader prefix tuning, exact equality to the base full-model prefix is expected to break. That is not automatically a failure. At that point, judge AR by:

```text
matched reconstruction improvement
> predict-mean baseline
> shuffled-target baseline
> random matched-norm baseline
> PCA/nearest-neighbor/leakage baselines
```

and by whether the AR remains stable under reload and frozen-snapshot RL evaluation.

**Implementation Requirements**

- Do not slice the graph in a way that changes:
  - attention masks,
  - position ids,
  - Mamba conv/SSM initial states,
  - MoE routing behavior,
  - dtype/kernel path,
  - final normalization convention.
- Add equality diagnostics:
  - relative L2 error,
  - RMS ratio error,
  - `1 - cosine`,
  - downstream KL through the shared suffix when applicable.

**Validation Gate**

- Full-prefix equality holds within numerical tolerance before any AR prefix training.
- Frozen-prefix head-only AR beats predict-mean baseline or establishes a clear lower bound.
- If LoRA is enabled, base-prefix equality is no longer required; tether loss should remain bounded and AR must beat shuffled-target controls.

**Feasibility**

- Exact prefix diagnostics: high feasibility.
- Frozen-prefix AR: high feasibility, uncertain quality.
- Tethered LoRA prefix: medium feasibility, better quality odds.

## Issue 3: Boundary Selection, Stability, And MoE Routing Diagnostics

**Problem**

Nano's 52 blocks are not interchangeable. A post-Mamba residual, post-MoE residual, and post-GQA residual are different residual boundaries. Numeric depth alone is not enough to choose `h_b`.

The earlier plan also overstated MoE routing variance. In eval mode, with no router noise, no token dropping/capacity clipping, fixed position IDs, and deterministic kernels, a top-k MoE router should route deterministically as a function of the token hidden state. If the same `(x, tau)` routes differently only because the batch was reshuffled, that is a systems/numerics/packing/capacity diagnostic, not an inherent property of MoE.

The correct target is:

```text
h_b = R_b^target(x)_tau
```

where `b` is a residual boundary, not a vague "layer".

Mamba-specific caveat: a single token residual `R_b[tau]` is a valid activation target, but it is not the whole recurrent model state. During streaming, sequence information also lives in Mamba conv/SSM states. Post-Mamba boundaries should therefore remain candidates; attention boundaries are plausible first choices, not guaranteed semantic optima.

**Decision**

Select `h_b` by residual boundary, block type, semantic usefulness, AR cost, and AV suffix capacity.

Primary candidate boundaries:

| Boundary | Meaning | Why test it |
|---|---|---|
| `R_27` | post-GQA layer 26 | earlier, leaves more AV suffix depth, still plausibly semantic |
| `R_34` | post-GQA layer 33 | closest middle-late candidate and leaves later GQA at layer 42 |

Ablation candidates:

| Boundary | Meaning | Why test it |
|---|---|---|
| `R_33` | post-Mamba layer 32, just before GQA 33 | isolates whether the GQA write helps |
| `R_35` | post-MoE layer 34, just after GQA 33 | tests immediate MoE effect |
| `R_43` | post-GQA layer 42 | late semantic ablation; not first choice for residual-boundary AV because suffix capacity is smaller |

For `R_33` and `R_34`, compare the GQA write:

```text
A_33 = R_34 - R_33
```

Do not assume MoE boundaries are noisy. Measure route stability and residual movement. Reject a site only if extraction instability is large relative to natural activation variation or relative to the expected AR reconstruction signal.

**Implementation Requirements**

- Enumerate all 52 blocks and record:
  - zero-based block index,
  - block type (`Mamba-2`, `MoE`, `GQA`),
  - module path,
  - residual boundary name (`R_{i+1}` for post-layer `i`),
  - whether router internals are accessible.
- For each candidate boundary, measure activation scale:
  - L2 norm p05/p50/p95,
  - L2 norm mean/std,
  - RMS of `R_b`,
  - dtype and precision path.
- Measure repeated-run stability:

```text
epsilon_repeat = 1 - cos(h_b^(1), h_b^(2))
```

- Measure batch-reshuffle stability:

```text
epsilon_batch = 1 - cos(h_b^solo, h_b^batched)
```

- Compare extraction noise to natural activation variation:

```text
noise_b = E[1 - cos(h_b^(1), h_b^(2))]
data_b  = E[1 - cos(h_b(x,tau), h_b(x',tau'))]
rho_b   = noise_b / data_b
```

- For every MoE layer before `b`, log if accessible:
  - selected expert set `S_i(t)`,
  - router gates `g_i(t)`,
  - top-k margin `m_i = g_(k) - g_(k+1)`,
  - Jaccard similarity under repeated/reshuffled runs,
  - route-load entropy and effective expert count:

```text
H_i = -sum_e p_i,e log p_i,e
N_eff,i = exp(H_i)
```

- Record router configuration from the loaded checkpoint:

```yaml
router:
  num_experts_per_tok:
  n_routed_experts:
  n_shared_experts:
  norm_topk_prob:
  routed_scaling_factor:
  n_group:
  topk_group:
```

- Distinguish route-set instability, top-k weight instability, post-MoE residual instability, and downstream reconstruction instability.

- Bucket all results by token position:
  - `tau in [0,10)`,
  - `tau in [10,50)`,
  - `tau in [50,200)`,
  - `tau >= 200`.

Early-token failures should not condemn a site because early activations have seen little context.

- Measure causal usefulness of `R_b[tau]`:

```text
patch original h_b into matched context
patch original h_b into mismatched context
patch mean h_b
patch shuffled h_b
patch nearest-neighbor h_b
measure downstream next-token KL and target-token logit effects
```

**Validation Gate**

- For repeated extraction, median `1 - cos` should be near numerical noise and much smaller than the expected AR reconstruction error.
- `rho_b` should be much less than 1.
- Batch-reshuffle drift should be traced to deterministic-kernel, padding/position ID, packed-sequence, cache, or capacity settings before blaming MoE.
- MoE route flips are only disqualifying if they cause substantial residual movement, not merely because the expert set changes.
- The site must pass three practical constraints:
  - enough semantic signal for AR reconstruction,
  - enough downstream AV suffix capacity for the chosen injection mode,
  - feasible AR prefix cost.
- Sidecar records boundary, block type, candidate rationale, extraction noise, route diagnostics, and token-position buckets.

**Feasibility**

- Boundary enumeration and activation scale diagnostics: high feasibility.
- Repeated/batch stability diagnostics: high feasibility.
- Router margin/Jaccard/load logging: medium feasibility, dependent on Nano implementation internals.
- Candidate-boundary AR smoke comparison: medium feasibility, but highly useful.

## Issue 4: Serving And Generation Must Respect The Actual Residual Representation

**Problem**

Track A input-embedding injection can stay close to the released NLA serving contract:

```text
input_embeds with one row replaced -> generate
```

Track C residual-boundary injection is a different serving contract:

```text
prefill to boundary b -> materialize R_b -> patch R_b[p] -> continue suffix -> decode with valid cache/state
```

This patch must happen during prefill, before downstream attention KV caches and Mamba/SSM states are built. Running the whole prompt and patching afterward is invalid.

For Nano, the cache is hybrid. A correct Track C prefill must build both downstream attention KV cache and downstream Mamba conv/SSM states from the patched computation. Editing only an attention KV entry, patching after downstream blocks have run, or reusing token-ID-only prefix caches is invalid.

The hardest correctness issue is that high-performance hybrid implementations may carry the residual stream as a pair like:

```text
(hidden_states, residual)
```

rather than as a single materialized `R_b` tensor. Patching the wrong member of that pair is a silent correctness bug.

**Decision**

Use three serving tracks:

- Track A: paper-faithful input-embedding injection using the released-style `input_embeds` path.
- Track B: learned embedding adapter, still using `input_embeds` if justified by Issue 1 probes.
- Track C: residual-boundary oracle using an HF split-forward implementation first.

Do not assume SGLang/vLLM support arbitrary residual-boundary patching. Treat high-throughput residual patching as a later model-runner project.

For the pilot, Track C is an oracle/debugging intervention, not a serving-equivalent variant.

For any patched request, disable token-prefix/radix/Mamba prefix caching unless the cache key includes:

```text
injection_mode, boundary_b, patch_position, hash(patch_vector)
```

Before Track C, pass a Nano cache sanity gate:

```text
no-cache logits ~= cached logits
one-token cached decode ~= full-sequence recompute
returned cache is non-null
attention key/value caches populated for attention blocks
Mamba conv_states and ssm_states populated for Mamba blocks
cache dtype recorded
```

**Implementation Requirements**

- For Track A and Track B:
  - use the existing NLA-style embedding replacement path,
  - disable radix/prefix cache for injected requests,
  - record `input_embeds` injection metadata in the sidecar.
  - prove local/server first-token logits match when raw embeddings are supplied.
- For Track C, implement an explicit HF split-forward oracle:

```python
def forward_prefill_with_patch(
    model,
    input_ids,
    attention_mask,
    boundary_b,
    patch_pos,
    patch_vec,
):
    # 1. embed tokens
    # 2. run layers 0..b-1 with the model's native hidden/residual carriers
    # 3. materialize the tensor the next layer will normalize as R_b
    # 4. replace R_b[batch, patch_pos, :] = patch_vec
    # 5. convert R_b back into the representation expected by layer b
    # 6. run layers b..L-1 with cache/state construction
    # 7. return logits and patched caches
```

- For fused residual-add/norm implementations, materialize:

```text
R_post = residual + hidden_states
```

then patch:

```text
R_post[p] <- T(h_b)
```

and convert back to the representation expected by the next layer. A safe canonical representation to test is:

```text
residual <- R_post
hidden_states <- 0
```

provided the next layer computes its norm over `residual + hidden_states`. Verify this against the exact HF/NVIDIA/vLLM/SGLang code path used.

- Record whether a rollout used:
  - Track A input embedding,
  - Track B adapted embedding,
  - Track C HF residual patch,
  - custom SGLang residual patch,
  - custom vLLM residual patch.

**Validation Gate**

- Boundary materialization identity:

```text
normal logits ~= split-forward logits
hooked R_b ~= split-forward materialized R_b
```

- Self-replacement identity:

```text
v_self = R_b^AV(u)[p]
patch R_b[p] <- v_self
patched logits ~= normal logits
```

Track:

```text
max_abs_logit_diff
KL(P_normal || P_patched)
```

- Cache contamination test:
  - run two identical-token prompts with different injected vectors in alternating order `A, B, A, B`;
  - output for `A` must not depend on whether `B` ran immediately before it.
- Batched vs unbatched consistency for the same patched request.
- Downstream cached decode matches full recompute for 1, 4, and 16 generated tokens.
- Mamba fast-kernel vs fallback-kernel drift is measured on a small sample.
- Correct-vs-shuffled patch gap survives batching, but this is a signal test, not the main correctness test.
- Main downstream success remains the round trip:

```text
h_b -> z -> h_hat_b
cos(h_hat_b, h_b) >> cos(h_hat_b, h_b_shuffled)
```

**Feasibility**

- Track A serving: high feasibility and closest to the released repo.
- Track B serving: medium-high feasibility if it still emits `input_embeds`.
- Track C HF split-forward oracle: medium feasibility and high diagnostic value.
- Track C high-throughput SGLang/vLLM serving: low-medium feasibility; defer until the HF oracle proves the method.

## Issue 5: Training Stack Must Use Bridge/EP-First PEFT And Prove Checkpoint Portability

**Problem**

Nano has 30B total parameters, about 3.5B active parameters, and 128 routed experts. The NLA training stack needs a frozen target, AV, AR, rollout generation, reward scoring, and checkpoint reload. This is both a scientific and infrastructure problem.

Bridge support for Nano training is necessary but not sufficient for NLA. NLA still requires custom extraction hooks, `input_embeds` or split-forward AV injection, AR value-head training, normalized vector MSE, sidecar-controlled scales, and checkpoint export that preserves LoRA adapters plus the AR value head.

Several details matter:

- Expert parallelism is not optional for serious Nano training; use NVIDIA's supported stack first.
- Avoiding routed-expert LoRA is a safe starting point, not a mathematical rule.
- Freezing router weights does not freeze route assignments, because upstream hidden states can change under LoRA.
- Excluding expert FFN LoRA does not remove MoE communication; token dispatch/all-to-all still exists.
- Public docs have had expert-count discrepancies; trust the loaded checkpoint config and weight shapes.

**Decision**

Use Megatron Bridge's Nano recipe as the serious-training baseline, with EP configured according to the loaded Nano recipe. Start with PEFT, not full-parameter AV/AR training.

Model roles:

```text
Target Nano:
    theta_T frozen
    used only for h_b = R_b^theta_T(x)_tau

AV:
    theta_A = theta_0 + Delta_A^LoRA
    maps h_b -> z

AR:
    theta_R = theta_0 + Delta_R^LoRA
    value head V_psi
    maps z -> h_hat_b
```

The target model should never receive LoRA or training updates.

LoRA staging:

1. Conservative portable LoRA:
   - GQA/attention projection modules,
   - Mamba projection modules,
   - AR value head `V_psi`,
   - no router modules,
   - no routed expert FFNs.
2. Add always-active/shared expert components only if:
   - they are clearly identifiable in the loaded module tree,
   - Stage 1 plateaus,
   - export/reload remains clean.
3. Routed-expert LoRA only after:
   - EP training is stable,
   - HF export/reload works,
   - serving loads the adapters,
   - logical expert IDs are preserved across sharding/export.

Do not hardcode module names until inspecting the loaded Nano checkpoint. Candidate names include:

```text
HF attention: q_proj, k_proj, v_proj, o_proj
HF mamba: in_proj, out_proj
HF mlp/moe: up_proj, down_proj
HF router: gate
Bridge/Megatron: linear_qkv, linear_proj, linear_fc1, linear_fc2, in_proj, out_proj
```

but the actual target list must come from the loaded module tree and export path.

**Implementation Requirements**

- Pin and record:
  - Megatron Bridge commit/container,
  - HF model revision,
  - loaded `num_experts_per_tok`,
  - EP/TP/PP/SP settings,
  - PEFT target modules,
  - HF target modules,
  - Megatron target modules,
  - HF-to-Megatron name map,
  - excluded modules,
  - logical expert mapping if routed experts are ever adapted.
- Verify Bridge/HF conversion before training:

```text
logits_Bridge ~= logits_HF
R_b_Bridge ~= R_b_HF
```

- Verify adapter reload before long training:

```text
logits_trained ~= logits_reloaded
R_b_trained ~= R_b_reloaded
```

- Monitor route load before and after training:

```text
H_i = -sum_e p_i,e log p_i,e
N_eff,i = exp(H_i)
KL(p_i^base || p_i^trained)
```

Freezing router parameters is not enough; route assignments can still change if upstream hidden states change.

- Profile memory by layer type before deciding checkpointing policy. Do not assume "checkpoint MoE blocks only". Choose checkpointing by measured memory saved per recompute cost.
- Train in order:
  1. extraction validation,
  2. AR SFT,
  3. AV SFT,
  4. offline/sequential RL only after AR and AV SFT pass.
- Version the AR used for reward:

```yaml
reward_model:
  ar_checkpoint:
  ar_lora_hash:
  extraction_boundary:
  normalization:
```

**Validation Gate**

- Bridge/HF reload equivalence passes before any long run.
- Adapter reload equivalence passes before any long run.
- Smoke PEFT SFT stays within memory budget.
- Route-load entropy/effective expert count does not collapse unless route changes are intentional and improve validation.
- Checkpoint sidecar is complete enough to reproduce extraction, injection, AR scoring, LoRA loading, and cache policy.
- AR/AV SFT are trained and evaluated separately before concurrent RL is attempted.

**Feasibility**

- Bridge-based PEFT SFT on 8x H100: medium-high feasibility if kept close to NVIDIA recipes.
- Full-parameter AV/AR on 8x H100: low feasibility.
- Routed-expert LoRA: medium-low initially; defer.
- Sequential/offline RL after SFT: medium feasibility.
- Concurrent AV+AR+rollout RL: low-medium feasibility; only after sequential RL works.

## Cross-Cutting Audit Fixes

- Treat teacher-summary SFT as distillation warm start, not unsupervised NLA evidence.
- Default Nano chat/template control to `enable_thinking=false`; record tokenizer revision, remote-code revision, template, parser, and prompt format.
- Report relative reconstruction improvement rather than classical variance-explained framing when using direction-only normalized MSE.
- Add leakage controls:
  - token identity only,
  - token position only,
  - source-context summary only,
  - teacher summary with activation shuffled,
  - teacher summary with source context removed.
- Add PCA/low-rank and nearest-neighbor baselines.
- Add causal patch validation:

```text
patch original h_b
patch reconstructed h_hat_b
patch shuffled h_b
patch mean h_b
compare downstream logit/KL effects
```

- Add explanation perturbation tests to verify that AR uses the text bottleneck semantically.

## Iteration-1 Priority Order

1. Pin Nano revision, tokenizer/chat template, reasoning mode, Mamba kernel mode, and architecture adapter.
2. Implement residual-boundary definitions and full-vs-prefix equality diagnostics.
3. Run hook-site, cache-sanity, and MoE routing stability audits.
4. Probe Track A input-embedding injection and Track C HF residual-boundary oracle.
5. Train frozen-prefix AR head-only baseline with leakage/PCA/nearest-neighbor controls.
6. Decide whether Track B adapter, LoRA/tethered AR, or high-throughput serving are needed based on measured reconstruction improvement and injection signal.

## Current Feasibility Summary

The Nano adaptation is feasible as a pilot, but it should not be treated as a direct port of the released NLA repo. The viable route is:

```text
paper-faithful Track A baseline first
-> Track C residual-boundary oracle for diagnosis
-> Track B reconstruction-trained embedding adapter only if A is weak and C is strong
-> AR/AV SFT only after cache/template/adapter gates pass
-> serving/training throughput only after the science works locally
```

The main risk is not that Nano cannot support NLA; it is that a wrapper, final-norm, cache, reasoning-template, teacher-leakage, or adapter mismatch creates plausible-looking results with invalid reconstruction science.
