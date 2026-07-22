# Qwen NLA Inference QC Report

Date: 2026-05-19
Run directory: `runs/introspection/qwen-nla-inference-qc-20260519T192605Z`
Cluster project path: `/lustre/fsw/portfolios/llmservice/users/rigarg/mech_interp/research-projects-stage3/nano30b-nla-pilot`
Final Slurm job: `28041646`
Final status: `QC passed: True`

## Executive Summary

This QC run reproduced the released Qwen NLA inference path using the Qwen AV/AR checkpoints from Hugging Face and a small layer-20 Qwen activation parquet. The run was inference-only: no training, PEFT, LoRA, RL, or large datagen was performed. The final pass used one A100 80GB GPU.

The released Qwen pair behaved as expected under the reference sidecar, injection, AV generation, and AR critic-scoring contract:

- Sidecar/tokenizer checks passed.
- Layer-20 activation extraction was verified against `output_hidden_states`.
- AV generation completed for all 56 activation rows.
- All 56 AV outputs parsed successfully as explanations.
- AR scoring separated correct explanations from shuffled, random, and mean-target controls on every row.
- Correct explanations scored far better than controls:
  - Correct mean MSE: `0.16280749892549856`
  - Shuffled mean MSE: `0.9283932048295226`
  - Random mean MSE: `1.0225361649479185`
  - Mean-target mean MSE: `0.6622565016150475`
  - Correct median cosine: `0.924675315618515`

The comparison point against the earlier Nano R34 failure remains stark: Nano R34 had heldout correct normalized MSE `0.9222`, heldout correct cosine `0.5389`, train-mean normalized MSE `0.8829`, shuffled normalized MSE `0.9648`, and `scientific_passed: false`. Since Qwen passed cleanly, the reference NLA flow is less likely to be the source of the Nano failure. Nano-specific boundary, objective, data mixture, split design, or target issues remain more likely.

## Original QC Objective

The handoff brief asked for an inference-only QC over the released Qwen NLA pair:

- AV checkpoint: `kitft/nla-qwen2.5-7b-L20-av`
- AR checkpoint: `kitft/nla-qwen2.5-7b-L20-ar`
- Base model for activation extraction: `Qwen/Qwen2.5-7B-Instruct`
- Extraction layer: `20`
- GPU limit: one visible GPU
- Explicit exclusions: no training, no PEFT/LoRA, no RL, no Nano serving, no large datagen, no multi-GPU experiment

The key question was:

```text
Does the released Qwen NLA pair behave as expected under the reference
sidecar/injection/critic-scoring contract?
```

The answer from this run is yes.

## Final Run Location

All artifacts were written under:

```text
/lustre/fsw/portfolios/llmservice/users/rigarg/mech_interp/research-projects-stage3/nano30b-nla-pilot/runs/introspection/qwen-nla-inference-qc-20260519T192605Z
```

Important generated files:

```text
manifest.json
commands.sh
comparison_notes.md
checkpoints/checkpoint_paths.json
checkpoints/tokenizer_probe.json
checkpoints/sidecar_hashes.json
activations/qwen_l20_activations.parquet
activations/qwen_l20_activations_manifest.json
activations/extraction_probe.json
outputs/av_outputs.jsonl
outputs/av_outputs_sample.md
scores/ar_scores.jsonl
scores/ar_summary.json
scores/controls_summary.json
logs/av_client_resume.log
logs/ar_score.log
logs/finalize.log
logs/nvidia_smi_end.txt
logs/output_row_counts.txt
logs/artifact_listing_final.txt
```

## Checkpoints And HF Resolution

The run resolved the three model repos through Hugging Face cache:

```json
{
  "av_repo": "kitft/nla-qwen2.5-7b-L20-av",
  "av_revision": "b88469162777ae6553bc14208eb0cb579336f8f4",
  "ar_repo": "kitft/nla-qwen2.5-7b-L20-ar",
  "ar_revision": "e2c9e57eac213d37a31612087f645ab6332c1bb6",
  "base_repo": "Qwen/Qwen2.5-7B-Instruct",
  "base_revision": "a09a35458c702b33eeacc393d103063234e8bc28"
}
```

Resolved snapshot paths:

```text
AV:
/lustre/fsw/portfolios/llmservice/users/rigarg/mech_interp/.cache/huggingface/hub/models--kitft--nla-qwen2.5-7b-L20-av/snapshots/b88469162777ae6553bc14208eb0cb579336f8f4

AR:
/lustre/fsw/portfolios/llmservice/users/rigarg/mech_interp/.cache/huggingface/hub/models--kitft--nla-qwen2.5-7b-L20-ar/snapshots/e2c9e57eac213d37a31612087f645ab6332c1bb6
```

`checkpoints/checkpoint_paths.json` recorded both AV and AR as not cached before the run:

```json
{
  "av_cached_before_run": false,
  "ar_cached_before_run": false
}
```

That means the required released AV/AR artifacts were pulled into the shared HF cache during this QC.

Sidecar/config SHA-256 hashes:

```json
{
  "checkpoints/ar_config.json": "cc1539659f9b87ded3af46887d65dd981d2be2e9303cc73ce9741a445ffd1a55",
  "checkpoints/ar_nla_meta.yaml": "553eb4ac1a585d598c6460032a9237f6951a68070afce42595dcc72b73c250ce",
  "checkpoints/av_config.json": "1a879a797e1473705ab3d7153bba1794f79c4cc87d948e927a3f1afa76f10786",
  "checkpoints/av_nla_meta.yaml": "2ff1aef3fcab48caf2e799733fbcc7d0ba0dd74a18e52d4b91b278d4abe2bddd"
}
```

## Sidecar And Tokenizer Contract Checks

The tokenizer probe passed the required sidecar invariants:

- AV injection encoding matched the sidecar: `true`
- AV neighbor IDs matched the sidecar: `true`
- AR critic suffix token IDs matched the sidecar: `true`

Recorded probe values:

```json
{
  "av_injection_char": "\\u320e",
  "av_injection_token_id": 149705,
  "live_tokenizer_encoding_of_injection_char": [149705],
  "canonical_prompt_token_count": 125,
  "injection_token_position": 111,
  "observed_left_neighbor_id": 29,
  "observed_right_neighbor_id": 522,
  "sidecar_left_neighbor_id": 29,
  "sidecar_right_neighbor_id": 522,
  "ar_critic_suffix_token_ids_from_sidecar": [1318, 29, 366, 1708, 29],
  "live_tokenizer_tail_ids_for_critic_template_suffix": [1318, 29, 366, 1708, 29]
}
```

This is a key result because it verifies that the released sidecars, prompt templates, injection marker, neighbor tokens, and critic suffix were being interpreted in the expected way before generation or scoring.

## Activation Extraction

The run created a small activation parquet:

```text
activations/qwen_l20_activations.parquet
```

Manifest:

```json
{
  "base_model": "Qwen/Qwen2.5-7B-Instruct",
  "created_utc": "2026-05-19T19:43:37Z",
  "dimension": 3584,
  "doc_count": 12,
  "extraction_convention": "forward hook on model.model.layers[20] capturing block output; output_hidden_states index 21 verified",
  "layer_index": 20,
  "path": "activations/qwen_l20_activations.parquet",
  "row_count": 56
}
```

Extraction probe:

```json
{
  "base_model": "Qwen/Qwen2.5-7B-Instruct",
  "hook_vs_output_hidden_states_max_abs_diff": 0.0,
  "hooked_module": "model.model.layers[20]",
  "layer_index": 20,
  "selected_output_hidden_states_index": 21,
  "selected_token_position_policy": "sample positions >= 8 plus fractional positions across multiple public text prompts; right padding and right truncation",
  "selected_vector_dimension": 3584
}
```

The `0.0` max absolute difference confirms the selected `output_hidden_states` index matched the forward hook convention for layer 20.

## SGLang AV Serving

The first SGLang launch loaded weights but failed during FlashInfer CUDA graph setup:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
```

`ninja` was installed in the NLA Python environment, but the env `bin` directory was not initially on `PATH` inside the allocation. The resumed run fixed this by exporting:

```bash
PATH=/lustre/fs11/portfolios/llmservice/projects/llmservice_nemo_mlops/users/rigarg/conda_env/nla/bin:$PATH
```

To avoid spending more time in the FlashInfer JIT/CUDA graph path for this small inference QC, the successful resumed SGLang launch used:

```bash
python -m sglang.launch_server \
  --model-path "$AV_SNAPSHOT" \
  --port 30000 \
  --disable-radix-cache \
  --context-length 512 \
  --mem-fraction-static 0.70 \
  --trust-remote-code \
  --disable-cuda-graph \
  --attention-backend triton \
  --sampling-backend pytorch
```

SGLang startup evidence:

```text
Load weight end. type=Qwen2ForCausalLM, dtype=torch.bfloat16
KV Cache is allocated.
Uvicorn running on http://127.0.0.1:30000
```

The server was shut down cleanly after AV:

```text
SIGTERM received. Draining requests and shutting down...
Gracefully exiting... Remaining number of requests 0.
```

## AV Generation Results

AV generation ran against the local SGLang server using the released AV sidecar and `input_embeds` path. The client log recorded:

```text
[NLAClient] b88469162777ae6553bc14208eb0cb579336f8f4: d_model=3584 inj_scale=150.0 embed_scale=1.00 inj_char='\\u320e'(id=149705)
```

All 56 rows completed and parsed:

```text
AV row 0: parse=True latency=2.79s
...
AV row 55: parse=True latency=2.55s
```

The final row count was:

```text
56 outputs/av_outputs.jsonl
```

The final summary reports:

```json
{
  "parse_success_rate": 1.0,
  "row_count": 56
}
```

## AR Scoring Results

AR scoring used the released Qwen AR checkpoint and sidecar critic.

The AR checkpoint loaded, but Hugging Face emitted this warning:

```text
Some weights of Qwen2ForCausalLM were not initialized from the model checkpoint ... and are newly initialized: ['lm_head.weight', 'model.norm.weight']
```

This warning is preserved in `logs/ar_score.log`. It should be noted when interpreting the checkpoint package, although the critic still separated correct explanations from controls strongly.

The critic initialization line was:

```text
[NLACritic] 21 layers  d_model=3584  mse_scale=59.87
```

All 56 rows parsed during AR scoring:

```text
AR row 0: parse=parsed
...
AR row 55: parse=parsed
```

The final score file contains four score types for each of 56 rows:

```text
56 outputs/av_outputs.jsonl
224 scores/ar_scores.jsonl
```

## Control Comparisons

The run scored four control conditions:

- `correct`: the AV-generated explanation for the matching activation row
- `shuffled_text`: explanation text shuffled across rows
- `random_text`: unrelated fixed random explanation text
- `mean_target`: mean-target baseline

Summary:

| Control | Rows | Mean MSE | Median MSE | Mean Cosine | Median Cosine |
|---|---:|---:|---:|---:|---:|
| correct | 56 | 0.16280749892549856 | 0.15064983814954758 | 0.9185963645577431 | 0.924675315618515 |
| mean_target | 56 | 0.6622565016150475 | 0.6348138153553009 | 0.6688720285892487 | 0.682593435049057 |
| shuffled_text | 56 | 0.9283932048295226 | 0.9018042683601379 | 0.5358037586723056 | 0.5490983128547668 |
| random_text | 56 | 1.0225361649479185 | 1.0334786772727966 | 0.4887322708964348 | 0.48326103389263153 |

Gaps:

```json
{
  "correct_vs_mean_mse_gap": 0.4994490026895489,
  "correct_vs_random_mse_gap": 0.8597286660224199,
  "correct_vs_shuffled_mse_gap": 0.7655857059040241,
  "fraction_correct_beats_mean": 1.0,
  "fraction_correct_beats_random": 1.0,
  "fraction_correct_beats_shuffled": 1.0
}
```

The critical result is that correct explanations beat every control on every row.

## Representative Examples

Best example:

```text
row_id: 11
token_text: " carrying"
mse: 0.08831863105297089
cosine: 0.9558408856391907
source preview: The short story opens with a traveler arriving at a quiet station, carrying a letter that changes how the family understands the previous winter.
```

Median example:

```text
row_id: 39
token_text: " crates"
mse: 0.15147648751735687
cosine: 0.9242620468139648
source preview: At dawn the harbor crews checked the tide tables, loaded the last crates, and waited for the fog to lift before clearing the channel.
```

Worst correct example:

```text
row_id: 31
token_text: ","
mse: 0.36912593245506287
cosine: 0.8154370784759521
source preview: The tutorial demonstrates how to normalize a vector, project it onto a basis, and interpret the residual as information not captured by the chosen subspace.
```

Even the worst correct example remained substantially better than the shuffled/random aggregate controls.

## Cluster And Job Notes

The final successful allocation was:

```text
JobID: 28041646
JobName: bash
Partition: interactive
State: COMPLETED
ExitCode: 0:0
Elapsed: 00:09:42
ReqMem: 32G
```

End-state GPU log showed no running GPU processes:

```text
No running processes found
```

There was unrelated tmux contamination from another agent:

- `28041506` (`nla-identity-r3427`) was submitted into the same broad session and failed quickly with exit code `1:0`.
- A later pending stray job `28041593` was canceled before consuming GPU resources.
- The Qwen run recorded the contamination under:
  - `logs/stray_identity_files.txt`
  - `logs/stray_identity_command_tail.txt`
  - `logs/stray_identity_sacct.txt`
  - `logs/post_interrupt_status.txt`

The first Qwen interactive allocation ended with exit code `130` while clearing the stale SGLang health wait. The run then resumed from the same output directory in a fresh tmux window and completed under job `28041646`.

## Interpretation Against Nano R34

The Qwen pair passed the intended QC, while the Nano R34 reference point was weak:

| Metric | Qwen QC | Nano R34 Failure Point |
|---|---:|---:|
| Correct mean MSE | 0.1628 | 0.9222 heldout normalized MSE |
| Correct median cosine | 0.9247 | 0.5389 heldout cosine |
| Mean-target / train-mean MSE | 0.6623 | 0.8829 |
| Shuffled MSE | 0.9284 | 0.9648 |
| Scientific pass | true | false |

Qwen correct explanations are clearly separated from controls. Nano R34 correct explanations were close to the mean baseline and only weakly separated from shuffled text. This makes a generic misunderstanding of the released NLA sidecar/injection/critic contract less likely.

The remaining likely Nano-specific issues are:

- boundary selection or residual target definition
- objective or target scaling mismatch
- data mixture or explanation quality mismatch
- train/eval split design
- mean-vector baseline dominance
- Nano model-specific representation behavior

The report in `comparison_notes.md` explicitly recommends keeping the sidecar/tokenizer/injection/critic contract unchanged if using the Qwen pass as validation, while revisiting Nano split design, target boundary, objective/data mixture, and mean-baseline controls before interpreting Nano R34 quality.

## Files To Inspect First

For a quick audit, inspect these in order:

```text
runs/introspection/qwen-nla-inference-qc-20260519T192605Z/comparison_notes.md
runs/introspection/qwen-nla-inference-qc-20260519T192605Z/scores/ar_summary.json
runs/introspection/qwen-nla-inference-qc-20260519T192605Z/scores/controls_summary.json
runs/introspection/qwen-nla-inference-qc-20260519T192605Z/checkpoints/tokenizer_probe.json
runs/introspection/qwen-nla-inference-qc-20260519T192605Z/activations/extraction_probe.json
runs/introspection/qwen-nla-inference-qc-20260519T192605Z/logs/av_client_resume.log
runs/introspection/qwen-nla-inference-qc-20260519T192605Z/logs/ar_score.log
runs/introspection/qwen-nla-inference-qc-20260519T192605Z/logs/output_row_counts.txt
```

## Bottom Line

The released Qwen NLA AV/AR pair passes the inference-only QC under the reference flow. The evidence supports treating the Qwen sidecar/injection/critic path as working in this environment. The Nano R34 failure should be investigated as a Nano-specific scientific or data/objective issue rather than as a first-order failure of the reference NLA inference contract.
