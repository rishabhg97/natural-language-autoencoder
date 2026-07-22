# Qwen NLA Inference-Only QC Agent Brief

This brief is for a new agent that should reproduce the released Qwen NLA
inference flow and save enough artifacts for comparison against the Nano30B NLA
pilot. This is an implementation-regression experiment, not Nano training.

## Scope

Run an inference-only quality-control pass for the released Qwen NLA pair:

- AV: `kitft/nla-qwen2.5-7b-L20-av`
- AR: `kitft/nla-qwen2.5-7b-L20-ar`
- Base model for extracting test vectors: `Qwen/Qwen2.5-7B-Instruct`
- Extraction layer: `20`

Do not run training, PEFT, LoRA, RL, serving work for Nano, large datagen, or
multi-GPU experiments. Use at most one visible GPU.

The purpose is to answer:

```text
Does the released Qwen NLA pair behave as expected under the reference
sidecar/injection/critic-scoring contract?
```

If Qwen passes cleanly, the Nano failure is more likely boundary/objective/data
specific. If Qwen fails, pause Nano work and debug our understanding of the
reference flow.

## Read First

From this project:

- `README.md`
- `docs/nano30b-nla-core-plan.md`
- `docs/cluster_runbook.md`
- `docs/execution_log.md`
- `external/natural_language_autoencoders/README.md`
- `external/natural_language_autoencoders/docs/inference.md`
- `external/natural_language_autoencoders/nla_inference.py`
- `external/natural_language_autoencoders/nla/schema.py`
- `external/natural_language_autoencoders/nla/injection.py`
- `external/natural_language_autoencoders/nla/reward.py`
- `external/natural_language_autoencoders/nla/datagen/extractors.py`

Important reference facts:

- `nla_inference.py` contains both `NLAClient` for AV generation through
  SGLang `input_embeds` and `NLACritic` for AR reconstruction/scoring.
- Actor sidecars define the injection marker, prompt template, neighbor token
  IDs, and `injection_scale`.
- Critic sidecars define the critic prompt template and `mse_scale`.
- Qwen does not need the Gemma `sqrt(d)` embedding-scale correction.
- For AR scoring, `MSE = 2 * (1 - cosine)` under the sidecar `mse_scale`.

## Suggested Cluster Setup

Use the existing local tmux session if working from the cluster:

```bash
tmux attach -t nano30b-nla
```

Use the project path that is already synced on the cluster:

```bash
cd /lustre/fsw/portfolios/llmservice/users/rigarg/mech_interp/research-projects-stage3/nano30b-nla-pilot
```

Use one GPU only. A single A100 80GB is enough for Qwen AV serving plus AR
scoring. CPU memory should not need to exceed `32G`.

If using Slurm, prefer a one-GPU interactive/batch job, for example:

```bash
# After creating the OUT directory below:
get_gpu | tee "$OUT/logs/get_gpu.txt"

# Choose one available one-GPU account/partition from get_gpu output, then
# write the exact srun command with concrete values into $OUT/commands.sh
# before executing it.
srun --gres=gpu:1 --mem=32G --time=04:00:00 --pty bash
```

If a dependency is missing, install only into a user-controlled conda env. Do
not use system Python and do not alter unrelated environments.

## Output Directory

Create one timestamped run directory:

```bash
TS=qwen-nla-inference-qc-$(date -u +%Y%m%dT%H%M%SZ)
OUT=runs/introspection/$TS
mkdir -p "$OUT"/{checkpoints,activations,logs,outputs,scores,scripts}
```

Every command that matters should be saved into:

```text
$OUT/commands.sh
```

Every process log should be tee'd into `logs/`.

## Required Artifacts

Save these files before handing back:

```text
runs/introspection/<TS>/
  manifest.json
  commands.sh
  logs/
    env.txt
    get_gpu.txt
    pip_freeze.txt
    nvidia_smi_start.txt
    nvidia_smi_end.txt
    sglang_server.log
    av_client.log
    ar_score.log
  checkpoints/
    av_model_info.json
    ar_model_info.json
    av_nla_meta.yaml
    ar_nla_meta.yaml
    av_config.json
    ar_config.json
    tokenizer_probe.json
  activations/
    qwen_l20_activations.parquet
    qwen_l20_activations_manifest.json
    extraction_probe.json
  outputs/
    av_outputs.jsonl
    av_outputs_sample.md
  scores/
    ar_scores.jsonl
    ar_summary.json
    controls_summary.json
  comparison_notes.md
```

The `manifest.json` must include:

- run timestamp
- host/node
- Slurm job id if any
- git commit for this repo
- git status summary
- Python executable
- relevant package versions
- `CUDA_VISIBLE_DEVICES`
- checkpoint ids and resolved HF revisions if available
- exact AV/AR sidecar hashes
- exact activation parquet row count and dimension

## Step 1: Checkpoint And Sidecar Capture

Download or access the released Qwen AV/AR checkpoints through Hugging Face.
Record whether they were already cached or freshly downloaded.

Use `huggingface_hub.model_info` or equivalent to save:

```text
checkpoints/av_model_info.json
checkpoints/ar_model_info.json
```

Copy the sidecars and configs from the local HF cache/snapshot if accessible:

```text
checkpoints/av_nla_meta.yaml
checkpoints/ar_nla_meta.yaml
checkpoints/av_config.json
checkpoints/ar_config.json
```

Also write `checkpoints/tokenizer_probe.json` with:

- AV injection char
- AV injection token id
- live tokenizer encoding of the injection char
- canonical prompt token count
- injection token position
- left/right neighbor IDs observed from the tokenizer
- whether observed values match the AV sidecar
- AR critic suffix token IDs from sidecar
- live tokenizer tail IDs for the critic template suffix

If any sidecar/tokenizer invariant fails, stop and report. Do not continue to
generate outputs from a mismatched checkpoint.

## Step 2: Build A Small Qwen Activation Parquet

Create `activations/qwen_l20_activations.parquet` with `20-100` rows. Keep the
first run small. The file must contain at least:

```text
activation_vector: fixed/list float32, width 3584
source_text: string
doc_id: string
n_raw_tokens: int
token_position: int
token_id: int
token_text: string
layer_index: int
extraction_method: string
```

Prefer the reference datagen convention over ad hoc `output_hidden_states`
indexing. The reference extractor says `layer_index=K` means a hook on
`model.model.layers[K]`, capturing that block's output. For Qwen L20, capture
the output of `model.model.layers[20]`.

If using `output_hidden_states`, explicitly verify which index matches the hook
before writing the parquet. Save that result in:

```text
activations/extraction_probe.json
```

The probe should include max absolute differences for:

```text
hooked layer-20 output vs selected output_hidden_states index
selected vector dimension
selected token position policy
```

Use simple public text prompts or a tiny FineWeb slice. Do not generate a large
dataset. Include several token positions from different documents if possible,
but avoid making this a datagen job.

## Step 3: Launch Qwen AV Through SGLang

Launch SGLang with the released Qwen AV checkpoint:

```bash
python -m sglang.launch_server \
  --model-path kitft/nla-qwen2.5-7b-L20-av \
  --port 30000 \
  --disable-radix-cache \
  --context-length 512 \
  --mem-fraction-static 0.70 \
  --trust-remote-code \
  2>&1 | tee "$OUT/logs/sglang_server.log"
```

Notes:

- `--disable-radix-cache` is required for `input_embeds`.
- Send only `input_embeds`, not `input_ids`.
- Use the AV sidecar prompt template exactly.
- Do not hardcode the injection marker or scale; read them from `nla_meta.yaml`.

Run AV generation over the activation parquet using
`external/natural_language_autoencoders/nla_inference.py` or a small run-local
wrapper that imports `NLAClient`. If a wrapper is needed, save it under:

```text
$OUT/scripts/qwen_inference_qc_runner.py
```

Save per-row AV records to:

```text
outputs/av_outputs.jsonl
```

Each JSONL row must include:

- row id
- source text preview
- token position and token text
- original activation norm
- raw AV response text
- parsed explanation text
- parse success/failure
- generation parameters
- request latency

Also save a human-readable sample:

```text
outputs/av_outputs_sample.md
```

## Step 4: Score With Released Qwen AR

Load the released Qwen AR checkpoint with `NLACritic` from
`external/natural_language_autoencoders/nla_inference.py`.

For each row, compute at least:

```text
correct:          AR(parsed AV explanation_i) vs activation_i
shuffled_text:    AR(parsed AV explanation_j) vs activation_i, j != i
random_text:      AR(short unrelated explanation) vs activation_i
mean_target:      train/eval mean-vector baseline if enough rows exist
```

If using a train/eval split for mean baseline, split by document when possible.
For the tiny run, it is acceptable to report both all-row mean and doc-heldout
mean if there are enough documents.

Save per-row scores:

```text
scores/ar_scores.jsonl
```

Each row must include:

- row id
- control type
- explanation row id used
- MSE
- cosine
- original activation norm
- reconstructed activation norm
- parse status

Save aggregate metrics:

```text
scores/ar_summary.json
scores/controls_summary.json
```

The aggregate summary should report:

- row count
- parse success rate
- mean/median MSE and cosine for each control
- correct-vs-shuffled MSE gap
- correct-vs-random MSE gap
- correct-vs-mean MSE gap if computed
- fraction of rows where correct beats shuffled
- fraction of rows where correct beats random
- examples of best, median, and worst correct reconstructions

## Expected Signal

Do not require a specific absolute threshold on the first run, because the
sample may include awkward tokens. But the released Qwen pair should show a
clear ordering:

```text
correct AV explanation better than shuffled explanation
correct AV explanation better than unrelated/random explanation
parse success high enough to inspect outputs, ideally > 90%
```

The reference docs describe cosine around `0.9` / MSE around `0.2` as a good
clean decode, cosine around `0.5` / MSE around `1.0` as mediocre, and
orthogonal as MSE around `2.0`.

If correct does not beat shuffled/random, treat that as a reference-flow
failure and debug before drawing Nano conclusions.

## Comparison Notes To Write

Create:

```text
comparison_notes.md
```

Answer these questions directly:

1. Did released Qwen sidecar loading pass all tokenizer/marker/template checks?
2. Did Qwen AV generation produce well-formed `<explanation>` outputs?
3. Did Qwen AR score correct explanations better than shuffled/random controls?
4. What were the typical Qwen correct MSE/cosine values?
5. How large was the correct-vs-shuffled gap?
6. Did the Qwen target residuals have a strong mean-vector baseline like Nano
   R34?
7. Which Nano failure hypotheses become more or less likely after this run?
8. Which parts of our Nano harness should be changed or left alone?

Use the Nano comparison points below:

```text
Nano R34 Haiku 1244-row AR check:
  heldout correct normalized MSE: 0.9222
  heldout correct cosine:         0.5389
  train-mean normalized MSE:      0.8829
  shuffled normalized MSE:        0.9648
  scientific_passed:              false
  eval split was random-row and all eval docs also appeared in train
```

The Qwen run should help distinguish:

```text
reference-flow misunderstanding
vs
Nano-specific boundary/objective/teacher bottleneck
```

## Stop Conditions

Stop and report instead of continuing if:

- AV or AR sidecar is missing.
- tokenizer encoding does not match sidecar injection token IDs.
- canonical prompt neighbor IDs do not match sidecar.
- SGLang cannot serve the AV checkpoint on one GPU.
- `nla_inference.py` cannot parse any generated explanations.
- AR checkpoint lacks `value_head.safetensors`.
- Qwen correct scores do not beat shuffled/random controls.

## Handoff Summary Required

When finished, report:

- run directory
- exact commands
- checkpoint revisions
- activation row count and extraction convention
- AV parse success rate
- mean/median correct MSE and cosine
- mean/median shuffled and random control MSE/cosine
- whether Qwen inference QC passed
- blockers and any environment changes
- the most important contrast with the Nano R34 AR failure
