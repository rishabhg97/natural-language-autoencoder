# Nano30B NLA Pilot Handover - 2026-05-19

Historical status: superseded. The current starting points are
`docs/current_state.md` and `docs/runs/r33_rl_hero_20260708.md`.

This handover summarizes the `2026-05-19` Nano30B NLA pilot state after the
introspection, identity, source replay, capacity, exact-provenance, Haiku, and
Kimi diagnostics.

## Current Decision

Do not start 10% -> 30% -> 50% -> 75% -> 100% AR SFT scaling yet.

Reason: the regenerated exact-provenance data path is clean, and raw source-prefix replay is a near-perfect oracle, but teacher explanation prompts still fail heldout controls. Both Haiku and Kimi teacher prompts lose to blank/generic and train-mean controls. This is not currently a data-scale problem.

Next scientific step: run the same source replay and prompt-signal diagnostics at `R_27`, then audit AR math/template/boundary behavior before training or PEFT.

## Hard Scope Boundaries

Do not start any of the following until the prompt-signal gates pass heldout controls:

- No PEFT or LoRA.
- No serving work.
- No RL.
- No large dataset generation.
- No 10%+ scale run.
- Do not treat Track C as serving.
- Do not treat teacher summaries/explanations as ground truth.

Teacher text is only a warm-start label candidate. It must beat shuffled, blank, generic, source-context, and train-mean controls before scaling.

## Code And Branch State

Primary GitHub branch for this handover:

```text
origin/feature/nano30b-nla-diagnostics-20260519
df05de9 Record Nano exact signal gates and Kimi parser fix
```

Base branch it extends:

```text
origin/feature/nano30b-nla-harness
28b73a0 Add Nano AR prompt signal gate
```

Clean sync worktree on cluster:

```text
/lustre/fsw/portfolios/llmservice/users/rigarg/mech_interp/research-projects-stage3-sync-nla-20260519
```

Dirty experiment worktree with run artifacts:

```text
/lustre/fsw/portfolios/llmservice/users/rigarg/mech_interp/research-projects-stage3/nano30b-nla-pilot
```

Local macOS source path from original task:

```text
/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot
```

The local macOS checkout was intermittently unreadable to the tool because of macOS permission behavior. The cluster run artifacts and the pushed branch above are the reliable state.

## Files To Read First

Read these in order:

```text
nano30b-nla-pilot/README.md
nano30b-nla-pilot/docs/nano30b-nla-core-plan.md
nano30b-nla-pilot/docs/execution_log.md
nano30b-nla-pilot/docs/cluster_runbook.md
nano30b-nla-pilot/docs/qwen_nla_inference_qc_agent_brief.md
nano30b-nla-pilot/docs/handover_20260519_nano30b_nla.md
```

Reference implementation files:

```text
nano30b-nla-pilot/external/natural_language_autoencoders/nla/injection.py
nano30b-nla-pilot/external/natural_language_autoencoders/nla/models.py
nano30b-nla-pilot/external/natural_language_autoencoders/nla/loss.py
nano30b-nla-pilot/external/natural_language_autoencoders/nla/datagen/extractors.py
nano30b-nla-pilot/external/natural_language_autoencoders/nla_inference.py
nano30b-nla-pilot/external/natural_language_autoencoders/docs/inference.md
```

## Cluster Access

Use the existing local tmux session:

```bash
tmux attach -t nano30b-nla
```

Login-03 was reliable for polling:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 cs-oci-ord-login-03.nvidia.com
```

Use the project artifact worktree for experiments:

```bash
cd /lustre/fsw/portfolios/llmservice/users/rigarg/mech_interp/research-projects-stage3/nano30b-nla-pilot
source scripts/cluster_nano_env.sh
export PYTHONPATH="$PWD/external/natural_language_autoencoders:${PYTHONPATH:-}"
```

Use this conda/Python environment:

```text
/lustre/fs11/portfolios/llmservice/projects/llmservice_nemo_mlops/users/rigarg/conda_env/nla/bin/python
```

Use at most one GPU. Previous GPU jobs used:

```text
#SBATCH --account=nemotron_edge_omni
#SBATCH --partition=interactive_singlenode
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
```

CPU-only API jobs used:

```text
#SBATCH --partition=cpu_long
#SBATCH --account=llmservice_nemo_mlops
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
```

## Scientific Contract

The intended NLA contract remains:

```text
frozen target Nano: x, tau -> h_b = R_b^target(x)_tau
AV: h_b -> z
AR: z -> h_hat_b
loss/eval: h_hat_b reconstructs h_b through text
```

AR must reconstruct from explanation text `z`, not from source context `x`.

## Key Implemented Scripts

Core harness:

```text
scripts/nano_introspection.py
scripts/nano_extraction_identity.py
```

Real-data and diagnostics:

```text
scripts/nano_realdata_stage0_extract.py
scripts/nano_realdata_ar_build.py
scripts/nano_realdata_stage3_build.py
scripts/nano_source_replay_probe.py
scripts/nano_ar_capacity_probe.py
scripts/nano_ar_signal_gate.py
```

Provider/parser fixes in the current diagnostics branch:

```text
external/natural_language_autoencoders/nla/datagen/providers.py
external/natural_language_autoencoders/nla/datagen/stage2_api_explain.py
```

The Kimi/NVIDIA reasoning fix is important: NVIDIA-hosted Kimi can return useful output in `message.reasoning_content` while `message.content` is null. The provider now falls back to `reasoning_content`, and Stage 2 extracts the last complete `<analysis>...</analysis>` block from verbose reasoning traces.

## Confirmed Nano Wrapper Assumptions

The initial introspection confirmed the expected Nano wrapper/module paths:

```text
.backbone
.backbone.layers
.backbone.norm_f
.backbone.embeddings
```

`R_34` and `R_27` were the required boundaries to verify before training. Identity checks were done earlier for both; the later detailed diagnostic focus has been `R_34`.

## Major Positive Controls

### Qwen Released NLA Inference QC

Run directory:

```text
/lustre/fsw/portfolios/llmservice/users/rigarg/mech_interp/research-projects-stage3/nano30b-nla-pilot/runs/introspection/qwen-nla-inference-qc-20260519T192605Z
```

Results:

```text
QC passed: true
AV rows: 56, parse success 1.0
AR score rows: 224
Correct AR mean MSE: 0.1628
Shuffled mean MSE: 0.9284
Random mean MSE: 1.0225
Correct median cosine: 0.9247
Correct beats shuffled/random/mean target: 100% of rows
```

Interpretation: reference NLA released checkpoints behave as expected, so the local scoring/QC idea is meaningful.

### Nano R_34 Source Replay

Run directory:

```text
runs/introspection/source-replay-r34-haiku-20260519T201404Z
```

Results:

```text
256/256 exact token-count rows
correct cosine: 0.99909
normalized MSE: 0.00182
```

Interpretation: source-prefix replay can recover Nano `R_34` targets. The residual target and source provenance are not obviously broken.

## Failed R_34 Capacity Diagnostics

### Tail-1 Capacity Probe

Run directory:

```text
runs/introspection/ar-capacity-r34-tail1-haiku-singlenode-20260519T204033Z
job 28042647
```

Results:

```text
train NMSE: 1.3703 -> 0.7891
train cosine: 0.3149 -> 0.6055
heldout correct NMSE: 0.9674
heldout shuffled NMSE: 0.9747
heldout mean NMSE: 0.8919
scientific_passed: false
```

Interpretation: the model can fit train rows, but heldout teacher-conditioned reconstruction loses to mean.

### Capacity Ablation Bundle

Run directory:

```text
runs/introspection/ar-capacity-r34-diagnostics-haiku-20260519T210032Z
job 28043387
```

Summary:

```text
head0 doc_random 80: train 0.7992, heldout correct 0.9708, mean 0.8919, false
tail1 doc_random 400: train 0.2103, heldout correct 1.1277, mean 0.8919, false
tail1 row_random 80: train 0.7859, heldout correct 0.9484, mean 0.8904, false
tail2 doc_random 80: train 0.7621, heldout correct 0.9603, mean 0.8919, false
```

Interpretation: capacity is not the first-order blocker. More training overfits; more tail depth does not create heldout explanation signal.

## Prompt Signal Gates

### Old Haiku Gate, No Exact Prefix Metadata

Run directory:

```text
runs/introspection/ar-signal-gate-r34-haiku-20260519T213303Z
job 28044029
```

Key result:

```text
passed: true
scientific_passed: false
source_raw feature NMSE: 0.002116
source_raw cosine: 0.99894
teacher NMSE: 1.1437
mean NMSE: 0.8919
blank NMSE: 0.8916
generic NMSE: 0.8913
```

Caveat: source_raw had to re-tokenize text because the old parquet lacked `token_ids_prefix`.

### Exact-Provenance Haiku Gate

Run directory:

```text
runs/introspection/exact-r34-haiku-signal-20260519T215409Z
```

Jobs:

```text
stage0 extraction: 28044572, COMPLETED 0:0, 00:04:21, MaxRSS 1,402,452K
stage2/stage3: 28044573, COMPLETED 0:0, 00:02:19, MaxRSS 521,184K
signal gate: 28044574, COMPLETED 0:0, 00:07:47, MaxRSS 1,920,552K
```

Data:

```text
stage0 rows: 256/256 with token_ids_prefix
stage2/stage3 rows: 254/256 kept
final AR rows: 254/254 token_ids_prefix and api_explanation
exact_token_prefix_fraction: 1.0
warnings: []
blockers: []
```

Metrics:

```text
passed: true
scientific_passed: false
teacher heldout NMSE: 1.1089
teacher heldout cosine: 0.4456
teacher_shuffled NMSE: 1.2060
blank NMSE: 0.8599
generic NMSE: 0.8597
source_context NMSE: 1.0743
train-mean NMSE: 0.8595
source_raw feature NMSE: 0.003019
source_raw feature cosine: 0.99849
```

Interpretation: exact provenance did not fix the teacher signal failure.

### Exact-Provenance Kimi Reasoning Gate

Run directory:

```text
runs/introspection/kimi-exact-r34-32-signal-20260519T221433Z
```

Jobs:

```text
stage2/stage3: 28044907, COMPLETED 0:0, 00:05:44, MaxRSS 481,788K
signal gate: 28044908, COMPLETED 0:0, 00:01:46, MaxRSS 1,480,576K
```

Provider config:

```text
provider: nla.datagen.providers.OpenAIChatCompletionsProvider
model: nvidia/moonshotai/kimi-k2.6
max_tokens: 8192
temperature: 0.2
concurrency: 4
extra_body.chat_template_kwargs.thinking: true
```

Data:

```text
source rows: first 32 exact rows from the Haiku exact run
stage2/stage3 rows: 31/32 kept
final AR rows: 31/31 token_ids_prefix and api_explanation
exact_token_prefix_fraction: 1.0
warnings: []
blockers: []
```

Metrics:

```text
passed: true
scientific_passed: false
teacher heldout NMSE: 1.0729
teacher heldout cosine: 0.4636
teacher_shuffled NMSE: 1.1460
blank NMSE: 0.9578
generic NMSE: 0.9570
source_context NMSE: 1.0701
train-mean NMSE: 0.9595
source_raw feature NMSE: 0.001743
source_raw feature cosine: 0.99913
```

Interpretation: Kimi reasoning parses and runs now, but stronger teacher text still does not beat blank/generic/mean controls.

## Current Diagnosis

Supported by the artifacts above:

1. Nano extraction and source-token provenance are not the immediate blocker.
   - Exact rows have `token_ids_prefix`.
   - Source_raw replay is near-perfect.

2. AR train capacity is not the immediate blocker.
   - The probe can fit train rows.
   - More steps/depth overfit or fail heldout controls.

3. Current teacher explanations do not carry enough heldout reconstruction signal in the current critic channel.
   - Haiku and Kimi both beat shuffled teacher, so they contain some row-specific information.
   - Both lose badly to blank/generic/mean controls, so that information is not mapping to target residuals under the current AR feature/head/template setup.

4. Source_context inside the critic template fails, while raw source prefix passes.
   - This suggests prompt/template/boundary/feature geometry issues, not just bad teacher labels.

## Recommended Next Work

### Step 1: Repeat Source Replay And Signal Gate At R_27

Goal: determine whether the failure is specific to `R_34`.

Use the same FineWeb slice and exact-token data pattern as the R_34 exact run. If using the same text rows, Stage 2 explanations can be cached from R_34 because the teacher prompt depends on `detokenized_text_truncated`, not the boundary.

Suggested GPU Stage 0 extraction command inside an sbatch with one A100:

```bash
"$NANO_PYTHON" scripts/nano_realdata_stage0_extract.py \
  --local-files-only \
  --boundary R_27 \
  --corpus HuggingFaceFW/fineweb \
  --corpus-config sample-10BT \
  --corpus-split train \
  --text-column text \
  --corpus-start 0 \
  --corpus-length 256 \
  --positions-per-doc 1 \
  --chunk-size 4 \
  --batch-size 1 \
  --max-length 1024 \
  --keep-token-metadata \
  --output "$OUT/base_R27.parquet" \
  --metadata-output "$OUT/base_R27.metadata.json"
```

Then split AR-only:

```bash
"$NANO_PYTHON" -m nla.datagen.stage1_split \
  --base "$OUT/base_R27.parquet" \
  --av-sft-frac 0.0 \
  --ar-sft-frac 1.0 \
  --rl-frac 0.0 \
  --seed 42 \
  --output-dir "$OUT/splits"
```

For Stage 2, first try cache reuse from the exact Haiku R_34 run:

```bash
"$NANO_PYTHON" -m nla.datagen.stage2_api_explain \
  --input "$OUT/splits/ar_sft_raw.parquet" \
  --output "$OUT/splits/ar_sft_explained.parquet" \
  --cache-from runs/introspection/exact-r34-haiku-signal-20260519T215409Z/splits/ar_sft_explained.parquet \
  --provider-cls nla.datagen.providers.OpenAIChatCompletionsProvider \
  --provider-kwargs '{"model":"aws/anthropic/claude-haiku-4-5-v1","max_tokens":600,"temperature":1.0,"concurrency":16,"max_retries":10}' \
  --chunk-size 32
```

Build AR parquet:

```bash
"$NANO_PYTHON" scripts/nano_realdata_ar_build.py \
  --local-files-only \
  --input "$OUT/splits/ar_sft_explained.parquet" \
  --output "$OUT/ar_sft.parquet"
```

Run source replay first:

```bash
"$NANO_PYTHON" scripts/nano_source_replay_probe.py \
  --local-files-only \
  --ar-sft-parquet "$OUT/ar_sft.parquet" \
  --boundaries R_27 \
  --source-token-ids-column token_ids_prefix \
  --source-max-length 2048 \
  --source-feature-batch-size 4 \
  --max-records 256 \
  --timestamp "$TS-source-replay"
```

Then run the signal gate:

```bash
"$NANO_PYTHON" scripts/nano_ar_signal_gate.py \
  --local-files-only \
  --ar-sft-parquet "$OUT/ar_sft.parquet" \
  --boundaries R_27 \
  --max-records 256 \
  --train-fraction 0.75 \
  --split-strategy doc_random \
  --variants teacher,teacher_shuffled,blank,generic,source_context,source_raw \
  --prompt-max-length 2048 \
  --feature-batch-size 2 \
  --max-steps 80 \
  --lr 1e-3 \
  --mse-margin 0.05 \
  --cosine-margin 0.02 \
  --oracle-mse-threshold 0.05 \
  --log-every 20 \
  --timestamp "$TS"
```

Pass condition before any scale-up:

```text
source_raw_oracle_passed = true
exact_token_prefix_fraction = 1.0
teacher heldout NMSE < shuffled, blank, generic, and mean controls by the configured margin
teacher heldout cosine > controls by the configured margin
scientific_passed = true
```

### Step 2: Audit AR Geometry Against Reference NLA

Focus on these questions before adding training scale:

- Does the Nano target vector use the same residual boundary semantics as the reference AR critic expects?
- Are we comparing normalized vectors exactly like reference `nla/loss.py` expects, including scaling to `sqrt(d)`?
- Is the current trainable value head a faithful stand-in for the reference frozen AR path, or does it hide a missing projection/injection step?
- Does the current critic template `Summary of the following text: <text>{explanation}</text> <summary>` erase or wash out the explanation signal for Nano?
- Is source_context failing because the critic template is wrong, while source_raw passes because it uses the model's native continuation path?
- Does Nano's hybrid Mamba/MoE block structure change which residual boundary is most appropriate for AR reconstruction?

### Step 3: Track A And Track C On Exact-Provenance Rows

After R_27 source replay/signal gate, run Track A/C probes on the same rows. Goal: decide whether the AV text channel has usable signal before training.

Keep Track A paper-faithful and marker-based. Do not use common English tokens such as `a`, `an`, or `the`; the plan requires CJK/single-token marker handling based on the reference NLA marker contract.

## Useful Status Commands

Check no jobs are still running:

```bash
squeue -u "$USER" -o '%.18i %.24j %.10T %.10M %.20P %.30R' | grep -E 'nla|nano|JOBID' || true
```

Summarize completed diagnostic jobs:

```bash
sacct -j 28044572,28044573,28044574,28044907,28044908 \
  --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem --parsable2
```

Parse Haiku and Kimi signal gate summaries:

```bash
python3 - <<'PY'
import json
from pathlib import Path
runs = [
    'runs/introspection/exact-r34-haiku-signal-20260519T215409Z',
    'runs/introspection/kimi-exact-r34-32-signal-20260519T221433Z',
]
for run in runs:
    d = json.loads((Path(run) / 'ar_signal_gate.json').read_text())
    c = d['comparison']
    p = d['data_source']['provenance']
    print(run)
    print('rows', p['row_count'], 'exact_frac', p['exact_token_prefix_fraction'], 'api', p['api_explanation_count'])
    print('passed', d['passed'], 'scientific', d['scientific_passed'], 'warnings', d.get('warnings', []), 'blockers', d.get('blockers', []))
    print('teacher_nmse', c['teacher_heldout_normalized_mse'])
    print('teacher_cos', c['teacher_heldout_cosine'])
    print('mean_nmse', c['mean_heldout_normalized_mse'])
    print('source_raw_nmse', c['source_raw_feature_normalized_mse'])
    print('controls', {k: v['control_normalized_mse'] for k, v in c['control_comparisons'].items()})
PY
```

Validate parser files in the clean sync worktree:

```bash
cd /lustre/fsw/portfolios/llmservice/users/rigarg/mech_interp/research-projects-stage3-sync-nla-20260519
python3 -m py_compile \
  nano30b-nla-pilot/external/natural_language_autoencoders/nla/datagen/providers.py \
  nano30b-nla-pilot/external/natural_language_autoencoders/nla/datagen/stage2_api_explain.py
```

## Known Blockers And Ambiguities

Current blockers:

```text
R_34 teacher prompts fail heldout blank/generic/mean controls.
R_27 real-data source replay and prompt signal gate are not yet run.
AR math/template/boundary equivalence to reference NLA is not fully resolved for Nano's hybrid Mamba/MoE architecture.
Track A/C exact-provenance probes are incomplete.
```

Resolved or downgraded blockers:

```text
remote-code/model load: works in current cluster env
mamba_ssm/selective_scan_cuda import: fixed by environment/library setup
torch missing: was pip path confusion; use python -m pip and NANO_PYTHON
exact token provenance: fixed for regenerated rows, exact fraction 1.0
Kimi parsing: fixed for reasoning_content fallback and last analysis block extraction
memory: 32G CPU + one A100 80GB was enough for all diagnostics above
```

Still watch for:

```text
final-norm mismatch
output_hidden_states mismatch
critic template ambiguity
cache API ambiguity
teacher response truncation or bad XML tag formatting
row drops in Stage 2
```

## What To Report Back After The Next Step

For each next diagnostic, report:

```text
script paths
exact commands or sbatch paths
run directory
job IDs and slurm states
row counts and drop counts
exact_token_prefix_fraction
source_raw oracle metrics
teacher vs shuffled/blank/generic/source_context/mean metrics
scientific_passed boolean
whether scaling is still blocked
new blocker classification
```

If identity/source_raw fails at `R_27`, stop and debug boundary extraction before touching AV/AR training.
