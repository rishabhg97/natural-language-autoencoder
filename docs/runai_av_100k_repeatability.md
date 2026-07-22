# RunAI AV 100k Repeatability

This runbook makes the current Nano30B AV rsLoRA experiment reproducible on
RunAI `train-dev` and keeps post-run analysis anchored to the JSON report.

## Inputs

```text
code: /workspace/interp/code/nano30b-nla-pilot-nano-av-h200-20260523T031253Z
model: /workspace/interp/models/nano-30b-a3b-bf16-hf
data: /workspace/interp/artifacts/nano30b-nla-pilot/super-teacher-r27-100k-thinking-merged-20260525T2150Z/base_R27_super_thinking_99570_explained.parquet
rows: 99,570
split: doc, 90/10
```

## Canonical Command

Inside the `train-dev` workspace:

```bash
cd /workspace/interp/code/nano30b-nla-pilot-nano-av-h200-20260523T031253Z
source /workspace/interp/.venv/bin/activate

OUTPUT_ROOT=/workspace/interp/outputs/nano30b-nla-pilot/av-r27-100k-rslora-repro \
WANDB_GROUP=av-r27-100k-rslora-repro \
TRAIN_LR=1e-5 \
scripts/run_nano_av_100k_rslora_runai.sh
```

The launcher defaults to:

```text
rsLoRA rank 192, alpha 384
row limit: 99,570
experiment class: complete-performance
split: doc-level 90/5/5 train/validation/test
targets: q/k/v/o/in/out/up/down
trainable params: about 5.3B, 14.4%
injection scale: 75
steps: 800
batch size: 1
eval train limit: 64
eval validation limit: 128
eval test limit: 128
generate examples: 8
offline W&B: enabled
save_trainable_state: enabled
```

For a true full train-split pass over the 100k dataset, use epoch mode instead
of a fixed smoke-step count:

```bash
OUTPUT_ROOT=/workspace/interp/outputs/nano30b-nla-pilot/av-r27-100k-rslora-b8-epoch1-v1 \
WANDB_GROUP=av-r27-100k-rslora-b8-epoch1-v1 \
TRAIN_LR=1e-5 \
TRAIN_BATCH_SIZE=8 \
EXPERIMENT_CLASS=complete-performance \
TRAIN_FRACTION=0.9 \
VALIDATION_FRACTION=0.05 \
TEST_FRACTION=0.05 \
TRAIN_EPOCHS=1 \
TRAIN_SAMPLING=epoch \
TRAIN_LOG_EVERY=125 \
scripts/run_nano_av_100k_rslora_runai.sh
```

With the current 99,570-row dataset and 90/5/5 doc split, expected training
coverage is about 89.6k train rows. That is about 22.4k optimizer steps at
batch 4, or 11.2k optimizer steps at batch 8.

Override only the variable being tested, for example:

```bash
OUTPUT_ROOT=/workspace/interp/outputs/nano30b-nla-pilot/av-r27-100k-rslora-hpo-v2 \
WANDB_GROUP=av-r27-100k-rslora-hpo-v2 \
TRAIN_LR=2e-5 \
scripts/run_nano_av_100k_rslora_runai.sh
```

## Monitoring

```bash
RUN=av-r27-99570-rslora-r192-broad-scale75-lr1e5-s800-save-gen8-2gpu-offline-YYYYMMDDTHHMMZ
OUT=/workspace/interp/outputs/nano30b-nla-pilot/av-r27-100k-rslora-repro/$RUN
LOG=/workspace/interp/tmp/nano_av_peft_logs/$RUN.log
PID=$(cat /workspace/interp/tmp/nano_av_peft_logs/$RUN.pid)

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
ps -p "$PID" -o pid,stat,etime --no-headers
test -f "$OUT/av_warmstart_smoke.json" && echo report_exists || echo report_missing
test -f "$OUT/trainable_state.pt" && ls -lh "$OUT/trainable_state.pt"
tail -80 "$LOG" | grep -E "step|loss|Saved|report|wandb|blocker|heldout" | tail
```

## Summarize

When `av_warmstart_smoke.json` exists:

```bash
python scripts/summarize_nano_av_run.py "$OUT"
python scripts/summarize_nano_av_run.py "$OUT" --json > "$OUT/summary.json"
```

Standard gates to record:

```text
blockers
train/validation/test rows and doc overlap
train curve first/min/last loss and grad norm tail
validation/test NLL for real, shuffled, zero, mean, no-injection
validation/test real-vs-control NLL gaps
generation parse rate and content F1 by control
trainable parameter count/fraction
trainable_state path/size
offline W&B directory
```

## Fetch And Sync W&B

From the local Mac:

```bash
SYNC=1 scripts/fetch_runai_wandb_offline.sh "$OUT"
```

The `.env` file may contain `wandb_api_key`; the script maps it to
`WANDB_API_KEY` without printing it.

## Reference Result

Completed run:

```text
run: av-r27-99570-rslora-r192-broad-scale75-lr1e5-s800-save-gen8-2gpu-offline-20260526T1645Z
blockers: []
split: 89,604 train / 9,966 heldout / 0 doc overlap
trainable: 5,303,242,752 params, 14.38%
trainable_state: 20G
wandb: https://wandb.ai/rishabhga97/nano30b-nla-pilot/runs/ggrd168y
```

Heldout NLL:

| Control | NLL |
|---|---:|
| real | 1.1707 |
| zero | 1.4172 |
| mean | 1.4671 |
| no injection | 1.4797 |
| shuffled | 1.5454 |

Real-vs-control gaps:

```text
vs zero:     +0.2465
vs mean:     +0.2964
vs none:     +0.3090
vs shuffled: +0.3747
```

Generation examples:

```text
real closing-tag rate: 7/8
mean content F1: real 0.4466, shuffled 0.3288, zero 0.2842, no-injection 0.2770
no-injection closing-tag rate: 0/8
```

Interpretation: the AV likelihood signal is strong and row-specific, and free
generation is meaningfully better than the earlier repeated-`<` collapse. It is
still a separate gate: parse/closure rate and content F1 must be tracked rather
than inferred from NLL alone.
