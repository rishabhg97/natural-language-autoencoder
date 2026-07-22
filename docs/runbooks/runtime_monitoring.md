# Runtime Monitoring

Use runtime monitoring to capture enough system context to explain speed,
memory, OOMs, and evictions without keeping bulky artifacts.

## Current Monitoring Outputs

Remote monitoring directory:

`/workspace/interp/outputs/nano30b-nla-pilot/monitoring`

Known CSV patterns:

- `gpu_memory_*.csv`
- `gpu_apps_*.csv`
- `process_memory_*.csv`

These CSVs are useful but can become stale if the monitor is not running. Always
pair them with a live process and `nvidia-smi` check before making a claim.

## What To Record Per Run

- Run id and config path.
- GPU type and count.
- Global batch, microbatch, sequence length, packed-token cap.
- Current and peak GPU memory per GPU.
- Current and peak GPU utilization if available.
- Active training/eval process IDs.
- Disk use for `/workspace/interp`, `/workspace/models`, `/tmp`, `/dev/shm`.
- OOM, traceback, or eviction signs.
- Whether W&B system metrics were logged offline.

## W&B Integration Direction

The preferred long-term path is to log compact system metrics into the offline
W&B run itself, then keep CSV monitors as fallback/debug artifacts. This avoids a
split between training metrics and resource metrics.

Do not log secrets or full environment dumps.

## Quick Health Check

Run from the active RunAI workspace:

```bash
pwd
df -h /workspace/interp /workspace/models /tmp /dev/shm
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader
ps -eo pid,etime,cmd | grep -E "train_actor|nano_av|nano_ar|eval_nano|monitor" | grep -v grep || true
```

