# Scripts Map

This directory contains experiment scripts, verifiers, queues, and operational
helpers. Prefer extending existing families over adding one-off scripts.

## Dataset And Extraction

| Script | Role |
|---|---|
| `nano_prefix_activation_extract.py` | Prefix-key activation extraction for exact teacher-backed rows |
| `nano_prefix_dataset_pipeline.sh` | End-to-end prefix dataset pipeline for base, AR-SFT, AV-SFT, critic init, verifiers |
| `nano_prefix_dataset_queue.py` | Fail-closed sequential queue for config-driven dataset and critic preparation jobs |
| `nano_prefix_dataset_sidecar.py` | Shared sidecar metadata writer |
| `nano_realdata_stage0_extract.py` | Older real-data extraction path |
| `nano_realdata_ar_build.py` | AR dataset build helper |
| `verify_nano_miles_ar_dataset.py` | AR dataset verifier |
| `verify_nano_miles_av_dataset.py` | AV dataset verifier |

## Training And Queues

| Script | Role |
|---|---|
| `nano_ar_hpo_queue.py` | AR queue runner |
| `nano_av_probe_queue.py` | AV probe queue runner |
| `nano_rl_queue.py` | Config-driven RL queue runner with source/runtime contracts, managed SGLang, guard rules, checkpoint retention, and chained round-trip post-eval |
| `nano_queue_gate.py` | Evidence gate for promoting a blocked queue item after dependency status and artifact paths are present |
| `nano_queue_chain.py` | Wait for a named queue item and launch AR, AV, RL, round-trip, layer, or prefix-dataset queues |
| `nano_av_runner.py` | AV Miles/FSDP2 config-driven runner |
| `run_nano_av_miles_fsdp2_sft.sh` | AV SFT launch wrapper |
| `run_nano_av_100k_rslora_runai.sh` | Legacy AV rsLoRA launch wrapper |

## Evaluation

| Script | Role |
|---|---|
| `eval_nano_ar_miles_checkpoint.py` | AR checkpoint eval with controls |
| `eval_nano_av_miles_checkpoint.py` | AV checkpoint eval with controls |
| `eval_nano_av_ar_roundtrip_gate.py` | AV-generated-text to AR reconstruction gate |
| `analyze_rl_reward_gate_correlation.py` | RL reward/gate correlation diagnostics for completed generated outputs |
| `summarize_nano_av_run.py` | AV run summarizer |

## Operations

| Script | Role |
|---|---|
| `nano_s3.py` | S3 helper for RunAI/Mac artifact transfer |
| `nano_wandb.py` | W&B helper |
| `fetch_runai_wandb_offline.sh` | Fetch offline W&B snapshots |
| `prune_nano_miles_checkpoints.py` | Storage-conscious checkpoint pruning |
| `cluster_nano_env.sh` | Cluster environment setup |

## Organization Rule

When adding a script, also add:

- a short row in this README,
- a test when behavior is non-trivial,
- a runbook entry if operators need to execute it manually,
- a registry entry when it produces decision-changing results.
