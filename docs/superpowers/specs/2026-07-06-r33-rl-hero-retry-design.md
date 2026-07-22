# R33 RL Hero Retry Design

## Objective

Run a storage-safe, fully guarded 342-update R33 corrected-K3 RL hero retry
that preserves the clean `lr=1e-5` HPO winner, tolerates isolated heavy-tail
K3 samples, and still stops on sustained policy divergence. Promotion remains
conditional on matched round-trip evaluation against the clean R33 SFT
baseline.

## Evidence And Diagnosis

The first hero attempt generated rollout IDs 0-63, totaling 24,576 responses.
Its raw reward improved from a first-ten mean of `-0.346596` to a last-ten
mean of `-0.272935`; close and usable rates averaged `99.784%/99.752%`, no
response was truncated, and actor/rollout log-prob absolute difference stayed
below `0.304` versus the `0.75` guard.

The run stopped because raw `train/kl_loss` exceeded `5.0` on two consecutive
steps: `25.9797955` at step 62 and `5.1258636` at step 63. The KL distribution
was heavy-tailed rather than monotonically increasing: median `0.6301`, with
isolated maxima `233.0444`, `15.4012`, `178.4859`, and `25.9798`. Eight-step
window medians remained between `0.28` and `0.91`. The successful clean
32-update confirmation showed the same pattern, including isolated KL values
`24.4045`, `33.8230`, and `8.5366`, while producing the selected update-16
round-trip improvement.

The working diagnosis is therefore guard calibration against a heavy-tailed
per-update K3 statistic, not CUDA failure, rollout/parser failure, actor/ref
desynchronization, or demonstrated sustained policy divergence.

## Storage Cleanup

Preserve these required inputs and evidence:

- R33 AV-SFT actor DCP `iter_0001291`.
- R33 actor-reference HF export `actor_sft_hf_iter_0001291`.
- R33 AR-SFT critic HF checkpoint `iter_0001289/hf`.
- Independent R33 critic final checkpoint and eval report.
- Corrected-K3 selected update-16 checkpoint and Stage-2 evidence.
- RL train-only parquet, SFT baseline reports, cross-critic gate, and Stage-2
  gate.
- Text logs, offline W&B runs, generated JSONL, and eval reports from all
  historical experiments.

Delete only model shard directories from these superseded checkpoints:

- Corrected-K3 unified-environment update 8.
- Historical Qwen-scale `lr=2e-5` confirmation update 16.
- Historical Qwen-scale `lr=2e-5` medium update 32.

The deletion must use a retention policy and write a manifest before applying
the cleanup. It should reclaim approximately 177 GiB from Longhorn while
leaving the lightweight experimental record intact. The already removed
obsolete crash core dumps and completed iteration-393 continuation checkpoint
must also be recorded in the logbook.

## Retry Configuration

Create a new queue YAML and run directory; do not mutate or reuse the failed
run directory. Preserve:

- 8 H100 NVL topology: six actor, one rollout/SGLang, one frozen critic.
- Clean R33 RL train-only dataset.
- `rollout_batch_size=48`, `n_samples_per_prompt=8`, and
  `global_batch_size=384`.
- `num_rollout=342`, producing 131,328 sampled responses.
- Actor LR `1e-5`, constant schedule, K3 KL loss, KL coefficient `1e-3`.
- Actor microbatch 32, GRPO standard-deviation normalization, unnormalized
  advantages, frozen critic, offline W&B role runs, and external SGLang.
- Existing parser, response-length, provenance, runtime-contract, and
  Stage-2 prerequisite gates.

Change only the safety and checkpoint envelope:

- Stop on `train/kl_loss > 5.0` for three consecutive actor updates.
- Keep the actor/rollout log-prob absolute-difference guard at `0.75` for two
  consecutive updates.
- Add `train/grad_norm > 100` for two consecutive actor updates.
- Save actor checkpoints at updates 114, 228, and 342.
- Retain checkpoints 114 and 342 after successful training; delete 228 through
  the declared retention policy.

The three-step KL rule would not have fired on either clean historical trace.
It still fails closed on sustained moderate divergence, while the independent
log-prob and gradient guards cover synchronization failure and repeated large
updates.

## Evaluation And Promotion

After successful training:

1. Evaluate update 114 on bounded 64/64 round-trip rows against the clean SFT
   baseline.
2. Evaluate update 342 on the same 64/64 rows against update 114.
3. Run the update-342 512/512 promotion evaluation against the clean SFT
   baseline only if the preceding gate passes.

Each evaluation requires matching dataset provenance, candidate row wins above
50%, positive document-clustered confidence intervals, all declared controls,
and healthy close/usable rates. The final 512/512 gate additionally requires
at least 10% relative improvement over SFT. AV NLL or training reward alone
cannot promote the checkpoint.

## Failure Handling

Any failed prerequisite, runtime contract, parser gate, sustained KL guard,
gradient guard, log-prob drift guard, training exception, missing checkpoint,
or post-eval gate leaves the queue failed and prevents promotion. Preserve the
run log, offline W&B data, checkpoint metadata, and failure report. Do not
automatically weaken a guard or launch another retry.

## Verification

Before launch:

- Validate the queue with dry-run/preflight.
- Run focused queue, guard, retention, and hero static-contract tests in the
  RunAI venv.
- Confirm all protected inputs and prerequisite gate reports exist.
- Confirm no relevant training/eval process is active.
- Confirm free Longhorn space safely covers three approximately 59-GiB actor
  checkpoints plus logs.
- Sync the reviewed source/config through S3 and verify source fingerprints.

After launch, verify SGLang health, all four offline W&B role runs, the declared
6+1+1 GPU topology, and at least one genuine actor optimizer step. Then leave
the guarded cluster process running without a laptop-side dependency.
