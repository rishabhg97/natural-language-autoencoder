# R33 Family-Clean Internal Online-RL Hero

## Scope

This is a declared internal stability run, not a sealed publication study. The
retained checkpoint pair now has a protocol-matched validation report whose
report status is `confirmatory`, but that label applies to the frozen matched
evaluation protocol, not to the corpus boundary. The v6 full-exposure audit
found that all in-corpus R33 candidate families have historical exposure.
Therefore no R33 in-corpus validation or test result can be described as a
pristine sealed result. A publication-grade generalization claim requires a
new teacher-backed external corpus family.

## Fixed Recipe

- Queue: `configs/nano_rl/hero/r33_family_clean_online_joint_a3e5_internal_hero_u342_queue_8h100.yaml`
- Dataset and clean SFT actor/critic inputs: the same family-clean assets used
  by the completed `a3e-5/u24` promotion probe.
- Topology: four actor GPUs, three online-critic GPUs, and one SGLang rollout
  GPU on eight H100 NVLs.
- Actor LR: `3e-5`; critic LR: `2e-6`; constant schedule; K3 KL coefficient
  `0.003`; actor microbatch `2`; gradient checkpointing enabled.
- Rollout: 24 prompts x 8 samples = 192 generated responses per update,
  context cap 512, response cap 256, 342 updates, and new rollout seed
  `20260719`.
- Final actor and critic checkpoints are retained at iteration 342. Optimizer
  shards are not saved.

## Completed Online-RL Training

The selected pair was produced by a full online-RL run, not merely evaluated:

- Initialization: qualified clean R33 AV+AR SFT checkpoints.
- Duration: approximately 43 hours.
- Optimization horizon: 342 online-RL optimizer updates.
- Per update: 24 prompts x 8 rollouts = 192 generated responses.
- Total generated responses: 342 x 192 = `65,664`.
- Allocation: four GPUs for actor/AV training, three GPUs for online
  critic/AR training, and one GPU for the SGLang rollout engine.
- Output: the paired actor and critic `iter_0000342` checkpoints below.

## Selected Checkpoint Pair

- Actor/AV:
  `/workspace/interp/outputs/nano30b-nla-pilot/publication/r33_online_rl/internal_hero/a3e5_u342/actor/iter_0000342`
- Critic/AR:
  `/workspace/interp/outputs/nano30b-nla-pilot/publication/r33_online_rl/internal_hero/a3e5_u342/critic/iter_0000342`

This is the existing selected internal-hero pair produced by the 43-hour RL
run. The matched 384-token result below is a later validation of those trained
weights; the evaluation did not create a second checkpoint.

## Matched 384-Token Validation

- Evidence directory:
  `/workspace/interp/outputs/nano30b-nla-pilot/publication/r33_online_rl/internal_hero/a3e5_u342/eval384_matched_v122`
- Clean-SFT baseline report: `sft_roundtrip_report.json`.
- Online-RL report: `rl_roundtrip_report.json`.
- Execution record: `eval384_chain.log`.
- Boundary: 122 held-out validation rows representing 122 independent content
  families.
- Generation: identical SFT/RL rows and protocol, maximum 384 new tokens.
- Protocol SHA-256:
  `fcc431ec4450adb8817cd946d6c194fa2a45b53b0c6c42c8682c1e9f12f94d4d`.
- Controls: shuffled, zero, mean, and no-injection.
- Reproduction configs:
  `configs/nano_roundtrip/publication/r33_internal_hero_eval384_{sft,rl}_{generate,score,queue}.yaml`.

| Metric | Matched SFT | Online RL | Change |
|---|---:|---:|---:|
| Round-trip directional NMSE | `0.309055` | `0.224386` | `-0.084669` (`27.4%` lower) |
| Raw MSE | `9.5523` | `7.2665` | `-2.2858` (`23.9%` lower) |
| Generated text has lower round-trip error than teacher text | `62/122` (`50.8%`) | `103/122` (`84.4%`) | `+41` rows (`+33.6` percentage points) |
| Parse/close rate | not recorded here | `121/122` (`99.18%`) | one RL row did not close |

The RL AV+AR path beat every required control and both configured gates passed.
The RL report is marked confirmatory and generation-protocol compatible. The
execution record reports no OOM, traceback, or evaluation error.

This is strong evidence that online RL materially improved R33 AV-generated
text -> AR reconstruction over its exact matched SFT baseline. It is not a
sealed test-set result, does not establish R33-over-R27 superiority, and does
not by itself establish semantic faithfulness of the generated explanations.
Because SFT is scored by the SFT AR while RL is scored by the jointly trained
RL AR, the `27.4%` estimate is a **pair-level AV+AR improvement**, not an
actor-only effect.

## Evaluation And Claim Boundary

- The automatic post-eval is validation-only on the declared 122-family
  development boundary with the protocol-matched clean-SFT baseline.
- It requires all controls, healthy parse/close rates, at least 50% paired
  baseline wins, and a positive clustered baseline-improvement interval.
- The queue does not evaluate a test split, does not use a legacy H1 result as
  a promotion gate, and must not be summarized as a sealed or publication
  result.
- After completion, preserve the final actor/critic pair, source contract,
  generated outputs, report, offline W&B logs, and cleanup manifest. Archive
  them before any later storage reclamation.

## Publication Follow-Up

- Synchronize and hash the two JSON reports, generated records, and execution
  log into the local evidence bundle; the exact remote paths and protocol hash
  are recorded above, but this worker has not independently read those files.
- Record the report's centered R2, norm ratio, cosine, family-clustered
  interval, sign-flip result, and paired SFT win rate if present.
- Inspect the single non-closing RL generation and report content-usability
  separately from tag closure.
- Run the four-way component decomposition (SFT AV/SFT AR, RL AV/SFT AR, SFT
  AV/RL AR, RL AV/RL AR), including the independently initialized AR critic,
  to distinguish better verbalization from actor/critic co-adaptation.
- Compare SFT and RL in the functional reinjection eval on downstream logits
  and loss, then replicate the online-RL result with a second seed.
- After all choices are frozen, use a new external teacher-backed family
  boundary for the one-shot publication test. Add a matched R27 run only if
  claiming R33 outperforms R27.
