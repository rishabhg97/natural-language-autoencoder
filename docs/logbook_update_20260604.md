# Google Doc Update — New Findings (2026-06-02 / 2026-06-03)

This file contains the content added to `docs/experiment_logbook.md` after the
Google Doc was last compiled (2026-06-01). Paste the new sections below into the
existing doc **after section 9 ("2026-06-01 Nano AR Fullscan And HPO Milestone")
and before "Practical Lessons For Future Runs."** Two small edits to existing
sections are noted at the end.

---

## 9b. iter_0001547 Larger Heldout Confirmation (2048/2048)

The larger confirmation eval that section 9 listed as "running" has completed.
It confirms the quick 512/512 HPO improvement on a 2048/2048 sample.

| Split | Teacher NMSE | Teacher cosine | FVE vs mean | Teacher beats mean rowwise |
| :-: | :-: | :-: | :-: | :-: |
| Validation | 0.436878 | 0.781561 | 0.492958 | 98.54% |
| Test | 0.450516 | 0.774742 | 0.475775 | 97.71% |

**Read.** The continuation checkpoint holds up on the larger sample: teacher NMSE
stays near 0.44 and FVE near 0.49, beating the mean control on the large majority
of rows. The current AR milestone target is teacher NMSE 0.25–0.30 on both splits
without major new data or new training algorithms, so this is strong row-specific
signal but not yet at target. The next phase continues from iter_0001547 with
bounded, storage-conscious probes, then reruns the winning recipe with
exact-resume optimizer state before naming a final AR checkpoint.

# **10. Nano AR Wide Probe Queue (2026-06-02)**

On 2026-06-02T19:06Z, a serial AR HPO queue watcher was launched in the RunAI
`train` workspace to run short bounded probes without manual train/eval handoffs.

**Queue artifacts.** queue.yaml, watcher.pid, and watcher.log under
`/workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/`; remote queue script
`scripts/nano_ar_hpo_queue.py`.

**Policy.** Run one probe at a time on the `train` workspace; use 512/512
validation/test evals only; do not auto-queue 2048/2048 evals; keep W&B offline;
keep one model-only checkpoint per probe with `NLA_KEEP_LOCAL=1`; do not start RL
or AV+AR tuning during this phase.

**Initial queued probes.** r27-wide-best1547-lr3e5-cos128,
r27-wide-best1547-lr1e5-constant128, r27-wide-best1547-lr5e6-cos128,
r27-wide-fullscan-lr2e5-cos192, r27-wide-fullscan-lr5e5-cos128.

**Launch verification.** A remote dry run resolved the first probe to
`--num-rollout 128`, expected checkpoint iter_0000128, and 512/512 eval controls
teacher / teacher_shuffled / blank / generic / mean / source_context / source_raw.
Immediately after launch the queue had 1 item training and 4 pending; watcher
PID 1142.

# **11. Nano AR HPO Results Captured So Far (2026-06-03)**

On 2026-06-03, the RunAI `train` workspace was recreated under the canonical name
with the original 2-GPU GH200 shape after the prior pod failed. The recreated
workspace reached scheduler allocation but stayed blocked in `ContainerCreating`
because the `/workspace/interp` PVC reported `volume ... is not ready for
workloads`. Latest remote queue/eval artifacts may exist on the PVC but were not
readable from the pod, so remaining queued probes were not treated as complete
until their JSON reports were recovered.

**Confirmed local AR eval reports** under `artifacts/nano_ar_hpo_study/`:

| run | eval | val teacher NMSE | test teacher NMSE | val FVE | test FVE | val cosine | test cosine |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| r27-best1547-lr1e5-iter0000256 | 512/512 | 0.4417968094 | 0.4392344058 | 0.5017487246 | 0.4923874972 | 0.7791016102 | 0.7803828120 |
| r27-best1547-lr2e5-iter0000256 | 512/512 | 0.4412533045 | 0.4374333322 | 0.5023616807 | 0.4944689540 | 0.7793734074 | 0.7812833190 |
| r27-best1547-lr2e5-iter0000256 | 2048/2048 | 0.4351932406 | 0.4476521015 | 0.4949136702 | 0.4791074058 | 0.7824034095 | 0.7761739492 |

The 512/512 lr=2e-5 continuation is the best confirmed quick heldout point so far
by test teacher NMSE (0.4374333322). It beats mean and shuffled teacher controls
strongly: teacher beats mean 0.98828125 / 0.98828125 (val/test), teacher beats
shuffled teacher 0.998046875 / 1.0, and teacher beats source-context
0.64453125 / 0.685546875.

Additional wide-queue results captured in-session before the PVC became
unreadable:

| run | val teacher NMSE | test teacher NMSE | val FVE | test FVE |
| :-- | :-: | :-: | :-: | :-: |
| r27-wide-best1547-lr3e5-cos128 | 0.4417034388 | 0.4410030842 | 0.5018540266 | 0.4903434786 |
| r27-wide-best1547-lr1e5-constant128 | 0.4420669079 | 0.4399488568 | 0.5014441120 | 0.4915618235 |

**Read.** Nano AR is learning a real explanation-to-activation mapping and beats
the main negative controls, but confirmed teacher NMSE is still around 0.44, not
the 0.25–0.30 target.

# **12. RunAI PVC Recovery and Completed Wide AR Probe Readback (2026-06-03)**

Later on 2026-06-03 the `train` workspace was recovered with the original
persistent PVC attached (project trustworthy-ai-inference, pod train-0-0,
2 x NVIDIA H200; `/workspace/interp` Longhorn PVC 1008G total / 854G used /
154G free; `/workspace/models` 1.4T total / 968G used / 460G free).

**Root cause.** Stale Longhorn attach/detach state from the old `train-dev` path
plus node disk pressure / stale replica state. Recovery deleted only the completed
old pod, repaired the stale Longhorn replica/snapshot state, pruned unused node
runtime artifacts, and recreated `train` with the same PVC mounts. **PVC data was
not deleted.**

Queue readback showed all five wide probes complete (5 complete, 0 pending,
0 training, 0 eval_running, 0 failed).

**Recovered wide AR results:**

| run | val teacher NMSE | test teacher NMSE | val FVE | test FVE | val cosine | test cosine |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| r27-wide-best1547-lr3e5-cos128 | 0.4417034388 | 0.4410030842 | 0.5018540266 | 0.4903434786 | 0.7791483402 | 0.7794984579 |
| r27-wide-best1547-lr1e5-constant128 | 0.4420669079 | 0.4399488568 | 0.5014441120 | 0.4915618235 | 0.7789665461 | 0.7800256014 |
| r27-wide-best1547-lr5e6-cos128 | 0.4429176748 | 0.4404751062 | 0.5004846309 | 0.4909536499 | 0.7785412073 | 0.7797624469 |
| r27-wide-fullscan-lr2e5-cos192 | 0.4926010668 | 0.4873670042 | 0.4444525072 | 0.4367618257 | 0.7536994815 | 0.7563165426 |
| r27-wide-fullscan-lr5e5-cos128 | 0.4728046060 | 0.4699196219 | 0.4667786347 | 0.4569253403 | 0.7635977268 | 0.7650401592 |

**Read.** The best1547 basin remains stable around 0.44 teacher NMSE. The fullscan
escape probes were worse, not better, so the next AR step should diagnose
bottlenecks rather than simply widen this same short-run sweep. The best confirmed
quick point remains r27-best1547-lr2e5-iter0000256 at 512/512 with test teacher
NMSE 0.4374333322.

# **13. Final AV Checkpoint Standalone Generation Sanity (2026-06-03)**

After RunAI recovery, a short standalone-generation sanity was run on the
completed AV hero checkpoint hf_iter_0000467 (8 validation rows, 8 test rows,
2 generated validation examples, 80 max new tokens).

**Teacher-forced NLL controls:**

| Control | Validation NLL | Test NLL |
| :-- | :-: | :-: |
| real | 0.8636959568 | 1.0225046575 |
| shuffled | 1.2146585882 | 1.3821892142 |
| zero | 1.1028810516 | 1.2680757344 |
| mean | 1.0978926346 | 1.3128505871 |
| none | 1.2453771383 | 1.4175161719 |

**Real activation gaps vs controls.** Validation: +0.3510 shuffled, +0.2392 zero,
+0.2342 mean, +0.3817 none. Test: +0.3597 shuffled, +0.2456 zero, +0.2903 mean,
+0.3950 none.

**Standalone generation.** Example 0 (job-advertisement continuation after "And"):
real AV generation correctly described coordinating-conjunction/list continuation,
employee benefits/growth, and corporate promotional tone (content_f1 = 0.4835);
shuffled, zero, and no-injection generations drifted to unrelated
legal/weather/biography domains. Example 1 (customer-service responsibility list):
real AV generation stayed on job responsibilities and formal job-description
register (content_f1 = 0.4694); controls again drifted off-domain.

**Read.** AV standalone verbalization is coherent on this short sanity check and
the real activation clearly carries document-specific information that the controls
do not. This is not a broad qualitative eval, but it supports treating the AV hero
checkpoint as a usable standalone verbalizer for manual inspection while AR remains
the weaker leg.

# **14. AR Phase-2 Gate, S3 Sync, and Queued Bounded Probes (2026-06-03)**

S3 sync between the Mac and RunAI was made reliable using existing RunAI S3
credentials plus a proxy override: removing `pdx.s8k.io` / `.s8k.io` from
`NO_PROXY`/`no_proxy` lets S3 traffic use the egress proxy (the prior setting
forced direct pod-to-S3 traffic and timed out). Source bundles were uploaded by
presigned PUT and downloaded in the pod by presigned GET.

**PVC cleanup/archival** ran in the background to
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/runai-outputs-archive/`, archiving
only old bulky AR wide/fullscan/critic-init checkpoint dirs; the final AV hero and
current best tiny AR run metadata were left local. The current-best AR checkpoint
(nano-ar-r27-best1547-continue-lr2e5-cosine-256steps-20260602T0710Z) had its model
shards archived to S3 (65,980,972,670 bytes) and its restore was started so bounded
continuation probes can resume from exact Miles shards. Reusable AR split parquets
were copied to a protected dir (ar-r27-275k-splits-20260530; 247,870 train /
13,761 validation / 13,765 test; doc overlap 0).

**Correctness audit on restored current-best metadata** (iter_0000256/hf): passed
= true; HF num_hidden_layers = 28; sidecar critic.extraction_layer_index = 27;
value-head identity distance = 0.0932226554; doc split overlap = 0. This confirms
the intended R_27 / K+1 critic interpretation: R_27 has extraction hidden-state
index 27 and the truncated critic has 28 hidden layers.

**Information-ceiling diagnostic** (train 5,000; val/test 512/512; feature dim 256;
kNN k=8). Scaling the per-dimension diagnostic by d_model=2688 puts it on the eval
scale; the simple text-hash kNN explanation floor is worse than the mean baseline:

| split | kNN x2688 | mean-baseline x2688 |
| :-- | :-: | :-: |
| validation | 0.9193 | 0.8864 |
| test | 0.9035 | 0.8647 |

**Read.** There is no easy duplicate/retrieval shortcut in teacher explanation
text. Learned AR at test teacher NMSE 0.4374 remains far better than this retrieval
floor and the mean control, but the source-raw control around 0.13 still shows much
more information lives in the original token stream than in explanation text alone.

**Queued phase-2 bounded probes** (both W&B offline, 128 continuation steps,
512/512 evals, protected 275k split path, current-best resume):

| queue item | config | purpose |
| :-- | :-- | :-- |
| r27-best256-polish-lr1e6-cos128 | r27_best256_polish_lr1e6_cosine_128steps.yaml | gentle low-LR polish from current best |
| r27-best256-batch384-lr2e5-cos128 | r27_best256_batch384_lr2e5_cosine_128steps.yaml | larger-batch current-basin probe |

---

## Small edits to existing doc sections

- **Section 9 → "Follow-up":** replace "The larger 2048/2048 eval for iter_0001547
  is running …" with the completed result now in section 9b above (validation
  teacher NMSE 0.436878, test 0.450516; FVE 0.492958 / 0.475775).
- **Section 7 → "Not yet proven":** the bullet stating the larger iter_0001547
  v2048/t2048 eval "is still the confirmation gate" can be updated — that eval has
  now run and confirmed the quick result near 0.44 NMSE; the open AR question is
  now reaching the 0.25–0.30 target and selecting a final recipe, not the
  confirmation eval itself.
