# Nano30B NLA Current State

<!-- R33-HERO-BASELINE-PROTOCOL-INVALIDATED -->

> [!CAUTION]
> Publication status (`2026-07-16`): the selected family-clean R33 SFT AV+AR
> checkpoint pair is qualified for directional AV-to-AR reconstruction and
> stored-snapshot functional recovery. It is not qualified for raw-magnitude
> recovery without post-hoc calibration, exact fresh-forward replay,
> external-boundary generalization, R33-over-R27 superiority, or RL gain.
> The historical July 8 RL effect remains invalidated. Metric definitions are
> fixed in `docs/methods/measurement_contract.md`; the release record is
> `docs/runs/r33_clean_sft_av_ar_20260715.md`.

Last organized: `2026-07-16`.

This is the short canonical entry point for the active milestone. Long-form
chronology remains in `docs/experiment_logbook.md`, active queue notes remain in
`docs/nano_av_job_tracker.md`, and completed run history remains in
`docs/nano_av_run_history.md`.

## Qualified SFT Checkpoint Pair

The deterministic dataset and v2 family-overlap audit pass. The manifest has
`27,647` documents, `275,396` rows, and `5,009` exact-refined content
families. Clean AR and AV training exposure covers `4,504` families;
validation/test contain `250 / 255` disjoint families and
`13,766 / 13,765` eligible rows, with zero train-family or cross-split-family
overlap.

Both component gates and both end-to-end gates pass:

- AR teacher-text validation directional MSE/cosine/FVE-NRM:
  `0.281703 / 0.859148 / 0.584534`; centered raw R2 is `-0.201696`.
- AV validation real NLL: `0.776775`, versus shuffled `1.311727`, zero
  `1.176494`, mean `1.237522`, and no-injection `1.220974`.
- AV-generated-text to AR directional MSE: validation `0.307004`, test
  `0.319225`; teacher text is `0.304714 / 0.302637`.
- Generated explanations are closed and usable on `512/512` rows in each
  split. Every registered control passes with positive family-bootstrap lower
  bounds and rowwise win rates from `99.61%` to `100%` on test.
- Stored-snapshot functional reinjection is teacher-level within
  family-clustered uncertainty and decisively beats mean, zero, and shuffled
  controls.

The fail-closed pair manifest reports `qualified: true` for release ID
`r33-clean-sft-av-ar-iter1291-20260715`. Both checkpoint fingerprints and all
six verifier reports are bound into that manifest. The AV and AR model payloads
are preserved on S3; lightweight evidence, generated text, training/eval logs,
and offline W&B data are local under
`artifacts/runai_eval/r33-clean-sft-av-ar-qualified-20260715/`.

The test boundary is family-disjoint from clean SFT training, but is not
guaranteed untouched by every historical exploratory run. The stored target
also failed exact fresh-forward replay, and centered raw R2 remains negative.
Those caveats bound the public claim; they do not invalidate the directional
stored-snapshot AV+AR result. No publication-valid RL model has been trained
from this clean pair.

## Online Joint RL Canary

The first clean-lineage online joint AV+AR canary completed on `2026-07-17`
after the accidentally suspended `train` workspace was resumed on eight H100
NVLs. The run used four actor GPUs, three critic GPUs, one SGLang rollout GPU,
24 rollout samples per update, and two updates. Both actor and critic
checkpoints committed atomically.

An exact 64-row family-stratified comparison was then run against the clean SFT
pair with identical row keys, dataset hashes, generation protocol, five
controls, and 64 independent families. Online/SFT directional MSE was
`0.291993 / 0.292173`, but the relative gain was only `0.0618%`, paired wins
were `32 / 64`, the family-bootstrap interval crossed zero, and the sign-flip
test gave `p=0.4824`. Raw MSE worsened from `8.797533` to `8.969927`.

The strict promotion gate therefore reports `passed: false`. This proves the
online actor+critic path can train, checkpoint, generate, and evaluate
correctly; it does not show that two RL updates improve the SFT checkpoint.
The canonical run summary is
`docs/runs/r33_online_joint_canary_20260717.md` and lightweight evidence is
local under
`artifacts/runai_eval/r33-online-joint-canary-evidence-20260717T0951Z/`.

## Publication Boundary And Magnitude Audit

The `2026-07-16` selected-pair exposure audit supersedes the earlier
evaluation-only v4 audit. It enumerated `136` selected-pair training,
validation, test, and historical evaluation sources spanning `28,665` unique
documents. All `5,009` canonical content families were exposed and no document
was unmapped, so an in-corpus confirmatory split is impossible for this pair.
The report/inventory/joint-manifest SHA-256 values are
`373e2988...088b`, `c193f2ef...abb`, and `9d68a894...bc20`.

An independent inventory found `63` candidate teacher tables, `53` usable
tables, and no teacher-backed numeric document ID outside the already exposed
`0..38161` range. Under the standing rule not to generate new teacher text,
the next genuinely confirmatory boundary must therefore come from an external
teacher-backed corpus.

The 64-row validation-only activation audit also bounds the snapshot claim:

- all strict fresh-vs-stored identity checks fail;
- repeated fresh forwards are exact, and full-forward versus
  extraction-forward values are exact;
- fresh-vs-stored cosine is `0.999142` on average (`0.983146` minimum), with
  centered raw R2 `0.997255` when the stored vector predicts the fresh vector.

This is stable runtime drift relative to the archived snapshot, not stochastic
forward instability. Keep the functional claim scoped to stored-snapshot
counterfactual reinjection.

A one-parameter magnitude audit fitted a nonnegative origin scalar on
validation teacher reconstructions only. The selected scalar is `0.560604`.
It leaves directional MSE unchanged while changing AV reconstruction as
follows:

| Split | Raw MSE before | Raw MSE after | Centered raw R2 before | Centered raw R2 after |
|---|---:|---:|---:|---:|
| Validation | `9.446685` | `3.648806` | `-0.326250` | `0.487733` |
| Exploratory test | `9.647148` | `3.770353` | `-0.335374` | `0.478102` |

The test family-clustered 95% interval for raw-MSE improvement is
`[5.634404, 6.123574]`, with the calibrated candidate better in every
bootstrap draw. This supports a simple global scale-mismatch diagnosis, but it
is post-hoc and test-exposed. It does not establish native or exact
raw-magnitude recovery.

The complete lightweight follow-up evidence is local under
`artifacts/runai_eval/r33-clean-sft-publication-evidence-20260716/` and mirrored
at
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/publication/evidence/20260716_r33_clean_pair/`.

The frozen-cache subgroup audit is also complete. Validation-fitted bins were
applied unchanged to exploratory test for source length, target length,
activation norm, and family frequency. Every one of the 16 validation/test
bins has sufficient rows and families, and every registered family-clustered
control interval stays positive. The weakest test bin is the lowest activation
norm quartile (`0.370077` directional MSE; calibrated centered raw R2
`0.415606`). A 50-validation/50-test source-and-teacher-grounded qualitative
packet has zero automatic structural flags. The original panel builder omitted
the resolved source-text column; that evidence was replaced, not reused. The
corrected source-grounded panel SHA-256 is
`4f5d61486330b1104dd0a256ea185d8c1c99512ee9ff4731f8135305924f81c8`,
and its structural-review SHA-256 is
`33de2720d96bda3f663318be7fe8c10765740a3d603bcdf76682ca23661297da`.
It has not received blinded human semantic ratings, so no semantic-quality
claim is made.

An automatic release-text triage scanned all `1,024` frozen generated
explanations and found no configured PII, credential, internal-path, or private
endpoint pattern and no source-copying failure. The report SHA-256 is
`00e501ff644483e614d0b60071f726f33575c95ba8b81d5c6b59c4bd79d13419`.
Fourteen phone-like patterns occur in source excerpts, not generated output;
source examples remain internal pending human adjudication or redaction. This
is automatic triage, not proof of privacy or consent.

The independent critic rebuild is now provenance-complete. Canonical manifest
hashing ignores only the transient `value_head.before_sha256` field, while all
final state and initialization checks remain mandatory. The semantic manifest
SHA-256 is
`34e863f756e0749ca19fc8c138b7bd71b5da69c907ee42ad021517542e5c8941`;
the independence verifier passes all 16 checks and has SHA-256
`4639285fea694f7f850c766b31d8ddea4e2b2bdd61c710f3b2a4cdd2109fb6e3`.
The seed-`314159` independent clean AR queue completed 1,291 optimizer updates
on four H100-NVL GPUs with the primary recipe and validation-only selection.
Its 512-row component verifier passes: teacher directional MSE/cosine/FVE-NRM
are `0.286169 / 0.856916 / 0.577948`, close to the primary seed's
`0.281703 / 0.859148 / 0.584534`. The component report SHA-256 is
`368c84cad8bb0b8a7235b1a1e96c862b69d33be72d49c1f07a563e62d0be65aa`.
Frozen AV-generated validation text was then scored through this independent
AR. The 512-row verifier passes: AV-text directional MSE is `0.310963`,
independent teacher-text directional MSE is `0.308533`, all rows are closed and
usable, and all five controls pass family-clustered inference. Rowwise wins are
`99.80%` against mean and `100%` against the other controls. The
report/verifier SHA-256 values are
`6f0829a61b03ac584b109c1c7a54f689f0a14b61f7fee803afd3d5ca29bf552b`
and
`dd3de6e1dd10f0f64c25e23ff3152638c9cf46785133dfb6c0466c353e434331`.
This qualifies validation-only cross-critic directional replication; raw
magnitude remains unsupported.

The independent HF checkpoint is fingerprinted at 10 files and
`38,462,226,688` bytes with directory SHA-256
`c2eea74f5baccee97128617b05636187804c7e59aedc560d088dbf65d52f1925`.
S3 contains the exact object count and byte total. After that verification, a
manifest-first retention pass deleted only the redundant 36G model DCP and
72G optimizer state while preserving HF, logs, split metadata, and offline
W&B evidence.

Hash-bound selected-run compute accounting covers all 1,291 logged optimizer
steps for primary AR, primary AV, and independent AR. Successful training totals
`138.7456` H100-NVL GPU-hours: primary AR `3.8867h x 4`, primary AV
`13.4608h x 8`, and independent AR `3.8781h x 4`. The report SHA-256 is
`7bde74be3a874d2ae305463ca8da211c069ce0bf1001802b6bdf7ab091fd7238`.
It explicitly excludes extraction, conversion, evaluation, historical HPO/RL,
and failed diagnostic compute that cannot be reconstructed exactly.

A static security audit of the source-only internal snapshot found no
checkpoint/data payload, forbidden heavy extension, symlink, oversized text,
or unallowlisted credential fixture. It is not a public bundle: 31 files still
contain local-home, internal S3/endpoint, or cluster references and must be
redacted or excluded from the exact staged release. Audit SHA-256 is
`bf697acdad268e219cfd729428233b1259b07aea988529e8c7e4bb01b8a62c42`.

That internal-snapshot failure has now been resolved for a narrower candidate
public bundle. The deterministic builder stages release-relevant source,
configs, tests, aggregate evidence, and compact curves for all 3,873 selected
optimizer steps while redacting workstation, cluster, and internal object-store
locators. The exact 496-file staged tree passes the security audit with zero
failed finding files, forbidden paths, symlinks, binaries, or oversized files.
Its audited tree SHA-256 is
`df175c5f61cefbfc1a02451a7bd242ba69e1cb602cdd97ca4b8bd8fe9c263b77`;
the audit-report SHA-256 is
`a130d8e4295a06d0372bd0d920d9dfd0c8f7649710652939b33dbc764089f096`.
The deterministic 6,859,370-byte archive has SHA-256
`3eb8e64ed0d9d61ed2d6b0694fbaf96b99051a63f2ce1a6c99372d93832e573a`,
and archive reinspection produced the same tree hash. No weights, generated
text, optimizer state, activation parquet, credentials, or W&B binary files are
included. This closes the technical staging blocker, not the human or legal
release gates.

The archive, SHA-256 sidecar, manifest, security report, and attestation are
also preserved as five objects under the internal S3 release-candidate prefix
`publication/release-candidates/r33-clean-sft-av-ar-iter1291-20260716/`.
The listed remote archive size is the expected `6,859,370` bytes.
The exact human-readable index is
`docs/releases/r33_clean_sft_release_candidate_attestation.md`.

## Superseded Remediation Record

The bullets below preserve the pre-qualification state and decisions for audit
history. Statements that an AV eval or round trip was pending are superseded by
the qualified result above.

- Completed clean AR run:
  `nano-ar-r33-publication-deterministic-family-clean-4gpu-unfusedtorchconv-expertscan-cudablock-lr5e5-warmup25-gb192-mb48`.
  It completed 1,291 updates and validation-only 512-row evaluation. Teacher
  directional MSE/FVE-NRM/cosine are `0.281703 / 0.584534 / 0.859148`;
  raw MSE is `8.537785` versus train-mean `7.104776`, so centered raw R2 is
  `-0.201696`. Shuffled directional MSE is
  `0.968888`, mean is `0.678041`, source context is `0.301252`, and source raw
  is `0.083248`. Teacher beat shuffled on all 512 rows and beat blank,
  generic, and mean on at least 98.4% of rows. The confirmatory test split is
  still unopened. The hash-bound verifier passes with claim scope
  `directional_activation_reconstruction`; centered raw R2 is negative, so no
  raw-magnitude claim is supported.
- A new fail-closed AV packed-vs-padded gate found and fixed a second packed
  contamination path: `NemotronHForCausalLM.forward` dropped packed
  `position_ids` before the backbone. Commit `6abfe18` repairs and validates
  the contract. All earlier packed AV/RL actor checkpoints remain internal
  evidence only.
- Corrected `dyn4096` AV proof passed with max absolute/relative response-NLL
  difference `0.01632524 / 0.00704313`, steady step `67.33s`, and post-Adam
  peak allocated/reserved memory `70.45 / 78.24 GiB` on four H100-NVL GPUs.
- Selected clean AV config:
  `configs/nano_av/publication/r33_family_clean_sft_8gpu_dyn4096.yaml`; queue:
  `configs/nano_av/publication/r33_family_clean_sft_8gpu_dyn4096_queue.yaml`.
  It keeps the clean family split, `gb192`, LR `1e-4`, warmup 25, offline W&B,
  validation-only 512-row selection, and the mandatory equivalence gate. The
  live eight-GPU gate passed at packed/padded NLL `2.563666 / 2.565836`, max
  absolute/relative difference `0.022072 / 0.007950`. Training completed all
  1,291 updates at `2026-07-10T14:09Z`; final loss/gradient norm/LR were
  `0.683009 / 0.585938 / 1e-5`. All 1,291 logged losses, gradients, LRs, and
  router metrics were finite, all 128 experts remained active, and no OOM,
  CUDA, or traceback signal occurred. Its validation-only `512`-row eval now
  passes a hash-bound verifier: real NLL is `0.776775`, versus shuffled
  `1.311727`, zero `1.176494`, mean `1.237522`, and no-injection `1.220974`.
  No test rows were consumed.
- The queue-chain watcher promoted AV only after the AR queue recorded both its
  checkpoint and evaluation report as complete.
- Manifest-first pre-hero cleanup removed only superseded publication paths
  after freezing lightweight metadata. Protected paths all passed post-check,
  and `/workspace/models` free space increased from about `196 GiB` to
  `255 GiB`.
- The selected clean AR HF checkpoint is mirrored to S3 and verified at 10
  objects and `38,462,226,607` bytes. Its redundant `72G` optimizer and `36G`
  model DCP copies were removed through a completed retention manifest; the
  local HF model remains protected and directly evaluable.
- The corrected AR audit proves the full R33 geometry contract: extraction and
  last retained block index 33, 34 configured blocks, tensor blocks 0 through
  33, stripped LM head/final norm, finite value head, and zero document overlap.
- At this historical point, a validation-only clean AV->AR round-trip queue
  still required rerendering after identity-binding and metric changes. That
  work is now complete: the selected pair passes hash-bound 512/512
  validation/test round-trip gates, and the test boundary is explicitly
  classified as exploratory rather than pristine confirmatory evidence.
- At this historical point, the first independent critic copy was incomplete
  and could not initialize a clean run. The replacement seed-`314159` critic
  was subsequently rebuilt, fingerprinted, trained, evaluated, preserved in
  internal S3, and retention-cleaned. Its validation-only cross-critic gate
  passes at AV-text directional MSE `0.310963` versus teacher `0.308533` on
  512 rows, with all controls passed.
- At this historical point, confirmatory RL was blocked on the clean SFT
  round-trip baseline and a genuinely independent clean critic. Both SFT gates
  now pass, but no clean confirmatory RL is claimed or required for the
  supervised checkpoint release. Every queue item captures source and
  environment provenance in an immutable launch contract; an optional
  `preregistration` mapping hash-binds the study plan, family evidence, SFT
  baseline, seed, guards, endpoints, and validation-only policy. The draft finite stability grid and final
  decision rule are in `docs/runs/r33_publication_preregistration.md`; no clean
  RL job has been launched.

- Deterministic extraction source commit:
  `0dabaade33ee35a3ff7419d2f99be2551439ab13`.
- Source archive SHA-256:
  `cbbde5ee91c2513f69a5abf8a5c57d0c5bcb9a9c905bd473e5d4b71e2ad58e27`.
- Runtime SHA-256:
  `1b7ca243028de224a06702c4bd3e1e2d3d9f75fe84f19b24610c146d66d70ff1`.
- Base-model SHA-256:
  `abd6d1368f9d2baa1b6f5b4047916db780466193af85b4772bbf5dc64c218019`.
- Clean-SFT readiness code/config snapshot:
  `294cc1e1619c42ea454c8d29e9f477e7ae4d4322`. The readiness evidence was
  generated from this snapshot, and its complete 588-file tree fingerprint is
  `1c7c0abbad68bebc426acff74ea1b14ed1559adacda28496d2b526154032c020`;
  later launch-critical fixes are not ancestors of that commit, so it is not
  described as the final immutable launch source. Each future run must bind its
  own resolved source fingerprint. The recorded readiness fingerprint is
  `a79128f8c479620b43df9e69b13f9e24d678bfecf7a6484d4823c4e188aa43b6`.
- Deterministic extraction profile: PyTorch deterministic algorithms on,
  TF32 off, cuDNN benchmarking off, float32 matmul precision `highest`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, seed `20260709`.
- Dataset root:
  `/workspace/interp/outputs/nano30b-nla-pilot/publication/r33_deterministic_full275396`.
- Base, AR-SFT, and AV-SFT each contain exactly `275,396` rows with
  `d_model=2688`, zero nonfinite activations, and zero empty/malformed text.
- The refined manifest has `5,009` content families across `27,647` documents.
  Train/validation/test contain `247,865 / 13,766 / 13,765` rows and
  `4,504 / 250 / 255` families, with zero document, family, or exact-prefix
  content overlap. Manifest SHA-256 is
  `479cbab5d21cd031cb72a770eebb3428e0d5419ebf8cce38c2ca6025e49741b6`.
- An independent full eight-shard replay produced the exact same merged
  Parquet SHA-256 as the primary extraction:
  `e3008a150831b8e894eac0de9f360a46823ffbfbd7cc73a9673f7e61e84521ac`.
  This proves deterministic replay, not equality to a fresh model forward.
- The 32-row live-vs-stored diagnostic remains decisive for fidelity. It failed
  all rows at the predeclared tolerance, with `2.562%` mean and `7.728%` max
  relative L2 drift and `publication_ready=false`. Stored-snapshot paired
  comparisons remain useful, but the repo does not claim exact fresh-forward
  activation reconstruction.
- The primary deterministic R33 critic init and fresh seed-`314159`
  independent critic init both passed the explicit initialization verifier:
  shared base/data/layer/dtype, identity primary head and router, distinct
  seeded independent value head, and changed independent router parameters.
  The first incomplete copy was superseded by a fully fingerprinted rebuild;
  that rebuild initialized the completed independent AR training run.
- The original primary AR, independent AR, and AV readiness dry-runs all exited
  `0`, resolved only the deterministic dataset, and retained validation-only
  selection. Primary AR, primary AV, and independent AR later completed all
  1,291 optimizer steps and their hash-bound validation gates. The selected
  release pair and independent AR checkpoint are preserved in internal S3.
- Readiness evidence is local under
  `artifacts/runai_evidence/20260709_r33_publication_remediation/clean_sft_queue_readiness/`
  with archive SHA-256
  `f6045b10e1e4573635c00cd49418137870b9309a2323faeb82db9e3adcb85c3a`.
  Its S3 mirror is pending recovery of the RunAI egress proxy; this does not
  weaken the local/RunAI checksum match or launch gate.

Pre-remediation SFT/RL checkpoints remain internal salvage evidence only. The
clean AV component gate, protocol-matched SFT round trip, and independent AR
lineage are complete. Remaining release blockers are the exact staged public
bundle audit, blinded human review, license/teacher-terms approval, and an
external teacher-backed boundary for any confirmatory generalization claim.

## Historical Internal Milestone (Publication-Invalidated)

On `2026-07-08`, the project selected the following internal R33 RL hero
checkpoint. That selection is now historical and publication-invalidated:

```text
R33 h -> RL-tuned AV-generated explanation -> fixed R33 AR reconstruction
```

- Run:
  `r33-corrected-k3-hero-lr1e5-update342-resume228-retry3`
- Actor checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_hero/r33_corrected_k3_hero_lr1e5_update342_resume228_retry3/actor/iter_0000342`
- Final `512/512` report:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_hero/r33_corrected_k3_hero_lr1e5_update342_resume228_retry3/roundtrip_iter_0000342_v512_t512_report.json`
- Gate: `passed=true` against exact matched clean-SFT rows.
- Validation/test RL AV-real NMSE:
  `0.000087528 / 0.000091176`.
- Relative improvement over clean SFT:
  `30.97% / 32.34%`.
- Rowwise wins over SFT:
  `83.40% / 88.67%`.
- Closed/usable generation rates:
  `99.02% / 100%` validation and `99.41% / 100%` test.
- All generation controls were beaten, dataset hashes and row identities
  matched, and document-clustered bootstrap intervals were positive.

The historical component-full SFT checkpoints used by that old gate were:

- AR:
  `nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96`,
  checkpoint `iter_0001289`.
- AV:
  `nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512`,
  checkpoint `iter_0001291`.
- SFT round-trip validation/test NMSE:
  `0.000109680 / 0.000121664` in its original report; the hardened row-matched
  baseline used by the RL gate was `0.000126796 / 0.000134752`.

Canonical result details and evidence hashes are in
`docs/runs/r33_rl_hero_20260708.md`.

Publication caveat: this old comparison is protocol-invalidated and cannot be
repaired merely by adding R27 or another seed. It is retained as internal
systems and hypothesis evidence only.

## Historical July 8 RL Run Status

- Queue completed at `2026-07-08T03:20:52Z` with no recorded error.
- Selected topology: six actor FSDP GPUs, one SGLang rollout GPU, and one
  frozen AR critic GPU on `8x H100 NVL`.
- Selected recipe: constant actor LR `1e-5`, K3 coefficient `0.001`, global
  batch `384`, actor microbatch `32`, and eight samples per prompt.
- Selected lineage: 342 optimizer updates and 131,328 generated responses.
- The update-342 `64/64` prerequisite gate and `512/512` promotion gate both
  passed.
- Final model-only actor checkpoint is retained; temporary HF eval exports were
  cleaned.
- All eight GPUs were idle at the final status check. Free space was about
  `342 GB` on `/workspace/interp` and `456 GB` on `/workspace/models`.
- Lightweight evidence is mirrored locally and on S3 with SHA-256
  `78cbf98d27188594c25cbf9c0d695f0b3b1754df978961585bbaa6fc178f0bc7`.

## Critical Status Correction

The Nano/Nemotron-H packed-boundary contamination bug and the Miles LR schedule
bug were confirmed after the R33 AR/AV scouting runs below. Treat all pre-fix
AR/AV training results as directional evidence only:

- AR/AV checkpoints trained before the packed-boundary fix are not clean hero
  checkpoints.
- Metrics from those checkpoints still matter as scouting signal because
  evals were held out and control-based, but the training path was noisy and
  mismatched.
- Runs labeled `cosine` before the LR-schedule remediation should be read as
  requested configs, not proof that a decayed LR schedule actually ran, unless
  the run has an LR-decay canary or final-LR evidence.

That path produced the historical component-full R33 AV+AR milestone above.
The later activation and packed-position audits superseded it for publication;
pre-remediation R33/R27 results remain directional context only.

## Objective

Reach a credible R33 Nano30B NLA hero milestone:

```text
h -> AV-generated explanation -> AR reconstruction
```

Teacher-text AR metrics and AV real-vs-control losses are useful proxies, but
they are not sufficient proof of a good NLA. The promotion gate is the actual
AV-generated-text to AR reconstruction round trip, compared against the mature
R27 fallback.

This gate passed first for the clean R33 SFT hero and then for the selected
R33 RL hero against exact matched SFT rows.

## Historical RL Systems Smoke Status (2026-06-23)

As of `2026-06-23`, the 4-H200 RL smoke/debug line has one clean live-sync
systems pass. This is not yet a quality-bearing RL result, but it does prove
the actor/rollout/critic loop can execute two rollouts with live
actor-to-SGLang weight sync, frozen AR reward scoring, and a saved actor
checkpoint.

Completed live-sync smoke:

- Queue:
  `configs/nano_rl/r33_component_full_pilot_queue_4h200_len512_rb2_sync2_nosaveoptim_unifiedenv_mambawheels_tokcompat_nopackedcheck_criticfwd_evalmode_nofastpath_timesteplimit_allocseg.yaml`
- Run dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_pilot/r33_component_full_sft_init_512row_4h200_len512_rb2_sync2_nosaveoptim_unifiedenv_mambawheels_tokcompat_nopackedcheck_criticfwd_evalmode_nofastpath_timesteplimit_allocseg`
- Topology: `2` actor H200s, `1` rollout/SGLang H200, `1` critic H200.
- Runtime controls:
  unified SGLang/Torch env, Mamba wheels, TokenizersBackend fallback,
  `NLA_ASSERT_PACKED_EQUIV=0`, critic forward in eval/no-grad mode,
  Mamba fast-path disabled only during critic reward scoring,
  AR-critic `time_step_limit` JSON float sentinel normalization,
  `NLA_FREEZE_CRITIC_TRAIN=1`, and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Result:
  queue completed at `2026-06-23T20:26:56Z`; two rollouts completed
  (`4/4` generations each); actor checkpoint saved at `actor/iter_0000002`.
- Rollout snapshots:
  rollout 0 raw reward `-0.946493`, response length `235.75`,
  actor train time `175.0s`; rollout 1 raw reward `-0.619056`, response
  length `234.0`, actor train time `10.9s`.
- Memory/checkpoint notes:
  rollout 1 actor train reached about `143.1 GiB` on each actor GPU but did
  not OOM. The final actor DCP checkpoint took about `2m20s` to flush and
  wrote two model shards of about `31.6GB` each plus metadata.

Earlier skip-sync smoke:

- Queue:
  `configs/nano_rl/r33_component_full_smoke_queue_4h200_len256_rb2_fix2_freezecritic.yaml`
- Run dir:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_smoke/r33_component_full_sft_init_512row_4h200_len256_rb2_fix2_freezecritic`
- Topology: `2` actor H200s, `1` rollout/SGLang H200, `1` critic H200.
- Runtime caveats:
  `NLA_SKIP_ROLLOUT_WEIGHT_SYNC=1` and `NLA_FREEZE_CRITIC_TRAIN=1`.
- Result:
  rollout `4/4`, actor step `0` completed, reward/train MSE equivalence
  `mean=1.0000`, `max|r-1|=0.0000`, `n=4`, actor checkpoint saved at
  `actor/iter_0000001`.
- Key scalar snapshot:
  raw reward `-0.280732`, shaped reward `5.349516868591309e-06`, advantage
  `5.304813385009766e-06`, actor loss `-5.304813385009766e-06`, grad norm
  `9.3125`.

Superseded failed live-weight-sync pilots:

- The no-save-optim pilot failed during distributed rollout weight sync after
  the first `backbone.embeddings.weight` bucket metadata was sent.
- The barrier-fix retry is stale in queue state but logs indicate SIGTERM and
  SGLang healthcheck timeouts.
- The unified-env retry failed because the SGLang env lacked `accelerate`, which
  Miles FSDP imports during actor initialization.

Do not describe the RL smoke as improved NLA quality. The completed live-sync
run is systems evidence only: two rollout/update cycles can run on 4 H200s.
Next RL work should run a slightly larger quality smoke with interval
checkpointing disabled or delayed, then evaluate round-trip/AR reconstruction
before any hero-scale RL promotion.

## Superseded Strict-Dedup Candidate Decision

On `2026-06-15`, this clean R33 AV+AR pair was selected as the final candidate
for hero planning. It has since been superseded by the component-full
`2026-06-21` hero milestone above, but remains useful proof that the clean
round-trip gate was working before the full run.

Candidate AR checkpoint:

- Run id:
  `nano-ar-r33-dedup-clean56k-lr5e5-cosine-warmup25-gb192-mb96-128step-padded`
- Checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-clean56k/nano-ar-r33-dedup-clean56k-lr5e5-cosine-warmup25-gb192-mb96-128step-padded/checkpoints/iter_0000128`
- Bounded `512/512` eval:
  validation/test teacher NMSE `0.361513 / 0.352040`.
- Teacher text beat shuffled almost perfectly and is good enough for the AV
  round-trip gate.

Candidate AV checkpoint:

- Run id:
  `nano-av-r33-dedup-20k-lr1e4-cosine-warmup5-gb192-mb2-seq1152-dyn512-32steps`
- Checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-dedup-smoke/nano-av-r33-dedup-20k-lr1e4-cosine-warmup5-gb192-mb2-seq1152-dyn512-32steps/checkpoints/iter_0000032`
- AV eval:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-dedup-smoke/nano-av-r33-dedup-20k-lr1e4-cosine-warmup5-gb192-mb2-seq1152-dyn512-32steps/eval_iter_0000032_v512_t512_gen4_report.json`
- Round-trip eval:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-dedup-smoke/nano-av-r33-dedup-20k-lr1e4-cosine-warmup5-gb192-mb2-seq1152-dyn512-32steps/roundtrip_iter_0000032_v64_t64_report.json`

Round-trip result:

- AV real NLL beat shuffled, zero, mean, and no-injection controls on
  validation/test.
- Generated explanation parse health was perfect on validation/test:
  closed fraction `1.0`, usable fraction `1.0`, empty fraction `0.0`.
- R33 AV-generated text beat all round-trip controls:
  validation/test AV-real NMSE `0.000128805 / 0.000135702`.
- Validation controls: mean `0.000233943`, none `0.000294733`, zero
  `0.000295612`, shuffled `0.000327986`, AV-mean `0.000346130`.
- Test controls: mean `0.000241159`, none `0.000301629`, zero
  `0.000303576`, shuffled `0.000335947`, AV-mean `0.000346187`.
- Rowwise win rates versus generated controls were `63-64/64` or `64/64`.
- The strict report-level gate still records `passed=false` only because the
  R27 baseline comparison is not row-identical
  (`baseline_row_identity_match=false`) and was intentionally waived for this
  decision. Do not reinterpret that field as an in-run control failure.

Next mode update: cleanup and hero-run review completed, then the R33
component-preserving full hero path ran to completion. Do not start RL from this
alone; the next research decision should be whether to run a fresh row-matched
R27 comparison or move toward carefully scoped post-hero analysis.

## Component-Full Hero Execution

As of `2026-06-15T22:31:59Z`, the active clean hero path is the
component-preserving `275,396` row R33 run, not the earlier strict-dedup
`56,351` row candidate.

Dataset gate:

- Root:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_component_fullscan275396`
- Source rows: `275,396`; `d_model=2688`.
- AR verifier:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_component_fullscan275396/verify_ar_R33_component_fullscan275396.json`
- AV verifier:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_component_fullscan275396/verify_av_R33_component_fullscan275396.json`
- Materialized component splits report doc/content overlap `0`; synthetic
  split checks are intentionally skipped by the verifiers for this
  materialized split mode.

AR hero:

- Run id:
  `nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96`
- Checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001289`
- Eval:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/eval_iter_0001289_v512_t512_winrates_report.json`
- Validation/test teacher NMSE: `0.320616 / 0.292730`.
- Validation/test source_raw NMSE: `0.095084 / 0.080078`.
- Teacher text beats shuffled, blank, generic, and mean controls strongly;
  source_raw remains the lower-bound control as expected.

AV smoke gate:

- Run id:
  `nano-av-r33-component-full-smoke-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512-32steps`
- Checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-smoke-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512-32steps/checkpoints/iter_0000032`
- AV eval:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-smoke-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512-32steps/eval_iter_0000032_v512_t512_gen8_report.json`
- Round-trip eval:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-smoke-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512-32steps/roundtrip_iter_0000032_v64_t64_report.json`
- AV real NLL validation/test: `1.051308 / 1.049516`, beating shuffled,
  zero, mean, and no-injection controls by large gaps.
- Round-trip gate passed with closed/usable parse fractions `1.0 / 1.0`;
  validation/test AV-real NMSE `0.000140105 / 0.000135508`; AV-real beat all
  in-run controls.

Full AV hero:

- Queue:
  `configs/nano_av/hpo/r33_component_full_hero_queue.yaml`
- Run id:
  `nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512`
- Status: complete.
- Checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291`
- Corrected AV eval:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/eval_iter_0001291_v512_t512_gen8_report.json`
- Round-trip gate:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/roundtrip_iter_0001291_v256_t256_report.json`
- Validation/test real NLL:
  `0.798672 / 0.819993`.
- Validation/test AV-real round-trip NMSE:
  `0.000109680 / 0.000121664`.
- Gate passed with `baseline_required=false`, closed/usable parse fractions
  `1.0 / 1.0`, and all in-run controls beaten.

## Selected Family-Clean Online-RL Internal Hero

The selected family-clean internal-hero checkpoint pair was produced by an
approximately 43-hour online-RL run initialized from the qualified clean R33
AV+AR SFT checkpoints. It completed 342 optimizer updates, each using 24
prompts x 8 rollouts = 192 responses, for approximately `65,664` generated
responses. Training used four actor/AV GPUs, three online critic/AR GPUs, and
one SGLang rollout GPU. The later exact-matched 384-token evaluation validated
those RL-trained weights; it did not create another checkpoint.

- Actor/AV:
  `/workspace/interp/outputs/nano30b-nla-pilot/publication/r33_online_rl/internal_hero/a3e5_u342/actor/iter_0000342`
- Critic/AR:
  `/workspace/interp/outputs/nano30b-nla-pilot/publication/r33_online_rl/internal_hero/a3e5_u342/critic/iter_0000342`
- Evidence:
  `/workspace/interp/outputs/nano30b-nla-pilot/publication/r33_online_rl/internal_hero/a3e5_u342/eval384_matched_v122`
- Boundary and protocol: 122 held-out validation rows, 122 independent content
  families, identical SFT/RL generation with max 384 new tokens, protocol
  SHA-256 `fcc431ec4450adb8817cd946d6c194fa2a45b53b0c6c42c8682c1e9f12f94d4d`.
- Directional NMSE: matched SFT `0.309055`, online RL `0.224386`; absolute
  reduction `0.084669`, relative reduction `27.4%`.
- Raw MSE: `9.5523 -> 7.2665`, a `23.9%` relative reduction.
- RL generations have lower AR reconstruction error than teacher text on
  `103/122` rows (`84.4%`), versus `62/122` (`50.8%`) for SFT; RL close rate
  is `121/122` (`99.18%`). This is not a semantic-quality comparison.
- RL beat shuffled, zero, mean, and no-injection controls; both gates passed;
  the report is marked confirmatory and generation-protocol compatible.

This is strong matched-validation evidence that online RL improved the R33
AV+AR round trip. It is not a final publication test result and does not
support a matched R33-over-R27 claim. The comparison changes both AV and AR, so
the `27.4%` is a pair-level gain rather than an actor-only effect. Before a
public RL claim, synchronize and hash the evidence, inspect the one non-closing
generation, report the complete metric/interval set, run the four-way SFT/RL
AV x AR decomposition and functional reinjection delta, replicate with a
second RL seed, and evaluate once on a new external teacher-backed boundary.

## Layer Decision And Historical Evidence

R33 remains the selected layer for clean SFT and future confirmatory RL. R27 is
the mature fallback and is needed only for an explicit cross-layer superiority
claim. R34 is not the immediate target because its AR trajectory is weaker even
though its AV 20k probe was marginally better.

Key evidence:

| Candidate | Evidence |
|---|---|
| R33 AR 20k | validation/test teacher NMSE `0.381983 / 0.388301`; source_raw `0.071066 / 0.076216` |
| R34 AR 20k | validation/test teacher NMSE `0.490728 / 0.501399`; weaker than R33 |
| R27 tuned AR | mature fallback around `0.441 / 0.437` teacher NMSE |
| R33 AV 20k | corrected real NLL `1.040335 / 1.015130` |
| R34 AV 20k | corrected real NLL `1.037261 / 1.013677`; marginal AV win, not enough to offset AR weakness |
| Historical R33 RL hero | old gate reported AV-real NMSE `0.000087528 / 0.000091176` and `30.97% / 32.34%`; effect later invalidated by mixed generation protocols |

## Historical R33 Artifacts

Pre-fix R33 hero-size dataset, now reclassified as contaminated scouting
provenance:

- Root:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_fullscan275396`
- Rows: `275,396`
- `d_model`: `2688`
- Old AR/AV verifiers passed only doc-id overlap checks. The 2026-06-10 audit
  found duplicate source content under distinct doc IDs, so this is not a clean
  heldout dataset.
- R33 critic init:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-critic-init/nano-ar-r33-critic-init`

Clean strict content-dedup target:

- RunAI root:
  `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_dedup_content_fullscan`
- Source table:
  `/workspace/interp/outputs/nano30b-nla-pilot/ar-r27-r30-fullscan-20260528T234403Z/R_27/ar_sft.parquet`
- Source rows/docs: `275,396 / 27,647`
- Kept rows/docs after verifier-equivalent token/text content dedup:
  `56,351 / 5,657`
- Empty explanations: `0`; `d_model=2688`; nonfinite activations: `0`.
- AR/AV verifier JSONs passed with content-hash cross-split overlap `0` for
  both `80/10/10` and `90/5/5`.
- This is a clean leak-free smoke/gate dataset, not the original 275k hero-size
  dataset. Future hero-scale recovery should preserve duplicate-content
  components within one split instead of dropping all duplicate rows.

R33 AR full275k pre-fix scouting result:

- Run id:
  `nano-ar-r33-full275k-lr5e5-cosine-warmup25-gb192-mb96`
- Bounded `512/512` eval:
  validation/test teacher NMSE `0.277565 / 0.276665`
- Source_raw NMSE:
  validation/test `0.096948 / 0.091568`
- Status correction: trained before the packed-boundary contamination fix and
  before the LR-schedule canary path. Its checkpoint tree was removed from
  RunAI on `2026-06-10T23:50:13Z`; logs/eval reports were preserved in the
  RunAI evidence archive.

This is a strong AR scouting signal, not a clean AR hero checkpoint and not the
final NLA milestone.

## Next Gates

1. Keep the qualified clean SFT pair as the public baseline and the retained
   `a3e5_u342` pair as the selected internal online-RL candidate.
2. Synchronize and hash the matched 384-token reports, generated records, and
   execution log; inspect the one non-closing RL sample.
3. Record the full metric and inference payload, including centered R2, norm
   ratio, cosine, family-clustered interval, sign-flip result, and paired wins.
4. Run the four SFT/RL AV x AR combinations, including the independent AR,
   and compare SFT/RL functional reinjection effects on downstream logits and
   loss. Then replicate the training result with a second online-RL seed.
5. Freeze all choices and perform one evaluation on a new external
   teacher-backed family boundary. Add a row-matched R27 comparison only if
   making a public R33-over-R27 claim.

## Historical R27 Round-Trip Baseline Status

R27 round-trip evaluator/HPO plumbing now runs on RunAI and has a clean
full-control `64/64` baseline:

- Report:
  `/workspace/interp/outputs/nano30b-nla-pilot/roundtrip/r27_baseline/r27_roundtrip_v64_t64_full_controls_prefix256_report.json`
- Generated JSONL:
  `/workspace/interp/outputs/nano30b-nla-pilot/roundtrip/r27_baseline/r27_roundtrip_v64_t64_full_controls_prefix256_generated.jsonl`
- Gate: `passed=true` with `control_margin=5e-5` and
  `min_control_win_fraction=0.9`.
- Validation teacher/AV-real NMSE:
  `0.000156636 / 0.000174863`.
- Test teacher/AV-real NMSE:
  `0.000143537 / 0.000173753`.
- AV-real beats all generated-text controls with rowwise win fraction
  `>=0.96875` on validation and `1.0` on test.

This R27 result remains historical baseline evidence. The R33 RL hero was
promoted against the exact clean R33 SFT baseline, not against a newly trained
row-matched R27 checkpoint.

## Historical Clean R33 Work

- Clean R33 AR packed smoke `gb192/mb96` failed at step 0 on
  `2026-06-11T11:00:34Z`: reward-path and training-path MSE diverged by
  `17.9%` under packed real rollout data. This confirms packed AR training is
  not clean for Nano/Nemotron-H today.
- First clean fallback AR smoke ran on RunAI:
  `nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb1-2gpu`.
  It uses `micro_batch_size=1` across two GPUs to avoid packed samples.
- First observed step for the mb1 fallback:
  `train/loss=1.2053946`, `train/fve_nrm=-1.1587539`, LR `1e-6`,
  step time about `387.6s`. Full 96-step smoke ETA is roughly `10` hours.
- Later health check at `2026-06-11T12:43:52Z`: step 17 completed with
  `train/loss=0.7070697`, `train/fve_nrm=-0.2662987`, LR `1.8e-5`. Both H200
  actor processes were alive, using about
  `96.7 GiB` each, and `/workspace/interp` had about `767G` free.
- Final correction: this checkpoint is diagnostic only. The run failed the
  LR-decay canary with a flat final LR (`2e-05 >= 1.8e-05`), and its
  heavyweight checkpoint tree has been removed from RunAI/S3 while logs and
  the diagnostic eval report were preserved.
- Current clean AR rerun is training on RunAI:
  `nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb1-2gpu-lrfix`.
  It uses the same mb1/two-GPU shape after the live Miles FSDP actor patch that
  preserves fresh SFT cosine schedules. Live scheduler preflight on
  `2026-06-12T16:06Z` logged on both ranks:
  `lr=2e-05, decay=cosine, min_lr=2e-06`.
- First completed step for the `lrfix` rerun at `2026-06-12T16:13:38Z`:
  `train/loss=1.2055051`, `train/fve_nrm=-1.1589518`, LR `1e-6`,
  actor-train time `406.3s`, and step time `409.9s`. This is expected warmup
  behavior, not the old flat-LR failure. Queue status is `training`; final
  checkpoint/eval/canary are still pending.
- The first clean R33 AV smoke is prepared but still blocked in
  `configs/nano_av/hpo/r33_dedup_clean_queue.yaml`. It is wired as an
  `av_roundtrip` trial against the expected mb1 AR checkpoint and the R27
  `256/256` full-controls round-trip baseline. Do not unblock it until the AR
  checkpoint, bounded eval, and R27 `256/256` report exist.
- `scripts/nano_queue_gate.py` is now the checked handoff utility for this
  dependency. A RunAI dry run correctly refused to unblock the AV smoke while
  AR was still `training`, the expected checkpoint was absent, and
  `eval_report` was not yet set.
- `scripts/eval_nano_av_ar_roundtrip_gate.py` now validates generated JSONL
  coverage before AR scoring. It rejects missing rows, duplicate/extra rows,
  and missing requested controls, preventing a partial streamed generation file
  from being scored as a valid gate report.
- The round-trip evaluator and AV queue now support `--resume-generated`.
  Existing complete row/control generations are reused and incomplete rows are
  regenerated. The staged first clean R33 AV smoke opts into this via
  `roundtrip.resume_generated: true`.
- The standalone config renderer `scripts/nano_roundtrip_eval_config.py` also
  passes through `eval.resume_generated`, so config-driven R27/R33 round-trip
  runs can use the same resumable generation path.
- The checked-in R27 full-control baseline configs
  `configs/nano_roundtrip/r27_baseline_64_full_controls_prefix256.yaml` and
  `configs/nano_roundtrip/r27_baseline_256_full_controls_prefix256.yaml` now
  enable `resume_generated: true`.
- The R27 `256/256` config renders the cached two-worker AV generation engine
  with worker devices `0,1`, streaming, resume, all five controls, and
  `max_new_tokens=256`. RunAI preflight verified the R27 AV/AR checkpoints and
  train/validation/test parquets exist; the `256/256` generated JSONL/report
  are still absent and will be created by the sequencer when GPUs are free.
- `configs/nano_roundtrip/r27_baseline_queue.yaml` now records the verified
  `64/64` full-control R27 baseline as `complete` and leaves the larger
  `256/256` full-control baseline as `pending` for the next GPU-safe window.
- The R27 round-trip queue has a launch guard that blocks pending round-trip
  evals while Nano train/eval/conversion processes are active. A RunAI
  `run-once` check returned `blocked_active_process` while the clean R33 AR
  smoke was training, leaving the `256/256` item pending.
- The launch guard now uses stricter process regexes for actual Python script
  invocations, avoiding false positives from sync filenames such as
  `/tmp/test_nano_av_probe_queue.py.b64`.
- A remote sequential driver is staged at
  `/workspace/interp/outputs/nano30b-nla-pilot/queue_drivers/r33_roundtrip_av_sequence_20260611T1218Z.sh`
  with log
  `/workspace/interp/outputs/nano30b-nla-pilot/queue_drivers/r33_roundtrip_av_sequence_20260611T1218Z.log`.
  It waits for active Nano train/eval processes to clear, runs the R27
  `256/256` baseline queue item, then unblocks and launches the first R33 AV
  smoke only if the AR checkpoint/eval are present and the R27 `256/256`
  report has `gate.passed=true`.
  The driver was restarted with the stricter process guard at
  `2026-06-11T12:50:02Z`, then restarted again at `2026-06-11T12:57:04Z`
  with a live-compatible AR evidence gate requiring `expected_checkpoint` and
  `eval_report`.
- Future `scripts/nano_ar_hpo_queue.py` completions now write
  `checkpoint_dir` as an alias for `expected_checkpoint`, but the currently
  running AR queue process was launched before that patch and is expected to
  write only `expected_checkpoint`.
- Fresh `2026-06-11T13:32:37Z` status: the clean AR smoke was still training at
  step `27/96` with loss `0.604653` and `fve_nrm=-0.082879`; the final
  checkpoint, `_winrates_report.json` eval, and R27 `256/256` round-trip report
  were all still absent. Focused local and RunAI CPU-only regression shards for
  the modular generation/round-trip/queue/gate path both passed with
  `55 passed`.
- Follow-up `2026-06-11T13:34:17Z`: the AR smoke advanced to step `28/96`
  with loss `0.586248` and `fve_nrm=-0.049918`. A dry-run of
  `scripts/nano_queue_gate.py` still refused to unblock the first R33 AV smoke
  because the AR checkpoint/eval and R27 `256/256` report are not present yet.
  `scripts/nano_roundtrip_queue.py status` confirmed R27 `64/64` complete and
  R27 `256/256` pending, with no running/scoring round-trip item.
- Follow-up `2026-06-11T16:44Z`: the last successful RunAI status check showed
  the clean R33 AR mb1 smoke still training at step `70/96`, loss `0.427511`,
  `fve_nrm=0.234366`; final AR checkpoint/eval and the R27 `256/256`
  round-trip report were still absent. The RunAI venv focused
  generation/round-trip/queue shard passed with `44 passed`. An ignored
  repo-local `.venv` now provides Mac-side focused verification, and the same
  local shard passed with `44 passed`. The local RunAI CLI token then expired,
  so live checks require `runai login` from the Mac before continuing. No new
  training/eval work was launched while auth was expired.
- Follow-up `2026-06-12T15:26Z`: RunAI access was restored. No active Nano
  train/eval/conversion process was running; both H200s were idle. The R27
  `256/256` full-control round-trip baseline is now complete and `gate.passed`
  is true, with validation/test AV-real normalized MSE
  `0.000180003 / 0.000175571`, teacher normalized MSE
  `0.000157706 / 0.000155285`, closed/usable parse fractions `1.0`, and
  AV-real beating all controls. The clean R33 AR mb1 smoke saved
  `checkpoints/iter_0000096` and reached final train loss/FVE
  `0.387694 / 0.305674`, but the queue marked it `failed` before eval because
  the LR-decay canary observed a flat final LR (`2e-05 >= 1.8e-05`). The AR
  bounded eval report is absent, so the R33 AV smoke remains blocked. Next
  action should be either a clean AR rerun with verified decay or an explicitly
  labeled diagnostic eval of this checkpoint that must not unblock AV by
  itself.
- Diagnostic-only AR eval `2026-06-12T15:36Z`: the failed-LR-canary checkpoint
  above was evaluated manually without mutating the queue or unblocking AV.
  Report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-smoke/nano-ar-r33-dedup-smoke-20k-lr2e5-cosine-warmup20-gb192-mb1-2gpu/eval_iter_0000096_v512_t512_winrates_diagnostic_lrflat_report.json`.
  Validation/test teacher NMSE was `0.389577 / 0.396610`; source_raw NMSE was
  `0.064048 / 0.067254`; source_context NMSE was `0.458013 / 0.459581`.
  Teacher text beat shuffled on every row, beat blank/generic/mean on roughly
  `97-99%` of rows, and beat source_context on `75.8% / 72.9%` of
  validation/test rows. Source_raw still beat teacher almost everywhere. The
  temporary HF load report listed missing `backbone.norm_f.weight` and
  `lm_head.weight`, which should be kept in mind for this diagnostic. This is
  encouraging patched-path signal, but it is not promotion evidence because the
  underlying training run failed the LR-decay canary.
- Cleanup `2026-06-12T15:58Z`: inventory was saved locally under
  `artifacts/cleanup/20260612T1558Z_contaminated_checkpoint_cleanup/` and on
  S3 under
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/sync_exports/20260612T1558Z_contaminated_checkpoint_cleanup/`.
  Heavy contaminated model state was removed from RunAI and S3 while preserving
  lightweight reports/logs/manifests. RunAI `/workspace/interp` dropped from
  about `313G` used to `152G` used (`856G` free). The old R27 AV/AR S3
  checkpoint prefix is empty, the R27 AR HPO checkpoint archive prefix now
  contains only tiny manifest JSONs, and the old RunAI outputs archive dropped
  to about `14.8M` of non-checkpoint evidence.
- LR policy fix `2026-06-12`: the Megatron actor no longer unconditionally
  rewrites fresh SFT runs to constant LR. It refreshes stale checkpoint LR
  bounds while preserving requested cosine decay for `finetune` or
  `no_load_optim` fresh SFT; `NLA_FORCE_CONSTANT_LR=1` remains as an explicit
  stale-resume escape hatch. A new clean AR queue item/config was added:
  `r33-dedup-smoke-20k-lr2e5-warmup20-gb192-mb1-2gpu-lrfix` /
  `configs/nano_ar/hpo/r33_dedup_smoke_20k_lr2e5_cosine_warmup20_gb192_mb1_2gpu_lrfix.yaml`.
- Throughput correction `2026-06-12T17:24Z`: the clean `gb192/mb1` LR-fix
  rerun proved the LR fix at steps `0-2` (`1e-6`, `2e-6`, `3e-6`) but was
  manually stopped because its 96-step ETA was about 11 hours. Packed AR
  throughput probes at `gb192/mb32` and `gb192/mb16` were rejected by the live
  reward/train equivalence guard with about `20-22%` max deviation, so packed
  Nemotron-H AR critic training remains unsafe. The active replacement is the
  correctness-preserving `gb64/mb1/16step` smoke
  `nano-ar-r33-dedup-throughput-lr2e5-warmup4-gb64-mb1-16step`, which
  completed at `2026-06-12T17:15:21Z` from
  `configs/nano_ar/hpo/r33_dedup_throughput_queue.yaml`. Scheduler evidence
  confirmed cosine LR (`lr=2e-5`, `min_lr=2e-6`) and final step `15` decayed
  to `2e-6`. Checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-throughput/nano-ar-r33-dedup-throughput-lr2e5-warmup4-gb64-mb1-16step/checkpoints/iter_0000016`.
  Eval report:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-dedup-throughput/nano-ar-r33-dedup-throughput-lr2e5-warmup4-gb64-mb1-16step/eval_iter_0000016_v64_t64_winrates_report.json`.
  Validation/test teacher NMSE was `0.660454 / 0.654646`; source_raw NMSE was
  `0.064698 / 0.067230`. Teacher beat shuffled and blank/generic, but not mean
  or source_raw. Interpretation: the fast schedule-correct smoke completed and
  is operationally useful, but 16 steps is too shallow for quality selection.
- AR batching fix `2026-06-12T19:44Z`: the `mb>1` correctness blocker is now
  understood and fixed in the AR critic path. Root cause was two-part:
  unpatched copied Nemotron-H remote code leaked Mamba state across packed
  samples, and even after patching, Nano/Nemotron-H packed THD critic training
  was not equivalent to the padded-mask reward/eval path. The modular fix makes
  AR critic SFT use padded masked microbatches with explicit last-token value
  indexing. Fresh RunAI proof
  `nano-ar-r33-dedup-throughput-smoke-lr2e5-warmup2-gb192-mb16-4step-padded`
  passed the live guard exactly:
  `reward/train MSE ratio mean=1.0000 max|r-1|=0.0000 n=32`, completed
  4 steps at steady-state `~24-25s/step`, and its temporary 72G checkpoint was
  deleted after preserving logs. New AR configs no longer need the legacy
  `allow_packed_critic_training` flag for batching.
- Post-hero execution `2026-06-21T19:30Z`: R33 component-full SFT remains the
  selected candidate. A post-hero comparability/reward dry-run report was
  produced under
  `/workspace/interp/outputs/nano30b-nla-pilot/posthero/20260621T163000Z/`.
  R33 beat the nearest-valid existing R27 `256/256` round-trip baseline by
  `39.1%` validation and `30.7%` test AV-real NMSE, but this is not a fresh
  row-identical R27 retrain. RL preflight was corrected to the current padded
  critic path and passed with `max |reward/train MSE ratio - 1| = 0.0000`.
  A 512-row R33 `stage=rl` smoke parquet was staged at
  `/workspace/interp/outputs/nano30b-nla-pilot/posthero/20260621T163000Z/rl_smoke/rl_R33_fullscan275396_smoke512.parquet`.
  RL was not launched because the reference actor+critic+rollout topology needs
  at least 3 GPU groups and the current RunAI workspace has 2 GPUs.
  Local evidence archive:
  `artifacts/runai_sync/20260621T155000Z_r33_component_full_hero/posthero/20260621T163000Z/posthero_runai_evidence_20260621T193000Z.tgz`
  (`sha256=5e636b7e13759983abc44056b9241473c0d7dda77046c1ba6043746400556e53`).

## Operating Rules

- Keep W&B offline on RunAI unless explicitly syncing.
- Use the DCP -> temporary HF -> eval -> cleanup path for AV checkpoint evals.
- Preserve compact logs, eval reports, run specs, registry entries, and W&B
  offline logs.
- Do not preserve unnecessary optimizer/checkpoint shards after eval unless they
  are selected candidates.
- Use S3 as the hub for Mac <-> RunAI source sync; RunAI is not assumed to have
  GitHub access.
- Keep large artifacts out of Git.

## Canonical Files

| File | Role |
|---|---|
| `docs/current_state.md` | This short active-state pointer |
| `docs/runs/r33_rl_hero_20260708.md` | Canonical RL hero result, retry lineage, metrics, and evidence |
| `docs/runs/r33_gate_matrix.md` | Gate checklist for R33 promotion |
| `docs/runs/r33_ar_hpo_202606.md` | Organized R33 AR HPO and hero result |
| `docs/runs/r33_av_hpo_202606.md` | Organized R33 AV HPO state |
| `artifacts/runai_sync/20260610T234644Z/` | Local copy of RunAI lightweight evidence archive, source sync archive, and cleanup manifests |
| `artifacts/runai_rl/20260708T151400Z_r33_rl_hero_final/` | Local lightweight final RL hero reports, generated outputs, eval logs, and metadata |
| `runs/registry/experiments.yaml` | Lightweight structured run registry |
| `docs/runbooks/runai_s3_sync.md` | Source sync runbook |
| `docs/runbooks/runtime_monitoring.md` | Runtime telemetry runbook |
| `docs/incidents/2026-06-longhorn-diskpressure.md` | Storage incident record |
