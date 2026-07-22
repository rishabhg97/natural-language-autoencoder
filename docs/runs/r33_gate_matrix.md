# R33 Hero Gate Matrix

<!-- R33-HERO-BASELINE-PROTOCOL-INVALIDATED -->

> [!CAUTION]
> Publication status (`2026-07-16`): the family-clean R33 SFT AV+AR pair is
> qualified for directional reconstruction and stored-snapshot functional
> recovery. Its signal also passes a validation-only gate through an
> independently initialized and trained seed-`314159` AR. The archived
> `30.97% / 32.34%` RL comparison remains invalidated.
> See `docs/runs/r33_clean_sft_av_ar_20260715.md`.

Last organized: `2026-07-16`.

The active result is the clean deterministic SFT lineage. The July 8
component-full SFT/RL matrix below remains historical debugging evidence only.

## Clean Confirmatory Matrix

| Gate | Status | Evidence | Next action |
|---|---|---|---|
| Deterministic R33 snapshot extraction | replay passed; fresh-forward fidelity failed | `275,396` rows, exact eight-shard replay; 32-row live check failed with `2.562%` mean and `7.728%` max relative L2 drift | Treat stored vectors as deterministic snapshot targets; do not claim exact fresh-forward fidelity |
| Refined family boundary | passed for clean SFT qualification | `5,009` families; train/validation/test use `4,504 / 250 / 255` families with zero train-family or cross-split overlap | Disclose that test is not guaranteed untouched by all historical runs; use a pristine external split for a stronger publication test claim |
| Clean primary AR SFT | passed directionally | 1,291 updates; teacher directional MSE `0.281703`, cosine `0.859148`; raw MSE `8.537785`, centered raw R2 `-0.201696` | Preserve selected HF checkpoint; do not claim raw-magnitude recovery |
| Corrected packed AV SFT | passed | Live packed/padded gate passed; 1,291 finite updates; validation real NLL `0.776775` beats shuffled `1.311727`, zero `1.176494`, mean `1.237522`, and none `1.220974` | Preserve selected DCP checkpoint and verifier evidence |
| Clean SFT round trip | passed on validation and test | `512/512` rows; candidate directional MSE `0.307004 / 0.319225`, teacher `0.304714 / 0.302637`; `100%` usable; all family-clustered controls pass | Qualified for directional AV-to-AR reconstruction |
| Stored-snapshot functional recovery | passed on validation and test | Candidate is teacher-level within family uncertainty and decisively beats mean, zero, and shuffled controls | Keep claim scoped to stored-snapshot counterfactual reinjection, not fresh-forward identity |
| Checkpoint-pair release manifest | passed | Release `r33-clean-sft-av-ar-iter1291-20260715`; exact AV/AR fingerprints plus six passing verifiers; `qualified: true` | Use this pair as the supervised R33 NLA milestone |
| Independent critic/AR | passed on validation | Seed-`314159` initialization passed all 16 independence checks. The independently trained AR has component directional MSE `0.286169`; frozen selected-AV text reconstructs at `0.310963` versus teacher text at `0.308533` on 512 rows across 250 families, with `100%` usable generations and all five controls passed. The checkpoint is fully fingerprinted and preserved in internal S3 | Treat as cross-critic replication for this selected AV only; train a second AV seed or use an external teacher boundary before an architecture-level generalization claim |
| Selected-run compute and curves | passed | All 1,291 updates are present for primary AR, primary AV, and independent AR; `138.7456` H100-NVL GPU-hours; compact curves cover 3,873 updates | Cite explicit exclusions; do not present this as full-project compute |
| Redacted no-weights candidate | passed static gate | 496 files; zero failed findings, forbidden paths, symlinks, binaries, or oversized files; archive/tree SHA-256 `df175c5f...263b77`; five objects preserved in internal S3 | Keep `weights_included=false` and `legal_clearance_granted=false` until approval |
| Blinded semantic review | pending human action | Corrected 50-validation/50-test source-grounded panel and two blinded packets exist; automatic structural flags `0` | Obtain two independent reviews and report agreement before semantic-quality claims |
| License and teacher-output rights | blocked on approval | Terms inventory and draft NOTICE exist; exact repository license, weight redistribution approval, and teacher-API agreement remain unresolved | Owner/legal decision required before public source or weight distribution |
| Confirmatory RL | not launched | Draft preregistration exists; universal source/env/launch guards are implemented, while clean neighbor checks, kernel delta, power calculation, and final registration remain pending | Complete all upstream gates before two independent registered seeds |

## Historical Internal Matrix

Historical `NMSE` labels below refer to direction-only metrics. Some reports
also divided by `d_model=2688`; they are retained for provenance and are not
raw-magnitude reconstruction measurements.

| Gate | Status | Evidence | Next action |
|---|---|---|---|
| R33 layer selected | passed | R33 20k AR beats R34 and mature R27 tuned AR on teacher NMSE trajectory | Keep R27 as fallback baseline |
| R33 strict-dedup clean dataset | passed for smoke/gate | `56,351` rows, `d_model=2688`, nonfinite `0`, empty explanations `0`, AR/AV content-overlap `0` | Use for clean smoke and round-trip gating; design component-preserving split before any 275k hero claim |
| R33 AR packed smoke | failed as intended | `gb192/mb96` failed live reward/train equivalence with `17.9%` max MSE-ratio divergence | Treat packed AR as unsafe; use mb1 fallback |
| R33 AR clean candidate | selected | `nano-ar-r33-dedup-clean56k-lr5e5-cosine-warmup25-gb192-mb96-128step-padded`; checkpoint `iter_0000128`; bounded `512/512` teacher NMSE `0.361513 / 0.352040`; teacher beat shuffled almost perfectly | Freeze as AR side of current candidate; preserve checkpoint/eval evidence |
| R33 component-full dataset | passed | `275,396` rows under `/workspace/interp/outputs/nano30b-nla-pilot/r33_prefix_component_fullscan275396`; AR/AV verifiers pass with `d_model=2688`, nonfinite `0`, empty explanations `0`, materialized component split doc/content overlap `0` | Use as the active hero dataset |
| R33 component-full AR hero | passed | `nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96`; checkpoint `iter_0001289`; bounded `512/512` teacher NMSE `0.320616 / 0.292730`; teacher beats shuffled/blank/generic/mean controls strongly | Use as AR side for full AV round-trip gate |
| R33 AR 100k HPO | scouting passed, clean rerun required | Best pre-fix 100k run: `lr=5e-5`, `warmup=25`, `gb192/mb96`, teacher NMSE `0.300924 / 0.292944`; contaminated training risk and pre-canary LR schedule | Use only as LR prior; packed shape cannot be reused cleanly |
| R33 AR full275k proxy | scouting passed, checkpoint removed | Pre-fix `lr=5e-5`, `warmup=25`, `gb192/mb96`; bounded teacher NMSE `0.277565 / 0.276665`; checkpoint deleted after evidence sync | Reproduce clean before using as hero |
| R33 AV round-trip smoke | superseded historical smoke | Earlier dedup smoke remains useful candidate evidence: AV real NLL `1.003093 / 0.976148`; round-trip AV-real NMSE `0.000128805 / 0.000135702`; parse closed/usable `1.0 / 1.0` | Keep compact evidence only; active publication path uses deterministic family-clean data |
| R33 component-full AV smoke | passed | `nano-av-r33-component-full-smoke-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512-32steps`; checkpoint `iter_0000032`; AV real NLL `1.051308 / 1.049516`; round-trip AV-real NMSE `0.000140105 / 0.000135508`; parse closed/usable `1.0 / 1.0`; AV-real beats all controls | Gate passed; unblocked full AV hero |
| R33 component-full AV hero | passed | `nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512`; checkpoint `iter_0001291`; corrected AV eval real NLL `0.798672 / 0.819993`; `256/256` round-trip gate passed with AV-real NMSE `0.000109680 / 0.000121664`, closed/usable parse `1.0 / 1.0`, and all in-run controls beaten | Selected as the internal R33 AV+AR hero milestone; preserve checkpoint/evidence |
| R27 round-trip baseline | partial | `r27_roundtrip_v64_t64_full_controls_prefix256_report.json`; AV-real NMSE `0.000174766 / 0.000173755`; closed parse fraction `1.0 / 1.0`; all controls beaten with rowwise win fraction threshold. The checked queue records 64/64 complete and 256/256 pending; `r27_baseline_256_full_controls_prefix256.yaml` renders cached two-worker generation with streaming/resume enabled and all model/split prerequisites present | Use 64/64 only as smoke evidence; require 256/256 before the first R33 AV round-trip HPO smoke |
| R33 round-trip evaluator | ready | `scripts/eval_nano_av_ar_roundtrip_gate.py`; fresh focused shard passed locally and on RunAI with `55 passed`; generated JSONL coverage is validated before AR scoring; `--resume-generated` can reuse complete row/control generations | Run after bounded R33 AV and AR checkpoints are selected |
| R33 beats R27 full NLA | waived for internal hero | The final component-full round-trip report records `gate.passed=true` with `baseline_required=false`; R33 beats all in-run controls, but no fresh row-matched clean R27 baseline was required for this internal milestone. | Restore row-matched R27 comparison before external claims; not required before the current internal hero selection |
| R33 family-clean online-RL internal hero | passed on matched validation protocol; not sealed | Existing actor/critic `iter_0000342` pair evaluated on the same 122 validation rows/122 independent families as matched SFT, max 384 new tokens, protocol `fcc431ec...94d4d`; directional NMSE `0.309055 -> 0.224386` (`27.4%` lower), raw MSE `9.5523 -> 7.2665` (`23.9%` lower), RL generated text has lower reconstruction error than teacher on `103/122`, close rate `121/122`, all controls and both gates passed | Preserve as selected internal RL pair; sync/hash reports, inspect the one parse miss, run four-way AV/AR decomposition, functional delta, independent-critic and second-seed replication, then evaluate once on a new external boundary |
| R33 corrected-K3 RL hero | historical internal pass; publication-invalidated | `r33-corrected-k3-hero-lr1e5-update342-resume228-retry3`; selected checkpoint `iter_0000342`; old gate reported `30.97% / 32.34%`, but the SFT baseline mixed generation protocols and the stored activation lineage later failed publication identity checks | Preserve compact evidence only; do not use this effect as the clean baseline or current claim |

## Required Round-Trip Report Fields

The final gate report should include:

- R33 AV checkpoint and AR checkpoint identifiers.
- R27 baseline checkpoint identifiers.
- Validation row and independent-family counts; test counts only after the
  one-shot selection lock.
- AV generation parse/closure rate.
- AR directional MSE from generated text, plus raw MSE, centered R2, norm
  ratio, and cosine.
- Teacher-text directional metrics as a proxy, not a raw-space upper bound.
- Control comparisons: shuffled, blank/generic, mean, source_context,
  source_raw where applicable.
- Family-clustered confidence interval and rowwise win rates versus the matched
  baseline.

## Historical Component-Full Hero Pair

- AR checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001289`
- AV checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/checkpoints/iter_0001291`
- Queue:
  `configs/nano_av/hpo/r33_component_full_hero_queue.yaml`
- Current full-gate reports:
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/eval_iter_0001291_v512_t512_gen8_report.json`
  and
  `/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/roundtrip_iter_0001291_v256_t256_report.json`.

## Historical RL Hero

- Actor checkpoint:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_hero/r33_corrected_k3_hero_lr1e5_update342_resume228_retry3/actor/iter_0000342`
- Queue:
  `configs/nano_rl/r33_component_corrected_k3_hero_342_resume228_retry3_queue_8h100.yaml`
- Final report:
  `/workspace/interp/outputs/nano30b-nla-pilot/rl_hero/r33_corrected_k3_hero_lr1e5_update342_resume228_retry3/roundtrip_iter_0000342_v512_t512_report.json`
- Historical result:
  the old gate recorded `gate.passed=true`, matching row identities, and
  `30.97% / 32.34%` relative improvement. The later audit invalidated that
  effect estimate because generation protocols differed across the baseline;
  it is not a clean confirmatory result.
- Canonical summary:
  `docs/runs/r33_rl_hero_20260708.md`.

## Current Family-Clean Online-RL Internal Hero

- Actor/AV:
  `/workspace/interp/outputs/nano30b-nla-pilot/publication/r33_online_rl/internal_hero/a3e5_u342/actor/iter_0000342`
- Critic/AR:
  `/workspace/interp/outputs/nano30b-nla-pilot/publication/r33_online_rl/internal_hero/a3e5_u342/critic/iter_0000342`
- Evaluation directory:
  `/workspace/interp/outputs/nano30b-nla-pilot/publication/r33_online_rl/internal_hero/a3e5_u342/eval384_matched_v122`
- Reports: `sft_roundtrip_report.json` and `rl_roundtrip_report.json`; execution
  log: `eval384_chain.log`.
- Matched result: directional NMSE `0.309055 -> 0.224386`, an absolute
  reduction of `0.084669` and a relative reduction of `27.4%`. Raw MSE fell
  from `9.5523` to `7.2665` (`23.9%`).
- The evaluation is validation-only on 122 independent families. It supports
  the selected internal checkpoint and an online-RL improvement claim on this
  boundary, not a final publication test or a matched R27 claim.

## Preservation Status

- Local compact evidence archive:
  `artifacts/runai_sync/20260621T155000Z_r33_component_full_hero/20260621T155000Z_r33_component_full_hero_compact.tgz`
- S3 prefix:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/sync_exports/20260621T155000Z_r33_component_full_hero/`
- Cleanup manifest:
  `artifacts/runai_sync/20260621T155000Z_r33_component_full_hero/cleanup/20260621T155000Z_r33_component_full_cleanup.txt`
- Cleanup deleted the superseded component-full AV smoke checkpoint payload and
  kept the selected AR/AV hero checkpoint payloads on RunAI.
- RL hero lightweight archive:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/sync_exports/20260708T151400Z_r33_rl_hero_final/r33_rl_hero_final_lightweight_20260708T151400Z.tgz`
- RL hero archive SHA-256:
  `78cbf98d27188594c25cbf9c0d695f0b3b1754df978961585bbaa6fc178f0bc7`.

## Historical RL Systems Smoke Update

Latest RL status is documented in `docs/experiment_logbook.md` under
`rl-smoke: r33-component-full-4h200-post-leakfix-20260623`.

Completed systems smokes used:

- `2` actor H200s;
- `1` rollout/SGLang H200;
- `1` critic H200;
- frozen AR reward scoring with `NLA_FREEZE_CRITIC_TRAIN=1`.

The earlier `len256` smoke used `NLA_SKIP_ROLLOUT_WEIGHT_SYNC=1` and validated
topology/reward-train equivalence only. The later `len512` smoke used live
actor-to-SGLang sync, unified env, TokenizersBackend fallback, Mamba
`time_step_limit` normalization, reward-forward no-fast-path mode, and
`expandable_segments`; it completed two rollout/update cycles and saved
`actor/iter_0000002`.

These smokes validated systems execution and were later superseded by the
corrected-K3 probes and completed update-342 RL hero above.
