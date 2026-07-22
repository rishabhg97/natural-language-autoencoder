# Run Documentation

This directory holds compact, decision-oriented run summaries. It should answer:

- what was run,
- what evidence it produced,
- what decision changed,
- where the durable artifacts live,
- what gate remains blocked or passed.

Long chronological notes stay in `docs/experiment_logbook.md`. Live queue state
stays in `docs/nano_av_job_tracker.md`. This directory is the curated layer for
research decisions.

## R33 Files

| File | Purpose |
|---|---|
| `r33_clean_sft_av_ar_20260715.md` | Qualified family-clean R33 SFT AV+AR checkpoints, independent-AR follow-up, release-candidate evidence, and claim boundaries |
| `r33_family_clean_internal_rl_hero_20260719.md` | Selected family-clean online-RL internal hero, matched 384-token validation result, and publication boundary |
| `r33_sft_confirmatory_preregistration.md` | Frozen endpoints and one-shot policy for a future pristine SFT confirmatory boundary |
| `r33_rl_hero_20260708.md` | Archived, protocol-invalidated R33 RL hero recipe, retry lineage, and evidence locations |
| `r33_gate_matrix.md` | Promotion checklist for the R33 hero milestone |
| `r33_ar_hpo_202606.md` | R33 AR 20k, 100k, and full275k hero selection |
| `r33_av_hpo_202606.md` | R33 AV smoke/HPO state and next choices |

Archived historical checkpoint, not a current release candidate:

`/workspace/interp/outputs/nano30b-nla-pilot/rl_hero/r33_corrected_k3_hero_lr1e5_update342_resume228_retry3/actor/iter_0000342`

There is now a selected **internal** online-RL checkpoint pair. It passes a
protocol- and row-matched 122-family validation gate, improving directional
NMSE from `0.309055` to `0.224386` (`27.4%`) against clean SFT. It is not yet
an externally supportable final checkpoint: independent-critic and second-seed
replication plus a new external teacher-backed test boundary remain open.

## Update Rule

After every completed eval, add or update a row in the relevant run summary and
in `runs/registry/experiments.yaml`. Do not rely on queue YAML status alone,
because remote queues can be ahead of checked-in local files.
