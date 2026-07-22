# Nano30B NLA Pilot

<!-- R33-HERO-BASELINE-PROTOCOL-INVALIDATED -->

> [!CAUTION]
> Publication status (`2026-07-22`): the family-clean R33 supervised AV+AR
> checkpoint pair is qualified for directional activation reconstruction and
> stored-snapshot functional recovery. Validation/test AV-to-AR directional
> MSE is `0.307004 / 0.319225`, near teacher text at
> `0.304714 / 0.302637`, with `100%` parse usability and decisive
> family-clustered control separation. A validation-fitted scalar improves
> exploratory test centered raw R2 from `-0.335374` to `0.478102`, but this is
> post-hoc calibration rather than native raw-scale recovery. The selected-pair
> exposure audit found no unexposed in-corpus family. Frozen validation text
> also passes an independent seed-`314159` AR gate at directional MSE
> `0.310963`, versus independent teacher reconstruction at `0.308533`, with all
> controls passing. The selected online-RL actor and critic were subsequently
> trained for 342 joint updates (approximately 65,664 generated responses).
> On a matched 122-family validation protocol with a 384-token generation
> budget, round-trip directional MSE improved from `0.309055` for clean SFT to
> `0.224386` for RL: `0.084669` absolute and `27.4%` relative. The RL result
> beats every required activation control and has usable text on `122/122`
> rows. It is a joint AV+AR validation result, not actor attribution or a
> sealed test-set result. This does not establish exact fresh-forward replay,
> external generalization, or R33-over-R27 superiority.
> The historical July 8 `30.97% / 32.34%` RL comparison remains invalidated.
> See `docs/runs/r33_clean_sft_av_ar_20260715.md` and
> `docs/runs/r33_family_clean_internal_rl_hero_20260719.md`.

This project tracks the Nano30B Natural Language Autoencoder pilot.

## Interactive Dashboard

The checkpoint-free NLA Observatory is published from this repository with
GitHub Pages:

<https://rishabhg97.github.io/natural-language-autoencoder/>

Its source is isolated under [`dashboard/nla-observatory`](dashboard/nla-observatory).
The committed `public/data` directory contains the compact, verified static
evidence needed by CHANNEL, TRACE, BENCH, and AUDIT. It contains no model
weights and performs no live inference.

Run it locally:

```bash
cd dashboard/nla-observatory
npm ci
npm run data:verify:published
npm test
npm run dev
```

GitHub Pages deployment is defined in
[`dashboard-pages`](.github/workflows/pages.yml). In the repository settings,
select **GitHub Actions** as the Pages source after the first push.

## Active Docs

Read in this order:

| File | Purpose |
|---|---|
| [docs/current_state.md](docs/current_state.md) | Short canonical current state, active R33 milestone, next gates |
| [docs/methods/measurement_contract.md](docs/methods/measurement_contract.md) | Canonical activation-fidelity, metric, inference, and critic-dependence definitions |
| [docs/runs/r33_clean_sft_av_ar_20260715.md](docs/runs/r33_clean_sft_av_ar_20260715.md) | Qualified R33 supervised AV+AR pair, exact checkpoints, evidence, and claim limits |
| [docs/runs/r33_online_joint_canary_20260717.md](docs/runs/r33_online_joint_canary_20260717.md) | First clean-lineage online AV+AR update canary, exact paired result, and non-promotion decision |
| [docs/releases/r33_clean_sft_publication_checklist.md](docs/releases/r33_clean_sft_publication_checklist.md) | Scientific, replication, legal, and distribution gates for a public release |
| [docs/releases/r33_clean_sft_model_card_draft.md](docs/releases/r33_clean_sft_model_card_draft.md) | Claim-scoped draft model card for the selected checkpoint pair |
| [docs/releases/r33_clean_sft_license_provenance.md](docs/releases/r33_clean_sft_license_provenance.md) | Base-model, source-data, teacher-output, and redistribution terms inventory |
| [docs/releases/r33_clean_sft_human_review.md](docs/releases/r33_clean_sft_human_review.md) | Blinded semantic-review protocol and scoring handoff |
| [docs/superpowers/plans/2026-07-08-r33-publication-correctness-remediation.md](docs/superpowers/plans/2026-07-08-r33-publication-correctness-remediation.md) | Publication-correctness remediation checklist and remaining gates |
| [docs/runs/r33_publication_preregistration.md](docs/runs/r33_publication_preregistration.md) | Draft sealed-test protocol for the future confirmatory RL replication |
| [docs/runs/r33_rl_hero_20260708.md](docs/runs/r33_rl_hero_20260708.md) | Historical, publication-invalidated R33 RL hero configuration and evidence |
| [docs/runs/r33_gate_matrix.md](docs/runs/r33_gate_matrix.md) | R33 hero promotion checklist |
| [docs/runs/r33_ar_hpo_202606.md](docs/runs/r33_ar_hpo_202606.md) | Organized R33 AR HPO and full275k hero proxy result |
| [docs/runs/r33_av_hpo_202606.md](docs/runs/r33_av_hpo_202606.md) | Organized R33 AV HPO state |
| [runs/registry/experiments.yaml](runs/registry/experiments.yaml) | Lightweight structured registry for decision-changing runs |
| [docs/execution_log.md](docs/execution_log.md) | Current phase, run history, blockers, additions/subtractions |
| [docs/nano_av_job_tracker.md](docs/nano_av_job_tracker.md) | Human/agent editable queue for active and planned Nano AV/AR runs |
| [docs/nano_av_run_history.md](docs/nano_av_run_history.md) | Completed/invalid/held Nano AV/AR run history and key results |
| [docs/nano30b-nla-core-plan.md](docs/nano30b-nla-core-plan.md) | Canonical scientific and engineering plan |
| [docs/research/20260610_nano_nla_research_directions.md](docs/research/20260610_nano_nla_research_directions.md) | Research directions memo: bottleneck hypothesis, prioritized experiments, R33 decision tree |
| [docs/cluster_runbook.md](docs/cluster_runbook.md) | Exact cluster commands |
| [docs/runbooks/README.md](docs/runbooks/README.md) | Operational runbooks for sync, monitoring, and cluster work |
| [docs/issues_iter1.md](docs/issues_iter1.md) | Archived detailed rationale and discarded alternatives |
| [external/natural_language_autoencoders](external/natural_language_autoencoders) | Reference implementation for adaptation |

The tree under `external/natural_language_autoencoders` is the modified
production adaptation, not a pristine upstream checkout. It includes Nano
architecture adapters and changes to loss, injection, reward, logging, and
runtime integration. Method comparisons must disclose those divergences; see
the measurement contract.

Cluster setup is centralized in [scripts/cluster_nano_env.sh](scripts/cluster_nano_env.sh).

## Current Phase

The project has a **qualified family-clean R33 SFT AV+AR pair**.

```text
deterministic 275,396-row stored R33 snapshot
-> family-disjoint AR and corrected-packed AV SFT checkpoints
-> hash-bound 512-row component validation
-> protocol-matched AV-generated-text -> AR validation gate
-> one-time family-disjoint test and stored-snapshot functional gate
-> qualified checkpoint-pair manifest and S3 preservation
```

The component AV real NLL is `0.776775`, versus `1.176494-1.311727`
for activation controls. The AV-to-AR directional MSE is
`0.307004 / 0.319225` on validation/test, with all controls beaten and all
generated rows usable. Stored-snapshot functional metrics are statistically
indistinguishable from teacher reconstruction at the content-family level and
decisively better than mean, zero, and shuffled controls.

The checkpoint pair is suitable for a carefully scoped public SFT result, and
its AV signal now replicates through an independently initialized and trained
AR seed on frozen validation text. A stronger publication claim still needs a
pristine external test boundary and blinded semantic review; a second AV seed
is strongly recommended for an architecture-level claim. Exact fresh-forward
identity remains false under the audited runtime, and a row-matched R27
comparison is required only if layer superiority is claimed.

The 342-update online-RL pair materially improves the matched validation round
trip: directional MSE falls from `0.309055` to `0.224386` (`27.4%` relative),
and RL-generated descriptions beat teacher text on `103/122` rows. This is
strong validation evidence for a joint actor+critic improvement. It is not a
sealed publication test, a component-level attribution, or a matched R27
comparison.

Weights, datasets, activation arrays, optimizer states, W&B directories, and
other heavy runtime artifacts are intentionally not committed to this source
repository. Release status and artifact provenance remain documented under
`docs/releases` and `docs/publication`.

## Core Pilot Sequence

```text
pin environment
-> prove Nano adapter and extraction identity
-> test R_34 and R_27
-> prove Qwen-faithful AV warm-start h -> z
-> improve AV decoding beyond lm_head-only smoke
-> scale R33 AV and AR with bounded controls
-> run held-out AV-generated round trip with AR
-> run guarded RL with generated-text round-trip promotion gates
-> run released Qwen/Gemma NLA checkpoint QC as implementation regression
```

## Repository Organization

| Path | Role |
|---|---|
| [docs/runs](docs/runs) | Curated run summaries and gate decisions |
| [docs/runbooks](docs/runbooks) | Operational procedures |
| [docs/architecture](docs/architecture) | Stable design notes and architecture pointers |
| [docs/incidents](docs/incidents) | Operational incidents affecting experiments |
| [configs](configs) | Source-controlled experiment plans |
| [scripts](scripts) | Dataset, training, eval, queue, and ops scripts |
| [external/natural_language_autoencoders](external/natural_language_autoencoders) | Modified NLA training/runtime implementation |
| [observatory](observatory) | Offline evidence and visualization preparation code |
| [dashboard/nla-observatory](dashboard/nla-observatory) | Static React dashboard and committed public evidence bundle |
| [tests](tests) | CPU unit, regression, verifier, and integration tests |
| [runs/registry](runs/registry) | Lightweight tracked run registry; heavy `runs/` artifacts remain ignored |
