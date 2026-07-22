# Nano30B R33 Supervised NLA Pair: Draft Model Card

Status: internal draft, not cleared for public redistribution.

## Model Description

This artifact is a pair of supervised Natural Language Autoencoder components
for Nano30B at residual boundary R33:

- **AV (activation verbalizer):** `h_R33 -> generated explanation text`.
- **AR (activation reconstructor/critic):** `explanation text -> h_hat_R33`.

The pair is evaluated through the complete
`h_R33 -> AV text -> AR h_hat_R33` round trip. It is not a standalone language
model and the text is a lossy proxy for activation information, not a guaranteed
faithful natural-language explanation of model reasoning.

## Frozen Checkpoints

- AR HF directory SHA-256:
  `5e792120ec1a00ebb4cf4abca50d2a6a962421ac4f45423479ae5061f4d2d760`.
- AV DCP model fingerprint:
  `dcp_model_sha256:43346232d2fc043260ee903191e20cce07801903e1e7b7956f16022eb463386a`.
- Release manifest ID:
  `r33-clean-sft-av-ar-iter1291-20260715`.

The release manifest, rather than a mutable directory name, is authoritative.

## Training Data

The supervised dataset contains `275,396` teacher-backed rows from `27,647`
documents, represented by `5,009` refined content families. The stored R33
activation dimension is `2,688`. The frozen split contains `4,504 / 250 / 255`
train/validation/test families with no cross-split family or document overlap.

This split is family-disjoint but not project-wide unexposed. A later exposure
audit found that all in-corpus families had appeared in selected-pair training
or evaluation, and the existing teacher-corpus inventory contains no external
document boundary. The test metrics are therefore exploratory.

## Evaluation Summary

Validation AV response NLL is `0.776775`, versus `1.176494-1.311727` for zero,
mean, no-injection, and shuffled controls.

| Split | Rows / families | AV-text directional MSE | Teacher directional MSE | Parse usable |
|---|---:|---:|---:|---:|
| Validation | `512 / 250` | `0.307004` | `0.304714` | `100%` |
| Exploratory test | `512 / 255` | `0.319225` | `0.302637` | `100%` |

The AV path decisively beats mean, shuffled, zero, and no-injection controls
under content-family clustered inference. Stored-snapshot functional recovery
is near the teacher-text reconstruction within uncertainty and substantially
better than nonsemantic controls.

Frozen validation generations were also scored through an independently
initialized and trained seed-`314159` AR. AV-text directional MSE is `0.310963`
versus independent teacher-text `0.308533` on `512` rows and `250` content
families. All rows are closed and usable, all five registered controls pass,
and rowwise wins are `99.80%` against mean and `100%` against the remaining
controls. This is validation-only cross-critic replication of directional
information, not an independent AV seed or a raw-magnitude result.

An exploratory subgroup audit fitted all quantile edges on validation and
applied them unchanged to test. All 16 validation/test bins across source
length, target length, target activation norm, and family frequency have
sufficient rows and families, and every family-clustered control interval stays
above zero. The weakest test bin is the lowest target-activation-norm quartile:
directional MSE `0.370077` and calibrated centered raw R2 `0.415606`.

Raw centered R2 is negative before calibration, so the primary figures
establish directional information recovery rather than exact raw activation
recovery. A validation-only teacher fit selected a single nonnegative origin
scalar of `0.560604`. Applied unchanged, it improves AV centered raw R2 from
`-0.326250` to `0.487733` on validation and from `-0.335374` to `0.478102` on
the exploratory test. Directional MSE is unchanged. This is post-hoc evidence
for a global scale mismatch, not native or exact magnitude recovery.

## Compute

The selected successful SFT trainings use H100-NVL GPUs and each record all
1,291 optimizer steps in a hashed train log:

| Run | GPUs | Wall time | GPU-hours | Mean logged GPU utilization | Peak logged GPU memory |
|---|---:|---:|---:|---:|---:|
| Primary AR | `4` | `3.8867h` | `15.5467` | `79.82%` | `61,527 MiB` |
| Primary AV | `8` | `13.4608h` | `107.6867` | `61.06%` | `49,724 MiB` |
| Independent AR | `4` | `3.8781h` | `15.5122` | `81.26%` | `61,438 MiB` |

Total selected-training compute is `138.7456` H100-NVL GPU-hours. Memory and
utilization values are rank-aggregated logger metrics, not per-device profiler
traces. The report excludes extraction, conversion, evaluation, historical
HPO/RL, and failed diagnostic compute not exactly recoverable from retained
logs. Report SHA-256 is
`7bde74be3a874d2ae305463ca8da211c069ce0bf1001802b6bdf7ab091fd7238`.

The complete internal source snapshot is not itself a public release bundle.
A narrower candidate stages redacted source/evidence files with no weights or
generated text. The exact tree passes the static security gate and its
deterministic archive reinspection matches the audited tree recorded in the
adjacent machine-readable attestation.
This closes static path/secret triage only; it does not grant legal, privacy, or
model-safety clearance.

## Intended Uses

- Mechanistic-interpretability research on activation-to-language bottlenecks.
- Controlled comparison of AV text, teacher text, and nonsemantic controls.
- Stored-snapshot counterfactual reinjection experiments.
- A supervised initialization for future research, provided later training is
  evaluated against the frozen SFT pair.

## Out-Of-Scope Uses

- Treating generated text as a verified chain of thought or causal explanation.
- Safety-critical interpretation, auditing, or automated decisions.
- Claims about fresh-forward exact activations, raw-magnitude recovery,
  R33-over-R27 superiority, or RL improvement.
- Unqualified comparison to Qwen/Gemma NLA results with different models,
  data, activation geometry, or metrics.

## Known Limitations

1. All available teacher-backed content is exposed somewhere in the project;
   there is no pristine confirmatory boundary.
2. Stored extraction replays byte-exactly, but isolated fresh forwards do not
   meet strict stored-activation identity tolerances.
3. The primary metric is `2 * (1 - cosine)` and is invariant to positive scale.
4. The magnitude correction is a validation-fitted global scalar and the test
   was already exposed to project-level analysis.
5. Only one selected AV seed is qualified. The primary and independently
   initialized seed-`314159` AR paths both pass, including frozen validation
   cross-critic scoring, but this does not measure AV training-seed variance.
6. Semantic factuality and usefulness require a blinded human review. The
   frozen 100-example review packet and automatic structural screen are
   available, but automatic checks are not semantic judgments.
7. Generated explanations can contain plausible but unsupported details and
   must not be presented as ground-truth model cognition.

## Reproducibility And Evidence

- Run summary: `docs/runs/r33_clean_sft_av_ar_20260715.md`.
- Measurement definitions: `docs/methods/measurement_contract.md`.
- Release config:
  `configs/nano_roundtrip/publication/r33_clean_sft_checkpoint_pair_release.yaml`.
- Local lightweight evidence:
  `artifacts/runai_eval/r33-clean-sft-av-ar-qualified-20260715/`.
- Follow-up exposure, fidelity, scorer-cache, and calibration evidence:
  `artifacts/runai_eval/r33-clean-sft-publication-evidence-20260716/`.
- Independent AR and cross-critic evidence:
  `artifacts/runai_eval/r33-independent-ar-publication-evidence-20260716/`.
- Subgroup report: `roundtrip_subgroup_audit_report.json` in that directory.
- Frozen qualitative packet and structural screen:
  `teacher_grounded_qualitative_panel.json` and
  `teacher_grounded_qualitative_auto_review.json` in that directory.
- Redacted no-weights candidate archive:
  `artifacts/publication/r33-clean-sft-public-release-candidate-20260716.tgz`;
  use the adjacent attestation for its exact SHA-256 and audited tree hash.
- Human-readable release-candidate attestation:
  `docs/releases/r33_clean_sft_release_candidate_attestation.md`.
- Compact primary-AR, primary-AV, and independent-AR metric curves:
  `training_metric_curves.json`, SHA-256
  `7d9c22b989c594e546ec08648d0319c37caad69ad502d2badb635d41706c42a6`.

Every public metric must cite the report and verifier hash used to compute it.
W&B curves and the exported text-log curves are supporting telemetry; JSON
verifier reports are authoritative.

## License And Distribution

No public license is assigned to this repository or checkpoint pair yet. The
local Nano30B model card identifies the NVIDIA Open Model License, and current
official terms permit derivative-model distribution subject to their
conditions. FineWeb is ODC-By-1.0 and also subject to Common Crawl terms.
Teacher explanations were generated through an NVIDIA inference API, but the
exact account/subscription terms have not been established; API Trial Terms,
if governing, restrict production use and distribution of Generated Content.
Upstream NLA code is Apache-2.0. Public distribution remains blocked pending
owner/legal confirmation, repository-license selection, required notices, and
the terms review in `docs/releases/r33_clean_sft_license_provenance.md`.
