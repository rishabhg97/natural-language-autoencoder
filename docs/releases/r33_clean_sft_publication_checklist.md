# R33 Clean SFT Publication Checklist

Status date: `2026-07-16`

This checklist is the release boundary for the selected family-clean R33
supervised AV+AR checkpoint pair. It does not revive the invalidated July 8 RL
claim and it does not turn an exploratory test into a pristine confirmatory
test.

## P0 Scientific Gates

| Gate | Status | Required evidence or action |
|---|---|---|
| Exact checkpoint identity | complete | Release manifest binds AR SHA-256 `5e792120...d760` and AV DCP model fingerprint `43346232...386a` |
| Component AV conditioning | complete | 512-row validation report and verifier; real NLL beats shuffled, zero, mean, and no-injection |
| Component AR reconstruction | complete | 512-row validation report and verifier; teacher direction beats nonsemantic controls |
| Generated-text AV->AR round trip | complete, exploratory boundary | 512 validation and 512 test rows, family-clustered controls, 100% usable generation |
| Functional recovery | complete, stored-snapshot scope | Validation/test counterfactual reinjection reports and verifiers |
| Project-wide exposure audit | complete, negative result | All `5,009` in-corpus content families have selected-pair train/eval exposure; no in-corpus confirmatory family remains |
| External teacher boundary inventory | complete, negative result | Existing usable teacher corpora contain no numeric document IDs outside the exposed `0..38161` range |
| Fresh-forward identity | failed exact claim | Fresh forwards are repeatable and highly aligned with stored rows, but the 64-row strict identity audit fails; retain stored-snapshot wording |
| Raw activation magnitude | complete, exploratory calibration | Validation teacher fit selected one origin scalar, `0.560604`; unchanged application raises exploratory test AV centered raw R2 from `-0.335374` to `0.478102`, but native uncalibrated scale remains unsupported |
| Independent critic seed | complete on validation | Seed-`314159` critic init passed all 16 independence checks; clean AR component evaluation passes, and frozen AV-text to independent-AR directional reconstruction is `0.310963` versus teacher `0.308533` on 512 rows. All five controls and the hash-bound verifier pass |
| Semantic quality | human review pending | Corrected source-grounded 50-validation/50-test panel and two blinded reviewer packets exist; structural screen has zero flags. Complete blinded human ratings and report inter-rater agreement |
| External replication | blocked by data | Obtain or generate a genuinely unexposed, legally usable teacher-backed boundary under a preregistered protocol |

The checkpoint pair may be described publicly before external replication only
as an exploratory, family-disjoint **within-corpus stored-snapshot** result.
Words such as "confirmatory", "fresh-forward exact", and "project-wide unseen"
remain prohibited.

## P0 Release And Legal Gates

| Gate | Status | Required evidence or action |
|---|---|---|
| Repository license | blocked | The repository has no top-level license. Obtain owner/legal approval and add the selected license before public source release |
| Base-model redistribution | legal approval pending | Local model card identifies the NVIDIA Open Model License, whose current text permits derivative-model distribution subject to its conditions. Confirm the exact acquired-model terms and release obligations with owner/legal before publishing weights |
| Dataset and teacher-text terms | legal approval pending | FineWeb is ODC-By-1.0 and subject to Common Crawl terms. Teacher text came from an NVIDIA inference API; the exact subscription/agreement must be identified because API Trial Terms restrict production and Generated Content distribution |
| Third-party notices | partial | Upstream NLA code is Apache-2.0 under `external/natural_language_autoencoders/LICENSE`; add notices for every redistributed dependency |
| Privacy and memorization review | automatic gate complete; human pending | All 1,024 generated explanations have zero configured sensitive-pattern findings and zero source-copy failures. Fourteen phone-like strings occur only in source excerpts; adjudicate or redact before releasing examples |
| Security review | candidate bundle complete | The narrower redacted candidate has zero failed findings, forbidden paths, symlinks, binaries, or oversized files. Its deterministic archive reinspection exactly matches the audited tree recorded in the adjacent attestation. Rerun only if the legal-cleared final bytes change |

No checkpoint or source archive should be uploaded to a public registry until
these legal gates are signed off. Internal S3 preservation is not public
redistribution approval.

## P1 Robustness And Scope

| Item | Need |
|---|---|
| Row-matched R27 retrain | Required only for a public R33-over-R27 or layer-optimality claim |
| Second AV training seed | Strongly recommended for an architecture/performance paper; not required to release one checkpoint as an artifact |
| Subgroup audit | Complete for available registered fields: 16 validation/test bins across source length, target length, activation norm, and family frequency; every bin has sufficient rows/families and every control CI remains positive. Source-type labels are unavailable in the frozen cache |
| Calibration sensitivity | Report identity, origin-scalar, and train-mean-scalar candidates; select on validation only |
| Numerical replay | Complete for this follow-up: cache/report hashes are bound; scorer replay showed small runtime-level metric drift and is not claimed bit-exact |
| Released Qwen/Gemma control | Useful implementation regression, but not a substitute for Nano external replication |
| Compute accounting | selected successful runs complete; broader project compute bounded | Hash-bound report covers all 1,291 steps for primary AR, primary AV, and independent AR: `138.7456` H100-NVL GPU-hours. It lists explicit exclusions for extraction, eval, historical HPO/RL, and unrecoverable failed diagnostics |

## Reproducibility Bundle

The public bundle should contain only lightweight, permitted artifacts:

1. Exact source commit/archive and SHA-256.
2. Resolved AV and AR configs, environment lock, runtime fingerprints, and
   launch commands.
3. Dataset manifests, family assignments, split hashes, verifier JSON, and
   provenance reports, but no restricted source text.
4. W&B offline histories or exported metric tables with run IDs.
5. Generated evaluation text only after data/privacy review.
6. Raw rowwise metrics, family IDs, bootstrap seeds, and analysis scripts.
7. Checkpoint fingerprints and model card; include weights only after license
   approval.
8. A machine-readable limitations/claim-boundary file.

The current no-weights candidate and its adjacent machine-readable attestation
are under `artifacts/publication/` and
`artifacts/runai_eval/r33-clean-sft-publication-evidence-20260716/`.
The attestation records the archive byte count, archive SHA-256, security-report
SHA-256, file count, and independently recomputed archive/tree hashes.
Compact exported curves cover all 1,291 updates for each of primary AR,
primary AV, and independent AR; their JSON SHA-256 is
`7d9c22b989c594e546ec08648d0319c37caad69ad502d2badb635d41706c42a6`.
The candidate remains explicitly marked `legal_clearance_granted=false` and
`weights_included=false`.

Exact values and the current scientific/release decision are indexed in
`docs/releases/r33_clean_sft_release_candidate_attestation.md`. That page is
intentionally excluded from the immutable candidate archive to avoid embedding
an archive hash inside the archive it identifies.

Large model shards, optimizer states, temporary HF conversions, raw activation
Parquets, credentials, and internal-only paths are excluded from the public
evidence archive.

Internal evidence paths:

- Primary follow-up:
  `artifacts/runai_eval/r33-clean-sft-publication-evidence-20260716/`.
- Independent AR replication:
  `artifacts/runai_eval/r33-independent-ar-publication-evidence-20260716/`.
- Blinded packets:
  `blinded_human_review/review_packet_reviewer_{1,2}.json`; do not give the
  answer key to reviewers.
- Terms inventory:
  `docs/releases/r33_clean_sft_license_provenance.md`.

## Claim Decision

Current defensible wording:

> On a deterministic stored R33 activation snapshot with content-family
> disjoint train/validation/test splits, the selected Nano30B AV-generated text
> carries row-specific activation information that a paired AR reconstructs
> directionally, with stored-snapshot functional recovery near the teacher-text
> path and far better than activation and semantic controls.

Required qualifiers: exploratory within-corpus boundary, stored-snapshot
semantics, direction-primary metric, one selected checkpoint pair, no valid RL
gain, no R33-over-R27 claim, and no native exact raw-magnitude claim. The
validation-fitted one-scalar result may be reported separately as post-hoc
calibration evidence.
