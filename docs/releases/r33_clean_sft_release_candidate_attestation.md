# R33 Clean SFT Release-Candidate Attestation

Status date: `2026-07-16`

This page is the human-readable index for the final lightweight candidate
associated with release ID `r33-clean-sft-av-ar-iter1291-20260715`. It is
maintained outside the immutable archive so it can cite that archive without a
self-referential hash.

## Scientific Result

The selected family-clean R33 supervised AV+AR pair is qualified for the
following bounded claim:

> On a deterministic stored R33 activation snapshot with content-family
> disjoint clean-SFT splits, AV-generated text contains row-specific activation
> information that the paired AR reconstructs directionally. The signal beats
> semantic and activation controls and supports stored-snapshot counterfactual
> functional recovery.

Selected-pair evidence:

| Gate | Result |
|---|---|
| AR component validation | teacher directional MSE `0.281703`, cosine `0.859148`, FVE-NRM `0.584534` |
| AV component validation | real NLL `0.776775`; shuffled `1.311727`, zero `1.176494`, mean `1.237522`, none `1.220974` |
| AV-text -> AR round trip | validation/test directional MSE `0.307004 / 0.319225`; teacher `0.304714 / 0.302637` |
| Parse health | `512/512` closed and usable on both validation and test |
| Controls | every registered family-clustered lower confidence bound positive; test rowwise wins `99.61-100%` |
| Functional recovery | passes on validation/test for stored-snapshot counterfactual reinjection |

The result does not establish exact fresh-forward identity, native
raw-magnitude recovery, project-wide unseen generalization, R33-over-R27
superiority, or RL gain.

## Independent Replication

The independently initialized seed-`314159` AR completed all 1,291 optimizer
updates. Its initialization verifier passes all 16 independence checks and its
512-row component evaluation reports teacher directional MSE `0.286169`,
cosine `0.856916`, and FVE-NRM `0.577948`.

Frozen selected-AV validation text scored through this independent AR gives
directional MSE `0.310963`, versus independent teacher-text reconstruction at
`0.308533`. All 512 rows across 250 families are closed and usable; all five
controls pass, with minimum rowwise win fraction `0.998047`. This is
validation-only cross-critic replication for one selected AV seed, not an
independent AV-seed result.

The independent HF checkpoint has 10 files totaling `38,462,226,688` bytes,
directory SHA-256
`c2eea74f5baccee97128617b05636187804c7e59aedc560d088dbf65d52f1925`,
and is preserved in internal S3. Redundant DCP model and optimizer copies were
removed only after upload verification and a manifest-first retention pass.

## Compute And Curves

All selected training logs contain exactly 1,291 optimizer steps:

| Run | GPUs | Wall time | GPU-hours |
|---|---:|---:|---:|
| Primary AR | 4 H100-NVL | `3.8867h` | `15.5467` |
| Primary AV | 8 H100-NVL | `13.4608h` | `107.6867` |
| Independent AR | 4 H100-NVL | `3.8781h` | `15.5122` |

Total selected successful training is `138.7456` H100-NVL GPU-hours. Compact
loss, LR, gradient, router, system, and performance curves cover all 3,873
optimizer steps; curve-file SHA-256 is
`7d9c22b989c594e546ec08648d0319c37caad69ad502d2badb635d41706c42a6`.
The accounting excludes extraction, conversion, evaluation, historical HPO/RL,
and failed diagnostics without exact retained timing.

## Candidate Archive

| Field | Value |
|---|---|
| Local archive | `artifacts/publication/r33-clean-sft-public-release-candidate-20260716.tgz` |
| Archive bytes | `6,859,370` |
| Archive SHA-256 | `3eb8e64ed0d9d61ed2d6b0694fbaf96b99051a63f2ce1a6c99372d93832e573a` |
| Files | `496` |
| Audited/archive tree SHA-256 | `df175c5f61cefbfc1a02451a7bd242ba69e1cb602cdd97ca4b8bd8fe9c263b77` |
| Security-report SHA-256 | `a130d8e4295a06d0372bd0d920d9dfd0c8f7649710652939b33dbc764089f096` |
| Weights included | `false` |
| Legal clearance granted | `false` |

The exact staged tree has zero failed security findings, forbidden paths,
symlinks, binary files, or oversized text files. The deterministic archive was
read back member-by-member and reproduced the audited tree hash.

Five objects are preserved under the internal S3 prefix
`publication/release-candidates/r33-clean-sft-av-ar-iter1291-20260716/`:
archive, SHA-256 sidecar, bundle manifest, security report, and archive
attestation. The remote listing reports the expected archive byte count.

## Verification

- Full local suite: `749 passed`.
- Publication-bundle focused suite: `20 passed`.
- `scripts/verify_docs_consistency.py`: passed.
- Corrected qualitative panel: 50 validation and 50 test rows, zero automatic
  structural flags.
- Release-text audit: 1,024 generated explanations, zero configured generated
  sensitive findings, zero source-copy failures.

## Remaining Gates

The checkpoint is scientifically usable under the bounded SFT claim, but
public distribution remains blocked on:

1. Two blinded human semantic reviews and inter-rater reporting.
2. Repository-license selection and owner/legal approval for weights.
3. Confirmation of the exact teacher-API service agreement and output rights.
4. Final third-party notices.

Fourteen phone-like strings occur only in source excerpts; examples containing
source text remain internal until adjudicated or redacted. A genuinely
unexposed external teacher-backed boundary is required only for a confirmatory
generalization claim. A second AV seed is recommended for an architecture-level
paper but is not required to describe this one checkpoint artifact.
