# R33 SFT Confirmatory Evaluation Preregistration

Status: **in-corpus boundary infeasible; external boundary required**

Protocol config:
`configs/nano_roundtrip/publication/r33_clean_sft_confirmatory_protocol.yaml`

## Purpose

This protocol evaluates the already selected family-clean R33 SFT AV+AR
checkpoint pair on a new confirmatory family boundary. The checkpoint pair,
primary endpoint, controls, thresholds, parser requirements, functional
metrics, and statistical procedures are frozen before that boundary is
materialized or inspected.

No additional AV or AR HPO is allowed under this study identity. Any retrained
seed, magnitude-calibration model, R27 baseline, or RL model is a separate
registered experiment.

## Primary Claim

Given an R33 activation, the fixed AV checkpoint emits an explanation from
which the fixed AR checkpoint reconstructs activation direction. The primary
metric is `2 * (1 - cosine)`, paired by row and analyzed over independent
content families.

The candidate must remain below directional MSE `0.35`, within `0.05` of
teacher-text reconstruction, and at least `0.1344` better than every registered
control. It must achieve at least `90%` rowwise wins, `95%` closed generations,
`99%` usable generations, and family sign-flip `p <= 0.01`.

## Boundary Policy

The confirmatory boundary must contain at least 512 rows and 100 independent
families. It must have zero family exposure in training, HPO, prior validation,
or prior test artifacts, and every historical document must map to a family or
an explicitly resolved out-of-universe record. Unmapped historical documents
fail the boundary.

The boundary is evaluated once. Generated outputs, failures, and exclusions
are retained. A failed result is not repaired under the same study ID.

## Functional And Magnitude Claims

Fresh-forward functional recovery is claimed only if the separately frozen
activation-fidelity diagnostic passes its predeclared tolerances. Otherwise the
functional result remains explicitly stored-snapshot counterfactual
reinjection.

Raw activation magnitude is secondary. It becomes a supported claim only when
raw MSE beats the train-mean predictor and centered raw R2 is positive without
test-set calibration. Directional success cannot substitute for that gate.

## Fixed Checkpoints

- AV fingerprint:
  `dcp_model_sha256:43346232d2fc043260ee903191e20cce07801903e1e7b7956f16022eb463386a`
- AR SHA-256:
  `5e792120ec1a00ebb4cf4abca50d2a6a962421ac4f45423479ae5061f4d2d760`
- Pair release:
  `r33-clean-sft-av-ar-iter1291-20260715`

## Superseded Evaluation-Only Audit

The v4 joint-family evaluation-exposure audit passed before confirmatory
generation. It
audited 130 historical validation/test parquets and 9,077 unique prior
documents. The resolver classified 8,059 by direct document ID, 185 by exact
content hash, 747 as near-duplicates of R33 candidate families, and 86 as
outside the R33 candidate universe. No historical document remained unmapped.

That audit would leave 13,146 rows across 242 independent families after
excluding 3,355 evaluation-exposed or near-duplicate families.

- Family manifest SHA-256:
  `26aec15511f0e9b70263bfe7a879b91e2d18b2c95f207a31c4a160608a535ea3`
- Exposure report SHA-256:
  `dc53092a4739b8462dd7a6e9f9cb769a6986031ffc5a3f2e35640ca5178204b8`
- Exposure inventory SHA-256:
  `32b29a4bbafaec7bdc0ec2ea66ecdf1bee6e4e20bdb5986bfc5b76d661419b30`
- Joint-family manifest SHA-256:
  `22461eaabc27928127108290e7b184e62028ed57c4b17b7e1f52c6d9509ce134`

The v4 result is not a valid confirmatory boundary because it omitted the exact
train splits consumed by the selected AV and AR checkpoints. It is preserved
as audit evidence and must not be used for confirmatory scoring. The v5 audit
adds both selected training sources to every historical validation/test source
and remains fail-closed. No confirmatory output may be generated until v5
passes and its hashes are frozen. If no in-corpus family remains, the study
requires a genuinely new teacher-backed corpus boundary rather than a weaker
overlap rule. The v6 audit names the exact train, validation, and test
partitions for both selected checkpoints, then adds the project-wide historical
validation/test ledger. Its infeasibility report is itself a required
publication artifact if those selected partitions exhaust the candidate
families.

## Full-Exposure Result

The v6 audit completed with a preserved fail-closed report. Across 136 exact
exposure sources it found 28,665 unique historical documents, zero unmapped
documents, and exposure of all 5,009 R33 candidate families. Split assignment
is therefore mathematically infeasible because the test split would contain no
family. No confirmatory generation was run.

- Exposure report SHA-256:
  `373e2988b32f2e4e68e2d7644a8b77430829ea7a5d5e2f6b72f870caa89d088b`
- Exposure inventory SHA-256:
  `c193f2efb7c8414f4f8a6a12ab2051e633b891107f92c2971535164128f9aabb`
- Joint-family manifest SHA-256:
  `9d68a894e763ed533fba11016ebc8b8f05c0d4a39443585196e7cd24ebffbc20`

The selected pair remains valid for the disclosed family-disjoint exploratory
evaluation already reported. A pristine confirmatory claim now requires a
teacher-backed external corpus family that was absent from checkpoint fitting,
hyperparameter selection, and prior evaluation. Repartitioning the same
275,396 rows cannot solve this constraint.
