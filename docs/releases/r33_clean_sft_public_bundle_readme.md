# R33 Clean SFT Public Bundle Candidate

This is the lightweight, redacted evidence and source bundle for release ID
`r33-clean-sft-av-ar-iter1291-20260715`.

## Defensible Claim

On a deterministic stored R33 activation snapshot with content-family-disjoint
clean SFT splits, the selected Nano30B AV-generated text carries row-specific
activation information that the paired AR reconstructs directionally. The
signal passes semantic and activation controls, stored-snapshot functional
recovery, and a validation-only cross-critic gate through an independently
initialized and trained AR.

This is an exploratory within-corpus, stored-snapshot result. It is not a claim
of exact fresh-forward activation identity, native raw-magnitude recovery,
project-wide unseen generalization, R33 superiority over R27, or RL gain.

## Bundle Contract

- `bundle_manifest.json` binds every staged file before the manifest itself is
  written and records all path redactions.
- `source/` contains the release-relevant source, configs, tests, and active
  documentation.
- `evidence/` contains aggregate reports, verifiers, fingerprints, family
  manifests, compute accounting, compact curves, retained AV/independent-AR
  package snapshots, and the independent-AR replication record.
- No checkpoint weights, optimizer state, activation parquet, generated text,
  credentials, or W&B binary files are included.
- Internal locators are replaced with `${NANO_LOCAL_HOME}`,
  `${NANO_INTERP_ROOT}`, `${NANO_MODEL_ROOT}`, `${NANO_INTERNAL_S3_ROOT}`,
  `${NANO_CLUSTER_HOST}`, or `${NANO_S3_ENDPOINT}`.

The primary AR run did not retain a separate `pip freeze` file. Its exact run
plan, command, runtime fingerprint, source/config hashes, and the same-family
independent-AR environment snapshot are included, but this remains a disclosed
reproducibility limitation rather than an invented lockfile.

## Release Blockers

This candidate is **not cleared for public distribution**. Before publishing
source or weights, complete the two-reviewer blinded semantic review, resolve
the top-level repository license and exact base-model/teacher-API terms, finish
third-party notices, adjudicate the fourteen source-only phone-like strings,
and rerun the security audit on the exact final archive. An external
teacher-backed boundary is additionally required for a confirmatory
generalization claim.

See `docs/releases/r33_clean_sft_publication_checklist.md`,
`docs/releases/r33_clean_sft_license_provenance.md`, and
`docs/releases/r33_clean_sft_human_review.md` for the authoritative gates.
