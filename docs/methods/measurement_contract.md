# NLA Measurement Contract

This document defines what the project metrics do and do not support. Code,
run summaries, and external writeups should use these definitions.

## Activation Target

The current R33 datasets contain deterministic stored snapshots from one
frozen extraction pipeline. A full eight-shard replay produced the same merged
Parquet hash, which proves deterministic extraction and storage.

It does not prove equality to an arbitrary fresh forward. The archived
32-row live-vs-stored check failed its predeclared tolerance on every row:

- mean relative L2 drift: `0.025623`;
- maximum observed relative L2 drift: `0.077284`;
- report field: `publication_ready: false`.

Accordingly, downstream paired SFT/RL comparisons may claim improvement on the
same deterministic snapshot target. They must not claim exact reconstruction
of fresh model activations unless a new live-vs-stored gate passes.

## Reconstruction Metrics

The primary directional metric is:

```text
directional_mse = ||unit(h_hat) - unit(h)||^2 = 2 * (1 - cosine)
```

It is dimension-independent and ignores activation magnitude. Historical
reports called this `normalized_mse` or NMSE. Another historical convention
divided the same quantity by `d_model=2688`; those values are exactly 2,688
times smaller. New reports use `directional_mse`; `normalized_mse` remains only
as a compatibility alias.

Every directional result must be accompanied by:

- `raw_mse`;
- `centered_r2` against the train-mean predictor;
- `norm_ratio_mean`;
- cosine or directional MSE.

For the clean 512-row AR validation artifact, teacher text gives directional
MSE `0.281703` and cosine `0.859148`, but raw MSE `8.537785` versus train-mean
raw MSE `7.104776`, hence centered R2 `-0.201696`. This supports directional
recovery from teacher text. It does not support raw-magnitude reconstruction.

## Statistical Unit

Selection uses validation only. Test remains sealed until the checkpoint and
analysis are locked. Paired uncertainty is resampled over independent content
families, not rows. Reports must state row count and independent-family count.

Shuffled controls are drawn from the same split and a different content family.
Cached generations are reusable only when generation protocol, model,
tokenizer, and all selected dataset hashes match exactly.

Content families use exact normalized-text union plus deterministic prefix
filtering that evaluates every possible pair at or above the declared `0.80`
five-token-shingle Jaccard threshold. Older manifests built with the
probabilistic candidate pass must be rebuilt before sealed-test use.

## Critic Dependence

RL reward and the primary round-trip score use a frozen AR critic. Functional
and invariance evaluations that rescore vectors from that same AR are not
critic-independent. A cross-critic claim requires an independently initialized
and independently trained AR checkpoint, evaluated under the same protocol.

## Data And Teacher Scope

The current teacher is `nemotron-3-super-v3`. The original preparation dropped
865 teacher parse failures and 1,015 unmatched activation rows. These exclusions
must remain visible in data lineage; future roots should emit row-level reject
manifests rather than only aggregate counts.

The May 28 teacher-backed root parquet is byte-bound by SHA-256
`76b78d2c34a251f004d53eb5d53766fa01879e2bf3744bc4d80d4fcc1d17825e`.
Active extraction configs must verify this digest before reading rows.

The modified code under `external/natural_language_autoencoders` is the
production adaptation, not a pristine upstream checkout. Material divergences
include the frozen-critic RL design, project FVE definitions, Nano architecture
adapters, and injection scale `75` versus the reference recipe's `150`. Claims
of method parity must enumerate these differences.
