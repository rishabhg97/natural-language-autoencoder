# R33 Independent Cross-Critic Validation Design

## Objective

Prevent an R33 RL actor from being promoted merely because it learned to
exploit the same frozen AR critic used for both reward and evaluation. The
hero gate must include an independently trained R33 AR critic that never
participated in RL reward computation.

## Independent Critic

- Reuse the verified component-full R33 AR dataset and the exact seed-42
  content-component train/validation/test split used by the selected critic.
- Reuse the selected architecture and stable AR recipe: layer 33, `d_model`
  2688, `lr=5e-5`, cosine decay, warmup 25, `gb192/mb96`, one epoch.
- Change only the training shuffle seed to `314159`. This preserves heldout
  identity while producing a separately optimized critic.
- Save one model checkpoint without optimizer state. Evaluate 512/512 teacher
  NMSE and all AR controls before using it as an audit critic.
- Require finite activations, zero split/document overlap, and successful AR
  dataset verification exactly as for the selected critic.

## Cross-Critic Data Flow

1. Reuse the candidate and SFT generated-text JSONL files from the trusted
   full-prefix 512/512 round-trip evaluations. Do not regenerate text.
2. Score both JSONL files with the independent AR critic against the same
   hash-bound validation and test parquet rows.
3. Produce independent candidate and SFT round-trip reports containing exact
   dataset hashes, row keys, document IDs, rowwise NMSE, and clustered
   bootstrap summaries.
4. Compare the primary-critic and independent-critic effects in a dedicated
   cross-critic report.

## Promotion Gate

The cross-critic report passes only when both heldout splits satisfy all of:

- complete row and dataset-hash identity;
- candidate independent-critic NMSE is at least 5% below independent SFT;
- candidate wins more than 50% of paired rows;
- the document-clustered 95% bootstrap CI for improvement is strictly
  positive;
- independent relative improvement is at least 25% of the primary-critic
  relative improvement, preventing a large critic-specific collapse;
- all candidate generations remain closed and usable under the trusted
  generated-text record.

The output path is
`/workspace/interp/outputs/nano30b-nla-pilot/validity/r33-corrected-cross-critic-gate.json`.
The hero queue already requires `passed: true` at this path.

## Failure Handling

- A weak independent critic fails preflight rather than weakening thresholds.
- Missing rows, hashes, reports, or nonfinite values fail closed.
- If primary performance improves but independent performance does not, stop
  before hero training and treat the candidate as possible reward hacking.
- Preserve reports and generated text; the temporary independent checkpoint
  may be deleted only after its selected model-only artifact is archived.

## Verification

- Unit-test shuffle-seed rendering, paired report alignment, all threshold
  boundaries, and missing-data failures.
- Run the existing local and RunAI test suites.
- Record the independent AR eval, cross-critic report, source fingerprint,
  queue hash, and artifact hashes in the RL logbook.
