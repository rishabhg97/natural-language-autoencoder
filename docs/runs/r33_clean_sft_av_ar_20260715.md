# R33 Family-Clean SFT AV+AR Qualification

Date: `2026-07-15`

Release ID: `r33-clean-sft-av-ar-iter1291-20260715`

## Decision

The selected R33 supervised checkpoints qualify as a good Nano30B Natural
Language Autoencoder pair for the following bounded claim:

```text
R33 stored activation h
-> AV-generated explanation z
-> AR reconstruction h_hat
```

On family-disjoint validation and test samples, the generated text reconstructs
activation direction close to teacher-text reconstruction, parses successfully
on every evaluated row, and decisively beats mean, shuffled, zero, mean-AV, and
no-injection controls. Stored-snapshot counterfactual reinjection also matches
teacher-level functional behavior within family-clustered uncertainty and beats
the nonsemantic controls.

This is a qualified **SFT AV+AR checkpoint pair**, not an RL result and not a
claim of raw activation-magnitude recovery or exact fresh-forward replay.

## Selected Checkpoints

### AV

- Run: `nano-av-r33-publication-deterministic-family-clean-8gpu-pospass-lr1e4-warmup25-gb192-mb2-dyn4096`
- Iteration: `iter_0001291`
- Format: Miles/FSDP2 distributed checkpoint
- Model fingerprint:
  `dcp_model_sha256:43346232d2fc043260ee903191e20cce07801903e1e7b7956f16022eb463386a`
- Tokenizer fingerprint:
  `tokenizer_files_sha256:924108201acc4872dca41bbd39caad108038634a428c2bb26fee7516d424ad0f`
- S3:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/publication/checkpoints/r33_deterministic_family_clean_av_sft/iter_0001291/`

### AR

- Run: `nano-ar-r33-publication-deterministic-family-clean-4gpu-unfusedtorchconv-expertscan-cudablock-lr5e5-warmup25-gb192-mb48`
- Iteration: `iter_0001291`
- Format: loadable Hugging Face checkpoint
- SHA-256:
  `5e792120ec1a00ebb4cf4abca50d2a6a962421ac4f45423479ae5061f4d2d760`
- S3:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/publication/checkpoints/r33_deterministic_family_clean_ar_sft/iter_0001291/`

## Data Boundary

- Rows: `275,396`
- Documents: `27,647`
- Exact-refined content families: `5,009`
- Clean train families: `4,504`
- Validation families: `250`; eligible rows: `13,766`
- Test families: `255`; eligible rows: `13,765`
- Train/validation/test family overlap: `0`
- Validation/test evaluation rows: `512 / 512`
- Primary statistical unit: `content_family_id`

Dataset hashes:

| Split | SHA-256 |
|---|---|
| Train | `cf618cb08b682f6316e2807a095b3bfcef597c8b2d1182c7cce753dca4c9fe6c` |
| Validation | `f543eb9e1017a655c3ab436c0140c5c334e8485b0d493083d27aaaed9d9ce1ff` |
| Test | `86973528e153c7d6bd9c0fd0fdb72ebfb614e2964e4bde2b76c742d7f5763c5f` |

The test partition is family-disjoint from the clean SFT training exposure. It
is not guaranteed to have been untouched by every historical exploratory run
in this long-lived project, so it should not be described as a pristine final
publication test set.

## Component Gates

### AR teacher-text reconstruction

The hash-bound validation verifier passed on 512 rows:

| Metric | Result |
|---|---:|
| Teacher directional MSE | `0.281703` |
| Teacher cosine | `0.859148` |
| Teacher FVE-NRM | `0.584534` |
| Teacher raw MSE | `8.537785` |
| Teacher centered raw R2 | `-0.201696` |
| Shuffled directional MSE | `0.968888` |
| Mean directional MSE | `0.678041` |

The AR checkpoint has strong row-specific directional signal. Negative centered
raw R2 means it does not establish raw-magnitude reconstruction.

### AV activation conditioning

The hash-bound validation verifier passed on 512 rows:

| Condition | Response NLL |
|---|---:|
| Real activation | `0.776775` |
| Shuffled activation | `1.311727` |
| Zero activation | `1.176494` |
| Mean activation | `1.237522` |
| No injection | `1.220974` |

Real activation conditioning beats every control by a large margin.

## AV-to-AR Round Trip

The primary metric is rowwise L2-normalized directional MSE,
`2 * (1 - cosine)`. Both strict verifiers passed.

| Split | Rows | Families | AV text | Teacher text | Raw MSE | Centered raw R2 | Closed / usable |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 512 | 250 | `0.307004` | `0.304714` | `9.449079` | `-0.326586` | `100% / 100%` |
| Test | 512 | 255 | `0.319225` | `0.302637` | `9.647148` | `-0.335374` | `100% / 100%` |

On test, the candidate beats the controls by the following family-clustered
directional-MSE margins:

| Control | Control minus candidate | Family-bootstrap 95% CI | Rowwise wins |
|---|---:|---:|---:|
| Mean | `0.361806` | `[0.352926, 0.370762]` | `99.80%` |
| AV mean | `0.522112` | `[0.511974, 0.532006]` | `100%` |
| AV none | `0.536178` | `[0.524715, 0.548039]` | `99.61%` |
| AV shuffled | `0.645884` | `[0.634535, 0.657831]` | `100%` |
| AV zero | `0.663983` | `[0.654222, 0.674222]` | `100%` |

All one-sided family sign-flip p-values are approximately `1e-5`. These
statistics support activation-specific information transfer through generated
language; they do not support raw-scale recovery.

## Functional Reinjection

Functional evaluation reinjects reconstructed activations into the stored
counterfactual snapshot and compares output distributions. It is not a fresh
model-forward identity claim.

| Split | Variant | KL(original || patched) | JS | Logit Pearson | Top-10 overlap | Top-50 overlap |
|---|---|---:|---:|---:|---:|---:|
| Validation | AV candidate | `1.088345` | `0.155017` | `0.901570` | `0.622656` | `0.637852` |
| Validation | Teacher | `1.199089` | `0.156927` | `0.901211` | `0.626563` | `0.641016` |
| Test | AV candidate | `0.949545` | `0.152073` | `0.907847` | `0.625586` | `0.639336` |
| Test | Teacher | `0.970104` | `0.145112` | `0.911889` | `0.631055` | `0.645313` |
| Test | Mean | `4.124133` | `0.488665` | `0.735405` | `0.217383` | `0.286797` |
| Test | Zero | `6.297471` | `0.594679` | `0.576885` | `0.106250` | `0.169648` |
| Test | Shuffled | `9.528919` | `0.656315` | `0.621783` | `0.031818` | `0.098636` |

Candidate-versus-teacher family intervals include zero on all registered
functional metrics. Candidate-versus-mean, zero, and shuffled intervals are
decisively favorable.

## Qualification And Evidence

The fail-closed release builder produced
`nano_nla_checkpoint_pair_manifest.v1` with `qualified: true`. It binds both
checkpoint fingerprints, all six passing verifiers, the family boundary,
source archives, raw reports, generated text caches, and explicit limitations.

- Local lightweight evidence:
  `artifacts/runai_eval/r33-clean-sft-av-ar-qualified-20260715/`
- Evidence archive SHA-256:
  `b9043eae03cbb71c38a3feb81262361e764d4a147ba98d66735293b75b24f6bc`
- Evidence S3 object:
  `s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/sync_exports/r33_clean_sft_av_ar_qualified_20260715.tgz`
- Release config:
  `configs/nano_roundtrip/publication/r33_clean_sft_checkpoint_pair_release.yaml`
- Release builder:
  `scripts/build_nano_nla_checkpoint_pair_manifest.py`

## Claim Boundaries

Supported:

- This is a good R33 supervised AV+AR NLA checkpoint pair under the frozen,
  family-disjoint stored-snapshot protocol.
- AV-generated text carries row-specific activation information that AR can
  reconstruct directionally.
- The generated-text reconstruction is close to teacher-text reconstruction
  and decisively better than semantic and activation controls.
- Stored-snapshot functional reinjection is teacher-level within uncertainty
  and much better than nonsemantic controls.

Not supported:

- raw activation-magnitude recovery;
- exact equality to a fresh Nano30B forward pass;
- a pristine never-before-seen project-wide test set;
- R33 superiority over a new row-matched R27 retrain;
- AV training-seed invariance or external-corpus generalization;
- RL improvement over this SFT pair.

## Publication Follow-Up (2026-07-16)

The selected-pair exposure audit is now complete. The earlier v4 audit looked
only at historical evaluation partitions and is superseded. The full v6 audit
included selected AV/AR train, validation, and test partitions plus historical
evaluation sources: `136` sources, `28,665` unique documents, all `5,009`
candidate families exposed, and zero unmapped documents. There is no unused
in-corpus family suitable for a confirmatory claim. An inventory of existing
teacher tables likewise found no numeric document ID outside `0..38161`.

Fresh-forward fidelity was rechecked on 64 validation-only rows. Repeated fresh
forwards and full-vs-extraction forward paths are exact, but every strict
fresh-vs-stored comparison fails. Alignment remains high (mean cosine
`0.999142`, minimum `0.983146`, centered raw R2 `0.997255`). This confirms that
the release claim must remain stored-snapshot scoped.

The round-trip scorer was rerun to emit immutable prediction caches. The
validation and test gates still pass on `512/512` rows. The new runtime is very
close to, but not bit-identical with, the July report, so the follow-up binds
the new report and cache hashes rather than claiming exact metric replay.

A validation-only teacher fit compared identity, origin-scalar, and
train-mean-scalar calibration. Raw MSE selected the one-parameter origin scalar
`0.560604`. Applied unchanged to AV predictions, it yields:

| Split | Raw MSE before | Raw MSE after | Centered raw R2 before | Centered raw R2 after |
|---|---:|---:|---:|---:|
| Validation | `9.446685` | `3.648806` | `-0.326250` | `0.487733` |
| Exploratory test | `9.647148` | `3.770353` | `-0.335374` | `0.478102` |

The exploratory-test family-clustered 95% interval for raw-MSE improvement is
`[5.634404, 6.123574]`. This is evidence that much of the raw error is a global
scale mismatch. It is post-hoc, does not change directional reconstruction,
and does not upgrade the pair to native exact magnitude recovery.

Follow-up evidence is under
`artifacts/runai_eval/r33-clean-sft-publication-evidence-20260716/` and the
matching S3 prefix
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/publication/evidence/20260716_r33_clean_pair/`.

### Subgroup and qualitative evidence

The subgroup audit fitted quartile edges on validation only and applied those
edges unchanged to the exploratory test. It covers source-token length,
teacher-target word count, target activation norm, and sample-family
frequency. All 16 validation/test bins have sufficient rows and families; the
lowest 95% control-improvement bound across the test bins is still positive
(`0.319044`). The weakest test directional slice is the lowest activation-norm
quartile (`0.370077`); its calibrated centered raw R2 is `0.415606`. This is a
post-hoc robustness description, not a new confirmatory analysis.

A deterministic panel sampled 50 validation and 50 test generations and binds
each output to source text and its teacher explanation. The first panel build
silently omitted the resolved `detokenized_text_truncated` source column; it
was invalidated and replaced after the resolver and panel builder were made
fail-closed on empty source. The corrected panel has zero missing source rows,
and its automatic structural screen found zero empty, encoded-looking,
repetitive, or severe length-regression outputs. Human semantic review remains
incomplete and is the only evidence that may support factuality or usefulness
judgments.

- Subgroup report SHA-256:
  `fd24360bfcc4ff51538b2b21f5d1b931befa7b31ee0af45551701a95835c972c`.
- Qualitative panel SHA-256:
  `4f5d61486330b1104dd0a256ea185d8c1c99512ee9ff4731f8135305924f81c8`.
- Structural review SHA-256:
  `33de2720d96bda3f663318be7fe8c10765740a3d603bcdf76682ca23661297da`.

The automatic release-text audit scanned all `1,024` frozen generated
explanations and found zero configured sensitive-pattern findings and zero
source-copy failures. The corrected 100-row panel has no candidate copy flag;
its maximum contiguous source match is five words. Fourteen phone-like
patterns occur only in source excerpts, so source-containing examples remain
internal pending human adjudication or redaction. Report SHA-256 is
`00e501ff644483e614d0b60071f726f33575c95ba8b81d5c6b59c4bd79d13419`.
This is automated triage, not proof of privacy, consent, or factuality.

### Independent AR replication status

The seed-`314159` critic initialization was rebuilt from the frozen base model.
The first byte-hash gate exposed one intentionally overwritten random
pre-initialization head hash; all final component hashes matched. The generic
queue gate now hashes canonical JSON while ignoring only
`value_head.before_sha256`. The preserved and rebuilt manifests share canonical
SHA-256
`34e863f756e0749ca19fc8c138b7bd71b5da69c907ee42ad021517542e5c8941`,
and the independent verifier passes all 16 checks. Its report SHA-256 is
`4639285fea694f7f850c766b31d8ddea4e2b2bdd61c710f3b2a4cdd2109fb6e3`.

The clean independent AR queue completed with seed `314159`, four H100-NVL
GPUs, `lr=5e-5`, 25 warmup updates, `gb192/mb48`, cosine decay, 1,291 updates,
validation-only selection, and offline W&B. Final train loss/FVE-NRM are
`0.295026 / 0.472163`. Its 512-row teacher-text component verifier passes:
directional MSE `0.286169`, cosine `0.856916`, and FVE-NRM `0.577948`, versus
primary-seed `0.281703 / 0.859148 / 0.584534`. Teacher beats shuffled on all
512 rows and mean on 507/512. Report SHA-256 is
`368c84cad8bb0b8a7235b1a1e96c862b69d33be72d49c1f07a563e62d0be65aa`.
Frozen validation AV generations were then scored through this independent AR.
The hash-bound verifier passes on 512 rows and 250 families. AV-text
directional MSE is `0.310963`; independent teacher-text reconstruction is
`0.308533`. All generations are closed and usable. All five controls pass
family-clustered inference with lower confidence bounds above zero; rowwise
wins are `99.80%` against mean and `100%` against AV-mean, no-injection,
shuffled, and zero controls. The report/verifier SHA-256 values are
`6f0829a61b03ac584b109c1c7a54f689f0a14b61f7fee803afd3d5ca29bf552b`
and
`dd3de6e1dd10f0f64c25e23ff3152638c9cf46785133dfb6c0466c353e434331`.
Cross-critic directional replication is therefore qualified on validation.
The independent centered raw R2 remains negative (`-0.399865`), so no native
raw-magnitude claim follows.

The independent HF directory contains 10 files totaling `38,462,226,688`
bytes. Its directory fingerprint is
`c2eea74f5baccee97128617b05636187804c7e59aedc560d088dbf65d52f1925`,
and the manifest file SHA-256 is
`79408abd1e7cafadbc68ebe627bca99381e1a3e4486aa950ca0e7a42c97bb1ed`.
S3 was verified at the exact object count and byte total under
`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/publication/checkpoints/r33_deterministic_family_clean_independent_ar_sft_seed314159/iter_0001291/hf/`.
After verification, manifest-first retention removed only the redundant 36G
model DCP and 72G optimizer state. HF, logs, split metadata, eval reports, and
offline W&B records remain preserved.

### Selected-run compute accounting

All three selected train logs contain exactly 1,291 unique optimizer steps and
hash-bound start/completion timestamps. Primary AR used 4 H100-NVL GPUs for
`3.8867h` (`15.5467` GPU-hours), primary AV used 8 for `13.4608h`
(`107.6867` GPU-hours), and independent AR used 4 for `3.8781h` (`15.5122`
GPU-hours). Total selected successful training is `138.7456` H100-NVL
GPU-hours. The report preserves logged utilization/memory envelopes and
explicitly excludes extraction, conversion, evaluation, historical HPO/RL,
and failed diagnostics without exact retained timing. Its SHA-256 is
`7bde74be3a874d2ae305463ca8da211c069ce0bf1001802b6bdf7ab091fd7238`.

### Internal source-bundle security audit

The source-only internal snapshot contains 680 files. It has zero forbidden
heavy/checkpoint paths, symlinks, binaries, oversized text files, or
unallowlisted credential fixtures. It is not public-release clean: 31 files
contain local-home, internal S3/endpoint, or cluster references. They must be
redacted or excluded from the exact public bundle, which must then be scanned
again. Tree-manifest SHA-256 is
`3469d92cc535e42b4e70098ce5f3640cb980e051c43427794a6afe4fef0cf8e8`;
audit-report SHA-256 is
`bf697acdad268e219cfd729428233b1259b07aea988529e8c7e4bb01b8a62c42`.

### Redacted public-bundle candidate

The failed full-snapshot audit was followed by a deterministic staging pass.
Release-relevant source, configs, tests, aggregate evidence, family manifests,
and compact metrics for all 3,873 selected optimizer steps were copied into a
candidate tree. Workstation, RunAI, internal S3, endpoint, and cluster locators
were replaced with named placeholders. Checkpoints, optimizer state,
activations, generated text, credentials, and W&B binaries were excluded.

The exact staged tree has 496 files and passes the static security audit with
zero failed findings, forbidden paths, symlinks, binaries, or oversized text.
Audited tree SHA-256 is
`df175c5f61cefbfc1a02451a7bd242ba69e1cb602cdd97ca4b8bd8fe9c263b77`;
audit-report SHA-256 is
`a130d8e4295a06d0372bd0d920d9dfd0c8f7649710652939b33dbc764089f096`.
A normalized deterministic archive was then read back member-by-member and
matched the same tree hash. The 6,859,370-byte archive SHA-256 is
`3eb8e64ed0d9d61ed2d6b0694fbaf96b99051a63f2ce1a6c99372d93832e573a`.
The compact metric-curve JSON SHA-256 is
`7d9c22b989c594e546ec08648d0319c37caad69ad502d2badb635d41706c42a6`.
The archive attestation explicitly records `weights_included=false` and
`legal_clearance_granted=false`.

Five lightweight release-candidate objects are preserved in internal S3 under
`publication/release-candidates/r33-clean-sft-av-ar-iter1291-20260716/`:
archive, SHA-256 sidecar, bundle manifest, security report, and attestation.
The remote listing reports the expected 6,859,370-byte archive.
The exact cross-document index is
`docs/releases/r33_clean_sft_release_candidate_attestation.md`.
