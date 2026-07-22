# R33 Clean SFT License And Provenance Inventory

Status date: `2026-07-16`

This is an engineering inventory, not legal advice or redistribution approval.
It records the terms found by the project and the decisions an authorized
owner or legal reviewer must make before source, weights, teacher text, or
source-grounded examples are published.

## Artifact Lineage

| Element | Recorded source | Evidence | Release status |
|---|---|---|---|
| Base model | `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | Local model card at `/workspace/interp/models/nano-30b-a3b-bf16-hf/README.md`; card identifies `nvidia-nemotron-open-model-license` | Legal confirmation pending |
| Source corpus | `HuggingFaceFW/fineweb`, `sample-10BT` | Extraction configs and `docs/methods/measurement_contract.md` | Attribution and terms review pending |
| Teacher model | `nvidia/nvidia/nemotron-3-super-v3` | `scripts/nano_stage2_super_teacher.py` | Exact service agreement pending |
| Teacher endpoint | `https://inference-api.nvidia.com/v1/chat/completions` | `scripts/nano_stage2_super_teacher.py` | Exact service agreement pending |
| Adapted NLA code | `external/natural_language_autoencoders/` | Apache-2.0 text in `external/natural_language_autoencoders/LICENSE` | Preserve license and notices |
| Project source | This repository | No top-level license is present | Owner must select and approve a license |

## Base-Model Terms

The local Nano30B model card points to the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
The official page, last modified October 24, 2025, says that models are
commercially usable and derivative models may be created and distributed,
subject to the complete agreement. Redistribution conditions include providing
the agreement and a NOTICE attribution when distributing the model. The terms
also incorporate NVIDIA Trustworthy AI requirements, guardrail conditions,
separate-component licenses, and trade-compliance obligations.

Required owner/legal checks:

1. Confirm that the exact Nano30B files acquired for this project are governed
   by that version of the license and not a different enterprise, evaluation,
   or internal-use agreement.
2. Decide whether the AV and AR checkpoints are derivative models under the
   governing agreement and which base-model files, if any, will be bundled.
3. Include the governing agreement and final NOTICE text when required.
4. Confirm Trustworthy AI, guardrail, export, trademark, and separate-component
   obligations for the intended hosting location and audience.

## Source-Corpus Terms

The official
[FineWeb dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
states that FineWeb is released under ODC-By-1.0 and that use is also subject
to Common Crawl's Terms of Use. It also warns that web-derived data can retain
harmful or copyrighted material.

Required owner/legal checks:

1. Preserve FineWeb attribution and citation in the release documentation.
2. Review ODC-By-1.0 and Common Crawl obligations for any distributed source
   excerpts, derived tables, or examples.
3. Do not distribute raw FineWeb rows, activation Parquets, or source-grounded
   qualitative packets by default. The current public-bundle plan needs only
   hashes, manifests, aggregate metrics, and permitted redacted examples.

## Teacher-Output Terms

The project generated teacher explanations with the NVIDIA inference endpoint
and model recorded above. The credential/account provenance does not yet bind
those calls to a specific subscription, enterprise agreement, or trial terms.
That missing contract identity is the highest-risk legal blocker.

The current official
[NVIDIA API Trial Terms of Service](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf)
limit trial access to internal testing/evaluation, restrict production use,
and restrict making Generated Content available to others. They also say that
an accompanying model or third-party license may govern Generated Content.
These terms may or may not govern the account used here; the project must not
guess.

Required owner/legal checks:

1. Identify the account, service provider, subscription, and terms effective
   when the teacher rows were generated.
2. Obtain written confirmation that using those outputs to train and publicly
   distribute the AV/AR checkpoint pair is permitted.
3. Unless cleared, keep teacher text, answer keys, generated evaluation text,
   and source-grounded examples internal. Aggregate non-reversible metrics and
   hashes should still be reviewed before release.

## Current Decision

Internal preservation on project S3 is allowed operationally but is not a
public redistribution decision. No checkpoint, source archive, teacher table,
or qualitative packet should be uploaded to a public registry until:

- the exact base-model and teacher-service agreements are identified;
- owner/legal signs off on the intended artifacts and claims;
- a repository license is selected;
- final NOTICE/license copies are assembled; and
- privacy/security review passes on the exact staged public bundle.

The draft attribution file is
`docs/releases/r33_clean_sft_NOTICE.txt`. It is intentionally incomplete until
the decisions above are recorded.
