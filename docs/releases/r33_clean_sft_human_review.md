# R33 Clean SFT Blinded Human Review

Status: packets ready; human ratings pending.

The human review is the remaining semantic-quality gate. Automatic structural,
privacy-pattern, and copying checks cannot establish factuality, coverage, or
usefulness.

## Materials

Internal evidence root:

`artifacts/runai_eval/r33-clean-sft-publication-evidence-20260716/blinded_human_review/`

- Give `review_packet_reviewer_1.json` only to reviewer 1.
- Give `review_packet_reviewer_2.json` only to reviewer 2.
- Do not give either reviewer `answer_key.json`.
- Keep source-containing packets internal. Fourteen source excerpts trigger a
  phone-like pattern and require adjudication or redaction before any example
  is public.

The answer key SHA-256 is
`27585eae51d55deb9bb3821afbd1f5d1d3e7cfd0e7c4167e4111566ff06c1856`.
The corrected source-grounded panel SHA-256 is
`4f5d61486330b1104dd0a256ea185d8c1c99512ee9ff4731f8135305924f81c8`.

## Review Contract

Reviewers score each candidate independently and without discussing examples:

- factuality;
- source coverage;
- coherence;
- unsupported-detail flag;
- privacy/sensitive-content flag; and
- pairwise preference.

Use the fixed scales and field schema embedded in each packet. Reviewers may
mark an item unreviewable with a reason; they must not infer which candidate is
the teacher or AV output. Do not change packet order, IDs, or hashes after
rating starts.

## Scoring

1. Save the two completed packet files outside the answer-key directory.
2. Point a copy of
   `configs/nano_roundtrip/publication/r33_clean_sft_blinded_review_score_template.yaml`
   at those completed files and the internal answer key.
3. Run `scripts/score_nano_blinded_review.py`.
4. Preserve the scored report, config, packet hashes, answer-key hash, reviewer
   role descriptions, and any adjudication log.

The scorer reports fixed quality thresholds, preference results, Cohen kappa,
and weighted kappa. A failed threshold is a result, not a reason to modify or
rerun the frozen packets. Any follow-up panel must be labeled a new study.

## Claim Boundary

Until two completed blinded reviews and the hash-bound score report exist, do
not claim that AV text is factually accurate, complete, useful, or preferred to
the teacher. The current supported statement is only that the generated text
is structurally usable and carries activation information through AR.
