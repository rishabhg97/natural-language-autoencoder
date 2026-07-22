# R33 Nano30B NLA — Open Issues Before Any External Publication

Status date: `2026-07-16`. Companion to
`docs/publication/r33_nano30b_nla_paper_draft.md`. Ordered by category;
within each category, roughly by severity. "Blocker" = must be resolved
before the corresponding release action; "disclosure" = may remain open if
stated plainly in the published text (the current drafts do state them).

## A. Scientific

1. **No confirmatory boundary exists (blocker for any confirmatory claim).**
   The v6 exposure audit (report `373e2988…088b`) shows all 5,009 in-corpus
   content families were exposed to selection or historical evaluation, and
   the teacher-table inventory found no external teacher-backed documents.
   A genuinely new, legally usable, teacher-backed external corpus boundary
   — evaluated once under the frozen protocol
   (`configs/nano_roundtrip/publication/r33_clean_sft_confirmatory_protocol.yaml`)
   — is required before the word "confirmatory" may be used. Repartitioning
   the existing 275,396 rows cannot fix this.
2. **Raw magnitude is not recovered (disclosure; blocker for any
   "activation reconstruction" claim without the directional qualifier).**
   Native centered raw R² is −0.327/−0.335; the 0.5606 scalar repair is
   validation-fitted and test-exposed (post-hoc). A magnitude-aware AR head
   or calibrated training objective is future work.
3. **Fresh-forward identity fails (disclosure).** All 64 audited rows
   violate strict tolerances (mean rel-L2 3.1%, max 18.6%) despite cosine
   0.9991. Claims must stay stored-snapshot scoped. Root cause (runtime
   drift between extraction-era and current stacks) is characterized but
   not eliminated; a frozen-runtime re-extraction plus immediate evaluation
   in the same runtime would be needed for fresh-forward claims.
4. **Single AV seed (disclosure; blocker for architecture-level claims).**
   Cross-critic replication covers the AR side only. A second AV training
   seed is strongly recommended for any method paper.
5. **No semantic-faithfulness evidence (blocker for any explanation-quality
   claim).** Blinded human review packets exist
   (`PUB/blinded_human_review/review_packet_reviewer_{1,2}.json`) but no
   ratings have been collected; the automatic structural screen is not a
   semantic judgment. The July-8 audit documented frequent hallucinated
   details in RL-era generations; the clean SFT generations have not been
   human-audited either.
6. **Teacher/source confound (disclosure).** AR reconstruction from raw
   source context (0.3013) is statistically close to teacher text (0.2817),
   and the teacher never saw `h`. How much reconstructed direction reflects
   activation-specific computation versus source-predictable content is
   unresolved; av_shuffled/av_none margins bound but do not settle it.
7. **R33 boundary-label conventions are unreconciled (disclosure; verified
   at code level 2026-07-16).** The deployed extractor
   (`nano_ar_layer_sweep.py::_forward_selected_boundaries`,
   `boundary = layer_idx + 1`) and the core plan define `R_b` as the
   residual state **after b blocks** — R33 = output of zero-based block 32
   (Mamba-2), immediately before the attention block at index 33,
   supporting the research memo's "post-Mamba, pre-attention" reading. The
   2026-07-10 "corrected convention" (staging-design doc;
   `nano_ar_correctness_audit.v2`; followed by `docs/current_state.md`)
   instead asserts R33 = module index 33 with a 34-block critic. The
   functional/identity code hooks `layers[boundary − 1]`, consistent with
   the extractor; the AR checkpoint physically retains 34 blocks. All
   measurements are internally consistent; only the label-to-block mapping
   and hence any architectural-type claim is blocked until the audit
   convention and extractor convention are explicitly reconciled.
8. **Layer selection is pre-fix scouting (disclosure).** R33-vs-R27/R34
   probes predate the packed-boundary and LR fixes; no clean row-matched
   R27 comparison exists, so no layer-superiority claim.
9. **No valid RL result (disclosure; blocker for any RL claim).** July-8
   headline invalidated; salvage rescore is internal-only on a superseded
   activation lineage. Future RL must follow the draft preregistration:
   registered guards, four-point stability grid, two independent seeds,
   independent-critic primary endpoint, length-matched controls, one-shot
   test after selection lock. The preregistration itself is stale (see C.6)
   and must be revised for the external-boundary reality before
   registration.
10. **Injection scale 75 never re-derived for R33 (minor).** Inherited from
    R27; upstream recipe uses 150. `NANO_FORK.md` concedes no clean scale
    sweep exists.
11. **Validation split carries no exposure guarantee (disclosure).** The
    split builder forbids historically exposed families from *test* only;
    validation-side exposure was never constrained. Validation numbers are
    selection-adjacent evidence, not held-out evidence.
12. **Functional metric uses oracle norms (disclosure).** Reinjected
    reconstructions are rescaled per-row to the stored activation's norm,
    so functional recovery is directional-under-oracle-norm; without the
    rescale, the negative raw-R² problem would apply to reinjection too.

## B. Reproducibility and evidence integrity

1. **Public bundle ships the wrong exposure report (blocker for release;
   flagged prominently).** The staged tree and archive
   (`3eb8e64e…573a`, 496 files) contain, as their only exposure/split
   evidence, the obsolete v2-era report with
   `unmapped_prior_document_count=833`, 130 sources, 3,083 forbidden
   families, a 4,612/356/41-family split summary that contradicts the
   frozen 4,504/250/255 split shipped beside it, and `passed=false` for the
   wrong reasons (staged SHA-256 `682aeec2…4ad1`; source copy
   `bd9cc5d9…f30f` inside the July-15 qualified evidence tree at
   `…/confirmatory/r33_content_families_v2/`). The authoritative v6
   report/inventory/joint-manifest JSONs are *absent* from the bundle (only
   hash-referenced in its prereg doc and protocol YAML) and exist locally
   only inside `PUB/r33_exposure_audits_v4_v6.tgz`. No bundled document
   flags the shipped report as superseded, while the bundle's own staged
   docs/configs cite the v6 numbers — the archive contradicts itself.
   `bundle_manifest.json` cryptographically binds the obsolete file, and
   the **current** builder config
   (`configs/nano_release/r33_public_bundle_candidate.yaml`, SHA-256
   `03ca4a47…fa7b`, line 89) still stages it, so a naive rebuild reproduces
   the defect — fix the copy list, add the v6 artifacts, restage, re-audit,
   re-attest. (The live config has also drifted from the shipped bundle's
   recorded build config, `a48c5359…` → `03ca4a47…`, so the bundle no
   longer reflects current intended contents either way.) Until then the
   archive must not be described as final or self-contained.
2. **v6 exposure artifacts have no loose local copy (medium).** They live
   in one local tgz, on the cluster, and in internal S3. Extract and pin
   them as first-class local artifacts (and into the restaged bundle).
3. **Upstream NLA commit pin unverifiable (disclosure).**
   `REFERENCE_REPOS.md` cites commit `047eb8e…`, but the vendored tree has
   no `.git`, the hash resolves nowhere, and `NANO_FORK.md` states the
   import did not preserve an upstream ID. Method-parity claims are
   accordingly barred.
4. **No `pip freeze` for the primary AV/AR runs (disclosure).** Environment
   snapshots exist for the independent AR only; the bundle README discloses
   this.
5. **v5 exposure audit configured but never executed (minor).** The chain
   went v4 → v6; the preregistration doc still describes v5 as a required
   gate. Annotate or remove.
6. **Preregistration doc is stale (medium).** It still claims the clean AV
   eval and SFT round-trip are "pending" (both passed 2026-07-15/16) and
   presumes an in-corpus sealed test that the v6 audit has made impossible.
   Revise before any registration.
7. **rl_logbook.md is stale (minor).** "Last updated 2026-07-10"; still
   presents the update-342 checkpoint as "Selected run" below the caution
   banner, and pre-dates the July 15/16 qualification.
8. **Registry coverage of failures is partial (minor).** Guard-stopped RL
   heroes and the removed pre-fix AR are registered, but many failed runs
   (independent-critic launch failures, confirm-32 retries, AV startup/save
   failures, OOM probes) exist only in `docs/nano_av_job_tracker.md`.
   `runs/registry/experiments.yaml` also still lists
   `r33-prefix-fullscan275396-dataset` as `passed` without the
   "contaminated scouting provenance" reclassification from
   `docs/current_state.md`.
9. **Two documented metric-value generations for validation (disclosure).**
   July-15 report vs July-16 immutable-cache rescore differ at the 4th
   decimal (0.307004 vs 0.306969 directional; −0.326586 vs −0.326250 R²).
   Not bit-exact scorer replay; every quoted validation number must name
   its report. The evidence table does.
10. **Docs say "16 subgroup bins"; the report realizes 13 per split
    (minor).** The family-frequency dimension collapses to one bin
    (fit edge [2.0]). Correct the prose in
    `docs/current_state.md` / model card, or report requested-vs-realized.
11. **AV padded train table has 247,872 rows vs 247,865 manifest rows
    (minor).** Presumed padding rows; document the exact mechanism in the
    dataset notes so the mismatch is not read as leakage.
12. **Tree-hash recipes undocumented (minor).** Neither the 496-file
    audited tree hash (`df175c5f…`) nor the manifest's 495-file
    `tree_manifest_sha256` (`ed1d0f9c…`) ships with a recomputation recipe;
    A future reader cannot independently reproduce either without reading
    the builder source.
13. **AV/AR kernel-profile mismatch (disclosure).** The AV and AR ran with
    different validated Mamba kernel profiles (registry caveat); a
    cross-kernel delta was planned but only the per-run equivalence gates
    exist.
14. **AV component verifier does not bind checkpoint/dataset identity
    (minor code gap).** `verify_nano_av_eval_report.py` hashes only the
    report file, unlike the roundtrip/functional verifiers; the AV report's
    linkage to the fingerprinted checkpoint rests on the pair manifest and
    run directory, not the verifier.
15. **Roundtrip verifier does not recompute aggregate directional MSE from
    rowwise arrays (minor code gap).** It recomputes control statistics and
    win fractions but trusts the report's aggregate primary/teacher values
    and does not hash-chain rowwise arrays back to the generation cache.
16. **docs/runs/README.md omits two run docs (trivial).**
    `r33_publication_preregistration.md` and `r33_hero_handoff_20260622.md`
    are unlisted.
17. **Fidelity evidence rests on a single artifact with an internal field
    inconsistency (disclosure).** The base (non-mb8) 64-row fidelity run
    OOM'd (`activation_fidelity_validation64_runner.json` returncode 1;
    CUDA OOM at the `lm_head` projection) and produced no JSON, so
    `activation_fidelity_validation64_mb8.json` is the only fidelity
    artifact. Inside it, the aggregate `activation_fidelity` block records
    max rel-L2 `0.195980` while the per-row `rows` array and
    `fidelity_assessments` record `0.185983` (means differ too: `0.031405`
    vs `0.031329`). The verdict (strict identity fails on all 64 rows;
    cosine ≈ 0.9991) is unaffected, but quoted values must name their
    field, and the aggregate-vs-per-row divergence should be explained or
    fixed in the diagnostic before external use.
18. **Independent-AR evidence bundle does not hash its own split parquets
    (medium).** `IND/run_artifacts/splits/split_manifest.json` and
    `split_content_verify.json` record family/row counts and overlap checks
    but no parquet content hashes; the expected train/validation hashes
    appear only via the roundtrip report's provenance pointing at the *AV
    run's* splits directory, and the test-split hash appears nowhere in the
    IND bundle (test was not scored there). Row/family counts are the only
    content binding for the independent run's own materialized splits.
    Additionally, the independent AR's optimizer state was retention-deleted
    (HF export only) although its `run_spec.yaml` sets
    `checkpoint.require_optimizer_state_for_hero: true` — fine for a
    replication artifact, but the run is no longer resumable.
19. **AR component eval violates the metric contract's companion-metric
    rule, and its verifier fails open on missing metrics (minor code gaps).**
    `eval_nano_ar_miles_checkpoint.py::metric_summary` omits
    `norm_ratio_mean` (contract: every directional result carries raw_mse,
    centered R², norm_ratio_mean, cosine), and
    `verify_nano_ar_eval_report.py` checks teacher thresholds and
    control gaps only inside `if _finite(...)` guards — a missing or NaN
    teacher/control metric silently skips the check unless that control is
    separately listed in `required_controls`. The roundtrip path is
    compliant and fail-closed; only the AR component leg has these gaps.
20. **Functional "shuffled" control does not match the contract's shuffled
    definition (disclosure).** The measurement contract defines shuffled
    controls as same-split, different-content-family; the functional
    evaluator's `within_document_shuffle` rotates stored vectors *within
    the same document* (78/88 rows). It is a deliberately harder control,
    but any text calling it "shuffled" without the within-document
    qualifier misdescribes it. The AV/roundtrip shuffles do follow the
    contract (with the cross-family constraint waived only for rows lacking
    a family ID).
21. **Family-clustered inference is verifier-enforced, not
    evaluator-enforced (disclosure).** The evaluators silently degrade the
    statistical unit (`content_family_id` → `doc_id` → row) when family IDs
    are missing, gate defaults are permissive (`min_control_win_fraction=0`,
    `min_closed_fraction=0`, `require_family_level_inference=False`), and
    `eval_nano_r33_functional_recovery.py --min-independent-families`
    defaults to 1 (vs the roundtrip gate's 100). The qualified runs used
    strict configs and the verifiers recompute clustered statistics
    fail-closed, so the published numbers are safe — but "gate passed"
    without the paired verifier does not certify the statistical-unit
    contract.
22. **`fve_nrm` name collision with upstream (disclosure).** The project's
    `fve_nrm` is `1 − directional_mse / directional_mse(train-mean
    predictor)` — the upstream codebase's `fve_nrm_meannorm` concept —
    whereas upstream's field named `fve_nrm` uses a stricter raw-variance
    baseline. Identical names, different baselines; never compare across
    codebases without renaming.
23. **Naming nits (trivial).** The release ID is
    `…-iter1291-20260715` but the S3 release-candidate prefix is
    `…-iter1291-20260716` (candidate-dated; visually collides with the
    release-ID scheme). The license-provenance table and NOTICE record the
    teacher as `nvidia/nvidia/nemotron-3-super-v3` (doubled org prefix);
    confirm the canonical model string before any public document quotes
    it.

## C. Packaging and release engineering

1. **Restage the public bundle** after fixing B.1/B.2 (new archive, new
   security audit, new attestation; keep the old hashes quarantined as
   superseded).
2. **Attestation and security report live outside the archive (by design,
   but document it).** The archive alone cannot prove its own audit status;
   the S3 release-candidate prefix holds all five objects together — say so
   wherever the archive is cited.
3. **Internal source snapshot is not public-clean (known).** 31 files with
   internal references must be redacted/excluded if a fuller source release
   is ever wanted (current 496-file candidate already excludes them).
4. **Weights are not staged anywhere public.** `weights_included=false`;
   S3-internal only. Any weight release needs its own bundle, audit, and
   attestation round.

## D. Human review

1. **Blinded semantic review (blocker for quality claims; strongly
   recommended before any release).** Two reviewer packets are frozen; no
   ratings collected; inter-rater agreement and adjudication unreported.
   Do not hand reviewers the answer key.
2. **Phone-like source excerpts (blocker for releasing examples).**
   Fourteen phone-like patterns in source excerpts require human
   adjudication or redaction before any source-grounded example leaves the
   project. Generated text itself screened clean (automatic only).
3. **Human pass over the drafts.** The paper/blog drafts in this directory
   were machine-verified against artifacts but need a human owner's
   read-through, authorship decisions, and venue formatting.

## E. Legal and licensing

1. **Teacher-output terms unidentified (highest-risk blocker).** The exact
   account/subscription/agreement governing the `nemotron-3-super-v3`
   inference-API calls is unknown; if API Trial Terms govern, distribution
   of Generated Content is restricted. Written confirmation required before
   publishing teacher text, generated evaluation text, checkpoints trained
   on them, or possibly even derived examples.
2. **No repository license (blocker for source release).** Owner must
   select and approve one.
3. **Base-model redistribution (blocker for weight release).** Confirm the
   acquired Nano30B files are governed by the NVIDIA Open Model License as
   the local card states, decide derivative-model status for AV/AR
   checkpoints, and assemble the required agreement + NOTICE.
4. **FineWeb/Common-Crawl obligations (needed for any data-derived
   release).** ODC-By-1.0 attribution plus Common Crawl ToU review for any
   distributed excerpts or derived tables; default plan (hashes, manifests,
   aggregates only) minimizes this.
5. **Third-party notices incomplete.** Upstream NLA Apache-2.0 notice is
   present; a complete NOTICE for every redistributed dependency is not.
   `docs/releases/r33_clean_sft_NOTICE.txt` is intentionally unfinished.

## What the evidence currently supports

- **Checkpoint artifact release:** supported *after* D.1–D.2 and E.1–E.3
  close — the scientific gates for describing the pair as a qualified
  supervised NLA artifact under the bounded claim are all green.
- **Exploratory technical report** (the paper draft's claim, exactly):
  supported now, subject to the disclosures above and A-category wording
  limits.
- **Confirmatory scientific claim:** not supported; blocked on A.1.
- **Validation-only RL claim:** supported for the retained family-clean
  `a3e5_u342` pair against its exact matched SFT baseline (`27.4%` lower
  directional NMSE on 122 independent validation families). This is a
  pair-level AV+AR claim; actor-only attribution remains open.
- **Publication-level RL claim:** not supported; local artifact hash binding,
  four-way AV/AR component decomposition, functional reinjection delta,
  independent-critic transfer, a second seed, and a new external test boundary
  remain open.
