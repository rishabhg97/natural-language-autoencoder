# R33 Nano30B NLA — Claim-to-Evidence Table

<!-- R33-HERO-BASELINE-PROTOCOL-INVALIDATED -->

Status date: `2026-07-16`. Every SHA-256 below was recomputed locally with
`shasum -a 256` from the file at the stated repo-relative path during
preparation of the publication drafts, except where marked
*(docs-cited; cluster-side)*. JSON values were read directly from the
artifacts; prose documents were not treated as authoritative.

Shared identities for all selected-pair rows:

| Identity | Value |
|---|---|
| Release ID | `r33-clean-sft-av-ar-iter1291-20260715` (pair manifest `qualified: true`) |
| AV checkpoint fingerprint | `dcp_model_sha256:43346232d2fc043260ee903191e20cce07801903e1e7b7956f16022eb463386a` (iter_0001291) |
| AR checkpoint fingerprint (HF dir SHA-256) | `5e792120ec1a00ebb4cf4abca50d2a6a962421ac4f45423479ae5061f4d2d760` (iter_0001291) |
| Independent AR checkpoint | HF dir SHA-256 `c2eea74f5baccee97128617b05636187804c7e59aedc560d088dbf65d52f1925`, 10 files, 38,462,226,688 bytes *(docs-cited; fingerprint report `artifacts/runai_eval/r33-independent-ar-publication-evidence-20260716/independent_ar_checkpoint_fingerprint.json`, SHA-256 `79408abd1e7cafadbc68ebe627bca99381e1a3e4486aa950ca0e7a42c97bb1ed`)* |
| Split parquet SHA-256 | train `cf618cb08b682f6316e2807a095b3bfcef597c8b2d1182c7cce753dca4c9fe6c`; validation `f543eb9e1017a655c3ab436c0140c5c334e8485b0d493083d27aaaed9d9ce1ff`; test `86973528e153c7d6bd9c0fd0fdb72ebfb614e2964e4bde2b76c742d7f5763c5f` |
| Refined content-family base manifest | `confirmatory/r33_content_families_v2/r33_exact_prefix_refined_base_manifest.json`, SHA-256 `d2756e39066ec722e0059868cebfec6f03890e38c6e0a5286f2b76bc4490d30f` (5,009 families / 27,647 docs). The split-manifest hash `479cbab5d21cd031cb72a770eebb3428e0d5419ebf8cce38c2ca6025e49741b6` cited in docs is cluster-side and was not locally recomputed |
| Generated-text caches (frozen) | validation `8075966743dc4a56f2c9dd05d3f22e97d1578b9d8c4cff88dbe0d4f6562042ab`; exploratory test `545b433d2d74948142fd27a530693aeaf5801c23321daca974e5b98d2b4900e1` |
| Generation protocol SHA-256 | `e5e3a2658d28975514dd962be18c149012ee1fc85f1d6f52ccc834f59c95d416` (empty prefix, greedy, max_new_tokens 256, stop `</explanation>`, injection scale 75, seed 20260709) |
| Pair manifest | `artifacts/runai_eval/r33-clean-sft-av-ar-qualified-20260715/publication/evidence/20260715_r33_clean_pair/checkpoint_pair_manifest.json`, SHA-256 `37166ce740ec7443b66929f99a83c2944a1110710f86978a4d8f3b6d429e3edb` |

Evidence-directory roots (abbreviated in the table):

- `QUAL` = `artifacts/runai_eval/r33-clean-sft-av-ar-qualified-20260715/publication`
- `PUB`  = `artifacts/runai_eval/r33-clean-sft-publication-evidence-20260716`
- `IND`  = `artifacts/runai_eval/r33-independent-ar-publication-evidence-20260716`

Classification key — **none of the rows is confirmatory**: *validation
(exploratory)* = family-disjoint validation used for selection; *test
(exploratory)* = family-disjoint test with documented historical exposure;
*post-hoc* = analysis performed after test exposure or fitted after the
fact; *negative/diagnostic* = a result that bounds or blocks a claim.

## 1. Component claims

| # | Claim | Exact metric (value) | Split / rows / families | Report (path, SHA-256) | Verifier (path, SHA-256, verdict) | Class |
|---|---|---|---|---|---|---|
| 1.1 | AV uses the real activation | real NLL 0.7767746376921423 vs shuffled 1.3117268836358562, zero 1.1764944556634873, mean 1.237522108363919, none 1.220973681542091 | validation / 512 / (250 families in split; verifier checks counts+gaps, not family stats) | `QUAL/evidence/20260715_r33_clean_av/eval_iter_0001291_v512_t512_gen8_report.json`, `ee8faecb42b11b4caaa35ee1a13e25090c2a4c9b22ae7bf0385434b703bd4a95` | `QUAL/evidence/20260715_r33_clean_av/eval_verifier.json`, `e4d2e2e9af60bb9ece5ee38f0ee7dfa2eedea4004be74b9788a7a77b28e8f82e`, passed | validation (exploratory) |
| 1.2 | AR recovers direction from teacher text | teacher directional MSE 0.2817034125328064, cosine 0.8591482639312744, FVE-NRM 0.5845335363401372; raw MSE 8.537784576416016; centered raw R2 −0.20169644255144026; shuffled 0.9688876867294312, mean 0.6780412793159485, source_context 0.30125200748443604, source_raw 0.08324814587831497 | validation / 512 | `QUAL/evidence/20260715_r33_clean_ar/eval_iter_0001291_v512_t512_winrates_report.json`, `d3a95611265cbf22b0e7912bfda502762fc2f77fc8f1bf94d4243ae5096e3ca1` | `QUAL/evidence/20260715_r33_clean_ar/eval_verifier.json`, `1e62f1118fabf2493e31ec3e00ea4aa21170e5a12f4094cc0c71c919c31f2518`, passed; `raw_magnitude_claim_supported=false` | validation (exploratory) |

## 2. End-to-end round-trip claims

| # | Claim | Exact metric (value) | Split / rows / families | Report (path, SHA-256) | Verifier (path, SHA-256, verdict) | Class |
|---|---|---|---|---|---|---|
| 2.1 | AV-text → AR directional reconstruction, validation | primary directional MSE 0.3070042379760435; teacher 0.3047138391504298; raw MSE 9.449078796932245; centered raw R2 −0.32658628172707593; norm ratio 1.532614490080084; closed/usable 1.0/1.0 | validation / 512 / 250 | `QUAL/r33_clean_sft_roundtrip/validation_roundtrip_report.json`, `948460b54800a29480e3498e8549f6145646f9f4bb0bb89fe4eca4c49e573c25` | `QUAL/r33_clean_sft_roundtrip/validation_roundtrip_verifier.json`, `e1adedd40c25bb30be3907192eb5c298a213289295941a908572f4927e454ae8`, passed; claim scope `directional_av_to_ar_reconstruction` | validation (exploratory) |
| 2.2 | Same, exploratory test | primary 0.3192247649882203; teacher 0.3026366898913825; raw MSE 9.647147982011639; centered raw R2 −0.33537367599208356; norm ratio 1.5289557933441031; closed/usable 1.0/1.0 | test / 512 / 255 | `QUAL/r33_clean_sft_roundtrip/final_test_roundtrip_report.json`, `4802cd70b172abe78ebef903b38ebc982335abfdc30f87a65f0470d3903e3123` | `QUAL/r33_clean_sft_roundtrip/final_test_roundtrip_verifier.json`, `f3823bd334977c58da5bafdf67aefe899b2fcd41e788228f2573120bd0e6a71e`, passed | test (exploratory) |
| 2.3 | All controls beaten, family-clustered (test) | control−candidate margins: mean 0.361806 CI95 [0.352926, 0.370762] wins 0.998046875; av_mean 0.522112 [0.511974, 0.532006] 1.0; av_none 0.536178 [0.524715, 0.548039] 0.99609375; av_shuffled 0.645884 [0.634535, 0.657831] 1.0; av_zero 0.663983 [0.654222, 0.674222] 1.0; all sign-flip p = 9.99990000099999e-06 (MC floor); unit `content_family_id`, 255 units | test / 512 / 255 | same as 2.2 (CIs recomputed by verifier from rowwise arrays) | same as 2.2 | test (exploratory) |
| 2.4 | Same, validation | mean 0.36197869667875704 [0.3519201072139356, 0.37243960610694876] wins 0.99609375; av_mean 0.5272687000241001 [0.5164387289130028, 0.5385450298375675] 1.0; av_none 0.5298305160276751 [0.5180428382893946, 0.5430307653240594] 0.998046875; av_shuffled 0.6590958460172425 [0.6466787300674407, 0.6719463654427031] 1.0; av_zero 0.6684007596085646 [0.6570486263360339, 0.6797505502599274] 1.0; p = 9.99990000099999e-06; 250 units | validation / 512 / 250 | same as 2.1 | same as 2.1 | validation (exploratory) |

## 3. Functional reinjection claims (stored-snapshot counterfactual)

| # | Claim | Exact metric (value) | Split / rows / families | Report (path, SHA-256) | Verifier (path, SHA-256, verdict) | Class |
|---|---|---|---|---|---|---|
| 3.1 | Candidate ≈ teacher functionally; both ≫ controls (validation) | means: candidate KL 1.0883452449490907, JS 0.15501706278645808, Pearson 0.9015696287819976, top-10 0.6226562499999999, top-50 0.6378515625; teacher KL 1.199089; stored_gold KL 0.002818; mean 3.996430; zero 6.163884; shuffled 9.765375 (78 rows). Candidate-vs-teacher family CIs all include 0 | validation / 512 (shuffled 78) / 250 clusters | `QUAL/r33_clean_sft_roundtrip/validation_functional_report.json`, `4f6116db46de973c1185b487b5da76c43ebf4c7fe7509a6b09fb06ba2f57f9ff` | `QUAL/r33_clean_sft_roundtrip/validation_functional_verifier.json`, `efe2ab8dc8634253adcae2f550859eb3ee69d8c75381a1b78fa700cffc584253`, passed; claim scope stored-snapshot | validation (exploratory) |
| 3.2 | Same, exploratory test | candidate KL 0.949545, JS 0.152073, Pearson 0.907847, top-10 0.625586, top-50 0.639336; teacher KL 0.970104, JS 0.145112; stored_gold KL 0.002860; mean 4.124133; zero 6.297471; shuffled 9.528919 (88 rows); cand-vs-teacher KL Δ +0.02118268495886979 CI [−0.10322676062485685, 0.14967010221340588] | test / 512 (shuffled 88) / 255 clusters | `QUAL/r33_clean_sft_roundtrip/final_test_functional_report.json`, `8cd9324fbe6348f9c33f3be137cb63d2c0ac91afd16ab29ebc787abaf7f944ab` | `QUAL/r33_clean_sft_roundtrip/final_test_functional_verifier.json`, `302a2f4da77820d5352e4d3322e98cfc12b46ee411628711d29ec38812e80d7f`, passed | test (exploratory) |

Method note: AR-derived vectors are rescaled per-row to the stored gold norm
before reinjection (`rescale_direction`), so functional results assess
direction under an oracle norm; reinjection identity gate passed
(`identity_passed=true`, 512 rows) on both splits. The functional `shuffled`
control is a **within-document rotation** of stored gold vectors
(`within_document_shuffle`; only rows whose document has ≥2 selected rows,
hence 78/88 rows) — a deliberately harder same-document control that differs
from the measurement contract's cross-family shuffled definition used by the
AV and round-trip evaluations (see open issues).

## 4. Independent-AR replication claims (validation-only)

| # | Claim | Exact metric (value) | Split / rows / families | Report (path, SHA-256) | Verifier (path, SHA-256, verdict) | Class |
|---|---|---|---|---|---|---|
| 4.1 | Independent init is genuinely independent | 16/16 independence checks passed; canonical init-manifest SHA-256 `34e863f756e0749ca19fc8c138b7bd71b5da69c907ee42ad021517542e5c8941` (canonical-JSON hash, ignores only `value_head.before_sha256`) | n/a | `IND/critic_initialization/critic_initialization.json`, `a6f50b01e6686fbbb274dc20113616a137e7caa6ec25bd9e170710c280912583` | `IND/critic_initialization/critic_initialization_verification.json`, `4639285fea694f7f850c766b31d8ddea4e2b2bdd61c710f3b2a4cdd2109fb6e3`, passed | supporting |
| 4.2 | Independent AR component matches primary | teacher directional MSE 0.28616863489151, cosine 0.8569157123565674, FVE-NRM 0.5779480636043055; raw MSE 9.082671165466309; centered raw R2 −0.27838949739266483; wins: shuffled 1.0, mean 0.990234375 | validation / 512 | `IND/run_artifacts/eval_iter_0001291_v512_t512_winrates_report.json`, `368c84cad8bb0b8a7235b1a1e96c862b69d33be72d49c1f07a563e62d0be65aa` | `IND/component_eval_verifier.json`, `461f92f9968b10d24e08f8d815c0ab0ee5154258b6342ebcc48924c109bd7009`, passed; `raw_magnitude_claim_supported=false` | validation (exploratory) |
| 4.3 | Cross-critic replication on frozen AV text | AV-text directional MSE 0.3109634728220453; teacher 0.3085329510257789; centered raw R2 −0.39986470045388844; closed/usable 1.0/1.0; control CIs: mean [0.348206, 0.368142] wins 0.998046875; av_mean [0.510652, 0.532637] 1.0; av_none [0.511522, 0.535871] 1.0; av_shuffled [0.640363, 0.665274] 1.0; av_zero [0.651312, 0.673726] 1.0; p = 9.99990000099999e-06 | validation / 512 / 250 | `IND/validation_roundtrip_report.json`, `6f0829a61b03ac584b109c1c7a54f689f0a14b61f7fee803afd3d5ca29bf552b` | `IND/validation_roundtrip_verifier.json`, `dd3de6e1dd10f0f64c25e23ff3152638c9cf46785133dfb6c0466c353e434331`, passed | validation (exploratory), cross-critic |

## 5. Fidelity, magnitude, subgroup, exposure (bounding results)

| # | Claim | Exact metric (value) | Split / rows / families | Report (path, SHA-256) | Verifier | Class |
|---|---|---|---|---|---|---|
| 5.1 | Strict fresh-forward identity FAILS; alignment high | 64/64 rows violate the joint tolerances (rel-L2 ≤ 0.01, 1−cos ≤ 1e-4, abs ≤ 0.01; 62/64 on rel-L2 alone); per-row `full_vs_stored` rel-L2 mean 0.03132927212573122, max 0.18598312139511108 (recomputed from the `rows` array; equals `fidelity_assessments.max_observed_relative_l2`); the report's aggregate `activation_fidelity` block separately records rel-L2 mean 0.03140473198970221 / max 0.19598004516026946 — the two in-file max fields disagree by ~0.01 and the aggregate max does not match the per-row array (flagged in open issues); cosine mean 0.9991419776016195, min 0.983146454872643; stored-predicts-fresh centered R2 0.9972547944645127; repeat-forward and full-vs-extraction deltas exactly zero. The non-mb8 companion run OOM'd (`activation_fidelity_validation64_runner.json` returncode 1), so the mb8 JSON is the only fidelity artifact | validation / 64 | `PUB/activation_fidelity_validation64_mb8.json`, `5a457245d196a7f4e6ed60855bfc6cb6120e139d14fc79351198f7d9fc26c999` | self-reporting fail-closed diagnostic (`passed=false` on strict identity) | negative / diagnostic |
| 5.2 | One validation-fitted scalar repairs most raw error | selected origin scalar 0.5606042252919684 (fit: validation teacher, raw-MSE criterion); AV validation raw 9.446684620944241 → 3.648805524155073, centered R2 −0.32625015558381953 → 0.4877325656265995; AV test raw 9.647147982011639 → 3.7703528171163097, R2 −0.33537367546872665 → 0.47810172409560125; test family CI for raw-MSE improvement [5.634403774563392, 6.123574335381406], candidate better fraction 1.0; directional MSE unchanged | validation fit; applied to validation + test / 512 each / 250, 255 clusters | `PUB/roundtrip_magnitude_calibration_report.json`, `facad9f4b764eb7f1c1f50f95389b8580c1ee8c72ce7d703c5f679c41630923a` | inputs bound to gate-passed immutable caches (`PUB/validation_roundtrip_magnitude_cache_report.json` `92d222dd00ca9eb707336f3e55b5efda93cb6b6e2089243364614d57f3018ea2`; `PUB/final_test_roundtrip_magnitude_cache_report.json` `a1b9f67c42b4323f199b21b4f78efbfc1d9fa1ea375944e00feeb29d27a6b4e0`) | post-hoc (report self-labels `exploratory_posthoc_exposed_test`) |
| 5.3 | Subgroup robustness | 13 realized groups per split (4 source-length + 4 target-length + 4 activation-norm quartiles + 1 collapsed family-frequency bin; docs' "16 bins" counts requested bins); weakest test group = lowest activation-norm quartile (121 rows / 99 families): `av_real_directional_mse` 0.3700773988427564, `calibrated_centered_raw_r2` 0.4156062127347079; minimum control CI95 lower bound across test groups 0.3190440820712961 (> 0, at `target_activation_norm[0]`/mean) | validation-fit edges; both splits / 512 each | `PUB/roundtrip_subgroup_audit_report.json`, `fd24360bfcc4ff51538b2b21f5d1b931befa7b31ee0af45551701a95835c972c` | self-labelled `exploratory_posthoc_exposed_test` | post-hoc |
| 5.4 | No in-corpus confirmatory boundary exists | 136 exposure sources; 28,665 unique prior documents; unmapped = 0; test-forbidden families = 5,009 of 5,009; `passed=false` (split infeasible: test would be empty) | corpus-wide | v6 report inside `PUB/r33_exposure_audits_v4_v6.tgz` (tgz `6a0cfc714f4d23ca3c011b181f9a40666c714e73f5c2b1ec9d0e87c7af16fa4f`): extracted member SHA-256 `373e2988b32f2e4e68e2d7644a8b77430829ea7a5d5e2f6b72f870caa89d088b`; inventory `c193f2efb7c8414f4f8a6a12ab2051e633b891107f92c2971535164128f9aabb`; joint manifest `9d68a894e763ed533fba11016ebc8b8f05c0d4a39443585196e7cd24ebffbc20` | fail-closed by design | negative (blocks confirmatory claims) |
| 5.5 | No external teacher table exists in-project | 63 candidate tables, 53 usable, 0 external; max numeric doc suffix 38161 | corpus-wide | `PUB/teacher_corpus_external_boundary_inventory_v2.json`, `8b7b55b61d3cafbd3208493d4128ee8ec08de105940732e263eccf7d248fa27f` (v1: `187d7ce4c048f1ed2eae667395abd923eb9bec1d71d1ef029b98171b8b24c0a5`) | — | negative |
| 5.6 | July-16 scorer replay drifts at 4th decimal (validation) | cache rescore: validation primary directional 0.30696873336933334 vs July-15 report 0.3070042379760435; test values bit-identical across dates | validation / 512 | 5.2 cache reports (above) vs 2.1 report | — | diagnostic (report-version disclosure) |

## 6. Quality, privacy, compute, packaging

| # | Claim | Exact metric (value) | Report (path, SHA-256) | Class |
|---|---|---|---|---|
| 6.1 | Structural quality of generations | 100-row source+teacher-grounded panel: 0 automatic structural flags; panel `PUB/teacher_grounded_qualitative_panel.json` `4f5d61486330b1104dd0a256ea185d8c1c99512ee9ff4731f8135305924f81c8`; screen `PUB/teacher_grounded_qualitative_auto_review.json` `33de2720d96bda3f663318be7fe8c10765740a3d603bcdf76682ca23661297da` | (see left) | automatic only; human semantic review pending |
| 6.2 | Release-text privacy triage | 1,024 generated rows: 0 configured sensitive findings, 0 source-copy failures; panel max contiguous source match 5 words; 14 phone-like patterns in source excerpts only | `PUB/release_text_privacy_memorization_audit.json`, `00e501ff644483e614d0b60071f726f33575c95ba8b81d5c6b59c4bd79d13419` | automatic triage only |
| 6.3 | Selected-run compute | primary AR 4×H100 NVL, 3.8867 h, 15.5467 GPU-h; primary AV 8×, 13.4608 h, 107.6867 GPU-h; independent AR 4×, 3.8781 h, 15.5122 GPU-h; total 138.7456 GPU-h; 1,291 logged optimizer steps each; explicit exclusions listed in-file | `PUB/compute_accounting.json`, `7bde74be3a874d2ae305463ca8da211c069ce0bf1001802b6bdf7ab091fd7238` | exact for selected runs; not project-total |
| 6.4 | Training curves | 3,873 optimizer steps across the three runs | `PUB/training_metric_curves.json`, `7d9c22b989c594e546ec08648d0319c37caad69ad502d2badb635d41706c42a6` | supporting telemetry |
| 6.5 | Public bundle candidate (NOT final) | archive `artifacts/publication/r33-clean-sft-public-release-candidate-20260716.tgz` SHA-256 `3eb8e64ed0d9d61ed2d6b0694fbaf96b99051a63f2ce1a6c99372d93832e573a`, 6,859,370 bytes, 496 files, tree `df175c5f61cefbfc1a02451a7bd242ba69e1cb602cdd97ca4b8bd8fe9c263b77`; security audit `PUB/public_bundle_security_audit.json` `a130d8e4295a06d0372bd0d920d9dfd0c8f7649710652939b33dbc764089f096` (0 failed findings); attestation `PUB/public_bundle_archive_attestation.json` `89ffa410316201bf36f055bb7badf82292a43e1394d626cf978c0987b0d2c77a` (`weights_included=false`, `legal_clearance_granted=false`). **Defect:** ships obsolete exposure report `evidence/family_boundary/r33_confirmatory_family_split_report.json` (staged SHA-256 `682aeec283477821a118d4531e9f133d44dc45c4f7080de3d63980ecae604ad1`; source copy `bd9cc5d9a15faaeb4c94190f68d7b03f77020315b20c3218ee5df286cfc7f30f` inside `QUAL/../confirmatory/r33_content_families_v2/`) with `unmapped_prior_document_count=833`, 130 sources, 3,083 forbidden families, split summary 4,612/356/41 families (contradicting the frozen 4,504/250/255 split shipped beside it), `passed=false` — while the authoritative v6 JSON (row 5.4) is absent from the archive. The bundle's own staged docs/configs cite the v6 numbers, so the archive contradicts itself. The **current** builder config `configs/nano_release/r33_public_bundle_candidate.yaml` (SHA-256 `03ca4a47b9046151657fdaa93a90263a055c8420d08f10595aeb63906dabfa7b`, line 89) still stages the obsolete source file, so a naive rebuild reproduces the defect | (see left) | packaging defect — do not describe as final/self-contained |
| 6.6 | Internal source snapshot not public-clean | 680 files; 31 files with internal references (183 local-home, 76 internal-S3, 10 endpoint, 3 cluster-hostname findings); `automatic_gate_passed=false` | `PUB/source_bundle_security_audit.json`, `bf697acdad268e219cfd729428233b1259b07aea988529e8c7e4bb01b8a62c42` | negative / packaging |

## 7. Current validation-only online-RL evidence

| # | Claim | Exact metric (value) | Report (path, SHA-256) | Class |
|---|---|---|---|---|
| 7.1 | The selected family-clean online-RL AV+AR pair materially improves matched round-trip reconstruction on the declared validation boundary | 122 rows / 122 independent families; max 384 new tokens; matched protocol `fcc431ec4450adb8817cd946d6c194fa2a45b53b0c6c42c8682c1e9f12f94d4d`; directional NMSE `0.309055 -> 0.224386` (`0.084669` absolute, `27.4%` relative); raw MSE `9.5523 -> 7.2665` (`23.9%` relative); generated text has lower reconstruction error than teacher on `103/122` RL rows vs `62/122` SFT rows (not a semantic-quality judgment); RL close `121/122`; all controls and both gates pass | remote `publication/r33_online_rl/internal_hero/a3e5_u342/eval384_matched_v122/{sft_roundtrip_report.json,rl_roundtrip_report.json,eval384_chain.log}`; local synchronization and file SHA-256 values pending | strong protocol-matched validation evidence; pair-level effect because both AV and AR differ; report status `confirmatory`, but not a sealed publication test |

The checkpoint pair is
`publication/r33_online_rl/internal_hero/a3e5_u342/{actor,critic}/iter_0000342`.
It was trained from the qualified clean R33 SFT pair for 342 online-RL updates
over approximately 43 hours, using 24 prompts x 8 rollouts per update
(`65,664` generated responses total) on a 4 actor / 3 critic / 1 rollout GPU
allocation. The 384-token evaluation was performed after training on those
retained weights.
This result supports an online-RL improvement over matched SFT on this boundary.
It does not support a final test-set claim or R33-over-R27 superiority.

## 8. Historical / invalidated (must not be cited as current results)

| # | Item | Exact metric (value) | Evidence | Class |
|---|---|---|---|---|
| 8.1 | July-8 RL headline (INVALID) | 30.97% / 32.34% relative directional improvement vs mixed-protocol SFT baseline; audit-corrected fair estimate ~23.6% / 22.5% on the uncorrupted half | `docs/reviews/2026-07-08-r33-rl-hero-publication-audit.md`; registry entry `r33-corrected-k3-hero-lr1e5-update342-resume228-retry3` (`publication_valid: false`, reason `stored_activation_identity_failure_and_exploratory_test_exposure`) | invalidated — do not cite |
| 8.2 | Protocol-matched salvage rescore (internal only) | primary-critic relative directional improvement ≈ 0.2262 (validation) / 0.2032 (test); independent-critic 0.2262 / 0.1989; win rates 83–87%; gain-transfer ratios 0.9993 / 0.9793; length-matched gain ≈ 0.13–0.16; all on 512 rows, 214/218 families, protocol SHA `5677d491a812baecb9fa21829866de7eb3750f7cdb0273ed3038123a3119a381` (registry records primary 0.22638/0.20308 from the strict-baseline variant; per-report values differ in the 4th decimal by baseline variant) | `artifacts/runai_evidence/20260709_r33_publication_remediation/r33_protocol_matched_salvage/r33_retained_salvage_cross_critic_gate.json`, `379746492a3a04bb4424ed5fd148adcaf0228dc7941c6af4c02bda0eb5da8adb` (+ hero/sft report JSONs in same dir) | internal salvage evidence on superseded lineage — not publication evidence; no RL claim |
| 8.3 | v4 evaluation-only exposure audit (superseded) | 130 sources, 9,077 unique documents, 0 unmapped; would have left 242 candidate families | v4 members of `PUB/r33_exposure_audits_v4_v6.tgz`: report `dc53092a4739b8462dd7a6e9f9cb769a6986031ffc5a3f2e35640ca5178204b8`, inventory `32b29a4bbafaec7bdc0ec2ea66ecdf1bee6e4e20bdb5986bfc5b76d661419b30` | superseded by v6 (5.4) |

## Cross-checks performed

1. All SHA-256 values in sections 1–4 match the hashes bound inside the
   pair manifest (`37166ce7…3edb`) and inside each verifier's
   `report_sha256` field.
2. Verifier CIs were recomputed by the verifiers from rowwise arrays with
   `independent_unit = content_family_id` (250/255 units); bootstrap
   10,000 resamples seed 20260709; sign-flip Monte-Carlo 100,000 samples
   seed 20260709 (reported p 9.99990000099999e-06 is the add-one floor).
3. Docs-vs-artifact discrepancies found and resolved in the drafts:
   validation centered raw R2 −0.326250 (July-16 cache rescore) vs
   −0.326586 (July-15 report) — both preserved, disclosure in 5.6;
   "16 subgroup bins" (docs) vs 13 realized groups (report) — 5.3;
   "833 unmapped" exists only in the obsolete bundled v2-era report and
   nowhere in the current audit chain — 6.5; AV train table
   `train_padded.parquet` has 247,872 rows vs 247,865 manifest rows
   (padding rows; noted in open issues).
4. Corrections applied on the 2026-07-16 re-verification pass (a second,
   independent read of every JSON above): row 5.1 previously quoted a max
   rel-L2 of `0.18598004516026946`, a value present in **no** artifact field
   (a conflation of the aggregate-block max `0.19598004516026946` and the
   per-row/assessment max `0.18598312139511108`); row 5.3 previously quoted
   `0.3700769722805304` / `0.4156057514683159` for the weakest bin, values
   present in **no** artifact field (the JSON records `0.3700773988427564` /
   `0.4156062127347079`). Both rows now quote field-attributed values. All
   other full-precision numbers in this table were re-verified byte-exact
   against their artifacts on this pass.
