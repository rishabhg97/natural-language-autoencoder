# Nano30B NLA Super Audit — 2026-06-10

Independent super audit of the Nano30B Natural Language Autoencoder pilot:
reference faithfulness, Nano/Nemotron-H adaptations, dataset/split validity,
training and eval code, configs, and run records. Audited from primary sources
(code, configs, sidecars, eval JSONs, the preserved RunAI evidence archive, and
the local sha256-verified copy of the 275,396-row source table). No prior audit
reports were used. All local verification was read-only except writes to `/tmp`.

Method: one auditor pass over the core NLA contract files plus four parallel
sub-audits (datasets, eval, training tooling, configs). Every finding rated
Critical or High below was either produced or independently re-verified by the
lead auditor (re-running code, re-parsing parquets, re-applying patches); the
verification matrix in §7 lists exact commands. Sub-audit-only findings are
marked with their source.

---

## 1. Executive verdict

**NOT CLEAN. Do not launch the next R33 hero attempt on the current data, code,
or gate.** Three independent critical blockers, any one of which is
disqualifying:

1. **The dataset's doc-level split is silently broken by duplicated documents
   (C-1).** 66.7% of source docs (18,430/27,647) are byte-identical text under
   distinct `doc_id`s — a stage-0 shard re-streaming bug (duplicate doc-index
   offsets are exactly multiples of the 5,000-doc shard size, up to 6 copies
   per text). Under the production seed-42 doc split, **65.9% of validation
   docs and 78.8% of the docs covering the bounded 512-row eval window have a
   same-text twin in train**. Every heldout AR/AV number in the current
   decision state — including the full275k teacher NMSE `0.2776` and the R27
   fallback `0.441` — is leakage-inflated to an unknown degree. This is
   independent of, and additional to, the known packed-boundary training
   contamination.
2. **The packed-boundary remediation does not work (C-2).** The remote-code
   patcher (`nla/remote_code_patches.py`) was tested against the only real
   snapshot of Nano's `modeling_nemotron_h.py` in this repo: it rewrites the
   two Mamba-kernel kwargs to `seq_idx=seq_idx` while failing to add `seq_idx`
   to any signature (every plumbing regex misses the real multi-line annotated
   defs) → `NameError` at the first Mamba forward in **both training and eval**
   fast paths. It is also non-idempotent (a second pass over a file its
   patterns do match produces `def forward(..., seq_idx=seq_idx)` →
   `NameError` at import) and runs by default, in place, on shared checkpoint
   dirs at every actor init/load/save — the second queue item against the same
   critic init bricks it. Separately, Miles patches 0001/0003/0004 are
   malformed and cannot be applied to a fresh checkout, so the RunAI code
   state is not reproducible from this repo.
3. **The round-trip promotion gate can pass vacuously (C-3).**
   `build_gate_summary` uses `primary <= value - margin` with default margin
   `0.0` and `all(...)` over filtered controls: an injection-dead AV model
   (identical greedy text for every control → identical NMSE) **passes**, and
   a report with no control variants at all **passes**. The gate also feeds
   raw `<explanation>`-tagged generations to a critic trained on bare text,
   computes no parse/closure rate, and never checks row identity against the
   R27 baseline — all required by `docs/runs/r33_gate_matrix.md`.

What this does *not* mean: the architecture adaptation work is largely sound.
Boundary indexing (R33 = `hidden_states[33]`, critic kept at K+1=34 blocks) is
internally consistent and identity-checked; the injection/sidecar/normalized-MSE
contract faithfully matches the vendored reference (eval NMSE = `2(1−cos)`
verified to fp32 precision against the preserved eval JSON); the documented
metric values match their primary artifacts exactly; W&B-offline is enforced at
four layers; and the new step-0 packed-vs-padded assert is a sound design.
The pilot's problem is not the science design — it is that the dataset premise
(`doc_id` ⇔ unique text) is false, and the remediation/gate tooling shipped
today is unexercised and broken in ways the string-grep test suite cannot see.

**Decision impact on the R33-vs-R34-vs-R27 layer choice:** probably survives.
The 20k layer probes shared the same corpus and the same leakage, so the
*relative* ordering (R33 AR ≫ R34 AR; R33 ≈ R34 AV) is plausibly unaffected.
But absolute targets ("teacher NMSE in the 0.25–0.30 band") and all promotion
thresholds must be re-derived on deduplicated data.

---

## 2. Severity-ordered findings

Classes: `confirmed bug` / `likely correctness risk` / `missing check` /
`stale config` / `doc inconsistency` / `harmless detail` / `intentional
adaptation`. Every Critical/High item states its evidence.

### CRITICAL

**C-1 — Duplicated documents defeat the doc-level split; all heldout metrics are
leakage-biased.** `confirmed bug` (stage-0 lineage, propagated through R33).
Evidence (reproduced by lead auditor on
`runs/introspection/ar-r27-r30-fullscan-20260528T234403Z/handoff/R_27/ar_sft.parquet`,
275,396 rows):
- Keying each doc by its longest prefix's first 300 tokens: 18,430/27,647 docs
  (66.7%) are duplicates of another doc_id; group sizes 2–6; **every** pairwise
  doc-index offset within a duplicate group is a multiple of 5,000 (5 distinct
  offsets observed, all ≡ 0 mod 5000) — i.e. shard re-streaming in the original
  R27 stage-0 extraction (shards `start10500_len5000`, `start15500_len5000`, …
  visible under `runs/introspection/ar-r27-r30-fullscan-20260528T234403Z/runai_activations/`),
  not natural FineWeb duplication. Example group:
  `HuggingFaceFW/fineweb:train:{10508,15508,20508,25508,30508,35508}`.
- Replaying the exact production split (`sorted(docs)`,
  `random.Random(42).shuffle`, 0.9/0.05/0.05 —
  [nano_av_materialize_splits.py:96-99](../../scripts/nano_av_materialize_splits.py),
  [verify_nano_miles_av_dataset.py:35-81](../../scripts/verify_nano_miles_av_dataset.py)):
  validation 911/1,382 docs leaked (65.9% of docs and rows), test 910/1,382
  (65.8%). The bounded eval reads the **first 512 rows** of the doc-contiguous
  validation parquet ≈ 52 docs, of which 41 (78.8%) have same-text train twins.
- Sub-audit additionally measured 11,365 cross-doc row pairs with identical
  (text, position) — i.e. **identical activation vectors** straddling doc_ids.
- The dataset gates could not catch this: `doc_overlap_count` compares disjoint
  slices of a deduplicated sorted list — it is **0 by construction**
  ([nano_av_materialize_splits.py:105-111](../../scripts/nano_av_materialize_splits.py),
  [verify_nano_miles_av_dataset.py:94-97](../../scripts/verify_nano_miles_av_dataset.py)).
  The registry's `split_doc_overlap: 0` (runs/registry/experiments.yaml:25) is
  true and meaningless.
Impact: the R33 full275k `0.2776`/`0.2767`, the 100k HPO table, the R27
fallback `0.441/0.437`, and AV real-NLL numbers are all upper-bound-of-quality
(lower-bound-of-error) estimates contaminated by train/heldout text overlap.
Fix: content-hash dedup (keep one doc per text group → ~13.3k docs, ~133k rows)
**before** any clean rerun; add a text-hash cross-split assert to both
verifiers; re-baseline R27 on the deduplicated heldout.

**C-2 — The Nemotron-H remote-code patcher is broken, non-idempotent, and
default-on; the Miles patch set is unappliable.** `confirmed bug`. Four
compounding parts, all reproduced:
- *Broken against the real source.* Running
  `patch_nemotron_h_source` on the repo's only real Nano modeling snapshot
  (`runs/introspection/ar-r27-datagen-dryrun-20260528T230649Z/nano_tokenizer/modeling_nemotron_h.py`,
  1,740 lines) yields `seq_idx_replacements=2, attention_mask_replacements=0,
  moe_replaced=False`: only the two kernel kwargs (lines 437, 495) become
  `seq_idx=seq_idx`, while **no `def` gains a `seq_idx` parameter** — the
  plumbing regexes
  ([remote_code_patches.py:74-120](../../external/natural_language_autoencoders/nla/remote_code_patches.py))
  expect single-line unannotated signatures, but the real file uses multi-line
  annotated ones (mixer fwd at :707, block fwd at :762). Result: `NameError:
  name 'seq_idx' is not defined` at the first Mamba forward — in the training
  branch (`mamba_split_conv1d_scan_combined`) *and* the eval branch
  (`mamba_chunk_scan_combined`). The MoE rewrite never matches (real `def moe`
  has type annotations vs the pattern at :128), and the router-fp32 `post_init`
  injection never anchors (`post_init` is inherited, not defined in the file).
  A "healthy-looking" `modeling_nemotron_h.py.nla_patch_report.json` is still
  written (`changed=true`).
- *Non-idempotent.* On sources where the signature patterns do match (e.g. the
  shape used in `tests/test_nano_audit_remediation.py:16-29`), a second pass
  rewrites the pass-1-added `seq_idx=None` defaults to `seq_idx=seq_idx` →
  `NameError` at import; `patch_nemotron_h_file` writes the corrupted file
  back (report: pass-2 `already_patched=True` yet `seq_idx_replacements=2`,
  `changed=True`). Reproduced exactly (§7, check 5).
- *Default-on, in-place, on shared state.* `NLACriticModel.from_pretrained`
  patches the **load** dir in place
  ([models.py:176](../../external/natural_language_autoencoders/nla/models.py));
  `save_pretrained` copies the already-patched file then re-patches the copy
  ([models.py:115,276](../../external/natural_language_autoencoders/nla/models.py));
  `prepare_critic_checkpoint` re-patches a third time
  ([prepare_critic_checkpoint.py:77,146](../../external/natural_language_autoencoders/nla/scripts/prepare_critic_checkpoint.py));
  Miles patch 0003 patches `args.hf_checkpoint` at **every actor init** unless
  `NLA_PATCH_NEMOTRON_REMOTE_CODE=0`
  (miles_patches/0003:19-20), and `nano_av_runner.py:458` pins the env to `1`.
  Net effect: the first run pass-1-patches the shared critic init (breaking it
  against the real source per the first bullet); on pattern-matching sources,
  the *second* run corrupts it at import level. Either way the HPO queue dies
  — loudly, but after burning a launch, and having **mutated a shared
  checkpoint dir** that must then be re-synced.
- *The patch set cannot rebuild the cluster state.* `patch -p1` fails on
  `0001_miles_nla_integration.patch` (1/47 hunks malformed — the remediated
  hunk itself has header `+7` vs body `+8`), `0003` (10/17 hunks), `0004`
  (3/6). Reproduced with a hunk-count checker and `patch` (§7, check 8).
  `tests/test_nano_miles_launcher.py` only greps patch text; nothing applies
  them. Consequence: "patched code on RunAI" is hand-edited state that this
  repo can neither reproduce nor verify; the LR-schedule fix (patch 0001) and
  the patch-hook (0003) live in diffs that do not apply.
Fix direction (§6 step 0): make patching an explicit one-time, idempotent,
verified step (marker → skip everything; per-pattern counters; assert the
intended counters > 0 for Nemotron-H; `py_compile` + import + 2-token forward
after patching; never patch on load/save), or abandon regex patching and vendor
a corrected `modeling_nemotron_h.py` per checkpoint with a content hash.
Regenerate Miles patches as `git diff` from the actual working checkout and add
an apply-test.

**C-3 — Round-trip gate logic admits vacuous/tied passes; required report
fields missing.** `confirmed bug` (verified by eval sub-audit executing the
module; pass logic re-read by lead auditor at
[eval_nano_av_ar_roundtrip_gate.py:403-453](../../scripts/eval_nano_av_ar_roundtrip_gate.py)).
- `primary_beats = primary <= value - control_margin` with default
  `--control-margin 0.0` (:520) → exact ties pass. Greedy decoding with dead
  injection produces identical text for `av_real` and every `av_*` control →
  identical NMSE → **gate passes**.
- `beats_all_controls = all(item["primary_beats"] for item in controls.values()
  if item["normalized_mse"] is not None)` (:438) → `all([])` is `True` when
  controls are absent. A report containing only `av_real` **passes**.
- Generated text is fed to the critic raw — including `<explanation>` tags and
  unclosed tags at the 96-token cap — while the teacher side is tag-stripped
  (:90-92, :359-365); the AR critic trained on bare text inside
  `<text>...</text>` ([nano_realdata_stage3_build.py:59](../../scripts/nano_realdata_stage3_build.py)).
  This biases `av_real` vs `teacher` comparisons (av-vs-controls and vs-R27
  stay internally fair).
- No parse/closure-rate metric exists in the gate script; no rowwise win rates
  vs R27; no row/dataset identity check against the baseline JSON (:425-431,
  :468-469) — three fields required by
  [r33_gate_matrix.md](../runs/r33_gate_matrix.md) §Required Round-Trip Report
  Fields. `--reuse-generated` trusts recomputed row offsets (no dataset
  fingerprint in the JSONL).
Fix: require positive margins (e.g. `--control-margin 0.02`), fail when any
control variant is missing or when splits are empty, `extract_explanation()`
on generated text before critic templating (score parse failures as failures —
report closure rate), add baseline row-identity (split-manifest hash) and
rowwise R27 win rates.

**C-4 — Packed-boundary contamination is real for Nano and currently has no
working mitigation for packed runs.** `confirmed bug` (mechanism) +
`likely correctness risk` (magnitude unmeasured). The real remote code has no
packed-sequence handling at all: `_update_causal_mask` (copied from Jamba,
real file :1526-1559) builds a plain causal mask (or returns `None` for FA2)
with no position-id block-diagonal detection; `_update_mamba_mask` (:1561-1570)
returns `None` whenever the mask is all-ones/absent; the block forward drops
`attention_mask` before the mamba mixer (:781-784, the genuine N4 issue); the
kernel calls hardcode `seq_idx=None` (:437, :495). The `use_cache=False`
mitigation in
[train_actor.py:652-674](../../external/natural_language_autoencoders/nla/train_actor.py)
relies on transformers' `masking_utils` packed detection, which this
checkpoint-local code never calls — it protected Gemma/Qwen, not Nano. So AR
SFT at `gb192/mb96` = one packed 96-sample `[1,T]` stream per rank-step with
full cross-sample attention and Mamba state bleed. This confirms the docs'
reclassification of all pre-fix results as scouting-only, and — because C-2
means the fix doesn't exist yet — **no packed rerun can be clean today**. The
new step-0 packed-vs-padded assert
([train_actor.py:832-841](../../external/natural_language_autoencoders/nla/train_actor.py),
tol=0.02, default-on for critic/mb>1) is sound and will catch this loudly for
AR; the AV/actor path has **no equivalence check** (`_is_critic_model` gate at
:832), and AV `dyn512` can still pack pairs of ≤512-token samples (cap admits
two short samples; Miles batcher behavior unverified locally — §5).

### HIGH

**H-1 — Queue tooling cannot read 10 of 13 checked-in queue YAMLs.**
`confirmed bug` (fail-safe direction). `VALID_STATUSES = {pending, training,
eval_running, complete, failed}` in
[nano_ar_hpo_queue.py:28](../../scripts/nano_ar_hpo_queue.py) /
[nano_av_probe_queue.py:27](../../scripts/nano_av_probe_queue.py) (and
`{pending, running, complete, failed}` in
[nano_ar_layer_sweep.py:35](../../scripts/nano_ar_layer_sweep.py)); the
remediation commit `0812f8f` stamped `cancelled` (+ pre-existing `blocked`,
`blocked_missing_dataset`) across queue files and extended only
`update_nano_job_docs.py`. Any operation, including `--status`, raises
`QueueError` (verified: `tests/test_nano_av_probe_queue.py::test_checked_in_r33_100k_queue_is_valid`
fails on `status 'cancelled'`). Also misrepresents history: the **completed**
full275k hero run is recorded as `cancelled` with reason "removal of pending
jobs" ([r33_full275k_hero_queue.yaml:21-24](../../configs/nano_ar/hpo/r33_full275k_hero_queue.yaml)
vs registry/eval evidence). Fix: add terminal statuses to the runners (one
set, shared), correct mislabeled completed items, or author fresh queues.

**H-2 — No clean-rerun configs exist; the selected hero config is probe-grade;
AV hero/100k configs silently run constant LR at a contradicted LR.**
`stale config` + `likely correctness risk` (config sub-audit; spot-verified).
- `r33_full275k_lr5e5_cosine_warmup25_gb192_mb96.yaml` is
  `experiment_class: tuning-probe`, `no_save_optim: true`,
  `require_optimizer_state_for_hero: false` (:3,:47,:48) — reusing it for the
  hero violates the exact-resume policy and fails the runner's
  complete-performance gates if relabeled without checkpoint-field edits
  ([nano_av_runner.py:221-228](../../scripts/nano_av_runner.py)).
- `configs/nano_av/hpo/r33_full275k_lr1e5_gb192_mb2_seq1152_dyn512.yaml` and
  `r33_100k_...dyn512.yaml` have **no** `min_lr`/`lr_decay_style`/warmup keys →
  the runner passes no schedule flags ([nano_av_runner.py:313-323](../../scripts/nano_av_runner.py))
  → constant or Miles-default LR, with the canary recorded "not applicable";
  and they pin `lr=1e-5` although the AV smoke winner was `1e-4`
  (runs/registry/experiments.yaml:90-94).
- Zero configs reference router metrics or packed-equiv markers; every R33 AR
  config carries `allow_packed_critic_training: true` (the guardrail is
  boilerplate, not a brake).

**H-3 — LR-schedule verification has holes on exactly the paths that matter.**
`missing check` (training sub-audit; key lines re-read). Patch 0001's
neutering hunk is gone (good), but: the canary
([nano_ar_hpo_study.py:263-281](../../scripts/nano_ar_hpo_study.py)) is
log-only (never fails a run), silently "passes" when `final_train_lr` is
unparseable or `min_lr >= lr`, and exists **only** in the AR queue —
`nano_av_probe_queue.py` records nothing, while AV cosine smokes route through
it. Plain `--load` resume of a pre-fix checkpoint restores the neutered
`max_lr==min_lr` scheduler state (only `--finetune` skips scheduler state,
patch 0001:90-98). The shell launcher (`run_nano_av_miles_fsdp2_sft.sh`) never
plumbs schedule flags at all — shell-launched runs are always constant-LR.
Whether the Miles FSDP backend honors `--lr-decay-style` end-to-end remains
unverified from this repo (Miles not vendored) — that is precisely what the
canary must prove on the first clean smoke, and why it must be made a hard
gate there.

**H-4 — Critic HF checkpoint saves and AV trial cleanup can leave no usable
weights.** `confirmed bug` (corruption path) + `intentional adaptation`
(cleanup policy, sharp edges):
- Every critic `save_model` → `save_pretrained` → `_copy_remote_code_files_from_config`
  re-patches the copied modeling file (C-2): saved critic HF dirs carry broken
  remote code (NameError on next load).
- The AV queue's `cleanup_dcp_model_after_eval: true` +
  `cleanup_converted_hf_after_eval: true`
  ([r33_100k_scaling_queue.yaml:8-9](../../configs/nano_av/hpo/r33_100k_scaling_queue.yaml),
  [nano_av_probe_queue.py:292-297](../../scripts/nano_av_probe_queue.py))
  delete **all** weights of a completed trial after eval — an HPO winner must
  be retrained; in-train pruning at `keep_last: 1`
  ([train_actor.py:935-944](../../external/natural_language_autoencoders/nla/train_actor.py))
  deletes every intermediate hero checkpoint with no `NLA_BACKUP_REMOTE`
  configured in queues.
- The DCP→HF conversion cleanup is **not** in a `finally`: a failed eval
  leaves a full fp32 HF tree (order-100 GB for 30B) on the Longhorn PVC
  ([nano_av_probe_queue.py:280-302](../../scripts/nano_av_probe_queue.py)) —
  the exact failure mode of
  [2026-06-longhorn-diskpressure.md](../incidents/2026-06-longhorn-diskpressure.md);
  a partial tree without `config.json` then blocks the unforced converter
  (convert_fsdp_to_hf.py:378-379).

**H-5 — Remediation evidence is unverifiable by construction.**
`missing check`. Nothing consumes
`modeling_nemotron_h.py.nla_patch_report.json` (rg over scripts/tests/runbooks:
zero consumers); `PatchReport.seq_idx_replacements` lumps kwarg rewrites and
all plumbing additions into one counter, so even a manual reader cannot
distinguish "fully plumbed" from "rewrote kwargs to an undefined name" (the
broken state in C-2). The remediation-pass tests are predominantly
string-presence asserts (`tests/test_nano_audit_remediation.py` greps for
`"cosine_sum"`, `"nla_value_indices"` etc.); there is no numerical equivalence
test for `segmented_moe`, no idempotency/compile test for the patcher, no
patch-application test. Additionally the audit-remediation doc's own
verification command (`python3 -m unittest tests.test_nano_audit_remediation ...`)
fails — `tests/` has no `__init__.py`; only the pytest form works.

### MEDIUM

| # | Finding | Class | Evidence |
|---|---|---|---|
| M-1 | AV/actor packed training has no runtime equivalence check, and `dyn512` does not structurally prevent two ≤512-token samples from packing; "verified Miles single-sample-overflow behavior" (configs/README.md:59) is not independently verifiable locally | likely correctness risk | [train_actor.py:832-841](../../external/natural_language_autoencoders/nla/train_actor.py) critic-only gate; Miles not vendored (UPSTREAM_PIN `radixark/miles@051cd15`) |
| M-2 | `segmented_moe` (N2) never applies to the real file (type-annotated `def moe` defeats the regex), and if it did apply it returns `topk_weights.dtype` (fp32) without the stock final `.type(hidden_states.dtype)` cast — a dtype change mid-stack; no equivalence test | likely correctness risk (latent) | [remote_code_patches.py:123-142](../../external/natural_language_autoencoders/nla/remote_code_patches.py); [nemotron_moe.py:32-47](../../external/natural_language_autoencoders/nla/nemotron_moe.py); real file :862 `return final_hidden_states.type(hidden_states.dtype)` |
| M-3 | Router-fp32 remediation (N3) is largely redundant — the real router already computes `F.linear(hidden.float(), weight.float())` in-forward (:908-918); storage repinning helps only bias rounding; post-FSDP fp32 survival behind `NLA_REPIN_ROUTER_FP32_AFTER_FSDP` (default off) is unproven | harmless detail / unproven | real modeling file; [audit_runtime.py:119-136](../../external/natural_language_autoencoders/nla/audit_runtime.py); [train_actor.py:391-394](../../external/natural_language_autoencoders/nla/train_actor.py) |
| M-4 | AR/AV split agreement is convention-only (same seed/fractions/row_limit ⇒ same doc split); cross-scale pairing (full275k AR + 100k AV) silently puts ~90% of AV validation docs in AR train; no manifest-equality check at gate time | likely correctness risk | dataset sub-audit; [nano_av_materialize_splits.py:91-99,142-147](../../scripts/nano_av_materialize_splits.py) |
| M-5 | Legacy launcher `run_nano_av_miles_fsdp2_sft.sh` performs no split (first-ROW_LIMIT slice straight to Miles; fractions recorded but unused, :97-151) — metric-bearing use would have zero heldout | likely correctness risk (if used) | dataset sub-audit; :106-134, :158-233 |
| M-6 | Bounded 512/512 eval = first-N rows of doc-contiguous splits ≈ ~51 docs, not 512 independent rows; fair for A/B, anti-conservative for CIs (bootstrap defaults off; report JSON records no seed/limits/dtype) | likely correctness risk (statistical) | eval sub-audit; [eval_nano_ar_miles_checkpoint.py:212-213](../../scripts/eval_nano_ar_miles_checkpoint.py); reproduced doc-count (52) in §7 check 10 |
| M-7 | AV eval `--injection-scale` default `"75"` is a CLI constant that happens to match configs; not read from the trained model's sidecar (and ≠ sqrt(2688)=51.8); manual eval of a non-75 run silently mismatches | likely correctness risk | eval sub-audit; eval_nano_av_miles_checkpoint.py:280, gate :513 |
| M-8 | Converter skeleton-fallback writes config.json without `auto_map` → eval silently runs builtin transformers Nemotron-H instead of checkpoint remote code; fall-through printed, not fatal | likely correctness risk (guarded by exact-match primary path) | eval sub-audit; convert_fsdp_to_hf.py:256-261, 304-351 |
| M-9 | Two stale tests encode the pre-remediation verbatim-copy contract and now fail (`tests/test_nano_critic_model_arch.py:70-116`) — they are, in fact, flagging the C-2 auto-patch-on-copy behavior; plus the AV-probe-queue test failure (H-1) | confirmed bug (test rot) | pytest run, §7 check 2 |
| M-10 | `_assert_reward_train_paths_agree` silently skips when n<4 and runs only at rollout 0 (never re-checked after resume); skip is print-only | missing check | [train_actor.py:140-143](../../external/natural_language_autoencoders/nla/train_actor.py) |
| M-11 | In-train prune (`ls \| head -n -keep_n \| xargs rm -rf`) can race the one-behind background push when `NLA_BACKUP_REMOTE` is set with `keep_n=1` (prune of iter_{N-1} while its gsutil copy runs) | likely operational risk (env-gated, unused in queues) | [train_actor.py:935-964](../../external/natural_language_autoencoders/nla/train_actor.py) |

### LOW / INFO (selected; full lists in sub-audit appendices)

- `<text>` extraction is find-first and would silently truncate explanations
  containing `</text>` — empirically 0 such rows in all 275,396 prompts
  (verified scan), but no guard exists
  ([nano_prefix_activation_extract.py:64-72](../../scripts/nano_prefix_activation_extract.py)).
- Scaling-path teacher join drops misses silently up to a 5% floor
  (nano_ar_r33_scaling_pipeline.sh:259-292); the exact-count verifier
  downstream catches net loss; the older layer-probe variant lacks both guards.
- Verifiers never cross-check sidecar `layer_index` vs the parquet
  `activation_layer` column; CLI `--expected-d-model` overrides rather than
  cross-checks the sidecar.
- AV verifier accepts empty explanation bodies (`<explanation>\n\n</explanation>`).
- Design-doc drift: design.md §0 says absent `injection_scale` defaults to
  `sqrt_d_model`; the code deliberately resolves absent→None and asserts
  ([config.py:176-183](../../external/natural_language_autoencoders/nla/config.py)) —
  code is right, doc is stale. Same-name `fve_nrm` uses the mean-norm baseline
  in eval but `rawvar` in training logs.
- `data_source.add_samples` comment still claims "get_samples deepcopies"
  (it shallow-copies post-E6); `summarize_nano_av_run._parsed_count` keys on a
  `parsed_explanation` field nothing produces (closure = substring heuristic).
- `RouterEntropyTracker._hook` does a `.cpu()` bincount per router forward —
  one blocking D2H per MoE layer per microbatch when `NLA_ROUTER_METRICS=1`
  ([system_metrics.py:168](../../external/natural_language_autoencoders/nla/system_metrics.py)).
- `tests/test_nla_system_metrics.py` lacks a `unittest.main()` guard; queue
  `--dry-run` is not side-effect-free (re-materializes splits, rewrites specs).
- Job-tracker top section stale (claims R33 100k "not launched"); registry
  lacks the AR 20k layer-decision probes; `execution_log.md` frozen at
  2026-05-26.

### What was checked and found SOUND (positive results worth keeping)

- **Boundary indexing**: R_b = output of block b−1 = `hidden_states[b]`;
  `"R33" → [33]`; manual `.backbone.layers` walk with per-block-type masks;
  no `norm_f`; identity-harness-validated
  ([nano_ar_layer_sweep.py:910-932](../../scripts/nano_ar_layer_sweep.py),
  [nano_extraction_identity.py:260-315](../../scripts/nano_extraction_identity.py)).
  Critic K+1 convention (`--num-layers 33` → 34 blocks, value head reads
  output of block 33) is the reference's documented design
  ([models.py:156-175](../../external/natural_language_autoencoders/nla/models.py)),
  pinned by `nano_ar_correctness_audit.py:39-41` and sidecars
  (`critic.extraction_layer_index: 33` in both the dataset sidecar and the
  critic-init model sidecar extracted from the evidence archive). Internally
  consistent end-to-end; document the dual convention (sidecar stores the
  boundary index; reference help text calls the same number a block index).
- **Reference faithfulness of the metric/normalization contract**: raw vectors
  in parquet (`norm: none` in real sidecars), normalization only at
  injection/loss/eval; eval NMSE ≡ `2(1−cos)` for normalized rows — verified
  numerically and against the preserved full275k eval JSON (teacher cos
  0.861217 → 0.277566 vs recorded NMSE 0.277565); `mse_scale = 51.846 =
  √2688` in the critic-init sidecar; both-sides normalization in loss and eval
  identical to `nla_inference.py:655-665`.
- **Documented metrics match primary artifacts exactly** (eval JSON in the
  evidence archive ≡ registry ≡ docs, all seven controls present, 512/512
  rows, win rates included). The numbers are faithfully recorded — they are
  just measured on a leaky split (C-1).
- **Injection path**: pure, vectorized, neighbor-checked, count-asserted hook
  with distributed-abort on mismatch; sidecar/tokenizer drift asserts at load;
  injection marker `々` (id 42019) with canonical neighbors verified per-row
  by the AV verifier. Suffix-anchored critic extraction with one-time suffix
  verification.
- **Mean/teacher-shuffled controls**: train-only mean fit (`train_targets`
  from the train parquet only), true derangement (verified over 1,400
  seed/count combos), within-split shuffles, tie-aware win rates.
- **W&B offline** enforced at spec, command, env, and launcher layers.
- **E1–E6 remediations are genuinely implemented** in the actor: local-shard
  grad-norm clipping (decomposable-L2-correct), value-head gather on selected
  rows, embedding-size-gated microbatch sync, vectorized injection, shallow
  sample copies, SFT log-prob skip.

---

## 3. Verification matrix

Commands run by the lead auditor (all from repo root; `/tmp/nla_audit` scratch):

| # | Check | Command (abbrev.) | Result |
|---|---|---|---|
| 1 | Documented unittest form | `python3 -m unittest tests.test_nano_audit_remediation ... -q` | **FAILS** — `ModuleNotFoundError` (no `tests/__init__.py`); doc bug |
| 2 | Full test suite | `python3 -m pytest tests/ -q` | **3 failed, 203 passed, 88 subtests** — failures: `test_nano_av_probe_queue` (status `cancelled`), 2× `test_nano_critic_model_arch` (remote-code copy now auto-patched) |
| 3 | Remediation subset | `python3 -m pytest tests/test_nano_audit_remediation.py tests/test_nano_ar_hpo_study.py tests/test_nano_av_runner_validation.py tests/test_nano_miles_launcher.py -q` | 27 passed, 62 subtests |
| 4 | Static checks | `git diff --check`; `python3 -m py_compile <11 listed files>` | clean / all OK |
| 5 | Patcher idempotency | double-`patch_nemotron_h_source` on the test's synthetic source; `exec` both passes | pass-1 OK; **pass-2 `NameError: name 'seq_idx' is not defined`** at exec; pass-2 `changed=True` |
| 6 | Patcher vs real source | run patcher on `runs/introspection/.../nano_tokenizer/modeling_nemotron_h.py` | `seq_idx_replacements=2, attention_mask=0, moe=False`; **no def gains seq_idx** → NameError at first Mamba forward; fp32 hook not wired |
| 7 | Real-file packed handling | read `_update_causal_mask` / `_update_mamba_mask` / block fwd / kernel calls | no packed detection; mask dropped before mamba mixer; `seq_idx=None` hardcoded |
| 8 | Miles patch integrity | hunk-count checker over all 6 patches | 0001: 1/47 malformed; 0003: 10/17; 0004: 3/6; 0002/0005/0006 clean |
| 9 | Metric identity | recompute `2(1−cos)` vs recorded NMSE from preserved eval JSON | teacher val: 2(1−0.861217)=0.277566 vs 0.277565 ✓; all 7 controls, 512/512 rows ✓; registry numbers ✓ |
| 10 | Duplicate docs + leakage | pyarrow scan of `handoff/R_27/ar_sft.parquet` (275,396 rows); replay `Random(42)` 0.9/0.05/0.05 doc split | **18,430/27,647 docs (66.7%) duplicated** (longest-prefix-300 key); all pairwise offsets ≡ 0 mod 5000; val leak 911/1,382 docs (65.9%); first-512-val-rows ≈ 52 docs, 41 leaked (78.8%) |
| 11 | Gate logic | read `build_gate_summary`; eval sub-audit executed module with tied/no-control reports | tie-pass and `all([])` vacuous-pass confirmed |
| 12 | Sidecar contract | extract critic-init + dataset sidecars from evidence archive | `extraction_layer_index: 33` both; `mse_scale=√2688`; `norm: none`; templates match reference suffix convention |
| 13 | Hero config grade / AV schedule fields / queue statuses | grep configs | `tuning-probe` + `no_save_optim: true` on hero family; AV hero has `lr: 1e-5`, no schedule keys; `cancelled`/`blocked` present |

Sub-audit-verified (not re-run by lead, methods stated in their reports):
derangement property over 1,400 combos; gate module execution on synthetic
reports; `patch -p1` failure reproduction; AR-eval mean control train-only
trace; 11,365 identical (text,position) pairs; per-file config tables.

Skipped (and why): no RunAI access from this session — packed-vs-padded live
check, Miles scheduler end-to-end behavior, dynamic-batcher pack/overflow
semantics, actual cluster Miles checkout state, current checkpoint-local
modeling file, and HF modules cache staleness are all **cluster-side
verifications** listed in §5. No training, deletion, or RL was launched.
W&B stayed offline (nothing was synced).

---

## 4. Correctness risks that remain unproven

1. **Magnitude of leakage inflation (C-1).** The 0.2776 → honest-NMSE gap is
   unknown until a dedup rerun; same for R27's 0.441 (the fallback must be
   re-evaluated on dedup splits, not retrained, since its checkpoint exists).
2. **Magnitude of packed contamination (C-4)** on final metrics: mechanism
   proven, effect size unmeasured. The step-0 padded-vs-packed ratio from the
   first post-fix run will quantify it.
3. **Miles scheduler end-to-end behavior** for `--lr-decay-style cosine` on
   the FSDP backend, and the cluster checkout's actual patch state (H-3, C-2).
4. **Miles dynamic-batcher semantics** for `max_tokens_per_gpu` (oversize =
   solo microbatch vs drop; whether two short samples co-pack) — decides if
   AV ever packed at all (M-1).
5. **Whether the RunAI checkpoint-local `modeling_nemotron_h.py` matches the
   May-28 snapshot** the patcher was tested against (it carries a hand-edit
   marker `# was seq_idx`). If it differs, C-2's NameError specifics may
   differ — but the patterns are fragile either way.
6. **Router fp32 storage survival under FSDP2 mixed precision** (M-3) and
   grad-norm parity for any replicated (non-DTensor-sharded) params.
7. **HF dynamic-module cache staleness** (`HF_MODULES_CACHE`) serving
   pre-patch code even after a dir is patched — needs a loaded-module hash
   check on cluster.

---

## 5. Training/eval optimization opportunities (ranked by expected impact)

1. **Don't ship `timing_debug: true` in active probe configs** — patch 0004
   brackets ~12 regions per microbatch with `torch.cuda.synchronize()`;
   at AV dyn512 (~48 microbatches/step) that is O(10³) forced syncs per step.
   Already rejected for complete-performance; flip the default to false
   everywhere and opt in explicitly. (Largest single wall-time lever on
   probes; the historical full275k run trained with it **on**.)
2. **Keep local-shard grad clipping** (E1, done): documented old global path
   was 86.56 s/step vs 8.98 s forward (runai_miles_fsdp2_integration.md:313).
   Use `global_clip` only for a one-time parity check.
3. **Dedup the dataset (C-1)**: ~275k → ~133k rows for the same unique
   information ⇒ hero epoch cost roughly halves (1,291 → ~620 steps at
   gb192). Correctness fix that is also the second-largest throughput win.
4. **Stop convert-then-delete per AV trial**: the DCP→fp32-HF conversion
   (order-100 GB IO) is repaid zero times and serializes train→convert→eval
   on idle GPUs. Convert once to bf16 (`--torch-dtype bfloat16` at conversion
   if supported, else cast at save like the critic path), eval, clean in
   `finally`; or eval directly from DCP via a loader shim. Also fixes H-4's
   disk risk.
5. **Skip re-materializing splits per queue item**: `prepare_run` re-reads the
   full parquet (99,570×2688 fp32) and rewrites 4 split parquets (+ an
   unconditional `train_padded.parquet`) on every item *and on dry-run*; cache
   keyed on (source sha, row_limit, seed, fractions).
6. **MoE**: the segmented-MoE rewrite currently never applies (M-2). If MoE
   FLOPs matter, fix the pattern + add an equivalence test + final dtype cast;
   a grouped-GEMM fork remains the bigger win later. Until then the stock
   per-expert `torch.where` loop is what actually runs — budget accordingly.
7. **Router metrics**: accumulate bincounts on-GPU and `.cpu()` once per
   `collect()` instead of per router forward (only matters when
   `NLA_ROUTER_METRICS=1`; keep it for smokes, off for heroes).
8. **AR eval batching**: batch 4 with right-padding is fine at 512 rows;
   don't increase without re-checking bf16 padding sensitivity (M-6 caveat).
9. Keep the already-implemented wins: SFT log-prob skip (~2× step), embed-dump
   skip (~2.2 s/step), `clear_memory` every 50, value-head gather (E2/E4),
   microbatch-sync gating (E3 — Nano embeddings 352M < 1B threshold ⇒ sync
   skipped).

---

## 6. Proposed clean R33 run plan

Ordered; each step gates the next. "1 GPU"/"2 GPU" refers to GH200/H100-class.
Wall-times are estimates from pre-fix runs (full275k AR completed overnight on
2 GPUs; 100k runs ~3–6 h; 20k probes ~1–2 h) — treat as planning numbers.

**Step 0 — Code fixes (local, no GPU).** Must all land before any launch:
- Patcher (C-2): make idempotent (marker ⇒ return unchanged), per-pattern
  counters, hard-fail for Nemotron-H when `seq_idx` plumbing counters are 0 or
  kwarg rewrites occur without signature additions, `py_compile` + import the
  patched file, **never** patch inside `from_pretrained`/`save_pretrained`
  (explicit `python -m nla.remote_code_patches <dir>` step only); restore
  verbatim remote-code copies (the two failing critic tests then pass as
  written). Alternative accepted: drop regex patching; vendor a hand-fixed
  `modeling_nemotron_h.py` with a recorded sha256 and a paired unit test.
- Regenerate Miles patches 0001/0003/0004 as valid diffs from the actual
  working checkout; add a CI test that applies all six against the pinned
  `radixark/miles@051cd15` in a temp dir.
- Gate (C-3): positive default margin (0.02), fail on missing controls/empty
  splits, `extract_explanation()` before critic scoring + closure-rate field,
  baseline split-manifest identity check, rowwise R27 win rates.
- Queues (H-1): add `cancelled`/`blocked`/`blocked_missing_dataset` to both
  runners' `VALID_STATUSES` (shared constant); fix the three mislabeled
  completed items; AV queue gets `--reset-active` parity.
- Verifiers (C-1): add content-hash (first-300-token prefix hash per doc)
  cross-split assert + duplicate-group report to both verifiers; verify the
  **materialized** split parquets, not a recomputed hypothetical split.
- LR canary (H-3): make it a hard gate for non-constant schedules in *both*
  queue runners (fail the item when `applicable && !passed`, or when the
  final-LR is unparseable on a cosine run).
- Acceptance: full pytest green (206+), including the two updated critic
  tests and a new patcher idempotency + apply-patches test.

**Step 1 — Dataset dedup + re-verification (CPU, ~1–2 h).**
From the existing 275,396-row table (no new extraction, no new teacher text):
group docs by longest-prefix-300-token hash, keep the lowest doc-index member
per group, drop rows of the other docs ⇒ expected ~13.3k docs / ~133k rows.
Write `r33_prefix_dedup_fullscan<N>` root + sidecars (same `layer_index: 33`,
`norm: none`); run both upgraded verifiers (must now include the cross-split
content-hash check); record exact N in the registry. Materialize splits once
(seed 42, 0.9/0.05/0.05) and pin the split manifest hash that AR, AV, and the
gate must all reference (M-4).

**Step 2 — Cluster preflight (1 GPU, <1 h).** On RunAI:
(a) snapshot + sha256 the live `modeling_nemotron_h.py`; run the fixed patcher
explicitly; require the new per-pattern counters all >0; `py_compile`+import.
(b) 2-token forward smoke through the patched critic init (catches NameError
class). (c) `NLA_ASSERT_PACKED_EQUIV=1` step-0 packed-vs-padded run on real
rows (must pass at tol=0.02 — this is the C-4 acceptance test). (d) one
50-step cosine canary run; assert `final_lr < 0.9·lr` from `train.log`
(H-3 acceptance). If (c) cannot pass because seq_idx plumbing is still
incomplete, fall back to **mb1** (no packing) for Step 3 and treat packed mode
as blocked.
Note: the existing critic init at
`/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-critic-init/nano-ar-r33-critic-init`
predates the patcher (created 2026-06-08) — verify its modeling file is
pristine before first use, and re-prepare it from the dedup sidecar if the
sidecar row_count/dataset_id is consumed anywhere downstream.

**Step 3 — AR dedup smoke (1 GPU, ~1–2 h).** New config
`configs/nano_ar/hpo/r33_dedup_smoke_20k_lr2e5_cosine_warmup20_gb192_mb96.yaml`:
first 20k dedup rows, lr 2e-5, min_lr 2e-6 cosine, warmup 20, gb192/mb96 (or
mb1 per Step 2 fallback), `timing_debug: false`, `NLA_ROUTER_METRICS=1`,
save/eval at final step only, bounded 512/512 eval with all 7 controls.
Promotion signal: run completes with packed-equiv pass + LR canary pass;
expect teacher NMSE **worse** than the leaky 0.38 — this sets the honest
baseline. Re-evaluate the existing R27 fallback checkpoint on the dedup
heldout in the same step (eval-only, no retrain) so the comparison target is
honest too.

**Step 4 — AR medium/hero confirmation (2 GPU).** New config
`r33_dedup_full_lr5e5_cosine_warmup25_gb192_mb96.yaml`, class
`complete-performance`, `no_save_optim: false`,
`require_optimizer_state_for_hero: true`, `keep_last: 2`, save/eval at final
(~620 steps for ~119k train rows at gb192; overnight). Start from lr 5e-5
(best pre-fix family) but accept the dedup smoke may shift the optimum; if
smoke teacher NMSE degrades >0.10 vs scouting, insert a 3-point LR re-probe
(3e-5/5e-5/8e-5, 100-step, 1 GPU each) before the hero. Promotion: beats the
re-based R27 teacher NMSE on identical dedup splits, all controls sane,
canary+packed-equiv evidence archived.

**Step 5 — AV LR confirmation (1 GPU, 2 runs × ~1 h).** New configs
`r33_dedup_av_20k_lr1e4_cosine_warmup5_gb192_mb2_seq1152_dyn1152.yaml` and a
5e-5 twin: **explicit** min_lr/cosine/warmup fields, `injection_scale: 75`,
response cap 1024, and — until M-1 is resolved on cluster —
`max_tokens_per_gpu: 1152` with `micro_batch_size: 1` so every microbatch is
provably single-sample (no packing exposure), accepting the throughput cost at
probe scale. Extend the AV queue to record the LR canary. Promotion: real NLL
ordering reproduces (1e-4 best), real-vs-shuffled/zero/mean gaps healthy,
generation closure rate reported.

**Step 6 — AV hero (2 GPU, overnight).** Winner LR from Step 5 on the full
dedup set, complete-performance class, optimizer state retained, `keep_last:
2`, S3 push of the final checkpoint before any cleanup; convert-eval with
cleanup-in-`finally`.

**Step 7 — Round-trip gate (1 GPU, ~1–2 h).** Fixed gate (Step 0) with
`--control-margin 0.02`, AV checkpoint from Step 6, AR checkpoint from Step 4,
R27 baseline gate-report regenerated by the same script on the same dedup
splits (M-4: assert split-manifest hash equality). Report must include closure
rate, av_real vs teacher vs all controls, rowwise win rates vs R27. **Only a
pass here, against the re-based R27, completes the milestone. RL stays
blocked until then.**

**Optional layer re-check:** only if Step 3's honest R33 numbers land within
noise of the re-based R27 — then rerun the 20k AR probes for R27/R33/R34 on
dedup data (3 × 1 GPU × ~1 h) before committing the hero. Otherwise skip; the
pre-fix ordering was same-leakage and likely valid directionally.

**Configs that must NOT be used as-is:** every existing
`configs/nano_ar/hpo/r33_*` and `configs/nano_av/hpo/r33_*` file (pre-fix,
leaky dataset root, probe-grade hero, missing AV schedule fields, packed
acknowledgement boilerplate); the two non-seq1152 AV layer probes; all `r27_*`
HPO configs; `r25_r51_20k_queue.yaml`; the legacy
`run_nano_av_miles_fsdp2_sft.sh` for any metric-bearing run (M-5). Keep them
for provenance; author the new `*dedup*` families above.

---

## 7. Artifact/checkpoint cleanup guidance (Longhorn)

Preserve (cheap, evidentiary):
- `artifacts/runai_sync/20260610T234644Z/` (local) and its S3 mirror — now
  also the canonical archive of pre-fix eval JSONs validated by this audit.
- All `eval_*_report.json`, `train.log`, `run_spec.yaml`, W&B offline dirs,
  split manifests, sidecars for any run referenced by the registry.
- The R27 fallback AR checkpoint (needed for the re-based eval in Step 3) and
  the R33 critic init (after Step 2 pristine-check).
- The May-28 `nano_tokenizer/modeling_nemotron_h.py` snapshot — it is the only
  local ground truth for patcher testing.

Prune / fix:
- Hunt orphaned fp32 DCP→HF conversion trees from failed AV evals
  (`find /workspace/interp/outputs -maxdepth 4 -name 'hf_iter_*' -o -name
  '*_hf_tmp*'` and compare against queue items marked `failed`) — H-4's
  cleanup gap means any failed eval left ~100 GB behind. After the Step 0 fix,
  cleanup runs in `finally`.
- Keep `keep_last: 2` (not 1) for heroes + S3 push of final and best
  checkpoints before queue cleanup; never enable
  `cleanup_dcp_model_after_eval` on a queue whose winner you intend to keep.
- Pre-fix R33 AR/AV checkpoint trees other than retained candidates: already
  removed per registry; do not re-sync them from S3 except for forensic needs.
- The leaky-dataset derivative parquets (`r33_prefix_fullscan275396` splits,
  `train_padded.parquet` copies under run dirs) become obsolete after Step 1 —
  delete the per-run split copies, keep one canonical copy of the source table
  + the new dedup root.
- Track PVC usage before each hero (incident doc protocol); temporary HF
  conversions are the known killer.

---

## 8. What would invalidate this audit

1. **A different cluster modeling file.** All C-2 specifics were proven
   against the May-28 snapshot (which carries a `# was seq_idx` hand-edit
   marker). If the live checkpoint-local `modeling_nemotron_h.py` on RunAI has
   materially different shape, the patcher might match there (or break
   differently). Step 2(a) settles this; the non-idempotency and
   counter-conflation findings hold regardless.
2. **A hand-patched cluster Miles/checkpoint state.** If the RunAI checkout
   already contains correct, manually applied equivalents of patches
   0001/0003/0004 and a correctly plumbed modeling file, then *new runs there*
   could be cleaner than this repo can prove — the reproducibility finding
   (cannot rebuild from the repo) would still stand.
3. **Duplicate-doc analysis key too coarse.** The 66.7% figure keys docs on
   their longest prefix's first 300 tokens. Legitimately repeated FineWeb
   documents could inflate it — but the offset structure (100% of pairwise
   offsets ≡ 0 mod 5000, groups capped at 6 = number of shards) is
   incompatible with natural duplication. If stage-0 sampled *different
   positions* per shard copy, leakage is same-text/different-position for
   most pairs (still leakage; 11,365 pairs are identical-position regardless).
   A content-hash dedup build (Step 1) measures the true unique count.
4. **Miles batcher semantics.** If `max_tokens_per_gpu` provably never co-packs
   two samples (and oversize = solo microbatch), AV contamination exposure
   (M-1) shrinks to nil and the AV-side mb1 constraint in Step 5 can relax.
5. **Tolerance choices.** The packed-vs-padded tol=0.02 and the eval's bf16
   nondeterminism (~1e-3) are assumed adequate to separate contamination from
   GEMM noise; if the step-0 check flaps on clean data, re-derive tolerances
   in fp32 before concluding either way.
6. This audit ran on commit `0812f8f` (working tree with minor doc edits).
   Any code landed after it must be re-checked against §2 before launch.

---

## Appendix: sub-audit reports

The four parallel sub-audits (datasets, eval, training tooling, configs)
produced detailed per-question evidence beyond what is summarized here —
including full config tables, per-control eval traces, and queue lifecycle
analyses. Their headline findings are merged above with lead-auditor
verification noted per item; remaining low-severity items are listed in §2
LOW/INFO. For the next agent: §6 Step 0 is the complete actionable subset.
