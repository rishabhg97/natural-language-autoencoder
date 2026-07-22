# Nano30B Cluster Harness Runbook

Use the local `nano30b-nla` tmux session on the cluster and keep the harness to
one visible GPU. These commands run introspection, extraction identity, small
extraction serialization probes, injection probes, and the constrained frozen AR
value-head baseline.

For current status, blockers, and additions/subtractions across runs, see
[execution_log.md](execution_log.md). This runbook should stay command-focused:
environment setup, exact invocations, expected output paths, and smoke targets.

```bash
cd /lustre/fsw/portfolios/llmservice/users/rigarg/mech_interp/research-projects/nano30b-nla-pilot
source scripts/cluster_nano_env.sh
```

The env wrapper pins:

- `NANO_MODEL_REVISION=cbd3fa9f933d55ef16a84236559f4ee2a0526848`
- `NANO_TOKENIZER_REVISION=$NANO_MODEL_REVISION`
- `HF_MODULES_CACHE=/lustre/fsw/portfolios/llmservice/users/rigarg/mech_interp/research-projects/.hf_nla_cache/modules`
- `HF_HOME=/lustre/fsw/portfolios/llmservice/users/rigarg/.cache-models`
- `CUDA_VISIBLE_DEVICES=0`

Run the metadata harness:

```bash
"$NANO_PYTHON" scripts/nano_introspection.py \
  --load-mode meta \
  --local-files-only \
  --timestamp pinned-meta-$(date -u +%Y%m%dT%H%M%SZ)
```

Run the extraction identity harness:

```bash
"$NANO_PYTHON" scripts/nano_extraction_identity.py \
  --local-files-only \
  --timestamp pinned-identity-$(date -u +%Y%m%dT%H%M%SZ)
```

Run the tiny extraction serialization probe:

```bash
"$NANO_PYTHON" scripts/nano_extraction_serialize_probe.py \
  --local-files-only \
  --timestamp tiny-serialize-$(date -u +%Y%m%dT%H%M%SZ)
```

Run the Track A input-embedding injection probe:

```bash
"$NANO_PYTHON" scripts/nano_track_a_probe.py \
  --local-files-only \
  --timestamp track-a-$(date -u +%Y%m%dT%H%M%SZ)
```

Run the Track C HF residual-boundary oracle probe:

```bash
"$NANO_PYTHON" scripts/nano_track_c_probe.py \
  --local-files-only \
  --cache-smoke \
  --timestamp track-c-$(date -u +%Y%m%dT%H%M%SZ)
```

Run the frozen AR value-head baseline smoke:

```bash
"$NANO_PYTHON" scripts/nano_ar_frozen_baseline.py \
  --local-files-only \
  --boundaries R_34 \
  --prompt-names raw,reasoning_off_chat,av_marker,ar_critic \
  --max-records 8 \
  --train-fraction 0.5 \
  --split-strategy alternating \
  --explanation-template prompt_label \
  --max-steps 50 \
  --lr 5e-5 \
  --timestamp ar-frozen-$(date -u +%Y%m%dT%H%M%SZ)
```

The AR harness follows the reference NLA critic contract: bias-free identity
`d_model -> d_model` head, NLA-scale normalized MSE where `MSE = 2(1-cos)`,
critic prompt suffix `</text> <summary>`, and train-split mean/RRI controls.

Run the bounded frozen AR smoke grid:

```bash
"$NANO_PYTHON" scripts/nano_ar_smoke_grid.py \
  --local-files-only \
  --boundaries-grid R_34,R_27 \
  --max-records-grid 8 \
  --train-fractions-grid 0.5 \
  --split-strategies-grid alternating \
  --explanation-templates-grid generic,prompt_label \
  --lrs-grid 2e-5,5e-5 \
  --max-steps-grid 50 \
  --seeds-grid 1234 \
  --max-runs 8 \
  --timestamp ar-smoke-grid-$(date -u +%Y%m%dT%H%M%SZ)
```

Expected outputs are written under `runs/introspection/<timestamp>/`.

## Actual-Data AR Smoke

The synthetic AR grid is now only a regression harness. Use this path for
scientific smoke runs because it follows the reference NLA datagen contract on
real public text: Stage 0 extracts Nano activations, Stage 1 splits by document,
Stage 2 asks a teacher for `api_explanation`, and the Nano AR builder formats
those explanations into critic prompts.

Reference NLA used public corpora and generated model/layer-specific activation
parquets rather than publishing a reusable universal activation dataset. The
repo configs use `HuggingFaceFW/fineweb` `sample-10BT` for fresh reproduction
and note that the Ultra-FineWeb subset used for released checkpoints is no
longer hosted as that exact slice.

For the Stage 1 dry run before AV/AR PEFT, keep the reference marker contract:
AV/Track A uses the upstream rare CJK single-token injection marker metadata,
not common English tokens. The training parquet should keep `<INJECT>` as the
placeholder, while the sidecar records the resolved `injection_char`,
`injection_token_id`, neighbor IDs, prompt templates, and AR critic suffix IDs.
If Nano has no single-token entry in the upstream enclosed-CJK block, use a
verified rare CJK-symbol fallback rather than common English tokens.
This makes Nano data compatible with the released NLA loader assumptions while
only swapping the extraction backend for Nano `R_b`.

The reference GitHub reproduction config uses `HuggingFaceFW/fineweb`
`sample-10BT`. Use that same source for Nano. Full reference scale is `100k`
docs x `10` positions/doc, split `25/25/50`; dry runs should keep the same
dataset and split but smaller `corpus-length`.

## Reference-Aligned Datagen Dry Run

This is the next path before AV/AR PEFT. It builds all three reference-style
training parquets for Nano using FineWeb `sample-10BT`, not synthetic data.

```bash
cd /lustre/fsw/portfolios/llmservice/users/rigarg/mech_interp/research-projects/nano30b-nla-pilot
source scripts/cluster_nano_env.sh
export PYTHONPATH="$PWD/external/natural_language_autoencoders:${PYTHONPATH:-}"

TS=datagen-r34-$(date -u +%Y%m%dT%H%M%SZ)
OUT=runs/introspection/$TS
mkdir -p "$OUT/splits"

"$NANO_PYTHON" scripts/nano_realdata_stage0_extract.py \
  --local-files-only \
  --boundary R_34 \
  --corpus HuggingFaceFW/fineweb \
  --corpus-config sample-10BT \
  --corpus-split train \
  --text-column text \
  --corpus-start 0 \
  --corpus-length 1000 \
  --positions-per-doc 5 \
  --chunk-size 4 \
  --batch-size 1 \
  --max-length 1024 \
  --output "$OUT/base_R34.parquet"

"$NANO_PYTHON" -m nla.datagen.stage1_split \
  --base "$OUT/base_R34.parquet" \
  --av-sft-frac 0.25 \
  --ar-sft-frac 0.25 \
  --rl-frac 0.50 \
  --output-dir "$OUT/splits"

# Requires ANTHROPIC_API_KEY. Teacher labels are warm-start data, not ground truth.
"$NANO_PYTHON" -m nla.datagen.stage2_api_explain \
  --input "$OUT/splits/av_sft_raw.parquet" \
  --output "$OUT/splits/av_sft_explained.parquet" \
  --provider-kwargs '{"model":"claude-haiku-4-5-20251001","max_tokens":300,"concurrency":8}' \
  --chunk-size 64

"$NANO_PYTHON" -m nla.datagen.stage2_api_explain \
  --input "$OUT/splits/ar_sft_raw.parquet" \
  --output "$OUT/splits/ar_sft_explained.parquet" \
  --provider-kwargs '{"model":"claude-haiku-4-5-20251001","max_tokens":300,"concurrency":8}' \
  --chunk-size 64

"$NANO_PYTHON" scripts/nano_realdata_stage3_build.py \
  --local-files-only \
  --input "$OUT/splits/av_sft_explained.parquet" \
  --stage av_sft \
  --output "$OUT/av_sft.parquet"

"$NANO_PYTHON" scripts/nano_realdata_stage3_build.py \
  --local-files-only \
  --input "$OUT/splits/ar_sft_explained.parquet" \
  --stage ar_sft \
  --output "$OUT/ar_sft.parquet"

"$NANO_PYTHON" scripts/nano_realdata_stage3_build.py \
  --local-files-only \
  --input "$OUT/splits/rl_raw.parquet" \
  --stage rl \
  --output "$OUT/rl.parquet"

"$NANO_PYTHON" -m nla.datagen.stage_shuffle \
  --input "$OUT/av_sft.parquet" \
  --output "$OUT/av_sft_shuf.parquet" \
  --seed 42

"$NANO_PYTHON" -m nla.datagen.stage_shuffle \
  --input "$OUT/ar_sft.parquet" \
  --output "$OUT/ar_sft_shuf.parquet" \
  --seed 42

"$NANO_PYTHON" -m nla.datagen.stage_shuffle \
  --input "$OUT/rl.parquet" \
  --output "$OUT/rl_shuf.parquet" \
  --seed 42
```

If the dry run passes, repeat with `--boundary R_27`. If `detokenized_text_truncated`
matches the R_34 run for the same corpus slice and tokenizer, use Stage 2
`--cache-from` to avoid paying for duplicate teacher explanations.

## Minimal AR-Only Smoke

```bash
cd /lustre/fsw/portfolios/llmservice/users/rigarg/mech_interp/research-projects/nano30b-nla-pilot
source scripts/cluster_nano_env.sh
export PYTHONPATH="$PWD/external/natural_language_autoencoders:${PYTHONPATH:-}"

TS=realdata-r34-$(date -u +%Y%m%dT%H%M%SZ)
OUT=runs/introspection/$TS
mkdir -p "$OUT/splits"

"$NANO_PYTHON" scripts/nano_realdata_stage0_extract.py \
  --local-files-only \
  --boundary R_34 \
  --corpus HuggingFaceFW/fineweb \
  --corpus-config sample-10BT \
  --corpus-split train \
  --text-column text \
  --corpus-start 0 \
  --corpus-length 64 \
  --positions-per-doc 2 \
  --chunk-size 4 \
  --batch-size 1 \
  --max-length 1024 \
  --output "$OUT/base_R34.parquet"

"$NANO_PYTHON" -m nla.datagen.stage1_split \
  --base "$OUT/base_R34.parquet" \
  --av-sft-frac 0.0 \
  --ar-sft-frac 1.0 \
  --rl-frac 0.0 \
  --output-dir "$OUT/splits"

# Requires ANTHROPIC_API_KEY. This is teacher-distilled z, not ground truth.
"$NANO_PYTHON" -m nla.datagen.stage2_api_explain \
  --input "$OUT/splits/ar_sft_raw.parquet" \
  --output "$OUT/splits/ar_sft_explained.parquet" \
  --provider-kwargs '{"model":"claude-haiku-4-5-20251001","max_tokens":300,"concurrency":8}' \
  --chunk-size 32

"$NANO_PYTHON" scripts/nano_realdata_ar_build.py \
  --local-files-only \
  --input "$OUT/splits/ar_sft_explained.parquet" \
  --output "$OUT/ar_sft.parquet"

"$NANO_PYTHON" scripts/nano_ar_frozen_baseline.py \
  --local-files-only \
  --ar-sft-parquet "$OUT/ar_sft.parquet" \
  --boundaries R_34 \
  --max-records 128 \
  --train-fraction 0.75 \
  --split-strategy random \
  --max-steps 200 \
  --lr 2e-5 \
  --mse-margin 0.02 \
  --cosine-margin 0.01 \
  --min-rri 0.05 \
  --timestamp "$TS-ar"
```

Target for the first longer smoke: train loss should decrease, heldout
correct-pair MSE should beat shuffled and random controls, and heldout
RRI-vs-train-mean should be at least `0.05`. If R_34 passes, repeat the same
commands with `R_27`; if it does not pass, vary row count, train steps, LR, and
teacher prompt before moving to AV/AR model training.
