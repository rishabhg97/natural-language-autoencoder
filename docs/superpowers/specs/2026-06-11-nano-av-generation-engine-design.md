# Nano AV Generation Engine Design

## Goal

Build a reusable, faster AV text-generation path for Nano NLA evals, starting with AV -> generated explanation -> AR reconstruction round-trip gates. The immediate purpose is to turn the current slow R27/R33 round-trip gates into practical HPO metrics without changing AV or AR SFT training semantics.

## Current Bottleneck

The round-trip evaluator currently generates text in `scripts/eval_nano_av_ar_roundtrip_gate.py` by calling `generate_with_control()` from `scripts/nano_av_warmstart_smoke.py` once per row/control pair. For a 64/64 full-control gate this is 128 rows times five controls. Each decode is single-sample greedy generation with `use_cache=False`, so every generated token recomputes the full prompt and all prior generated tokens.

Live RunAI observations for the R27 64/64 gate show large memory headroom and low-to-moderate GPU utilization. The bottleneck is decoder inefficiency and serial scheduling, not VRAM capacity.

## Scope

In scope:

- A focused `scripts/nano_av_generation.py` module with generation job planning, cache-aware greedy decoding, parse metadata, streaming JSONL output, and resume-friendly keys.
- Integration into `scripts/eval_nano_av_ar_roundtrip_gate.py`.
- Config switches for backend selection, worker sharding, cache use, progress logging, and parser thresholds.
- Tests that verify planning, parser/resume behavior, command rendering, and cache-mode invocation with lightweight fake models.
- RunAI sync and a replacement R27 64/64 round-trip gate only after local and remote tests pass.

Out of scope for this pass:

- Changing AV or AR SFT training loops.
- Changing checkpoint formats.
- Full multi-process distributed serving.
- Batched injected decoding across heterogeneous prompt lengths. This belongs in a separate pass after cache-aware generation and worker sharding are stable.

## Architecture

### Generation Jobs

The engine represents each generation as a `GenerationJob`:

- `row_index`, `split`, `doc_id`
- `control_name`
- `prompt` or source row
- `control_vector`
- `target_explanation`

The planner emits one job per row/control pair, preserving the existing eval semantics. A stable `job_key` is `"{split}:{row_index}:{control_name}"`, used for streaming/resume.

### Generator Backend

The first backend is a cache-aware greedy decoder for a single loaded AV model:

- Build injected prompt embeddings exactly as the current path does.
- Append optional `generation_prefix`.
- Decode with `use_cache=True` after the initial forward pass.
- Feed only the newest token embedding after the first step when `past_key_values` are available.
- Stop on EOS or `stop_text`.
- Return generated text including the prefix, so the existing parser behavior remains comparable.

If the model or remote code does not return usable cache values, the backend falls back to the current full-context greedy loop and records `cache_used=false` in metadata.

### Streaming And Resume

The engine writes one JSONL record per completed row/control job, not one record per row. This gives finer-grained resume and makes progress visible sooner. The round-trip evaluator can regroup records into the existing per-row report shape before AR scoring.

Resume behavior:

- Read existing JSONL records.
- Skip completed `job_key`s unless `--overwrite-generated` is set.
- Keep partial files useful after interruption.

### Worker Sharding

The second acceleration tier is one worker per GPU, each loading one model replica on a specific device and processing a disjoint subset of jobs. This uses memory headroom directly and avoids model-parallel communication overhead during generation.

Initial worker policy:

- `--generation-workers 1` keeps current single-process behavior.
- `--generation-workers 2 --generation-worker-devices cuda:0 cuda:1` launches two local worker processes.
- Each worker writes a shard JSONL; the parent merges shards deterministically by original job order.

Worker sharding is optional. Cache-aware single-worker generation is the minimum viable optimization.

### Round-Trip Integration

`scripts/eval_nano_av_ar_roundtrip_gate.py` will call the new engine and then convert job-level records into the existing row-level `controls` structure. AR scoring and final gate summaries remain unchanged, except that reports include generation backend metadata.

## Data Flow

1. Load eval rows and activation vectors.
2. Build control vectors: real, shuffled, zero, mean, none.
3. Plan generation jobs.
4. Generate job-level JSONL with cache-aware decoder and optional worker sharding.
5. Regroup job records by row.
6. Run AR reconstruction scoring in batches.
7. Write the round-trip report and HPO-study record.

## Error Handling

- If a generation job fails, write a failed job record with error metadata and fail the queue item unless `--allow-generation-failures` is explicitly set.
- If parse health is below configured thresholds, the report is written but the gate fails.
- If a worker exits nonzero, preserve all shard JSONL files and fail the queue item.
- If cache mode is requested but unavailable, log fallback metadata and continue by default.

## Testing

Unit tests:

- Job planning creates expected row/control keys.
- Streaming resume skips completed job keys.
- Cache-aware decoder calls the model with full embeddings once and then one-token embeddings with `past_key_values`.
- Fallback decoder still works when no cache is returned.
- Round-trip command/config rendering includes generation backend options.
- Round-trip regrouping preserves the existing row-level report schema.

RunAI verification:

- Sync source via S3.
- Run focused tests in `/workspace/interp/.venv`.
- Run a tiny generated-text smoke if needed.
- Stop the old slow R27 64/64 gate only after the faster path passes tests.
- Relaunch R27 64/64 full-control gate with the new engine.

## Success Criteria

- Existing focused local and RunAI tests pass.
- R27 64/64 full-control gate produces healthy parse rates and a final report.
- Generation throughput materially improves versus the current serial no-cache path.
- The generated report remains comparable to previous round-trip reports.
- No AV/AR SFT training code behavior changes are introduced in this pass.
