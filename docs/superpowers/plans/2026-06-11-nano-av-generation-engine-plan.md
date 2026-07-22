# Nano AV Generation Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable cache-aware AV generation engine and wire it into round-trip evals to improve R27/R33 gate throughput.

**Architecture:** Create `scripts/nano_av_generation.py` for job planning, streaming/resume, and cache-aware greedy decoding. Keep `scripts/eval_nano_av_ar_roundtrip_gate.py` responsible for AV/AR eval orchestration and report schema. Leave AV/AR SFT training behavior unchanged.

**Tech Stack:** Python 3.12, PyTorch, Hugging Face-style model calls, pytest, RunAI, S3 sync helper.

---

### Task 1: Generation Job Planning And Resume

**Files:**
- Create: `scripts/nano_av_generation.py`
- Test: `tests/test_nano_av_generation.py`

- [ ] Write failing tests for `GenerationJob`, stable `job_key`, `plan_generation_jobs`, `load_completed_job_keys`, and `append_generation_record`.
- [ ] Verify the tests fail because `scripts/nano_av_generation.py` does not yet exist.
- [ ] Implement the dataclass and JSONL helpers.
- [ ] Run `pytest tests/test_nano_av_generation.py -q` and verify the new tests pass.

### Task 2: Cache-Aware Greedy Decoder

**Files:**
- Modify: `scripts/nano_av_generation.py`
- Test: `tests/test_nano_av_generation.py`

- [ ] Add a fake tokenizer/model test showing the decoder calls the model once with the full prompt and then with one-token embeddings plus `past_key_values`.
- [ ] Verify the test fails before decoder implementation.
- [ ] Implement `greedy_generate_with_cache`, preserving prefix, EOS, and stop-text behavior.
- [ ] Add fallback metadata when no usable cache is returned.
- [ ] Run `pytest tests/test_nano_av_generation.py -q`.

### Task 3: Round-Trip Integration

**Files:**
- Modify: `scripts/eval_nano_av_ar_roundtrip_gate.py`
- Test: `tests/test_nano_av_ar_roundtrip_gate.py`

- [ ] Add tests that job-level generation records regroup into the existing row-level `controls` shape.
- [ ] Verify tests fail before integration.
- [ ] Replace the row/control nested generation loop with generation jobs from `nano_av_generation.py`.
- [ ] Keep old no-cache generation available through `--generation-backend legacy`.
- [ ] Run `pytest tests/test_nano_av_ar_roundtrip_gate.py tests/test_nano_av_generation.py -q`.

### Task 4: Config And Queue Flags

**Files:**
- Modify: `scripts/nano_av_probe_queue.py`
- Modify: `scripts/nano_roundtrip_eval_config.py`
- Modify: `configs/nano_roundtrip/r27_baseline_64_full_controls_prefix256.yaml`
- Modify: `configs/nano_roundtrip/r27_baseline_256_full_controls_prefix256.yaml`
- Test: `tests/test_nano_av_probe_queue.py`
- Test: `tests/test_nano_roundtrip_eval_config.py`

- [ ] Add failing tests for `--generation-backend cache`.
- [ ] Add command-rendering support for generation backend selection.
- [ ] Set R27 configs to `generation_backend: cache`.
- [ ] Run focused queue/config tests.

### Task 5: Local And RunAI Verification

**Files:**
- No new source files.

- [ ] Run local focused tests:
  `pytest tests/test_nano_av_generation.py tests/test_nano_av_ar_roundtrip_gate.py tests/test_nano_roundtrip_eval_config.py tests/test_nano_av_probe_queue.py tests/test_nano_roundtrip_queue.py -q`
- [ ] Sync changed source/config/tests to S3.
- [ ] Pull the tarball on RunAI and run the same tests in `/workspace/interp/.venv`.
- [ ] Check the existing slow R27 process status and preserve generated JSONL/logs.
- [ ] Stop the old slow R27 run only after the cache path passes RunAI tests.
- [ ] Reset the R27 queue item to pending and relaunch the 64/64 full-control gate with the cache backend.

### Task 6: Report Outcome

**Files:**
- Modify docs only if the faster run completes or fails with actionable evidence.

- [ ] Report queue status, GPU use, streamed rows, parse health, and ETA.
- [ ] If final report exists, summarize round-trip NMSE, controls, and parse health.
- [ ] Keep the 256/256 item blocked until 64/64 parse health is proven.
