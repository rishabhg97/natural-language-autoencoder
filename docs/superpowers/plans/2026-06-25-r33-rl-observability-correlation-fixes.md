# R33 RL Observability, Reward-Gate Alignment, and Throughput Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Nano30B R33 RL runs interpretable, gate-aligned, and throughput-tunable before launching more expensive RL training.

**Architecture:** First add low-risk rollout observability that cannot corrupt training semantics. Then add a critic-recomputed reward-vs-gate diagnostic on the exact rows used by the paired round-trip gate. In parallel, keep the throughput work YAML-driven: resource topology, SGLang tensor parallelism, FSDP/offload switches, and canary variants must be expressed in queue configs and rendered by `scripts/nano_rl_queue.py` / `configs/rl.sh`, not by editing training scripts between runs. Only after the diagnostics pass should the next RL queue item or throughput canary be unblocked.

**Tech Stack:** Python 3.12, pytest/unittest, existing NLA/Miles hooks under `external/natural_language_autoencoders`, Miles patch files, YAML queue configs, RunAI 8x H100 runtime, W&B offline logs.

---

## Critical Corrections From Plan Review

- Do **not** use positional grouping (`idx // n_samples_per_prompt`) as a safety signal. `external/natural_language_autoencoders/CLAUDE.md` says Miles reorders samples; positional grouping can be wrong.
- Do **not** compute per-group reward stats in the rollout hook: rollout `Sample`s carry no group/prompt id (only `metadata["activation_vector"]`, verified in `nla/reward.py:116` and `nla/rollout/nla_generate.py:293`), so positional or metadata grouping is wrong or inert in production. Advantage variance comes from the **real** `rollout_data["advantages"]` (Task 2) only.
- The current 5e-6 probe ran on pre-patch code: it has no `rollout/nla_advantage/*`. The advantage-variance go/no-go therefore applies to the **next instrumented run's first rollouts**, not the current probe.
- Do **not** use preexisting rollout reward summaries for reward-vs-gate correlation. Recompute reward with the frozen NLA critic on the exact generated gate rows.
- Do **not** soften failure penalty to `-1.0`: valid rewards can be below `-1.0`, so this can make failed parses better than poor valid reconstructions.
- Do **not** set `grad_norm = NaN` in the Miles patch. Omit or sentinel the metric safely, and verify patches apply.
- Do **not** push directly to `main` until tests pass and the live queue is not about to consume a broken moving code root.
- Do **not** treat throughput canaries as quality wins. The new topology/runtime knobs are for measuring speed, utilization, and OOM headroom; promotion still requires paired gate and reward-alignment evidence.
- Do **not** hand-edit `rl.sh` or launch scripts for throughput experiments. Use structured queue keys (`resources`, `sglang`, `training`) so runs are reproducible, reviewable, and test-covered.

## File Structure

- Modify `external/natural_language_autoencoders/nla/rollout/rl_metrics.py`
  - Responsibility: reward/parse/failure stdout summaries only (no per-group stats — samples carry no group id), plus the `advantage_stats_from_rollout_data` helper consumed by Task 2.
- Modify `tests/test_nla_rl_metrics.py`
  - Responsibility: unit tests for rollout stats, stdout summaries, and safe behavior when grouping metadata is missing.
- Modify `external/natural_language_autoencoders/nla/reward.py`
  - Responsibility: configurable failure penalty with unchanged default; no unsafe ablation value baked into configs.
- Create `tests/test_nla_reward_config.py`
  - Responsibility: verify default and env override for failure penalty.
- Create `scripts/analyze_nla_rl_run.py`
  - Responsibility: compact parser for `train.log` and SGLang logs.
- Create `tests/test_analyze_nla_rl_run.py`
  - Responsibility: parser tests from synthetic logs.
- Create `scripts/analyze_rl_reward_gate_correlation.py`
  - Responsibility: recompute NLA reward on exact paired gate generated rows and correlate reward with gate rowwise NMSE.
- Create `tests/test_analyze_rl_reward_gate_correlation.py`
  - Responsibility: pure-function tests for row pairing, correlation, and failure handling.
- Modify `scripts/nano_rl_queue.py`
  - Responsibility: render structured RL topology/throughput config, rewrite managed SGLang `--tp-size` / `--base-gpu-id`, pass topology env vars, and reject unsupported runtime combinations before launch.
- Modify `external/natural_language_autoencoders/configs/rl.sh`
  - Responsibility: forward topology and throughput env vars into Miles, echo the active GPU topology, and map config toggles to FSDP/offload/gradient-checkpointing/colocation CLI flags.
- Modify `external/natural_language_autoencoders/nla/system_metrics.py`
  - Responsibility: log topology metrics from env and stay import-safe when `torch` is unavailable.
- Modify `external/natural_language_autoencoders/nla/miles_patches/0004_fsdp_timing_debug.patch`
  - Responsibility: add true post-advantage stats from `rollout_data["advantages"]` after Miles computes them.
- Do **not** modify `external/natural_language_autoencoders/nla/miles_patches/0005_fsdp_skip_grad_norm_debug.patch` in this pass
  - The `grad_norm=0.0` sentinel is documented as a caveat in `docs/rl_logbook.md` (Task 7) instead. A runtime change (omit/sentinel the metric) needs its own patch-apply test against a clean Miles and is out of scope here. Do not set `grad_norm=NaN` (it can trip `check_grad_norm` in `0004` and poison aggregates).
- Modify `configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml`
  - Responsibility: expose stdout logging env vars, keep follow-up quality runs blocked, and define throughput canaries via structured topology/training fields.
- Modify `tests/test_nano_rl_queue.py`
  - Responsibility: verify queue gating, topology rendering, SGLang command rewriting, runtime validation, and env/config changes.
- Modify `tests/test_nla_system_metrics.py`
  - Responsibility: verify topology metrics and no-torch import behavior.
- Modify `docs/rl_logbook.md`
  - Responsibility: record the audit, fixes, and go/no-go criteria.

---

### Task 0: Freeze Training Decisions Until the Current Probe Finishes

**Files:**
- Modify: `docs/rl_logbook.md`

- [ ] **Step 1: Check current probe status**

Run:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
RUN_DIR=/workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/r33_component_rl_8h100_tier1_probe_rb64_n8_gb512_lr5e6_rollout8_v256t256
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
ls -lh "$RUN_DIR"/roundtrip_iter_*_report.json "$RUN_DIR"/actor/iter_* 2>/dev/null || true
tail -80 "$RUN_DIR/train.log" | perl -pe "s/\e\[[0-9;]*[mK]//g" | cut -c1-220
'
```

Expected while running: actor/logprob progress. Expected if complete: final `roundtrip_iter_0000008_v256_t256_report.json`.

- [ ] **Step 2: Append guardrail to logbook**

Append to `docs/rl_logbook.md`:

```markdown
## 2026-06-25 - RL Audit Guardrail

- Status: do not launch additional RL training until the active `5e-6` rollout-8 probe completes and the paired 256/256 gate is read.
- Audit conclusion: pipeline health is good, but reward/gate correlation and true advantage variance are not yet proven.
- Required before next RL launch:
  - rollout stdout must include reward, parse, failure, and reliable variance metrics;
  - a critic-recomputed reward-vs-gate correlation report must exist for the final probe artifacts;
  - queued `1e-5` follow-up must remain blocked unless the correlation report and final paired gate justify it.
```

- [ ] **Step 3: Commit**

```bash
/Users/rigarg/.local/bin/visor run git add docs/rl_logbook.md
/Users/rigarg/.local/bin/visor run git commit -m "docs: add rl audit guardrail"
```

---

### Task 1: Surface Reward and Parse Stats to `train.log` Without Unsafe Positional Grouping

**Files:**
- Modify: `external/natural_language_autoencoders/nla/rollout/rl_metrics.py`
- Modify: `tests/test_nla_rl_metrics.py`

- [ ] **Step 1: Add tests for stdout and metadata-only grouping**

Add to `tests/test_nla_rl_metrics.py`:

```python
class Sample:
    def __init__(
        self,
        reward: float | None,
        response: str,
        length: int,
        status: str = "COMPLETED",
        metadata: dict | None = None,
    ) -> None:
        self._reward = reward
        self.response = response
        self.effective_response_length = length
        self.status = types.SimpleNamespace(name=status)
        self.metadata = metadata or {}

    def get_reward_value(self, _args):
        return self._reward
```

If this class already exists, update it to match the signature above.

Add this test method:

```python
def test_prints_compact_rollout_summary_to_stdout(self):
    module = load_module()
    metrics: dict[str, float | int] = {}
    samples = [
        Sample(-0.2, "<explanation>a</explanation>", 10),
        Sample(-0.6, "<explanation>b</explanation>", 11),
    ]

    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        module.log_rollout_data(4, object(), samples, metrics, 1.0)

    output = buffer.getvalue()
    self.assertIn("[NLA ROLLOUT]", output)
    self.assertIn('"rollout_id": 4', output)
    self.assertIn('"reward_mean": -0.4', output)
    self.assertIn('"usable_frac": 1.0', output)
```

Do **not** add per-group reward-std tests. Rollout `Sample`s carry no group/prompt id
(only `metadata["activation_vector"]`, verified in `nla/reward.py:116` and
`nla/rollout/nla_generate.py:293`), so any in-hook grouping is `None` for every real
sample and the metric would be inert (it would only "pass" against synthetic test
metadata). Advantage variance is measured from the real `rollout_data["advantages"]` in
Task 2. The `metadata` parameter on the `Sample` stub above is harmless to keep but no
Task 1 test depends on it.

- [ ] **Step 2: Run tests and confirm failure**

```bash
/Users/rigarg/.local/bin/visor run python3 -m pytest tests/test_nla_rl_metrics.py -q
```

Expected: failures for missing stdout and group metadata metrics.

- [ ] **Step 3: Implement safe stdout and metadata grouping**

In `external/natural_language_autoencoders/nla/rollout/rl_metrics.py`, add imports:

```python
import json
import os
```

Add helpers below `_prefix`:

```python
def _stdout_enabled() -> bool:
    raw = os.environ.get("NLA_ROLLOUT_SUMMARY_STDOUT", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _emit_stdout_summary(rollout_id: int, metrics: dict[str, float | int]) -> None:
    if not _stdout_enabled():
        return
    payload = {
        "rollout_id": rollout_id,
        "reward_mean": metrics.get("rollout/nla_reward/mean"),
        "reward_std": metrics.get("rollout/nla_reward/std"),
        "reward_p10": metrics.get("rollout/nla_reward/p10"),
        "reward_p50": metrics.get("rollout/nla_reward/p50"),
        "reward_p90": metrics.get("rollout/nla_reward/p90"),
        "closed_frac": metrics.get("rollout/nla_parse/closed_frac"),
        "usable_frac": metrics.get("rollout/nla_parse/usable_frac"),
        "failed_frac": metrics.get("rollout/nla_status/failed_frac"),
        "truncated_frac": metrics.get("rollout/nla_status/truncated_frac"),
        "length_corr": metrics.get("rollout/nla_reward/length_corr"),
    }
    print("[NLA ROLLOUT] " + json.dumps(payload, sort_keys=True), flush=True)
```

Change the first argument name from `_rollout_id` to `rollout_id`. Leave the existing
reward-collection loop unchanged.

Before the **final** `return False` (the one at the end of `log_rollout_data`, after the
reward/parse metrics are populated — NOT the early `if rollout_extra_metrics is None:
return False` guard), add only the stdout emission:

```python
    _emit_stdout_summary(rollout_id, rollout_extra_metrics)
```

Do **not** add `_group_id`, `_grouped_reward_stats`, positional grouping, or any
`rollout/nla_group/*` or `rollout/nla_grpo_advantage_proxy/*` metric in this task: rollout
`Sample`s carry no group id, so those stats are inert or wrong. Advantage variance is
covered by Task 2 (real `rollout_data["advantages"]`).

- [ ] **Step 4: Run tests**

```bash
/Users/rigarg/.local/bin/visor run python3 -m pytest tests/test_nla_rl_metrics.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
/Users/rigarg/.local/bin/visor run git add external/natural_language_autoencoders/nla/rollout/rl_metrics.py tests/test_nla_rl_metrics.py
/Users/rigarg/.local/bin/visor run git commit -m "feat: log safe nla rollout stats"
```

---

### Task 2: Log True Advantage Distribution From Miles Rollout Data

**Files:**
- Modify: `external/natural_language_autoencoders/nla/miles_patches/0004_fsdp_timing_debug.patch`
- Modify: `tests/test_nla_rl_metrics.py`

- [ ] **Step 1: Add pure helper tests**

Add to `tests/test_nla_rl_metrics.py`:

```python
def test_advantage_stats_from_rollout_data_tensor(self):
    module = load_module()
    import torch

    metrics = module.advantage_stats_from_rollout_data({"advantages": torch.tensor([0.0, 1.0, -1.0, 0.0])})

    self.assertEqual(metrics["rollout/nla_advantage/count"], 4)
    self.assertGreater(metrics["rollout/nla_advantage/std"], 0.0)
    self.assertAlmostEqual(metrics["rollout/nla_advantage/frac_zero"], 0.5)

def test_advantage_stats_empty_when_missing(self):
    module = load_module()
    self.assertEqual(module.advantage_stats_from_rollout_data({}), {})
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
/Users/rigarg/.local/bin/visor run python3 -m pytest tests/test_nla_rl_metrics.py -q
```

Expected: missing `advantage_stats_from_rollout_data`.

- [ ] **Step 3: Implement helper in `rl_metrics.py`**

Add:

```python
def advantage_stats_from_rollout_data(rollout_data: dict[str, Any]) -> dict[str, float | int]:
    values_obj = rollout_data.get("advantages")
    if values_obj is None:
        return {}
    try:
        import torch

        if isinstance(values_obj, torch.Tensor):
            values = values_obj.detach().float().cpu().view(-1).tolist()
        else:
            values = [float(value) for value in values_obj]
    except Exception:
        return {}
    stats = _prefix("rollout/nla_advantage", _stats([float(value) for value in values]))
    if values:
        stats["rollout/nla_advantage/frac_zero"] = sum(abs(float(value)) <= 1e-12 for value in values) / len(values)
    return stats
```

- [ ] **Step 4: Patch Miles rollout logging after `compute_advantages_and_returns`**

In `external/natural_language_autoencoders/nla/miles_patches/0004_fsdp_timing_debug.patch`, near the existing added lines after:

```diff
         compute_advantages_and_returns(self.args, self.parallel_state, rollout_data)
         self._nla_timing_log("nla_timing_compute_advantages", nla_timing_advantages_start, rollout_id=rollout_id)

         nla_timing_log_rollout_start = self._nla_timing_start()
         log_rollout_data(rollout_id, self.args, rollout_data, self.parallel_state)
```

add a patch hunk that logs true advantage stats to `train.log` (no `wandb.log`):

```diff
+        try:
+            from nla.rollout.rl_metrics import advantage_stats_from_rollout_data
+            nla_adv_metrics = advantage_stats_from_rollout_data(rollout_data)
+            if nla_adv_metrics:
+                logger.info("[NLA ADVANTAGE] " + " ".join(f"{k}={v}" for k, v in sorted(nla_adv_metrics.items())))
+        except Exception as exc:
+            logger.warning("[NLA ADVANTAGE] failed to collect advantage stats: %s", exc)
```

Constraints (these prevent the issues found in review):

- Do **not** call `wandb.log(nla_adv_metrics, step=rollout_id)`. This patch runs on every
  actor rank, and an explicit `step=rollout_id` (0..7) collides with Miles's own
  monotonically-increasing train-step axis — W&B drops or forks out-of-order steps. The
  log line is the deliverable; Task 4's analyzer parses `[NLA ADVANTAGE]` from `train.log`.
- Guard the emission to the data-parallel **source rank** using whatever rank-0 accessor
  the surrounding patched file already uses (the same one gating other rank-0-only logs).
  Without a guard you get 4 duplicate lines, each computed over that rank's **shard** of
  `advantages` (so the std would be a partial-batch std, not the global one). If no such
  accessor is readily available, document that the stat is per-rank-shard.
- `logger` is in scope at this insertion point — it is already used by the
  `_nla_timing_log` lines added by this same patch (`0004_fsdp_timing_debug.patch:46`).
- Do not use positional sample grouping.

- [ ] **Step 5: Verify the edited patch applies to a PRISTINE Miles, then re-apply to the run code root**

The live `/workspace/interp/code/miles-051cd15` is **already patched**, so a dry-run
against it would falsely fail. Check against a fresh checkout reset to the pinned upstream
SHA (`051cd15`) instead:

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
set -euo pipefail
PRISTINE=/tmp/miles-pristine-051cd15
PATCHES=/workspace/interp/code/nano30b-nla-pilot-current/external/natural_language_autoencoders/nla/miles_patches
rm -rf "$PRISTINE"
git clone --quiet /workspace/interp/code/miles-051cd15 "$PRISTINE"
cd "$PRISTINE"
git reset --hard 051cd15   # drop any committed/applied patches -> true upstream base
for p in "$PATCHES"/*.patch; do
  git apply --check "$p" || { echo "PATCH FAILS TO APPLY: $p"; exit 1; }
done
echo "ALL PATCHES APPLY CLEAN"
'
```

**Re-apply is required and is NOT automatic.** Editing `0004_*.patch` does not change the
Miles that training imports. Before the next instrumented run:

1. Confirm how this project applies patches to the run code root — `scripts/check_miles_patches.py` is the **checker**; find the corresponding apply step (env/setup).
2. Re-run that apply against the `defaults.code_root` the next queue item will use, then run `python3 scripts/check_miles_patches.py` and confirm it reports all patches applied.
3. Do **not** launch the next run until the edited `0004` patch is confirmed live in that code root (otherwise `[NLA ADVANTAGE]` lines never appear and the Stage-B gate in Task 8 cannot be evaluated).

- [ ] **Step 6: Run tests and commit**

```bash
/Users/rigarg/.local/bin/visor run python3 -m pytest tests/test_nla_rl_metrics.py -q
/Users/rigarg/.local/bin/visor run git add external/natural_language_autoencoders/nla/rollout/rl_metrics.py external/natural_language_autoencoders/nla/miles_patches/0004_fsdp_timing_debug.patch tests/test_nla_rl_metrics.py
/Users/rigarg/.local/bin/visor run git commit -m "feat: log true rl advantage stats"
```

---

### Task 3: Make Failure Penalty Configurable But Preserve the Safe Default

**Files:**
- Modify: `external/natural_language_autoencoders/nla/reward.py`
- Create: `tests/test_nla_reward_config.py`

- [ ] **Step 1: Write tests**

Create `tests/test_nla_reward_config.py`:

```python
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NLA_ROOT = ROOT / "external" / "natural_language_autoencoders"

def load_reward_module(env: dict[str, str] | None = None):
    path = NLA_ROOT / "nla" / "reward.py"
    spec = importlib.util.spec_from_file_location("nla_reward_under_test", path)
    module = importlib.util.module_from_spec(spec)
    old_env = os.environ.copy()
    sys.path.insert(0, str(NLA_ROOT))
    try:
        os.environ.clear()
        os.environ.update(old_env)
        if env:
            os.environ.update(env)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        sys.path.pop(0)

class NLARewardConfigTests(unittest.TestCase):
    def test_default_failure_penalty_stays_minus_two(self):
        module = load_reward_module()
        self.assertEqual(module.failed_extraction_reward(), -2.0)

    def test_env_overrides_failure_penalty(self):
        module = load_reward_module({"NLA_FAILED_EXTRACTION_REWARD": "-2.5"})
        self.assertEqual(module.failed_extraction_reward(), -2.5)

    def test_bad_env_value_raises(self):
        with self.assertRaises(ValueError):
            load_reward_module({"NLA_FAILED_EXTRACTION_REWARD": "not-a-number"})

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Implement helper**

Replace the constant with:

```python
def failed_extraction_reward() -> float:
    raw = os.environ.get("NLA_FAILED_EXTRACTION_REWARD")
    if raw is not None and raw.strip() != "":
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"NLA_FAILED_EXTRACTION_REWARD must be numeric, got {raw!r}") from exc
    return -math.log(2.0) if _USE_LOG_MSE_REWARD else -2.0
```

Replace:

```python
rewards = [FAILED_EXTRACTION_REWARD] * len(samples)
```

with:

```python
rewards = [failed_extraction_reward()] * len(samples)
```

- [ ] **Step 3: Run tests and commit**

```bash
/Users/rigarg/.local/bin/visor run python3 -m pytest tests/test_nla_reward_config.py tests/test_nla_rl_metrics.py -q
/Users/rigarg/.local/bin/visor run git add external/natural_language_autoencoders/nla/reward.py tests/test_nla_reward_config.py
/Users/rigarg/.local/bin/visor run git commit -m "feat: configure nla rl failure penalty"
```

Do not set `NLA_FAILED_EXTRACTION_REWARD=-1.0` in any queue. Future ablations should use either a stricter penalty below the empirical valid floor or an explicit "exclude failed samples from group baseline" implementation.

---

### Task 4: Add Compact RL Log Analyzer

**Files:**
- Create: `scripts/analyze_nla_rl_run.py`
- Create: `tests/test_analyze_nla_rl_run.py`

- [ ] **Step 1: Write parser tests**

Create `tests/test_analyze_nla_rl_run.py`:

```python
from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import textwrap
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_nla_rl_run.py"

def load_module():
    spec = importlib.util.spec_from_file_location("analyze_nla_rl_run", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

class AnalyzeNLARLRunTests(unittest.TestCase):
    def test_parses_rollout_perf_and_warning_summary(self):
        module = load_module()
        log = textwrap.dedent(
            """
            [x] log_utils.py:52 - rollout 0: {'rollout/response_lengths': 122.0, 'rollout/raw_reward': -0.48, 'rollout/truncated': 0.0, 'rollout/advantages': 0.0}
            [x] train_metric_utils.py:44 - perf 0: {'perf/actor_train_time': 1051.8, 'perf/ref_log_probs_time': 283.4, 'perf/log_probs_time': 242.6, 'perf/step_time': 1729.4}
            [x] log_utils.py:429 - step 0: {'train/ppo_kl': 0.0, 'train/kl_loss': 0.0, 'train/pg_clipfrac': 0.0, 'train/grad_norm': 0.0}
            [x] WARNING something benign
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "train.log"
            path.write_text(log)
            summary = module.analyze_train_log(path)

        self.assertEqual(summary["rollout_count"], 1)
        self.assertEqual(summary["step_count"], 1)
        self.assertEqual(summary["latest_rollout"]["rollout/raw_reward"], -0.48)
        self.assertEqual(summary["latest_perf"]["perf/step_time"], 1729.4)
        self.assertEqual(summary["latest_step"]["train/ppo_kl"], 0.0)
        self.assertEqual(summary["warning_count"], 1)
```

- [ ] **Step 2: Implement analyzer**

Create `scripts/analyze_nla_rl_run.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")

def _clean(line: str) -> str:
    return ANSI_RE.sub("", line)

def _parse_dict_after(pattern: str, line: str) -> tuple[int, dict[str, Any]] | None:
    match = re.search(pattern, line)
    if not match:
        return None
    return int(match.group(1)), ast.literal_eval(match.group(2))

def analyze_train_log(path: Path) -> dict[str, Any]:
    rollouts: list[tuple[int, dict[str, Any]]] = []
    perfs: list[tuple[int, dict[str, Any]]] = []
    steps: list[tuple[int, dict[str, Any]]] = []
    warnings: list[str] = []
    errors: list[str] = []
    for raw in path.read_text(errors="ignore").splitlines():
        line = _clean(raw)
        for pattern, target in (
            (r"rollout (\d+): (\{.*\})", rollouts),
            (r"perf (\d+): (\{.*\})", perfs),
            (r"step (\d+): (\{.*\})", steps),
        ):
            parsed = _parse_dict_after(pattern, line)
            if parsed is not None:
                target.append(parsed)
                break
        lower = line.lower()
        if "warning" in lower:
            warnings.append(line[:500])
        if any(token in lower for token in ("traceback", "error", "oom", "out of memory")):
            errors.append(line[:500])
    return {
        "path": str(path),
        "rollout_count": len(rollouts),
        "perf_count": len(perfs),
        "step_count": len(steps),
        "latest_rollout": rollouts[-1][1] if rollouts else None,
        "latest_perf": perfs[-1][1] if perfs else None,
        "latest_step": steps[-1][1] if steps else None,
        "warning_count": len(warnings),
        "error_count": len(errors),
        "warnings_tail": warnings[-10:],
        "errors_tail": errors[-10:],
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    summary = analyze_train_log(args.train_log)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests and commit**

```bash
/Users/rigarg/.local/bin/visor run python3 -m pytest tests/test_analyze_nla_rl_run.py -q
/Users/rigarg/.local/bin/visor run git add scripts/analyze_nla_rl_run.py tests/test_analyze_nla_rl_run.py
/Users/rigarg/.local/bin/visor run git commit -m "feat: add nla rl log analyzer"
```

---

### Task 5: Add Critic-Recomputed Reward-vs-Gate Correlation Diagnostic

**Files:**
- Create: `scripts/analyze_rl_reward_gate_correlation.py`
- Create: `tests/test_analyze_rl_reward_gate_correlation.py`

- [ ] **Step 1: Write pure tests**

Create `tests/test_analyze_rl_reward_gate_correlation.py`:

```python
from __future__ import annotations

import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_rl_reward_gate_correlation.py"

def load_module():
    spec = importlib.util.spec_from_file_location("analyze_rl_reward_gate_correlation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

class RewardGateCorrelationTests(unittest.TestCase):
    def test_spearman_positive_and_negative(self):
        module = load_module()
        self.assertAlmostEqual(module.spearman_corr([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(module.spearman_corr([1, 2, 3], [30, 20, 10]), -1.0)

    def test_pairs_report_losses_with_reward_rows(self):
        module = load_module()
        split = {
            "row_indices": [10, 11, 12],
            "rowwise_normalized_mse": {"av_real": [0.3, 0.2, 0.1]},
        }
        rewards = {12: -0.1, 10: -0.5}
        pairs = module.pair_rewards_with_gate_losses(split, rewards)
        self.assertEqual(pairs["row_indices"], [10, 12])
        self.assertEqual(pairs["rewards"], [-0.5, -0.1])
        self.assertEqual(pairs["gate_losses"], [0.3, 0.1])
```

- [ ] **Step 2: Implement pure core**

Create `scripts/analyze_rl_reward_gate_correlation.py` with rank/correlation/pairing helpers. Use the same helpers from the previous plan only for:

```text
_rank
pearson_corr
spearman_corr
pair_rewards_with_gate_losses
```

Do not implement a rollout-rewards shortcut as a decision gate.

- [ ] **Step 3: Add critic-recompute CLI**

The CLI must require generated rows and critic inputs:

```text
--roundtrip-report-json
--generated-jsonl
--critic-checkpoint-dir
--validation-parquet
--test-parquet
--output-json
--batch-size
--device
```

The critic prompt template and `mse_scale` are read from the **critic checkpoint's
sidecar** (`--critic-checkpoint-dir`), exactly as training does. Do **not** add a parquet
`--critic-template-source` flag and do **not** add `--train-parquet`: the gate report only
has `validation`/`test` splits, and sourcing the template from anywhere but the critic
sidecar would diverge from the reward RL actually optimized.

Implementation requirements:

1. Read final `roundtrip_iter_0000008_v256_t256_report_generated.jsonl`.
2. For each generated record, extract the explanation by **reusing `nla.schema.extract_explanation`** on `controls.real.generated` (fall back to `controls.real.parsed.explanation` if already parsed) — the same extractor `nla/reward.py` uses, not a re-implementation.
3. Load gold `activation_vector` for validation/test rows via `load_eval_rows` + `_rows_by_index` from `scripts/eval_nano_av_ar_roundtrip_gate.py`.
4. Load the sidecar with `nla.config.load_nla_config(critic_checkpoint_dir, tokenizer)` and format prompts with its `critic_prompt_template`; **right-pad** the tokenizer (matches `nla/reward.py:86`).
5. Load `NLACriticModel.from_pretrained(critic_checkpoint_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)`. **Hard-assert the value head loaded:** verify `<critic_dir>/value_head.safetensors` exists AND `model.value_head.weight.is_meta is False` after load, and **raise** otherwise. `models.py:238-240` loads the head only `if head_path.exists()`, so a missing file leaves `value_head` randomly initialized **with no error** → garbage reward → meaningless correlation.
6. Compute rewards by **reusing `nla.reward._mse_to_reward(pred, gold, cfg.mse_scale)`** (the same L2-normalize + `-MSE` the trainer used), selecting the last non-pad token for the critic forward. Do not hand-roll the normalization.
7. Pair rewards to `splits[*].row_indices` and `rowwise_normalized_mse.av_real`.
8. Interpret useful alignment as **negative reward-vs-gate-loss correlation**. This is a fixed-policy diagnostic, not proof that a policy-gradient step will improve the gate.

- [ ] **Step 4: Add a smoke test for CLI pairing without loading HF**

Add a test that monkeypatches reward recomputation to return `{row_index: reward}` and verifies output has:

```python
"spearman_reward_vs_gate_loss"
"pearson_reward_vs_gate_loss"
"interpretation": "fixed_policy_correlation_not_policy_gradient_proof"
```

- [ ] **Step 5: Run tests and commit**

```bash
/Users/rigarg/.local/bin/visor run python3 -m pytest tests/test_analyze_rl_reward_gate_correlation.py -q
/Users/rigarg/.local/bin/visor run git add scripts/analyze_rl_reward_gate_correlation.py tests/test_analyze_rl_reward_gate_correlation.py
/Users/rigarg/.local/bin/visor run git commit -m "feat: add critic reward gate correlation diagnostic"
```

---

### Task 6: Queue Gating, Throughput Topology, and Config Updates

**Files:**
- Modify: `configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml`
- Modify: `scripts/nano_rl_queue.py`
- Modify: `external/natural_language_autoencoders/configs/rl.sh`
- Modify: `external/natural_language_autoencoders/nla/system_metrics.py`
- Modify: `tests/test_nano_rl_queue.py`
- Modify: `tests/test_nla_system_metrics.py`

- [ ] **Step 1: Update queue tests for diagnostics and structured throughput**

Update the checked-in queue tests to assert:

```python
self.assertEqual(item_by_name["r33-component-rl-8h100-tier1-probe-rb64-n8-gb512-lr1e5-rollout8-v256t256"]["status"], "blocked")
self.assertEqual(queue_doc["defaults"]["env"]["NLA_ROLLOUT_SUMMARY_STDOUT"], "1")
self.assertEqual(queue_doc["defaults"]["env"]["NLA_FAILED_EXTRACTION_REWARD"], "-2.0")
```

Also add/keep coverage for the throughput work already introduced in the code:

- Structured topology rendering from YAML:
  - `resources.actor_gpus`, `resources.critic_gpus`, `resources.rollout_gpus`, `resources.min_actor_gpus`;
  - `sglang.tensor_parallel_size`, `sglang.base_gpu_id`, `sglang.rollout_num_gpus_per_engine`.
- Managed SGLang command rewriting:
  - `--tp-size` and `--base-gpu-id` are rewritten from the structured `sglang` config;
  - `sglang.tensor_parallel_size` must match `sglang.rollout_num_gpus_per_engine`.
- Runtime validation before launch:
  - reject `training.async_training=true` with `training.colocate=true`;
  - reject `training.ref_log_probs_placement=critic` until the Miles runtime actually supports critic-side ref logprobs;
  - reject `training.colocate=true` with external/managed SGLang.
- Env propagation:
  - `NLA_WORKSPACE_GPUS`, `NLA_ACTOR_GPUS`, `NLA_CRITIC_GPUS`, `NLA_ROLLOUT_GPUS`, `NLA_ROLLOUT_GPUS_PER_ENGINE`, `NLA_SGLANG_TP_SIZE`, `NLA_SGLANG_BASE_GPU_ID`;
  - throughput flags such as `GRADIENT_CHECKPOINTING`, `FSDP_CPU_OFFLOAD`, `FSDP_CPU_BACKEND`, `OFFLOAD_TRAIN`, `OFFLOAD_ROLLOUT`, `OFFLOAD_ROLLOUT_LEVEL`, `COLOCATE`, and `NLA_REF_LOG_PROBS_PLACEMENT`.

- [ ] **Step 2: Update queue config**

Under `defaults.env`, add the diagnostics defaults:

```yaml
    NLA_ROLLOUT_SUMMARY_STDOUT: "1"
    NLA_FAILED_EXTRACTION_REWARD: "-2.0"
```

Keep the completed `5e-6 rollout8` run as the Stage-A pre-patch probe. Keep the `1e-5`,
medium, and hero quality runs blocked until Stage A/Stage B pass. Do **not** change LR and
KL together unless that confound is explicitly recorded; the current checked-in follow-up
may keep `kl_loss_coef: 0.0003` while blocked, and a later KL change should preferably add
a `5e-6 + KL=1e-3` reference arm.

Represent throughput experiments as explicit queue items, not shell/script edits:

```yaml
- name: r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-tp1-actor5
  status: blocked
  resources:
    actor_gpus: 5
    critic_gpus: 2
    rollout_gpus: 1
    min_actor_gpus: 5
  sglang:
    tensor_parallel_size: 1
    base_gpu_id: 5
    rollout_num_gpus_per_engine: 1

- name: r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-tp1-actor5-cpuoffload-nockpt-mb2
  status: blocked
  training:
    actor_micro_batch: 2
    gradient_checkpointing: false
    fsdp_cpu_offload: true
    fsdp_cpu_backend: gloo

- name: r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-async-tp2
  status: blocked
  training:
    async_training: true

- name: r33-component-rl-8h100-tier1-fit-rb32-n16-gb512-lr5e6-tp1-actor5
  status: pending
  rollout:
    rollout_batch_size: 32
    n_samples_per_prompt: 16
    global_batch_size: 512
```

The `tp1-actor5` variants are engineering canaries: compare generation time, ref-logprob
time, actor-train time, GPU utilization, and OOM headroom against the completed `tp2`
actor4 fit. The `rb32/n16/gb512` item is a signal canary: it increases GRPO samples per
prompt while keeping generated samples per update at 512. None of these canaries promotes
RL by itself.

- [ ] **Step 3: Update `rl.sh` throughput plumbing**

Ensure `external/natural_language_autoencoders/configs/rl.sh`:

- exports topology env vars (`NLA_WORKSPACE_GPUS`, `NLA_ACTOR_GPUS`, `NLA_CRITIC_GPUS`, `NLA_ROLLOUT_GPUS`, `NLA_ROLLOUT_GPUS_PER_ENGINE`, `NLA_SGLANG_TP_SIZE`, `NLA_SGLANG_BASE_GPU_ID`);
- prints a `[NLA RL CONFIG] topology ...` line before launch;
- includes the topology/throughput env vars in `NLA_TRAIN_ENV_KEYS`;
- maps `GRADIENT_CHECKPOINTING`, `OFFLOAD_TRAIN`, `OFFLOAD_ROLLOUT`, `OFFLOAD_ROLLOUT_LEVEL`, `COLOCATE`, `FSDP_CPU_OFFLOAD`, and `FSDP_CPU_BACKEND` to the Miles CLI flags.

- [ ] **Step 4: Update system metrics**

Ensure `external/natural_language_autoencoders/nla/system_metrics.py`:

- logs `nla/system/topology_*` metrics from the topology env vars;
- imports `torch` lazily so parser/tests still work in non-CUDA or minimal local envs;
- keeps all-GPU parsing tolerant of NVIDIA-SMI field drift.

- [ ] **Step 5: Run tests and commit**

```bash
/Users/rigarg/.local/bin/visor run python3 -m pytest tests/test_nano_rl_queue.py tests/test_nla_system_metrics.py -q
/Users/rigarg/.local/bin/visor run git add configs/nano_rl/r33_component_qwen_comparable_queue_8h100_len512.yaml scripts/nano_rl_queue.py external/natural_language_autoencoders/configs/rl.sh external/natural_language_autoencoders/nla/system_metrics.py tests/test_nano_rl_queue.py tests/test_nla_system_metrics.py
/Users/rigarg/.local/bin/visor run git commit -m "feat: add rl throughput topology controls"
```

---

### Task 7: Grad-Norm and KL Caveats Without Risky Runtime Patch

**Files:**
- Modify: `docs/rl_logbook.md`

- [ ] **Step 1: Add documentation caveat**

Append:

```markdown
### PPO KL and Grad-Norm Metric Caveats

The built-in `train/ppo_kl` has repeatedly logged exactly `0.0` under current single-epoch GRPO settings. Treat it as an implementation/measurement artifact, not proof of no policy movement. Until a post-step heldout KL probe is implemented, use `rollout/log_probs - rollout/ref_log_probs`, `train/train_rollout_logprob_abs_diff`, true advantage stats, and the final paired gate as movement indicators.

When `--nla-skip-grad-norm` is enabled, `train/grad_norm=0.0` is a sentinel from the Miles patch path, not proof of zero gradient. Do not use it for learning diagnosis.
```

- [ ] **Step 2: Commit**

```bash
/Users/rigarg/.local/bin/visor run git add docs/rl_logbook.md
/Users/rigarg/.local/bin/visor run git commit -m "docs: clarify rl kl and grad norm caveats"
```

Do not edit `0005_fsdp_skip_grad_norm_debug.patch` in this pass. A safe runtime change needs a separate patch-application test and a known-good Miles checkout.

---

### Task 8: Final Probe Artifact Sync and Go/No-Go

**Files:**
- Modify: `docs/rl_logbook.md`

- [ ] **Step 1: Sync final artifacts using the established staging archive pattern**

On RunAI, create a staging dir, copy expected files, write an explicit missing-file manifest, then tar the staging dir. Do not suppress tar errors.

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
set -euo pipefail
RUN_DIR=/workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/r33_component_rl_8h100_tier1_probe_rb64_n8_gb512_lr5e6_rollout8_v256t256
STAGE=/tmp/r33_probe8_final_stage
OUT=/tmp/r33_probe8_final_logs.tgz
rm -rf "$STAGE" "$OUT"
mkdir -p "$STAGE/current_run" "$STAGE/status"
missing=0
for f in train.log sglang_service_0.log roundtrip_iter_0000008_v256_t256_report.json roundtrip_iter_0000008_v256_t256_report_generated.jsonl; do
  if [ -f "$RUN_DIR/$f" ]; then cp "$RUN_DIR/$f" "$STAGE/current_run/"; else echo "missing $f" | tee -a "$STAGE/status/missing_files.txt"; missing=1; fi
done
[ -d "$RUN_DIR/wandb" ] && cp -a "$RUN_DIR/wandb" "$STAGE/current_run/wandb"
find "$RUN_DIR" -maxdepth 4 -type f -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" | sort > "$STAGE/status/file_inventory.txt"
tar -czf "$OUT" -C "$STAGE" .
sha256sum "$OUT"
exit "$missing"
'
```

If this exits nonzero because final report/generated JSONL is missing, do not proceed to diagnostics; poll completion again.

- [ ] **Step 2: Stream the archive to local and decode (macOS-compatible)**

Stream the remote tgz as base64 over the exec channel, decode locally with BSD `base64`:

```bash
/Users/rigarg/.local/bin/visor run bash -lc '
set -euo pipefail
LOCAL_DIR=/Users/rigarg/Desktop/agents/research-projects/nano30b-nla-pilot/artifacts/runai_rl/20260625_r33_rl_probe8_final
mkdir -p "$LOCAL_DIR"
/Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- \
  bash -lc "base64 /tmp/r33_probe8_final_logs.tgz" > "$LOCAL_DIR/r33_probe8_final_logs.tgz.b64"
base64 -D -i "$LOCAL_DIR/r33_probe8_final_logs.tgz.b64" -o "$LOCAL_DIR/r33_probe8_final_logs.tgz"
rm -f "$LOCAL_DIR/r33_probe8_final_logs.tgz.b64"
tar -xzf "$LOCAL_DIR/r33_probe8_final_logs.tgz" -C "$LOCAL_DIR"
ls -R "$LOCAL_DIR" | head -40
'
```

base64-over-exec can truncate on multi-MB payloads. If `tar -xzf` fails, the stream was
truncated: re-do Step 1 **excluding `wandb`** (the analyzer + correlation diagnostic only
need `train.log`, the report JSON, and the generated JSONL), or use a file-copy/rsync
mechanism instead of base64-over-exec.

- [ ] **Step 3: Run log analyzer and correlation diagnostic**

First run the log analyzer locally:

```bash
/Users/rigarg/.local/bin/visor run python3 scripts/analyze_nla_rl_run.py \
  --train-log artifacts/runai_rl/20260625_r33_rl_probe8_final/current_run/train.log \
  --output-json artifacts/runai_rl/20260625_r33_rl_probe8_final/train_log_summary.json
```

The critic recompute must run **where the critic checkpoint lives (RunAI)**. Preflight that
the value head survived `cleanup_hf`/`cleanup_actor_checkpoint` (config:120-121) before
trusting any reward number, then run the diagnostic with the template/`mse_scale` sourced
from the critic dir (no parquet template flag, no `--train-parquet`):

```bash
/Users/rigarg/.local/bin/visor run /Users/rigarg/.runai/bin/2.116.2/runai workspace exec train -p trustworthy-ai-inference -- bash -lc '
set -euo pipefail
RUN_DIR=/workspace/interp/outputs/nano30b-nla-pilot/rl_qwen_scale/r33_component_rl_8h100_tier1_probe_rb64_n8_gb512_lr5e6_rollout8_v256t256
AVDIR=/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-av-sft-r33-component-full/nano-av-r33-component-full-hero-lr1e4-cosine-warmup25-gb192-mb2-seq1152-dyn512/splits
CDIR=/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-r33-component-full/nano-ar-r33-component-full-hero-lr5e5-cosine-warmup25-gb192-mb96/checkpoints/iter_0001289/hf
test -f "$CDIR/value_head.safetensors" || { echo "MISSING $CDIR/value_head.safetensors (cleaned up?) — cannot recompute reward"; exit 1; }
cd /workspace/interp/code/nano30b-nla-pilot-current
python3 scripts/analyze_rl_reward_gate_correlation.py \
  --roundtrip-report-json "$RUN_DIR/roundtrip_iter_0000008_v256_t256_report.json" \
  --generated-jsonl "$RUN_DIR/roundtrip_iter_0000008_v256_t256_report_generated.jsonl" \
  --critic-checkpoint-dir "$CDIR" \
  --validation-parquet "$AVDIR/validation.parquet" \
  --test-parquet "$AVDIR/test.parquet" \
  --output-json "$RUN_DIR/reward_gate_correlation_report.json" \
  --batch-size 16 --device cuda
'
```

Then sync `reward_gate_correlation_report.json` back into the local
`artifacts/runai_rl/20260625_r33_rl_probe8_final/` directory.

- [ ] **Step 4: Apply decision gate (TWO stages — the current probe pre-dates Task 2 logging)**

The current 5e-6 probe ran on pre-patch code, so it has **no** `rollout/nla_advantage/*`
(its offline W&B holds only the mean advantage ≈0, and per-sample rewards are not stored,
so advantage std is **not recoverable** from its artifacts). Do not gate on advantage
variance using the current probe. Split the decision:

**Stage A — evaluate the current probe (artifacts in hand):**

```text
- parse usable_fraction >= 0.95 AND failed_fraction <= 0.04 (from train_log_summary / [NLA ROLLOUT] lines / W&B);
- the critic-recomputed reward-vs-gate-loss report has >= 64 paired rows on EACH split.
Outcome:
- If the final paired 256/256 gate already beats clean SFT on BOTH splits -> promote RL; the blocked
  `1e-5` quality follow-up is not needed.
- Else if reward-vs-gate-loss correlation is >= 0 (not negative) on either split -> the reward proxy is
  decoupled from the gate. Do NOT unblock quality follow-ups or throughput/signal canaries; fix the
  reward first (e.g. exclude failed samples from the group baseline, or use a gate-aligned reward).
- Else (correlation negative on both splits, gate not yet beaten) -> eligible for Stage B.
```

**Stage B — guard the NEXT instrumented run (this is where advantage stats exist).** The
next item must carry Task 1 stdout + Task 2 `[NLA ADVANTAGE]` logging (only after the
0004 patch is re-applied — Task 2 Step 5). Prefer the low-risk throughput/signal canary
`r33-component-rl-8h100-tier1-fit-rb32-n16-gb512-lr5e6-tp1-actor5` before the `1e-5`
quality follow-up: it keeps generated samples per update at 512 while testing `n=16` GRPO
groups and the 5-actor / 1-rollout TP1 topology. Check its **first 2-3 rollouts**,
aborting early if either fails:

```text
- true rollout/nla_advantage/std >= 0.5;
- rollout/nla_advantage/frac_zero <= 0.25.
```

Interpret the correlation as a **fixed-policy diagnostic only**: a negative correlation is
encouraging but does not prove a policy-gradient step will improve the gate. The Stage-B
checks are the live guard that advantages carry usable signal before spending the full
rollout budget. Also compare throughput metrics against the completed TP2 actor4 fit:
generation time, ref-logprob time, actor-train time, all-GPU utilization, and OOM headroom.

- [ ] **Step 5: Record decision**

Append a concrete decision entry to `docs/rl_logbook.md` with paths to final paired report, train summary, correlation report, and decision.

---

### Task 9: Verification, Branching, and Push

**Files:**
- No new files beyond previous tasks.

- [ ] **Step 1: Run focused tests**

```bash
/Users/rigarg/.local/bin/visor run python3 -m pytest \
  tests/test_nla_rl_metrics.py \
  tests/test_nla_reward_config.py \
  tests/test_analyze_nla_rl_run.py \
  tests/test_analyze_rl_reward_gate_correlation.py \
  tests/test_nano_rl_queue.py \
  tests/test_nla_system_metrics.py \
  -q
```

- [ ] **Step 2: Run broader RL-adjacent tests**

```bash
/Users/rigarg/.local/bin/visor run python3 -m pytest \
  tests/test_nano_rl_queue.py \
  tests/test_nla_system_metrics.py \
  tests/test_nla_rl_metrics.py \
  tests/test_nano_av_ar_roundtrip_gate.py \
  -q
```

- [ ] **Step 3: Use a branch unless explicitly told to push main**

```bash
/Users/rigarg/.local/bin/visor run git checkout -b codex/r33-rl-observability-throughput
```

If already on a branch, keep it. Do not push to `main` until the user asks.

- [ ] **Step 4: Push branch**

```bash
/Users/rigarg/.local/bin/visor run git push origin codex/r33-rl-observability-throughput
```

---

## Self-Review

- First-round review blockers addressed:
  - B1: correlation diagnostic requires critic recomputation on the exact gate generated rows (Task 5).
  - B2: no positional proxy; true advantage stats come from `rollout_data["advantages"]` via the Miles patch (Task 2).
  - B3: the arithmetically-wrong group-stat test is gone (the group block was removed — see N1).
  - B4: N/A — no in-hook reward/group pairing remains.
  - D1: advantage gate uses a real threshold (`std >= 0.5`, `frac_zero <= 0.25`) — see N2 for where it is applied.
  - D2: fixed-policy correlation caveat added (Task 5, Task 8).
  - D3: `-1.0` ablation removed (Task 3).
  - P1: LR+KL confound flagged (Task 6); P2: branch-first push (Task 9); P3: staging fails loudly (Task 8).
- Second-round review blockers addressed:
  - N1: Task 1 group/metadata stats removed (samples carry no group id → inert); only stdout reward/parse surfacing remains. Advantage variance is Task 2 only.
  - N2: go/no-go split into Stage A (current pre-patch probe: gate + correlation) and Stage B (next instrumented run's first rollouts: advantage std/frac_zero). The unsatisfiable "advantage std on the current probe" condition is gone.
  - N3: the `rollout/nla_group/count` KeyError test is removed with the group block.
  - N4: `wandb.log(..., step=rollout_id)` removed from the patch; rank-0-guarded `logger.info` only.
  - N5: critic recompute hard-asserts `value_head.safetensors` exists and is loaded (not meta); Task 8 preflights it against `cleanup_hf`.
  - N6: correlation script reuses `nla.reward._mse_to_reward` / `nla.schema.extract_explanation` and sources the template from the critic sidecar; bogus `--critic-template-source`/`--train-parquet` flags dropped.
  - N7: patch dry-run runs against a pristine `git reset --hard 051cd15` checkout; an explicit re-apply-to-code-root step is required before the next run.
  - N8: concrete base64-stream + macOS decode for artifact sync, with a wandb-exclude fallback if it truncates.
- Throughput alignment addressed:
  - Structured topology controls are now part of Task 6 (`resources.*`, `sglang.*`, `training.*`) instead of undocumented shell edits.
  - `rl.sh` must forward topology/offload/gradient-checkpointing env vars and print the active topology before launch.
  - System metrics must log topology fields so W&B/offline logs can compare TP2 actor4, TP1 actor5, CPU-offload/mb2, async, and n=16 canaries.
  - Stage B now prefers the `rb32/n16/gb512` TP1 actor5 signal canary before a `1e-5` quality follow-up, while still requiring advantage and gate-alignment evidence.
- Remaining hard work:
  - The critic-recompute script loads the large critic checkpoint and must run on RunAI; validate value-head load there.
  - The `0004` advantage patch must be re-applied to the run code root (Task 2 Step 5) before the next launch, or `[NLA ADVANTAGE]` never appears and the Stage-B gate cannot be evaluated.
  - Throughput canaries should be judged on measured timing/utilization and OOM headroom only; do not promote RL quality from a speed canary without paired gate improvement.
