# Nano AR Wide Probe Queue Implementation Plan

**Goal:** Build and launch a lightweight YAML-driven AR-SFT probe queue so short
Nano AR HPO train/eval jobs run sequentially without manual pinging.

**Architecture:** Add a Python queue CLI that reuses `scripts/nano_av_runner.py`
to prepare Miles/FSDP2 training plans from existing experiment YAML configs,
then runs the standard `scripts/eval_nano_ar_miles_checkpoint.py` `512/512`
heldout eval. The queue manifest remains YAML, lives on the RunAI PVC, and is
updated after each train/eval transition.

**Tech Stack:** Python standard library, PyYAML, existing Nano AR YAML configs,
existing Miles runner, existing AR eval script, existing HPO study helper,
`pytest`/`unittest`, RunAI `train` workspace.

---

### Task 1: Queue Model And Validation

**Files:**
- Create: `scripts/nano_ar_hpo_queue.py`
- Create: `tests/test_nano_ar_hpo_queue.py`

- [ ] **Step 1: Write failing tests for queue parsing, status selection, and long-eval rejection**

Add tests that import `scripts/nano_ar_hpo_queue.py` and verify:

```python
def test_load_queue_rejects_long_eval_limits():
    queue_path.write_text(
        '''
        schema_version: nano_ar_hpo_queue.v1
        defaults: {validation_limit: 2048, test_limit: 512}
        items:
          - name: too-long
            config: configs/nano_ar/hpo/example.yaml
            status: pending
        '''
    )
    with pytest.raises(queue.QueueError, match="validation_limit"):
        queue.load_queue(queue_path)


def test_next_pending_skips_complete_and_running_items():
    loaded = queue.validate_queue(
        {
            "schema_version": "nano_ar_hpo_queue.v1",
            "defaults": {"validation_limit": 512, "test_limit": 512},
            "items": [
                {"name": "done", "config": "a.yaml", "status": "complete"},
                {"name": "active", "config": "b.yaml", "status": "training"},
                {"name": "next", "config": "c.yaml", "status": "pending"},
            ],
        },
        source=queue_path,
    )
    assert queue.next_pending_index(loaded) == 2
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_nano_ar_hpo_queue.py -q
```

Expected: fail because `scripts/nano_ar_hpo_queue.py` does not exist.

- [ ] **Step 3: Implement the queue data model**

Create `scripts/nano_ar_hpo_queue.py` with:

```python
VALID_STATUSES = {"pending", "training", "eval_running", "complete", "failed"}
MAX_AUTOMATED_EVAL_LIMIT = 512

class QueueError(ValueError):
    pass

def load_queue(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    return validate_queue(data, source=path)

def validate_queue(data: dict[str, Any], *, source: Path) -> dict[str, Any]:
    if data.get("schema_version") != "nano_ar_hpo_queue.v1":
        raise QueueError("schema_version must be nano_ar_hpo_queue.v1")
    defaults = data.setdefault("defaults", {})
    validation_limit = int(defaults.get("validation_limit", 512))
    test_limit = int(defaults.get("test_limit", 512))
    if validation_limit > MAX_AUTOMATED_EVAL_LIMIT:
        raise QueueError("validation_limit above 512 is not allowed in automated queue")
    if test_limit > MAX_AUTOMATED_EVAL_LIMIT:
        raise QueueError("test_limit above 512 is not allowed in automated queue")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise QueueError("items must be a non-empty list")
    for index, item in enumerate(items):
        if item.get("status", "pending") not in VALID_STATUSES:
            raise QueueError(f"item {index} has invalid status")
        if not item.get("name") or not item.get("config"):
            raise QueueError(f"item {index} requires name and config")
        item.setdefault("status", "pending")
    return data

def next_pending_index(queue_doc: dict[str, Any]) -> int | None:
    for index, item in enumerate(queue_doc["items"]):
        if item.get("status") == "pending":
            return index
    return None
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
python -m pytest tests/test_nano_ar_hpo_queue.py -q
```

Expected: queue parsing tests pass.

### Task 2: Atomic Queue Updates And Command Construction

**Files:**
- Modify: `scripts/nano_ar_hpo_queue.py`
- Modify: `tests/test_nano_ar_hpo_queue.py`

- [ ] **Step 1: Write failing tests for atomic status updates and eval command rendering**

Add tests that verify:

```python
def test_update_item_status_writes_yaml_with_artifacts():
    queue.write_queue(queue_path, queue_doc)
    queue.update_item(queue_path, 0, status="training", run_dir="/runs/a")
    reloaded = yaml.safe_load(queue_path.read_text())
    assert reloaded["items"][0]["status"] == "training"
    assert reloaded["items"][0]["run_dir"] == "/runs/a"


def test_eval_command_uses_512_limits_and_standard_controls():
    command = queue.build_eval_command(
        python_bin="/venv/bin/python",
        code_root=Path("/code"),
        checkpoint_dir=Path("/run/checkpoints/iter_0000128"),
        train_parquet=Path("/splits/train_padded.parquet"),
        validation_parquet=Path("/splits/validation.parquet"),
        test_parquet=Path("/splits/test.parquet"),
        report_json=Path("/run/eval.json"),
        validation_limit=512,
        test_limit=512,
        batch_size=4,
        controls=["teacher", "teacher_shuffled", "blank", "generic", "mean", "source_context", "source_raw"],
    )
    assert command[:2] == ["/venv/bin/python", "scripts/eval_nano_ar_miles_checkpoint.py"]
    assert "--validation-limit" in command
    assert command[command.index("--validation-limit") + 1] == "512"
    assert "--controls" in command
    assert "source_raw" in command
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_nano_ar_hpo_queue.py -q
```

Expected: fail because update and command helpers are missing.

- [ ] **Step 3: Implement atomic write and eval command helpers**

Add:

```python
def write_queue(path: Path, queue_doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(queue_doc, sort_keys=False))
    tmp.replace(path)

def update_item(path: Path, index: int, **fields: Any) -> dict[str, Any]:
    queue_doc = load_queue(path)
    item = queue_doc["items"][index]
    item.update({key: value for key, value in fields.items() if value is not None})
    write_queue(path, queue_doc)
    return item

def build_eval_command(...args...) -> list[str]:
    return [
        python_bin,
        "scripts/eval_nano_ar_miles_checkpoint.py",
        "--checkpoint-dir", str(checkpoint_dir),
        "--train-parquet", str(train_parquet),
        "--validation-parquet", str(validation_parquet),
        "--test-parquet", str(test_parquet),
        "--validation-limit", str(validation_limit),
        "--test-limit", str(test_limit),
        "--batch-size", str(batch_size),
        "--controls", *controls,
        "--report-json", str(report_json),
    ]
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
python -m pytest tests/test_nano_ar_hpo_queue.py -q
```

Expected: all queue helper tests pass.

### Task 3: Watcher Execution Loop

**Files:**
- Modify: `scripts/nano_ar_hpo_queue.py`
- Modify: `tests/test_nano_ar_hpo_queue.py`

- [ ] **Step 1: Write failing dry-run tests for preparing a pending item without launching training**

Add a test that creates a temporary queue with one pending item and a fake config,
then calls:

```python
result = queue.process_next_item(queue_path, once=True, dry_run=True)
assert result["status"] == "dry_run"
assert "train_command" in result
assert "eval_command" in result
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_nano_ar_hpo_queue.py -q
```

Expected: fail because `process_next_item` is missing.

- [ ] **Step 3: Implement `process_next_item`**

Behavior:

```python
def process_next_item(queue_path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    queue_doc = load_queue(queue_path)
    index = next_pending_index(queue_doc)
    if index is None:
        return {"status": "idle"}
    item = queue_doc["items"][index]
    spec = nano_av_runner.load_and_validate_spec(resolve_path(item["config"], queue_path))
    plan = nano_av_runner.prepare_run(spec, run_id=item.get("run_id") or item["name"])
    expected_checkpoint = expected_checkpoint_for_plan(plan)
    eval_paths = eval_paths_for_plan(plan)
    eval_command = build_eval_command(...)
    if dry_run:
        return {"status": "dry_run", "train_command": plan["command"], "eval_command": eval_command}
    update_item(queue_path, index, status="training", run_dir=plan["run_dir"], expected_checkpoint=str(expected_checkpoint))
    run training synchronously with stdout/stderr to train.log
    update_item(queue_path, index, status="eval_running", eval_report=str(eval_report))
    run eval synchronously with stdout/stderr to eval log
    record trial through nano_ar_hpo_study.upsert_trial
    update_item(queue_path, index, status="complete", completed_at=utc timestamp)
```

Failure handling:

```python
except subprocess.CalledProcessError as exc:
    update_item(queue_path, index, status="failed", failure=str(exc), failed_at=utc timestamp)
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
python -m pytest tests/test_nano_ar_hpo_queue.py -q
```

Expected: dry-run and helper tests pass.

### Task 4: First Probe Configs And Queue YAML

**Files:**
- Create: `configs/nano_ar/hpo/r27_wide_probe_best1547_lr3e5_cosine_128steps.yaml`
- Create: `configs/nano_ar/hpo/r27_wide_probe_best1547_lr1e5_constant_128steps.yaml`
- Create: `configs/nano_ar/hpo/r27_wide_probe_best1547_lr5e6_cosine_128steps.yaml`
- Create: `configs/nano_ar/hpo/r27_wide_probe_fullscan_lr2e5_cosine_192steps.yaml`
- Create: `configs/nano_ar/hpo/r27_wide_probe_fullscan_lr5e5_cosine_128steps.yaml`
- Create: `configs/nano_ar/hpo/r27_wide_probe_queue.yaml`
- Modify: `tests/test_nano_av_runner_spec.py`

- [ ] **Step 1: Write failing validation coverage for checked-in HPO configs**

Extend `test_checked_in_specs_are_valid` to include the five new HPO configs.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_nano_av_runner_spec.py::NanoAVRunnerSpecTests::test_checked_in_specs_are_valid -q
```

Expected: fail because the configs do not exist.

- [ ] **Step 3: Add the probe configs**

Each config must:

- use `training.objective: ar_sft`
- set `dataset.materialize_splits: false`
- point `paths.input_ar_sft` at the existing fullscan `splits/train_padded.parquet`
- use `eval.validation_limit: 512` and `eval.test_limit: 512`
- use `checkpoint.keep_last: 1`
- use `checkpoint.no_save_optim: true`
- set `checkpoint.save_interval` equal to `training.resume_steps`

The first batch should include:

```text
best1547 lr=3e-5 cosine warmup=25 min_lr_ratio=0.1 steps=128
best1547 lr=1e-5 constant warmup=0 steps=128
best1547 lr=5e-6 cosine warmup=10 min_lr_ratio=0.1 steps=128
fullscan iter1291 lr=2e-5 cosine warmup=25 min_lr_ratio=0.1 steps=192
fullscan iter1291 lr=5e-5 cosine warmup=10 min_lr_ratio=0.1 steps=128
```

- [ ] **Step 4: Add queue YAML**

`configs/nano_ar/hpo/r27_wide_probe_queue.yaml` should include the five configs,
default `validation_limit: 512`, `test_limit: 512`, `batch_size: 4`, standard
AR controls, and all item statuses as `pending`.

- [ ] **Step 5: Run validation tests**

Run:

```bash
python -m pytest tests/test_nano_av_runner_spec.py::NanoAVRunnerSpecTests::test_checked_in_specs_are_valid tests/test_nano_ar_hpo_queue.py -q
```

Expected: pass.

### Task 5: Remote Dry Run And Watcher Launch

**Files:**
- Modify: `docs/nano_ar_hpo_study.md`
- Modify: `docs/experiment_logbook.md`

- [ ] **Step 1: Mirror code and configs into the RunAI `train` workspace**

Run from the local repo:

```bash
rsync -a scripts/nano_ar_hpo_queue.py tests/test_nano_ar_hpo_queue.py configs/nano_ar/hpo/r27_wide_probe_*.yaml configs/nano_ar/hpo/r27_wide_probe_queue.yaml docs/superpowers/specs/2026-06-02-nano-ar-wide-probe-queue-design.md /workspace/interp/code/nano30b-nla-pilot-current/
```

Use the actual RunAI copy mechanism available in this environment if direct
`rsync` is not available.

- [ ] **Step 2: Copy queue YAML to the PVC queue root**

Run inside `train`:

```bash
mkdir -p /workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue
cp /workspace/interp/code/nano30b-nla-pilot-current/configs/nano_ar/hpo/r27_wide_probe_queue.yaml \
  /workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/queue.yaml
```

- [ ] **Step 3: Remote dry run**

Run inside `train`:

```bash
cd /workspace/interp/code/nano30b-nla-pilot-current
/workspace/interp/.venv/bin/python scripts/nano_ar_hpo_queue.py \
  /workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/queue.yaml \
  --dry-run --once
```

Expected: prints one train command and one eval command, both with `512` limits.

- [ ] **Step 4: Launch watcher**

Run inside `train`:

```bash
cd /workspace/interp/code/nano30b-nla-pilot-current
nohup /workspace/interp/.venv/bin/python scripts/nano_ar_hpo_queue.py \
  /workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/queue.yaml \
  --poll-seconds 60 \
  > /workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/watcher.log 2>&1 &
echo $! > /workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/watcher.pid
```

- [ ] **Step 5: Verify watcher status**

Run inside `train`:

```bash
cat /workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/watcher.pid
tail -40 /workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/watcher.log
python scripts/nano_ar_hpo_queue.py \
  /workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/queue.yaml \
  --status
```

Expected: one item is `training` or `eval_running`; remaining items are
`pending`.

- [ ] **Step 6: Document launch**

Update `docs/nano_ar_hpo_study.md` and `docs/experiment_logbook.md` with:

- queue path
- watcher PID/log path
- probe list
- eval limit policy
- no `2048/2048` automated evals

- [ ] **Step 7: Commit implementation milestone**

Run:

```bash
git add scripts/nano_ar_hpo_queue.py tests/test_nano_ar_hpo_queue.py configs/nano_ar/hpo/r27_wide_probe_*.yaml configs/nano_ar/hpo/r27_wide_probe_queue.yaml docs/nano_ar_hpo_study.md docs/experiment_logbook.md
git commit -m "nano30b: add AR HPO probe queue"
```

## Self-Review

- Spec coverage: the plan implements YAML queueing, serial watcher behavior,
  `512/512` evals, status persistence, storage-conscious checkpointing, HPO
  ledger updates, tests, remote dry run, and watcher launch.
- Placeholder scan: no task contains `TBD`, `TODO`, or an unspecified command.
- Type consistency: queue statuses are consistently `pending`, `training`,
  `eval_running`, `complete`, and `failed`; queue item fields match the design
  spec.
