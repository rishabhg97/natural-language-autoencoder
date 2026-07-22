# NLA Offline Observatory

This package owns the reproducible evidence pipeline for the CPU-only NLA
dashboard. It deliberately excludes generated evidence, checkpoints, datasets,
and the dashboard application bundle.

## Layout

- `qualify_evidence.py`: validate and bind source evidence.
- `build_corpus.py`: select the fixed validation panel.
- `run_model_batches.py`: AV, AR, functional, and trace model phases.
- `poetry_planning.py`: resumable poetry planning, reconstruction, and causal
  steering phases for the TRACE planning lens.
- `model_runtime.py`: shared scoring, reconstruction, and generation runtime.
- `compute_geometry.py`: offline geometry tables.
- `compute_interventions.py`: offline causal and attribution aggregates.
- `build_bundle.py`: construct the static dashboard bundle.
- `verify_bundle.py`: fail-closed bundle integrity and provenance checks.
- `queue.py`, `queue_after_state.py`, and `pipeline_supervisor.py`: durable,
  config-driven orchestration.

Run modules from the repository root, for example:

```bash
python -m observatory.queue --queue configs/nano_viz/offline_observatory_gpu_queue.yaml
python -m observatory.verify_bundle --config configs/nano_viz/offline_observatory_bundle.yaml
python -m observatory.queue --queue configs/nano_viz/offline_observatory_poetry_queue.yaml
```

The legacy `scripts/nano_viz_*.py` commands are thin compatibility launchers.
New code should import from `observatory` and use module execution.

All heavyweight outputs belong outside Git under the configured evidence root.
Only configuration, source, tests, schemas, and documentation are committed.
