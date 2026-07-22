# Nano AR Wide Probe Queue Design

## Context

Nano30B AR-SFT is working but not yet at the desired AR milestone. The best
recent `2048/2048` heldout confirmation is around `0.441` average teacher
normalized MSE, while the project target is roughly `0.25-0.30` and the usable
fallback threshold is `<=0.35`.

The `1e-5` and `2e-5` bounded continuations from the current best checkpoint are
near-tied, which suggests the search may be too narrow around one basin. Long
`2048/2048` evals cost about 20 minutes each, so the next phase should favor
many cheap probes with `512/512` evals.

## Goal

Run a wide, lightweight Nano AR-SFT probe campaign that explores whether the
current `~0.44` teacher NMSE region is a local minimum and identifies a better
training regime before any RL or AV+AR tuning.

## Scope

- Run bounded AR-SFT probes only.
- Use `512/512` validation/test evals as the default automated gate.
- Do not run `2048/2048` evals in the automated queue.
- Explore both current-best continuations and a small number of escape probes
  from earlier/fullscan checkpoints.
- Keep W&B offline for all launched runs.
- Keep storage bounded to one exact-resume checkpoint per probe plus lightweight
  eval reports and logs.
- Do not start RL.
- Do not begin AV+AR tuning until an AR milestone checkpoint is selected.

## Success Criteria

- Any probe below the current `512/512` objective `0.4393` is interesting.
- A probe at or below `0.42` is a meaningful improvement.
- A probe at or below `0.40` should stop broad search and trigger exploitation
  around that regime.
- A promoted probe must still beat mean, shuffled teacher, blank, generic, and
  source-context controls on both validation and test splits.

## Probe Families

The phase should mix three families:

1. Narrow continuations from current best:
   - Start from the current best AR checkpoint region.
   - Sweep learning rate, min-LR ratio, warmup, and short step budgets.
   - Prefer `128-192` step probes unless a specific candidate justifies `256`.

2. Escape probes from earlier checkpoints:
   - Start some runs from the one-epoch fullscan checkpoint or earlier strong
     checkpoints.
   - Use wider LR/schedule variants to test whether the `iter_0001547` basin is
     limiting.

3. Cheap diagnostics through standard reports:
   - Use the existing `512/512` heldout report and controls to compare regimes.
   - Avoid long doc-source or per-bucket evals during this phase.

## Queue Design

Reuse YAML experiment configs as the unit of work. A lightweight queue file will
list config paths and expected run artifacts. The queue lives on the RunAI PVC
so it survives workspace restarts.

Suggested queue directory:

```text
/workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/
```

Suggested queue manifest:

```text
/workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/queue.yaml
```

Each queued item should include:

- `name`
- `config`
- `run_dir`
- `expected_checkpoint`
- `eval_report`
- `status`
- optional `notes`

Statuses:

- `pending`
- `training`
- `eval_running`
- `complete`
- `failed`

## Watcher Behavior

A single watcher process runs inside the RunAI `train` workspace.

Loop:

1. Acquire a lock to prevent two watchers from launching the same work.
2. Load `queue.yaml`.
3. Select the first `pending` item.
4. Launch training from that item config.
5. Wait for the expected checkpoint.
6. Launch the standard `512/512` AR eval.
7. Parse the eval report.
8. Update the queue item status.
9. Append or update the local HPO result ledger.
10. Move to the next pending item.

Failure handling:

- If training exits without the expected checkpoint, mark the item `failed`.
- If eval fails, retry once, then mark `failed`.
- If the watcher restarts, it resumes from `queue.yaml` and existing artifacts.
- It must not print secrets.
- It must keep W&B offline.

## Data Flow

```text
YAML experiment configs
  -> queue.yaml
  -> watcher in RunAI train workspace
  -> train run directory
  -> expected checkpoint
  -> 512/512 eval report
  -> queue status update
  -> AR HPO study ledger
  -> refreshed suggestions
```

## Testing And Verification

- Unit-test queue parsing and state transitions locally.
- Unit-test command construction without launching RunAI jobs.
- Dry-run the watcher against a fake queue and fake artifact files.
- Smoke-run one real queued item before filling the queue.
- Verify the watcher never schedules `2048/2048` evals by default.

## Non-Goals

- No Optuna service or database.
- No concurrent multi-GPU job packing.
- No RL launch.
- No AV+AR round-trip tuning.
- No long evals unless explicitly requested later.
