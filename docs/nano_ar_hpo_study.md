# Nano NLA Offline HPO Study

This is a local recommendation layer for Nano NLA tuning. It currently supports
AR-SFT reconstruction trials and AV-SFT Phase 1/hero logs. It does not launch
RunAI jobs and does not select from train loss alone.

## Files

- AR study JSONL: `artifacts/nano_ar_hpo_study/trials.jsonl`
- AR suggestions JSON/Markdown: `artifacts/nano_ar_hpo_study/suggestions.json`,
  `artifacts/nano_ar_hpo_study/suggestions.md`
- AR Optuna-style export: `artifacts/nano_ar_hpo_study/optuna_trials.json`
- AV study JSONL: `artifacts/nano_av_hpo_study/trials.jsonl`
- AV suggestions JSON/Markdown: `artifacts/nano_av_hpo_study/suggestions.json`,
  `artifacts/nano_av_hpo_study/suggestions.md`
- AV Optuna-style export: `artifacts/nano_av_hpo_study/optuna_trials.json`
- Utility: `scripts/nano_ar_hpo_study.py`

`artifacts/` is gitignored, so force-add these small study files when they are
part of a milestone record.

## Current Seeded AV Logs

The AV study is seeded with two completed runs:

- Phase 1 lm_head-only baseline from `experiment_0523.md`: 29,913 rows,
  scale 75, lr `1e-4`, 10,000 steps, heldout real NLL `1.6051`; real h beats
  shuffled `1.7151`, zero `1.6812`, mean `1.7029`, and no-injection `1.8309`.
- Full Nano30B AV hero checkpoint:
  `nano-av-miles-fsdp2-r27-super-thinking-100k-hero-gloo-tokenized-gb192-mb8-save100-20260528T0110Z`.
  The v64/t64 objective NLL is `0.930576216429472`; validation real NLL is
  `0.9046209901571274`, test real NLL is `0.9565314427018166`, and real h beats
  mean, shuffled, zero, and no-injection controls on both splits.

## Record A Completed Trial

After a checkpoint eval report exists:

```bash
python scripts/nano_ar_hpo_study.py record \
  --task ar \
  --study-jsonl artifacts/nano_ar_hpo_study/trials.jsonl \
  --trial-name r27-best1547-lr1e5-bounded \
  --config configs/nano_ar/hpo/r27_best1547_continue_lr1e5_cosine_256steps.yaml \
  --run-dir /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-hpo/<run_id> \
  --eval-report /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-hpo/<run_id>/eval_<checkpoint>_v512_t512_winrates_report.json \
  --train-log /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-hpo/<run_id>/train.log \
  --status complete
```

The active objective is validation teacher directional MSE only. Test metrics
in older records are historical contamination and must not drive selection.
Controls and validation rowwise win rates are supporting metrics; the test
split is opened once after the checkpoint and analysis are locked.

For AV eval reports, pass `--task av` and use
`artifacts/nano_av_hpo_study/trials.jsonl`. The objective is heldout real-h NLL
(`objective_nll`); control gaps are retained as supporting metrics.

## Suggest Next Trials

```bash
python scripts/nano_ar_hpo_study.py suggest \
  --task ar \
  --study-jsonl artifacts/nano_ar_hpo_study/trials.jsonl \
  --top-n 6 \
  --out-json artifacts/nano_ar_hpo_study/suggestions.json \
  --out-md artifacts/nano_ar_hpo_study/suggestions.md
```

The suggestions are advisory. The current best completed trial is the long
`2e-5` Qwen-style finetune to `iter_0001547`, so the seeded study currently
prefers a bounded `2e-5` replay. The live `1e-5` bounded probe is recorded as
`running` so the suggester does not recommend the exact same parameter set.

For AV, run the same command with `--task av` and the
`artifacts/nano_av_hpo_study/` paths. The seeded AV study currently prefers
scale 75 / lr `1e-5` follow-ups around the successful hero configuration, with
warmup and cosine-min-LR variants made explicit in the Markdown output.

## Export For Optuna

```bash
python scripts/nano_ar_hpo_study.py export-optuna \
  --task ar \
  --study-jsonl artifacts/nano_ar_hpo_study/trials.jsonl \
  --out-json artifacts/nano_ar_hpo_study/optuna_trials.json
```

This writes a dependency-free Optuna-style payload with `direction=minimize`,
`objective=objective_nmse` for AR or `objective=objective_nll` for AV, and
trial records containing `number/state/value/params/user_attrs`. A notebook or
later wrapper can turn these into real Optuna `study.tell(...)` calls if we
decide to add the Optuna dependency.

## Selection Rule

- Green AR selection milestone: validation teacher directional MSE `<=0.30`.
- Usable AR selection milestone: validation `<=0.35` if bounded probes plateau.
- After selection lock, report the one-shot test metric without reopening HPO.
- Always require teacher to beat mean, shuffled, blank, generic, and
  source-context controls before promoting a checkpoint.
- Do not start RL until a milestone AR checkpoint is selected and documented.

## Wide Probe Queue

On `2026-06-02T19:06Z`, a lightweight YAML-driven AR HPO queue watcher was
launched in the RunAI `train` workspace.

- Queue manifest:
  `/workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/queue.yaml`
- Watcher PID file:
  `/workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/watcher.pid`
- Watcher log:
  `/workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/watcher.log`
- Remote code path:
  `/workspace/interp/code/nano30b-nla-pilot-current/scripts/nano_ar_hpo_queue.py`
- Automated eval limit policy: `512/512` only; no `2048/2048` evals are queued
  by the watcher.
- Controls: `teacher`, `teacher_shuffled`, `blank`, `generic`, `mean`,
  `source_context`, and `source_raw`.

Queued probes:

- `r27-wide-best1547-lr3e5-cos128`
- `r27-wide-best1547-lr1e5-constant128`
- `r27-wide-best1547-lr5e6-cos128`
- `r27-wide-fullscan-lr2e5-cos192`
- `r27-wide-fullscan-lr5e5-cos128`

The watcher runs one probe at a time, marks queue items as `training`,
`eval_running`, `complete`, or `failed`, and writes completed records to:

```text
/workspace/interp/outputs/nano30b-nla-pilot/ar_hpo_queue/trials.jsonl
```
