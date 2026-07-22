# Experiment Registry

This directory is intentionally tracked even though most of `runs/` is ignored.
It stores lightweight structured metadata only. Do not place logs, checkpoints,
parquets, W&B payloads, or activation artifacts here.

Primary file:

`experiments.yaml`

## Registry Rule

Every run that changes a decision should get a compact registry entry with:

- run id,
- task,
- layer,
- dataset,
- config,
- status,
- key metrics,
- artifact pointers,
- decision.

The registry is not a replacement for eval JSONs or W&B. It is the table of
contents that prevents run history from living only in chat and queue YAMLs.

Archived historical selection, invalidated for external claims:

`r33-corrected-k3-hero-lr1e5-update342-resume228-retry3`

No current registry entry is a release candidate.
