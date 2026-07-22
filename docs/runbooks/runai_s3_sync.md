# RunAI Source Sync Via S3

Use this when RunAI cannot reach GitHub directly. S3 is the hub between the Mac
workspace and the RunAI workspace.

Bucket prefix:

`s3://team-ipp-trustworthy-ai/nano30b-nla-pilot/source-sync/`

## Goals

- Keep the Mac repo, RunAI code root, and GitHub in sync for source/config/docs.
- Exclude heavy data, activations, checkpoints, temporary HF conversions, W&B
  run payloads, caches, and secrets.
- Make RunAI a source peer, not an invisible fork.

## Suggested Flow

1. On Mac, verify local source state:

   ```bash
   git status --short
   git diff --stat
   ```

2. Build a source archive from tracked files plus explicitly selected untracked
   source/config/docs files. Do not include ignored artifacts.

3. Upload the archive to S3 under a timestamped key.

4. On RunAI, download the archive and unpack into:

   `/workspace/interp/code/nano30b-nla-pilot-current`

5. Run the lightweight tests in the RunAI venv:

   ```bash
   /workspace/interp/.venv/bin/python -m pytest \
     tests/test_nano_av_runner_spec.py \
     tests/test_nano_ar_hpo_queue.py -q
   ```

6. If RunAI generated source/config/doc changes, package those back to S3 and
   merge them on Mac before committing to GitHub.

7. Push from Mac to GitHub only after the Mac tree contains the intended
   superset.

## Verification Checklist

- `git status --short` on Mac has only intended changes.
- RunAI code root has the same source/config/doc files as Mac.
- No secrets are printed or archived.
- No large artifacts are included.
- Tests pass in the RunAI venv for changed code paths.
- Commit and push happen from Mac.

## Exclude Patterns

Keep these out of source-sync archives:

```text
artifacts/
outputs/
runs/introspection/
wandb/
offline-run-*/
*.parquet
*.arrow
*.safetensors
*.pt
*.pth
*.ckpt
*.bin
*.npy
*.npz
*.log
*.pid
.env
```

