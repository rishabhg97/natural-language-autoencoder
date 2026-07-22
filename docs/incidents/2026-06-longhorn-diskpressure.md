# 2026-06 Longhorn DiskPressure Incident

## Summary

RunAI `train` was repeatedly evicted because the GH200 node went under
kubelet ephemeral-storage pressure. The training container itself was not the
large disk user; it was evicted as a victim.

## Observed Facts

- Eviction reason: node low on ephemeral storage.
- Container `train` ephemeral usage was small compared with the node pressure.
- `/workspace/interp` is a Longhorn-backed PVC and had grown large through
  checkpoints, temporary conversions, and experiment artifacts.
- Longhorn replica placement on GH200 node-local disks made node storage
  pressure sensitive to PVC actual size and replica rebuilds.
- The workspace became usable again after cleanup/trim/redeploy, but the
  structural risk remains if the PVC grows by hundreds of GB again while
  replicas sit on constrained GH200 node disks.

## Research Impact

- R33 AR `lr4e-5` reached a final checkpoint before eviction and was later
  evaluated from the saved checkpoint.
- Non-candidate checkpoint trees were cleaned to reduce pressure.
- Compact eval reports, train logs, run specs, and W&B offline logs were
  retained when possible.

## Prevention

- Keep only selected model/checkpoint candidates locally.
- Delete temporary HF conversions after eval.
- Use DCP -> temporary HF -> eval -> cleanup for AV evals.
- Upload worth-preserving compact model-only artifacts to S3.
- Prefer node/disk placement that keeps large Longhorn replicas off constrained
  GH200 root disks, or confirm adequate healthy replicas before moving/removing
  replicas.
- Track `/workspace/interp` usage before launching long or checkpoint-heavy
  jobs.

## Do Not Do

- Do not delete the PVC to fix pressure.
- Do not wipe `/var/lib/longhorn`.
- Do not force-remove the only healthy replica.
- Do not infer that GPU/CPU memory was the cause of this incident.

