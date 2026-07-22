from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "reconcile_miles_runtime.py"
    spec = importlib.util.spec_from_file_location("reconcile_miles_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STALE_ACTOR = """import torch
import torch.distributed as dist
from miles.backends.training_utils.log_utils import aggregate_train_losses

class Actor:
    def train(self):
        if getattr(self.args, "nla_skip_grad_norm", False):
            grad_norm = 0.0
        else:
            nla_timing_clip_start = self._nla_timing_start()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            self._nla_timing_log("nla_timing_clip_grad_norm_raw", nla_timing_clip_start, rollout_id=rollout_id, step_id=step_id)

            nla_timing_full_tensor_start = self._nla_timing_start()
            grad_norm = grad_norm.full_tensor().item()
            self._nla_timing_log(
                "nla_timing_clip_grad_norm_full_tensor",
                nla_timing_full_tensor_start,
                rollout_id=rollout_id,
                step_id=step_id,
                grad_norm=grad_norm,
            )
        nla_timing_optimizer_start = self._nla_timing_start()
        loss_dict = aggregate_train_losses(losses_reduced, self.parallel_state)
"""

STALE_ARGUMENTS = """class FSDPArgs:
    nla_timing_debug: bool = False
    nla_skip_grad_norm: bool = False
    warmup_ratio: float = 0.03
"""


class ReconcileMilesRuntimeTests(unittest.TestCase):
    def test_applies_local_shard_grad_norm_contract_idempotently(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            miles_root = pathlib.Path(tmp)
            fsdp_root = miles_root / "miles" / "backends" / "fsdp_utils"
            fsdp_root.mkdir(parents=True)
            actor = fsdp_root / "actor.py"
            arguments = fsdp_root / "arguments.py"
            actor.write_text(STALE_ACTOR)
            arguments.write_text(STALE_ARGUMENTS)
            (miles_root / "train.py").write_text("print('train')\n")
            patches_root = pathlib.Path(tmp) / "miles_patches"
            patches_root.mkdir()
            (patches_root / "runtime.py").write_text("PATCH = True\n")

            report = module.reconcile_miles_runtime(
                miles_root,
                apply=True,
                miles_patches_root=patches_root,
            )

            self.assertTrue(report["passed"])
            self.assertIn(
                "clip_grad_norm_local_shards",
                actor.read_text(),
            )
            self.assertIn("clip_grad_norm_local_shards(", actor.read_text())
            self.assertIn(
                "from nla.audit_runtime import aggregate_train_losses_by_key, clip_grad_norm_local_shards",
                actor.read_text(),
            )
            self.assertIn(
                "aggregate_train_losses_by_key(losses_reduced, self.parallel_state)",
                actor.read_text(),
            )
            self.assertIn("nla_local_grad_norm: bool = True", arguments.read_text())
            self.assertGreaterEqual(report["miles_tree"]["file_count"], 3)
            self.assertEqual(report["miles_patches_tree"]["file_count"], 1)
            self.assertEqual(len(report["runtime_sha256"]), 64)

            second = module.reconcile_miles_runtime(
                miles_root,
                apply=False,
                miles_patches_root=patches_root,
            )
            self.assertTrue(second["passed"])
            self.assertEqual(second["changed_files"], [])


if __name__ == "__main__":
    unittest.main()
