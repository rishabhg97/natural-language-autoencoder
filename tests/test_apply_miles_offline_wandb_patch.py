"""Regression coverage for the resilient offline W&B role installer."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "apply_miles_offline_wandb_patch.py"


def load_module():
    spec = importlib.util.spec_from_file_location("apply_miles_offline_wandb_patch", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ApplyMilesOfflineWandbPatchTests(unittest.TestCase):
    def test_applies_all_role_ownership_changes_and_is_idempotent(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            miles = Path(tmp)
            files = {
                "miles/utils/tracking_utils.py": """from . import wandb_utils\n\n\ndef init_tracking(args, primary: bool = True, **kwargs):\n    if primary:\n        wandb_utils.init_wandb_primary(args, **kwargs)\n    else:\n        wandb_utils.init_wandb_secondary(args, **kwargs)\n""",
                "miles/ray/rollout.py": """class RolloutManager:\n    def __init__(self, args):\n        init_tracking(args, primary=False, router_addr=f\"http://{args.sglang_router_ip}:{args.sglang_router_port}\")\n""",
                "miles/backends/fsdp_utils/actor.py": """class FSDPTrainRayActor:\n    def __init__(self, args, role):\n        if dist.get_rank() == 0:\n            init_tracking(args, primary=False)\n""",
                "miles/utils/wandb_utils.py": """def init_wandb_primary(args):\n    init_kwargs = {\n        \"entity\": args.wandb_team,\n        \"project\": args.wandb_project,\n        \"group\": group,\n        \"name\": run_name,\n        \"config\": _compute_config_for_logging(args),\n    }\n\n    wandb.init(**init_kwargs)\n\n\ndef init_wandb_secondary(args, router_addr=None):\n    wandb_run_id = getattr(args, \"wandb_run_id\", None)\n    if wandb_run_id is None:\n        return\n\n    offline = _is_offline_mode(args)\n\n    if (not offline) and args.wandb_key is not None:\n        wandb.login(key=args.wandb_key, host=args.wandb_host)\n\n    init_kwargs = {\n        \"id\": wandb_run_id,\n        \"entity\": args.wandb_team,\n        \"project\": args.wandb_project,\n        \"config\": args.__dict__,\n        \"resume\": \"allow\",\n        \"reinit\": True,\n        \"settings\": wandb.Settings(**settings_kwargs),\n    }\n\n    # Add custom directory if specified\n    if args.wandb_dir:\n        init_kwargs[\"dir\"] = args.wandb_dir\n""",
            }
            for relative, content in files.items():
                target = miles / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)

            first = module.apply_offline_wandb_role_patch(miles)
            self.assertEqual(first["changed"], 4)
            self.assertEqual(module.apply_offline_wandb_role_patch(miles)["changed"], 0)

            tracking = (miles / "miles/utils/tracking_utils.py").read_text()
            rollout = (miles / "miles/ray/rollout.py").read_text()
            actor = (miles / "miles/backends/fsdp_utils/actor.py").read_text()
            wandb = (miles / "miles/utils/wandb_utils.py").read_text()
            self.assertIn("role: str | None = None", tracking)
            self.assertIn("role=role", tracking)
            self.assertIn('role="rollout"', rollout)
            self.assertIn("role=role", actor)
            self.assertIn('init_kwargs["id"] = args.wandb_run_id', wandb)
            self.assertIn('offline_run_id = f"{wandb_run_id}-{offline_role}"', wandb)
            self.assertIn('"id": offline_run_id', wandb)
            self.assertIn('init_kwargs["name"] = f"{args.wandb_group}-{offline_role}"', wandb)


if __name__ == "__main__":
    unittest.main()
