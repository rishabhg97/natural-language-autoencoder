import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoCheckpointRetentionTests(unittest.TestCase):
    def test_retention_never_selects_protected_best_or_challenger(self):
        module = load_script("nano_checkpoint_retention")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            paths = {name: root / name for name in ("sft", "ar", "best", "challenger", "loser")}
            for path in paths.values():
                path.mkdir()
            policy = module.RetentionPolicy(
                output_root=root,
                protected={paths["sft"], paths["ar"]},
                current_best=paths["best"],
                keep_challenger=paths["challenger"],
            )

            plan = module.build_cleanup_plan(policy, candidates=list(paths.values()))

        self.assertEqual(plan.delete, [paths["loser"].resolve()])
        self.assertIn(paths["best"].resolve(), plan.keep)

    def test_retention_rejects_symlink_and_out_of_root_candidates(self):
        module = load_script("nano_checkpoint_retention")
        with tempfile.TemporaryDirectory() as tmp:
            parent = pathlib.Path(tmp)
            root = parent / "outputs"
            root.mkdir()
            target = root / "target"
            target.mkdir()
            symlink = root / "link"
            symlink.symlink_to(target, target_is_directory=True)
            outside = parent / "outside"
            outside.mkdir()
            policy = module.RetentionPolicy(output_root=root)

            with self.assertRaisesRegex(module.RetentionError, "symlink"):
                module.build_cleanup_plan(policy, candidates=[symlink])
            with self.assertRaisesRegex(module.RetentionError, "outside output_root"):
                module.build_cleanup_plan(policy, candidates=[outside])

    def test_apply_writes_manifest_before_first_deletion(self):
        module = load_script("nano_checkpoint_retention")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            loser = root / "loser"
            loser.mkdir()
            manifest = root / "cleanup_manifest.json"
            policy = module.RetentionPolicy(output_root=root)
            plan = module.build_cleanup_plan(policy, candidates=[loser])
            observed = []

            def delete(path):
                observed.append((path, manifest.is_file()))

            module.execute_cleanup(
                plan,
                manifest_path=manifest,
                apply=True,
                delete_function=delete,
            )

            final_manifest = json.loads(manifest.read_text())

        self.assertEqual(observed, [(loser.resolve(), True)])
        self.assertEqual(final_manifest["deleted"], [str(loser.resolve())])


if __name__ == "__main__":
    unittest.main()
