import importlib.util
import pathlib
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoFunctionalEvalConfigTests(unittest.TestCase):
    def _write_config(self, root: pathlib.Path, *, validation_limit: int = 8) -> pathlib.Path:
        config = root / "functional.yaml"
        config.write_text(
            textwrap.dedent(
                f"""
                schema_version: nano_functional_eval.v1
                python: /venv/bin/python
                paths:
                  generated_jsonl: /data/generated.jsonl
                  sft_generated_jsonl: /data/sft.jsonl
                  ar_checkpoint_dir: /models/ar
                  source_base_parquet: /data/base.parquet
                  mean_activation_parquet: /data/train.parquet
                  target_model: /models/nano30b
                  report_json: /out/report.json
                eval:
                  boundary: 33
                  eval_splits: [validation]
                  validation_limit: {validation_limit}
                  test_limit: 8
                  batch_size: 4
                  identity_relative_l2: 0.01
                  identity_max_abs: 0.01
                  identity_one_minus_cos: 0.0001
                  control: real
                  bootstrap_resamples: 1000
                  selection_strategy: longest_prefix
                  min_independent_families: 100
                  require_generation_identity: true
                """
            )
        )
        return config

    def test_config_renders_required_paths_and_limits(self):
        module = load_script("nano_functional_eval_config")
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_config(pathlib.Path(tmp))
            config = module.load_config(config_path)
            command = module.build_command(config, config_path=config_path)

        self.assertEqual(command[0], "/venv/bin/python")
        self.assertIn("scripts/eval_nano_r33_functional_recovery.py", command)
        self.assertEqual(
            command[command.index("--source-base-parquet") + 1],
            "/data/base.parquet",
        )
        self.assertEqual(command[command.index("--boundary") + 1], "33")
        self.assertEqual(command[command.index("--validation-limit") + 1], "8")
        self.assertEqual(command[command.index("--eval-splits") + 1], "validation")
        self.assertEqual(
            command[command.index("--sft-generated-jsonl") + 1],
            "/data/sft.jsonl",
        )
        self.assertEqual(
            command[command.index("--mean-activation-parquet") + 1],
            "/data/train.parquet",
        )
        self.assertEqual(
            command[command.index("--selection-strategy") + 1],
            "longest_prefix",
        )
        self.assertEqual(
            command[command.index("--min-independent-families") + 1],
            "100",
        )
        self.assertIn("--require-generation-identity", command)

    def test_config_rejects_missing_paths_and_nonpositive_limits(self):
        module = load_script("nano_functional_eval_config")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config_path = self._write_config(root, validation_limit=0)
            with self.assertRaisesRegex(module.FunctionalEvalConfigError, "positive"):
                module.load_config(config_path)

            config_path.write_text(
                textwrap.dedent(
                    """
                    schema_version: nano_functional_eval.v1
                    paths: {}
                    eval: {}
                    """
                )
            )
            with self.assertRaisesRegex(module.FunctionalEvalConfigError, "missing"):
                module.load_config(config_path)


if __name__ == "__main__":
    unittest.main()
