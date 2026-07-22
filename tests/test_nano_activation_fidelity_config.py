import importlib.util
import pathlib
import tempfile
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts/nano_activation_fidelity_config.py"
    spec = importlib.util.spec_from_file_location(
        "nano_activation_fidelity_config",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoActivationFidelityConfigTests(unittest.TestCase):
    def test_publication_config_is_validation_only_and_deterministic(self):
        module = load_script()
        config_path = (
            ROOT
            / "configs/nano_functional/"
            "r33_clean_sft_validation64_activation_fidelity.yaml"
        )
        config = module.load_config(config_path)
        command = module.build_command(config, config_path=config_path)

        split_index = command.index("--eval-splits")
        self.assertEqual(command[split_index + 1], "validation")
        self.assertNotIn("test", command[split_index + 1 : split_index + 2])
        self.assertIn("--publication-mode", command)
        self.assertIn("--repeat-full-forward", command)
        self.assertIn("--deterministic-algorithms", command)
        self.assertIn("--no-allow-tf32", command)
        self.assertIn("--no-cudnn-benchmark", command)
        self.assertNotIn("--extraction-source-parquet", command)
        self.assertTrue(config["paths"]["log_file"].endswith(".log"))
        self.assertTrue(
            config["paths"]["runner_report_json"].endswith("_runner.json")
        )

    def test_rejects_missing_required_path(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": module.SCHEMA_VERSION,
                        "paths": {
                            "generated_jsonl": "generated.jsonl",
                            "source_base_parquet": "base.parquet",
                            "report_json": "report.json",
                        },
                        "eval": {
                            "boundary": 33,
                            "validation_limit": 64,
                            "test_limit": 1,
                        },
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "paths.target_model"):
                module.load_config(path)


if __name__ == "__main__":
    unittest.main()
