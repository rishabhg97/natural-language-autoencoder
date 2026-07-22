import importlib.util
import pathlib
import tempfile
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "nano_openai_compatible_server.py"
    spec = importlib.util.spec_from_file_location("nano_openai_compatible_server", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoOpenAICompatibleServerTests(unittest.TestCase):
    def test_config_renders_owned_local_server(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            path = root / "server.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": module.SCHEMA_VERSION,
                        "server": {
                            "python": "/venv/bin/python",
                            "module": "sglang.launch_server",
                            "model_path": "/models/qwen",
                            "served_model_name": "paraphraser",
                            "host": "127.0.0.1",
                            "port": 30080,
                            "gpu_device": "7",
                            "tp_size": 1,
                        },
                        "runtime": {
                            "pid_file": str(root / "server.pid"),
                            "log_file": str(root / "server.log"),
                        },
                    }
                )
            )
            config = module.load_config(path)
            command = module.build_command(config)

        self.assertEqual(command[:3], ["/venv/bin/python", "-m", "sglang.launch_server"])
        self.assertEqual(command[command.index("--model-path") + 1], "/models/qwen")
        self.assertEqual(command[command.index("--tp-size") + 1], "1")
        self.assertEqual(module.endpoint(config), "http://127.0.0.1:30080")

    def test_model_stage_materializes_symlink_snapshot_and_is_reusable(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            blobs = root / "blobs"
            snapshot = root / "snapshot"
            staged = root / "staged"
            blobs.mkdir()
            snapshot.mkdir()
            (blobs / "config").write_text("{}\n")
            (blobs / "weights").write_bytes(b"model" * 41)
            (snapshot / "config.json").symlink_to(blobs / "config")
            (snapshot / "model.safetensors").symlink_to(blobs / "weights")
            config = {
                "schema_version": module.SCHEMA_VERSION,
                "server": {
                    "python": "/venv/bin/python",
                    "module": "sglang.launch_server",
                    "model_path": str(snapshot),
                    "served_model_name": "paraphraser",
                },
                "runtime": {
                    "pid_file": str(root / "server.pid"),
                    "log_file": str(root / "server.log"),
                    "model_stage": {
                        "output_dir": str(staged),
                        "manifest_json": str(root / "stage.json"),
                        "workers": 2,
                        "task_size_bytes": 17,
                        "follow_symlinks": True,
                    },
                },
            }

            first = module.prepare_model_stage(config)
            second = module.prepare_model_stage(config)
            command = module.build_command(config)

            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertFalse((staged / "model.safetensors").is_symlink())
            self.assertEqual((staged / "model.safetensors").read_bytes(), b"model" * 41)
            self.assertEqual(command[command.index("--model-path") + 1], str(staged))


if __name__ == "__main__":
    unittest.main()
