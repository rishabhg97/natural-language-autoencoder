import importlib.util
import json
import pathlib
import tempfile
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "prepare_nano_publication_runtime.py"
    spec = importlib.util.spec_from_file_location("prepare_nano_publication_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PrepareNanoPublicationRuntimeTests(unittest.TestCase):
    def test_production_config_pins_model_and_container_digests(self):
        config = yaml.safe_load(
            (
                ROOT
                / "configs"
                / "nano_data"
                / "publication"
                / "r33_frozen_runtime.yaml"
            ).read_text()
        )

        self.assertEqual(
            config["expected_model_sha256"],
            "abd6d1368f9d2baa1b6f5b4047916db780466193af85b4772bbf5dc64c218019",
        )
        self.assertEqual(
            config["container_image_digest"],
            "sha256:3c90e38f5ec51e51d1c73bd7eb3d83674a254f451147c5cadc4344314258a112",
        )
        self.assertTrue(
            config["outputs"]["runtime_provenance_json"].endswith(
                "/publication/runtime/extraction_runtime_provenance.json"
            )
        )
        self.assertEqual(
            config["critical_files"]["source_snapshot_archive"],
            "/workspace/interp/code/nano30b-nla-pilot-publication-current/source.tgz",
        )
        self.assertTrue(
            config["critical_files"]["source_snapshot_manifest"].endswith(
                "/publication/runtime/source_snapshot_manifest.json"
            )
        )

    def test_writes_complete_runtime_and_validated_model_fingerprint(self):
        preparer = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code = root / "code"
            for relative in (
                "scripts",
                "external/natural_language_autoencoders/nla",
                "external/natural_language_autoencoders/configs",
            ):
                path = code / relative
                path.mkdir(parents=True)
                (path / "source.py").write_text(f"# {relative}\n")
            miles = root / "miles"
            patches = root / "patches"
            miles.mkdir()
            patches.mkdir()
            (miles / "train.py").write_text("print('train')\n")
            (patches / "patch.py").write_text("PATCH = True\n")
            critical = root / "critical.yaml"
            critical.write_text("frozen: true\n")
            model_root = root / "model"
            model_root.mkdir()
            model_source = root / "model-fingerprint-source.json"
            model_source.write_text(
                json.dumps(
                    {
                        "label": "nano",
                        "root": str(model_root.resolve()),
                        "file_count": 73,
                        "total_bytes": 63174971861,
                        "sha256": "a" * 64,
                    }
                )
            )
            output = root / "runtime"
            config = root / "runtime.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "nano_publication_runtime.v1",
                        "paths": {
                            "code_root": str(code),
                            "miles_root": str(miles),
                            "miles_patches_root": str(patches),
                            "model_root": str(model_root),
                            "model_fingerprint_source": str(model_source),
                        },
                        "expected_model_sha256": "a" * 64,
                        "container_image_digest": "sha256:" + "b" * 64,
                        "critical_files": {"frozen_config": str(critical)},
                        "outputs": {
                            "model_fingerprint_json": str(output / "model.json"),
                            "runtime_provenance_json": str(output / "runtime.json"),
                        },
                    },
                    sort_keys=False,
                )
            )

            report = preparer.run_prepare(config)
            runtime = json.loads((output / "runtime.json").read_text())
            model = json.loads((output / "model.json").read_text())

        self.assertTrue(report["passed"])
        self.assertTrue(runtime["runtime"]["complete"])
        self.assertEqual(model["sha256"], "a" * 64)
        self.assertEqual(runtime["runtime"]["container_image_digest"], "sha256:" + "b" * 64)
        self.assertIn("frozen_config", runtime["runtime"]["critical_files"])


if __name__ == "__main__":
    unittest.main()
