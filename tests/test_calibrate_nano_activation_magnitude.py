from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import textwrap
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_cache(path: pathlib.Path) -> None:
    metadata = {
        "schema_version": "nano_roundtrip_prediction_cache.v1",
        "ar_checkpoint_dir": "/models/ar",
        "ar_hf_dir": "/models/ar",
        "critic_template_sha256": "template-sha",
        "dataset_provenance": {"train": {"sha256": "train-sha"}},
        "splits": {"validation": {"row_count": 4}},
    }
    train_mean = np.asarray([10.0, 10.0], dtype=np.float32)
    teacher = np.asarray(
        [[11.0, 10.0], [10.0, 11.0], [9.0, 10.0], [10.0, 9.0]],
        dtype=np.float32,
    )
    targets = train_mean[None, :] + 2.0 * (teacher - train_mean[None, :])
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(json.dumps(metadata), dtype=np.str_),
            train_mean=train_mean,
            validation__row_indices=np.arange(4),
            validation__doc_ids=np.asarray(["d1", "d2", "d3", "d4"]),
            validation__content_family_ids=np.asarray(["f1", "f2", "f3", "f4"]),
            validation__targets=targets,
            validation__prediction__teacher=teacher,
            validation__prediction__av_real=teacher,
        )


def write_score_report(path: pathlib.Path, cache: pathlib.Path) -> None:
    module = load_script("calibrate_nano_activation_magnitude")
    path.write_text(
        json.dumps(
            {
                "prediction_cache": module.file_provenance(cache),
                "gate": {"passed": True},
            }
        )
    )


class ActivationMagnitudeCalibrationTests(unittest.TestCase):
    def test_selects_train_mean_scalar_without_test_fit(self):
        module = load_script("calibrate_nano_activation_magnitude")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            cache = root / "cache.npz"
            score_report = root / "score_report.json"
            report = root / "report.json"
            config = root / "calibration.yaml"
            write_cache(cache)
            write_score_report(score_report, cache)
            config.write_text(
                textwrap.dedent(
                    f"""
                    schema_version: nano_activation_magnitude_calibration.v1
                    paths:
                      report_json: {report}
                      datasets:
                        validation:
                          cache_npz: {cache}
                          score_report_json: {score_report}
                          split: validation
                    protocol:
                      fit_dataset: validation
                      fit_variant: teacher
                      candidate_methods: [identity, origin_scalar, train_mean_scalar]
                      evaluation_variants: [teacher, av_real]
                      bootstrap_samples: 200
                      bootstrap_seed: 7
                      publication_status: exploratory_posthoc
                    """
                )
            )

            loaded = module.load_config(config)
            result = module.evaluate_config(loaded, config_path=config)

        self.assertEqual(result["fit"]["selected_method"], "train_mean_scalar")
        self.assertAlmostEqual(
            result["fit"]["candidate_parameters"]["train_mean_scalar"]["scalar"],
            2.0,
        )
        selected = result["evaluation"]["validation"]["variants"]["av_real"]
        self.assertAlmostEqual(selected["selected_metrics"]["raw_mse"], 0.0)
        self.assertGreater(
            selected["selected_vs_identity_clustered_bootstrap"]["mean_improvement"],
            0.0,
        )

    def test_rejects_test_as_fit_dataset(self):
        module = load_script("calibrate_nano_activation_magnitude")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config = root / "calibration.yaml"
            config.write_text(
                textwrap.dedent(
                    """
                    schema_version: nano_activation_magnitude_calibration.v1
                    paths:
                      report_json: report.json
                      datasets:
                        final_test:
                          cache_npz: cache.npz
                          score_report_json: score_report.json
                          split: test
                    protocol:
                      fit_dataset: final_test
                      fit_variant: teacher
                      candidate_methods: [identity]
                      evaluation_variants: [teacher]
                    """
                )
            )

            with self.assertRaisesRegex(module.CalibrationError, "test data"):
                module.load_config(config)


if __name__ == "__main__":
    unittest.main()
