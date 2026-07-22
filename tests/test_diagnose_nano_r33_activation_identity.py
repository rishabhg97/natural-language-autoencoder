import hashlib
import importlib.util
import inspect
import pathlib
import tempfile
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "diagnose_nano_r33_activation_identity.py"
    spec = importlib.util.spec_from_file_location(
        "diagnose_nano_r33_activation_identity",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoR33ActivationIdentityDiagnosticTests(unittest.TestCase):
    def test_assess_identity_rows_fails_when_any_tolerance_is_exceeded(self):
        module = load_script()
        rows = [
            {
                "full_vs_stored": {
                    "relative_l2": 0.005,
                    "max_abs": 0.004,
                    "one_minus_cos": 0.00001,
                }
            },
            {
                "full_vs_stored": {
                    "relative_l2": 0.02,
                    "max_abs": 0.003,
                    "one_minus_cos": 0.00002,
                }
            },
        ]

        assessment = module.assess_identity_rows(
            rows,
            comparison="full_vs_stored",
            max_relative_l2=0.01,
            max_abs=0.01,
            max_one_minus_cos=0.0001,
        )

        self.assertFalse(assessment["passed"])
        self.assertEqual(assessment["violating_rows"], 1)
        self.assertEqual(assessment["max_observed_relative_l2"], 0.02)
        self.assertEqual(assessment["thresholds"]["max_relative_l2"], 0.01)

    def test_inference_call_runs_forward_with_gradients_disabled(self):
        module = load_script()
        state = {"disabled": False}

        class NoGrad:
            def __enter__(self):
                state["disabled"] = True

            def __exit__(self, *_args):
                state["disabled"] = False

        class FakeTorch:
            @staticmethod
            def no_grad():
                return NoGrad()

        def forward(value):
            self.assertTrue(state["disabled"])
            return value + 1

        self.assertEqual(module.inference_call(FakeTorch, forward, 4), 5)
        self.assertFalse(state["disabled"])

    def test_build_fidelity_manifest_binds_inputs_code_checkpoint_and_samples(self):
        module = load_script()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            generated = root / "generated.jsonl"
            source = root / "source.parquet"
            mean_source = root / "train.parquet"
            extraction_source = root / "extract.parquet"
            family_manifest = root / "families.json"
            code = root / "extract.py"
            for path, payload in (
                (generated, b"generated"),
                (source, b"source"),
                (mean_source, b"mean"),
                (extraction_source, b"extract"),
                (family_manifest, b"families"),
                (code, b"code"),
            ):
                path.write_bytes(payload)

            manifest = module.build_activation_fidelity_manifest(
                generated_jsonl=generated,
                source_base_parquet=source,
                mean_activation_parquet=mean_source,
                extraction_source_parquet=extraction_source,
                content_family_manifest=family_manifest,
                target_model="nano-checkpoint",
                target_model_fingerprint="checkpoint-sha",
                target_revision="revision-a",
                boundary=33,
                target_torch_dtype="bfloat16",
                selection_strategy="family_stratified",
                selection_seed=20260708,
                sample_identities=[("raw_tokens", "doc-a", 4)],
                code_paths=[code],
                publication_mode=True,
                execution_profile={
                    "deterministic_algorithms": True,
                    "allow_tf32": False,
                    "cudnn_benchmark": False,
                    "float32_matmul_precision": "highest",
                    "cublas_workspace_config": ":4096:8",
                    "seed": 20260709,
                },
            )

        self.assertEqual(
            manifest["inputs"]["source_base_parquet"]["sha256"],
            hashlib.sha256(b"source").hexdigest(),
        )
        self.assertEqual(
            manifest["inputs"]["content_family_manifest"]["sha256"],
            hashlib.sha256(b"families").hexdigest(),
        )
        self.assertEqual(
            manifest["activation_extraction"]["checkpoint"]["fingerprint"],
            "checkpoint-sha",
        )
        self.assertEqual(manifest["activation_extraction"]["boundary"], 33)
        self.assertEqual(manifest["activation_extraction"]["dtype"], "bfloat16")
        self.assertTrue(
            manifest["activation_extraction"]["execution_profile"][
                "deterministic_algorithms"
            ]
        )
        self.assertEqual(
            manifest["selection"]["sample_identities"],
            [["raw_tokens", "doc-a", 4]],
        )
        self.assertEqual(
            manifest["activation_extraction"]["code"][0]["sha256"],
            hashlib.sha256(b"code").hexdigest(),
        )
        self.assertTrue(manifest["publication_complete"])

    def test_publication_manifest_rejects_missing_checkpoint_fingerprint(self):
        module = load_script()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "input"
            path.write_text("x")
            with self.assertRaisesRegex(ValueError, "checkpoint fingerprint"):
                module.build_activation_fidelity_manifest(
                    generated_jsonl=path,
                    source_base_parquet=path,
                    mean_activation_parquet=path,
                    extraction_source_parquet=None,
                    content_family_manifest=path,
                    target_model="nano-checkpoint",
                    target_model_fingerprint=None,
                    target_revision=None,
                    boundary=33,
                    target_torch_dtype="bfloat16",
                    selection_strategy="row_order",
                    selection_seed=0,
                    sample_identities=[("uuid", "a")],
                    code_paths=[path],
                    publication_mode=True,
                    execution_profile={
                        "deterministic_algorithms": True,
                        "allow_tf32": False,
                        "cudnn_benchmark": False,
                        "float32_matmul_precision": "highest",
                        "cublas_workspace_config": ":4096:8",
                        "seed": 20260709,
                    },
                )

    def test_summarize_activation_fidelity_reports_error_and_metric_floor(self):
        module = load_script()

        summary = module.summarize_activation_fidelity(
            live_values=np.asarray([[3.0, 4.0], [0.0, 2.0]]),
            stored_values=np.asarray([[3.0, 4.0], [0.0, 1.0]]),
            train_mean=np.zeros(2),
        )

        self.assertEqual(summary["row_count"], 2)
        self.assertAlmostEqual(summary["absolute_l2_mean"], 0.5)
        self.assertAlmostEqual(summary["absolute_l2_max"], 1.0)
        self.assertAlmostEqual(summary["relative_l2_mean"], 0.25)
        self.assertAlmostEqual(summary["cosine_agreement_mean"], 1.0)
        self.assertAlmostEqual(summary["live_over_stored_norm_ratio_mean"], 1.5)
        floor = summary["stored_as_prediction_vs_live_target"]
        self.assertAlmostEqual(floor["directional_mse"], 0.0)
        self.assertAlmostEqual(floor["raw_mse"], 0.25)
        self.assertAlmostEqual(floor["mean_predictor_raw_mse"], 7.25)
        self.assertAlmostEqual(floor["centered_r2"], 1.0 - 0.25 / 7.25)

    def test_build_comparisons_reports_all_method_pairs_per_row(self):
        module = load_script()

        def metric(lhs, rhs):
            return {"delta": lhs - rhs}

        rows = module.build_identity_comparisons(
            keys=[("raw_tokens", "doc-a", 4), ("raw_tokens", "doc-b", 7)],
            prefix_lengths=[4, 7],
            stored=[10, 20],
            full_forward=[11, 22],
            extraction_forward=[10, 21],
            metric_fn=metric,
        )

        self.assertEqual(rows[0]["provenance_key"], ["raw_tokens", "doc-a", 4])
        self.assertEqual(rows[0]["prefix_length"], 4)
        self.assertEqual(rows[0]["full_vs_stored"], {"delta": 1})
        self.assertEqual(rows[0]["extraction_vs_stored"], {"delta": 0})
        self.assertEqual(rows[0]["full_vs_extraction"], {"delta": 1})
        self.assertEqual(rows[1]["full_vs_extraction"], {"delta": 1})

    def test_build_comparisons_rejects_misaligned_methods(self):
        module = load_script()

        with self.assertRaisesRegex(ValueError, "same row count"):
            module.build_identity_comparisons(
                keys=[("uuid", "a")],
                prefix_lengths=[3],
                stored=[1],
                full_forward=[1, 2],
                extraction_forward=[1],
                metric_fn=lambda lhs, rhs: {"delta": lhs - rhs},
            )

    def test_original_batch_starts_selects_each_source_batch_once(self):
        module = load_script()
        groups = [
            {"doc_id": "a"},
            {"doc_id": "b"},
            {"doc_id": "c"},
            {"doc_id": "d"},
            {"doc_id": "e"},
        ]

        starts = module.original_batch_starts(
            groups,
            target_doc_ids={"b", "c", "d"},
            batch_size=2,
        )

        self.assertEqual(starts, [0, 2])
        with self.assertRaisesRegex(ValueError, "missing target documents"):
            module.original_batch_starts(
                groups,
                target_doc_ids={"missing"},
                batch_size=2,
            )

    def test_original_extraction_geometry_is_captured_before_other_forwards(self):
        module = load_script()
        source = inspect.getsource(module.run)
        original_forward = source.index('"stage": "original_extraction_forward"')

        for later_stage in (
            '"stage": "full_model_forward"',
            '"stage": "repeat_full_model_forward"',
            '"stage": "extraction_prefix_forward"',
        ):
            self.assertLess(
                original_forward,
                source.index(later_stage),
                f"original extraction geometry must precede {later_stage}",
            )

    def test_parser_exposes_loader_and_repeatability_diagnostics(self):
        module = load_script()
        args = module.build_parser().parse_args(
            [
                "--generated-jsonl",
                "generated.jsonl",
                "--source-base-parquet",
                "base.parquet",
                "--content-family-manifest",
                "families.json",
                "--target-model",
                "model",
                "--report-json",
                "report.json",
                "--boundary",
                "33",
                "--validation-limit",
                "2",
                "--test-limit",
                "2",
                "--batch-size",
                "8",
                "--eval-splits",
                "validation",
                "--mean-activation-parquet",
                "train.parquet",
                "--selection-strategy",
                "family_stratified",
                "--selection-seed",
                "20260708",
                "--target-model-fingerprint",
                "checkpoint-sha",
                "--publication-mode",
                "--fidelity-max-relative-l2",
                "0.01",
                "--fidelity-max-abs",
                "0.02",
                "--fidelity-max-one-minus-cos",
                "0.0001",
                "--model-loader",
                "extraction",
                "--repeat-full-forward",
                "--deterministic-algorithms",
                "--no-allow-tf32",
                "--no-cudnn-benchmark",
                "--float32-matmul-precision",
                "highest",
                "--cublas-workspace-config",
                ":4096:8",
                "--seed",
                "20260709",
            ]
        )

        self.assertEqual(args.model_loader, "extraction")
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.eval_splits, ["validation"])
        self.assertTrue(args.repeat_full_forward)
        self.assertTrue(args.deterministic_algorithms)
        self.assertFalse(args.allow_tf32)
        self.assertFalse(args.cudnn_benchmark)
        self.assertEqual(args.float32_matmul_precision, "highest")
        self.assertEqual(args.cublas_workspace_config, ":4096:8")
        self.assertEqual(args.seed, 20260709)
        self.assertEqual(args.mean_activation_parquet, pathlib.Path("train.parquet"))
        self.assertEqual(
            args.content_family_manifest,
            pathlib.Path("families.json"),
        )
        self.assertEqual(args.selection_strategy, "family_stratified")
        self.assertEqual(args.selection_seed, 20260708)
        self.assertEqual(args.target_model_fingerprint, "checkpoint-sha")
        self.assertTrue(args.publication_mode)
        self.assertEqual(args.fidelity_max_relative_l2, 0.01)
        self.assertEqual(args.fidelity_max_abs, 0.02)
        self.assertEqual(args.fidelity_max_one_minus_cos, 0.0001)

    def test_validate_args_fails_closed_for_publication_sampling(self):
        module = load_script()
        parser = module.build_parser()
        common = [
            "--generated-jsonl",
            "generated.jsonl",
            "--source-base-parquet",
            "base.parquet",
            "--target-model",
            "model",
            "--report-json",
            "report.json",
            "--boundary",
            "33",
            "--validation-limit",
            "2",
            "--test-limit",
            "2",
            "--publication-mode",
            "--deterministic-algorithms",
            "--no-allow-tf32",
            "--no-cudnn-benchmark",
            "--float32-matmul-precision",
            "highest",
            "--cublas-workspace-config",
            ":4096:8",
            "--seed",
            "20260709",
        ]

        with self.assertRaisesRegex(ValueError, "checkpoint fingerprint"):
            module.validate_args(parser.parse_args(common))
        with self.assertRaisesRegex(ValueError, "family_stratified"):
            module.validate_args(
                parser.parse_args(
                    common + ["--target-model-fingerprint", "checkpoint-sha"]
                )
            )
        with self.assertRaisesRegex(ValueError, "content-family manifest"):
            module.validate_args(
                parser.parse_args(
                    common
                    + [
                        "--target-model-fingerprint",
                        "checkpoint-sha",
                        "--selection-strategy",
                        "family_stratified",
                    ]
                )
            )
        module.validate_args(
            parser.parse_args(
                common
                + [
                    "--target-model-fingerprint",
                    "checkpoint-sha",
                    "--selection-strategy",
                    "family_stratified",
                    "--content-family-manifest",
                    "families.json",
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
