import importlib.util
import json
import pathlib
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_base(path: pathlib.Path) -> None:
    pq.write_table(
        pa.table(
            {
                "doc_id": ["train-a", "train-b", "validation-a", "test-a"],
                "split_unit_id": ["family-a", "family-b", "family-v", "family-t"],
                "activation_layer": [33, 33, 33, 33],
                "token_position": [4, 5, 6, 7],
                "n_raw_tokens": [5, 6, 7, 8],
                "token_id": [14, 15, 16, 17],
                "token_ids_prefix": [[1, 14], [1, 15], [1, 16], [1, 17]],
                "activation_vector": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "api_explanation": ["teacher-a", "teacher-b", "teacher-v", "teacher-t"],
            }
        ),
        path,
    )


def write_actor_sidecar(path: pathlib.Path) -> None:
    sidecar = pathlib.Path(str(path) + ".nla_meta.yaml")
    sidecar.write_text(
        yaml.safe_dump(
            {
                "kind": "nla_dataset",
                "schema_version": 1,
                "dataset_id": "av_sft_r33_test",
                "stage": "av_sft",
                "row_count": 4,
                "extraction": {
                    "base_model": "nano-test",
                    "d_model": 4,
                    "layer_index": 33,
                    "norm": "none",
                },
                "tokens": {
                    "injection_char": "々",
                    "injection_token_id": 42019,
                    "injection_left_neighbor_id": 1062,
                    "injection_right_neighbor_id": 1885,
                    "critic_suffix_ids": None,
                },
                "prompt_templates": {
                    "actor": "Inspect <concept>{injection_char}</concept> and respond.",
                    "critic": "Summary: <text>{explanation}</text> <summary>",
                },
                "parent_datasets": ["base_r33_test"],
            },
            sort_keys=False,
        )
    )


def write_manifest(path: pathlib.Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "nano_split_manifest.v2",
                "split_mode": "content_family_manifest",
                "split_unit_column": "split_unit_id",
                "split_unit_kind": "content_family",
                "splits": {
                    "train": {
                        "docs": ["train-a", "train-b"],
                        "split_unit_ids": ["family-a", "family-b"],
                    },
                    "validation": {
                        "docs": ["validation-a"],
                        "split_unit_ids": ["family-v"],
                    },
                    "test": {
                        "docs": ["test-a"],
                        "split_unit_ids": ["family-t"],
                    },
                },
                "content_verification": {"content_cross_split_overlap_count": 0},
            }
        )
    )


def write_family_contract(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    manifest = root / "content_families.json"
    coverage = root / "content_family_coverage.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "nano_content_family_manifest.v1",
                "doc_assignments": {
                    "train-a": "family-a",
                    "train-b": "family-b",
                    "validation-a": "family-v",
                    "test-a": "family-t",
                },
                "family_splits": {
                    "family-a": "train",
                    "family-b": "train",
                    "family-v": "validation",
                    "family-t": "test",
                },
                "split_assignment": {
                    "seed": 20260708,
                    "weights": {"train": 0.5, "validation": 0.25, "test": 0.25},
                },
            }
        )
    )
    coverage.write_text(
        json.dumps(
            {
                "schema_version": "nano_content_family_exposure_report.v1",
                "splits": {
                    "validation": {
                        "eligible_doc_ids": ["validation-a"],
                        "eligible_family_ids": ["family-v"],
                    },
                    "test": {
                        "eligible_doc_ids": ["test-a"],
                        "eligible_family_ids": ["family-t"],
                    },
                },
                "retain_existing_sft_checkpoints": True,
            }
        )
    )
    return manifest, coverage


class NanoR33RLDatasetTests(unittest.TestCase):
    def test_builder_accepts_v2_actor_model_sidecar_with_bound_training_lineage(self):
        builder = load_script("build_nano_r33_rl_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            base = root / "base.parquet"
            split_manifest = root / "split_manifest.json"
            family_manifest, family_coverage = write_family_contract(root)
            write_base(base)
            write_actor_sidecar(base)
            write_manifest(split_manifest)
            dataset_meta = yaml.safe_load(
                pathlib.Path(str(base) + ".nla_meta.yaml").read_text()
            )
            model_dir = root / "actor_model"
            model_dir.mkdir()
            (model_dir / "nla_meta.yaml").write_text(
                yaml.safe_dump(
                    {
                        "kind": "nla_model",
                        "schema_version": 2,
                        "role": "actor",
                        "stage": "sl",
                        "d_model": 4,
                        "extraction": {"injection_scale": 2.0},
                        "tokens": dataset_meta["tokens"],
                        "prompt_templates": dataset_meta["prompt_templates"],
                        "trained_on": [str(base)],
                    },
                    sort_keys=False,
                )
            )

            report = builder.build_dataset(
                base_parquet=base,
                actor_sidecar_source=model_dir,
                split_manifest=split_manifest,
                content_family_manifest=family_manifest,
                content_family_coverage=family_coverage,
                expected_rows=2,
                expected_layer=33,
                output=root / "rl_train.parquet",
                report_json=root / "build_report.json",
                batch_size=2,
            )

        self.assertEqual(report["actor_sidecar_kind"], "nla_model")
        self.assertEqual(report["actor_sidecar_schema_version"], 2)
        self.assertEqual(report["rows"], 2)

    def test_configured_pipeline_builds_then_reuses_only_verified_output(self):
        pipeline = load_script("nano_rl_dataset_pipeline")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            base = root / "base.parquet"
            split_manifest = root / "split_manifest.json"
            family_manifest, family_coverage = write_family_contract(root)
            output = root / "rl_train.parquet"
            write_base(base)
            write_actor_sidecar(base)
            write_manifest(split_manifest)
            config = root / "pipeline.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "nano_rl_dataset_pipeline.v1",
                        "paths": {
                            "base_parquet": str(base),
                            "actor_sidecar_source": str(base),
                            "split_manifest": str(split_manifest),
                            "content_family_manifest": str(family_manifest),
                            "content_family_coverage": str(family_coverage),
                            "output": str(output),
                            "build_report_json": str(root / "build.json"),
                            "verify_report_json": str(root / "verify.json"),
                            "pipeline_report_json": str(root / "pipeline.json"),
                        },
                        "expectations": {
                            "rows": 2,
                            "d_model": 4,
                            "activation_layer": 33,
                        },
                        "runtime": {"batch_size": 2, "reuse_verified": True},
                    },
                    sort_keys=False,
                )
            )

            built = pipeline.run_pipeline(config)
            reused = pipeline.run_pipeline(config)

        self.assertTrue(built["passed"])
        self.assertEqual(built["action"], "built")
        self.assertEqual(reused["action"], "reused_verified")
        self.assertEqual(reused["rows"], 2)

    def test_materialized_family_split_is_directly_buildable(self):
        splitter = load_script("nano_av_materialize_splits")
        builder = load_script("build_nano_r33_rl_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            base = root / "base.parquet"
            write_base(base)
            pq.write_table(pq.read_table(base).drop(["split_unit_id"]), base)
            write_actor_sidecar(base)
            family_manifest, family_coverage = write_family_contract(root)
            split_dir = root / "splits"
            manifest = splitter.materialize_splits(
                base,
                split_dir,
                train_fraction=0.5,
                validation_fraction=0.25,
                test_fraction=0.25,
                seed=20260708,
                split_mode="content_family_manifest",
                content_family_manifest=family_manifest,
            )

            report = builder.build_dataset(
                base_parquet=split_dir / "train.parquet",
                actor_sidecar_source=split_dir / "train.parquet",
                split_manifest=split_dir / "split_manifest.json",
                content_family_manifest=family_manifest,
                content_family_coverage=family_coverage,
                expected_rows=2,
                output=root / "rl_train.parquet",
                report_json=root / "build_report.json",
                batch_size=2,
            )

        self.assertEqual(manifest["splits"]["train"]["split_unit_count"], 2)
        self.assertTrue(report["split_unit_filter_applied"])
        self.assertEqual(report["rows"], 2)

    def test_builder_binds_family_contract_and_exact_rows(self):
        builder = load_script("build_nano_r33_rl_dataset")
        verifier = load_script("verify_nano_r33_rl_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            base = root / "base.parquet"
            split_manifest = root / "split_manifest.json"
            output = root / "rl_train.parquet"
            write_base(base)
            write_actor_sidecar(base)
            write_manifest(split_manifest)
            family_manifest, family_coverage = write_family_contract(root)

            report = builder.build_dataset(
                base_parquet=base,
                actor_sidecar_source=base,
                split_manifest=split_manifest,
                content_family_manifest=family_manifest,
                content_family_coverage=family_coverage,
                expected_rows=2,
                output=output,
                report_json=root / "build_report.json",
                batch_size=2,
            )
            verification = verifier.verify_dataset(
                dataset=output,
                split_manifest=split_manifest,
                content_family_manifest=family_manifest,
                content_family_coverage=family_coverage,
                expected_rows=2,
                expected_d_model=4,
            )
            table = pq.read_table(output)

        self.assertTrue(report["split_unit_filter_applied"])
        self.assertTrue(report["family_filter_applied"])
        self.assertEqual(set(table.column("content_family_id").to_pylist()), {"family-a", "family-b"})
        self.assertTrue(verification["passed"])
        self.assertEqual(verification["heldout_family_overlap_count"], 0)

    def test_builder_derives_split_units_from_sealed_family_membership(self):
        builder = load_script("build_nano_r33_rl_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            base = root / "base.parquet"
            split_manifest = root / "split_manifest.json"
            write_base(base)
            table = pq.read_table(base).drop(["split_unit_id"])
            pq.write_table(table, base)
            write_actor_sidecar(base)
            write_manifest(split_manifest)
            family_manifest, family_coverage = write_family_contract(root)

            report = builder.build_dataset(
                base_parquet=base,
                actor_sidecar_source=base,
                split_manifest=split_manifest,
                content_family_manifest=family_manifest,
                content_family_coverage=family_coverage,
                expected_rows=2,
                output=root / "rl_train.parquet",
                report_json=root / "build_report.json",
                batch_size=2,
            )
            output = pq.read_table(root / "rl_train.parquet")

        self.assertEqual(report["train_membership_mode"], "doc_content_family")
        self.assertTrue(report["derived_split_unit_ids"])
        self.assertEqual(
            output.column("split_unit_id").to_pylist(),
            ["family-a", "family-b"],
        )

    def test_builder_keeps_only_train_rows_and_preserves_lineage(self):
        builder = load_script("build_nano_r33_rl_dataset")
        verifier = load_script("verify_nano_r33_rl_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            base = root / "base.parquet"
            manifest = root / "split_manifest.json"
            output = root / "rl_train.parquet"
            build_report = root / "build_report.json"
            write_base(base)
            write_actor_sidecar(base)
            write_manifest(manifest)
            family_manifest, family_coverage = write_family_contract(root)

            report = builder.build_dataset(
                base_parquet=base,
                actor_sidecar_source=base,
                split_manifest=manifest,
                content_family_manifest=family_manifest,
                content_family_coverage=family_coverage,
                expected_rows=2,
                output=output,
                report_json=build_report,
                batch_size=2,
            )
            table = pq.read_table(output)
            verification = verifier.verify_dataset(
                dataset=output,
                split_manifest=manifest,
                content_family_manifest=family_manifest,
                content_family_coverage=family_coverage,
                expected_rows=2,
                expected_d_model=4,
            )
            output_sidecar = pathlib.Path(str(output) + ".nla_meta.yaml")
            output_sidecar_exists = output_sidecar.is_file()
            sidecar = yaml.safe_load(output_sidecar.read_text())

        self.assertEqual(report["schema_version"], "nano_r33_rl_dataset.v3")
        self.assertEqual(report["rows"], 2)
        self.assertEqual(set(table.column("doc_id").to_pylist()), {"train-a", "train-b"})
        self.assertIn("token_ids_prefix", table.column_names)
        self.assertNotIn("api_explanation", table.column_names)
        self.assertEqual(
            set(table.column("split_unit_id").to_pylist()), {"family-a", "family-b"}
        )
        self.assertEqual(
            table.column("prompt").to_pylist(),
            [
                [{"role": "user", "content": "Inspect <concept><INJECT></concept> and respond."}],
                [{"role": "user", "content": "Inspect <concept><INJECT></concept> and respond."}],
            ],
        )
        self.assertIn(b"source_base_sha256", table.schema.metadata)
        self.assertTrue(output_sidecar_exists)
        self.assertEqual(sidecar["stage"], "rl")
        self.assertEqual(sidecar["row_count"], 2)
        self.assertEqual(sidecar["tokens"]["injection_token_id"], 42019)
        self.assertTrue(verification["passed"])
        self.assertEqual(verification["heldout_doc_overlap_count"], 0)
        self.assertEqual(verification["canonical_prompt_rows"], 2)
        self.assertEqual(verification["activation_layer_counts"], {"33": 2})

    def test_builder_rejects_actor_sidecar_dimension_mismatch(self):
        builder = load_script("build_nano_r33_rl_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            base = root / "base.parquet"
            manifest = root / "split_manifest.json"
            write_base(base)
            write_actor_sidecar(base)
            write_manifest(manifest)
            family_manifest, family_coverage = write_family_contract(root)
            sidecar_path = pathlib.Path(str(base) + ".nla_meta.yaml")
            sidecar = yaml.safe_load(sidecar_path.read_text())
            sidecar["extraction"]["d_model"] = 5
            sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))

            with self.assertRaisesRegex(
                builder.RLDatasetBuildError,
                "activation dimension",
            ):
                builder.build_dataset(
                    base_parquet=base,
                    actor_sidecar_source=base,
                    split_manifest=manifest,
                    content_family_manifest=family_manifest,
                    content_family_coverage=family_coverage,
                    expected_rows=2,
                    output=root / "rl_train.parquet",
                    report_json=root / "build_report.json",
                    batch_size=2,
                )

    def test_verifier_fails_closed_when_lineage_files_are_missing(self):
        builder = load_script("build_nano_r33_rl_dataset")
        verifier = load_script("verify_nano_r33_rl_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            base = root / "base.parquet"
            manifest = root / "split_manifest.json"
            output = root / "rl_train.parquet"
            write_base(base)
            write_actor_sidecar(base)
            write_manifest(manifest)
            family_manifest, family_coverage = write_family_contract(root)
            builder.build_dataset(
                base_parquet=base,
                actor_sidecar_source=base,
                split_manifest=manifest,
                content_family_manifest=family_manifest,
                content_family_coverage=family_coverage,
                expected_rows=2,
                output=output,
                report_json=root / "build_report.json",
                batch_size=2,
            )
            base.unlink()
            pathlib.Path(str(base) + ".nla_meta.yaml").unlink()

            report = verifier.verify_dataset(
                dataset=output,
                split_manifest=manifest,
                content_family_manifest=family_manifest,
                content_family_coverage=family_coverage,
                expected_rows=2,
                expected_d_model=4,
            )

        self.assertFalse(report["passed"])
        self.assertIn("source_base_provenance", report["blockers"])
        self.assertIn("actor_sidecar_provenance", report["blockers"])

    def test_verifier_rejects_nonfinite_wrong_dimension_duplicates_and_heldout_docs(self):
        verifier = load_script("verify_nano_r33_rl_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            manifest = root / "split_manifest.json"
            write_manifest(manifest)
            family_manifest, family_coverage = write_family_contract(root)
            dataset = root / "bad.parquet"
            pq.write_table(
                pa.table(
                    {
                        "doc_id": ["validation-a", "validation-a"],
                        "split_unit_id": ["family-v", "family-v"],
                        "content_family_id": ["family-v", "family-v"],
                        "token_position": [4, 4],
                        "prompt": ["a", "b"],
                        "token_ids_prefix": [[1, 2], [1, 2]],
                        "activation_vector": [
                            [1.0, float("nan"), 0.0],
                            [1.0, 0.0, 0.0],
                        ],
                    }
                ),
                dataset,
            )

            report = verifier.verify_dataset(
                dataset=dataset,
                split_manifest=manifest,
                content_family_manifest=family_manifest,
                content_family_coverage=family_coverage,
                expected_rows=2,
                expected_d_model=4,
            )

        self.assertFalse(report["passed"])
        self.assertIn("nonfinite_activations", report["blockers"])
        self.assertIn("d_model", report["blockers"])
        self.assertIn("duplicate_provenance_keys", report["blockers"])
        self.assertIn("heldout_doc_overlap", report["blockers"])
        self.assertIn("prompt_schema", report["blockers"])
        self.assertIn("dataset_sidecar", report["blockers"])


if __name__ == "__main__":
    unittest.main()
