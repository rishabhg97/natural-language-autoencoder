import importlib.util
import hashlib
import json
import pathlib
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "nano_roundtrip_research_study.py"
    spec = importlib.util.spec_from_file_location("nano_roundtrip_research_study", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def generated_record(row_index, family, text, lineage):
    return {
        "split": "validation",
        "row_index": row_index,
        "sample_uuid": f"sample-{row_index}",
        "content_family_id": family,
        "generation_protocol_sha256": "a" * 64,
        "generation_protocol": {
            "controls": ["real"],
            "max_new_tokens": 384,
            "seed": 17,
            "prefix": "",
            "stop_text": "</explanation>",
            "generated_text_fallback": "raw",
            "backend": "legacy_batch",
            "worker_count": 2,
            "injection_scale": "75",
            "torch_dtype": "bfloat16",
        },
        "generation_provenance_sha256": "b" * 64,
        "generation_provenance": {
            "checkpoint": f"/{lineage}/av",
            "model_fingerprint": f"hf_model_sha256:{lineage[0] * 64}",
        },
        "target_explanation": f"teacher {row_index}",
        "controls": {
            "real": {
                "generated": f"<explanation>{text}</explanation>",
                "parsed": {"explanation": text, "closed": True, "usable": True},
            }
        },
    }


def report(nmse, losses):
    return {
        "splits": {
            "validation": {
                "row_count": 2,
                "row_indices": [0, 1],
                "content_family_ids": ["family-a", "family-b"],
                "independent_family_count": 2,
                "variants": {
                    name: {"directional_mse": value}
                    for name, value in {
                        "av_real": nmse,
                        "teacher": 0.3,
                        "source_context": 0.4,
                        "source_raw": 0.1,
                    }.items()
                },
                "rowwise_directional_mse": {"av_real": losses},
            }
        }
    }


def semantic_report(losses_by_transform):
    rowwise = {
        ("av_real" if name == "real" else f"av_{name}"): values
        for name, values in losses_by_transform.items()
    }
    return {
        "combined_score": {
            "splits": {
                "validation": {
                    "row_count": 2,
                    "row_indices": [0, 1],
                    "content_family_ids": ["family-0", "family-1"],
                    "independent_family_count": 2,
                    "rowwise_directional_mse": rowwise,
                }
            }
        }
    }


class NanoRoundtripResearchStudyTests(unittest.TestCase):
    def make_config(self, root):
        sft = root / "sft.jsonl"
        rl = root / "rl.jsonl"
        for path, label in ((sft, "sft"), (rl, "rl")):
            path.write_text(
                "".join(
                    json.dumps(
                        generated_record(
                            index,
                            f"family-{index}",
                            f"{label} {index}",
                            label,
                        )
                    )
                    + "\n"
                    for index in range(2)
                )
            )
        return {
            "schema_version": "nano_roundtrip_research_study.v1",
            "paths": {
                "code_root": str(ROOT),
                "render_dir": str(root / "rendered"),
                "output_root": str(root / "output"),
                "train_parquet": str(root / "train.parquet"),
                "validation_parquet": str(root / "validation.parquet"),
                "validation_control_parquet": str(
                    root / "validation_controls.parquet"
                ),
                "thin_validation_parquet": str(root / "thin_validation.parquet"),
                "control_source_parquet": str(root / "control_source.parquet"),
                "control_join_columns": ["doc_id", "n_raw_tokens"],
                "content_family_manifest": str(root / "families.json"),
                "content_family_coverage": str(root / "coverage.json"),
            },
            "texts": {
                name: {
                    "generated_jsonl": str(path),
                    "av_hf_checkpoint": f"/{name}/av",
                    "av_model_fingerprint": f"hf_model_sha256:{name[0] * 64}",
                    "av_tokenizer_fingerprint": f"tokenizer_files_sha256:{'c' * 64}",
                }
                for name, path in (("sft", sft), ("rl", rl))
            },
            "critics": {
                name: {"checkpoint": f"/{name}/critic"}
                for name in ("sft", "rl", "independent")
            },
            "evaluation": {
                "validation_limit": 2,
                "max_new_tokens": 384,
                "seed": 17,
                "selection_seed": 19,
                "min_independent_families": 2,
                "bootstrap_samples": 100,
            },
            "execution": {"python": "python", "gpu_devices": ["0", "1"]},
            "semantic_stress": {
                "deterministic_transforms": ["format_normalized"],
                "external_transforms": {},
            },
        }

    def test_materialize_dataset_controls_preserves_order_and_exact_prefixes(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config = self.make_config(root)
            thin = pa.table(
                {
                    "doc_id": ["doc-b", "doc-a"],
                    "n_raw_tokens": [9, 4],
                    "detokenized_text_truncated": ["second", "first"],
                    "activation_vector": [[2.0, 3.0], [0.0, 1.0]],
                }
            )
            source = pa.table(
                {
                    "doc_id": ["doc-a", "doc-b"],
                    "n_raw_tokens": [4, 9],
                    "detokenized_text_truncated": ["first", "second"],
                    "token_position": [3, 8],
                    "token_id": [13, 19],
                    "token_text": ["a", "b"],
                    "token_ids_prefix": [[1, 2, 3, 13], [5, 6, 7, 8, 19]],
                }
            )
            pq.write_table(thin, config["paths"]["thin_validation_parquet"])
            pq.write_table(source, config["paths"]["control_source_parquet"])

            result = module.materialize_dataset_controls(config)
            enriched = pq.read_table(config["paths"]["validation_control_parquet"])

        self.assertTrue(result["passed"])
        self.assertEqual(enriched["doc_id"].to_pylist(), ["doc-b", "doc-a"])
        self.assertEqual(
            enriched["token_ids_prefix"].to_pylist(),
            [[5, 6, 7, 8, 19], [1, 2, 3, 13]],
        )

    def test_preflight_and_render_preserve_protocol_identity(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config = self.make_config(root)
            config["critics"]["sft"]["runtime_checkpoint"] = "/staged/sft-critic"
            config_path = root / "study.yaml"
            config_path.write_text(yaml.safe_dump(config))

            loaded = module.load_config(config_path)
            preflight = module.audit_generated_pair(loaded)
            manifest = module.render(config_path, loaded)

        self.assertTrue(preflight["passed"])
        self.assertEqual(len(manifest["grid_jobs"]), 6)
        self.assertEqual(len(manifest["semantic_stress_jobs"]), 6)
        semantic_job = next(
            item
            for item in manifest["semantic_stress_jobs"]
            if item["name"] == "sft_text__sft_critic__semantic_stress"
        )
        checkpoint_index = semantic_job["command"].index("--ar-checkpoint-dir") + 1
        self.assertEqual(
            semantic_job["command"][checkpoint_index], "/staged/sft-critic"
        )

    def test_preflight_rejects_actor_fingerprint_mismatch(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config = self.make_config(root)
            config["texts"]["rl"]["av_model_fingerprint"] = "wrong"

            preflight = module.audit_generated_pair(config)

        self.assertFalse(preflight["passed"])
        self.assertFalse(preflight["texts"]["rl"]["model_fingerprint_match"])

    def test_critic_staging_is_verified_and_resumable(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config = self.make_config(root)
            source = root / "critic_source"
            destination = root / "critic_stage"
            source.mkdir()
            (source / "config.json").write_text('{"model_type":"test"}\n')
            (source / "model.safetensors").write_bytes(b"weights")
            config["critics"]["sft"].update(
                {
                    "stage_from": str(source),
                    "runtime_checkpoint": str(destination),
                }
            )

            first = module.stage_critic_checkpoints(config)
            second = module.stage_critic_checkpoints(config)

        self.assertTrue(first["passed"])
        self.assertFalse(first["critics"]["sft"]["reused"])
        self.assertTrue(second["critics"]["sft"]["reused"])

    def test_multipart_file_identity_matches_s3_etag_construction(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "object.bin"
            parts = [b"abcd", b"efgh", b"ij"]
            path.write_bytes(b"".join(parts))
            expected = hashlib.md5(
                b"".join(
                    hashlib.md5(part, usedforsecurity=False).digest()
                    for part in parts
                ),
                usedforsecurity=False,
            ).hexdigest() + "-3"

            identity = module.multipart_file_identity(path, part_size_bytes=4)

        self.assertEqual(identity["multipart_etag"], expected)

    def test_analysis_separates_transfer_from_joint_gain(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config = self.make_config(root)
            grid = pathlib.Path(config["paths"]["output_root"]) / "grid"
            grid.mkdir(parents=True)
            values = {
                "sft": {"sft": (0.31, [0.30, 0.32]), "rl": (0.28, [0.27, 0.29]), "independent": (0.32, [0.31, 0.33])},
                "rl": {"sft": (0.25, [0.24, 0.26]), "rl": (0.22, [0.21, 0.23]), "independent": (0.26, [0.25, 0.27])},
            }
            for source, critics in values.items():
                for critic, (nmse, losses) in critics.items():
                    (grid / f"{source}_text__{critic}_critic.json").write_text(
                        json.dumps(report(nmse, losses))
                    )

            result = module.analyze(config)

        self.assertAlmostEqual(result["matched_joint_gain"], 0.09)
        self.assertAlmostEqual(result["actor_gain_by_fixed_critic"]["sft"]["mean"], 0.06)
        self.assertAlmostEqual(
            result["actor_gain_transfer_fraction_of_joint"]["independent"],
            2.0 / 3.0,
        )

    def test_semantic_analysis_reports_transform_effects_and_actor_retention(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config = self.make_config(root)
            config["semantic_stress"]["external_transforms"] = {
                source: {"light_paraphrase": f"/{source}/light.jsonl"}
                for source in ("sft", "rl")
            }
            config["semantic_stress"]["cross_source_contrasts"] = [
                {
                    "name": "rl_light_vs_sft_original",
                    "left": {"source": "sft", "transform": "real"},
                    "right": {"source": "rl", "transform": "light_paraphrase"},
                    "positive_interpretation": "transformed RL text wins",
                }
            ]
            semantic = pathlib.Path(config["paths"]["output_root"]) / "semantic_stress"
            semantic.mkdir(parents=True)
            for source in ("sft", "rl"):
                for critic in ("sft", "rl", "independent"):
                    base = [0.30, 0.32] if source == "sft" else [0.20, 0.22]
                    payload = semantic_report(
                        {
                            "real": base,
                            "format_normalized": [value + 0.01 for value in base],
                            "light_paraphrase": [value + 0.02 for value in base],
                        }
                    )
                    path = semantic / f"{source}_text__{critic}_critic__semantic_stress.json"
                    path.write_text(json.dumps(payload))

            result = module.analyze_semantic_stress(config)

        self.assertEqual(
            result["transforms"],
            ["real", "format_normalized", "light_paraphrase"],
        )
        self.assertAlmostEqual(
            result["transform_effects"]["sft"]["sft"]["format_normalized"]["mean"],
            0.01,
        )
        self.assertAlmostEqual(
            result["actor_gain_after_transform"]["independent"]["real"]["mean"],
            0.10,
        )
        self.assertAlmostEqual(
            result["actor_gain_retention"]["rl"]["light_paraphrase"],
            1.0,
        )
        self.assertAlmostEqual(
            result["cross_source_contrasts"]["rl_light_vs_sft_original"]
            ["by_critic"]["independent"]["mean"],
            0.08,
        )


if __name__ == "__main__":
    unittest.main()
