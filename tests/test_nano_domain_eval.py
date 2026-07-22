import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import nano_domain_eval as domain_eval  # noqa: E402


def test_load_config_supports_recursive_overlays(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        """\
schema_version: nano_domain_eval.v1
paths:
  output_root: /base/output
  manifest_jsonl: /shared/manifest.jsonl
models:
  base_hf: /models/base
  av_dcp: /models/base-av
evaluation:
  boundary: 33
  nla_controls: [real, shuffled, none]
manifest:
  expected_pairs: 16
"""
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        """\
schema_version: nano_domain_eval.v1
extends: base.yaml
paths:
  output_root: /child/output
models:
  av_dcp: /models/child-av
evaluation:
  nla_controls: [real]
"""
    )

    config = domain_eval.load_config(child)

    assert config["paths"] == {
        "output_root": "/child/output",
        "manifest_jsonl": "/shared/manifest.jsonl",
    }
    assert config["models"] == {
        "base_hf": "/models/base",
        "av_dcp": "/models/child-av",
    }
    assert config["evaluation"]["boundary"] == 33
    assert config["evaluation"]["nla_controls"] == ["real"]
    assert config["manifest"]["expected_pairs"] == 16
    assert "extends" not in config


def test_load_config_rejects_extends_cycle(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n")
    second.write_text("extends: first.yaml\n")

    with pytest.raises(domain_eval.DomainEvalError, match="extends cycle"):
        domain_eval.load_config(first)


def _config(tmp_path: pathlib.Path):
    return {
        "schema_version": domain_eval.SCHEMA_VERSION,
        "paths": {
            "manifest_jsonl": str(tmp_path / "manifest.jsonl"),
            "manifest_report_json": str(tmp_path / "manifest_report.json"),
            "activations_jsonl": str(tmp_path / "activations.jsonl"),
            "activation_report_json": str(tmp_path / "activation_report.json"),
            "descriptions_jsonl": str(tmp_path / "descriptions.jsonl"),
            "description_report_json": str(tmp_path / "description_report.json"),
            "behavior_jsonl": str(tmp_path / "behavior.jsonl"),
            "behavior_report_json": str(tmp_path / "behavior_report.json"),
            "analysis_report_json": str(tmp_path / "analysis.json"),
        },
        "models": {},
        "evaluation": {},
        "manifest": {
            "expected_pairs": 2,
            "seed": 7,
            "families": {
                "trust": {
                    "system_prompt": "system",
                    "prompt_template": "START {document} END QUESTION",
                    "position_anchors": {
                        "pre_condition": "START",
                        "condition_close": "END",
                        "pre_decision": "QUESTION",
                    },
                    "concept_lexicon": ["untrusted"],
                    "conditions": {
                        "clean": {
                            "label": 0,
                            "expected_decision": "SAFE",
                            "values": {"document": "ordinary"},
                        },
                        "injection": {
                            "label": 1,
                            "expected_decision": "SAFE",
                            "values": {"document": "untrusted instruction"},
                        },
                    },
                    "cases": [
                        {"pair_id": "p1", "template_id": "t1", "shared": {}},
                        {"pair_id": "p2", "template_id": "t2", "shared": {}},
                    ],
                }
            },
        },
    }


def test_build_manifest_is_paired_and_deterministic(tmp_path):
    config = _config(tmp_path)
    first = domain_eval.build_manifest(config)
    first_rows = domain_eval.read_jsonl(pathlib.Path(config["paths"]["manifest_jsonl"]))
    second = domain_eval.build_manifest(config)
    second_rows = domain_eval.read_jsonl(pathlib.Path(config["paths"]["manifest_jsonl"]))
    assert first["passed"] and second["passed"]
    assert first_rows == second_rows
    assert len(first_rows) == 4
    assert {row["condition_label"] for row in first_rows} == {0, 1}


def test_build_manifest_rejects_ambiguous_anchor(tmp_path):
    config = _config(tmp_path)
    config["manifest"]["families"]["trust"]["position_anchors"]["pre_condition"] = "T"
    with pytest.raises(domain_eval.DomainEvalError, match="exactly once"):
        domain_eval.build_manifest(config)


def test_analyze_reports_paired_lexicon_effect(tmp_path):
    config = _config(tmp_path)
    domain_eval.build_manifest(config)
    manifest = domain_eval.read_jsonl(pathlib.Path(config["paths"]["manifest_jsonl"]))
    descriptions = []
    for row in manifest:
        for position in ("pre_condition", "condition_close", "pre_decision"):
            text = (
                "shared prefix state"
                if position == "pre_condition"
                else "untrusted instruction detected"
                if row["condition_label"]
                else "ordinary text"
            )
            descriptions.append(
                {
                    **row,
                    "position_name": position,
                    "parsed_decision": "SAFE",
                    "decision_matches_expected": True,
                    "controls": {
                        control: {
                            "generated": text,
                            "parsed": {"explanation": text, "usable": True, "closed": True}
                        }
                        for control in ("real", "shuffled", "none")
                    },
                }
            )
    domain_eval.write_jsonl(
        pathlib.Path(config["paths"]["descriptions_jsonl"]), descriptions
    )
    result = domain_eval.analyze(config)
    assert result["passed"]
    assert result["decision_accuracy"] == 1.0
    assert result["paired_effects"]["trust/pre_decision"]["paired_hit_rate_difference"] == 1.0


def test_decision_parser_accepts_only_preregistered_labels():
    assert domain_eval._parse_decision("Done. DECISION: PROCEED. Thanks") == "PROCEED"
    assert domain_eval._parse_decision("DECISION: MAKE_EXPLOIT") is None


def test_pair_preserving_shards_keep_conditions_together():
    rows = [
        {"row_id": "p2:a", "pair_id": "p2"},
        {"row_id": "p1:a", "pair_id": "p1"},
        {"row_id": "p2:b", "pair_id": "p2"},
        {"row_id": "p1:b", "pair_id": "p1"},
    ]

    shard_zero = domain_eval._pair_preserving_shard_rows(
        rows, shard_index=0, shard_count=2
    )
    shard_one = domain_eval._pair_preserving_shard_rows(
        rows, shard_index=1, shard_count=2
    )

    assert {row["pair_id"] for row in shard_zero} == {"p2"}
    assert {row["pair_id"] for row in shard_one} == {"p1"}
    assert len(shard_zero) == len(shard_one) == 2


def test_analyze_prefers_identity_bound_behavior_evidence(tmp_path):
    config = _config(tmp_path)
    domain_eval.build_manifest(config)
    manifest = domain_eval.read_jsonl(pathlib.Path(config["paths"]["manifest_jsonl"]))
    descriptions = []
    for row in manifest:
        for position in ("pre_condition", "condition_close", "pre_decision"):
            descriptions.append(
                {
                    **row,
                    "position_name": position,
                    "parsed_decision": None,
                    "decision_matches_expected": False,
                        "controls": {
                            "real": {
                                "generated": "ordinary text",
                                "parsed": {
                                "explanation": "ordinary text",
                                "usable": True,
                                "closed": True,
                            }
                        }
                    },
                }
            )
    domain_eval.write_jsonl(
        pathlib.Path(config["paths"]["descriptions_jsonl"]), descriptions
    )
    domain_eval.write_jsonl(
        pathlib.Path(config["paths"]["behavior_jsonl"]),
        [
            {
                "row_id": row["row_id"],
                "parsed_decision": row["expected_decision"],
                "decision_matches_expected": True,
            }
            for row in manifest
        ],
    )

    result = domain_eval.analyze(config)

    assert result["passed"]
    assert result["decision_source"] == "behavior_regeneration"
    assert result["decision_parse_rate"] == 1.0
    assert result["decision_accuracy"] == 1.0


def test_merge_descriptions_restores_activation_order(tmp_path):
    config = _config(tmp_path)
    activation_path = pathlib.Path(config["paths"]["activations_jsonl"])
    output_path = pathlib.Path(config["paths"]["descriptions_jsonl"])
    report_path = pathlib.Path(config["paths"]["description_report_json"])
    activation_rows = [
        {"row_id": f"row-{index}", "position_name": "pre_decision"}
        for index in range(4)
    ]
    domain_eval.write_jsonl(activation_path, activation_rows)
    for shard_index, indices in enumerate(((0, 2), (1, 3))):
        rows = [
            {
                **activation_rows[index],
                "controls": {
                    "real": {
                        "parsed": {"usable": True, "closed": True, "explanation": "x"}
                    }
                },
            }
            for index in indices
        ]
        shard_path = domain_eval._shard_path(output_path, shard_index, 2)
        shard_report_path = domain_eval._shard_path(
            report_path, shard_index, 2
        )
        domain_eval.write_jsonl(shard_path, rows)
        domain_eval._write_description_report(
            config,
            activation_path=activation_path,
            output_path=shard_path,
            report_path=shard_report_path,
            rows=rows,
            expected_rows=2,
            shard_index=shard_index,
            shard_count=2,
        )

    report = domain_eval.merge_descriptions(config, shard_count=2)
    merged = domain_eval.read_jsonl(output_path)

    assert report["passed"]
    assert [row["row_id"] for row in merged] == [
        "row-0",
        "row-1",
        "row-2",
        "row-3",
    ]


def test_merge_behavior_applies_parse_gate_after_global_merge(tmp_path):
    config = _config(tmp_path)
    config["evaluation"] = {"behavior_min_parse_rate": 0.75}
    domain_eval.build_manifest(config)
    manifest_path = pathlib.Path(config["paths"]["manifest_jsonl"])
    manifest = domain_eval.read_jsonl(manifest_path)
    activation_path = pathlib.Path(config["paths"]["activations_jsonl"])
    domain_eval.write_jsonl(
        activation_path,
        [
            {"row_id": row["row_id"], "position_name": "pre_decision"}
            for row in manifest
        ],
    )
    output_path = pathlib.Path(config["paths"]["behavior_jsonl"])
    report_path = pathlib.Path(config["paths"]["behavior_report_json"])
    for shard_index in range(2):
        shard_rows = []
        for index, row in enumerate(manifest):
            if index % 2 != shard_index:
                continue
            parsed = row["expected_decision"] if index < 3 else None
            shard_rows.append(
                {
                    "row_id": row["row_id"],
                    "parsed_decision": parsed,
                    "decision_matches_expected": parsed == row["expected_decision"],
                    "generated_token_count": 10,
                }
            )
        shard_path = domain_eval._shard_path(output_path, shard_index, 2)
        domain_eval.write_jsonl(shard_path, shard_rows)
        shard_report_path = domain_eval._shard_path(report_path, shard_index, 2)
        domain_eval.write_json(
            shard_report_path,
            {
                "passed": True,
                "manifest_sha256": domain_eval.sha256_file(manifest_path),
                "activation_sha256": domain_eval.sha256_file(activation_path),
                "behavior": {"sha256": domain_eval.sha256_file(shard_path)},
            },
        )

    report = domain_eval.merge_behavior(config, shard_count=2)

    assert report["passed"]
    assert report["decision_parse_rate"] == 0.75
    assert report["merged_shards"] == 2


def test_merge_activations_applies_global_prefix_invariance_gate(tmp_path):
    config = _config(tmp_path)
    config["evaluation"] = {
        "boundary": 33,
        "pre_condition_invariance_max_rel_l2": 1e-5,
    }
    domain_eval.build_manifest(config)
    manifest_path = pathlib.Path(config["paths"]["manifest_jsonl"])
    manifest = domain_eval.read_jsonl(manifest_path)
    output_path = pathlib.Path(config["paths"]["activations_jsonl"])
    report_path = pathlib.Path(config["paths"]["activation_report_json"])
    for shard_index in range(2):
        selected = [
            row for index, row in enumerate(manifest) if index % 2 == shard_index
        ]
        shard_rows = []
        for row in selected:
            for position_index, position in enumerate(domain_eval.POSITION_NAMES):
                pair_number = int(row["pair_id"][-1])
                shard_rows.append(
                    {
                        "row_id": row["row_id"],
                        "pair_id": row["pair_id"],
                        "position_name": position,
                        "causal_prefix_sha256": (
                            f"pair-{row['pair_id']}-prefix"
                            if position == "pre_condition"
                            else f"{row['row_id']}-{position}"
                        ),
                        "activation_vector": [
                            float(pair_number),
                            float(position_index),
                        ],
                    }
                )
        shard_path = domain_eval._shard_path(output_path, shard_index, 2)
        shard_report_path = domain_eval._shard_path(report_path, shard_index, 2)
        domain_eval.write_jsonl(shard_path, shard_rows)
        domain_eval._write_activation_report(
            config,
            manifest_path=manifest_path,
            output_path=shard_path,
            report_path=shard_report_path,
            rows=shard_rows,
            manifest_rows=len(selected),
            manifest_total_rows=len(manifest),
            boundary=33,
            capture_backend="truncated_causal_prefix_per_anchor",
            apply_invariance_gate=False,
            shard_index=shard_index,
            shard_count=2,
        )

    report = domain_eval.merge_activations(config, shard_count=2)

    assert report["passed"]
    assert report["pre_condition_invariance"]["prefix_hash_equal_pairs"] == 2
    assert report["pre_condition_invariance"]["max_relative_l2"] == 0.0
