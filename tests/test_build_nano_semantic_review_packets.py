import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_nano_semantic_review_packets as builder  # noqa: E402
import score_nano_semantic_review as scorer  # noqa: E402


def _write_inputs(root: pathlib.Path, sources=("sft", "rl"), rows=18):
    result = {}
    for source in sources:
        generated_path = root / f"{source}_generated.jsonl"
        generated_rows = []
        transform_paths = {}
        for index in range(rows):
            explanation = " ".join(
                [f"original {source} {index}", *(["word"] * (index + 1))]
            )
            generated_rows.append(
                {
                    "split": "validation",
                    "row_index": index,
                    "doc_id": f"doc-{index}",
                    "content_family_id": f"family-{index}",
                    "controls": {
                        "real": {
                            "generated": f"<explanation>{explanation}</explanation>",
                            "parsed": {"explanation": explanation},
                        }
                    },
                }
            )
        generated_path.write_text(
            "".join(json.dumps(row) + "\n" for row in generated_rows)
        )
        for transform in ("light", "aggressive"):
            path = root / f"{source}_{transform}.jsonl"
            transformed_rows = []
            for index, generated in enumerate(generated_rows):
                original = generated["controls"]["real"]["parsed"]["explanation"]
                raw_source = generated["controls"]["real"]["generated"]
                transformed_rows.append(
                    {
                        "row_key": f"validation:{index}",
                        "transform": transform,
                        "transformed_text": f"<explanation>{transform} {original}</explanation>",
                        "source_sha256": builder._text_sha256(raw_source),
                        "prompt_sha256": f"prompt-{source}-{transform}-{index}",
                        "model": "test-model",
                    }
                )
            path.write_text("".join(json.dumps(row) + "\n" for row in transformed_rows))
            transform_paths[transform] = str(path)
        result[source] = {
            "generated_jsonl": str(generated_path),
            "transforms": transform_paths,
        }
    return result


def _config(tmp_path: pathlib.Path, output_name: str = "review"):
    return {
        "schema_version": builder.SCHEMA_VERSION,
        "paths": {
            "sources": _write_inputs(tmp_path),
            "output_dir": str(tmp_path / output_name),
        },
        "protocol": {
            "sample_size": 12,
            "length_bins": 3,
            "reviewer_ids": ["reviewer_1", "reviewer_2"],
            "seed": 20260722,
        },
    }


def test_review_packet_is_balanced_blinded_and_deterministic(tmp_path):
    config = _config(tmp_path)
    first = builder.build_packets(config)
    config["paths"]["output_dir"] = str(tmp_path / "review_again")
    second = builder.build_packets(config)

    assert first["passed"]
    assert first["sample_size"] == 12
    assert first["unique_content_families"] == 12
    assert set(first["source_counts"]) == {"rl", "sft"}
    assert set(first["transform_counts"]) == {"aggressive", "light"}

    first_packet = json.loads(
        pathlib.Path(first["packet_paths"]["reviewer_1"]).read_text()
    )
    second_packet = json.loads(
        pathlib.Path(second["packet_paths"]["reviewer_1"]).read_text()
    )
    assert first_packet["rows"] == second_packet["rows"]
    assert len(first_packet["rows"]) == 12
    for row in first_packet["rows"]:
        assert set(row) == {
            "review_id",
            "original_explanation",
            "transformed_explanation",
            "rating",
        }
        assert "source" not in row
        assert "transform" not in row


def test_review_packet_rejects_source_hash_mismatch(tmp_path):
    config = _config(tmp_path)
    path = pathlib.Path(config["paths"]["sources"]["sft"]["transforms"]["light"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["source_sha256"] = "bad"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(builder.SemanticReviewError, match="source hash mismatch"):
        builder.build_packets(config)


def test_score_review_unblinds_and_reports_agreement(tmp_path):
    result = builder.build_packets(_config(tmp_path))
    for packet_name in result["packet_paths"].values():
        packet_path = pathlib.Path(packet_name)
        packet = json.loads(packet_path.read_text())
        for row in packet["rows"]:
            row["rating"] = {
                "meaning_preservation": 5,
                "omission_severity": 0,
                "unsupported_addition_severity": 0,
                "contradiction_present": False,
                "fluent_and_interpretable": True,
                "notes": "",
            }
        packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")

    report = scorer.score(
        {
            "schema_version": scorer.SCHEMA_VERSION,
            "paths": {
                "answer_key": result["answer_key"],
                "reviews": result["packet_paths"],
                "output_json": str(tmp_path / "score.json"),
            },
            "protocol": {
                "minimum_reviewers": 2,
                "gated_transforms": ["light", "aggressive"],
                "thresholds": {
                    "minimum_meaning_preservation": 4.0,
                    "maximum_contradiction_fraction": 0.05,
                    "maximum_unsupported_addition_severity": 0.5,
                    "minimum_fluent_fraction": 0.9,
                },
            },
        }
    )
    assert report["passed"]
    assert report["review_complete"]
    assert report["meaning_preservation_gate_passed"]
    assert report["ratings"] == 24
    assert report["agreement"]["reviewer_1/reviewer_2"][
        "meaning_preservation_quadratic_weighted_kappa"
    ] == 1.0
