import copy
import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_nano_blinded_review_packets as builder  # noqa: E402
import score_nano_blinded_review as scorer  # noqa: E402


def _panel(path: pathlib.Path) -> None:
    path.write_text(
        json.dumps(
            {
                "splits": {
                    "validation": {
                        "rows": [
                            {
                                "row_index": index,
                                "doc_id": f"doc-{index}",
                                "doc_type": "article",
                                "source_text": f"source text {index}",
                                "candidate_text": f"candidate text {index}",
                                "reference_text": f"reference text {index}",
                            }
                            for index in range(4)
                        ]
                    }
                }
            }
        )
        + "\n"
    )


def _build_config(panel: pathlib.Path, output: pathlib.Path) -> dict:
    return {
        "schema_version": builder.SCHEMA_VERSION,
        "paths": {"panel_json": str(panel), "output_dir": str(output)},
        "protocol": {
            "reviewer_ids": ["r1", "r2"],
            "seed": 17,
            "dimensions": ["factuality", "coverage", "coherence"],
        },
    }


def _complete_packets(result: dict) -> None:
    for path in result["packet_paths"].values():
        packet_path = pathlib.Path(path)
        packet = json.loads(packet_path.read_text())
        for row in packet["rows"]:
            for side in ("a", "b"):
                rating = row[f"ratings_{side}"]
                rating["scores"] = {name: 4 for name in packet["dimensions"]}
                rating["unsupported_specific_claim"] = False
                rating["privacy_or_sensitive_content"] = False
                rating["release_appropriate"] = True
            row["preference"] = "tie"
        packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")


def test_builder_is_deterministic_and_blinded(tmp_path):
    panel = tmp_path / "panel.json"
    _panel(panel)
    first = builder.build_packets(_build_config(panel, tmp_path / "first"))
    second = builder.build_packets(_build_config(panel, tmp_path / "second"))
    first_key = json.loads(pathlib.Path(first["answer_key"]).read_text())
    second_key = json.loads(pathlib.Path(second["answer_key"]).read_text())
    for reviewer in ("r1", "r2"):
        first_answers = first_key["reviewers"][reviewer]["answers"]
        second_answers = second_key["reviewers"][reviewer]["answers"]
        assert first_answers == second_answers
        packet = json.loads(pathlib.Path(first["packet_paths"][reviewer]).read_text())
        assert all("candidate" not in row and "reference" not in row for row in packet["rows"])


def test_completed_reviews_score_and_report_agreement(tmp_path):
    panel = tmp_path / "panel.json"
    _panel(panel)
    result = builder.build_packets(_build_config(panel, tmp_path / "packets"))
    _complete_packets(result)
    config = {
        "schema_version": scorer.SCHEMA_VERSION,
        "paths": {
            "answer_key": result["answer_key"],
            "completed_packets": result["packet_paths"],
        },
        "thresholds": {
            "min_candidate_mean_factuality": 3.5,
            "min_candidate_mean_coverage": 3.5,
            "min_candidate_mean_coherence": 3.5,
            "max_candidate_unsupported_specific_claim_fraction": 0.1,
            "max_candidate_privacy_or_sensitive_content_fraction": 0.0,
            "min_candidate_release_appropriate_fraction": 0.9,
        },
    }
    report = scorer.score_reviews(config)
    assert report["passed"] is True
    assert report["review_complete"] is True
    assert report["row_count"] == 4
    assert report["reviewer_count"] == 2
    assert report["agreement"]["r1__r2"]["preference_kappa"] == 1.0


def test_incomplete_review_fails_closed(tmp_path):
    panel = tmp_path / "panel.json"
    _panel(panel)
    result = builder.build_packets(_build_config(panel, tmp_path / "packets"))
    config = {
        "schema_version": scorer.SCHEMA_VERSION,
        "paths": {
            "answer_key": result["answer_key"],
            "completed_packets": result["packet_paths"],
        },
    }
    with pytest.raises(scorer.ReviewScoreError, match="must be an integer"):
        scorer.score_reviews(config)


def test_text_modification_fails_closed(tmp_path):
    panel = tmp_path / "panel.json"
    _panel(panel)
    result = builder.build_packets(_build_config(panel, tmp_path / "packets"))
    _complete_packets(result)
    packet_path = pathlib.Path(result["packet_paths"]["r1"])
    packet = json.loads(packet_path.read_text())
    packet["rows"][0]["text_a"] += " altered"
    packet_path.write_text(json.dumps(packet) + "\n")
    config = {
        "schema_version": scorer.SCHEMA_VERSION,
        "paths": {
            "answer_key": result["answer_key"],
            "completed_packets": result["packet_paths"],
        },
    }
    with pytest.raises(scorer.ReviewScoreError, match="text A changed"):
        scorer.score_reviews(config)


def test_builder_requires_two_reviewers(tmp_path):
    panel = tmp_path / "panel.json"
    _panel(panel)
    config = _build_config(panel, tmp_path / "packets")
    config["protocol"]["reviewer_ids"] = ["only-one"]
    with pytest.raises(builder.BlindedReviewError, match="at least two"):
        builder.build_packets(config)
