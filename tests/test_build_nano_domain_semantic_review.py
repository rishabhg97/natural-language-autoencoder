import csv
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_nano_domain_semantic_review as review  # noqa: E402
import score_nano_domain_semantic_review as scorer  # noqa: E402


def _rows(source):
    rows = []
    for family in ("injection", "cyber"):
        for condition in ("negative", "positive"):
            for case in range(2):
                row_id = f"{family}_{condition}_{case}"
                for position_index, position in enumerate(
                    ("pre_condition", "condition_close", "pre_decision")
                ):
                    rows.append(
                        {
                            "row_id": row_id,
                            "pair_id": f"{family}_{case}",
                            "scenario_family": family,
                            "condition": condition,
                            "position_name": position,
                            "rendered_prompt_sha256": f"prompt-{row_id}",
                            "causal_prefix_sha256": f"prefix-{row_id}-{position}",
                            "token_index": position_index,
                            "token_text": position,
                            "system_prompt": "system",
                            "user_prompt": f"user {family} {condition}",
                            "visible_continuation": "DECISION: PROCEED",
                            "controls": {
                                "real": {
                                    "parsed": {
                                        "usable": True,
                                        "explanation": f"{source} explanation {row_id} {position}",
                                    }
                                }
                            },
                        }
                    )
    return rows


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_build_review_is_balanced_and_blinded(tmp_path):
    sft = tmp_path / "sft.jsonl"
    rl = tmp_path / "rl.jsonl"
    _write_jsonl(sft, _rows("sft"))
    _write_jsonl(rl, _rows("rl"))
    output = tmp_path / "review"
    report = review.build_review(
        {
            "inputs": {"sft": str(sft), "rl": str(rl)},
            "output_dir": str(output),
            "seed": 17,
            "samples_per_stratum": 1,
        }
    )

    assert report["passed"]
    assert report["matched_cells"] == 12
    assert report["review_items"] == 24
    assert report["source_counts"] == {"rl": 12, "sft": 12}
    with (output / "review_packet_reviewer_1.csv").open() as handle:
        packet = list(csv.DictReader(handle))
    assert len(packet) == 24
    assert "source" not in packet[0]
    assert all(not row["prompt_grounding_0_2"] for row in packet)
    answer_key = json.loads((output / "answer_key.json").read_text())
    assert {item["source"] for item in answer_key["items"].values()} == {
        "sft",
        "rl",
    }

    for reviewer in (1, 2):
        path = output / f"review_packet_reviewer_{reviewer}.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        for row in rows:
            row["prompt_grounding_0_2"] = "2"
            row["condition_relevance_0_2_or_na"] = (
                "NA" if row["position_name"] == "pre_condition" else "2"
            )
            row["hallucination_severity_0_2"] = "0"
            row["syntactic_only_yes_no"] = "no"
            row["behavior_prediction_usefulness_0_2"] = "2"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    score_report = scorer.score_reviews(
        {
            "answer_key": str(output / "answer_key.json"),
            "review_packets": {
                "reviewer_1": str(output / "review_packet_reviewer_1.csv"),
                "reviewer_2": str(output / "review_packet_reviewer_2.csv"),
            },
            "output_json": str(output / "score_report.json"),
        }
    )
    assert score_report["passed"]
    assert score_report["items_per_reviewer"] == 24
    assert score_report["inter_reviewer_agreement"]["prompt_grounding"][
        "exact_agreement_fraction"
    ] == 1.0
