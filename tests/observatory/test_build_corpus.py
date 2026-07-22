from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from observatory.build_corpus import parse_sections, run
from observatory.common import ObservatoryConfigError, load_config


def _explanation(index: int) -> str:
    return (
        f"Syntax/continuation feature: Expects a complement for item {index}.\n\n"
        f"Discourse/semantic feature: Signals topic {index} and its context.\n\n"
        "Genre/register feature: Uses a formal explanatory register.\n\n"
        "Final-token constraint: The last token requires a following phrase."
    )


def _fixture_config(tmp_path: Path) -> Path:
    panel_rows = []
    source_text = " ".join(f"source{position}" for position in range(180))
    for index in range(50):
        panel_rows.append(
            {
                "row_index": 1000 + index,
                "doc_id": f"dataset:train:{index}",
                "token_position": 140 + index,
                "activation_norm": 100.0 + index,
                "source_text": source_text,
                "reference_text": _explanation(index),
                "candidate_text": _explanation(index).replace("Expects", "Anticipates"),
                "stratum": {"token_position_bin": index % 4},
            }
        )
    panel = {
        "schema_version": "nano_r33_qualitative_panel.v1",
        "splits": {
            "validation": {"row_count": 50, "rows": panel_rows},
            "test": {"row_count": 0, "rows": []},
        },
    }
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(panel))
    cache_path = tmp_path / "cache.npz"
    np.savez_compressed(
        cache_path,
        validation__row_indices=np.arange(1000, 1050, dtype=np.int64),
        validation__content_family_ids=np.asarray(
            [f"cf_{index:03d}" for index in range(50)]
        ),
        validation__doc_ids=np.asarray(
            [f"dataset:train:{index}" for index in range(50)]
        ),
    )
    config = {
        "schema_version": "nano_viz_offline_observatory.v1",
        "paths": {
            "qualitative_panel_json": str(panel_path),
            "validation_prediction_cache_npz": str(cache_path),
            "corpus_dir": str(tmp_path / "corpus"),
        },
        "selection": {
            "deep_dive_rows": 50,
            "behavior_rows": 24,
            "canary_rows": 8,
            "film_rows": 10,
            "film_min_position": 130,
            "seed": 17,
        },
        "grid": {
            "word_occlusion_limit": 80,
            "truncation_points": 10,
            "corruption_rates": [0.1, 0.25, 0.5],
            "alternate_tellings": 8,
        },
        "evaluation": {},
        "gates": {},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def test_build_corpus_is_deterministic_and_control_complete(tmp_path: Path) -> None:
    config_path = _fixture_config(tmp_path)
    first = run(config_path, tmp_path / "first")
    second = run(config_path, tmp_path / "second")

    assert first["passed"] is True
    assert first["rows"] == 50
    assert first["interventions"] == second["interventions"]
    assert first["grid_spec_sha256"] == second["grid_spec_sha256"]
    assert first["selection_manifest_sha256"] == second["selection_manifest_sha256"]
    assert first["grid_summary"]["control_group_count"] == 300
    assert first["grid_summary"]["control_groups_complete"] is True
    assert first["grid_summary"]["family_counts"]["clause_swap"] == 1332

    selection = json.loads((tmp_path / "first" / "selection_manifest.json").read_text())
    assert len(selection["behavior_row_ids"]) == 24
    assert len(selection["canary_row_ids"]) == 8
    assert len(set(selection["canary_row_ids"]) & set(selection["behavior_row_ids"])) == 4
    assert len(selection["film_row_ids"]) == 10


def test_parse_sections_rejects_noncanonical_layout() -> None:
    with pytest.raises(ObservatoryConfigError, match="four canonical sections"):
        parse_sections("Syntax/continuation feature: only one section")


def test_config_rejects_scope_drift(tmp_path: Path) -> None:
    config_path = _fixture_config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["selection"]["deep_dive_rows"] = 49
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ObservatoryConfigError, match="deep_dive_rows must be 50"):
        load_config(config_path)
