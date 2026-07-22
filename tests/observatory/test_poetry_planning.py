from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from observatory.common import ObservatoryConfigError, config_fingerprint
from observatory.poetry_planning import (
    build_case_records,
    contains_term,
    edit_explanation,
    load_poetry_config,
    normalized_words,
    steering_replacement,
)
from observatory.queue import VALID_PHASES, build_command


CONFIG = (
    Path(__file__).parents[2]
    / "configs"
    / "nano_viz"
    / "offline_observatory_poetry.yaml"
)


def test_poetry_config_has_leakage_free_cases() -> None:
    config = load_poetry_config(CONFIG)
    assert len(config_fingerprint(config)) == 64
    cases = build_case_records(config)
    assert len(cases) == 8
    assert len({case["case_id"] for case in cases}) == 8
    for case in cases:
        prefix_words = set(normalized_words(case["prefix_text"]))
        assert case["target_word"] not in prefix_words
        assert not prefix_words.intersection(case["target_terms"])
        assert case["target_word"] in normalized_words(case["second_line"])


def test_poetry_future_leakage_is_rejected() -> None:
    config = load_poetry_config(CONFIG)
    config["cases"] = [dict(config["cases"][0])]
    config["gates"]["minimum_cases"] = 1
    config["cases"][0]["first_line"] = "The rabbit waits to grab it,"
    with pytest.raises(ObservatoryConfigError, match="leaks future target"):
        build_case_records(config)


def test_contains_term_uses_whole_words() -> None:
    assert contains_term("The model anticipates a rabbit.", ["rabbit"])
    assert not contains_term("The model anticipates rabbitholes.", ["rabbit"])
    assert contains_term("A HOUSE or a mouse may complete it.", ["mouse", "house"])


def test_edit_explanation_is_case_insensitive_and_auditable() -> None:
    edited, changed = edit_explanation(
        "A Rabbit with a carrot has a familiar habit.",
        {"rabbit": "mouse", "carrot": "cheese", "habit": "house"},
    )
    assert edited == "A mouse with a cheese has a familiar house."
    assert changed == ["rabbit", "carrot", "habit"]


def test_steering_replacement_matches_requested_gold_norm_step() -> None:
    gold = np.array([3.0, 4.0], dtype=np.float32)
    delta = np.array([0.0, 2.0], dtype=np.float32)
    replacement = steering_replacement(gold, delta, 0.5)
    np.testing.assert_allclose(replacement, [3.0, 6.5])
    np.testing.assert_allclose(steering_replacement(gold, delta, 0.0), gold)


def test_steering_rejects_zero_direction() -> None:
    with pytest.raises(ObservatoryConfigError, match="zero-norm"):
        steering_replacement(np.ones(2), np.zeros(2), 1.0)


@pytest.mark.parametrize(
    "phase",
    [
        "poetry-prepare",
        "poetry-extract",
        "poetry-describe",
        "poetry-score",
        "poetry-reconstruct",
        "poetry-intervene",
    ],
)
def test_poetry_phases_are_queueable(phase: str) -> None:
    assert phase in VALID_PHASES
    command = build_command(
        python_bin="python",
        code_root=Path("/code"),
        config_path=Path("/code/poetry.yaml"),
        phase=phase,
    )
    assert command[-1] == phase
