#!/usr/bin/env python3
"""Deterministic synthetic fixture generator for the NLA Observatory test suite.

Every value here is synthetic (no real evidence text or numbers). The shapes
mirror src/data/types.ts exactly; the generated JSON is committed next to this
script so tests do not need Python at run time. Re-run with:

    python3 tests/fixtures/make_fixture.py
"""

from __future__ import annotations

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "nla_observatory_dashboard.v2"
ROW_IDS = ["validation-1", "validation-2"]
CRITICS = ["primary", "independent"]


def sha(seed: str) -> str:
    """Deterministic fake-but-plausible sha256 hex digest."""
    return hashlib.sha256(f"nla-fixture:{seed}".encode()).hexdigest()


def slim_metric(i: int, critic: str) -> dict:
    bump = 0.01 if critic == "independent" else 0.0
    return {
        "dmse": round(0.10 + 0.02 * i + bump, 6),
        "raw_mse": round(4.0 + 0.5 * i + bump, 6),
        "cosine": round(0.95 - 0.01 * i - bump, 6),
        "norm_ratio": round(1.10 + 0.05 * i, 6),
    }


def write(path: str, obj: dict) -> None:
    full = os.path.join(HERE, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {path}")


# --------------------------------- rows.json ---------------------------------

rows = {
    "schema_version": SCHEMA,
    "kind": "rows",
    "rows": [
        {
            "row_id": rid,
            "row_index": i + 1,
            "doc_id": f"synthetic:corpus:{i + 1}",
            "content_family_id": f"cf_fixture_{i + 1}",
            "n_raw_tokens": 32 + i,
            "token_position": 8 + i,
            "activation_norm": 100.0 + i,
            "source_text": (
                f"Synthetic source text {i + 1}: a fixture sentence about weather patterns."
                " tok3 tok4 tok5"
            ),
            "teacher_text": f"Synthetic teacher encoding {i + 1} of a stored activation.",
            "av_text": f"Synthetic learned description {i + 1} of a stored activation.",
            "release_status": "synthetic_fixture",
            "claim_scope": "stored_snapshot",
            "stratum": {"bucket": "fixture", "index": i + 1},
        }
        for i, rid in enumerate(ROW_IDS)
    ],
}
write("rows.json", rows)

# -------------------------------- channel.json --------------------------------


def aggregate(mean: float) -> dict:
    return {
        "mean": mean,
        "ci_low": round(mean - 0.02, 6),
        "ci_high": round(mean + 0.02, 6),
        "rows": 2,
        "families": 2,
        "bootstrap_samples": 100,
    }


def waterfall_variant(dmse: float, cosine: float) -> dict:
    return {
        "dmse": dmse,
        "cosine_mean": cosine,
        "ci_low": round(cosine - 0.02, 6),
        "ci_high": round(cosine + 0.02, 6),
        "rows": 2,
        "families": 2,
        "norm_ratio_mean": 1.2,
    }


channel = {
    "schema_version": SCHEMA,
    "kind": "channel",
    "matched_online_rl": {
        "status": "validation_only_matched",
        "row_count": 2,
        "independent_family_count": 2,
        "max_new_tokens": 384,
        "generation_protocol_sha256": sha("matched-protocol"),
        "sft": {
            "roundtrip_nmse": 0.4,
            "raw_mse": 10.0,
            "cosine": 0.8,
            "centered_r2": 0.2,
            "norm_ratio": 1.1,
            "teacher_nmse": 0.35,
            "teacher_win_count": 1,
            "teacher_win_fraction": 0.5,
            "parse": {"closed_count": 2, "closed_fraction": 1.0, "usable_count": 2, "usable_fraction": 1.0, "row_count": 2},
            "controls": [],
        },
        "rl": {
            "roundtrip_nmse": 0.3,
            "raw_mse": 7.5,
            "cosine": 0.85,
            "centered_r2": 0.35,
            "norm_ratio": 1.05,
            "teacher_nmse": 0.34,
            "teacher_win_count": 2,
            "teacher_win_fraction": 1.0,
            "parse": {"closed_count": 1, "closed_fraction": 0.5, "usable_count": 2, "usable_fraction": 1.0, "row_count": 2},
            "controls": [
                {"key": "av_shuffled", "label": "shuffled activation", "roundtrip_nmse": 1.0},
                {"key": "av_zero", "label": "zero activation", "roundtrip_nmse": 0.95},
                {"key": "av_mean", "label": "mean activation", "roundtrip_nmse": 0.8},
                {"key": "av_none", "label": "no activation", "roundtrip_nmse": 0.98},
            ],
        },
        "improvement": {
            "nmse_absolute": 0.1,
            "nmse_relative": 0.25,
            "raw_mse_absolute": 2.5,
            "raw_mse_relative": 0.25,
            "teacher_win_fraction_gain": 0.5,
        },
        "source_reports": {"sft": "roundtrip_sft_384", "rl": "roundtrip_rl_384"},
        "scope_note": "Synthetic matched validation evidence. Joint AV+AR effect; not a sealed test result.",
    },
    "aggregates": {
        "primary.identity.directional_mse": aggregate(0.31),
        "primary.identity.cosine": aggregate(0.84),
        "primary.paraphrase.directional_mse": aggregate(0.36),
        "primary.corruption.cosine": aggregate(0.71),
        "independent.identity.directional_mse": aggregate(0.32),
    },
    "court_thresholds": {
        "primary": {
            "threshold": 0.981,
            "balanced_accuracy": 0.95,
            "positive_recall": 0.9,
            "negative_recall": 1.0,
        },
        "independent": {
            "threshold": 0.984,
            "balanced_accuracy": 0.94,
            "positive_recall": 0.88,
            "negative_recall": 1.0,
        },
    },
    "fit_split": "validation",
    "identity": [
        {"row_id": rid, "critic": critic, **slim_metric(i, critic)}
        for i, rid in enumerate(ROW_IDS)
        for critic in CRITICS
    ],
    "twin_critics": {
        "per_row": [
            {
                "row_id": rid,
                "primary_dmse": round(0.12 + 0.02 * i, 6),
                "independent_dmse": round(0.13 + 0.02 * i, 6),
                "primary_cosine": round(0.94 - 0.01 * i, 6),
                "independent_cosine": round(0.93 - 0.01 * i, 6),
            }
            for i, rid in enumerate(ROW_IDS)
        ],
        "e3_summaries": {
            "primary": {"cells": 4, "mean_directional_mse": 0.38, "mean_cosine": 0.81},
            "independent": {"cells": 4, "mean_directional_mse": 0.36, "mean_cosine": 0.82},
        },
        "p2_summaries": {
            "primary": {"cells": 8, "mean_directional_mse": 0.35, "mean_cosine": 0.82},
            "independent": {"cells": 8, "mean_directional_mse": 0.34, "mean_cosine": 0.83},
        },
        "confound": "Synthetic fixture: twin critics share a teacher target and are not semantically independent.",
    },
    "retrieval": [
        {
            "row_id": rid,
            "critic": critic,
            "rank": 1,
            "nearest_row_id": rid,
            "expected_cosine": round(0.93 - 0.01 * i, 6),
        }
        for i, rid in enumerate(ROW_IDS)
        for critic in CRITICS
    ],
    "waterfall": {
        "metric": "directional_mse_equals_2_times_one_minus_cosine",
        "split": "validation",
        "variants": {
            "teacher": waterfall_variant(0.30, 0.85),
            "av_real": waterfall_variant(0.42, 0.79),
            "av_shuffled": waterfall_variant(0.83, 0.58),
        },
        "source_report": "e0",
    },
    "capacity_ladder": {
        "ladder": [
            {
                "gallery_size": 2,
                "gallery_bits": 1.0,
                "top1_accuracy": 0.99,
                "top5_accuracy": 1.0,
                "median_rank": 1.0,
                "mean_reciprocal_rank": 0.995,
                "fano_information_lower_bound_bits": 0.92,
            },
            {
                "gallery_size": 4,
                "gallery_bits": 2.0,
                "top1_accuracy": 0.95,
                "top5_accuracy": 1.0,
                "median_rank": 1.0,
                "mean_reciprocal_rank": 0.97,
                "fano_information_lower_bound_bits": 1.71,
            },
        ],
        "assumptions": {
            "decoder": "nearest stored target by cosine",
            "gallery_prior": "uniform by construction",
        },
        "top_confusions": [
            {"count": 1, "retrieved_family": "cf_fixture_2", "source_family": "cf_fixture_1"}
        ],
        "distance": "cosine",
        "variant": "av_real",
        "source_report": "e0",
    },
    "real_vs_control": {
        "e1_av": {
            "losses": {"real": 0.74, "shuffled": 1.29, "zero": 1.14, "none": 1.16, "mean": 1.22},
            "rows": 2,
            "parse": {
                v: {"closed_fraction": 1.0, "usable_fraction": 1.0}
                for v in ["real", "shuffled", "zero", "none", "mean"]
            },
            "source_report": "e1_av",
        },
        "e2": {
            "mean_loss": {"real": 0.80, "shuffled": 1.30, "zero": 1.18, "none": 1.23, "mean": 1.25},
            "paired": {
                v: {"mean_real_minus_control": round(-0.4 - 0.02 * k, 6), "real_win_fraction": 1.0}
                for k, v in enumerate(["shuffled", "zero", "none", "mean"])
            },
            "rows": 2,
            "records": 8,
            "source_report": "e2",
        },
    },
    "truncation": {
        rid: [
            {"fraction": 0.25, "words": 4, "dmse": round(0.70 - 0.02 * i, 6), "cosine": round(0.65 + 0.01 * i, 6)},
            {"fraction": 0.5, "words": 8, "dmse": round(0.45 - 0.02 * i, 6), "cosine": round(0.77 + 0.01 * i, 6)},
            {"fraction": 1.0, "words": 16, "dmse": round(0.14 - 0.02 * i, 6), "cosine": round(0.93 + 0.01 * i, 6)},
        ]
        for i, rid in enumerate(ROW_IDS)
    },
    "occlusion": {
        rid: [
            {
                "word_index": 0,
                "word": "Synthetic",
                "char_start": 0,
                "char_end": 9,
                "dmse": 0.15,
                "d_dmse": 0.004,
            },
            {
                "word_index": 1,
                "word": "fixture",
                "char_start": 10,
                "char_end": 17,
                "dmse": 0.18,
                "d_dmse": 0.034,
            },
        ]
        for rid in ROW_IDS
    },
    "tellings": {
        rid: [
            {
                "cell_id": f"cell-telling-{i + 1}-{k}",
                "sample_index": k,
                "text": f"Synthetic alternate telling {k} for fixture row {i + 1}.",
                "dmse": round(0.20 + 0.03 * k, 6),
                "cosine": round(0.90 - 0.02 * k, 6),
            }
            for k in range(2)
        ]
        for i, rid in enumerate(ROW_IDS)
    },
    "shapley": {
        rid: {
            "sections": {"syntax": 0.28, "discourse": 0.21, "register": 0.05, "final_token": 0.19},
            "efficiency_error": 1e-09,
        }
        for rid in ROW_IDS
    },
}
write("channel.json", channel)

# -------------------------------- rewrites.json -------------------------------

REWRITE_CELLS = [
    ("cell-rw-1", "validation-1", "paraphrase", "shuffled_sections", {"kind": "shuffle_sections"}, "positive"),
    ("cell-rw-2", "validation-2", "paraphrase", "formal_tone", {"kind": "reword", "semantic_intent": "preserve"}, "positive"),
    ("cell-rw-3", "validation-1", "corruption", "delete_0.25", {"kind": "delete", "rate": 0.25}, "context"),
    ("cell-rw-4", "validation-2", "corruption", "shuffle_0.5", {"kind": "shuffle", "rate": 0.5}, "negative"),
]

rewrites = {
    "schema_version": SCHEMA,
    "kind": "rewrites",
    "identity": [
        {
            "row_id": rid,
            "metrics": {critic: slim_metric(i, critic) for critic in CRITICS},
            "text": f"Synthetic learned description {i + 1} of a stored activation.",
        }
        for i, rid in enumerate(ROW_IDS)
    ],
    "cells": [
        {
            "cell_id": cell_id,
            "row_id": rid,
            "family": family,
            "variant": variant,
            "text": f"Synthetic {family} rewrite ({variant}) of the fixture description.",
            "spec": spec,
            "metrics": {critic: slim_metric(j + 2, critic) for critic in CRITICS},
            "court": {
                critic: {
                    "identity_cosine": round(0.99 - 0.01 * j, 6),
                    "calibration_label": label,
                    "semanticity_verdict": label != "negative",
                }
                for critic in CRITICS
            },
        }
        for j, (cell_id, rid, family, variant, spec, label) in enumerate(REWRITE_CELLS)
    ],
}
write("rewrites.json", rewrites)

# --------------------------------- trace.json ---------------------------------

trace = {
    "schema_version": SCHEMA,
    "kind": "trace",
    "claim_scope": "fresh_forward_exploratory",
    "boundary": 3,
    "shuffled_control": {
        "available": False,
        "note": "Synthetic fixture: no shuffled-position trace control ships in this bundle.",
    },
    "source_alignment": {
        "exact_positions": 6,
        "unavailable_positions": 0,
        "note": "Synthetic fixture contexts align exactly.",
    },
    "docs": [
        {
            "row_id": rid,
            "doc_id": f"synthetic:corpus:{i + 1}",
            "content_family_id": f"cf_fixture_{i + 1}",
            "positions": [
                {
                    "position": p,
                    "n_context_tokens": p + 1,
                    "token_id": 100 + p,
                    "token_text": f" tok{p}",
                    "source_alignment": "exact",
                    "source_char_start": 67 + 5 * (p - 3),
                    "source_char_end": 72 + 5 * (p - 3),
                    "source_before": f"Synthetic source context before position {p}",
                    "source_token": f" tok{p}",
                    "source_after": " with later text shown only for orientation.",
                    "source_prefix_omitted": p > 3,
                    "source_suffix_omitted": True,
                    "description": f"Synthetic learned description of the stored activation at position {p} of fixture doc {i + 1}.",
                    "parse_state": "usable_closed",
                    "usable": True,
                }
                for p in (3, 4, 5)
            ],
            "drift": {
                "one_minus_cos": round(0.002 + 0.001 * i, 6),
                "relative_l2": round(0.05 + 0.01 * i, 6),
                "rms_ratio": 1.004,
                "max_abs": 0.9,
            },
        }
        for i, rid in enumerate(ROW_IDS)
    ],
}
write("trace.json", trace)

# --------------------------------- poetry.json --------------------------------

POETRY_CFG = sha("poetry-config")


def poetry_samples(case_id: str, usable_all: bool) -> list[dict]:
    return [
        {
            "position": 4 + k,
            "relative_offset": -2 + k,
            "variant": "real" if k == 0 else "shuffled",
            "sample_index": k,
            "source_case_id": case_id,
            "usable": True if usable_all else k == 0,
            "target_exact": False,
            "target_family": k == 0,
            "alternate_family": False,
            "explanation": f"Synthetic encoding sample {k} for case {case_id}.",
            "parse": {"closed": True, "extraction_mode": "strict", "repetition_loop": False},
        }
        for k in range(2)
    ]


def poetry_position_scores() -> list[dict]:
    scores = []
    for p in (4, 5, 6):
        for variant in ("real", "shuffled"):
            scores.append(
                {
                    "position": p,
                    "relative_offset": p - 6,
                    "variant": variant,
                    "samples": 2,
                    "usable_rate": 1.0,
                    "target_exact_rate": 0.0,
                    "target_family_rate": 0.5 if variant == "real" else 0.25,
                    "alternate_family_rate": 0.0,
                }
            )
    return scores


case_alpha = {
    "case_id": "fixture-alpha",
    "framing": "Synthetic couplet prompt (fixture).",
    "first_line": "A fixture lantern starts to glow,",
    "second_line": "and casts a light on rows below.",
    "prefix_text": "A fixture lantern starts to glow,\nand casts a light on rows",
    "cue": "glow",
    "target_word": "glow",
    "target_terms": ["glow", "flow"],
    "alternate_terms": ["night", "light"],
    "edit_map": {"glow": "night", "flow": "light"},
    "anchor_position": 6,
    "analysis": [
        {"position": p, "relative_offset": p - 6, "token_id": 200 + p, "token_text": f" w{p}"}
        for p in (4, 5, 6)
    ],
    "baseline_continuation": "below the fixture rows they flow.",
    "baseline_hits_target_family": True,
    "planning_onset_position": 4,
    "planning_onset_relative_offset": -2,
    "anchor_lift": 0.25,
    "anchor_real_target_family_rate": 0.5,
    "anchor_shuffled_target_family_rate": 0.25,
    "position_scores": poetry_position_scores(),
    "samples": poetry_samples("fixture-alpha", True),
    "reconstruction": {
        "original_explanation": "Synthetic original encoding mentioning glow.",
        "edited_explanation": "Synthetic edited encoding mentioning night.",
        "changed_terms": ["glow", "night"],
        "original_cosine": 0.9,
        "original_dmse": 0.4,
        "edit_delta_norm": 12.5,
    },
    "interventions": [
        {
            "direction": direction,
            "dose": dose,
            "continuation_text": f"Synthetic continuation ({direction}, dose {dose}).",
            "hits_target_family": direction == "edited" and dose == 0.5,
            "hits_alternate_family": False,
        }
        for direction in ("edited", "random")
        for dose in (0.5, 1.0)
    ],
}

case_beta = {
    "case_id": "fixture-beta",
    "framing": "Synthetic couplet prompt without onset (fixture).",
    "first_line": "A fixture window frames the rain,",
    "second_line": "and hums beside a passing train.",
    "prefix_text": "A fixture window frames the rain,\nand hums beside a passing",
    "cue": "rain",
    "target_word": "train",
    "target_terms": ["train", "rain"],
    "alternate_terms": ["sea", "tree"],
    "edit_map": {},
    "anchor_position": 6,
    "analysis": [
        {"position": p, "relative_offset": p - 6, "token_id": 300 + p, "token_text": f" v{p}"}
        for p in (4, 5, 6)
    ],
    "baseline_continuation": "a passing cart on the old lane.",
    "baseline_hits_target_family": False,
    "planning_onset_position": None,
    "planning_onset_relative_offset": None,
    "anchor_lift": 0.0,
    "anchor_real_target_family_rate": 0.25,
    "anchor_shuffled_target_family_rate": 0.25,
    "position_scores": poetry_position_scores(),
    "samples": poetry_samples("fixture-beta", False),
    "reconstruction": None,
    "interventions": [],
}

poetry = {
    "schema_version": SCHEMA,
    "kind": "poetry",
    "claim_scope": "fresh_forward_exploratory",
    "config_sha256": POETRY_CFG,
    "gates": {"planning_onset_rate": 0.25, "minimum_usable_fraction": 0.8},
    "aggregates": {
        "cases": 2,
        "positions": 6,
        "samples": 4,
        "usable_fraction": 0.75,
        "mean_anchor_lift": 0.125,
        "cases_with_planning_onset": 1,
        "cases_with_baseline_target_rhyme": 1,
        "editable_cases": 1,
        "steering_doses": [0.5, 1.0],
        "edited_alternate_hit_rate": 0.0,
        "random_alternate_hit_rate": 0.0,
        "mean_original_dmse": 0.4,
    },
    "interpretation": {
        "signal": "weak",
        "notes": [
            "Synthetic fixture: anchor lift is positive for 1 of 2 cases.",
            "Synthetic fixture: no causal edit produced an alternate-family hit at any dose.",
        ],
    },
    "reports": {
        phase: {"passed": True, "config_sha256": POETRY_CFG}
        for phase in [
            "poetry_prepare",
            "poetry_extract",
            "poetry_describe",
            "poetry_score",
            "poetry_reconstruct",
            "poetry_intervene",
        ]
    },
    "cases": [case_alpha, case_beta],
}
write("poetry.json", poetry)

# ------------------------------- bench shards ---------------------------------

LANES = ["edit", "paraphrase_placebo", "random_edit"]
LANE_CELL_IDS = {"edit": "cell-b-edit", "paraphrase_placebo": "cell-b-para", "random_edit": "cell-b-rand"}

bench_index = {
    "schema_version": SCHEMA,
    "kind": "bench_index",
    "banner": {
        "statement": "Synthetic fixture: you are choosing among 4 precomputed experiments, not editing a live model.",
        "total_cells": 4,
        "behavior_cells": 4,
        "behavior_rows": 1,
        "grid_spec_sha256": sha("grid-spec"),
        "claim_scope": "stored_snapshot",
        "functional_claim_status": "validation_only_exploratory",
    },
    "rows": [
        {
            "row_id": "validation-1",
            "has_behavior": True,
            "families": {
                "identity": {"variants": ["teacher"], "depths": ["BEHAVIOR"]},
                "clause_swap": {
                    "variants": [f"syntax:{lane}:a1" for lane in LANES],
                    "depths": ["BEHAVIOR"],
                },
            },
        },
        {
            "row_id": "validation-2",
            "has_behavior": False,
            "families": {"identity": {"variants": ["teacher"], "depths": ["METRIC"]}},
        },
    ],
    "behavior_rows": ["validation-1"],
    "control_groups": {
        "cg-1": {
            "row_id": "validation-1",
            "chip": "syntax",
            "cells": {lane: {"1": LANE_CELL_IDS[lane]} for lane in LANES},
        }
    },
}
write("bench/index.json", bench_index)


def topk(base_id: int) -> list[dict]:
    return [
        {"id": base_id + k, "text": f" tk{base_id + k}", "p": round(0.6 - 0.2 * k, 6)}
        for k in range(3)
    ]


def wake_points(seed: int) -> list[dict]:
    return [
        {
            "offset": k + 1,
            "js": round(0.5 / (k + 1) + 0.01 * seed, 6),
            "kl": round(2.0 / (k + 1) + 0.01 * seed, 6),
            "top_10_overlap": round(min(1.0, 0.4 + 0.15 * k), 6),
            "top_50_overlap": round(min(1.0, 0.5 + 0.1 * k), 6),
        }
        for k in range(4)
    ]


def behavior_block(seed: int) -> dict:
    return {
        "js_divergence": round(0.3 + 0.05 * seed, 6),
        "kl_original_to_patched": round(1.5 + 0.2 * seed, 6),
        "logit_pearson": round(0.9 - 0.03 * seed, 6),
        "top_10_overlap": round(0.7 - 0.05 * seed, 6),
        "top_50_overlap": round(0.8 - 0.05 * seed, 6),
        "original_top1_rank": 1 + seed,
        "vocab_size": 1024,
    }


GEN_PROTOCOL = {
    "backend": "full_prefix",
    "boundary_replacement": "each_full_prefix_forward",
    "do_sample": False,
    "eos_stopping": False,
    "max_new_tokens": 8,
    "use_cache": True,
}

bench_cells = [
    {
        "cell_id": "cell-b-teacher",
        "family": "identity",
        "variant": "teacher",
        "depth": "BEHAVIOR",
        "control_group_id": None,
        "spec": {"kind": "teacher"},
        "text": "Synthetic teacher encoding 1 of a stored activation.",
        "metrics": {critic: slim_metric(0, critic) for critic in CRITICS},
        "geometry": {"x": 9.5, "y": -4.5, "z": 0.9},
        "behavior": behavior_block(0),
        "topk": {"original": topk(100), "patched": topk(110)},
        "wake": wake_points(0),
        "baseline_continuation": "Synthetic baseline continuation for the teacher cell.",
        "patched_continuation": "Synthetic patched continuation for the teacher cell.",
        "generation_protocol": GEN_PROTOCOL,
    }
]
for s, lane in enumerate(LANES, start=1):
    bench_cells.append(
        {
            "cell_id": LANE_CELL_IDS[lane],
            "family": "clause_swap",
            "variant": f"syntax:{lane}:a1",
            "depth": "BEHAVIOR",
            "control_group_id": "cg-1",
            "spec": {"chip": "syntax", "lane": lane, "dose": 1.0},
            "text": f"Synthetic clause-swap text for the {lane} lane.",
            "metrics": {"primary": slim_metric(s, "primary")},
            "geometry": {"x": 9.0 + s, "y": -4.0 - s, "z": 0.5 * s},
            "behavior": behavior_block(s),
            "topk": {"original": topk(100 + 10 * s), "patched": topk(105 + 10 * s)},
            "wake": wake_points(s),
            "baseline_continuation": f"Synthetic baseline continuation for the {lane} lane.",
            "patched_continuation": f"Synthetic patched continuation for the {lane} lane.",
            "generation_protocol": GEN_PROTOCOL,
        }
    )

bench_row = {
    "schema_version": SCHEMA,
    "kind": "bench_row",
    "row_id": "validation-1",
    "claim_scope": "stored_snapshot",
    "target_geometry": {"x": 10.0, "y": -5.0, "z": 1.0},
    "cells": bench_cells,
}
write("bench/row-validation-1.json", bench_row)

# --------------------------------- audit.json ---------------------------------

FIXTURE_FILES = [
    "audit.json",
    "bench/index.json",
    "bench/row-validation-1.json",
    "channel.json",
    "poetry.json",
    "rewrites.json",
    "rows.json",
    "trace.json",
]

manifest_files = [
    {"path": p, "sha256": sha(f"file:{p}"), "bytes": 1000 + 7 * i, "schema_version": SCHEMA}
    for i, p in enumerate(FIXTURE_FILES)
]

tokenizer_block = {
    "path": "fixtures/tokenizer.json",
    "sha256": sha("tokenizer"),
    "vocab_size": 1024,
    "spot_check": {"tokens": 8, "mismatches": 0},
}

source_counts = {"rows": 2, "interventions": 8, "behavior": 4, "trajectories": 6, "vectors": 12}

audit = {
    "schema_version": SCHEMA,
    "kind": "audit",
    "claim_ledger": {
        "claims": {
            "stored_snapshot_channel": "Synthetic fixture claim: bounded stored-snapshot channel claim on the 2-row panel.",
            "matched_online_rl_roundtrip": "validation_only_confirmatory",
            "fresh_forward_trace": "Synthetic fixture claim: fresh-forward traces are exploratory.",
            "functional_interventions": "Synthetic fixture claim: interventions are precomputed, validation-only exploratory.",
            "test_set": "Synthetic fixture claim: no test-set exposure in this fixture.",
        },
        "limitations": [
            "Synthetic fixture limitation: two rows only.",
            "Synthetic fixture limitation: all numbers are made up.",
        ],
    },
    "evidence_status_legend": {
        "qualified": "Bounded stored-snapshot channel claim on the fixture panel.",
        "exploratory": "Fresh-forward or validation-fitted views outside the qualified claim.",
        "negative": "Outcomes where the measured effect is absent or contradicts the hypothesis.",
        "unavailable": "Evidence the bundle does not contain; absence is shown, never silent.",
    },
    "provenance": {
        "bundle_id": sha("bundle"),
        "source_config_sha256": sha("source-config"),
        "bundle_config_sha256": sha("bundle-config"),
        "population": "QUALIFIED",
        "split": "validation",
        "counts": source_counts,
        "files": manifest_files,
        "excluded_files": ["vectors/all.f16.bin"],
        "report_bindings": {
            "e0": {"source_path": "/synthetic/model_outputs/e0/report.json", "sha256": sha("report:e0")},
            "e2": {"source_path": "/synthetic/model_outputs/e2/report.json", "sha256": sha("report:e2")},
        },
        "code_bindings": {"configs/fixture.yaml": sha("code:fixture.yaml")},
        "runtime": {"python": "3.12.0", "numpy": "2.0.0", "torch": "2.0.0"},
        "privacy_card": {
            "automatic_gate_passed": True,
            "human_review_required": True,
            "claim_boundary": "Synthetic fixture: automatic triage only.",
        },
        "source_provenance": {
            "primary_cache_sha256": sha("cache:primary"),
            "independent_cache_sha256": sha("cache:independent"),
        },
        "tokenizer": tokenizer_block,
    },
    "court": {
        "thresholds": channel["court_thresholds"],
        "fit_split": "validation",
        "confound": "Synthetic fixture: court thresholds are fit on the same split they judge.",
        "docket": [
            {
                "row_id": "validation-1",
                "critic": critic,
                "paraphrase_min_identity_cosine": 0.985,
                "paraphrase_mean_identity_cosine": 0.99,
                "paraphrase_verdicts_true": 2,
                "paraphrase_cells": [
                    {"variant": "shuffled_sections", "identity_cosine": 0.995, "label": "positive", "verdict": True},
                    {"variant": "formal_tone", "identity_cosine": 0.985, "label": "positive", "verdict": True},
                ],
                "corruption_cells": [
                    {"variant": "delete_0.25", "identity_cosine": 0.97, "label": "context", "verdict": False},
                    {"variant": "shuffle_0.5", "identity_cosine": 0.92, "label": "negative", "verdict": False},
                ],
                "row_verdict": "honest" if critic == "primary" else "mixed",
            }
            for critic in CRITICS
        ],
    },
    "parse_health": {
        "explanations_by_kind": {
            "qualified_av": {"usable_closed": 4},
            "alternate_telling": {"usable_closed": 4},
        },
        "trace_descriptions_usable": True,
        "e1_av_parse": {
            v: {"closed_fraction": 1.0, "usable_fraction": 1.0}
            for v in ["real", "shuffled", "zero", "none", "mean"]
        },
        "almanac_parse_health": {
            v: {"closed_fraction": 1.0, "usable_fraction": 1.0} for v in ["real", "zero", "none"]
        },
        "poetry_usable_fraction": 0.75,
    },
    "drift": {
        "card": {
            "activation_fidelity": {
                "cosine_agreement_mean": 0.999,
                "cosine_agreement_min": 0.98,
                "relative_l2_max": 0.07,
            }
        },
        "e5_per_doc": [
            {"row_id": rid, "one_minus_cos": round(0.002 + 0.001 * i, 6), "relative_l2": round(0.05 + 0.01 * i, 6)}
            for i, rid in enumerate(ROW_IDS)
        ],
    },
    "magnitude": {
        "claim_boundary": "Synthetic fixture: post-hoc magnitude calibration fit on validation only.",
        "publication_status": "internal_only",
        "fit": {"candidate_metrics": {"identity": {"centered_r2": -0.29, "raw_mse": 9.2}}},
    },
    "null_text": {
        "scope": "Synthetic fixture: real-vs-null enrichment on 2 rows.",
        "row_count": 2,
        "real_enriched_words": [
            {"token": "fixture", "log_odds_real_vs_zero": 5.3},
            {"token": "synthetic", "log_odds_real_vs_zero": 4.1},
        ],
        "zero_enriched_words": [
            {"token": "empty", "log_odds_real_vs_zero": -4.8},
            {"token": "null", "log_odds_real_vs_zero": -3.9},
        ],
        "e1_av_losses": channel["real_vs_control"]["e1_av"]["losses"],
        "e2_mean_loss": channel["real_vs_control"]["e2"]["mean_loss"],
        "e2_paired": channel["real_vs_control"]["e2"]["paired"],
        "backfill_note": "Synthetic fixture backfill note.",
    },
    "negative_results": [
        {
            "id": "poetry_anchor_lift",
            "status": "weak",
            "statement": "Synthetic fixture: mean anchor lift is 0.125 across 2 cases; 1 of 2 shows no lift.",
            "source": "data/poetry.json",
        },
        {
            "id": "fixture_steering_null",
            "status": "negative",
            "statement": "Synthetic fixture: no causal edit produced an alternate-family hit at any dose.",
            "source": "data/poetry.json",
        },
    ],
    "poetry_status": {
        "claim_scope": "fresh_forward_exploratory",
        "pipeline_passed": True,
        "pipeline_note": "Pipeline `passed` means phase completeness gates only; it is not evidence that a scientific planning hypothesis passed.",
        "config_sha256": POETRY_CFG,
    },
}
write("audit.json", audit)

# -------------------------------- manifest.json -------------------------------

manifest = {
    "schema_version": SCHEMA,
    "generated_at": "2026-01-01T00:00:00Z",
    "source": {
        "bundle_id": sha("bundle"),
        "source_config_sha256": sha("source-config"),
        "bundle_config_sha256": sha("bundle-config"),
        "population": "QUALIFIED",
        "split": "validation",
        "counts": source_counts,
        "manifest_sha256": sha("source-manifest"),
        "excluded_files": ["vectors/all.f16.bin"],
    },
    "poetry": {
        "config_sha256": POETRY_CFG,
        "phases_passed": [
            "poetry_prepare",
            "poetry_extract",
            "poetry_describe",
            "poetry_score",
            "poetry_reconstruct",
            "poetry_intervene",
        ],
    },
    "online_rl": {
        "status": "validation_only_matched",
        "row_count": 2,
        "independent_family_count": 2,
        "max_new_tokens": 384,
        "generation_protocol_sha256": sha("matched-protocol"),
        "report_sha256": {"sft": sha("matched-sft"), "rl": sha("matched-rl")},
    },
    "tokenizer": tokenizer_block,
    "counts": {
        "rows": 2,
        "interventions": 8,
        "behavior": 4,
        "court": 8,
        "explanations": 4,
        "geometry": 6,
        "metrics": 8,
        "poetry_cases": 2,
        "poetry_interventions": 4,
        "poetry_positions": 6,
        "poetry_samples": 4,
        "retrieval": 4,
        "shapley": 2,
        "trajectories": 6,
    },
    "files": manifest_files,
}
write("manifest.json", manifest)

print("fixture set complete")
