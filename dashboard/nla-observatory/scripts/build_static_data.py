#!/usr/bin/env python3
"""Deterministic static-data builder for the NLA Offline Observatory dashboard.

This is the ONLY layer allowed to read the raw parquet/JSONL evidence. It:

1. verifies the core observatory bundle manifest hashes before reading tables;
2. verifies the poetry planning reports and their shared config hash;
3. validates expected row counts, enumerations, and referential joins;
4. strips activation vectors and unneeded token arrays;
5. emits deterministic, versioned, lazy-loadable browser shards under
   public/data/;
6. emits dashboard_manifest.json binding source hashes, generated hashes,
   schema versions, counts, and the build timestamp;
7. fails closed on missing rows, duplicate ids, unknown variants, broken
   references, or non-finite metrics.

Shard bytes are canonical JSON (sorted keys, compact separators, ASCII) so a
rebuild from identical evidence produces identical bytes (the manifest's
generated_at field is the single intentional exception, and it lives only in
the manifest).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SCHEMA_VERSION = "nla_observatory_dashboard.v2"

# Word tokenization must match observatory/build_corpus.py exactly.
WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)

CRITICS = {"primary", "independent"}
FAMILIES = {
    "identity", "clause_swap", "section_ablation", "word_occlusion",
    "truncation", "paraphrase", "corruption", "alternate_telling",
}
LANES = {"edit", "paraphrase_placebo", "random_edit"}
EXPLANATION_KINDS = {"teacher", "qualified_av", "alternate_telling", "trace_description"}
PARSE_STATES = {"usable_closed", "usable_open"}
CALIBRATION_LABELS = {"positive", "negative", "context"}
COURT_FAMILIES = {"paraphrase", "corruption"}
SECTIONS = {0: "syntax", 1: "discourse", 2: "register", 3: "final_token"}

# Heavy files intentionally not copied from the RunAI workspace (the plan
# forbids shipping activation vectors to the browser bundle).
EXCLUDED_BUNDLE_FILES = {"vectors/all.f16.bin", "geometry_basis.npz"}

POETRY_PHASES = [
    "poetry_prepare", "poetry_extract", "poetry_describe",
    "poetry_score", "poetry_reconstruct", "poetry_intervene",
]

BUNDLE_REPORT_NAMES = [
    "e0", "e1_av", "e1_ar", "e2", "e3", "e4", "e5",
    "p1_tellings", "p1_trace", "p2", "p3", "geometry", "interventions",
]

MATCHED_RL_EVAL_REL = Path(
    "artifacts/runai_eval/r33_u342_research_abc_v1_20260722/"
    "eval384_matched_v122"
)
MATCHED_RL_REPORTS = {
    "sft": "sft_roundtrip_report.json",
    "rl": "rl_roundtrip_report.json",
}
MATCHED_RL_PROTOCOL_SHA256 = (
    "fcc431ec4450adb8817cd946d6c194fa2a45b53b0c6c42c8682c1e9f12f94d4d"
)
MATCHED_RL_CONTROLS = {"real", "shuffled", "zero", "mean", "none"}


class BuildError(RuntimeError):
    """Fail-closed build failure."""


def fail(msg: str) -> None:
    raise BuildError(msg)


def check(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_int(seed: int, *parts) -> int:
    payload = canonical_json([int(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def finite(value, ctx: str) -> float:
    v = float(value)
    check(math.isfinite(v), f"non-finite metric in {ctx}: {value!r}")
    return v


def round6(value: float) -> float:
    """Stable float formatting for shard determinism and size."""
    return float(f"{value:.6g}")


def read_jsonl(path: Path) -> list[dict]:
    check(path.is_file(), f"missing JSONL file: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_json(path: Path) -> dict:
    check(path.is_file(), f"missing JSON file: {path}")
    return json.loads(path.read_text())


def load_matched_online_rl(repo_root: Path) -> tuple[dict, dict[str, Path]]:
    """Validate and summarize the matched 384-token SFT-vs-RL evaluation."""
    eval_dir = repo_root / MATCHED_RL_EVAL_REL
    paths = {name: eval_dir / filename for name, filename in MATCHED_RL_REPORTS.items()}
    reports = {name: load_json(path) for name, path in paths.items()}

    for name, report in reports.items():
        check(report.get("schema_version") == "nano_av_ar_roundtrip_gate.v1",
              f"matched {name} report schema mismatch")
        check(report.get("generation_protocol_sha256") == MATCHED_RL_PROTOCOL_SHA256,
              f"matched {name} generation protocol mismatch")
        check(int(report.get("max_new_tokens", -1)) == 384,
              f"matched {name} generation budget is not 384")
        check(set(report.get("generation_controls", [])) == MATCHED_RL_CONTROLS,
              f"matched {name} controls are incomplete")
        check(report.get("gate", {}).get("passed") is True,
              f"matched {name} gate did not pass")
        check(report.get("gate", {}).get("current_generation_protocol_compatible") is True,
              f"matched {name} report is not generation-protocol compatible")

        split = report.get("splits", {}).get("validation", {})
        check(split.get("row_count") == 122, f"matched {name} row count is not 122")
        check(split.get("independent_family_count") == 122,
              f"matched {name} family count is not 122")
        check(len(split.get("row_keys", [])) == 122,
              f"matched {name} row keys are incomplete")
        variants = split.get("variants", {})
        required = {"av_real", "av_shuffled", "av_zero", "av_mean", "av_none", "teacher"}
        check(required.issubset(variants), f"matched {name} variants are incomplete")
        parse = split.get("generation_parse", {}).get("real", {})
        check(parse.get("row_count") == 122, f"matched {name} parse rows are incomplete")

    check(reports["sft"]["splits"]["validation"]["row_keys"] ==
          reports["rl"]["splits"]["validation"]["row_keys"],
          "matched SFT and RL reports use different rows")

    control_specs = [
        ("av_shuffled", "Shuffled description"),
        ("av_zero", "Zero activation"),
        ("av_mean", "Mean activation"),
        ("av_none", "No activation injection"),
    ]

    def stage(report: dict) -> dict:
        split = report["splits"]["validation"]
        variants = split["variants"]
        real = variants["av_real"]
        parse = split["generation_parse"]["real"]
        teacher_win = split["rowwise_win_rates"]["av_real_vs_teacher"]
        return {
            "roundtrip_nmse": round6(finite(real["normalized_mse"], "matched real nmse")),
            "raw_mse": round6(finite(real["raw_mse"], "matched real raw mse")),
            "cosine": round6(finite(real["cosine_mean"], "matched real cosine")),
            "centered_r2": round6(finite(real["centered_raw_r2"], "matched real r2")),
            "norm_ratio": round6(finite(real["norm_ratio_mean"], "matched norm ratio")),
            "teacher_nmse": round6(finite(variants["teacher"]["normalized_mse"],
                                                  "matched teacher nmse")),
            "teacher_win_count": int(teacher_win["candidate_better_count"]),
            "teacher_win_fraction": round6(finite(
                teacher_win["candidate_better_fraction"], "matched teacher win fraction")),
            "parse": {
                "closed_count": int(parse["closed_count"]),
                "closed_fraction": round6(finite(parse["closed_fraction"],
                                                    "matched close fraction")),
                "usable_count": int(parse["usable_count"]),
                "usable_fraction": round6(finite(parse["usable_fraction"],
                                                    "matched usable fraction")),
                "row_count": int(parse["row_count"]),
            },
            "controls": [
                {
                    "key": key,
                    "label": label,
                    "roundtrip_nmse": round6(finite(
                        variants[key]["normalized_mse"], f"matched control {key}")),
                }
                for key, label in control_specs
            ],
        }

    sft = stage(reports["sft"])
    rl = stage(reports["rl"])
    check(abs(sft["roundtrip_nmse"] - 0.309055) < 1e-6,
          "matched SFT headline NMSE drifted")
    check(abs(rl["roundtrip_nmse"] - 0.224386) < 1e-6,
          "matched RL headline NMSE drifted")
    check(rl["roundtrip_nmse"] < sft["roundtrip_nmse"],
          "matched RL checkpoint does not improve round-trip NMSE")

    improvement = {
        "nmse_absolute": round6(sft["roundtrip_nmse"] - rl["roundtrip_nmse"]),
        "nmse_relative": round6((sft["roundtrip_nmse"] - rl["roundtrip_nmse"]) /
                                  sft["roundtrip_nmse"]),
        "raw_mse_absolute": round6(sft["raw_mse"] - rl["raw_mse"]),
        "raw_mse_relative": round6((sft["raw_mse"] - rl["raw_mse"]) /
                                     sft["raw_mse"]),
        "teacher_win_fraction_gain": round6(
            rl["teacher_win_fraction"] - sft["teacher_win_fraction"]),
    }

    return ({
        "status": "validation_only_matched",
        "row_count": 122,
        "independent_family_count": 122,
        "max_new_tokens": 384,
        "generation_protocol_sha256": MATCHED_RL_PROTOCOL_SHA256,
        "sft": sft,
        "rl": rl,
        "improvement": improvement,
        "source_reports": {
            "sft": "roundtrip_sft_384",
            "rl": "roundtrip_rl_384",
        },
        "scope_note": (
            "Matched held-out validation evidence. The actor and critic were both updated, "
            "so the difference is a joint AV+AR RL effect, not an actor-only attribution. "
            "This is not a sealed test-set result."
        ),
    }, paths)


# ---------------------------------------------------------------------------
# Source verification
# ---------------------------------------------------------------------------

def verify_core_manifest(bundle_dir: Path) -> dict:
    manifest = load_json(bundle_dir / "observatory_manifest.json")
    check(manifest.get("schema_version") == "nano_viz_bundle.v1",
          f"unexpected bundle schema: {manifest.get('schema_version')}")
    check(manifest.get("population") == "QUALIFIED", "bundle population is not QUALIFIED")
    check(manifest.get("split") == "validation", "bundle split is not validation")

    payload = {k: v for k, v in manifest.items() if k != "bundle_id"}
    recomputed = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    check(recomputed == manifest.get("bundle_id"),
          f"bundle_id mismatch: recomputed {recomputed} != manifest {manifest.get('bundle_id')}")

    verified, excluded = [], []
    for entry in manifest["files"]:
        rel = entry["path"]
        target = bundle_dir / rel
        if rel in EXCLUDED_BUNDLE_FILES:
            check(not target.exists() or target.stat().st_size == entry["bytes"],
                  f"excluded file present with wrong size: {rel}")
            excluded.append(rel)
            continue
        check(target.is_file(), f"bundle file missing: {rel}")
        check(target.stat().st_size == entry["bytes"],
              f"bundle file size mismatch: {rel}")
        digest = sha256_file(target)
        check(digest == entry["sha256"],
              f"bundle file hash mismatch: {rel} ({digest} != {entry['sha256']})")
        verified.append(rel)
    check(sorted(excluded) == sorted(EXCLUDED_BUNDLE_FILES),
          f"excluded-file set mismatch: {excluded}")
    manifest["_verified_files"] = verified
    manifest["_excluded_files"] = sorted(excluded)
    return manifest


def verify_poetry_pack(poetry_dir: Path) -> dict:
    reports = {}
    config_hashes = set()
    for phase in POETRY_PHASES:
        rep = load_json(poetry_dir / "reports" / f"{phase}_report.json")
        check(rep.get("schema_version") == "nano_viz_poetry_planning_report.v1",
              f"poetry report schema mismatch: {phase}")
        check(rep.get("phase") == phase, f"poetry report phase mismatch: {phase}")
        check(rep.get("passed") is True, f"poetry phase did not pass: {phase}")
        config_hashes.add(rep.get("config_sha256"))
        reports[phase] = rep
    check(len(config_hashes) == 1,
          f"poetry reports disagree on config hash: {config_hashes}")
    config_sha = config_hashes.pop()

    queue_state = load_json(poetry_dir / "queue_state.json")
    check(queue_state.get("status") == "complete", "poetry queue not complete")
    check(queue_state.get("config_sha256") == config_sha,
          "poetry queue config hash disagrees with reports")

    # Hash-bound artifacts (poetry_score binds paths only, checked by count).
    bindings = [
        (reports["poetry_prepare"]["cases_artifact"], poetry_dir / "poetry_corpus" / "cases.jsonl"),
        (reports["poetry_extract"]["continuations"], poetry_dir / "poetry_extract" / "continuations.jsonl"),
        (reports["poetry_extract"]["trajectories"], poetry_dir / "poetry_extract" / "trajectories.parquet"),
        (reports["poetry_describe"]["descriptions"], poetry_dir / "poetry_describe" / "descriptions.jsonl"),
        (reports["poetry_reconstruct"]["metadata"], poetry_dir / "poetry_reconstruct" / "reconstructions.jsonl"),
        (reports["poetry_intervene"]["interventions"], poetry_dir / "poetry_intervene" / "interventions.jsonl"),
    ]
    for binding, local in bindings:
        digest = sha256_file(local)
        check(digest == binding["sha256"],
              f"poetry artifact hash mismatch: {local.name} ({digest} != {binding['sha256']})")
    return {"config_sha256": config_sha, "reports": reports, "queue_state": queue_state}


def verify_tokenizer(tokenizer_path: Path, trajectories) -> dict:
    from tokenizers import Tokenizer

    check(tokenizer_path.is_file(), f"tokenizer missing: {tokenizer_path}")
    tok = Tokenizer.from_file(str(tokenizer_path))
    mismatches = 0
    for row in trajectories:
        if tok.decode([int(row["token_id"])]).strip() != str(row["token_text"]).strip():
            mismatches += 1
    check(mismatches == 0,
          f"tokenizer failed spot-check on {mismatches}/{len(trajectories)} trace tokens")
    return {
        "tokenizer": tok,
        "provenance": {
            "path": str(tokenizer_path),
            "sha256": sha256_file(tokenizer_path),
            "vocab_size": tok.get_vocab_size(),
            "spot_check": {"tokens": len(trajectories), "mismatches": 0},
        },
    }


# ---------------------------------------------------------------------------
# Table loading + validation
# ---------------------------------------------------------------------------

def load_tables(bundle_dir: Path) -> dict:
    tables = {}
    for name in ["rows", "metrics", "geometry", "retrieval", "explanations",
                 "token_trajectories", "interventions", "behavior", "court",
                 "shapley", "aggregates"]:
        tables[name] = pq.read_table(bundle_dir / f"{name}.parquet").to_pylist()
    return tables


def validate_core(tables: dict, manifest: dict) -> None:
    rows = tables["rows"]
    check(len(rows) == 50, f"rows count {len(rows)} != 50")
    row_ids = [r["row_id"] for r in rows]
    check(len(set(row_ids)) == 50, "duplicate row_id in rows")
    for r in rows:
        check(r["population"] == "QUALIFIED", f"non-qualified row {r['row_id']}")
        check(r["split"] == "validation", f"non-validation row {r['row_id']}")
        check(r["source_text_release_status"] == "privacy_cleared_panel",
              f"row not privacy cleared: {r['row_id']}")
        check(r["claim_scope"] == "stored_snapshot", f"bad claim scope: {r['row_id']}")
        check(r["row_id"] == f"validation-{r['row_index']}", f"row_id convention: {r['row_id']}")
    row_set = set(row_ids)

    inter = tables["interventions"]
    check(len(inter) == 7434, f"interventions count {len(inter)} != 7434")
    cell_ids = [c["cell_id"] for c in inter]
    check(len(set(cell_ids)) == len(cell_ids), "duplicate cell_id in interventions")
    groups: dict[str, dict[float, set[str]]] = {}
    for c in inter:
        check(c["family"] in FAMILIES, f"unknown family {c['family']}")
        check(c["state"] == "ready", f"non-ready cell {c['cell_id']}")
        check(bool(str(c["text"]).strip()), f"blank intervention text {c['cell_id']}")
        check(c["row_id"] in row_set, f"orphan intervention row {c['row_id']}")
        expected_hash = hashlib.sha256(str(c["text"]).strip().encode("utf-8")).hexdigest()
        check(expected_hash == c["text_sha256"], f"text hash mismatch {c['cell_id']}")
        if c["control_group_id"]:
            spec = json.loads(c["spec_json"])
            lane, dose = spec.get("lane"), spec.get("dose")
            check(lane in LANES, f"unknown lane {lane} in {c['cell_id']}")
            groups.setdefault(c["control_group_id"], {}).setdefault(float(dose), set()).add(lane)
    check(len(groups) > 0, "no control groups found")
    for cg, doses in groups.items():
        for dose, lanes in doses.items():
            check(lanes == LANES, f"incomplete control group {cg} dose {dose}: {lanes}")

    cell_set = set(cell_ids)
    metrics = tables["metrics"]
    seen_mc = set()
    primary_cells = set()
    critic_values = set()
    for m in metrics:
        key = (m["critic"], m["cell_id"])
        check(key not in seen_mc, f"duplicate metric key {key}")
        seen_mc.add(key)
        check(m["cell_id"] in cell_set, f"orphan metric cell {m['cell_id']}")
        check(m["row_id"] in row_set, f"orphan metric row {m['row_id']}")
        critic_values.add(m["critic"])
        if m["critic"] == "primary":
            primary_cells.add(m["cell_id"])
        for col in ("directional_mse", "raw_mse", "cosine", "norm_ratio"):
            finite(m[col], f"metrics.{col} {m['cell_id']}")
    check(critic_values == CRITICS, f"critic set {critic_values} != {CRITICS}")
    check(primary_cells == cell_set, "primary critic does not cover every cell")

    behavior = tables["behavior"]
    check(len(behavior) == 888, f"behavior count {len(behavior)} != 888")
    beh_rows: dict[str, int] = {}
    for b in behavior:
        check(b["cell_id"] in cell_set, f"orphan behavior cell {b['cell_id']}")
        beh_rows[b["row_id"]] = beh_rows.get(b["row_id"], 0) + 1
        check(len(b["baseline_continuation_token_ids"]) == 32, "baseline continuation != 32 tokens")
        check(len(b["patched_continuation_token_ids"]) == 32, "patched continuation != 32 tokens")
    check(len(beh_rows) == 24 and set(beh_rows.values()) == {37},
          f"behavior coverage not 24 rows x 37 cells: {len(beh_rows)} rows")

    expl = tables["explanations"]
    check(len(expl) == 900, f"explanations count {len(expl)} != 900")
    refs = [e["ref"] for e in expl]
    check(len(set(refs)) == 900, "duplicate explanation ref")
    for e in expl:
        check(e["kind"] in EXPLANATION_KINDS, f"unknown explanation kind {e['kind']}")
        check(e["parse_state"] in PARSE_STATES, f"unknown parse state {e['parse_state']}")
        check(bool(str(e["text"]).strip()), f"blank explanation {e['ref']}")
    expl_refs = set(refs)

    traj = tables["token_trajectories"]
    check(len(traj) == 400, f"trajectories count {len(traj)} != 400")
    for t in traj:
        check(t["description_ref"] in expl_refs, f"missing trace description {t['description_ref']}")
        check(bool(t["description_usable"]), f"unusable trace description {t['ref']}")
        check(t["row_id"] in row_set, f"orphan trace row {t['row_id']}")

    retrieval = tables["retrieval"]
    check(len(retrieval) == 100, f"retrieval count {len(retrieval)} != 100")
    for r in retrieval:
        check(r["critic"] in CRITICS, f"unknown retrieval critic {r['critic']}")
        finite(r["expected_cosine"], f"retrieval {r['row_id']}")

    court = tables["court"]
    check(len(court) == 1200, f"court count {len(court)} != 1200")
    for c in court:
        check(c["family"] in COURT_FAMILIES, f"unexpected court family {c['family']}")
        check(c["calibration_label"] in CALIBRATION_LABELS,
              f"unknown calibration label {c['calibration_label']}")
        finite(c["identity_cosine"], f"court {c['cell_id']}")

    shapley = tables["shapley"]
    check(len(shapley) == 200, f"shapley count {len(shapley)} != 200")
    for s in shapley:
        check(s["section"] == SECTIONS[s["section_index"]], f"section mismatch {s}")
        check(s["utility"] == "one_minus_directional_mse", "unexpected shapley utility")
        finite(s["shapley_value"], f"shapley {s['row_id']}")

    geometry = tables["geometry"]
    expected_geo = len(metrics) + 50
    check(len(geometry) == expected_geo, f"geometry count {len(geometry)} != {expected_geo}")
    for g in geometry:
        for col in ("x", "y", "z", "native_norm"):
            finite(g[col], f"geometry.{col} {g['ref']}")

    counts = manifest["counts"]
    observed = {"rows": len(rows), "interventions": len(inter), "behavior": len(behavior),
                "trajectories": len(traj), "vectors": len(metrics) + 50 + 400}
    check(counts == observed, f"manifest counts {counts} != observed {observed}")


def validate_aggregates(tables: dict, bundle_dir: Path) -> dict:
    """Recompute the family-clustered bootstrap bit-exactly and compare."""
    agg_json = load_json(bundle_dir / "aggregates.json")
    check(agg_json.get("fit_split") == "validation", "aggregates fit split")
    check(agg_json.get("family_clustered") is True, "aggregates not family clustered")

    metrics = tables["metrics"]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for m in metrics:
        grouped.setdefault((m["critic"], m["family"]), []).append(m)

    seed = 20260716  # statistics.seed in offline_observatory_bundle.yaml
    samples, confidence = 2000, 0.95
    alpha = (1 - confidence) / 2
    expected_keys = set()
    for gi, (critic, family) in enumerate(sorted(grouped)):
        rows = grouped[(critic, family)]
        for metric in ("directional_mse", "cosine"):
            key = f"{critic}.{family}.{metric}"
            expected_keys.add(key)
            fam_ids = np.array([r["content_family_id"] for r in rows])
            values = np.array([float(r[metric]) for r in rows], dtype=np.float64)
            families = np.unique(fam_ids)
            fam_means = np.array([values[fam_ids == f].mean() for f in families])
            mean = float(fam_means.mean())
            rng = np.random.default_rng(stable_int(seed, gi, metric))
            draws = rng.integers(0, len(families), size=(samples, len(families)))
            boot = fam_means[draws].mean(axis=1)
            ci_low = float(np.quantile(boot, alpha))
            ci_high = float(np.quantile(boot, 1 - alpha))
            stored = agg_json["aggregates"].get(key)
            check(stored is not None, f"aggregate missing: {key}")
            for name, ours in (("mean", mean), ("ci_low", ci_low), ("ci_high", ci_high)):
                check(abs(stored[name] - ours) <= 1e-6,
                      f"aggregate {key}.{name} mismatch: {stored[name]} vs {ours}")
            check(stored["rows"] == len(rows) and stored["families"] == len(families),
                  f"aggregate {key} counts mismatch")
    check(set(agg_json["aggregates"].keys()) == expected_keys,
          f"aggregate key set mismatch: extra {set(agg_json['aggregates']) - expected_keys}")
    return agg_json


# ---------------------------------------------------------------------------
# Shard construction (pure transforms of validated tables)
# ---------------------------------------------------------------------------

def spec_of(cell: dict) -> dict:
    return json.loads(cell["spec_json"])


def slim_metric(m: dict) -> dict:
    return {
        "dmse": round6(float(m["directional_mse"])),
        "raw_mse": round6(float(m["raw_mse"])),
        "cosine": round6(float(m["cosine"])),
        "norm_ratio": round6(float(m["norm_ratio"])),
    }


def build_rows_shard(tables: dict) -> dict:
    rows = []
    for r in sorted(tables["rows"], key=lambda x: x["row_index"]):
        rows.append({
            "row_id": r["row_id"],
            "row_index": r["row_index"],
            "doc_id": r["doc_id"],
            "content_family_id": r["content_family_id"],
            "n_raw_tokens": r["n_raw_tokens"],
            "token_position": r["token_position"],
            "activation_norm": round6(float(r["activation_norm"])),
            "source_text": r["source_text"],
            "teacher_text": r["target_explanation"],
            "av_text": r["av_explanation"],
            "release_status": r["source_text_release_status"],
            "claim_scope": r["claim_scope"],
            "stratum": json.loads(r["stratum_json"]),
        })
    return {"schema_version": SCHEMA_VERSION, "kind": "rows", "rows": rows}


def build_channel_shard(tables: dict, agg_json: dict, reports: dict,
                        online_rl: dict) -> dict:
    metrics = tables["metrics"]
    rows_by_id = {r["row_id"]: r for r in tables["rows"]}
    by_cell: dict[str, dict[str, dict]] = {}
    inter_by_cell = {c["cell_id"]: c for c in tables["interventions"]}
    for m in metrics:
        by_cell.setdefault(m["cell_id"], {})[m["critic"]] = m

    identity = []
    identity_by_row: dict[str, dict[str, dict]] = {}
    for cell_id, critics in by_cell.items():
        cell = inter_by_cell[cell_id]
        if cell["family"] != "identity":
            continue
        for critic, m in critics.items():
            entry = {"row_id": m["row_id"], "critic": critic, **slim_metric(m)}
            identity.append(entry)
            identity_by_row.setdefault(m["row_id"], {})[critic] = entry
    identity.sort(key=lambda x: (x["row_id"], x["critic"]))
    check(len(identity) == 100, f"identity metric rows {len(identity)} != 100")

    twin_rows = []
    for row_id in sorted(identity_by_row):
        pair = identity_by_row[row_id]
        check(set(pair) == CRITICS, f"identity missing critic for {row_id}")
        twin_rows.append({
            "row_id": row_id,
            "primary_dmse": pair["primary"]["dmse"],
            "independent_dmse": pair["independent"]["dmse"],
            "primary_cosine": pair["primary"]["cosine"],
            "independent_cosine": pair["independent"]["cosine"],
        })

    truncation: dict[str, list[dict]] = {}
    occlusion: dict[str, list[dict]] = {}
    tellings: dict[str, list[dict]] = {}
    for cell_id, critics in by_cell.items():
        cell = inter_by_cell[cell_id]
        m = critics["primary"]
        row_id = cell["row_id"]
        spec = spec_of(cell)
        if cell["family"] == "truncation":
            teacher = rows_by_id[row_id]["target_explanation"]
            word_count = len(WORD_PATTERN.findall(teacher))
            truncation.setdefault(row_id, []).append({
                "fraction": spec["fraction"],
                "words": max(1, int(round(word_count * spec["fraction"]))),
                "dmse": round6(float(m["directional_mse"])),
                "cosine": round6(float(m["cosine"])),
            })
        elif cell["family"] == "word_occlusion":
            teacher = rows_by_id[row_id]["target_explanation"]
            matches = list(WORD_PATTERN.finditer(teacher))
            wi = spec["word_index"]
            check(wi < len(matches), f"word index out of range {cell_id}")
            match = matches[wi]
            rebuilt = (teacher[:match.start()] + spec["replacement"] + teacher[match.end():]).strip()
            check(rebuilt == cell["text"], f"occluded-word recovery failed for {cell_id}")
            base = identity_by_row[row_id]["primary"]["dmse"]
            occlusion.setdefault(row_id, []).append({
                "word_index": wi,
                "word": match.group(0),
                "char_start": match.start(),
                "char_end": match.end(),
                "dmse": round6(float(m["directional_mse"])),
                "d_dmse": round6(float(m["directional_mse"]) - base),
            })
        elif cell["family"] == "alternate_telling":
            tellings.setdefault(row_id, []).append({
                "cell_id": cell_id,
                "sample_index": spec["sample_index"],
                "text": cell["text"],
                "dmse": round6(float(m["directional_mse"])),
                "cosine": round6(float(m["cosine"])),
            })
    for rows in truncation.values():
        rows.sort(key=lambda x: x["fraction"])
    for rows in occlusion.values():
        rows.sort(key=lambda x: x["word_index"])
    for rows in tellings.values():
        rows.sort(key=lambda x: x["sample_index"])
    check(sum(len(v) for v in tellings.values()) == 400, "alternate tellings != 400")

    shapley: dict[str, dict] = {}
    for s in tables["shapley"]:
        entry = shapley.setdefault(s["row_id"], {"sections": {}, "efficiency_error": 0.0})
        entry["sections"][s["section"]] = round6(float(s["shapley_value"]))
        entry["efficiency_error"] = round6(float(s["efficiency_error"]))

    retrieval = [{
        "row_id": r["row_id"], "critic": r["critic"], "rank": r["rank"],
        "nearest_row_id": r["nearest_row_id"],
        "expected_cosine": round6(float(r["expected_cosine"])),
    } for r in sorted(tables["retrieval"], key=lambda x: (x["row_id"], x["critic"]))]

    e0 = reports["e0"]
    e1av, e2, e3, p2 = reports["e1_av"], reports["e2"], reports["e3"], reports["p2"]
    waterfall_variants = {}
    for name, stats in e0["information_waterfall"]["variants"].items():
        fam = stats["family_clustered_directional_mse"]
        waterfall_variants[name] = {
            "dmse": round6(stats["directional_mse"]),
            "cosine_mean": round6(stats["cosine_mean"]),
            "ci_low": round6(fam["ci95_low"]),
            "ci_high": round6(fam["ci95_high"]),
            "rows": fam["row_count"],
            "families": fam["family_count"],
            "norm_ratio_mean": round6(stats["norm_ratio_mean"]),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "channel",
        "matched_online_rl": online_rl,
        "aggregates": agg_json["aggregates"],
        "court_thresholds": agg_json["court_thresholds"],
        "fit_split": agg_json["fit_split"],
        "identity": identity,
        "twin_critics": {
            "per_row": twin_rows,
            "e3_summaries": e3["summaries"],
            "p2_summaries": p2["summaries"],
            "confound": "Twin critics share a teacher target and are not semantically independent.",
        },
        "retrieval": retrieval,
        "waterfall": {
            "metric": e0["information_waterfall"]["metric"],
            "split": e0["information_waterfall"]["split"],
            "variants": waterfall_variants,
            "source_report": "e0",
        },
        "capacity_ladder": {**e0["capacity_ladder"], "source_report": "e0"},
        "real_vs_control": {
            "e1_av": {"losses": e1av["losses"], "rows": e1av["rows"],
                      "parse": e1av["parse"], "source_report": "e1_av"},
            "e2": {"mean_loss": e2["mean_loss"], "paired": e2["paired"],
                   "rows": e2["rows"], "records": e2["records"], "source_report": "e2"},
        },
        "truncation": truncation,
        "occlusion": occlusion,
        "tellings": tellings,
        "shapley": shapley,
    }


def build_rewrites_shard(tables: dict) -> dict:
    court_by_cell: dict[str, dict] = {}
    for c in tables["court"]:
        entry = court_by_cell.setdefault(c["cell_id"], {
            "cell_id": c["cell_id"], "row_id": c["row_id"], "family": c["family"],
            "variant": c["variant"], "text": c["text"], "spec": spec_of(c),
            "metrics": {}, "court": {},
        })
        entry["metrics"][c["critic"]] = slim_metric(c)
        entry["court"][c["critic"]] = {
            "identity_cosine": round6(float(c["identity_cosine"])),
            "calibration_label": c["calibration_label"],
            "semanticity_verdict": bool(c["semanticity_verdict"]),
        }
    cells = sorted(court_by_cell.values(),
                   key=lambda x: (x["row_id"], x["family"], x["variant"]))
    check(len(cells) == 600, f"rewrite cells {len(cells)} != 600")
    for cell in cells:
        check(set(cell["metrics"]) == CRITICS, f"rewrite cell missing critic {cell['cell_id']}")

    inter_by_cell = {c["cell_id"]: c for c in tables["interventions"]}
    metrics_identity: dict[str, dict[str, dict]] = {}
    for m in tables["metrics"]:
        cell = inter_by_cell[m["cell_id"]]
        if cell["family"] == "identity":
            metrics_identity.setdefault(m["row_id"], {})[m["critic"]] = slim_metric(m)
    identity = [{"row_id": rid, "metrics": crit, "text": inter_by_cell[cid]["text"]}
                for rid, crit in sorted(metrics_identity.items())
                for cid in [next(c["cell_id"] for c in tables["interventions"]
                                 if c["family"] == "identity" and c["row_id"] == rid)]]
    return {"schema_version": SCHEMA_VERSION, "kind": "rewrites",
            "identity": identity, "cells": cells}


def build_geometry_shard(tables: dict, reports: dict) -> dict:
    points = [{
        "ref": g["ref"], "kind": g["kind"], "family": g["family"],
        "variant": g["variant"], "critic": g["critic"], "row_id": g["row_id"],
        "cell_id": g["cell_id"],
        "x": round6(float(g["x"])), "y": round6(float(g["y"])), "z": round6(float(g["z"])),
        "native_norm": round6(float(g["native_norm"])),
        "target_cosine": round6(float(g["target_cosine"])),
        "dmse": round6(float(g["directional_mse"])),
    } for g in sorted(tables["geometry"], key=lambda x: x["ref"])]
    geo_rep = reports["geometry"]
    return {
        "schema_version": SCHEMA_VERSION, "kind": "geometry",
        "fit_split": geo_rep["fit_split"],
        "components": geo_rep["components"],
        "explained_variance_ratio": [round6(v) for v in geo_rep["explained_variance_ratio"]],
        "note": "PCA basis fit on the 50 validation target vectors only (validation-fitted exploratory view).",
        "points": points,
    }


def build_trace_shard(tables: dict, reports: dict, tokenizer) -> dict:
    expl_by_ref = {e["ref"]: e for e in tables["explanations"]}
    rows_by_id = {r["row_id"]: r for r in tables["rows"]}
    source_encodings = {}
    e5 = reports["e5"]
    drift_by_row = {d["row_id"]: d for d in e5["stored_final_position_drift"]}
    docs: dict[str, dict] = {}
    for t in sorted(tables["token_trajectories"], key=lambda x: (x["row_id"], x["position"])):
        doc = docs.setdefault(t["row_id"], {
            "row_id": t["row_id"], "doc_id": t["doc_id"],
            "content_family_id": t["content_family_id"], "positions": [],
        })
        desc = expl_by_ref[t["description_ref"]]
        row = rows_by_id.get(t["row_id"])
        check(row is not None, f"trace row missing source text: {t['row_id']}")
        if t["row_id"] not in source_encodings:
            source_encodings[t["row_id"]] = tokenizer.encode(
                row["source_text"], add_special_tokens=False)
        encoding = source_encodings[t["row_id"]]
        position = int(t["position"])
        source_text = row["source_text"]
        source_aligned = (
            position < len(encoding.ids)
            and int(encoding.ids[position]) == int(t["token_id"])
        )
        if source_aligned:
            char_start, char_end = encoding.offsets[position]
            context_start = max(0, char_start - 320)
            context_end = min(len(source_text), char_end + 220)
            source_before = source_text[context_start:char_start]
            source_token = source_text[char_start:char_end]
            source_after = source_text[char_end:context_end]
        else:
            context_start = context_end = 0
            source_before = source_after = ""
            source_token = str(t["token_text"])
        doc["positions"].append({
            "position": position,
            "n_context_tokens": t["n_context_tokens"],
            "token_id": t["token_id"],
            "token_text": t["token_text"],
            "source_alignment": "exact" if source_aligned else "unavailable",
            "source_char_start": char_start if source_aligned else None,
            "source_char_end": char_end if source_aligned else None,
            "source_before": source_before,
            "source_token": source_token,
            "source_after": source_after,
            "source_prefix_omitted": source_aligned and context_start > 0,
            "source_suffix_omitted": source_aligned and context_end < len(source_text),
            "description": desc["text"],
            "parse_state": desc["parse_state"],
            "usable": bool(t["description_usable"]),
        })
    check(len(docs) == 10, f"trace docs {len(docs)} != 10")
    for row_id, doc in docs.items():
        check(len(doc["positions"]) == 40, f"trace doc {row_id} has {len(doc['positions'])} != 40")
        drift = drift_by_row.get(row_id)
        check(drift is not None, f"missing drift entry for trace doc {row_id}")
        doc["drift"] = {
            "one_minus_cos": round6(drift["one_minus_cos"]),
            "relative_l2": round6(drift["relative_l2"]),
            "rms_ratio": round6(drift["rms_ratio"]),
            "max_abs": round6(drift["max_abs"]),
        }
    exact_positions = sum(
        p["source_alignment"] == "exact"
        for doc in docs.values()
        for p in doc["positions"]
    )
    return {
        "schema_version": SCHEMA_VERSION, "kind": "trace",
        "claim_scope": e5["claim_scope"],
        "boundary": e5["boundary"],
        "shuffled_control": {
            "available": False,
            "note": "The core bundle ships no shuffled-position trace control; the poetry "
                    "planning lens carries the real-vs-shuffled control for this station.",
        },
        "source_alignment": {
            "exact_positions": exact_positions,
            "unavailable_positions": 400 - exact_positions,
            "note": "Exact tokenizer-offset context is shown only where the released row text "
                    "matches the trajectory token ids at every selected position. Unaligned rows "
                    "stay explicit and are never assigned a guessed highlight.",
        },
        "docs": [docs[k] for k in sorted(docs)],
    }


def build_poetry_shard(poetry_dir: Path, poetry_meta: dict) -> dict:
    reports = poetry_meta["reports"]
    cases = read_jsonl(poetry_dir / "poetry_corpus" / "cases.jsonl")
    check(len(cases) == 8, f"poetry cases {len(cases)} != 8")
    traj = pq.read_table(poetry_dir / "poetry_extract" / "trajectories.parquet",
                         columns=["case_id", "position", "relative_offset",
                                  "token_id", "token_text"]).to_pylist()
    check(len(traj) == 104, f"poetry positions {len(traj)} != 104")
    continuations = {c["case_id"]: c for c in read_jsonl(poetry_dir / "poetry_extract" / "continuations.jsonl")}
    descriptions = read_jsonl(poetry_dir / "poetry_describe" / "descriptions.jsonl")
    check(len(descriptions) == 624, f"poetry descriptions {len(descriptions)} != 624")
    sample_scores = read_jsonl(poetry_dir / "poetry_score" / "sample_scores.jsonl")
    check(len(sample_scores) == 624, f"poetry sample scores {len(sample_scores)} != 624")
    position_scores = read_jsonl(poetry_dir / "poetry_score" / "position_scores.jsonl")
    check(len(position_scores) == 208, f"poetry position scores {len(position_scores)} != 208")
    case_scores = {c["case_id"]: c for c in read_jsonl(poetry_dir / "poetry_score" / "case_scores.jsonl")}
    check(len(case_scores) == 8, "poetry case scores != 8")
    recon = {r["case_id"]: r for r in read_jsonl(poetry_dir / "poetry_reconstruct" / "reconstructions.jsonl")}
    interventions = read_jsonl(poetry_dir / "poetry_intervene" / "interventions.jsonl")
    check(len(interventions) == 30, f"poetry interventions {len(interventions)} != 30")

    desc_by_key = {}
    for d in descriptions:
        desc_by_key[(d["case_id"], d["position"], d["variant"], d["sample_index"])] = d

    samples_by_case: dict[str, list[dict]] = {}
    for s in sample_scores:
        key = (s["case_id"], s["position"], s["variant"], s["sample_index"])
        d = desc_by_key.get(key)
        check(d is not None, f"poetry sample without description: {key}")
        samples_by_case.setdefault(s["case_id"], []).append({
            "position": s["position"],
            "relative_offset": s["relative_offset"],
            "variant": s["variant"],
            "sample_index": s["sample_index"],
            "source_case_id": d["source_case_id"],
            "usable": bool(s["usable"]),
            "target_exact": bool(s["target_exact"]),
            "target_family": bool(s["target_family"]),
            "alternate_family": bool(s["alternate_family"]),
            "explanation": s["explanation"],
            "parse": {
                "closed": bool(d["parsed"]["closed"]),
                "extraction_mode": d["parsed"]["extraction_mode"],
                "repetition_loop": bool(d["parsed"]["repetition_loop"]),
            },
        })

    pos_by_case: dict[str, list[dict]] = {}
    for p in position_scores:
        pos_by_case.setdefault(p["case_id"], []).append({
            "position": p["position"], "relative_offset": p["relative_offset"],
            "variant": p["variant"], "samples": p["samples"],
            "usable_rate": round6(p["usable_rate"]),
            "target_exact_rate": round6(p["target_exact_rate"]),
            "target_family_rate": round6(p["target_family_rate"]),
            "alternate_family_rate": round6(p["alternate_family_rate"]),
        })

    # Recompute onset + anchor lift from position scores (fail closed on drift
    # between derived artifacts).
    onset_rate = 0.25  # gates.planning_onset_rate in offline_observatory_poetry.yaml
    for case_id, cs in case_scores.items():
        entries = pos_by_case[case_id]
        real = sorted([e for e in entries if e["variant"] == "real"], key=lambda e: e["position"])
        shuffled = {e["position"]: e for e in entries if e["variant"] == "shuffled"}
        onset = None
        for e in real:
            sh = shuffled[e["position"]]
            if e["target_family_rate"] >= onset_rate and e["target_family_rate"] > sh["target_family_rate"]:
                onset = e["position"]
                break
        check(onset == cs["planning_onset_position"],
              f"onset recompute mismatch for {case_id}: {onset} != {cs['planning_onset_position']}")
        anchor = max(e["position"] for e in real)
        lift = round(next(e["target_family_rate"] for e in real if e["position"] == anchor)
                     - shuffled[anchor]["target_family_rate"], 9)
        check(abs(lift - cs["anchor_lift"]) < 1e-9,
              f"anchor lift recompute mismatch for {case_id}")

    iv_by_case: dict[str, list[dict]] = {}
    edited_hits, random_hits = [], []
    for iv in interventions:
        iv_by_case.setdefault(iv["case_id"], []).append({
            "direction": iv["direction"], "dose": iv["dose"],
            "continuation_text": iv["continuation_text"],
            "hits_target_family": bool(iv["hits_target_family"]),
            "hits_alternate_family": bool(iv["hits_alternate_family"]),
        })
        (edited_hits if iv["direction"] == "edited" else random_hits).append(
            bool(iv["hits_alternate_family"]))
    check(abs(sum(edited_hits) / len(edited_hits)
              - reports["poetry_intervene"]["edited_alternate_hit_rate"]) < 1e-9,
          "edited alternate hit rate mismatch")
    check(abs(sum(random_hits) / len(random_hits)
              - reports["poetry_intervene"]["random_alternate_hit_rate"]) < 1e-9,
          "random alternate hit rate mismatch")

    out_cases = []
    for c in cases:
        cs = case_scores[c["case_id"]]
        analysis = sorted([t for t in traj if t["case_id"] == c["case_id"]],
                          key=lambda t: t["position"])
        check(len(analysis) == len(c["analysis_positions"]),
              f"analysis positions mismatch for {c['case_id']}")
        out_cases.append({
            "case_id": c["case_id"],
            "framing": c["framing"],
            "first_line": c["first_line"],
            "second_line": c["second_line"],
            "prefix_text": c["prefix_text"],
            "cue": c["cue"],
            "target_word": c["target_word"],
            "target_terms": c["target_terms"],
            "alternate_terms": c["alternate_terms"],
            "edit_map": c["edit_map"],
            "anchor_position": c["anchor_position"],
            "analysis": [{
                "position": t["position"], "relative_offset": t["relative_offset"],
                "token_id": t["token_id"], "token_text": t["token_text"],
            } for t in analysis],
            "baseline_continuation": cs["baseline_continuation"],
            "baseline_hits_target_family": bool(cs["baseline_hits_target_family"]),
            "planning_onset_position": cs["planning_onset_position"],
            "planning_onset_relative_offset": cs["planning_onset_relative_offset"],
            "anchor_lift": round6(cs["anchor_lift"]),
            "anchor_real_target_family_rate": round6(cs["anchor_real_target_family_rate"]),
            "anchor_shuffled_target_family_rate": round6(cs["anchor_shuffled_target_family_rate"]),
            "position_scores": sorted(pos_by_case[c["case_id"]],
                                      key=lambda e: (e["position"], e["variant"])),
            "samples": sorted(samples_by_case[c["case_id"]],
                              key=lambda e: (e["position"], e["variant"], e["sample_index"])),
            "reconstruction": ({
                "original_explanation": recon[c["case_id"]]["original_explanation"],
                "edited_explanation": recon[c["case_id"]]["edited_explanation"],
                "changed_terms": recon[c["case_id"]]["changed_terms"],
                "original_cosine": round6(recon[c["case_id"]]["original_cosine"]),
                "original_dmse": round6(recon[c["case_id"]]["original_directional_mse"]),
                "edit_delta_norm": round6(recon[c["case_id"]]["edit_delta_norm"]),
            } if c["case_id"] in recon else None),
            "interventions": sorted(iv_by_case.get(c["case_id"], []),
                                    key=lambda e: (e["direction"], e["dose"])),
        })

    rep_summary = {
        phase: {
            "passed": rep["passed"],
            "config_sha256": rep["config_sha256"],
        } for phase, rep in reports.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "poetry",
        "claim_scope": "fresh_forward_exploratory",
        "config_sha256": poetry_meta["config_sha256"],
        "gates": {"planning_onset_rate": onset_rate, "minimum_usable_fraction": 0.80},
        "aggregates": {
            "cases": len(cases),
            "positions": reports["poetry_extract"]["positions"],
            "samples": len(sample_scores),
            "usable_fraction": round6(reports["poetry_describe"]["usable_fraction"]),
            "mean_anchor_lift": round6(reports["poetry_score"]["mean_anchor_lift"]),
            "cases_with_planning_onset": reports["poetry_score"]["cases_with_planning_onset"],
            "cases_with_baseline_target_rhyme": reports["poetry_score"]["cases_with_baseline_target_rhyme"],
            "editable_cases": reports["poetry_reconstruct"]["eligible_cases"],
            "steering_doses": reports["poetry_intervene"]["steering_doses"],
            "edited_alternate_hit_rate": round6(reports["poetry_intervene"]["edited_alternate_hit_rate"]),
            "random_alternate_hit_rate": round6(reports["poetry_intervene"]["random_alternate_hit_rate"]),
            "mean_original_dmse": round6(reports["poetry_reconstruct"]["mean_original_directional_mse"]),
        },
        "interpretation": {
            "signal": "weak",
            "notes": [
                "Mean anchor lift is 0.03125 across 8 cases; only lantern-light shows a positive anchor lift.",
                "The unpatched baseline continuation lands in the target rhyme family for 1 of 8 cases.",
                "No causal edit produced an alternate-family rhyme at any dose (edited and random hit rates are both 0).",
                "This is a rough planning signal, not proof of an internal plan.",
            ],
        },
        "reports": rep_summary,
        "cases": out_cases,
    }


def build_bench_shards(tables: dict, geometry_shard: dict, tokenizer, manifest: dict) -> tuple[dict, dict[str, dict]]:
    inter_by_cell = {c["cell_id"]: c for c in tables["interventions"]}
    metrics_by_cell: dict[str, dict[str, dict]] = {}
    for m in tables["metrics"]:
        metrics_by_cell.setdefault(m["cell_id"], {})[m["critic"]] = m
    geo_by_ref = {p["ref"]: p for p in geometry_shard["points"]}

    behavior_by_row: dict[str, list[dict]] = {}
    for b in tables["behavior"]:
        behavior_by_row.setdefault(b["row_id"], []).append(b)
    behavior_rows = sorted(behavior_by_row)

    def decode(token_id: int) -> str:
        return tokenizer.decode([int(token_id)])

    # Coverage map for all 50 rows.
    coverage: dict[str, dict[str, dict]] = {}
    control_groups: dict[str, dict] = {}
    for c in tables["interventions"]:
        row_cov = coverage.setdefault(c["row_id"], {})
        fam = row_cov.setdefault(c["family"], {"variants": [], "depths": set()})
        fam["variants"].append(c["variant"])
        fam["depths"].add(c["depth"])
        if c["control_group_id"]:
            spec = spec_of(c)
            cg = control_groups.setdefault(c["control_group_id"], {
                "row_id": c["row_id"], "chip": spec["chip"], "cells": {}})
            cg["cells"].setdefault(spec["lane"], {})[f"{spec['dose']:g}"] = c["cell_id"]

    index_rows = []
    for r in sorted(tables["rows"], key=lambda x: x["row_index"]):
        row_id = r["row_id"]
        fams = {}
        for fam_name, fam in sorted(coverage.get(row_id, {}).items()):
            fams[fam_name] = {
                "variants": sorted(fam["variants"]),
                "depths": sorted(fam["depths"]),
            }
        index_rows.append({
            "row_id": row_id,
            "has_behavior": row_id in behavior_by_row,
            "families": fams,
        })

    index = {
        "schema_version": SCHEMA_VERSION,
        "kind": "bench_index",
        "banner": {
            "statement": f"You are choosing among {len(tables['interventions'])} precomputed "
                         f"experiments (grid spec {manifest['bundle_config_sha256'][:12]}…), "
                         "not editing a live model.",
            "total_cells": len(tables["interventions"]),
            "behavior_cells": len(tables["behavior"]),
            "behavior_rows": len(behavior_rows),
            "grid_spec_sha256": manifest["bundle_config_sha256"],
            "claim_scope": "stored_snapshot",
            "functional_claim_status": "validation_only_exploratory",
        },
        "rows": index_rows,
        "behavior_rows": behavior_rows,
        "control_groups": dict(sorted(control_groups.items())),
    }

    row_shards: dict[str, dict] = {}
    for row_id in behavior_rows:
        cells = []
        for b in sorted(behavior_by_row[row_id], key=lambda x: x["cell_id"]):
            cell = inter_by_cell[b["cell_id"]]
            spec = spec_of(cell)
            bm = json.loads(b["metrics_json"])
            wake = json.loads(b["wake_json"])
            otk = json.loads(b["original_topk_json"])
            ptk = json.loads(b["patched_topk_json"])
            geo = geo_by_ref.get(f"prediction:primary:{b['cell_id']}")
            crit_metrics = {crit: slim_metric(m)
                            for crit, m in metrics_by_cell[b["cell_id"]].items()}
            cells.append({
                "cell_id": b["cell_id"],
                "family": cell["family"],
                "variant": cell["variant"],
                "depth": cell["depth"],
                "control_group_id": cell["control_group_id"],
                "spec": spec,
                "text": cell["text"],
                "metrics": crit_metrics,
                "geometry": ({"x": geo["x"], "y": geo["y"], "z": geo["z"]} if geo else None),
                "behavior": {
                    "js_divergence": round6(finite(bm["js_divergence"], b["cell_id"])),
                    "kl_original_to_patched": round6(finite(bm["kl_original_to_patched"], b["cell_id"])),
                    "logit_pearson": round6(finite(bm["logit_pearson"], b["cell_id"])),
                    "top_10_overlap": round6(bm["top_10_overlap"]),
                    "top_50_overlap": round6(bm["top_50_overlap"]),
                    "original_top1_rank": bm["original_top1_rank"],
                    "vocab_size": bm["vocab_size"],
                },
                "topk": {
                    "original": [{"id": tid, "text": decode(tid), "p": round6(p)}
                                 for tid, p in zip(otk["token_ids"][:10], otk["probabilities"][:10])],
                    "patched": [{"id": tid, "text": decode(tid), "p": round6(p)}
                                for tid, p in zip(ptk["token_ids"][:10], ptk["probabilities"][:10])],
                },
                "wake": [{
                    "offset": w["offset"],
                    "js": round6(finite(w["js_divergence"], f"wake {b['cell_id']}")),
                    "kl": round6(finite(w["kl_original_to_patched"], f"wake {b['cell_id']}")),
                    "top_10_overlap": round6(w["top_10_overlap"]),
                    "top_50_overlap": round6(w["top_50_overlap"]),
                } for w in wake],
                "baseline_continuation": b["baseline_continuation_text"],
                "patched_continuation": b["patched_continuation_text"],
                "generation_protocol": json.loads(b["generation_protocol_json"]),
            })
        target_geo = geo_by_ref.get(f"target:{row_id}")
        row_shards[row_id] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "bench_row",
            "row_id": row_id,
            "claim_scope": "stored_snapshot",
            "target_geometry": ({"x": target_geo["x"], "y": target_geo["y"], "z": target_geo["z"]}
                                if target_geo else None),
            "cells": cells,
        }
    return index, row_shards


def build_audit_shard(tables: dict, manifest: dict, provenance: dict, claim_ledger: dict,
                      agg_json: dict, reports: dict, poetry_meta: dict,
                      tokenizer_prov: dict, online_rl: dict) -> dict:
    e0 = reports["e0"]

    # Court docket per (row, critic).
    docket: dict[tuple[str, str], dict] = {}
    for c in tables["court"]:
        key = (c["row_id"], c["critic"])
        d = docket.setdefault(key, {
            "row_id": c["row_id"], "critic": c["critic"],
            "paraphrase": [], "corruption": [],
        })
        d[c["family"]].append({
            "variant": c["variant"],
            "identity_cosine": round6(float(c["identity_cosine"])),
            "label": c["calibration_label"],
            "verdict": bool(c["semanticity_verdict"]),
        })
    docket_rows = []
    for (row_id, critic), d in sorted(docket.items()):
        para = sorted(d["paraphrase"], key=lambda x: x["variant"])
        corr = sorted(d["corruption"], key=lambda x: x["variant"])
        para_cos = [p["identity_cosine"] for p in para]
        verdicts = [p["verdict"] for p in para]
        docket_rows.append({
            "row_id": row_id, "critic": critic,
            "paraphrase_min_identity_cosine": min(para_cos),
            "paraphrase_mean_identity_cosine": round6(sum(para_cos) / len(para_cos)),
            "paraphrase_verdicts_true": sum(verdicts),
            "paraphrase_cells": para,
            "corruption_cells": corr,
            "row_verdict": ("honest" if all(verdicts)
                            else "mixed" if any(verdicts) else "suspect"),
        })

    expl_state_counts: dict[str, dict[str, int]] = {}
    for e in tables["explanations"]:
        kind = expl_state_counts.setdefault(e["kind"], {})
        kind[e["parse_state"]] = kind.get(e["parse_state"], 0) + 1

    misses = [r for r in tables["retrieval"] if r["rank"] != 1]
    negative_results = [
        {
            "id": "poetry_anchor_lift",
            "status": "weak",
            "statement": "Poetry planning: mean anchor lift is 0.03125 across 8 cases; "
                         "7 of 8 cases show zero or no lift at the anchor token.",
            "source": "data/poetry.json",
        },
        {
            "id": "poetry_baseline_rhyme",
            "status": "negative",
            "statement": "The unpatched model lands in the intended rhyme family for only "
                         "1 of 8 poetry baselines.",
            "source": "data/poetry.json",
        },
        {
            "id": "poetry_causal_steering",
            "status": "negative",
            "statement": "No alternate-family rhyme was produced by any causal edit at any "
                         "dose (edited hit rate 0.0; random-direction control 0.0).",
            "source": "data/poetry.json",
        },
        {
            "id": "retrieval_misses",
            "status": "negative",
            "statement": f"Identity retrieval misses rank 1 on {len(misses)} of 100 "
                         "(row, critic) pairs; worst rank "
                         f"{max((r['rank'] for r in tables['retrieval']), default=1)}.",
            "source": "data/channel.json",
        },
        {
            "id": "drift_not_publication_ready",
            "status": "caveat",
            "statement": "The stored-vs-fresh activation drift card is marked "
                         "publication_ready=false in the qualification report.",
            "source": "data/reports/e0.json",
        },
    ]

    claims = dict(claim_ledger["claims"])
    claims["matched_online_rl_roundtrip"] = "validation_only_confirmatory"
    limitations = list(claim_ledger["limitations"])
    limitations.append(online_rl["scope_note"])

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "audit",
        "claim_ledger": {
            "claims": claims,
            "limitations": limitations,
        },
        "evidence_status_legend": {
            "qualified": "Bounded stored-snapshot channel claim on the 50-row validation panel.",
            "exploratory": "Fresh-forward or validation-fitted views outside the qualified claim.",
            "negative": "Outcomes where the measured effect is absent or contradicts the hypothesis.",
            "unavailable": "Evidence the bundle does not contain; absence is shown, never silent.",
        },
        "provenance": {
            "bundle_id": manifest["bundle_id"],
            "source_config_sha256": manifest["source_config_sha256"],
            "bundle_config_sha256": manifest["bundle_config_sha256"],
            "population": manifest["population"],
            "split": manifest["split"],
            "counts": manifest["counts"],
            "files": manifest["files"],
            "excluded_files": manifest["_excluded_files"],
            "report_bindings": provenance["report_bindings"],
            "code_bindings": provenance["code_bindings"],
            "runtime": provenance["runtime"],
            "privacy_card": provenance["privacy_card"],
            "source_provenance": provenance["source_provenance"],
            "tokenizer": tokenizer_prov,
        },
        "court": {
            "thresholds": agg_json["court_thresholds"],
            "fit_split": agg_json["fit_split"],
            "confound": "Twin critics share a teacher target and are not semantically independent.",
            "docket": docket_rows,
        },
        "parse_health": {
            "explanations_by_kind": expl_state_counts,
            "trace_descriptions_usable": all(bool(t["description_usable"])
                                             for t in tables["token_trajectories"]),
            "e1_av_parse": reports["e1_av"]["parse"],
            "almanac_parse_health": e0["null_text_almanac"]["parse_health"],
            "poetry_usable_fraction": round6(poetry_meta["reports"]["poetry_describe"]["usable_fraction"]),
        },
        "drift": {
            "card": e0["drift_card"],
            "e5_per_doc": [{
                "row_id": d["row_id"],
                "one_minus_cos": round6(d["one_minus_cos"]),
                "relative_l2": round6(d["relative_l2"]),
            } for d in reports["e5"]["stored_final_position_drift"]],
        },
        "magnitude": {
            "claim_boundary": e0["magnitude_card"]["claim_boundary"],
            "publication_status": e0["magnitude_card"]["publication_status"],
            "fit": e0["magnitude_card"]["fit"],
        },
        "null_text": {
            "scope": e0["null_text_almanac"]["scope"],
            "row_count": e0["null_text_almanac"]["row_count"],
            "real_enriched_words": e0["null_text_almanac"]["real_enriched_words"],
            "zero_enriched_words": e0["null_text_almanac"]["zero_enriched_words"],
            "e1_av_losses": reports["e1_av"]["losses"],
            "e2_mean_loss": reports["e2"]["mean_loss"],
            "e2_paired": reports["e2"]["paired"],
            "backfill_note": "Null-text token log-probabilities require the E2 backfill before final use.",
        },
        "negative_results": negative_results,
        "poetry_status": {
            "claim_scope": "fresh_forward_exploratory",
            "pipeline_passed": all(r["passed"] for r in poetry_meta["reports"].values()),
            "pipeline_note": "Pipeline `passed` means phase completeness gates only; it is not "
                             "evidence that a scientific planning hypothesis passed.",
            "config_sha256": poetry_meta["config_sha256"],
        },
    }


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def write_shard(out_dir: Path, rel: str, payload: dict) -> dict:
    target = out_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json(payload).encode("utf-8")
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)
    return {
        "path": rel,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "schema_version": payload.get("schema_version", SCHEMA_VERSION),
    }


def copy_report(out_dir: Path, rel: str, source: Path) -> dict:
    target = out_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return {
        "path": rel,
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
        "schema_version": "source_copy.v1",
    }


def build(repo_root: Path, out_dir: Path, tokenizer_path: Path) -> dict:
    bundle_dir = repo_root / "artifacts/observatory/r33_offline_v1/bundle"
    poetry_dir = repo_root / "artifacts/observatory/r33_poetry_planning_v1"
    check(bundle_dir.is_dir(), f"bundle directory missing: {bundle_dir}")
    check(poetry_dir.is_dir(), f"poetry directory missing: {poetry_dir}")

    print("[1/8] verifying core bundle manifest hashes ...")
    manifest = verify_core_manifest(bundle_dir)
    print(f"      bundle_id {manifest['bundle_id'][:16]}… "
          f"({len(manifest['_verified_files'])} files verified, "
          f"{len(manifest['_excluded_files'])} heavy files excluded by design)")

    print("[2/8] verifying poetry planning reports ...")
    poetry_meta = verify_poetry_pack(poetry_dir)
    print(f"      poetry config {poetry_meta['config_sha256'][:16]}… (6 phases passed)")

    print("[3/8] loading + validating tables ...")
    tables = load_tables(bundle_dir)
    validate_core(tables, manifest)
    agg_json = validate_aggregates(tables, bundle_dir)
    print("      counts, joins, control groups, and aggregates verified")

    print("[4/8] verifying tokenizer against bundle token texts ...")
    tok_info = verify_tokenizer(tokenizer_path, tables["token_trajectories"])

    reports = {name: load_json(bundle_dir / "assets" / "reports" / f"{name}.json")
               for name in BUNDLE_REPORT_NAMES}
    provenance = load_json(bundle_dir / "provenance.json")
    claim_ledger = load_json(bundle_dir / "assets" / "claim_ledger.json")
    print("[5/8] validating matched online-RL reports ...")
    online_rl, online_rl_paths = load_matched_online_rl(repo_root)
    print("      122 matched validation families; SFT 0.309055 -> RL 0.224386 dMSE")

    print("[6/8] building shards ...")
    shards: dict[str, dict] = {}
    shards["rows.json"] = build_rows_shard(tables)
    shards["channel.json"] = build_channel_shard(tables, agg_json, reports, online_rl)
    shards["rewrites.json"] = build_rewrites_shard(tables)
    geometry_shard = build_geometry_shard(tables, reports)
    shards["trace.json"] = build_trace_shard(tables, reports, tok_info["tokenizer"])
    shards["poetry.json"] = build_poetry_shard(poetry_dir, poetry_meta)
    bench_index, bench_rows = build_bench_shards(tables, geometry_shard,
                                                 tok_info["tokenizer"], manifest)
    shards["bench/index.json"] = bench_index
    for row_id, shard in bench_rows.items():
        shards[f"bench/row-{row_id}.json"] = shard
    shards["audit.json"] = build_audit_shard(tables, manifest, provenance, claim_ledger,
                                             agg_json, reports, poetry_meta,
                                             tok_info["provenance"], online_rl)

    print("[7/8] writing shards ...")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    files = [write_shard(out_dir, rel, payload) for rel, payload in sorted(shards.items())]
    for name in BUNDLE_REPORT_NAMES:
        files.append(copy_report(out_dir, f"reports/{name}.json",
                                 bundle_dir / "assets" / "reports" / f"{name}.json"))
    for phase in POETRY_PHASES:
        files.append(copy_report(out_dir, f"reports/{phase}.json",
                                 poetry_dir / "reports" / f"{phase}_report.json"))
    for name, source_path in sorted(online_rl_paths.items()):
        files.append(copy_report(out_dir, f"reports/roundtrip_{name}_384.json", source_path))

    print("[8/8] writing dashboard manifest ...")
    dashboard_manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source": {
            "bundle_id": manifest["bundle_id"],
            "source_config_sha256": manifest["source_config_sha256"],
            "bundle_config_sha256": manifest["bundle_config_sha256"],
            "population": manifest["population"],
            "split": manifest["split"],
            "counts": manifest["counts"],
            "manifest_sha256": sha256_file(bundle_dir / "observatory_manifest.json"),
            "excluded_files": manifest["_excluded_files"],
        },
        "poetry": {
            "config_sha256": poetry_meta["config_sha256"],
            "phases_passed": POETRY_PHASES,
        },
        "online_rl": {
            "status": online_rl["status"],
            "row_count": online_rl["row_count"],
            "independent_family_count": online_rl["independent_family_count"],
            "max_new_tokens": online_rl["max_new_tokens"],
            "generation_protocol_sha256": online_rl["generation_protocol_sha256"],
            "report_sha256": {
                name: sha256_file(source_path)
                for name, source_path in sorted(online_rl_paths.items())
            },
        },
        "tokenizer": tok_info["provenance"],
        "counts": {
            "rows": 50, "interventions": 7434, "metrics": len(tables["metrics"]),
            "behavior": 888, "trajectories": 400, "explanations": 900,
            "court": 1200, "shapley": 200, "retrieval": 100,
            "geometry": len(tables["geometry"]),
            "poetry_cases": 8, "poetry_positions": 104, "poetry_samples": 624,
            "poetry_interventions": 30,
        },
        "files": sorted(files, key=lambda f: f["path"]),
    }
    payload = json.dumps(dashboard_manifest, sort_keys=True, indent=1).encode("utf-8")
    (out_dir / "manifest.json").write_bytes(payload)
    total = sum(f["bytes"] for f in files)
    print(f"done: {len(files) + 1} files, {total / 1e6:.2f} MB of shards -> {out_dir}")
    return dashboard_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "public" / "data")
    parser.add_argument(
        "--tokenizer", type=Path,
        default=default_root / "runs/introspection/ar-r27-datagen-dryrun-20260528T230649Z/nano_tokenizer/tokenizer.json")
    args = parser.parse_args()
    try:
        build(args.repo_root, args.out, args.tokenizer)
    except BuildError as exc:
        print(f"BUILD FAILED (fail-closed): {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
