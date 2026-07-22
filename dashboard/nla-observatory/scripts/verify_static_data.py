#!/usr/bin/env python3
"""Fail-closed verifier for the generated NLA Observatory dashboard data.

Checks, in order:
1. dashboard manifest exists and every generated file hash/byte-count matches;
2. the recorded source bundle_id / config hashes still match the local raw
   evidence (re-verified from the raw bundle manifest);
3. shard-level counts and referential joins (rows <- channel/rewrites/trace/
   bench/poetry/audit) hold;
4. every shard number spot-checked here is finite;
5. determinism: rebuilding into a temp directory reproduces identical bytes
   for every shard (manifest generated_at excluded).

Exit code 0 = verified; 1 = any failure (fail closed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_static_data as builder  # noqa: E402


class VerifyError(RuntimeError):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise VerifyError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict:
    check(path.is_file(), f"missing generated file: {path}")
    return json.loads(path.read_text())


def walk_numbers(obj, ctx: str) -> None:
    if isinstance(obj, float):
        check(math.isfinite(obj), f"non-finite number in {ctx}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            walk_numbers(v, f"{ctx}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:200000]):
            walk_numbers(v, f"{ctx}[{i}]")


def verify(data_dir: Path, repo_root: Path, tokenizer: Path, *, determinism: bool) -> None:
    manifest = load(data_dir / "manifest.json")
    check(manifest["schema_version"] == builder.SCHEMA_VERSION,
          "dashboard manifest schema mismatch")

    print("[1/5] verifying generated file hashes ...")
    listed = {f["path"] for f in manifest["files"]}
    for entry in manifest["files"]:
        target = data_dir / entry["path"]
        check(target.is_file(), f"generated file missing: {entry['path']}")
        check(target.stat().st_size == entry["bytes"], f"size mismatch: {entry['path']}")
        check(sha256_file(target) == entry["sha256"], f"hash mismatch: {entry['path']}")
    on_disk = {str(p.relative_to(data_dir)) for p in data_dir.rglob("*")
               if p.is_file() and p.name != "manifest.json"}
    check(on_disk == listed, f"unlisted/missing files: {on_disk ^ listed}")

    print("[2/5] re-verifying source evidence binding ...")
    bundle_dir = repo_root / "artifacts/observatory/r33_offline_v1/bundle"
    raw_manifest = builder.verify_core_manifest(bundle_dir)
    check(raw_manifest["bundle_id"] == manifest["source"]["bundle_id"],
          "source bundle_id changed since build")
    check(sha256_file(bundle_dir / "observatory_manifest.json")
          == manifest["source"]["manifest_sha256"], "raw manifest hash changed")
    poetry_meta = builder.verify_poetry_pack(
        repo_root / "artifacts/observatory/r33_poetry_planning_v1")
    check(poetry_meta["config_sha256"] == manifest["poetry"]["config_sha256"],
          "poetry config hash changed since build")
    online_rl, online_rl_paths = builder.load_matched_online_rl(repo_root)
    check(online_rl["generation_protocol_sha256"]
          == manifest["online_rl"]["generation_protocol_sha256"],
          "matched online-RL protocol changed since build")
    check(manifest["online_rl"]["report_sha256"] == {
        name: sha256_file(path) for name, path in sorted(online_rl_paths.items())
    }, "matched online-RL report hash changed since build")

    print("[3/5] verifying shard counts and joins ...")
    rows = load(data_dir / "rows.json")["rows"]
    check(len(rows) == 50, "rows shard != 50 rows")
    row_ids = {r["row_id"] for r in rows}
    rows_by_id = {r["row_id"]: r for r in rows}
    check(len(row_ids) == 50, "duplicate row ids in rows shard")

    channel = load(data_dir / "channel.json")
    matched_rl = channel["matched_online_rl"]
    check(matched_rl["status"] == "validation_only_matched",
          "matched online-RL evidence lost its validation-only status")
    check(matched_rl["row_count"] == 122
          and matched_rl["independent_family_count"] == 122,
          "matched online-RL population drifted")
    check(matched_rl["max_new_tokens"] == 384,
          "matched online-RL generation budget drifted")
    check(matched_rl["generation_protocol_sha256"]
          == builder.MATCHED_RL_PROTOCOL_SHA256,
          "matched online-RL protocol hash drifted")
    check(matched_rl["sft"]["roundtrip_nmse"] == 0.309055,
          "matched SFT round-trip headline drifted")
    check(matched_rl["rl"]["roundtrip_nmse"] == 0.224386,
          "matched RL round-trip headline drifted")
    check(matched_rl["rl"]["roundtrip_nmse"]
          < matched_rl["sft"]["roundtrip_nmse"],
          "matched RL no longer improves on SFT")
    check({c["key"] for c in matched_rl["rl"]["controls"]}
          == {"av_shuffled", "av_zero", "av_mean", "av_none"},
          "matched RL controls are incomplete")
    check(matched_rl["rl"]["parse"]["closed_count"] == 121,
          "matched RL parse-close count drifted")
    check(len(channel["identity"]) == 100, "channel identity != 100")
    check(len(channel["retrieval"]) == 100, "channel retrieval != 100")
    check(sum(len(v) for v in channel["tellings"].values()) == 400, "channel tellings != 400")
    check(sum(len(v) for v in channel["occlusion"].values()) == 3752, "channel occlusion != 3752")
    check(sum(len(v) for v in channel["truncation"].values()) == 500, "channel truncation != 500")
    check(len(channel["shapley"]) == 50, "channel shapley != 50 rows")
    check(len(channel["aggregates"]) == 22, "channel aggregates != 22")
    for coll in (channel["identity"], channel["retrieval"]):
        for item in coll:
            check(item["row_id"] in row_ids, f"orphan row {item['row_id']} in channel")

    rewrites = load(data_dir / "rewrites.json")
    check(len(rewrites["cells"]) == 600, "rewrites cells != 600")
    check(len(rewrites["identity"]) == 50, "rewrites identity != 50")
    for cell in rewrites["cells"]:
        check(cell["row_id"] in row_ids, f"orphan rewrite row {cell['row_id']}")
        check(set(cell["metrics"]) == {"primary", "independent"},
              f"rewrite cell missing critic {cell['cell_id']}")

    trace = load(data_dir / "trace.json")
    check(len(trace["docs"]) == 10, "trace docs != 10")
    check(all(len(d["positions"]) == 40 for d in trace["docs"]), "trace positions != 40/doc")
    check(trace["claim_scope"] == "fresh_forward_exploratory", "trace claim scope wrong")
    check(trace["shuffled_control"]["available"] is False,
          "trace shuffled control availability must be explicit")
    trace_positions = [p for doc in trace["docs"] for p in doc["positions"]]
    exact_contexts = [p for p in trace_positions if p["source_alignment"] == "exact"]
    unavailable_contexts = [p for p in trace_positions if p["source_alignment"] == "unavailable"]
    check(len(exact_contexts) == trace["source_alignment"]["exact_positions"] == 360,
          "trace exact source-context count drifted")
    check(len(unavailable_contexts) == trace["source_alignment"]["unavailable_positions"] == 40,
          "trace unavailable source-context count drifted")
    check(all(p["source_token"] for p in exact_contexts),
          "aligned trace position has blank source token")
    check(all(not p["source_before"] and not p["source_after"] for p in unavailable_contexts),
          "unaligned trace position must not contain guessed source context")
    for doc in trace["docs"]:
        source_text = rows_by_id[doc["row_id"]]["source_text"]
        for position in doc["positions"]:
            start = position["source_char_start"]
            end = position["source_char_end"]
            if position["source_alignment"] == "exact":
                check(isinstance(start, int) and isinstance(end, int),
                      "aligned trace position is missing integer character offsets")
                check(0 <= start < end <= len(source_text),
                      "aligned trace character offsets are outside source text")
                check(source_text[start:end] == position["source_token"],
                      "aligned trace character span does not reproduce source token")
            else:
                check(start is None and end is None,
                      "unaligned trace position must not contain character offsets")

    poetry = load(data_dir / "poetry.json")
    check(len(poetry["cases"]) == 8, "poetry cases != 8")
    check(sum(len(c["samples"]) for c in poetry["cases"]) == 624, "poetry samples != 624")
    check(sum(len(c["interventions"]) for c in poetry["cases"]) == 30,
          "poetry interventions != 30")
    check(sum(1 for c in poetry["cases"] if c["reconstruction"]) == 5,
          "poetry editable cases != 5")
    check(poetry["aggregates"]["mean_anchor_lift"] == 0.03125, "poetry anchor lift drifted")
    check(poetry["aggregates"]["edited_alternate_hit_rate"] == 0.0,
          "poetry negative steering result must be preserved")
    check(poetry["aggregates"]["cases_with_baseline_target_rhyme"] == 1,
          "poetry baseline rhyme count must be preserved")

    bench_index = load(data_dir / "bench" / "index.json")
    check(len(bench_index["rows"]) == 50, "bench index rows != 50")
    check(len(bench_index["behavior_rows"]) == 24, "bench behavior rows != 24")
    check(bench_index["banner"]["total_cells"] == 7434, "bench banner cell count")
    lanes = {"edit", "paraphrase_placebo", "random_edit"}
    for cg_id, cg in bench_index["control_groups"].items():
        check(set(cg["cells"]) == lanes, f"control group {cg_id} incomplete")
    total_cells = 0
    for row_id in bench_index["behavior_rows"]:
        shard = load(data_dir / "bench" / f"row-{row_id}.json")
        check(len(shard["cells"]) == 37, f"bench row {row_id} cells != 37")
        total_cells += len(shard["cells"])
        for cell in shard["cells"]:
            check(len(cell["wake"]) == 16, f"wake != 16 in {cell['cell_id']}")
            check(len(cell["topk"]["original"]) == 10, "topk original != 10")
            check(len(cell["topk"]["patched"]) == 10, "topk patched != 10")
    check(total_cells == 888, "bench behavior cells != 888")

    audit = load(data_dir / "audit.json")
    check(len(audit["court"]["docket"]) == 100, "audit docket != 100")
    check(audit["claim_ledger"]["claims"]["stored_snapshot_channel"] == "qualified",
          "claim ledger drifted")
    check(audit["claim_ledger"]["claims"]["matched_online_rl_roundtrip"]
          == "validation_only_confirmatory",
          "matched online-RL claim boundary drifted")
    check(len(audit["negative_results"]) >= 3, "negative results must be preserved")
    check(audit["poetry_status"]["pipeline_note"].startswith("Pipeline `passed`"),
          "poetry pipeline-vs-science note missing")

    print("[4/5] checking finiteness of shard numbers ...")
    for name in ["channel.json", "rewrites.json", "trace.json",
                 "poetry.json", "audit.json"]:
        walk_numbers(load(data_dir / name), name)

    if determinism:
        print("[5/5] determinism: rebuilding into a temp dir and diffing bytes ...")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_out = Path(tmp) / "data"
            builder.build(repo_root, tmp_out, tokenizer)
            for entry in manifest["files"]:
                a = (data_dir / entry["path"]).read_bytes()
                b = (tmp_out / entry["path"]).read_bytes()
                check(a == b, f"non-deterministic shard: {entry['path']}")
    else:
        print("[5/5] determinism check skipped (--skip-determinism)")

    print("VERIFIED: all generated data matches the raw evidence, fail-closed checks pass.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).resolve().parents[1] / "public" / "data")
    parser.add_argument(
        "--tokenizer", type=Path,
        default=default_root / "runs/introspection/ar-r27-datagen-dryrun-20260528T230649Z/nano_tokenizer/tokenizer.json")
    parser.add_argument("--skip-determinism", action="store_true")
    args = parser.parse_args()
    try:
        verify(args.data, args.repo_root, args.tokenizer,
               determinism=not args.skip_determinism)
    except (VerifyError, builder.BuildError) as exc:
        print(f"VERIFY FAILED (fail-closed): {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
