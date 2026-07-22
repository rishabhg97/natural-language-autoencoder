import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import nano_domain_eval  # noqa: E402
import nano_domain_eval_queue as queue  # noqa: E402


def test_sharded_phase_config_prefixes_match_public_config_names():
    assert queue.SHARDED_PHASE_CONFIG_PREFIX == {
        "extract": "extraction",
        "describe": "description",
        "behavior": "behavior",
    }


def test_record_phase_clears_stale_error(tmp_path):
    config = {"paths": {"output_root": str(tmp_path)}}
    state = {
        "status": "running",
        "phases": {"describe": {"status": "failed", "error": "old failure"}},
    }
    queue._record_phase(config, state, "describe", "running")
    assert state["phases"]["describe"]["status"] == "running"
    assert "error" not in state["phases"]["describe"]


def test_reuse_through_extract_validates_artifact_hashes(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    activations = tmp_path / "activations.jsonl"
    manifest.write_text('{"row_id":"row"}\n')
    activations.write_text('{"row_id":"row","activation_vector":[0]}\n')
    manifest_report = tmp_path / "manifest_report.json"
    activation_report = tmp_path / "activation_report.json"
    manifest_report.write_text(
        json.dumps(
            {
                "passed": True,
                "manifest": {"sha256": nano_domain_eval.sha256_file(manifest)},
            }
        )
    )
    activation_report.write_text(
        json.dumps(
            {
                "passed": True,
                "manifest_sha256": nano_domain_eval.sha256_file(manifest),
                "activations": {"sha256": nano_domain_eval.sha256_file(activations)},
                "boundary": 33,
                "capture_backend": "truncated_causal_prefix_per_anchor",
                "pre_condition_invariance": {"passed": True},
            }
        )
    )
    av_hf = tmp_path / "av_hf"
    av_hf.mkdir()
    (av_hf / "config.json").write_text("{}\n")
    config = {
        "paths": {
            "manifest_jsonl": str(manifest),
            "manifest_report_json": str(manifest_report),
            "activations_jsonl": str(activations),
            "activation_report_json": str(activation_report),
        },
        "models": {"av_hf": str(av_hf)},
        "evaluation": {"boundary": 33},
    }
    state = {"phases": {phase: {"status": "pending"} for phase in queue.PHASES}}
    queue._reuse_through_extract(config, state)
    assert all(
        state["phases"][phase]["status"] == "complete"
        for phase in ("build-manifest", "prepare-av", "extract")
    )


def test_reuse_through_extract_allows_fresh_av_preparation(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    activations = tmp_path / "activations.jsonl"
    manifest.write_text('{"row_id":"row"}\n')
    activations.write_text('{"row_id":"row","activation_vector":[0]}\n')
    manifest_report = tmp_path / "manifest_report.json"
    activation_report = tmp_path / "activation_report.json"
    manifest_report.write_text(
        json.dumps(
            {
                "passed": True,
                "manifest": {"sha256": nano_domain_eval.sha256_file(manifest)},
            }
        )
    )
    activation_report.write_text(
        json.dumps(
            {
                "passed": True,
                "manifest_sha256": nano_domain_eval.sha256_file(manifest),
                "activations": {
                    "sha256": nano_domain_eval.sha256_file(activations)
                },
                "boundary": 33,
                "capture_backend": "truncated_causal_prefix_per_anchor",
                "pre_condition_invariance": {"passed": True},
            }
        )
    )
    config = {
        "paths": {
            "manifest_jsonl": str(manifest),
            "manifest_report_json": str(manifest_report),
            "activations_jsonl": str(activations),
            "activation_report_json": str(activation_report),
        },
        "models": {"av_hf": str(tmp_path / "not-prepared")},
        "evaluation": {"boundary": 33},
    }
    state = {"phases": {phase: {"status": "pending"} for phase in queue.PHASES}}

    queue._reuse_through_extract(config, state)

    assert state["phases"]["build-manifest"]["status"] == "complete"
    assert state["phases"]["extract"]["status"] == "complete"
    assert state["phases"]["prepare-av"]["status"] == "pending"


def test_no_resume_with_extract_reuse_skips_reused_phases(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    activations = tmp_path / "activations.jsonl"
    manifest.write_text('{"row_id":"row"}\n')
    activations.write_text('{"row_id":"row","activation_vector":[0]}\n')
    manifest_report = tmp_path / "manifest_report.json"
    activation_report = tmp_path / "activation_report.json"
    manifest_report.write_text(
        json.dumps(
            {
                "passed": True,
                "manifest": {"sha256": nano_domain_eval.sha256_file(manifest)},
            }
        )
    )
    activation_report.write_text(
        json.dumps(
            {
                "passed": True,
                "manifest_sha256": nano_domain_eval.sha256_file(manifest),
                "activations": {"sha256": nano_domain_eval.sha256_file(activations)},
                "boundary": 33,
                "capture_backend": "truncated_causal_prefix_per_anchor",
                "pre_condition_invariance": {"passed": True},
            }
        )
    )
    av_hf = tmp_path / "av_hf"
    av_hf.mkdir()
    (av_hf / "config.json").write_text("{}\n")
    code_root = tmp_path / "code"
    code_root.mkdir()
    output_root = tmp_path / "output"
    config = {
        "paths": {
            "code_root": str(code_root),
            "output_root": str(output_root),
            "manifest_jsonl": str(manifest),
            "manifest_report_json": str(manifest_report),
            "activations_jsonl": str(activations),
            "activation_report_json": str(activation_report),
        },
        "models": {"av_hf": str(av_hf)},
        "evaluation": {"boundary": 33},
        "checkpoint_prepare": {"cleanup_after_queue": False},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text("placeholder: true\n")
    monkeypatch.setattr(queue.nano_domain_eval, "load_config", lambda _: config)
    commands = []
    monkeypatch.setattr(
        queue,
        "_run_command",
        lambda command, **_: commands.append(tuple(command)),
    )

    state = queue.run_queue(
        config_path,
        resume=False,
        reuse_through="extract",
    )

    assert [command[2] for command in commands] == [
        "describe",
        "behavior",
        "analyze",
    ]
    assert state["status"] == "complete"


def test_reuse_through_describe_validates_description_hash(tmp_path, monkeypatch):
    activations = tmp_path / "activations.jsonl"
    descriptions = tmp_path / "descriptions.jsonl"
    activations.write_text('{"row_id":"row"}\n')
    descriptions.write_text('{"row_id":"row","controls":{}}\n')
    report = tmp_path / "description_report.json"
    report.write_text(
        json.dumps(
            {
                "passed": True,
                "activation_sha256": nano_domain_eval.sha256_file(activations),
                "descriptions": {
                    "sha256": nano_domain_eval.sha256_file(descriptions)
                },
            }
        )
    )
    config = {
        "paths": {
            "activations_jsonl": str(activations),
            "descriptions_jsonl": str(descriptions),
            "description_report_json": str(report),
        }
    }
    state = {"phases": {phase: {"status": "pending"} for phase in queue.PHASES}}
    monkeypatch.setattr(queue, "_reuse_through_extract", lambda *_: None)

    queue._reuse_through_describe(config, state)

    assert state["phases"]["describe"]["status"] == "complete"
    assert state["phases"]["describe"]["reuse_validation"] == "passed"
