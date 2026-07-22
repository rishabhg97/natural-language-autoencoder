import importlib.util
import pathlib
import tempfile
import textwrap

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prefix_queue_runs_config_and_requires_declared_artifacts(monkeypatch):
    queue = load_script("nano_prefix_dataset_queue")
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        code = root / "code"
        scripts = code / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "nano_prefix_dataset_config.py").write_text("pass\n")
        config = code / "config.yaml"
        config.write_text("schema_version: nano_prefix_dataset_pipeline.v1\n")
        expected = root / "critic" / "critic_initialization.json"
        queue_path = root / "queue.yaml"
        queue_path.write_text(
            textwrap.dedent(
                f"""
                schema_version: nano_prefix_dataset_queue.v1
                defaults:
                  code_root: {code}
                  python: /venv/bin/python
                items:
                  - name: prepare-independent
                    config: {config}
                    status: pending
                    run_dir: {root / 'run'}
                    expected_artifacts:
                      - {expected}
                """
            )
        )

        def fake_run(spec):
            expected.parent.mkdir(parents=True)
            expected.write_text("{}")

        monkeypatch.setattr(queue, "_run_logged", fake_run)
        result = queue.process_next(queue_path)
        saved = yaml.safe_load(queue_path.read_text())

    assert result["status"] == "complete"
    assert saved["items"][0]["status"] == "complete"
    assert saved["items"][0]["gate_passed"] is True


def test_prefix_queue_dry_run_does_not_serialize_inherited_secrets(monkeypatch):
    queue = load_script("nano_prefix_dataset_queue")
    monkeypatch.setenv("WANDB_API_KEY", "must-not-appear")
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        code = root / "code"
        scripts = code / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "nano_prefix_dataset_config.py").write_text("pass\n")
        config = code / "config.yaml"
        config.write_text("schema_version: nano_prefix_dataset_pipeline.v1\n")
        queue_path = root / "queue.yaml"
        queue_path.write_text(
            textwrap.dedent(
                f"""
                schema_version: nano_prefix_dataset_queue.v1
                defaults:
                  code_root: {code}
                  environment:
                    WANDB_MODE: offline
                items:
                  - name: dry-run
                    config: {config}
                    expected_artifacts: [{root / 'output.json'}]
                """
            )
        )

        result = queue.process_next(queue_path, dry_run=True)

    assert result["status"] == "dry_run"
    assert "env" not in result
    assert result["environment"]["WANDB_MODE"] == "offline"
    assert "must-not-appear" not in repr(result)


def test_prefix_queue_fails_closed_when_artifact_is_missing(monkeypatch):
    queue = load_script("nano_prefix_dataset_queue")
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        code = root / "code"
        scripts = code / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "nano_prefix_dataset_config.py").write_text("pass\n")
        config = code / "config.yaml"
        config.write_text("schema_version: nano_prefix_dataset_pipeline.v1\n")
        queue_path = root / "queue.yaml"
        queue_path.write_text(
            textwrap.dedent(
                f"""
                schema_version: nano_prefix_dataset_queue.v1
                defaults:
                  code_root: {code}
                items:
                  - name: missing-output
                    config: {config}
                    expected_artifacts: [{root / 'missing.json'}]
                """
            )
        )
        monkeypatch.setattr(queue, "_run_logged", lambda spec: None)
        try:
            queue.process_next(queue_path)
        except queue.PrefixQueueError:
            pass
        else:
            raise AssertionError("missing artifact must fail the queue")
        saved = yaml.safe_load(queue_path.read_text())

    assert saved["items"][0]["status"] == "failed"
    assert saved["items"][0]["gate_passed"] is False


def test_prefix_queue_fails_closed_on_artifact_hash_mismatch(monkeypatch):
    queue = load_script("nano_prefix_dataset_queue")
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        code = root / "code"
        scripts = code / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "nano_prefix_dataset_config.py").write_text("pass\n")
        config = code / "config.yaml"
        config.write_text("schema_version: nano_prefix_dataset_pipeline.v1\n")
        artifact = root / "critic_initialization.json"
        queue_path = root / "queue.yaml"
        queue_path.write_text(
            textwrap.dedent(
                f"""
                schema_version: nano_prefix_dataset_queue.v1
                defaults:
                  code_root: {code}
                items:
                  - name: wrong-hash
                    config: {config}
                    expected_artifacts:
                      - path: {artifact}
                        sha256: "{'0' * 64}"
                """
            )
        )

        def fake_run(_spec):
            artifact.write_text("different")

        monkeypatch.setattr(queue, "_run_logged", fake_run)
        try:
            queue.process_next(queue_path)
        except queue.PrefixQueueError as exc:
            assert "sha256 mismatch" in str(exc)
        else:
            raise AssertionError("artifact hash mismatch must fail the queue")
        saved = yaml.safe_load(queue_path.read_text())

    assert saved["items"][0]["status"] == "failed"


def test_prefix_queue_canonical_json_hash_can_ignore_declared_transient(monkeypatch):
    queue = load_script("nano_prefix_dataset_queue")
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        code = root / "code"
        scripts = code / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "nano_prefix_dataset_config.py").write_text("pass\n")
        config = code / "config.yaml"
        config.write_text("schema_version: nano_prefix_dataset_pipeline.v1\n")
        artifact = root / "critic_initialization.json"
        expected_payload = {
            "value_head": {"before_sha256": "old", "after_sha256": "final"},
            "router": {"after_sha256": "router-final"},
        }
        canonical = dict(expected_payload)
        canonical["value_head"] = dict(expected_payload["value_head"])
        del canonical["value_head"]["before_sha256"]
        import hashlib
        import json

        expected_sha = hashlib.sha256(
            json.dumps(
                canonical,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        queue_path = root / "queue.yaml"
        queue_path.write_text(
            textwrap.dedent(
                f"""
                schema_version: nano_prefix_dataset_queue.v1
                defaults:
                  code_root: {code}
                items:
                  - name: semantic-hash
                    config: {config}
                    expected_artifacts:
                      - path: {artifact}
                        canonical_json_sha256: {expected_sha}
                        ignore_json_paths: [value_head.before_sha256]
                """
            )
        )

        def fake_run(_spec):
            changed = dict(expected_payload)
            changed["value_head"] = dict(expected_payload["value_head"])
            changed["value_head"]["before_sha256"] = "new-random-preimage"
            artifact.write_text(json.dumps(changed))

        monkeypatch.setattr(queue, "_run_logged", fake_run)
        result = queue.process_next(queue_path)

    assert result["status"] == "complete"


def test_prefix_queue_canonical_json_hash_rejects_final_state_change(monkeypatch):
    queue = load_script("nano_prefix_dataset_queue")
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        code = root / "code"
        scripts = code / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "nano_prefix_dataset_config.py").write_text("pass\n")
        config = code / "config.yaml"
        config.write_text("schema_version: nano_prefix_dataset_pipeline.v1\n")
        artifact = root / "critic_initialization.json"
        queue_path = root / "queue.yaml"
        queue_path.write_text(
            textwrap.dedent(
                f"""
                schema_version: nano_prefix_dataset_queue.v1
                defaults:
                  code_root: {code}
                items:
                  - name: changed-final-state
                    config: {config}
                    expected_artifacts:
                      - path: {artifact}
                        canonical_json_sha256: {'0' * 64}
                        ignore_json_paths: [value_head.before_sha256]
                """
            )
        )

        def fake_run(_spec):
            artifact.write_text(
                '{"value_head":{"before_sha256":"random",'
                '"after_sha256":"wrong"}}'
            )

        monkeypatch.setattr(queue, "_run_logged", fake_run)
        try:
            queue.process_next(queue_path)
        except queue.PrefixQueueError as exc:
            assert "canonical_json_sha256" in str(exc)
        else:
            raise AssertionError("changed final state must fail the queue")


def test_queue_chain_recognizes_prefix_dataset_queue():
    chain = load_script("nano_queue_chain")
    queue = {"schema_version": "nano_prefix_dataset_queue.v1", "items": []}
    assert chain.queue_type(queue) == "prefix_dataset"
    command = chain.build_watch_command(
        python_bin="/venv/bin/python",
        next_queue=pathlib.Path("/queues/prefix.yaml"),
        poll_seconds=30,
        queue_type="prefix_dataset",
    )
    assert command == [
        "/venv/bin/python",
        "scripts/nano_prefix_dataset_queue.py",
        "/queues/prefix.yaml",
        "--run-until-empty",
        "--poll-seconds",
        "30",
    ]


def test_checked_in_independent_critic_rebuild_is_queueable_and_offline():
    queue = load_script("nano_prefix_dataset_queue")
    path = (
        ROOT
        / "configs"
        / "nano_data"
        / "publication"
        / "r33_independent_critic_rebuild_queue.yaml"
    )
    payload = queue.load_queue(path)
    item = payload["items"][0]

    assert payload["defaults"]["environment"]["WANDB_MODE"] == "offline"
    assert item["status"] == "pending"
    assert "independent" in item["name"]
    assert len(item["expected_artifacts"]) == 3
