from __future__ import annotations

from pathlib import Path

import pytest

from observatory.queue import (
    build_command,
    build_item_command,
    build_module_command,
    build_script_command,
    queue_environment,
    run_queue,
)


def test_build_command_is_config_driven() -> None:
    command = build_command(
        python_bin="/venv/bin/python",
        code_root=Path("/code"),
        config_path=Path("/code/config.yaml"),
        phase="canary-av",
    )
    assert command == [
        "/venv/bin/python",
        "-m",
        "observatory.run_model_batches",
        "--config",
        "/code/config.yaml",
        "--phase",
        "canary-av",
    ]


def test_build_command_rejects_unknown_phase() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        build_command(
            python_bin="python",
            code_root=Path("/code"),
            config_path=Path("/code/config.yaml"),
            phase="mystery",
        )


def test_build_command_supports_full_lattice_phase() -> None:
    command = build_command(
        python_bin="python",
        code_root=Path("/code"),
        config_path=Path("/code/config.yaml"),
        phase="lattice-full",
    )
    assert command[-1] == "lattice-full"


def test_build_command_supports_trace_description_phase() -> None:
    command = build_command(
        python_bin="python",
        code_root=Path("/code"),
        config_path=Path("/code/config.yaml"),
        phase="trace-describe",
    )
    assert command[-1] == "trace-describe"


def test_build_command_supports_full_functional_phase() -> None:
    command = build_command(
        python_bin="python",
        code_root=Path("/code"),
        config_path=Path("/code/config.yaml"),
        phase="functional-full",
    )
    assert command[-1] == "functional-full"


def test_build_script_command_is_config_driven() -> None:
    command = build_script_command(
        python_bin="python",
        code_root=Path("/code"),
        config_path=Path("/code/bundle.yaml"),
        script="scripts/nano_viz_verify_bundle.py",
    )
    assert command == [
        "python",
        "/code/scripts/nano_viz_verify_bundle.py",
        "--config",
        "/code/bundle.yaml",
    ]


def test_build_module_command_is_config_driven() -> None:
    command = build_module_command(
        python_bin="python",
        config_path=Path("/code/bundle.yaml"),
        module="observatory.verify_bundle",
    )
    assert command == [
        "python",
        "-m",
        "observatory.verify_bundle",
        "--config",
        "/code/bundle.yaml",
    ]


@pytest.mark.parametrize(
    "module", ["scripts.verify", "observatory../verify", "observatory.bad-name"]
)
def test_build_module_command_rejects_unsafe_module(module: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        build_module_command(
            python_bin="python",
            config_path=Path("/code/bundle.yaml"),
            module=module,
        )


@pytest.mark.parametrize("script", ["/tmp/unsafe.py", "../unsafe.py", "scripts/no_suffix"])
def test_build_script_command_rejects_unsafe_path(script: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        build_script_command(
            python_bin="python",
            code_root=Path("/code"),
            config_path=Path("/code/bundle.yaml"),
            script=script,
        )


def test_build_item_command_requires_exactly_one_runner() -> None:
    arguments = {
        "python_bin": "python",
        "code_root": Path("/code"),
        "config_path": Path("/code/config.yaml"),
    }
    with pytest.raises(ValueError, match="exactly one"):
        build_item_command(item={}, **arguments)
    with pytest.raises(ValueError, match="exactly one"):
        build_item_command(
            item={"phase": "canary-av", "script": "scripts/verify.py"},
            **arguments,
        )
    with pytest.raises(ValueError, match="exactly one"):
        build_item_command(
            item={"phase": "canary-av", "module": "observatory.verify_bundle"},
            **arguments,
        )


def test_queue_environment_includes_scripts_and_vendored_nla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/inherited")
    environment = queue_environment(Path("/code"))
    assert environment["PYTHONPATH"].split(":") == [
        "/code/scripts",
        "/code/external/natural_language_autoencoders",
        "/code",
        "/inherited",
    ]
    assert environment["WANDB_MODE"] == "offline"
    assert environment["TOKENIZERS_PARALLELISM"] == "false"


def test_queue_environment_applies_scalar_config_overrides() -> None:
    environment = queue_environment(
        Path("/code"),
        {"NLA_ENABLE_EXPERIMENTAL_NEMOTRON_CACHE": 1, "CUSTOM_FLAG": True},
    )
    assert environment["NLA_ENABLE_EXPERIMENTAL_NEMOTRON_CACHE"] == "1"
    assert environment["CUSTOM_FLAG"] == "True"


def test_run_queue_reexecutes_complete_item_after_config_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import subprocess
    import yaml

    code_root = tmp_path / "code"
    (code_root / "scripts").mkdir(parents=True)
    config = tmp_path / "config.yaml"
    config.write_text("selection:\n  seed: 2\n")
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": "nano_viz_queue_state.v1",
                "items": {
                    "canary": {
                        "status": "complete",
                        "config_sha256": "stale",
                    }
                },
            }
        )
    )
    queue = tmp_path / "queue.yaml"
    queue.write_text(
        yaml.safe_dump(
            {
                "defaults": {
                    "code_root": str(code_root),
                    "config": str(config),
                    "python": "python",
                    "state_json": str(state),
                    "log_dir": str(tmp_path / "logs"),
                },
                "items": [{"name": "canary", "phase": "canary-av"}],
            }
        )
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_queue(queue)
    assert result["status"] == "complete"
    assert len(calls) == 1
    assert result["items"]["canary"]["config_sha256"] != "stale"
