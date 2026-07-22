import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "nano_queue_chain.py"
    spec = importlib.util.spec_from_file_location("nano_queue_chain", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoQueueChainTests(unittest.TestCase):
    def test_chain_environment_pins_child_queue_source(self):
        module = load_script()

        env = module.chain_environment(pathlib.Path("/immutable/source"))

        self.assertEqual(env["NANO_QUEUE_CODE_ROOT"], "/immutable/source")

    def test_queue_state_requires_every_item_to_complete(self):
        module = load_script()

        self.assertEqual(
            module.queue_state({"items": [{"status": "complete"}, {"status": "complete"}]}),
            "complete",
        )
        self.assertEqual(
            module.queue_state({"items": [{"status": "complete"}, {"status": "running"}]}),
            "waiting",
        )
        self.assertEqual(
            module.queue_state({"items": [{"status": "training"}, {"status": "eval_running"}]}),
            "waiting",
        )
        self.assertEqual(
            module.queue_state({"items": [{"status": "complete"}, {"status": "failed"}]}),
            "failed",
        )
        self.assertEqual(
            module.queue_state(
                {"items": [{"status": "complete"}, {"status": "blocked_missing_dataset"}]}
            ),
            "failed",
        )

    def test_queue_state_rejects_empty_or_unknown_statuses(self):
        module = load_script()

        with self.assertRaisesRegex(module.QueueChainError, "non-empty"):
            module.queue_state({"items": []})
        with self.assertRaisesRegex(module.QueueChainError, "unknown"):
            module.queue_state({"items": [{"status": "mystery"}]})

    def test_queue_state_can_select_one_prerequisite_item(self):
        module = load_script()
        queue = {
            "items": [
                {"name": "historical-failure", "status": "failed"},
                {"name": "selected-confirmation", "status": "complete"},
            ]
        }

        self.assertEqual(
            module.queue_state(queue, item_name="selected-confirmation"),
            "complete",
        )
        self.assertEqual(
            module.queue_state(queue, item_name="historical-failure"),
            "failed",
        )
        with self.assertRaisesRegex(module.QueueChainError, "not found"):
            module.queue_state(queue, item_name="missing-item")

    def test_queue_state_can_require_an_explicit_quality_gate(self):
        module = load_script()

        self.assertEqual(
            module.queue_state(
                {"items": [{"status": "complete", "gate_passed": True}]},
                require_gate_pass=True,
            ),
            "complete",
        )
        self.assertEqual(
            module.queue_state(
                {"items": [{"status": "complete", "gate_passed": False}]},
                require_gate_pass=True,
            ),
            "failed",
        )
        self.assertEqual(
            module.queue_state(
                {"items": [{"status": "complete"}]},
                require_gate_pass=True,
            ),
            "failed",
        )
        self.assertEqual(
            module.queue_state(
                {"items": [{"status": "eval_running"}]},
                require_gate_pass=True,
            ),
            "waiting",
        )

    def test_build_watch_command_formats_integral_poll_seconds_for_watcher(self):
        module = load_script()

        command = module.build_watch_command(
            python_bin="/venv/bin/python",
            next_queue=pathlib.Path("configs/next.yaml"),
            poll_seconds=30.0,
        )

        self.assertEqual(
            command,
            [
                "/venv/bin/python",
                "scripts/nano_ar_layer_sweep.py",
                "watch",
                "configs/next.yaml",
                "--run-until-empty",
                "--poll-seconds",
                "30",
            ],
        )

    def test_build_watch_command_rejects_fractional_poll_seconds(self):
        module = load_script()

        with self.assertRaisesRegex(module.QueueChainError, "whole number"):
            module.build_watch_command(
                python_bin="/venv/bin/python",
                next_queue=pathlib.Path("configs/next.yaml"),
                poll_seconds=0.5,
            )

    def test_build_watch_command_supports_rl_queue(self):
        module = load_script()

        command = module.build_watch_command(
            python_bin="/venv/bin/python",
            next_queue=pathlib.Path("configs/next.yaml"),
            poll_seconds=30.0,
            queue_type="rl",
        )

        self.assertEqual(
            command,
            [
                "/venv/bin/python",
                "scripts/nano_rl_queue.py",
                "configs/next.yaml",
                "--run-until-empty",
                "--poll-seconds",
                "30",
            ],
        )

    def test_build_watch_command_supports_ar_hpo_queue(self):
        module = load_script()

        self.assertEqual(
            module.queue_type({"schema_version": "nano_ar_hpo_queue.v1"}),
            "ar_hpo",
        )
        command = module.build_watch_command(
            python_bin="/venv/bin/python",
            next_queue=pathlib.Path("configs/ar-next.yaml"),
            poll_seconds=30.0,
            queue_type="ar_hpo",
        )

        self.assertEqual(
            command,
            [
                "/venv/bin/python",
                "scripts/nano_ar_hpo_queue.py",
                "configs/ar-next.yaml",
                "--run-until-empty",
                "--poll-seconds",
                "30",
            ],
        )

    def test_build_watch_command_supports_av_probe_queue(self):
        module = load_script()

        self.assertEqual(
            module.queue_type({"schema_version": "nano_av_probe_queue.v1"}),
            "av_probe",
        )
        command = module.build_watch_command(
            python_bin="/venv/bin/python",
            next_queue=pathlib.Path("configs/av-next.yaml"),
            poll_seconds=30.0,
            queue_type="av_probe",
        )

        self.assertEqual(
            command,
            [
                "/venv/bin/python",
                "scripts/nano_av_probe_queue.py",
                "configs/av-next.yaml",
                "--run-until-empty",
            ],
        )

    def test_build_watch_command_supports_roundtrip_queue(self):
        module = load_script()

        self.assertEqual(
            module.queue_type({"schema_version": "nano_roundtrip_queue.v1"}),
            "roundtrip",
        )
        command = module.build_watch_command(
            python_bin="/venv/bin/python",
            next_queue=pathlib.Path("configs/roundtrip-next.yaml"),
            poll_seconds=30.0,
            queue_type="roundtrip",
        )

        self.assertEqual(
            command,
            [
                "/venv/bin/python",
                "scripts/nano_roundtrip_queue.py",
                "run-loop",
                "configs/roundtrip-next.yaml",
                "--sleep-seconds",
                "30",
            ],
        )


if __name__ == "__main__":
    unittest.main()
