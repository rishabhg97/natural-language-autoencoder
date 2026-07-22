import importlib.util
import pathlib
import sys
import tempfile
import textwrap
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS))
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class NanoQueueGateTests(unittest.TestCase):
    def _write_yaml(self, path: pathlib.Path, text: str) -> pathlib.Path:
        path.write_text(textwrap.dedent(text))
        return path

    def test_unblocks_target_when_dependency_is_complete_and_paths_exist(self):
        gate = load_script("nano_queue_gate")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            checkpoint = root / "ckpt"
            report = root / "eval.json"
            checkpoint.mkdir()
            report.write_text("{}")
            dependency_queue = self._write_yaml(
                root / "ar_queue.yaml",
                f"""
                schema_version: nano_ar_hpo_queue.v1
                items:
                - name: ar-smoke
                  config: ar.yaml
                  status: complete
                  expected_checkpoint: {checkpoint}
                  eval_report: {report}
                """,
            )
            target_queue = self._write_yaml(
                root / "av_queue.yaml",
                """
                schema_version: nano_av_probe_queue.v1
                items:
                - name: av-smoke
                  config: av.yaml
                  status: blocked
                """,
            )

            result = gate.unblock_when_ready(
                dependency_queue=dependency_queue,
                dependency_item_name="ar-smoke",
                target_queue=target_queue,
                target_item_name="av-smoke",
                required_fields=["expected_checkpoint", "eval_report"],
                dry_run=False,
                now="2026-06-11T12:00:00Z",
            )
            updated = yaml.safe_load(target_queue.read_text())

        self.assertTrue(result["ready"])
        self.assertTrue(result["changed"])
        self.assertEqual(updated["items"][0]["status"], "pending")
        self.assertEqual(updated["items"][0]["previous_status"], "blocked")
        self.assertEqual(updated["items"][0]["unblocked_at"], "2026-06-11T12:00:00Z")
        self.assertEqual(updated["items"][0]["unblock_dependency_item"], "ar-smoke")

    def test_keeps_target_blocked_when_dependency_is_still_active(self):
        gate = load_script("nano_queue_gate")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            dependency_queue = self._write_yaml(
                root / "ar_queue.yaml",
                """
                schema_version: nano_ar_hpo_queue.v1
                items:
                - name: ar-smoke
                  config: ar.yaml
                  status: training
                """,
            )
            target_queue = self._write_yaml(
                root / "av_queue.yaml",
                """
                schema_version: nano_av_probe_queue.v1
                items:
                - name: av-smoke
                  config: av.yaml
                  status: blocked
                """,
            )

            result = gate.unblock_when_ready(
                dependency_queue=dependency_queue,
                dependency_item_name="ar-smoke",
                target_queue=target_queue,
                target_item_name="av-smoke",
                required_fields=["expected_checkpoint", "eval_report"],
                dry_run=False,
                now="2026-06-11T12:00:00Z",
            )
            updated = yaml.safe_load(target_queue.read_text())

        self.assertFalse(result["ready"])
        self.assertFalse(result["changed"])
        self.assertIn("dependency status is training, not complete", result["reasons"])
        self.assertEqual(updated["items"][0]["status"], "blocked")

    def test_requires_declared_dependency_paths_to_exist(self):
        gate = load_script("nano_queue_gate")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            dependency_queue = self._write_yaml(
                root / "ar_queue.yaml",
                f"""
                schema_version: nano_ar_hpo_queue.v1
                items:
                - name: ar-smoke
                  config: ar.yaml
                  status: complete
                  expected_checkpoint: {root / "missing-ckpt"}
                  eval_report: {root / "missing-report.json"}
                """,
            )
            target_queue = self._write_yaml(
                root / "av_queue.yaml",
                """
                schema_version: nano_av_probe_queue.v1
                items:
                - name: av-smoke
                  config: av.yaml
                  status: blocked
                """,
            )

            result = gate.unblock_when_ready(
                dependency_queue=dependency_queue,
                dependency_item_name="ar-smoke",
                target_queue=target_queue,
                target_item_name="av-smoke",
                required_fields=["expected_checkpoint", "eval_report"],
                dry_run=False,
                now="2026-06-11T12:00:00Z",
            )
            updated = yaml.safe_load(target_queue.read_text())

        self.assertFalse(result["ready"])
        self.assertFalse(result["changed"])
        self.assertTrue(any("missing dependency path expected_checkpoint" in reason for reason in result["reasons"]))
        self.assertTrue(any("missing dependency path eval_report" in reason for reason in result["reasons"]))
        self.assertEqual(updated["items"][0]["status"], "blocked")

    def test_requires_json_bool_evidence_before_unblocking(self):
        gate = load_script("nano_queue_gate")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            checkpoint = root / "ckpt"
            eval_report = root / "eval.json"
            baseline_report = root / "baseline.json"
            checkpoint.mkdir()
            eval_report.write_text("{}")
            baseline_report.write_text('{"gate": {"passed": false}}')
            dependency_queue = self._write_yaml(
                root / "ar_queue.yaml",
                f"""
                schema_version: nano_ar_hpo_queue.v1
                items:
                - name: ar-smoke
                  config: ar.yaml
                  status: complete
                  expected_checkpoint: {checkpoint}
                  eval_report: {eval_report}
                """,
            )
            target_queue = self._write_yaml(
                root / "av_queue.yaml",
                """
                schema_version: nano_av_probe_queue.v1
                items:
                - name: av-smoke
                  config: av.yaml
                  status: blocked
                """,
            )

            result = gate.unblock_when_ready(
                dependency_queue=dependency_queue,
                dependency_item_name="ar-smoke",
                target_queue=target_queue,
                target_item_name="av-smoke",
                required_fields=["expected_checkpoint", "eval_report"],
                required_json_bools=[(baseline_report, "gate.passed")],
                dry_run=False,
                now="2026-06-11T12:00:00Z",
            )
            baseline_report.write_text('{"gate": {"passed": true}}')
            ready = gate.unblock_when_ready(
                dependency_queue=dependency_queue,
                dependency_item_name="ar-smoke",
                target_queue=target_queue,
                target_item_name="av-smoke",
                required_fields=["expected_checkpoint", "eval_report"],
                required_json_bools=[(baseline_report, "gate.passed")],
                dry_run=True,
                now="2026-06-11T12:00:00Z",
            )

        self.assertFalse(result["ready"])
        self.assertTrue(any("gate.passed" in reason for reason in result["reasons"]))
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["would_set_status"], "pending")


if __name__ == "__main__":
    unittest.main()
