import importlib.util
import pathlib
import tempfile
import unittest
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoRoundtripGenerationWorkerTests(unittest.TestCase):
    def test_generation_shard_worker_does_not_write_parent_report(self):
        module = load_script("eval_nano_av_ar_roundtrip_gate")
        records = [{"row_index": 1}]
        written: list[tuple[pathlib.Path, object]] = []
        original_parse_args = module.parse_args
        original_generate = module.generate_roundtrip_records
        original_protocol = module.build_generation_protocol
        original_provenance = module.build_generation_provenance
        original_validate_protocols = module.validate_generated_record_protocols
        original_validate_provenance = module.validate_generated_record_provenance
        original_write_json = module.write_json
        try:
            with tempfile.TemporaryDirectory() as tmp:
                generated = pathlib.Path(tmp) / "shard.jsonl"
                report = pathlib.Path(tmp) / "generation_report.json"
                module.parse_args = lambda _argv: SimpleNamespace(
                    generation_only=True,
                    generated_jsonl=generated,
                    report_json=report,
                    generation_workers=1,
                    stream_generated=False,
                    resume_generated=False,
                    generation_shard_index=1,
                    generation_shard_count=4,
                    require_generation_protocol_match=True,
                )
                module.generate_roundtrip_records = lambda _args, **_kwargs: records
                module.build_generation_protocol = lambda _args: {"seed": 1}
                module.build_generation_provenance = lambda _args: {"model": "test"}
                module.validate_generated_record_protocols = lambda *_args, **_kwargs: {
                    "seed": 1
                }
                module.validate_generated_record_provenance = lambda *_args, **_kwargs: {
                    "model": "test"
                }
                module.write_json = lambda path, payload: written.append((path, payload))

                self.assertEqual(module.main([]), 0)
        finally:
            module.parse_args = original_parse_args
            module.generate_roundtrip_records = original_generate
            module.build_generation_protocol = original_protocol
            module.build_generation_provenance = original_provenance
            module.validate_generated_record_protocols = original_validate_protocols
            module.validate_generated_record_provenance = original_validate_provenance
            module.write_json = original_write_json

        self.assertEqual(written, [])

    def test_worker_concurrency_cap_launches_deterministic_shard_waves(self):
        module = load_script("eval_nano_av_ar_roundtrip_gate")
        events: list[tuple[str, int]] = []

        class FakeProcess:
            def __init__(self, command, **_kwargs):
                self.index = int(command[0])
                events.append(("start", self.index))

            def wait(self):
                events.append(("wait", self.index))
                return 0

        original_popen = module.subprocess.Popen
        original_command = module.build_generation_worker_command
        original_merge = module.merge_generated_shards
        try:
            module.subprocess.Popen = FakeProcess
            module.build_generation_worker_command = lambda _args, shard_index, **_kwargs: [
                str(shard_index)
            ]
            module.merge_generated_shards = lambda _paths, _output: []
            with tempfile.TemporaryDirectory() as tmp:
                records = module.generate_roundtrip_records_with_workers(
                    SimpleNamespace(
                        generation_workers=5,
                        generation_max_parallel_workers=2,
                        resume_generated=False,
                        stream_generated=False,
                        generation_worker_devices=[],
                    ),
                    pathlib.Path(tmp) / "generated.jsonl",
                )
        finally:
            module.subprocess.Popen = original_popen
            module.build_generation_worker_command = original_command
            module.merge_generated_shards = original_merge

        self.assertEqual(records, [])
        self.assertEqual(
            events,
            [
                ("start", 0),
                ("start", 1),
                ("wait", 0),
                ("wait", 1),
                ("start", 2),
                ("start", 3),
                ("wait", 2),
                ("wait", 3),
                ("start", 4),
                ("wait", 4),
            ],
        )


if __name__ == "__main__":
    unittest.main()
