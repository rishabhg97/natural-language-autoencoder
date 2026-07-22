import importlib.util
import pathlib
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_runner():
    stub = types.ModuleType("nano_av_materialize_splits")
    stub.materialize_splits = lambda *args, **kwargs: None
    previous = sys.modules.get("nano_av_materialize_splits")
    sys.modules["nano_av_materialize_splits"] = stub
    try:
        path = ROOT / "scripts" / "nano_av_runner.py"
        spec = importlib.util.spec_from_file_location("nano_av_runner_validation", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("nano_av_materialize_splits", None)
        else:
            sys.modules["nano_av_materialize_splits"] = previous


def base_spec(*, objective: str = "av_sft") -> dict:
    paths = {
        "code_root": "/workspace/interp/code/nano30b-nla-pilot-current",
        "miles_root": "/workspace/interp/code/miles-051cd15",
        "model_id": "/workspace/interp/models/nano-30b-a3b-bf16-hf",
    }
    if objective == "ar_sft":
        paths["critic_init_model_id"] = "/workspace/interp/outputs/nano30b-nla-pilot/critic"
        paths["input_ar_sft"] = "/tmp/ar_sft.parquet"
    else:
        paths["input_av_sft"] = "/tmp/av_sft.parquet"
    training = {
        "objective": objective,
        "backend": "miles_fsdp2",
        "epochs": 1,
        "global_batch_size": 8,
        "micro_batch_size": 1,
        "rollout_batch_size": 8,
        "lr": 1e-5,
        "grad_norm_policy": "clip",
    }
    if objective == "av_sft":
        training["injection_scale"] = 75
    return {
        "run": {
            "name": "unit",
            "experiment_class": "small-smoke",
            "output_root": "/tmp/nano-out",
            "wandb_mode": "offline",
        },
        "paths": paths,
        "dataset": {
            "row_limit": 96,
            "split_mode": "doc",
            "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "materialize_splits": True,
            "final_batch_policy": "pad_with_train_duplicates",
        },
        "training": training,
        "checkpoint": {"save_interval": 1, "keep_last": 1, "save_enabled": True},
        "eval": {"controls": ["real"]},
    }


class NanoAVRunnerValidationTests(unittest.TestCase):
    def test_allows_ar_padded_critic_batching_without_legacy_acknowledgement(self):
        runner = load_runner()
        spec = base_spec(objective="ar_sft")
        spec["training"]["micro_batch_size"] = 4

        self.assertIs(runner.validate_spec(spec), spec)

    def test_allows_ar_packed_critic_with_explicit_acknowledgement(self):
        runner = load_runner()
        spec = base_spec(objective="ar_sft")
        spec["training"]["micro_batch_size"] = 4
        spec["training"]["allow_packed_critic_training"] = True

        self.assertIs(runner.validate_spec(spec), spec)

    def test_rejects_dynamic_budget_smaller_than_sequence_cap_by_default(self):
        runner = load_runner()
        spec = base_spec(objective="av_sft")
        spec["training"].update(
            {
                "use_dynamic_batch_size": True,
                "max_tokens_per_gpu": 512,
                "max_sequence_tokens": 1152,
            }
        )

        with self.assertRaises(runner.SpecValidationError) as caught:
            runner.validate_spec(spec)

        self.assertIn("max_tokens_per_gpu must be >= training.max_sequence_tokens", str(caught.exception))

    def test_allows_oversized_dynamic_batch_with_explicit_acknowledgement(self):
        runner = load_runner()
        spec = base_spec(objective="av_sft")
        spec["training"].update(
            {
                "use_dynamic_batch_size": True,
                "max_tokens_per_gpu": 512,
                "max_sequence_tokens": 1152,
                "allow_oversized_dynamic_batch": True,
            }
        )

        self.assertIs(runner.validate_spec(spec), spec)

    def test_allows_content_component_split_mode(self):
        runner = load_runner()
        spec = base_spec(objective="av_sft")
        spec["dataset"]["split_mode"] = "content_component"

        self.assertIs(runner.validate_spec(spec), spec)

    def test_rejects_timing_debug_on_complete_performance(self):
        runner = load_runner()
        spec = base_spec(objective="av_sft")
        spec["run"]["experiment_class"] = "complete-performance"
        spec["dataset"].update(
            {
                "row_limit": 90000,
                "fractions": {"train": 0.9, "validation": 0.05, "test": 0.05},
            }
        )
        spec["training"].update({"global_batch_size": 192, "rollout_batch_size": 192, "timing_debug": True})
        spec["checkpoint"].update({"require_optimizer_state_for_hero": True})

        with self.assertRaises(runner.SpecValidationError) as caught:
            runner.validate_spec(spec)

        self.assertIn("complete-performance cannot enable training.timing_debug", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
