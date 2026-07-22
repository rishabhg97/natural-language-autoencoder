from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS))
    sys.modules[name] = module
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class NanoAVGenerationTests(unittest.TestCase):
    def test_plan_generation_jobs_creates_stable_row_control_keys(self):
        gen = load_script("nano_av_generation")
        rows = [
            {"row_index": 10, "source_row_index": 0, "split": "validation", "doc_id": "doc-10"},
            {"row_index": 11, "source_row_index": 1, "split": "test", "doc_id": "doc-11"},
        ]
        controls_by_row = {
            10: {"real": "v10", "mean": "m"},
            11: {"real": "v11", "mean": "m"},
        }

        jobs = gen.plan_generation_jobs(
            rows_by_index={int(row["row_index"]): row for row in rows},
            row_indices=[10, 11],
            controls_by_row=controls_by_row,
            controls_requested=["real", "mean"],
            target_explanations={10: "target 10", 11: "target 11"},
        )

        self.assertEqual([job.job_key for job in jobs], [
            "validation:10:real",
            "validation:10:mean",
            "test:11:real",
            "test:11:mean",
        ])
        self.assertEqual(jobs[0].control_vector, "v10")
        self.assertEqual(jobs[-1].target_explanation, "target 11")

    def test_jsonl_resume_reads_only_completed_job_keys(self):
        gen = load_script("nano_av_generation")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "generated.jsonl"

            gen.append_generation_record(path, {"job_key": "validation:1:real", "status": "complete"})
            gen.append_generation_record(path, {"job_key": "validation:1:mean", "status": "failed"})
            gen.append_generation_record(path, {"job_key": "validation:2:real"})

            completed = gen.load_completed_job_keys(path)

        self.assertEqual(completed, {"validation:1:real", "validation:2:real"})

    def test_greedy_generate_with_cache_uses_one_token_steps_after_initial_forward(self):
        gen = load_script("nano_av_generation")
        fake_torch = FakeTorch()
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            model = FakeCachedModel(next_ids=[101, 102, 2], cache_values=["kv0", "kv1", "kv2"])
            tokenizer = FakeTokenizer(eos_token_id=2)

            result = gen.greedy_generate_with_cache(
                model,
                tokenizer,
                initial_embeds=FakeTensor(length=7, dtype="bf16"),
                attention_mask=FakeTensor(length=7, dtype="long"),
                max_new_tokens=8,
            )
        finally:
            if original_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original_torch

        self.assertEqual([call["input_length"] for call in model.calls], [7, 1, 1])
        self.assertEqual([call["past_key_values"] for call in model.calls], [None, "kv0", "kv1"])
        self.assertEqual(result.token_ids, [101, 102, 2])
        self.assertTrue(result.cache_used)

    def test_greedy_generate_with_cache_can_disable_eos_stopping(self):
        gen = load_script("nano_av_generation")
        fake_torch = FakeTorch()
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            model = FakeCachedModel(
                next_ids=[101, 2, 102], cache_values=["kv0", "kv1", "kv2"]
            )
            tokenizer = FakeTokenizer(eos_token_id=2)

            result = gen.greedy_generate_with_cache(
                model,
                tokenizer,
                initial_embeds=FakeTensor(length=7, dtype="bf16"),
                attention_mask=FakeTensor(length=7, dtype="long"),
                max_new_tokens=3,
                eos_token_id=None,
            )
        finally:
            if original_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original_torch

        self.assertEqual(result.token_ids, [101, 2, 102])
        self.assertTrue(result.cache_used)

    def test_greedy_generate_with_cache_falls_back_when_model_returns_no_cache(self):
        gen = load_script("nano_av_generation")
        fake_torch = FakeTorch()
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            model = FakeCachedModel(next_ids=[101, 2], cache_values=[None, None])
            tokenizer = FakeTokenizer(eos_token_id=2)

            result = gen.greedy_generate_with_cache(
                model,
                tokenizer,
                initial_embeds=FakeTensor(length=7, dtype="bf16"),
                attention_mask=FakeTensor(length=7, dtype="long"),
                max_new_tokens=8,
            )
        finally:
            if original_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original_torch

        self.assertEqual([call["input_length"] for call in model.calls], [7, 8])
        self.assertFalse(result.cache_used)
        self.assertEqual(result.fallback_reason, "missing_past_key_values")

    def test_greedy_generate_batch_with_cache_decodes_independent_sequences(self):
        gen = load_script("nano_av_generation")
        fake_torch = FakeTorch()
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            model = FakeBatchCachedModel(
                next_ids_by_step=[
                    [101, 201],
                    [102, 2],
                    [2, 2],
                ],
                cache_values=["kv0", "kv1", "kv2"],
            )
            tokenizer = FakeTokenizer(eos_token_id=2)

            results = gen.greedy_generate_batch_with_cache(
                model,
                tokenizer,
                initial_embeds=FakeTensor(length=7, batch_size=2, dtype="bf16"),
                attention_mask=FakeTensor(length=7, batch_size=2, dtype="long"),
                max_new_tokens=8,
            )
        finally:
            if original_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original_torch

        self.assertEqual([call["input_length"] for call in model.calls], [7, 1, 1])
        self.assertEqual([call["past_key_values"] for call in model.calls], [None, "kv0", "kv1"])
        self.assertEqual([result.token_ids for result in results], [[101, 102, 2], [201, 2]])
        self.assertTrue(all(result.cache_used for result in results))

    def test_greedy_generate_batch_initializes_nemotron_hybrid_cache(self):
        gen = load_script("nano_av_generation")
        fake_torch = FakeTorch()
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            model = FakeNemotronCachedModel(
                next_ids_by_step=[
                    [101, 201],
                    [102, 2],
                    [2, 2],
                ],
            )
            tokenizer = FakeTokenizer(eos_token_id=2)

            results = gen.greedy_generate_batch_with_cache(
                model,
                tokenizer,
                initial_embeds=FakeTensor(length=7, batch_size=2, dtype="bf16"),
                attention_mask=FakeTensor(length=7, batch_size=2, dtype="long"),
                max_new_tokens=8,
            )
        finally:
            if original_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original_torch

        self.assertEqual([call["input_length"] for call in model.calls], [7, 1, 1])
        self.assertTrue(all(call["cache_params"] is model.calls[0]["cache_params"] for call in model.calls))
        self.assertEqual(
            [call["cache_position"] for call in model.calls],
            [list(range(7)), [7], [8]],
        )
        self.assertEqual([result.token_ids for result in results], [[101, 102, 2], [201, 2]])
        self.assertTrue(all(result.cache_used for result in results))

    def test_greedy_generate_batch_full_prefix_recomputes_complete_sequence(self):
        gen = load_script("nano_av_generation")
        fake_torch = FakeTorch()
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            model = FakeBatchCachedModel(
                next_ids_by_step=[
                    [101, 201],
                    [102, 202],
                    [2, 2],
                ],
                cache_values=["ignored0", "ignored1", "ignored2"],
            )
            tokenizer = FakeTokenizer(eos_token_id=2)

            results = gen.greedy_generate_batch_full_prefix(
                model,
                tokenizer,
                initial_embeds=FakeTensor(length=7, batch_size=2, dtype="bf16"),
                attention_mask=FakeTensor(length=7, batch_size=2, dtype="long"),
                max_new_tokens=8,
            )
        finally:
            if original_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original_torch

        self.assertEqual([call["input_length"] for call in model.calls], [7, 8, 9])
        self.assertEqual([call["past_key_values"] for call in model.calls], [None, None, None])
        self.assertEqual([result.token_ids for result in results], [[101, 102, 2], [201, 202, 2]])
        self.assertTrue(all(not result.cache_used for result in results))

    def test_real_nemotron_class_falls_back_from_unverified_cache(self):
        gen = load_script("nano_av_generation")
        fake_torch = FakeTorch()
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            model = NemotronHForCausalLM(next_ids_by_step=[[101], [102], [2]])
            tokenizer = FakeTokenizer(eos_token_id=2)

            results = gen.greedy_generate_batch_with_cache(
                model,
                tokenizer,
                initial_embeds=FakeTensor(length=7, batch_size=1, dtype="bf16"),
                attention_mask=FakeTensor(length=7, batch_size=1, dtype="long"),
                max_new_tokens=8,
            )
        finally:
            if original_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original_torch

        self.assertEqual([call["input_length"] for call in model.calls], [7, 8, 9])
        self.assertFalse(results[0].cache_used)
        self.assertEqual(results[0].fallback_reason, "nemotron_cache_equivalence_unverified")


class FakeTensor:
    def __init__(
        self,
        *,
        length: int,
        dtype: str = "float",
        device: str = "cuda",
        token_id: int | None = None,
        token_ids: list[int] | None = None,
        batch_size: int = 1,
    ):
        self.length = length
        self.dtype = dtype
        self.device = device
        self.token_id = token_id
        self.token_ids = token_ids
        self.batch_size = batch_size
        self.shape = (batch_size, length, 1)

    def __getitem__(self, item):
        if isinstance(item, int) and self.token_ids is not None:
            return FakeTensor(length=1, token_id=self.token_ids[item])
        return self

    def to(self, *, dtype=None, device=None):
        return FakeTensor(
            length=self.length,
            dtype=dtype or self.dtype,
            device=device or self.device,
            token_id=self.token_id,
            token_ids=self.token_ids,
            batch_size=self.batch_size,
        )

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        if self.token_ids is None:
            raise TypeError("fake tensor has no vector tokens")
        return list(self.token_ids)

    def item(self):
        if self.token_id is None:
            raise TypeError("fake tensor has no scalar token")
        return self.token_id

    def __int__(self):
        return int(self.item())


class FakeLogits:
    def __init__(self, token_id: int | None = None, token_ids: list[int] | None = None):
        self.token_id = token_id
        self.token_ids = token_ids

    def __getitem__(self, item):
        return self


class FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")

    def argmax(self, logits, dim=-1):
        if logits.token_ids is not None:
            return FakeTensor(length=1, token_ids=list(logits.token_ids), batch_size=len(logits.token_ids))
        return FakeTensor(length=1, token_id=logits.token_id)

    def cat(self, tensors, dim=0):
        if dim != 1:
            raise AssertionError(f"unexpected cat dim {dim}")
        first = tensors[0]
        return FakeTensor(length=sum(t.length for t in tensors), dtype=first.dtype, device=first.device, batch_size=first.batch_size)

    def ones(self, shape, device=None, dtype=None):
        return FakeTensor(length=int(shape[1]), dtype=dtype or "long", device=device or "cuda", batch_size=int(shape[0]))

    def arange(self, start, end=None, device=None):
        if end is None:
            start, end = 0, start
        values = list(range(int(start), int(end)))
        return FakeTensor(length=len(values), token_ids=values, device=device or "cuda")

    class no_grad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False


class FakeCachedModel:
    def __init__(self, *, next_ids: list[int], cache_values: list[object]):
        self.next_ids = list(next_ids)
        self.cache_values = list(cache_values)
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(
            {
                "input_length": kwargs["inputs_embeds"].shape[1],
                "past_key_values": kwargs.get("past_key_values"),
                "use_cache": kwargs.get("use_cache"),
            }
        )
        token_id = self.next_ids.pop(0)
        cache = self.cache_values.pop(0)
        return types.SimpleNamespace(logits=FakeLogits(token_id), past_key_values=cache)

    def get_input_embeddings(self):
        def embed(token_tensor):
            return FakeTensor(length=1, dtype="bf16", token_id=token_tensor.token_id, batch_size=token_tensor.batch_size)

        return embed


class FakeBatchCachedModel:
    def __init__(self, *, next_ids_by_step: list[list[int]], cache_values: list[object]):
        self.next_ids_by_step = list(next_ids_by_step)
        self.cache_values = list(cache_values)
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(
            {
                "input_length": kwargs["inputs_embeds"].shape[1],
                "past_key_values": kwargs.get("past_key_values"),
                "use_cache": kwargs.get("use_cache"),
                "batch_size": kwargs["inputs_embeds"].shape[0],
            }
        )
        token_ids = self.next_ids_by_step.pop(0)
        cache = self.cache_values.pop(0)
        return types.SimpleNamespace(logits=FakeLogits(token_ids=token_ids), past_key_values=cache)

    def get_input_embeddings(self):
        def embed(token_tensor):
            return FakeTensor(length=1, dtype="bf16", batch_size=token_tensor.batch_size)

        return embed


class HybridMambaAttentionDynamicCache:
    def __init__(self, config, batch_size, dtype, device=None):
        self.config = config
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device


class FakeNemotronCachedModel:
    config = object()

    def __init__(self, *, next_ids_by_step: list[list[int]]):
        self.next_ids_by_step = list(next_ids_by_step)
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask,
        use_cache,
        cache_params=None,
        cache_position=None,
    ):
        self.calls.append(
            {
                "input_length": inputs_embeds.shape[1],
                "cache_params": cache_params,
                "cache_position": (
                    list(cache_position.token_ids) if cache_position is not None else None
                ),
                "use_cache": use_cache,
            }
        )
        token_ids = self.next_ids_by_step.pop(0)
        return types.SimpleNamespace(
            logits=FakeLogits(token_ids=token_ids),
            cache_params=cache_params,
        )

    def get_input_embeddings(self):
        def embed(token_tensor):
            return FakeTensor(length=1, dtype="bf16", batch_size=token_tensor.batch_size)

        return embed


class NemotronHForCausalLM(FakeNemotronCachedModel):
    pass


class FakeTokenizer:
    def __init__(self, *, eos_token_id: int):
        self.eos_token_id = eos_token_id

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in token_ids)


if __name__ == "__main__":
    unittest.main()
