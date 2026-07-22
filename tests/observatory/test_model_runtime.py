from __future__ import annotations

from pathlib import Path

import pytest

from observatory.common import ObservatoryConfigError
from observatory.model_runtime import (
    _sample_next_tokens,
    baseline_wake_logits,
    causal_token_logprobs,
    compare_score_batches,
    functional_wake_metrics,
    greedy_generate_patched_full_prefix,
    greedy_generate_unpatched,
    hf_checkpoint_complete,
    rowwise_reconstruction_metrics,
    select_trajectory_positions,
    selected_rows,
    topk_distribution,
    write_prediction_parquet,
)
from observatory.run_model_batches import (
    _prediction_equivalence,
    _reusable_report,
    _resolved_lattice_cells,
    _validation_rows,
)


ROOT = Path(__file__).resolve().parents[2]


def test_mamba_rmsnorm_fallback_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    torch = pytest.importorskip("torch")
    nla_root = ROOT / "external" / "natural_language_autoencoders"
    monkeypatch.syspath_prepend(str(nla_root))
    from nla.remote_code_patches import PatchReport, _patch_rmsnorm_fallback

    source = """import torch
try:
    # Older Nemotron-H checkpoints leave the class import commented out.
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
except ImportError:
    raise ImportError(\"mamba-ssm is required by the Mamba model but cannot be imported\")
"""
    report = PatchReport()
    patched = _patch_rmsnorm_fallback(source, report)
    assert report.rmsnorm_fallback_replacements == 1

    original_import = builtins.__import__

    def reject_mamba(name: str, *args: object, **kwargs: object):
        if name.startswith("mamba_ssm"):
            raise ImportError("mamba intentionally absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_mamba)
    namespace: dict[str, object] = {}
    exec(patched, namespace)
    fallback = namespace["rmsnorm_fn"]

    torch.manual_seed(7)
    x = torch.randn(2, 3, 12, dtype=torch.bfloat16)
    z = torch.randn_like(x)
    weight = torch.randn(12, dtype=torch.bfloat16)
    bias = torch.randn(12, dtype=torch.bfloat16)
    for norm_before_gate in (False, True):
        x_float = x.float()
        z_float = z.float()
        if not norm_before_gate:
            x_float = x_float * torch.nn.functional.silu(z_float)
        grouped = x_float.reshape(2, 3, 3, 4)
        reciprocal_rms = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + 1e-5)
        expected = (grouped * reciprocal_rms).reshape_as(x_float) * weight.float()
        expected = expected + bias.float()
        if norm_before_gate:
            expected = expected * torch.nn.functional.silu(z_float)
        expected = expected.to(x.dtype)

        actual = fallback(
            x,
            weight,
            bias,
            z=z,
            eps=1e-5,
            group_size=4,
            norm_before_gate=norm_before_gate,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_validation_rows_join_bounded_cache_to_full_split(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    validation_path = tmp_path / "validation.parquet"
    pq.write_table(
        pa.table(
            {
                "doc_id": ["doc-a", "doc-b", "doc-c"],
                "n_raw_tokens": [10, 20, 30],
                "response": ["a", "b", "c"],
            }
        ),
        validation_path,
    )
    generated_path = tmp_path / "generated.jsonl"
    generated_path.write_text(
        "\n".join(
            [
                '{"doc_id":"doc-c","n_raw_tokens":30,"row_index":103,"source_row_index":2,"split":"validation"}',
                '{"doc_id":"doc-a","n_raw_tokens":10,"row_index":101,"source_row_index":0,"split":"validation"}',
            ]
        )
        + "\n"
    )
    cache_path = tmp_path / "cache.npz"
    np.savez_compressed(
        cache_path,
        validation__row_indices=np.asarray([101, 103], dtype=np.int64),
        validation__doc_ids=np.asarray(["doc-a", "doc-c"]),
        validation__content_family_ids=np.asarray(["cf-a", "cf-c"]),
    )
    config = {
        "paths": {
            "validation_parquet": str(validation_path),
            "generated_validation_jsonl": str(generated_path),
            "validation_prediction_cache_npz": str(cache_path),
        }
    }

    rows = _validation_rows(config, tmp_path / "config.yaml")

    assert [row["row_index"] for row in rows] == [101, 103]
    assert [row["source_row_index"] for row in rows] == [0, 2]
    assert [row["content_family_id"] for row in rows] == ["cf-a", "cf-c"]


def test_full_prefix_generation_and_wake_use_the_requested_boundary() -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(7, 4)
            self.boundary = torch.nn.Identity()
            self.head = torch.nn.Linear(4, 7, bias=False)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
            hidden = self.boundary(self.embedding(input_ids))
            return SimpleNamespace(logits=self.head(hidden))

    torch.manual_seed(17)
    model = ToyModel().eval()
    prefix = [1, 2, 3]
    replacement = torch.tensor([3.0, -2.0, 1.0, 0.5])
    generated = greedy_generate_patched_full_prefix(
        model=model,
        boundary_module=model.boundary,
        prefix=prefix,
        replacement=replacement,
        max_new_tokens=3,
        eos_token_id=None,
    )
    assert len(generated) == 3
    baseline_generated = greedy_generate_unpatched(
        model=model,
        tokenizer=None,
        prefix=prefix,
        max_new_tokens=3,
        pad_token_id=0,
        eos_token_id=None,
        backend="full_prefix",
    )
    assert len(baseline_generated) == 3
    wake = functional_wake_metrics(
        model=model,
        boundary_module=model.boundary,
        prefix=prefix,
        baseline_continuation=generated,
        replacement=replacement,
        wake_positions=3,
    )
    assert [row["offset"] for row in wake] == [1, 2, 3]
    assert all(np.isfinite(value) for row in wake for value in row.values())
    cached_baseline = baseline_wake_logits(
        model=model,
        prefix=prefix,
        baseline_continuation=generated,
        wake_positions=3,
    )
    cached_wake = functional_wake_metrics(
        model=model,
        boundary_module=model.boundary,
        prefix=prefix,
        baseline_continuation=generated,
        replacement=replacement,
        wake_positions=3,
        baseline_position_logits=cached_baseline,
    )
    assert cached_wake == wake


def test_mamba_rmsnorm_fallback_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    nla_root = ROOT / "external" / "natural_language_autoencoders"
    monkeypatch.syspath_prepend(str(nla_root))
    from nla.remote_code_patches import PatchReport, _patch_rmsnorm_fallback

    source = """try:
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
except ImportError:
    raise ImportError(\"missing\")
"""
    first_report = PatchReport()
    first = _patch_rmsnorm_fallback(source, first_report)
    second_report = PatchReport()
    second = _patch_rmsnorm_fallback(first, second_report)
    assert second == first
    assert second_report.rmsnorm_fallback_replacements == 0


def test_dynamic_module_cache_cleanup_handles_transformers_sanitization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nla_root = ROOT / "external" / "natural_language_autoencoders"
    monkeypatch.syspath_prepend(str(nla_root))
    from nla.remote_code_patches import clear_transformers_dynamic_module_cache

    monkeypatch.setenv("HOME", str(tmp_path))
    cache_root = (
        tmp_path
        / ".cache"
        / "huggingface"
        / "modules"
        / "transformers_modules"
    )
    literal = cache_root / "r33-clean.av"
    sanitized = cache_root / "r33_hyphen_clean_dot_av"
    unrelated = cache_root / "keep-me"
    for directory in (literal, sanitized, unrelated):
        directory.mkdir(parents=True)
        (directory / "modeling.py").write_text("cached")

    removed = clear_transformers_dynamic_module_cache(tmp_path / "r33-clean.av")
    assert set(removed) == {str(literal), str(sanitized)}
    assert unrelated.is_dir()


def test_selected_rows_preserves_requested_order() -> None:
    rows = [{"row_index": 2}, {"row_index": 5}]
    assert selected_rows(rows, ["validation-5", "validation-2"]) == [rows[1], rows[0]]


def test_resolved_lattice_cells_binds_only_parse_usable_tellings() -> None:
    interventions = [
        {
            "cell_id": "identity",
            "family": "identity",
            "state": "ready",
            "text": "teacher",
        },
        {
            "cell_id": "telling",
            "family": "alternate_telling",
            "state": "pending_model_generation",
            "text": None,
        },
    ]
    resolved = _resolved_lattice_cells(
        interventions,
        [
            {
                "cell_id": "telling",
                "parsed": {"usable": True, "explanation": "alternate"},
            }
        ],
    )
    assert [cell["text"] for cell in resolved] == ["teacher", "alternate"]
    assert resolved[1]["state"] == "ready"
    assert interventions[1]["state"] == "pending_model_generation"


def test_resolved_lattice_cells_rejects_unusable_telling() -> None:
    with pytest.raises(ObservatoryConfigError, match="not parse-usable"):
        _resolved_lattice_cells(
            [
                {
                    "cell_id": "telling",
                    "family": "alternate_telling",
                    "state": "pending_model_generation",
                    "text": None,
                }
            ],
            [
                {
                    "cell_id": "telling",
                    "parsed": {"usable": False, "explanation": ""},
                }
            ],
        )


def test_reusable_report_requires_current_config_hash(tmp_path: Path) -> None:
    import json

    from observatory.common import config_fingerprint

    config = {"selection": {"seed": 7}}
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps({"passed": True, "config_sha256": config_fingerprint(config)})
    )
    assert _reusable_report(path, config, force=False) is not None
    assert _reusable_report(path, {"selection": {"seed": 8}}, force=False) is None
    assert _reusable_report(path, config, force=True) is None


def test_selected_rows_fails_closed() -> None:
    with pytest.raises(ObservatoryConfigError, match="missing"):
        selected_rows([{"row_index": 2}], ["validation-3"])


def test_causal_token_logprobs_respects_response_span() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 4, 4), dtype=torch.float32)
    logits[0, 1, 2] = 4.0
    logits[0, 2, 3] = 4.0
    ids = torch.tensor([[0, 1, 2, 3]])
    result = causal_token_logprobs(logits, ids, label_starts=[2], lengths=[4])[0]
    assert result["positions"] == [2, 3]
    assert result["token_ids"] == [2, 3]
    assert result["target_tokens"] == 2
    assert result["loss"] < 0.1


def test_compare_score_batches_reports_maxima() -> None:
    left = [
        {
            "row_index": 1,
            "positions": [7],
            "token_ids": [2],
            "loss": 1.0,
            "logprobs": [-1.0],
        }
    ]
    right = [
        {
            "row_index": 1,
            "positions": [7],
            "token_ids": [2],
            "loss": 1.1,
            "logprobs": [-1.2],
        }
    ]
    report = compare_score_batches(left, right)
    assert report["max_abs_loss_delta"] == pytest.approx(0.1)
    assert report["max_abs_token_logprob_delta"] == pytest.approx(0.2)
    assert report["tokens"] == 1
    assert report["max_token"]["position"] == 7


def test_prediction_equivalence_is_exact_for_equal_arrays() -> None:
    np = pytest.importorskip("numpy")
    values = np.asarray([[1.0, 2.0], [2.0, 1.0]])
    report = _prediction_equivalence(values, values.copy())
    assert report["max_relative_l2"] == 0.0
    assert report["min_cosine"] == pytest.approx(1.0)


def test_rowwise_reconstruction_metrics_has_expected_geometry() -> None:
    np = pytest.importorskip("numpy")
    predictions = np.asarray([[1.0, 0.0], [0.0, 2.0]])
    targets = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    report = rowwise_reconstruction_metrics(predictions, targets)
    assert report["directional_mse"].tolist() == pytest.approx([0.0, 0.0])
    assert report["cosine"].tolist() == pytest.approx([1.0, 1.0])
    assert report["norm_ratio"].tolist() == pytest.approx([1.0, 2.0])


def test_write_prediction_parquet_uses_fixed_fp16_vectors(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "predictions.parquet"
    write_prediction_parquet(
        path,
        [
            {
                "cell_id": "cell-1",
                "row_id": "validation-1",
                "row_index": 1,
                "content_family_id": "family-1",
                "family": "identity",
                "variant": "teacher",
                "depth": "METRIC",
                "critic": "primary",
                "directional_mse": 0.1,
                "raw_mse": 0.2,
                "cosine": 0.95,
                "norm_ratio": 0.8,
                "prediction_vector": [0.0] * 2688,
            }
        ],
    )
    table = pq.read_table(path)
    assert table.num_rows == 1
    assert table.schema.field("prediction_vector").type.list_size == 2688


def test_top_p_sampling_is_seed_reproducible() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    first, _ = _sample_next_tokens(
        logits,
        temperature=0.7,
        top_p=0.95,
        generators=[torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)],
    )
    second, _ = _sample_next_tokens(
        logits,
        temperature=0.7,
        top_p=0.95,
        generators=[torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)],
    )
    assert first.tolist() == second.tolist()


def test_topk_distribution_returns_probabilities() -> None:
    torch = pytest.importorskip("torch")
    result = topk_distribution(torch.tensor([0.0, 1.0, 2.0]), k=2)
    assert result["token_ids"] == [2, 1]
    assert len(result["probabilities"]) == 2
    assert result["probabilities"][0] > result["probabilities"][1]


def test_hf_checkpoint_complete_validates_index_shards(tmp_path: Path) -> None:
    import json

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}})
    )
    assert not hf_checkpoint_complete(tmp_path)
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    assert hf_checkpoint_complete(tmp_path)


def test_select_trajectory_positions_includes_endpoints() -> None:
    positions = select_trajectory_positions(130, minimum_context=32, count=40)
    assert positions[0] == 31
    assert positions[-1] == 129
    assert len(positions) == 40
