from types import SimpleNamespace

import json

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file

from external.natural_language_autoencoders.tools.convert_fsdp_to_hf import (
    _build_hf_model_state,
    _convert_with_origin_safetensors_layout,
    _patch_config_for_skeleton_init,
    _strip_best_raw_prefix,
    _strip_best_prefix,
)


def test_patch_nemotron_h_config_adds_missing_moe_latent_size() -> None:
    config = SimpleNamespace(model_type="nemotron_h")

    patched = _patch_config_for_skeleton_init(config)

    assert patched is config
    assert config.moe_latent_size is None


def test_patch_nemotron_h_config_preserves_existing_moe_latent_size() -> None:
    config = SimpleNamespace(model_type="nemotron_h", moe_latent_size=1024)

    patched = _patch_config_for_skeleton_init(config)

    assert patched is config
    assert config.moe_latent_size == 1024


def test_patch_config_ignores_other_model_types() -> None:
    config = SimpleNamespace(model_type="llama")

    patched = _patch_config_for_skeleton_init(config)

    assert patched is config
    assert not hasattr(config, "moe_latent_size")


def test_strip_best_prefix_accounts_for_nemotron_backbone_alias() -> None:
    keys = [
        "model_state.model.backbone.embeddings.weight",
        "model_state.model.lm_head.weight",
    ]
    target_keys = {
        "model.embeddings.weight",
        "lm_head.weight",
    }

    prefix, matches = _strip_best_prefix(keys, target_keys)

    assert prefix == "model_state.model."
    assert matches == 2


def test_strip_best_raw_prefix_preserves_origin_backbone_layout() -> None:
    keys = [
        "model_state.model.backbone.embeddings.weight",
        "model_state.model.lm_head.weight",
    ]
    target_keys = {
        "backbone.embeddings.weight",
        "lm_head.weight",
    }

    prefix, matches = _strip_best_raw_prefix(keys, target_keys)

    assert prefix == "model_state.model."
    assert matches == 2


def test_build_hf_model_state_packs_nemotron_expert_weights() -> None:
    tensor_items = {
        "model_state.model.backbone.embeddings.weight": torch.tensor([1.0]),
        "model_state.model.backbone.layers.1.mixer.experts.0.up_proj.weight": torch.tensor([[1.0]]),
        "model_state.model.backbone.layers.1.mixer.experts.1.up_proj.weight": torch.tensor([[2.0]]),
        "model_state.model.backbone.layers.1.mixer.experts.0.down_proj.weight": torch.tensor([[3.0]]),
        "model_state.model.backbone.layers.1.mixer.experts.1.down_proj.weight": torch.tensor([[4.0]]),
        "model_state.model.lm_head.weight": torch.tensor([5.0]),
    }
    target_keys = {
        "model.embeddings.weight",
        "model.layers.1.mixer.experts.up_proj",
        "model.layers.1.mixer.experts.down_proj",
        "lm_head.weight",
    }

    model_state = _build_hf_model_state(tensor_items, "model_state.model.", target_keys)

    assert set(model_state) == target_keys
    assert model_state["model.embeddings.weight"].tolist() == [1.0]
    assert model_state["lm_head.weight"].tolist() == [5.0]
    assert model_state["model.layers.1.mixer.experts.up_proj"].shape == (2, 1, 1)
    assert model_state["model.layers.1.mixer.experts.up_proj"].flatten().tolist() == [1.0, 2.0]
    assert model_state["model.layers.1.mixer.experts.down_proj"].flatten().tolist() == [3.0, 4.0]


def test_convert_with_origin_safetensors_layout_preserves_backbone_keys(tmp_path) -> None:
    origin = tmp_path / "origin"
    output = tmp_path / "output"
    origin.mkdir()
    (origin / "config.json").write_text(
        json.dumps({"model_type": "nemotron_h", "auto_map": {"AutoModelForCausalLM": "modeling.Model"}})
    )
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {
            "backbone.embeddings.weight": "model-00001-of-00001.safetensors",
            "lm_head.weight": "model-00001-of-00001.safetensors",
        },
    }
    (origin / "model.safetensors.index.json").write_text(json.dumps(index))
    tensor_items = {
        "model_state.model.backbone.embeddings.weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        "model_state.model.lm_head.weight": torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
    }

    converted = _convert_with_origin_safetensors_layout(str(origin), tensor_items, str(output))

    assert converted is True
    saved = load_file(output / "model-00001-of-00001.safetensors")
    assert set(saved) == {"backbone.embeddings.weight", "lm_head.weight"}
    assert saved["backbone.embeddings.weight"].tolist() == [1.0, 2.0]
    config = json.loads((output / "config.json").read_text())
    assert config["moe_latent_size"] is None
