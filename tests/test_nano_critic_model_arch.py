import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


ROOT = pathlib.Path(__file__).resolve().parents[1]
NLA_ROOT = ROOT / "external" / "natural_language_autoencoders"
if str(NLA_ROOT) not in sys.path:
    sys.path.insert(0, str(NLA_ROOT))

NEMOTRON_PATCH_FIXTURE = """
import torch
from transformers import PreTrainedModel


def mamba_split_conv1d_scan_combined(*args, **kwargs):
    return args[0]


def mamba_chunk_scan_combined(*args, **kwargs):
    return args[0]


class NemotronHMamba2Mixer:
    def cuda_kernels_forward(self, hidden_states, cache_params, cache_position, attention_mask):
        return mamba_split_conv1d_scan_combined(hidden_states, seq_idx=None)

    def torch_forward(self, hidden_states, cache_params, cache_position, attention_mask):
        return mamba_chunk_scan_combined(hidden_states, seq_idx=None)

    def forward(self, hidden_states, cache_params, cache_position, attention_mask):
        return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)


class NemotronHBlock:
    def forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        if self.block_type == "mamba":
            hidden_states = self.mixer(hidden_states, cache_position=cache_position)
        elif self.block_type == "attention":
            hidden_states = self.attention(hidden_states, cache_position=cache_position)
        return hidden_states


class NemotronHModel:
    def _update_mamba_mask(self, attention_mask, cache_position):
        return attention_mask

    def forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None, position_ids=None):
        mamba_mask = self._update_mamba_mask(attention_mask, cache_position)
        for layer_idx, mixer_block in enumerate(self.layers):
            layer_mask = mamba_mask
            hidden_states = mixer_block(
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=layer_mask,
            )
        return hidden_states


class NemotronHPreTrainedModel(PreTrainedModel):
    pass


class NemotronHForCausalLM(NemotronHPreTrainedModel):
    pass


class NemotronH:
    pass
"""


class NanoCriticModelArchTests(unittest.TestCase):
    def test_seeded_givens_value_head_is_deterministic_and_orthogonal(self):
        from nla.scripts.prepare_critic_checkpoint import _initialize_value_head

        first = torch.empty(12, 12)
        second = torch.empty(12, 12)
        different = torch.empty(12, 12)

        first_report = _initialize_value_head(
            first,
            mode="seeded_givens",
            seed=314159,
            rotation_radians=0.2,
        )
        _initialize_value_head(
            second,
            mode="seeded_givens",
            seed=314159,
            rotation_radians=0.2,
        )
        _initialize_value_head(
            different,
            mode="seeded_givens",
            seed=271828,
            rotation_radians=0.2,
        )

        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(first, different))
        torch.testing.assert_close(first.T @ first, torch.eye(12), atol=1e-6, rtol=1e-6)
        self.assertEqual(first_report["mode"], "seeded_givens")
        self.assertEqual(first_report["seed"], 314159)
        self.assertNotEqual(first_report["before_sha256"], first_report["after_sha256"])

    def test_seeded_router_noise_changes_only_router_parameters(self):
        from nla.scripts.prepare_critic_checkpoint import _perturb_router_parameters

        class FakeGate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.full((4, 6), 0.25))
                self.e_score_correction_bias = torch.nn.Parameter(torch.full((4,), 0.1))

        class FakeMixer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = FakeGate()

        class FakeLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mixer = FakeMixer()

        class FakeCritic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Module()
                self.backbone.layers = torch.nn.ModuleList([FakeLayer(), FakeLayer()])
                self.backbone.embedding = torch.nn.Embedding(8, 6)

        first = FakeCritic()
        second = FakeCritic()
        second.load_state_dict(first.state_dict())
        different = FakeCritic()
        different.load_state_dict(first.state_dict())
        embedding_before = first.backbone.embedding.weight.detach().clone()

        first_report = _perturb_router_parameters(
            first,
            mode="seeded_relative_noise",
            seed=314159,
            relative_std=0.01,
        )
        _perturb_router_parameters(
            second,
            mode="seeded_relative_noise",
            seed=314159,
            relative_std=0.01,
        )
        _perturb_router_parameters(
            different,
            mode="seeded_relative_noise",
            seed=271828,
            relative_std=0.01,
        )

        for name, parameter in first.named_parameters():
            if ".mixer.gate." in name:
                torch.testing.assert_close(parameter, dict(second.named_parameters())[name])
                self.assertFalse(torch.equal(parameter, dict(different.named_parameters())[name]))
        torch.testing.assert_close(first.backbone.embedding.weight, embedding_before)
        self.assertEqual(first_report["parameter_count"], 4)
        self.assertNotEqual(first_report["before_sha256"], first_report["after_sha256"])

    def test_megatron_compat_uses_initialized_value_head(self):
        from safetensors.torch import load_file, save_file

        from nla.scripts.prepare_critic_checkpoint import _add_megatron_compat_keys

        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp)
            save_file({"backbone.weight": torch.ones(2, 2)}, output / "model.safetensors")
            value_head = torch.tensor([[0.8, -0.6], [0.6, 0.8]], dtype=torch.float32)

            _add_megatron_compat_keys(output, value_head, torch.float32)

            compat = load_file(output / "model-megatron-compat.safetensors")
            torch.testing.assert_close(compat["lm_head.weight"], value_head)
            torch.testing.assert_close(compat["model.norm.weight"], torch.ones(2))

    def test_rl_preflight_sets_right_padding_and_eos_pad_fallback(self):
        from nla.scripts.rl_preflight import _prepare_tokenizer_for_preflight

        tokenizer = SimpleNamespace(
            padding_side="left",
            pad_token_id=None,
            pad_token=None,
            eos_token_id=2,
            eos_token="<eos>",
        )

        _prepare_tokenizer_for_preflight(tokenizer)

        self.assertEqual(tokenizer.padding_side, "right")
        self.assertEqual(tokenizer.pad_token, "<eos>")

    def test_rl_preflight_padded_reward_and_train_paths_select_same_rows(self):
        from nla.scripts.rl_preflight import _predict_padded_reward_path, _predict_padded_train_path

        class FakeCritic:
            def __init__(self):
                self.calls = []
                self.values = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)

            def __call__(self, *, input_ids, attention_mask, use_cache, nla_value_indices=None):
                self.calls.append(nla_value_indices)
                if nla_value_indices is not None:
                    rows = torch.arange(self.values.shape[0])
                    return SimpleNamespace(values=self.values[rows, nla_value_indices])
                return SimpleNamespace(values=self.values)

        critic = FakeCritic()
        ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])

        reward_pred = _predict_padded_reward_path(critic, ids, mask)
        train_pred = _predict_padded_train_path(critic, ids, mask)

        torch.testing.assert_close(reward_pred, train_pred)
        self.assertIsNone(critic.calls[0])
        torch.testing.assert_close(critic.calls[1], torch.tensor([2, 3]))

    def test_inner_transformer_accepts_nemotron_backbone_attribute(self):
        from nla.models import _inner_transformer

        class FakeInner:
            pass

        class FakeNemotronForCausalLM:
            def __init__(self):
                self.backbone = FakeInner()

        model = FakeNemotronForCausalLM()

        self.assertIs(_inner_transformer(model), model.backbone)

    def test_resolve_decoder_layers_accepts_nemotron_backbone_attribute(self):
        import torch
        from nla.arch_adapters import resolve_decoder_layers

        class FakeBackbone:
            def __init__(self):
                self.layers = torch.nn.ModuleList([torch.nn.Identity()])

        class FakeNemotronForCausalLM:
            def __init__(self):
                self.backbone = FakeBackbone()

        model = FakeNemotronForCausalLM()

        self.assertIs(resolve_decoder_layers(model), model.backbone.layers)

    def test_tied_weight_key_save_shim_restores_list_shape(self):
        import torch
        from nla.models import _coerce_tied_weight_keys_for_save

        module = torch.nn.Module()
        module._tied_weights_keys = ["lm_head.weight"]

        with _coerce_tied_weight_keys_for_save(module):
            self.assertEqual(module._tied_weights_keys, {"lm_head.weight": "lm_head.weight"})

        self.assertEqual(module._tied_weights_keys, ["lm_head.weight"])

    def test_truncate_config_layers_slices_nano_hybrid_pattern(self):
        from nla.models import _truncate_config_layers

        config = SimpleNamespace(
            num_hidden_layers=52,
            hybrid_override_pattern="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*E",
        )

        _truncate_config_layers(config, 28)

        self.assertEqual(config.num_hidden_layers, 28)
        self.assertEqual(len(config.hybrid_override_pattern), 28)

    def test_prepare_critic_copies_remote_code_auto_map_modules(self):
        from nla.scripts.prepare_critic_checkpoint import _copy_remote_code_files

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            (source / "configuration_nemotron_h.py").write_text("CONFIG = 1\n")
            (source / "modeling_nemotron_h.py").write_text("MODEL = 1\n")

            _copy_remote_code_files(
                str(source),
                str(output),
                {
                    "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
                    "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
                },
            )

            self.assertEqual((output / "configuration_nemotron_h.py").read_text(), "CONFIG = 1\n")
            self.assertEqual((output / "modeling_nemotron_h.py").read_text(), "MODEL = 1\n")

    def test_prepare_critic_patches_copied_nemotron_remote_code(self):
        from nla.remote_code_patches import PATCH_MARKER
        from nla.scripts.prepare_critic_checkpoint import _copy_remote_code_files

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            (source / "configuration_nemotron_h.py").write_text("CONFIG = 1\n")
            (source / "modeling_nemotron_h.py").write_text(NEMOTRON_PATCH_FIXTURE)

            _copy_remote_code_files(
                str(source),
                str(output),
                {
                    "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
                    "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
                },
            )

            copied = (output / "modeling_nemotron_h.py").read_text()
            self.assertIn(PATCH_MARKER, copied)
            self.assertIn("seq_idx = _nla_seq_idx_from_position_ids(position_ids, hidden_states.shape[0])", copied)
            self.assertIn("seq_idx=seq_idx", copied)

    def test_critic_save_copies_remote_code_from_backbone_config(self):
        from nla.models import _copy_remote_code_files_from_config

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            (source / "configuration_nemotron_h.py").write_text("CONFIG = 1\n")
            (source / "modeling_nemotron_h.py").write_text("MODEL = 1\n")
            config = SimpleNamespace(
                _name_or_path=str(source),
                auto_map={
                    "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
                    "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
                },
            )

            _copy_remote_code_files_from_config(config, output)

            self.assertEqual((output / "configuration_nemotron_h.py").read_text(), "CONFIG = 1\n")
            self.assertEqual((output / "modeling_nemotron_h.py").read_text(), "MODEL = 1\n")

    def test_critic_save_patches_copied_nemotron_remote_code_from_backbone_config(self):
        from nla.models import _copy_remote_code_files_from_config
        from nla.remote_code_patches import PATCH_MARKER

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            (source / "configuration_nemotron_h.py").write_text("CONFIG = 1\n")
            (source / "modeling_nemotron_h.py").write_text(NEMOTRON_PATCH_FIXTURE)
            config = SimpleNamespace(
                _name_or_path=str(source),
                auto_map={
                    "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
                    "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
                },
            )

            _copy_remote_code_files_from_config(config, output)

            copied = (output / "modeling_nemotron_h.py").read_text()
            self.assertIn(PATCH_MARKER, copied)
            self.assertIn("seq_idx = _nla_seq_idx_from_position_ids(position_ids, hidden_states.shape[0])", copied)
            self.assertIn("seq_idx=seq_idx", copied)


if __name__ == "__main__":
    unittest.main()
