import pathlib
import sys
import types
import unittest
from unittest import mock

import pytest

pytest.importorskip("torch")


ROOT = pathlib.Path(__file__).resolve().parents[1]
NLA_ROOT = ROOT / "external" / "natural_language_autoencoders"
if str(NLA_ROOT) not in sys.path:
    sys.path.insert(0, str(NLA_ROOT))


class NanoAuditRemediationTests(unittest.TestCase):
    def _read_optional_nemotron_remote_source(self) -> tuple[pathlib.Path, str]:
        source_path = (
            ROOT
            / "runs"
            / "introspection"
            / "ar-r27-datagen-dryrun-20260528T230649Z"
            / "nano_tokenizer"
            / "modeling_nemotron_h.py"
        )
        if not source_path.exists():
            self.skipTest(f"optional ignored Nemotron remote-code fixture is absent: {source_path}")
        return source_path, source_path.read_text()

    def test_nemotron_remote_code_patcher_plumbs_boundaries_and_router_fp32(self):
        from nla.remote_code_patches import patch_nemotron_h_source

        source = """
class NemotronHMixer:
    def forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        y = mamba_split_conv1d_scan_combined(x, seq_idx=None, return_final_states=False)
        z = mamba_chunk_scan_combined(y, seq_idx=None)

class NemotronHBlock:
    def forward(self, hidden_states, cache_position=None, attention_mask=None):
        hidden_states = self.mixer(hidden_states, cache_position=cache_position)
        hidden_states = self.attention(hidden_states, cache_position=cache_position)

class NemotronHModel:
    pass
"""

        patched, report = patch_nemotron_h_source(source)

        self.assertTrue(report.changed)
        self.assertIn("seq_idx=seq_idx", patched)
        self.assertIn("attention_mask=attention_mask", patched)
        self.assertIn("_nla_keep_router_buffers_fp32", patched)

    def test_nemotron_remote_code_patcher_repairs_hybrid_generation_cache(self):
        from nla.remote_code_patches import patch_nemotron_h_source

        source = """
class HybridMambaAttentionDynamicCache:
    def __init__(self, config, batch_size, dtype=None, device=None):
        conv_kernel_size = config.conv_kernel
        self.conv_states = []
        self.ssm_states = []

    def update_conv_state(self, layer_idx, new_conv_state, cache_init=False):
        self.conv_states[layer_idx] = new_conv_state.to(self.conv_states.device)

    def update_ssm_state(self, layer_idx, new_ssm_state):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states.device)

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()

class NemotronHMixer:
    def forward(self, cache_params):
        width = cache_params.conv_kernel_size
        cache_device = cache_params.ssm_states.device
        return width, cache_device

class NemotronHBlock:
    pass

class NemotronHModel:
    pass
"""

        patched, report = patch_nemotron_h_source(source)
        patched_again, second_report = patch_nemotron_h_source(patched)

        self.assertEqual(report.generation_cache_conv_kernel_replacements, 1)
        self.assertEqual(report.generation_cache_device_replacements, 3)
        self.assertEqual(report.generation_cache_reset_replacements, 1)
        self.assertIn("self.conv_kernel_size = conv_kernel_size", patched)
        self.assertIn("self.conv_states[layer_idx].device", patched)
        self.assertIn("self.ssm_states[layer_idx].device", patched)
        self.assertIn("cache_params.ssm_states[self.layer_idx].device", patched)
        self.assertIn("for state in self.conv_states:", patched)
        self.assertIn("for state in self.ssm_states:", patched)
        self.assertFalse(second_report.changed)
        self.assertEqual(patched_again, patched)

    def test_nemotron_remote_code_patcher_blocks_cross_sample_attention(self):
        from nla.remote_code_patches import patch_nemotron_h_source

        source = """
class NemotronHModel:
    def forward(self, attention_mask, inputs_embeds, cache_position, position_ids):
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)
        mamba_mask = self._update_mamba_mask(attention_mask, cache_position)
        seq_idx = _nla_seq_idx_from_position_ids(position_ids, hidden_states.shape[0])
        for layer_idx, mixer_block in enumerate(self.layers):
            inputs_embeds = mixer_block(inputs_embeds)
        return causal_mask

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = cache_position[-1] + 1
        causal_mask = torch.zeros(sequence_length, target_length, device=device)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        return causal_mask

class NemotronHForCausalLM:
    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        cache_params=None,
        output_attentions=None,
    ):
        nemotron_h_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )
        return nemotron_h_outputs
"""

        patched, report = patch_nemotron_h_source(source)
        patched_again, second_report = patch_nemotron_h_source(patched)

        self.assertEqual(report.causal_mask_signature_replacements, 1)
        self.assertEqual(report.causal_mask_call_replacements, 1)
        self.assertEqual(report.packed_attention_boundary_replacements, 1)
        self.assertEqual(report.causal_lm_position_ids_replacements, 1)
        self.assertIn(
            "def _update_causal_mask(self, attention_mask, input_tensor, cache_position, seq_idx=None):",
            patched,
        )
        self.assertIn(
            "self._update_causal_mask(attention_mask, inputs_embeds, cache_position, seq_idx)",
            patched,
        )
        self.assertLess(
            patched.index(
                "seq_idx = _nla_seq_idx_from_position_ids(position_ids, hidden_states.shape[0])"
            ),
            patched.index(
                "self._update_causal_mask(attention_mask, inputs_embeds, cache_position, seq_idx)"
            ),
        )
        self.assertIn("# NLA_PACKED_ATTENTION_BOUNDARY_MASK", patched)
        self.assertIn("causal_mask.masked_fill(~same_sequence, min_dtype)", patched)
        self.assertIn("position_ids=position_ids,", patched)
        self.assertFalse(second_report.changed)
        self.assertEqual(patched_again, patched)

    def test_nemotron_remote_code_patcher_handles_real_multiline_signatures_once(self):
        from nla.remote_code_patches import patch_nemotron_h_source

        source_path, source = self._read_optional_nemotron_remote_source()

        patched, report = patch_nemotron_h_source(source)
        patched_again, second_report = patch_nemotron_h_source(patched)

        self.assertTrue(report.changed)
        self.assertEqual(report.kernel_seq_idx_replacements, 2)
        self.assertGreaterEqual(report.mamba_signature_replacements, 3)
        self.assertEqual(report.block_signature_replacements, 1)
        self.assertGreaterEqual(report.block_mixer_call_replacements, 1)
        self.assertGreaterEqual(report.block_attention_call_replacements, 1)
        self.assertGreaterEqual(report.model_block_call_replacements, 2)
        self.assertIn("seq_idx = _nla_seq_idx_from_position_ids(position_ids, hidden_states.shape[0])", patched)
        self.assertIn("seq_idx=seq_idx", patched)
        compile(patched, str(source_path), "exec")

        self.assertTrue(second_report.already_patched)
        self.assertFalse(second_report.changed)
        self.assertEqual(patched_again, patched)

    def test_nemotron_remote_code_patcher_completes_partial_marker_patch(self):
        from nla.remote_code_patches import patch_nemotron_h_source

        source = """
from typing import Any

# NLA_AUDIT_PATCHED_NEMOTRON_H_20260610
def _nla_seq_idx_from_position_ids(position_ids):
    return None

class NemotronHMixer:
    def cuda_kernels_forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        return mamba_split_conv1d_scan_combined(x, seq_idx=seq_idx)

    def forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)

class NemotronHBlock:
    def forward(self, hidden_states, cache_position=None, attention_mask=None):
        hidden_states = self.mixer(hidden_states, cache_position=cache_position)
        return hidden_states

class NemotronHModel:
    def forward(self, position_ids=None):
        hidden_states = inputs_embeds
        mamba_mask = self._update_mamba_mask(attention_mask, cache_position)
        seq_idx = _nla_seq_idx_from_position_ids(position_ids)
        for layer_idx, mixer_block in enumerate(self.layers):
            hidden_states = mixer_block(
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=layer_mask,
            )
        return hidden_states

class NemotronHPreTrainedModel(PreTrainedModel):
    pass
"""

        patched, report = patch_nemotron_h_source(source)

        self.assertTrue(report.already_patched)
        self.assertTrue(report.changed)
        self.assertIn(
            "def cuda_kernels_forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None, seq_idx=None)",
            patched,
        )
        self.assertIn(
            "return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask, seq_idx)",
            patched,
        )
        self.assertIn("seq_idx = _nla_seq_idx_from_position_ids(position_ids, hidden_states.shape[0])", patched)
        self.assertNotIn("seq_idx = _nla_seq_idx_from_position_ids(position_ids)\n", patched)

    def test_nemotron_seq_idx_helper_keeps_batch_shape_for_fused_mamba(self):
        from nla.remote_code_patches import patch_nemotron_h_source

        _source_path, source = self._read_optional_nemotron_remote_source()

        patched, _report = patch_nemotron_h_source(source)

        self.assertIn("def _nla_seq_idx_from_position_ids(position_ids, batch_size=None):", patched)
        self.assertIn("seq_idx = _nla_seq_idx_from_position_ids(position_ids, hidden_states.shape[0])", patched)
        self.assertNotIn("position_ids.squeeze(0)", patched)
        self.assertIn("pos = pos.expand(batch_size, -1)", patched)
        self.assertIn("dtype=torch.int32", patched)

    def test_ar_eval_patches_restored_hf_remote_code_before_loading(self):
        text = (ROOT / "scripts" / "eval_nano_ar_miles_checkpoint.py").read_text()
        patch_text = (
            ROOT
            / "external"
            / "natural_language_autoencoders"
            / "nla"
            / "remote_code_patches.py"
        ).read_text()

        self.assertIn("prepare_nemotron_h_checkpoint_for_load", text)
        self.assertLess(
            text.index("prepare_nemotron_h_checkpoint_for_load(hf_dir)"),
            text.index("AutoTokenizer.from_pretrained"),
        )
        self.assertIn("transformers_modules", patch_text)
        self.assertIn("shutil.rmtree", patch_text)
        self.assertIn('replace("-", "_hyphen_")', patch_text)

    def test_ar_eval_tokenizer_loader_falls_back_for_tokenizers_backend_checkpoint(self):
        from scripts.eval_nano_ar_miles_checkpoint import _load_tokenizer_for_eval

        calls: list[tuple[str, str, dict[str, object]]] = []

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls.append(("auto", str(path), kwargs))
                raise ValueError("Tokenizer class TokenizersBackend does not exist or is not currently imported.")

        class PreTrainedTokenizerFast:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls.append(("fast", str(path), kwargs))
                return types.SimpleNamespace(
                    padding_side="left",
                    pad_token_id=None,
                    pad_token=None,
                    eos_token="<eos>",
                )

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=AutoTokenizer,
            PreTrainedTokenizerFast=PreTrainedTokenizerFast,
        )

        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            tokenizer = _load_tokenizer_for_eval(pathlib.Path("/tmp/critic"))

        self.assertEqual([call[0] for call in calls], ["auto", "fast"])
        self.assertEqual(calls[1][1], "/tmp/critic")
        self.assertTrue(calls[1][2]["trust_remote_code"])
        self.assertEqual(tokenizer.padding_side, "right")
        self.assertEqual(tokenizer.pad_token, "<eos>")

    def test_training_actor_contains_audit_runtime_hooks(self):
        text = (NLA_ROOT / "nla" / "train_actor.py").read_text()

        required = [
            "padded_critic_inputs_from_tokens",
            "nla_value_indices",
            "RouterEntropyTracker",
            "should_synchronize_microbatch",
            "NLA_ASSERT_PACKED_EQUIV",
            "tol=0.02",
        ]
        for needle in required:
            with self.subTest(needle=needle):
                self.assertIn(needle, text)
        self.assertNotIn("\n        torch.cuda.synchronize()\n        return self._attach_system_metrics", text)

    def test_critic_padded_inputs_preserve_last_token_indices(self):
        import torch
        from nla.audit_runtime import padded_critic_inputs_from_tokens

        ids, mask, last_idx = padded_critic_inputs_from_tokens(
            [
                torch.tensor([11, 12, 13], dtype=torch.long),
                torch.tensor([21], dtype=torch.long),
            ],
            device="cpu",
            pad_id=0,
        )

        self.assertEqual(ids.tolist(), [[11, 12, 13], [21, 0, 0]])
        self.assertEqual(mask.tolist(), [[1, 1, 1], [1, 0, 0]])
        self.assertEqual(last_idx.tolist(), [2, 0])

    def test_critic_training_uses_padded_masked_forward_not_packed_thd(self):
        text = (NLA_ROOT / "nla" / "train_actor.py").read_text()

        self.assertIn("_critic_forward_padded_last_values", text)
        self.assertIn("attention_mask=mask", text)
        self.assertIn("batch[\"_nla_backbone_last_hidden\"] = backbone_h.detach()", text)

    def test_critic_reward_forward_uses_indexed_padded_values_path(self):
        text = (NLA_ROOT / "nla" / "train_actor.py").read_text()

        self.assertIn("_critic_forward_padded_last_values_from_padded", text)
        self.assertIn("values, _backbone_h = _critic_forward_padded_last_values_from_padded", text)
        self.assertIn("was_training = self.model.training", text)
        self.assertIn("self.model.eval()", text)
        self.assertIn("self.model.train(was_training)", text)
        self.assertNotIn("self.model(input_ids=ids, attention_mask=mask, use_cache=False).values", text)

    def test_critic_reward_forward_can_disable_nemotron_mamba_fast_path(self):
        text = (NLA_ROOT / "nla" / "train_actor.py").read_text()
        runtime_text = (NLA_ROOT / "nla" / "mamba_runtime.py").read_text()

        self.assertIn("NLA_CRITIC_FWD_DISABLE_MAMBA_FAST_PATH", text)
        self.assertIn("_temporarily_disable_mamba_fast_path", text)
        self.assertIn("is_fast_path_available", runtime_text)
        self.assertIn("module.is_fast_path_available = False", runtime_text)
        self.assertIn("module.is_fast_path_available = old_value", runtime_text)
        self.assertIn("with _temporarily_disable_mamba_fast_path(self.model), torch.no_grad():", text)

    def test_training_can_select_unfused_torch_causal_conv(self):
        actor_text = (NLA_ROOT / "nla" / "train_actor.py").read_text()
        runtime_text = (NLA_ROOT / "nla" / "mamba_runtime.py").read_text()

        self.assertIn("temporarily_select_mamba_training_kernel", actor_text)
        self.assertIn("resolve_mamba_train_kernel_mode", actor_text)
        self.assertIn("NLA_TRAIN_MAMBA_KERNEL_MODE", runtime_text)
        self.assertIn("NLA_{str(role).strip().upper()}_TRAIN_MAMBA_KERNEL_MODE", runtime_text)
        self.assertIn('"unfused_torch_conv"', runtime_text)
        self.assertIn("torch_causal_conv1d", runtime_text)

    def test_reward_train_gate_uses_the_configured_training_kernel(self):
        actor_text = (NLA_ROOT / "nla" / "train_actor.py").read_text()

        self.assertIn('training_kernel_mode: str = "auto"', actor_text)
        self.assertIn("model, training_kernel_mode", actor_text)
        self.assertIn(
            "training_kernel_mode=self._nla_mamba_training_kernel_mode",
            actor_text,
        )

    def test_reward_train_gate_separates_layout_and_train_mode_tolerances(self):
        actor_text = (NLA_ROOT / "nla" / "train_actor.py").read_text()

        self.assertIn("reward/eval-layout MSE ratio", actor_text)
        self.assertIn("train/eval-layout MSE ratio", actor_text)
        self.assertIn("NLA_CRITIC_REWARD_LAYOUT_MSE_RATIO_TOL", actor_text)
        self.assertIn("NLA_CRITIC_TRAIN_MODE_MSE_RATIO_TOL", actor_text)
        self.assertIn("mse_ratio_agreement", actor_text)

    def test_actor_sft_can_fail_closed_on_packed_padded_drift(self):
        actor_text = (NLA_ROOT / "nla" / "train_actor.py").read_text()

        self.assertIn("NLA_ASSERT_ACTOR_PACKED_EQUIV", actor_text)
        self.assertIn("_maybe_assert_actor_packed_equivalence", actor_text)
        self.assertIn("build_packed_padded_inputs", actor_text)
        self.assertIn("packed_equivalence_metrics", actor_text)
        self.assertIn("actor packed-equivalence gate failed", actor_text)

    def test_rl_train_actor_emits_phase_observability_snapshots(self):
        text = (NLA_ROOT / "nla" / "train_actor.py").read_text()

        for phase in [
            '"critic_reward_fwd"',
            '"critic_frozen_train_skip"',
            '"actor_train_core"',
        ]:
            with self.subTest(phase=phase):
                self.assertIn(phase, text)
        self.assertGreaterEqual(text.count("emit_phase_snapshot"), 4)

    def test_rl_launcher_forwards_phase_observability_env(self):
        text = (NLA_ROOT / "configs" / "rl.sh").read_text()

        for name in [
            "NLA_PHASE_METRICS",
            "NLA_PHASE_METRICS_ALL_GPUS",
            "NLA_PHASE_METRICS_WANDB",
        ]:
            with self.subTest(name=name):
                self.assertIn(name, text)

    def test_train_actor_normalizes_json_float_mamba_time_step_limits(self):
        text = (NLA_ROOT / "nla" / "train_actor.py").read_text()

        self.assertIn("_decode_json_float_sentinel", text)
        self.assertIn("__float__", text)
        self.assertIn('float("inf")', text)
        self.assertIn("_normalize_mamba_time_step_limits", text)
        self.assertIn("_normalize_mamba_time_step_limits(self.model)", text)
        self.assertIn("time_step_limit", text)

    def test_critic_model_supports_indexed_value_head(self):
        text = (NLA_ROOT / "nla" / "models.py").read_text()

        self.assertIn("nla_value_indices", text)
        self.assertIn("h.index_select", text)
        self.assertIn("self.value_head(h_selected)", text)

    def test_critic_model_supports_indexed_value_head_for_padded_batches(self):
        text = (NLA_ROOT / "nla" / "models.py").read_text()

        self.assertIn("h[batch_rows, nla_value_indices.to(h.device)]", text)

    def test_loss_logs_directional_and_norm_metrics_for_m1(self):
        text = (NLA_ROOT / "nla" / "loss.py").read_text()

        self.assertIn("cosine_sum", text)
        self.assertIn("pred_norm_ratio_sum", text)
        self.assertIn("value_head_weight_norm", text)

    def test_injection_hook_avoids_python_match_loop(self):
        text = (NLA_ROOT / "nla" / "injection.py").read_text()

        self.assertIn("valid_positions", text)
        self.assertIn("local_positions", text)
        self.assertNotIn("for b, p in matches.tolist()", text)

    def test_miles_grad_norm_patch_uses_faithful_local_shard_path(self):
        text = (NLA_ROOT / "nla" / "miles_patches" / "0005_fsdp_skip_grad_norm_debug.patch").read_text()

        self.assertIn("clip_grad_norm_local_shards", text)
        self.assertIn("nla_local_grad_norm", text)
        self.assertIn("nla_timing_clip_grad_norm_local_shards", text)

    def test_system_metrics_has_router_entropy_tracker(self):
        text = (NLA_ROOT / "nla" / "system_metrics.py").read_text()

        self.assertIn("class RouterEntropyTracker", text)
        self.assertIn("router_entropy", text)
        self.assertIn("expert_count", text)

    def test_segmented_moe_restores_hidden_state_dtype(self):
        text = (NLA_ROOT / "nla" / "nemotron_moe.py").read_text()

        self.assertIn("to(hidden_states.dtype)", text)

    def test_nemotron_moe_has_configurable_expert_scan_fallback(self):
        text = (NLA_ROOT / "nla" / "nemotron_moe.py").read_text()

        self.assertIn("def expert_scan_moe", text)
        self.assertIn("NLA_MOE_ROUTING_IMPL", text)
        self.assertIn('"expert_scan"', text)


if __name__ == "__main__":
    unittest.main()
