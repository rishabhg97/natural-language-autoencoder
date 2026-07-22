import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeBlock(torch.nn.Module):
    def __init__(self, idx, block_type):
        super().__init__()
        self.idx = idx
        self.block_type = block_type
        self.delta = torch.nn.Parameter(torch.tensor(float(idx + 1)))

    def forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        del cache_params, cache_position, attention_mask
        return hidden_states + self.delta


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(32, 4)
        with torch.no_grad():
            self.embeddings.weight.copy_(torch.arange(128, dtype=torch.float32).view(32, 4) / 100.0)
        self.layers = torch.nn.ModuleList(
            [
                FakeBlock(0, "mamba"),
                FakeBlock(1, "moe"),
                FakeBlock(2, "attention"),
                FakeBlock(3, "moe"),
            ]
        )
        self.norm_f = torch.nn.Identity()

    def get_input_embeddings(self):
        return self.embeddings

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        del input_tensor, cache_position
        return attention_mask

    def _update_mamba_mask(self, attention_mask, cache_position):
        del cache_position
        return attention_mask

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        output_hidden_states=False,
        return_dict=True,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        del kwargs, use_cache
        hidden = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        if cache_position is None:
            cache_position = torch.arange(hidden.shape[1], device=hidden.device)
        causal_mask = self._update_causal_mask(attention_mask, hidden, cache_position)
        mamba_mask = self._update_mamba_mask(attention_mask, cache_position)
        states = [] if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                states.append(hidden)
            layer_mask = mamba_mask if layer.block_type == "mamba" else causal_mask if layer.block_type == "attention" else None
            hidden = layer(hidden, cache_position=cache_position, attention_mask=layer_mask)
        hidden = self.norm_f(hidden)
        if output_hidden_states:
            states.append(hidden)
        if return_dict:
            return SimpleNamespace(
                last_hidden_state=hidden,
                hidden_states=tuple(states) if states is not None else None,
                cache_params=None,
            )
        return hidden, None, tuple(states)


class FakeLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()
        self.lm_head = torch.nn.Linear(4, 7, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(28, dtype=torch.float32).view(7, 4) / 50.0)
        self.config = SimpleNamespace(
            hidden_size=4,
            num_hidden_layers=4,
            hybrid_override_pattern="ME*E",
            num_experts_per_tok=6,
            n_routed_experts=128,
            n_shared_experts=1,
            norm_topk_prob=True,
            routed_scaling_factor=2.5,
            n_group=1,
            topk_group=1,
        )

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def forward(self, *args, **kwargs):
        backbone_out = self.backbone(*args, **kwargs)
        logits = self.lm_head(backbone_out.last_hidden_state)
        return SimpleNamespace(
            logits=logits,
            last_hidden_state=backbone_out.last_hidden_state,
            hidden_states=backbone_out.hidden_states,
            cache_params=backbone_out.cache_params,
        )


class FakeTokenizer:
    chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    def __init__(self):
        self.calls = []
        self.name_or_path = "fake-nano-tokenizer"
        self.pad_token_id = 0
        self.eos_token_id = 0
        self.unk_token_id = -1

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        rendered = "".join(message["content"] for message in messages)
        if kwargs.get("tokenize"):
            return self._encode(rendered)
        return rendered

    def _encode(self, text):
        ids = []
        for ch in text:
            codepoint = ord(ch)
            if 0x3200 <= codepoint <= 0x33FF:
                ids.append(9000 + codepoint - 0x3200)
            else:
                ids.append(codepoint % 31 + 1)
        return ids

    def __call__(self, text, **kwargs):
        ids = self._encode(text)
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        return {"input_ids": ids}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return self._encode(text)


class NanoHarnessTests(unittest.TestCase):
    def test_wandb_env_loader_sets_missing_values_without_overwriting(self):
        nano_wandb = load_script("nano_wandb")
        with tempfile.TemporaryDirectory() as tmp:
            env_path = pathlib.Path(tmp) / ".env"
            env_path.write_text(
                "wandb_api_key=secret-value\n"
                "WANDB_PROJECT=nano-project\n"
                "EXISTING_VALUE=from-file\n"
            )
            old_key = os.environ.get("WANDB_API_KEY")
            old_alias = os.environ.get("WANB_API_KEY")
            old_lower = os.environ.get("wandb_api_key")
            old_project = os.environ.get("WANDB_PROJECT")
            old_existing = os.environ.get("EXISTING_VALUE")
            try:
                os.environ["EXISTING_VALUE"] = "already-set"
                loaded = nano_wandb.load_env_file(env_path)
                self.assertTrue(loaded)
                self.assertEqual(os.environ["WANDB_API_KEY"], "secret-value")
                self.assertEqual(os.environ["WANDB_PROJECT"], "nano-project")
                self.assertEqual(os.environ["EXISTING_VALUE"], "already-set")
            finally:
                for name, old in (
                    ("WANDB_API_KEY", old_key),
                    ("WANB_API_KEY", old_alias),
                    ("wandb_api_key", old_lower),
                    ("WANDB_PROJECT", old_project),
                    ("EXISTING_VALUE", old_existing),
                ):
                    if old is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = old

    def test_wandb_tracker_is_noop_when_disabled(self):
        nano_wandb = load_script("nano_wandb")
        tracker = nano_wandb.init_wandb(
            SimpleNamespace(
                wandb=False,
                wandb_project="nano",
                wandb_entity=None,
                wandb_name=None,
                wandb_group=None,
                wandb_tags="",
                wandb_mode=None,
                wandb_env_file=None,
            ),
            run_dir=pathlib.Path("/tmp/run"),
            job_type="test",
            config={"lr": 1e-4},
        )

        tracker.log({"loss": 1.0}, step=1)
        tracker.finish({"final/loss": 0.5})

        self.assertFalse(tracker.metadata["enabled"])
        self.assertEqual(tracker.metadata["status"], "disabled")

    def test_wandb_tracker_logs_history_and_numeric_summary(self):
        nano_wandb = load_script("nano_wandb")
        calls = []

        class FakeRun:
            def __init__(self):
                self.summary = {}

            def log(self, metrics, step=None):
                calls.append(("log", metrics, step))

            def finish(self):
                calls.append(("finish", dict(self.summary), None))

        def fake_init(**kwargs):
            calls.append(("init", kwargs, None))
            return FakeRun()

        fake_wandb = SimpleNamespace(init=fake_init)
        old_wandb = sys.modules.get("wandb")
        sys.modules["wandb"] = fake_wandb
        try:
            tracker = nano_wandb.init_wandb(
                SimpleNamespace(
                    wandb=True,
                    wandb_project="nano",
                    wandb_entity=None,
                    wandb_name="run-name",
                    wandb_group="group",
                    wandb_tags="av,smoke",
                    wandb_mode="offline",
                    wandb_env_file=None,
                ),
                run_dir=pathlib.Path("/tmp/run"),
                job_type="av",
                config={"lr": 1e-4},
            )
            tracker.log_history(
                [{"step": 1, "loss": 2.0, "batch_indices": [1, 2]}, {"step": 2, "loss": 1.5}],
                prefix="train",
            )
            tracker.log_summary({"eval": {"heldout": {"loss": 1.25}}, "examples": [{"skip": 1}]})
            tracker.finish({"status/passed": True})
        finally:
            if old_wandb is None:
                sys.modules.pop("wandb", None)
            else:
                sys.modules["wandb"] = old_wandb

        init_call = calls[0]
        self.assertEqual(init_call[0], "init")
        self.assertEqual(init_call[1]["project"], "nano")
        self.assertEqual(init_call[1]["name"], "run-name")
        self.assertEqual(calls[1], ("log", {"train/loss": 2.0}, 1))
        self.assertEqual(calls[2], ("log", {"train/loss": 1.5}, 2))
        self.assertEqual(calls[3], ("log", {"eval/heldout/loss": 1.25}, None))
        self.assertEqual(calls[4][0], "finish")
        self.assertTrue(calls[4][1]["status/passed"])

    def test_module_paths_and_metadata_are_nano_specific(self):
        nano_introspection = load_script("nano_introspection")
        model = FakeLM()

        resolved = nano_introspection.resolve_nano_module_paths(model)
        router = nano_introspection.router_config_from_config(model.config)
        pattern = nano_introspection.block_pattern_from_config(model.config, resolved["layers"].obj)

        self.assertEqual(resolved["backbone"].path, ".backbone")
        self.assertEqual(resolved["layers"].path, ".backbone.layers")
        self.assertEqual(resolved["norm_f"].path, ".backbone.norm_f")
        self.assertEqual(resolved["embeddings"].path, ".backbone.embeddings")
        self.assertEqual(pattern, "ME*E")
        self.assertEqual(router["num_experts_per_tok"], 6)
        self.assertEqual(router["n_routed_experts"], 128)

    def test_chat_prompt_renderer_defaults_enable_thinking_false(self):
        nano_introspection = load_script("nano_introspection")
        tokenizer = FakeTokenizer()

        rendered = nano_introspection.render_chat_prompt(
            tokenizer,
            [{"role": "user", "content": "hello"}],
            add_generation_prompt=True,
            enable_thinking=False,
        )

        self.assertEqual(rendered.token_ids, [12, 9, 16, 16, 19])
        self.assertEqual(tokenizer.calls[0]["enable_thinking"], False)
        self.assertTrue(rendered.enable_thinking_requested)
        self.assertTrue(rendered.enable_thinking_applied)

    def test_meta_model_load_does_not_pass_hub_only_kwargs_to_constructor(self):
        nano_introspection = load_script("nano_introspection")

        class FakeAutoModelForCausalLM:
            kwargs = None

            @classmethod
            def from_config(cls, config, **kwargs):
                cls.kwargs = kwargs
                return SimpleNamespace(config=config)

        original_transformers = sys.modules.get("transformers")
        sys.modules["transformers"] = SimpleNamespace(AutoModelForCausalLM=FakeAutoModelForCausalLM)
        try:
            model = nano_introspection.load_model_from_args(
                SimpleNamespace(
                    load_mode="meta",
                    torch_dtype="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                    attn_implementation=None,
                    model_revision=None,
                    device_map="auto",
                    model_id="fake",
                ),
                config=object(),
            )
        finally:
            if original_transformers is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = original_transformers

        self.assertIsNotNone(model)
        self.assertEqual(FakeAutoModelForCausalLM.kwargs, {"trust_remote_code": True})

    def test_device_mapped_model_load_uses_low_cpu_memory_path(self):
        nano_introspection = load_script("nano_introspection")

        class FakeModel:
            def eval(self):
                return self

        class FakeAutoModelForCausalLM:
            kwargs = None

            @classmethod
            def from_pretrained(cls, _model_id, **kwargs):
                cls.kwargs = kwargs
                return FakeModel()

        original_transformers = sys.modules.get("transformers")
        sys.modules["transformers"] = SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModelForCausalLM
        )
        try:
            model = nano_introspection.load_model_from_args(
                SimpleNamespace(
                    load_mode="full",
                    torch_dtype="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                    attn_implementation=None,
                    model_revision=None,
                    device_map="auto",
                    model_id="fake",
                )
            )
        finally:
            if original_transformers is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = original_transformers

        self.assertIsInstance(model, FakeModel)
        self.assertEqual(FakeAutoModelForCausalLM.kwargs["device_map"], "auto")
        self.assertTrue(FakeAutoModelForCausalLM.kwargs["low_cpu_mem_usage"])

    def test_prefix_forward_matches_output_hidden_state_boundary(self):
        nano_identity = load_script("nano_extraction_identity")
        model = FakeLM().eval()
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            full = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
            prefix = nano_identity.prefix_forward_to_R_b(model, input_ids, attention_mask, boundary_b=3)

        torch.testing.assert_close(prefix, full.hidden_states[3])
        self.assertFalse(torch.equal(prefix, full.hidden_states[4]))

    def test_identity_runner_checks_hook_hidden_and_prefix(self):
        nano_identity = load_script("nano_extraction_identity")
        model = FakeLM().eval()
        input_ids = torch.tensor([[4, 5, 6]])
        attention_mask = torch.ones_like(input_ids)

        result = nano_identity.run_boundary_identity_check(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_b=3,
            prompt_name="raw",
            tolerances=nano_identity.IdentityTolerances(relative_l2=1e-6, max_abs=1e-6, one_minus_cos=1e-6),
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["boundary_b"], 3)
        self.assertEqual(result["zero_based_block_index"], 2)
        self.assertEqual(result["hook_path"], ".backbone.layers.2")
        self.assertEqual(result["comparisons"]["hook_vs_output_hidden_states"]["relative_l2"], 0.0)

    def test_extraction_probe_serializes_and_reloads_boundary_tensor(self):
        nano_probe = load_script("nano_extraction_serialize_probe")
        model = FakeLM().eval()
        input_ids = torch.tensor([[4, 5, 6]])
        attention_mask = torch.ones_like(input_ids)

        tensor, hook_path = nano_probe.capture_boundary_tensor(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_b=3,
        )

        with tempfile.TemporaryDirectory() as tmp:
            record = nano_probe.serialize_tensor_record(
                tensor=tensor,
                run_dir=pathlib.Path(tmp),
                boundary_b=3,
                prompt_name="raw",
                hook_path=hook_path,
                token_count=3,
                save_tensors=True,
            )

            self.assertEqual(record["shape"], [1, 3, 4])
            self.assertEqual(record["dtype"], "torch.float32")
            self.assertTrue(record["reload_equal"])
            self.assertGreater(record["tensor_file_bytes"], 0)
            self.assertTrue(pathlib.Path(record["tensor_path"]).exists())

    def test_track_a_scales_vector_and_replaces_one_embedding_row(self):
        nano_track_a = load_script("nano_track_a_probe")
        vector = torch.tensor([3.0, 4.0, 0.0, 0.0])
        embeddings = torch.zeros(1, 3, 4)

        scaled = nano_track_a.normalize_activation(vector, target_scale=10.0)
        injected = nano_track_a.replace_embedding_row(embeddings, position=1, vector=scaled)

        self.assertAlmostEqual(float(scaled.float().norm()), 10.0, places=5)
        self.assertTrue(torch.equal(embeddings, torch.zeros_like(embeddings)))
        torch.testing.assert_close(injected[0, 1], scaled)
        torch.testing.assert_close(injected[0, 0], torch.zeros(4))
        torch.testing.assert_close(injected[0, 2], torch.zeros(4))

    def test_track_a_locates_marker_position_from_offsets(self):
        nano_track_a = load_script("nano_track_a_probe")
        rendered = "aa <NLA_ACTIVATION_MARKER> zz"
        offsets = [(0, 2), (2, 3), (3, 8), (8, 18), (18, 24), (24, 25), (25, 28)]

        position = nano_track_a.find_marker_token_position(
            rendered_text=rendered,
            offsets=offsets,
            marker_text="<NLA_ACTIVATION_MARKER>",
            strategy="first",
        )

        self.assertEqual(position, 2)
        with self.assertRaisesRegex(ValueError, "marker text not found"):
            nano_track_a.find_marker_token_position(
                rendered_text=rendered,
                offsets=offsets,
                marker_text="<MISSING>",
            )

    def test_track_a_locates_common_word_marker_inside_concept_context(self):
        nano_track_a = load_script("nano_track_a_probe")
        rendered = "the prompt has the word <concept>the</concept> later"
        offsets = [
            (0, 3),
            (4, 10),
            (11, 14),
            (15, 18),
            (19, 23),
            (24, 33),
            (33, 36),
            (36, 46),
            (47, 52),
        ]

        position = nano_track_a.find_marker_token_position(
            rendered_text=rendered,
            offsets=offsets,
            marker_text="the",
            left_context="<concept>",
            right_context="</concept>",
        )

        self.assertEqual(position, 6)

    def test_track_a_logit_metrics_and_alpha_response(self):
        nano_track_a = load_script("nano_track_a_probe")
        baseline = torch.tensor([[2.0, 0.0, -1.0]])
        variant = torch.tensor([[1.0, 1.0, -1.0]])

        metrics = nano_track_a.next_token_logit_metrics(baseline, variant)
        summary = nano_track_a.summarize_alpha_response(
            [
                {"alpha": 0.0, "control": "correct", "kl_baseline_to_variant": 0.0},
                {"alpha": 0.5, "control": "correct", "kl_baseline_to_variant": 0.05},
                {"alpha": 1.0, "control": "correct", "kl_baseline_to_variant": 0.20},
                {"alpha": 1.0, "control": "shuffled", "kl_baseline_to_variant": 0.08},
                {"alpha": 1.0, "control": "random_matched_norm", "kl_baseline_to_variant": 0.03},
            ]
        )

        self.assertGreater(metrics["kl_baseline_to_variant"], 0.0)
        self.assertGreater(metrics["max_abs_logit_delta"], 0.0)
        self.assertTrue(summary["correct_non_flat"])
        self.assertGreater(summary["correct_vs_shuffled_kl_gap_at_max_alpha"], 0.0)

    def test_track_c_split_forward_matches_full_forward_and_self_replacement(self):
        nano_identity = load_script("nano_extraction_identity")
        nano_track_c = load_script("nano_track_c_probe")
        model = FakeLM().eval()
        input_ids = torch.tensor([[4, 5, 6]])
        attention_mask = torch.ones_like(input_ids)
        boundary_b = 3
        patch_position = 1

        with torch.no_grad():
            full_logits = nano_track_c.forward_full_next_logits(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            residual = nano_identity.prefix_forward_to_R_b(
                model,
                input_ids,
                attention_mask,
                boundary_b=boundary_b,
            )
            split_logits = nano_track_c.forward_suffix_from_R_b(
                model,
                residual,
                attention_mask=attention_mask,
                boundary_b=boundary_b,
            )
            patched = nano_track_c.replace_residual_row(
                residual,
                position=patch_position,
                vector=residual[0, patch_position],
            )
            self_patch_logits = nano_track_c.forward_suffix_from_R_b(
                model,
                patched,
                attention_mask=attention_mask,
                boundary_b=boundary_b,
            )

        torch.testing.assert_close(split_logits, full_logits)
        torch.testing.assert_close(self_patch_logits, full_logits)

    def test_track_c_repairs_remote_cache_missing_fields_and_list_device_methods(self):
        nano_track_c = load_script("nano_track_c_probe")

        class BrokenCache:
            def __init__(self):
                self.conv_states = [torch.zeros(1, 3, 2)]
                self.ssm_states = [torch.zeros(1, 3, 4)]

        cache = BrokenCache()
        config = SimpleNamespace(conv_kernel=2)

        repairs = nano_track_c.make_hybrid_cache_compatible(cache, config)
        conv_state = cache.update_conv_state(0, torch.ones(1, 3, 2), cache_init=True).clone()
        ssm_state = cache.update_ssm_state(0, torch.ones(1, 3, 4)).clone()
        cache.reset()

        self.assertEqual(cache.conv_kernel_size, 2)
        self.assertIn("set conv_kernel_size", repairs)
        self.assertIn("patched update_conv_state", repairs)
        torch.testing.assert_close(conv_state, torch.ones(1, 3, 2))
        torch.testing.assert_close(ssm_state, torch.ones(1, 3, 4))
        torch.testing.assert_close(cache.conv_states[0], torch.zeros(1, 3, 2))
        torch.testing.assert_close(cache.ssm_states[0], torch.zeros(1, 3, 4))

    def test_ar_baseline_freezes_nano_and_keeps_value_head_trainable(self):
        nano_ar = load_script("nano_ar_frozen_baseline")
        model = FakeLM().eval()
        head = nano_ar.ValueHead(hidden_size=4)

        nano_ar.freeze_module(model)

        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in head.parameters()))

    def test_ar_value_head_is_identity_initialized_without_bias(self):
        nano_ar = load_script("nano_ar_frozen_baseline")
        head = nano_ar.ValueHead(hidden_size=4)
        features = torch.eye(4)

        self.assertIsNone(head.proj.bias)
        torch.testing.assert_close(head.proj.weight, torch.eye(4))
        torch.testing.assert_close(head(features), features)

    def test_ar_capacity_reference_layer_metadata_uses_boundary_not_block_index(self):
        capacity = load_script("nano_ar_capacity_probe")

        meta = capacity.reference_layer_metadata(boundary_b=34, block_count=52)

        self.assertEqual(meta["boundary_b"], 34)
        self.assertEqual(meta["zero_based_last_block_index"], 33)
        self.assertEqual(meta["reference_nla_layer_index_K"], 33)
        self.assertEqual(meta["reference_critic_num_hidden_layers"], 34)

    def test_extraction_identity_uses_real_module_device_before_fallback(self):
        nano_identity = load_script("nano_extraction_identity")
        real_block = torch.nn.Linear(2, 2)
        meta_block = torch.nn.Linear(2, 2, device="meta")
        fallback = torch.device("cpu")

        self.assertEqual(nano_identity._module_execution_device(real_block, torch.device("meta")), fallback)
        self.assertEqual(nano_identity._module_execution_device(meta_block, fallback), fallback)

    def test_ar_capacity_freezes_lower_prefix_and_trains_tail(self):
        capacity = load_script("nano_ar_capacity_probe")
        nano_ar = load_script("nano_ar_frozen_baseline")
        model = FakeLM().eval()
        head = nano_ar.ValueHead(hidden_size=4)
        start_b = capacity.tail_start_boundary(boundary_b=3, train_tail_blocks=1)

        plan = capacity.freeze_for_tail_training(
            model=model,
            head=head,
            prefix_start_b=start_b,
            boundary_b=3,
        )

        self.assertEqual(plan["prefix_start_b"], 2)
        self.assertEqual(plan["trainable_layer_indices"], [2])
        self.assertFalse(any(p.requires_grad for p in model.backbone.layers[0].parameters()))
        self.assertFalse(any(p.requires_grad for p in model.backbone.layers[1].parameters()))
        self.assertTrue(any(p.requires_grad for p in model.backbone.layers[2].parameters()))
        self.assertFalse(any(p.requires_grad for p in model.backbone.layers[3].parameters()))
        self.assertTrue(all(p.requires_grad for p in head.parameters()))

    def test_ar_capacity_tail_forward_matches_full_prefix_with_identity_head(self):
        capacity = load_script("nano_ar_capacity_probe")
        nano_identity = load_script("nano_extraction_identity")
        nano_ar = load_script("nano_ar_frozen_baseline")
        model = FakeLM().eval()
        head = nano_ar.ValueHead(hidden_size=4)
        capacity.freeze_for_tail_training(model=model, head=head, prefix_start_b=2, boundary_b=3)
        input_ids = torch.tensor([[4, 5, 6]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            cached = nano_identity.prefix_forward_to_R_b(model, input_ids, attention_mask, boundary_b=2)
            pred = capacity.forward_tail_from_cache(
                model=model,
                head=head,
                prefix_states=[cached[0].detach().cpu()],
                indices=[0],
                prefix_start_b=2,
                boundary_b=3,
                tau=-1,
            )
            full = nano_identity.prefix_forward_to_R_b(model, input_ids, attention_mask, boundary_b=3)

        torch.testing.assert_close(pred[0], full[0, -1])

    def test_ar_normalized_mse_uses_reference_sqrt_d_scale(self):
        nano_ar = load_script("nano_ar_frozen_baseline")
        x = torch.tensor([[1.0, 0.0]])
        y = torch.tensor([[0.0, 1.0]])

        self.assertAlmostEqual(float(nano_ar.normalized_vector_mse(x, y)), 2.0, places=6)
        self.assertAlmostEqual(nano_ar.cosine_mean(x, y), 0.0, places=6)

    def test_ar_specs_use_explanation_text_not_source_context(self):
        nano_ar = load_script("nano_ar_frozen_baseline")
        prompt = SimpleNamespace(name="raw", input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3), metadata={})

        specs = nano_ar.build_tiny_ar_specs(
            [prompt],
            boundaries=[3],
            max_records=1,
            explanation_template="prompt_label",
            critic_template=nano_ar.DEFAULT_CRITIC_TEMPLATE,
        )

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].source_prompt_name, "raw")
        self.assertEqual(specs[0].boundary_b, 3)
        self.assertTrue(specs[0].explanation_text.endswith(" <summary>"))
        self.assertIn("<text>", specs[0].explanation_text)
        self.assertIn("prompt mode raw", specs[0].explanation_text)
        self.assertNotIn("The Nano NLA pilot checks", specs[0].explanation_text)

    def test_ar_alternating_split_keeps_train_and_eval_interleaved(self):
        nano_ar = load_script("nano_ar_frozen_baseline")

        train, eval_indices = nano_ar.split_indices(8, 0.5, strategy="alternating", seed=123)

        self.assertEqual(train, [0, 2, 4, 6])
        self.assertEqual(eval_indices, [1, 3, 5, 7])

    def test_ar_doc_random_split_holds_out_whole_documents(self):
        nano_ar = load_script("nano_ar_frozen_baseline")
        records = [
            {"doc_id": "doc_a"},
            {"doc_id": "doc_a"},
            {"doc_id": "doc_b"},
            {"doc_id": "doc_b"},
            {"doc_id": "doc_c"},
            {"doc_id": "doc_c"},
        ]

        train, eval_indices = nano_ar.split_indices(
            6,
            0.5,
            strategy="doc_random",
            seed=123,
            records=records,
        )

        self.assertEqual(sorted(train + eval_indices), list(range(6)))
        self.assertTrue(train)
        self.assertTrue(eval_indices)
        train_docs = {records[idx]["doc_id"] for idx in train}
        eval_docs = {records[idx]["doc_id"] for idx in eval_indices}
        self.assertFalse(train_docs & eval_docs)

    def test_ar_centered_raw_r2_uses_training_mean(self):
        nano_ar = load_script("nano_ar_frozen_baseline")
        targets = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
        train_targets = torch.tensor([[0.0, 0.0], [2.0, 0.0]])

        perfect = nano_ar.centered_raw_diagnostics(targets, targets, train_targets)
        mean_baseline = nano_ar.mean_target_metrics(targets, train_targets)

        self.assertAlmostEqual(perfect["centered_raw_r2"], 1.0, places=6)
        self.assertAlmostEqual(mean_baseline["centered_raw_r2"], 0.0, places=6)
        self.assertGreater(mean_baseline["train_mean_l2"], 0.0)

    def test_source_replay_summarizes_token_count_deltas(self):
        replay = load_script("nano_source_replay_probe")
        rows = [
            {"record_id": "a", "stored_n_raw_tokens": 4, "replay_token_count": 4},
            {"record_id": "b", "stored_n_raw_tokens": 5, "replay_token_count": 7},
            {"record_id": "c", "stored_n_raw_tokens": 9, "replay_token_count": 8},
        ]

        summary = replay.summarize_token_count_deltas(rows)

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["exact_count"], 1)
        self.assertAlmostEqual(summary["exact_fraction"], 1 / 3)
        self.assertEqual(summary["delta_min"], -1)
        self.assertEqual(summary["delta_max"], 2)

    def test_source_replay_metric_bundle_reports_exact_subset(self):
        replay = load_script("nano_source_replay_probe")
        rows = [
            {"stored_n_raw_tokens": 2, "replay_token_count": 2},
            {"stored_n_raw_tokens": 3, "replay_token_count": 4},
        ]
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        bundle = replay.replay_metric_bundle(features, targets, rows, seed=123)

        self.assertEqual(bundle["all"]["count"], 2)
        self.assertEqual(bundle["exact_token_count"]["count"], 1)
        self.assertAlmostEqual(bundle["exact_token_count"]["correct"]["cosine_mean"], 1.0, places=6)
        self.assertIn("mean_target", bundle["all"])

    def test_source_replay_loads_token_id_prefix_when_available(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ModuleNotFoundError:
            self.skipTest("pyarrow not installed")

        replay = load_script("nano_source_replay_probe")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "ar_sft.parquet"
            table = pa.table(
                {
                    "activation_vector": pa.array([[1.0, 2.0, 3.0, 4.0]], type=pa.list_(pa.float32(), 4)),
                    "activation_layer": pa.array([34], type=pa.int64()),
                    "doc_id": pa.array(["doc0"], type=pa.string()),
                    "n_raw_tokens": pa.array([3], type=pa.int64()),
                    "detokenized_text_truncated": pa.array(["abc"], type=pa.string()),
                    "token_ids_prefix": pa.array([[11, 12, 13]], type=pa.list_(pa.int32())),
                }
            )
            pq.write_table(table, path)

            specs = replay.load_source_replay_specs(
                path,
                boundaries=[34],
                max_records=4,
                source_column="detokenized_text_truncated",
                source_token_ids_column="token_ids_prefix",
                source_mode="auto",
            )

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].source_token_ids, [11, 12, 13])
        self.assertEqual(specs[0].metadata["source_mode"], "token_ids")

    def test_ar_control_eval_reports_mean_baseline_and_rri(self):
        nano_ar = load_script("nano_ar_frozen_baseline")
        head = nano_ar.ValueHead(hidden_size=2)
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = features.clone()
        train_targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        controls = nano_ar.control_eval(
            head,
            features,
            targets,
            seed=123,
            train_targets_for_mean=train_targets,
            mse_margin=0.05,
            cosine_margin=0.02,
            min_rri=0.05,
        )

        self.assertEqual(controls["count"], 2)
        self.assertIn("mean_train", controls)
        self.assertIn("rri_vs_train_mean", controls["correct"])
        self.assertTrue(controls["correct_beats_mean"])

    def test_ar_value_head_tiny_overfit_reduces_normalized_mse(self):
        nano_ar = load_script("nano_ar_frozen_baseline")
        torch.manual_seed(0)
        features = torch.eye(4)
        targets = features.roll(shifts=1, dims=1)
        head = nano_ar.ValueHead(hidden_size=4)

        before = nano_ar.evaluate_head(head, features, targets)["normalized_mse"]
        history = nano_ar.train_value_head(head, features, targets, max_steps=80, lr=0.2)
        after = nano_ar.evaluate_head(head, features, targets)["normalized_mse"]

        self.assertLess(after, before)
        self.assertLess(history[-1]["loss"], history[0]["loss"])

    def test_ar_value_head_minibatch_training_reduces_normalized_mse(self):
        nano_ar = load_script("nano_ar_frozen_baseline")
        torch.manual_seed(0)
        features = torch.eye(8)
        targets = features.roll(shifts=1, dims=1)
        head = nano_ar.ValueHead(hidden_size=8)

        before = nano_ar.evaluate_head(head, features, targets, batch_size=3)["normalized_mse"]
        history = nano_ar.train_value_head(
            head,
            features,
            targets,
            max_steps=120,
            lr=0.2,
            batch_size=3,
            device="cpu",
            seed=123,
            eval_batch_size=3,
        )
        after = nano_ar.evaluate_head(head, features, targets, batch_size=3)["normalized_mse"]

        self.assertLess(after, before)
        self.assertLess(history[-1]["loss"], history[0]["loss"])

    def test_ar_predict_head_batches_match_full_prediction(self):
        nano_ar = load_script("nano_ar_frozen_baseline")
        features = torch.eye(5)
        head = nano_ar.ValueHead(hidden_size=5)

        full = nano_ar.predict_head(head, features)
        batched = nano_ar.predict_head(head, features, batch_size=2)

        torch.testing.assert_close(batched, full)

    def test_ar_loads_real_ar_sft_parquet_without_source_context(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ModuleNotFoundError:
            self.skipTest("pyarrow not installed")

        nano_ar = load_script("nano_ar_frozen_baseline")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "ar_sft.parquet"
            table = pa.table(
                {
                    "prompt": pa.array(
                        ["Summary of the following text: <text>feature one\n\nfeature two</text> <summary>"],
                        type=pa.string(),
                    ),
                    "activation_vector": pa.array([[1.0, 2.0, 3.0, 4.0]], type=pa.list_(pa.float32(), 4)),
                    "activation_layer": pa.array([3], type=pa.int64()),
                    "doc_id": pa.array(["fineweb:train:0"], type=pa.string()),
                    "n_raw_tokens": pa.array([64], type=pa.int64()),
                }
            )
            pq.write_table(table, path)

            specs = nano_ar.load_parquet_ar_specs(path, boundaries=[3], max_records=4)
            payload = nano_ar.parquet_spec_payload(specs[0])

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].boundary_b, 3)
        self.assertEqual(specs[0].record_id, "fineweb:train:0")
        self.assertIn("<summary>", specs[0].prompt)
        self.assertNotIn("detokenized_text_truncated", payload)
        self.assertIn("prompt_sha256", payload)
        self.assertNotIn("activation_vector", payload)

    def test_ar_batch_selects_last_nonpad_token_per_row(self):
        nano_ar = load_script("nano_ar_frozen_baseline")
        tensor = torch.arange(24, dtype=torch.float32).view(2, 3, 4)

        selected = nano_ar.select_token_vectors_by_lengths(tensor, [1, 3], tau=-1)

        torch.testing.assert_close(selected[0], tensor[0, 0])
        torch.testing.assert_close(selected[1], tensor[1, 2])

    def test_ar_parquet_materialization_batches_without_changing_features(self):
        nano_ar = load_script("nano_ar_frozen_baseline")
        specs = [
            nano_ar.ParquetARSpec(
                record_id="row0",
                boundary_b=3,
                prompt="a",
                activation_vector=[1.0, 2.0, 3.0, 4.0],
                metadata={"row_index": 0},
            ),
            nano_ar.ParquetARSpec(
                record_id="row1",
                boundary_b=3,
                prompt="abc",
                activation_vector=[4.0, 3.0, 2.0, 1.0],
                metadata={"row_index": 1},
            ),
        ]
        model = FakeLM().eval()
        tokenizer = FakeTokenizer()

        single_features, single_targets, single_records = nano_ar.materialize_parquet_ar_examples(
            model=model,
            tokenizer=tokenizer,
            specs=specs,
            ar_tau=-1,
            ar_prompt_max_length=8,
            hidden_size=4,
            ar_feature_batch_size=1,
        )
        batch_features, batch_targets, batch_records = nano_ar.materialize_parquet_ar_examples(
            model=model,
            tokenizer=tokenizer,
            specs=specs,
            ar_tau=-1,
            ar_prompt_max_length=8,
            hidden_size=4,
            ar_feature_batch_size=2,
        )

        torch.testing.assert_close(batch_features, single_features)
        torch.testing.assert_close(batch_targets, single_targets)
        self.assertEqual([record["explanation_token_count"] for record in batch_records], [1, 3])
        self.assertEqual([record["record_id"] for record in batch_records], [record["record_id"] for record in single_records])

    def test_ar_smoke_grid_expands_bounded_configs(self):
        nano_grid = load_script("nano_ar_smoke_grid")

        configs = nano_grid.expand_grid(
            boundaries=["R_34", "R_27"],
            max_records=[8],
            train_fractions=[0.5],
            split_strategies=["alternating"],
            explanation_templates=["generic", "prompt_label"],
            lrs=[2e-5, 5e-5],
            max_steps=[50],
            seeds=[1234],
            max_runs=8,
        )

        self.assertEqual(len(configs), 8)
        self.assertEqual(configs[0]["boundaries"], "R_34")
        self.assertEqual(configs[-1]["boundaries"], "R_27")

    def test_openai_chat_provider_accepts_full_chat_completions_endpoint(self):
        sys.path.insert(0, str(ROOT / "external" / "natural_language_autoencoders"))
        try:
            from nla.datagen import providers
        finally:
            sys.path.pop(0)

        self.assertEqual(
            providers._chat_completions_endpoint("https://inference-api.nvidia.com/v1/chat/completions"),
            "https://inference-api.nvidia.com/v1/chat/completions",
        )
        self.assertEqual(
            providers._chat_completions_endpoint("https://inference-api.nvidia.com/v1"),
            "https://inference-api.nvidia.com/v1/chat/completions",
        )

    def test_openai_chat_provider_extracts_text_response(self):
        sys.path.insert(0, str(ROOT / "external" / "natural_language_autoencoders"))
        try:
            from nla.datagen import providers
        finally:
            sys.path.pop(0)

        payload = {"choices": [{"message": {"content": "<analysis>feature one\n\nfeature two</analysis>"}}]}

        self.assertEqual(providers._chat_completion_text(payload), "<analysis>feature one\n\nfeature two</analysis>")

    def test_openai_chat_provider_drops_length_exhausted_no_content(self):
        sys.path.insert(0, str(ROOT / "external" / "natural_language_autoencoders"))
        try:
            from nla.datagen import providers
        finally:
            sys.path.pop(0)

        payload = {"choices": [{"finish_reason": "length", "message": {"content": None, "reasoning_content": "draft"}}]}

        self.assertIsNone(providers._chat_completion_text(payload))

    def test_openai_chat_provider_uses_reasoning_content_when_stopped_no_content(self):
        sys.path.insert(0, str(ROOT / "external" / "natural_language_autoencoders"))
        try:
            from nla.datagen import providers
        finally:
            sys.path.pop(0)

        payload = {"choices": [{"finish_reason": "stop", "message": {"content": None, "reasoning_content": "<analysis>x</analysis>"}}]}

        self.assertEqual(providers._chat_completion_text(payload), "<analysis>x</analysis>")

    def test_openai_chat_provider_accepts_extra_body_fields(self):
        sys.path.insert(0, str(ROOT / "external" / "natural_language_autoencoders"))
        try:
            from nla.datagen import providers
        finally:
            sys.path.pop(0)

        provider = providers.OpenAIChatCompletionsProvider(
            model="fake",
            base_url="https://example.test/v1",
            api_key="secret",
            extra_body={"chat_template_kwargs": {"thinking": False}},
        )

        self.assertEqual(provider.extra_body, {"chat_template_kwargs": {"thinking": False}})

    def test_injection_token_falls_back_to_cjk_symbol_block(self):
        sys.path.insert(0, str(ROOT / "external" / "natural_language_autoencoders"))
        try:
            from nla.datagen import injection_tokens
        finally:
            sys.path.pop(0)

        class EnclosedMissingTokenizer(FakeTokenizer):
            def __init__(self):
                super().__init__()
                self.name_or_path = "fake-enclosed-missing"

            def _encode(self, text):
                if len(text) == 1 and 0x3200 <= ord(text) <= 0x33FF:
                    return [101, 102]
                if text == "々":
                    return [777]
                return super()._encode(text)

        original_cache_path = injection_tokens._CACHE_PATH
        with tempfile.TemporaryDirectory() as tmp:
            injection_tokens._CACHE_PATH = pathlib.Path(tmp) / "injection_cache.yaml"
            try:
                char, token_id = injection_tokens.find_injection_token(EnclosedMissingTokenizer())
            finally:
                injection_tokens._CACHE_PATH = original_cache_path

        self.assertEqual(char, "々")
        self.assertEqual(token_id, 777)

    def test_stage3_builder_writes_av_parquet_with_cjk_marker_sidecar(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            import yaml
        except ModuleNotFoundError:
            self.skipTest("pyarrow/yaml not installed")

        nano_stage3 = load_script("nano_realdata_stage3_build")
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            nano_stage3.injection_tokens._CACHE_PATH = tmp_path / "injection_cache.yaml"
            src = tmp_path / "av_sft_explained.parquet"
            out = tmp_path / "av_sft.parquet"
            self._write_stage3_input(
                src,
                pa,
                pq,
                yaml,
                include_api_explanation=True,
                row_count=2,
            )

            result = nano_stage3.build_stage3(
                input_path=src,
                output_path=out,
                stage="av_sft",
                tokenizer=tokenizer,
                actor_template=nano_stage3.DEFAULT_ACTOR_TEMPLATE,
                critic_template=nano_stage3.DEFAULT_CRITIC_TEMPLATE,
                keep_debug_metadata=True,
            )

            table = pq.read_table(out)
            meta = yaml.safe_load(out.with_name(out.name + ".nla_meta.yaml").read_text())

        self.assertEqual(result["row_count"], 2)
        self.assertEqual(table.column_names[:3], ["prompt", "response", "activation_vector"])
        self.assertIn("token_position", table.column_names)
        self.assertIn("token_ids_prefix", table.column_names)
        prompt = table.column("prompt").to_pylist()[0][0]["content"]
        self.assertIn("<INJECT>", prompt)
        self.assertNotIn(meta["tokens"]["injection_char"], prompt)
        self.assertTrue(0x3200 <= ord(meta["tokens"]["injection_char"]) <= 0x33FF)
        self.assertIsInstance(meta["tokens"]["injection_token_id"], int)
        self.assertIsNone(meta["tokens"].get("critic_suffix_ids"))
        self.assertEqual(meta["stage"], "av_sft")
        self.assertIn("<explanation>", table.column("response").to_pylist()[0])

    def test_stage3_builder_writes_ar_prompts_and_suffix_metadata(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            import yaml
        except ModuleNotFoundError:
            self.skipTest("pyarrow/yaml not installed")

        nano_stage3 = load_script("nano_realdata_stage3_build")
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            nano_stage3.injection_tokens._CACHE_PATH = tmp_path / "injection_cache.yaml"
            src = tmp_path / "ar_sft_explained.parquet"
            out = tmp_path / "ar_sft.parquet"
            self._write_stage3_input(src, pa, pq, yaml, include_api_explanation=True, row_count=1)

            nano_stage3.build_stage3(
                input_path=src,
                output_path=out,
                stage="ar_sft",
                tokenizer=tokenizer,
                actor_template=nano_stage3.DEFAULT_ACTOR_TEMPLATE,
                critic_template=nano_stage3.DEFAULT_CRITIC_TEMPLATE,
                keep_debug_metadata=False,
            )

            table = pq.read_table(out)
            meta = yaml.safe_load(out.with_name(out.name + ".nla_meta.yaml").read_text())

        self.assertEqual(table.column_names, ["prompt", "activation_vector", "n_raw_tokens", "activation_layer", "doc_id"])
        prompt = table.column("prompt").to_pylist()[0]
        self.assertTrue(prompt.endswith(" <summary>"))
        self.assertIn("<text>feature one", prompt)
        self.assertNotIn("detokenized_text_truncated", table.column_names)
        self.assertEqual(meta["stage"], "ar_sft")
        self.assertGreaterEqual(len(meta["tokens"]["critic_suffix_ids"]), 1)
        self.assertEqual(meta["prompt_templates"]["critic"], nano_stage3.DEFAULT_CRITIC_TEMPLATE)

    def test_stage3_builder_writes_rl_without_teacher_response(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            import yaml
        except ModuleNotFoundError:
            self.skipTest("pyarrow/yaml not installed")

        nano_stage3 = load_script("nano_realdata_stage3_build")
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            nano_stage3.injection_tokens._CACHE_PATH = tmp_path / "injection_cache.yaml"
            src = tmp_path / "rl_raw.parquet"
            out = tmp_path / "rl.parquet"
            self._write_stage3_input(src, pa, pq, yaml, include_api_explanation=False, row_count=1)

            nano_stage3.build_stage3(
                input_path=src,
                output_path=out,
                stage="rl",
                tokenizer=tokenizer,
                actor_template=nano_stage3.DEFAULT_ACTOR_TEMPLATE,
                critic_template=nano_stage3.DEFAULT_CRITIC_TEMPLATE,
                keep_debug_metadata=True,
            )

            table = pq.read_table(out)
            meta = yaml.safe_load(out.with_name(out.name + ".nla_meta.yaml").read_text())

        self.assertNotIn("response", table.column_names)
        self.assertNotIn("api_explanation", table.column_names)
        self.assertEqual(meta["stage"], "rl")
        prompt = table.column("prompt").to_pylist()[0][0]["content"]
        self.assertIn("<INJECT>", prompt)
        self.assertTrue(0x3200 <= ord(meta["tokens"]["injection_char"]) <= 0x33FF)

    def test_av_warmstart_controls_build_real_shuffled_zero_mean_and_none(self):
        av_smoke = load_script("nano_av_warmstart_smoke")
        vectors = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
            ]
        )
        train_indices = [0, 1]

        controls = av_smoke.build_control_vectors(vectors, row_index=0, train_indices=train_indices, seed=7)

        torch.testing.assert_close(controls["real"], vectors[0])
        self.assertEqual(controls["none"], None)
        torch.testing.assert_close(controls["zero"], torch.zeros(4))
        torch.testing.assert_close(controls["mean"], vectors[train_indices].mean(dim=0))
        self.assertFalse(torch.equal(controls["shuffled"], vectors[0]))

    def test_av_warmstart_text_overlap_rewards_row_specific_terms(self):
        av_smoke = load_script("nano_av_warmstart_smoke")

        related = av_smoke.text_overlap_metrics(
            generated="The passage discusses neural routing and expert activation.",
            target="expert activation through neural routing in a model",
        )
        unrelated = av_smoke.text_overlap_metrics(
            generated="A recipe describes citrus, sugar, and a baked crust.",
            target="expert activation through neural routing in a model",
        )

        self.assertGreater(related["content_f1"], unrelated["content_f1"])
        self.assertGreater(related["content_recall"], 0.0)
        self.assertEqual(unrelated["content_f1"], 0.0)

    def test_av_warmstart_loads_config_from_parquet_not_sidecar_path(self):
        av_smoke = load_script("nano_av_warmstart_smoke")
        calls = []

        def fake_load_nla_config(source, tokenizer):
            del tokenizer
            calls.append(pathlib.Path(source))
            return SimpleNamespace(d_model=4)

        original = av_smoke.load_nla_config
        av_smoke.load_nla_config = fake_load_nla_config
        try:
            cfg = av_smoke.load_av_config(pathlib.Path("/tmp/av_sft.parquet"), object())
        finally:
            av_smoke.load_nla_config = original

        self.assertEqual(cfg.d_model, 4)
        self.assertEqual(calls, [pathlib.Path("/tmp/av_sft.parquet")])

    def test_av_warmstart_sets_lm_head_trainable_only(self):
        av_smoke = load_script("nano_av_warmstart_smoke")
        model = FakeLM()

        summary = av_smoke.configure_trainable_parameters(model, trainable_subset="lm_head")

        self.assertGreater(summary["trainable_parameters"], 0)
        self.assertLess(summary["trainable_parameters"], summary["total_parameters"])
        self.assertTrue(all(parameter.requires_grad for parameter in model.lm_head.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertEqual(summary["trainable_names"], ["lm_head.weight"])

    def test_av_warmstart_estimates_lora_parameters_for_target_modules(self):
        av_smoke = load_script("nano_av_warmstart_smoke")
        model = FakeLM()

        summary = av_smoke.estimate_lora_parameters(
            model,
            target_modules=["lm_head"],
            rank=8,
        )

        self.assertEqual(summary["matched_module_count"], 1)
        self.assertEqual(summary["estimated_lora_parameters"], 8 * (7 + 4))
        self.assertEqual(summary["matched_modules_sample"][0]["module"], "lm_head")

    def test_av_warmstart_parses_comma_separated_lora_modules(self):
        av_smoke = load_script("nano_av_warmstart_smoke")

        self.assertEqual(
            av_smoke.parse_csv_list(" q_proj, v_proj ,, down_proj "),
            ["q_proj", "v_proj", "down_proj"],
        )

    def test_av_warmstart_passes_rslora_and_dora_to_peft_config(self):
        av_smoke = load_script("nano_av_warmstart_smoke")
        captured = {}

        def fake_lora_config(**kwargs):
            captured["config_kwargs"] = kwargs
            return SimpleNamespace(**kwargs)

        fake_peft = SimpleNamespace(
            LoraConfig=fake_lora_config,
            TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
            get_peft_model=lambda model, config: model,
        )
        previous = sys.modules.get("peft")
        sys.modules["peft"] = fake_peft
        try:
            _, summary = av_smoke.apply_lora_adapters(
                FakeLM(),
                target_modules=["lm_head"],
                rank=4,
                alpha=8,
                dropout=0.0,
                bias="none",
                modules_to_save=[],
                use_rslora=True,
                use_dora=True,
            )
        finally:
            if previous is None:
                sys.modules.pop("peft", None)
            else:
                sys.modules["peft"] = previous

        self.assertTrue(captured["config_kwargs"]["use_rslora"])
        self.assertTrue(captured["config_kwargs"]["use_dora"])
        self.assertTrue(summary["lora_use_rslora"])
        self.assertTrue(summary["lora_use_dora"])

    def test_wandb_defaults_to_offline_enabled(self):
        nano_wandb = load_script("nano_wandb")
        import argparse

        parser = argparse.ArgumentParser()
        nano_wandb.add_wandb_args(parser)

        args = parser.parse_args([])

        self.assertTrue(args.wandb)
        self.assertEqual(args.wandb_mode, "offline")

    def test_summarize_nano_av_run_extracts_repeatability_gates(self):
        summarizer = load_script("summarize_nano_av_run")
        report = {
            "blockers": [],
            "requested_trainable_subset": "lm_head",
            "effective_trainable_subset": "peft:lora",
            "peft_method": "lora",
            "split": {"train_count": 9, "heldout_count": 1, "doc_overlap_count": 0},
            "peft": {
                "lora_rank": 192,
                "lora_alpha": 384,
                "lora_use_rslora": True,
                "lora_use_dora": False,
                "trainable_parameters": 5303242752,
                "trainable_fraction": 0.14379,
            },
            "training": {
                "train_steps": 800,
                "history": [{"step": 1, "loss": 2.0}, {"step": 800, "loss": 1.1}],
            },
            "evaluation": {
                "loss_summary": {
                    "real": {
                        "heldout": {"loss": 1.0},
                        "heldout_loss_gap_vs_none": 0.3,
                    },
                    "none": {"heldout": {"loss": 1.3}},
                },
                "examples": [
                    {
                        "real": {"parsed_explanation": "alpha", "content_f1": 0.5},
                        "no_injection": {"content_f1": 0.1},
                    }
                ],
            },
            "wandb": {"status": "enabled", "mode": "offline", "name": "run"},
        }

        summary = summarizer.summarize_report(report)

        self.assertEqual(summary["split"]["doc_overlap_count"], 0)
        self.assertEqual(summary["peft"]["effective_trainable_subset"], "peft:lora")
        self.assertEqual(summary["heldout_losses"]["real"], 1.0)
        self.assertEqual(summary["real_heldout_gaps"]["heldout_loss_gap_vs_none"], 0.3)
        self.assertEqual(summary["examples"]["parsed_real"], 1)
        self.assertEqual(summary["examples"]["mean_content_f1_none"], 0.1)
        self.assertIn("requested_trainable_subset=lm_head", summary["warnings"][0])

    def test_av_warmstart_actor_example_masks_prompt_tokens(self):
        av_smoke = load_script("nano_av_warmstart_smoke")
        tokenizer = FakeTokenizer()
        cfg = SimpleNamespace(injection_char="々")
        row = {
            "row_index": 0,
            "prompt": [{"role": "user", "content": "probe <concept><INJECT></concept>"}],
            "response": "<explanation>alpha beta</explanation>",
            "activation_vector": [1.0, 2.0, 3.0, 4.0],
        }

        example = av_smoke.build_actor_sft_example(
            tokenizer,
            cfg,
            row,
            max_target_tokens=64,
        )

        self.assertGreater(example["label_start"], 0)
        self.assertEqual(example["input_ids"].shape, example["labels"].shape)
        self.assertTrue(torch.equal(example["labels"][: example["label_start"]], torch.full_like(example["labels"][: example["label_start"]], -100)))
        self.assertTrue(torch.equal(example["labels"][example["label_start"] :], example["input_ids"][example["label_start"] :]))
        self.assertEqual(tuple(example["activation_vector"].shape), (4,))
        rendered = "".join(message["content"] for message in example["messages"])
        self.assertIn("々", rendered)
        self.assertNotIn("<INJECT>", rendered)

    def test_ar_signal_gate_builds_exact_source_and_shuffled_controls(self):
        signal_gate = load_script("nano_ar_signal_gate")
        rows = [
            signal_gate.SignalRow(
                row_index=0,
                record_id="a",
                boundary_b=34,
                target=[1.0, 0.0, 0.0, 0.0],
                teacher_prompt="teacher a",
                api_explanation="explanation a",
                source_text="source a",
                source_token_ids=[101, 102],
                metadata={"doc_id": "doc-a"},
            ),
            signal_gate.SignalRow(
                row_index=1,
                record_id="b",
                boundary_b=34,
                target=[0.0, 1.0, 0.0, 0.0],
                teacher_prompt="teacher b",
                api_explanation="explanation b",
                source_text="source b",
                source_token_ids=None,
                metadata={"doc_id": "doc-b"},
            ),
        ]

        raw_items = signal_gate.build_variant_items(
            rows,
            variant="source_raw",
            critic_template=signal_gate.DEFAULT_CRITIC_TEMPLATE,
            generic_explanation="generic",
            seed=7,
        )
        shuffled_items = signal_gate.build_variant_items(
            rows,
            variant="teacher_shuffled",
            critic_template=signal_gate.DEFAULT_CRITIC_TEMPLATE,
            generic_explanation="generic",
            seed=7,
        )

        self.assertEqual(raw_items[0].token_ids, [101, 102])
        self.assertIsNone(raw_items[0].text)
        self.assertEqual(raw_items[0].provenance, "source_token_ids_prefix")
        self.assertEqual(raw_items[1].text, "source b")
        self.assertEqual(raw_items[1].provenance, "source_text_retokenized")
        self.assertNotEqual(shuffled_items[0].source_row_index, 0)
        self.assertEqual(shuffled_items[0].text, "teacher b")

    def test_ar_signal_gate_teacher_comparison_requires_mean_and_controls(self):
        signal_gate = load_script("nano_ar_signal_gate")

        def variant(nmse, cosine, mean_nmse=0.9):
            return {
                "head_heldout_after": {"normalized_mse": nmse, "cosine_mean": cosine},
                "mean_heldout": {"normalized_mse": mean_nmse, "cosine_mean": 0.5},
            }

        variants = {
            "teacher": variant(0.7, 0.7),
            "teacher_shuffled": variant(0.8, 0.6),
            "blank": variant(0.85, 0.55),
            "generic": variant(0.83, 0.57),
            "source_context": variant(0.75, 0.65),
            "source_raw": {"feature_heldout": {"normalized_mse": 0.01, "cosine_mean": 0.99}},
        }
        passed = signal_gate.compare_teacher_controls(
            variants,
            mse_margin=0.05,
            cosine_margin=0.02,
            oracle_mse_threshold=0.05,
        )

        self.assertTrue(passed["teacher_beats_controls"])
        self.assertTrue(passed["source_raw_oracle_passed"])

        variants["teacher"] = variant(0.88, 0.7)
        failed = signal_gate.compare_teacher_controls(
            variants,
            mse_margin=0.05,
            cosine_margin=0.02,
            oracle_mse_threshold=0.05,
        )

        self.assertFalse(failed["teacher_beats_controls"])
        self.assertFalse(failed["teacher_beats_mean"])

    def _write_stage3_input(self, path, pa, pq, yaml, include_api_explanation, row_count):
        vectors = [[float(i + 1), float(i + 2), float(i + 3), float(i + 4)] for i in range(row_count)]
        cols = {
            "n_raw_tokens": pa.array([64 + i for i in range(row_count)], type=pa.int64()),
            "detokenized_text_truncated": pa.array([f"reference fineweb prefix {i}" for i in range(row_count)]),
            "activation_vector": pa.array(vectors, type=pa.list_(pa.float32(), 4)),
            "activation_layer": pa.array([34] * row_count, type=pa.int64()),
            "doc_id": pa.array([f"HuggingFaceFW/fineweb:train:{i}" for i in range(row_count)]),
            "token_position": pa.array([63 + i for i in range(row_count)], type=pa.int64()),
            "token_id": pa.array([1000 + i for i in range(row_count)], type=pa.int64()),
            "token_text": pa.array([f" token{i}" for i in range(row_count)], type=pa.string()),
            "token_ids_prefix": pa.array(
                [[101, 200 + i, 1000 + i] for i in range(row_count)],
                type=pa.list_(pa.int32()),
            ),
        }
        if include_api_explanation:
            cols["api_explanation"] = pa.array(
                ["feature one\n\nfeature two" for _ in range(row_count)],
                type=pa.string(),
            )
        pq.write_table(pa.table(cols), path)
        meta = {
            "dataset_id": "base_fake_nano_L34__raw_or_explained",
            "stage": "base",
            "row_count": row_count,
            "extraction": {
                "base_model": "fake-nano",
                "d_model": 4,
                "layer_index": 34,
                "norm": "none",
                "corpus": "HuggingFaceFW/fineweb",
                "corpus_slice": {"start": 0, "length": row_count},
                "positions_per_doc": 1,
            },
            "kind": "nla_dataset",
            "schema_version": 1,
            "keep_debug_metadata": True,
            "prompt_templates": {},
            "parent_datasets": [],
            "created_by": "test",
        }
        path.with_name(path.name + ".nla_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))

    def test_super_teacher_extracts_last_tagged_analysis_block(self):
        teacher = load_script("nano_stage2_super_teacher")
        raw = """
        Earlier quoted template:
        <analysis>
        bad draft
        </analysis>

        Final answer:
        <analysis>
        1. **Syntax pressure:** the final token opens a noun phrase.
        - Discourse state: speaker is contrasting two options.
        * Next-token cue: punctuation likely ends the clause.
        </analysis>
        """

        parsed = teacher.extract_teacher_analysis(raw)

        self.assertIsNotNone(parsed)
        self.assertEqual(
            parsed.text,
            "Syntax pressure: the final token opens a noun phrase.\n\n"
            "Discourse state: speaker is contrasting two options.\n\n"
            "Next-token cue: punctuation likely ends the clause.",
        )
        self.assertEqual(parsed.source, "analysis")
        self.assertEqual(parsed.feature_count, 3)

    def test_super_teacher_payload_enables_high_reasoning_budget(self):
        teacher = load_script("nano_stage2_super_teacher")
        payload = teacher.build_payload(
            model="nvidia/nvidia/nemotron-3-super-v3",
            system_prompt="system",
            user_prompt="user",
            temperature=0.2,
            max_tokens=8192,
            reasoning_effort="high",
            reasoning_budget=4096,
            enable_thinking=True,
        )

        self.assertEqual(payload["model"], "nvidia/nvidia/nemotron-3-super-v3")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(payload["reasoning_budget"], 4096)
        self.assertEqual(
            payload["chat_template_kwargs"],
            {
                "enable_thinking": True,
                "thinking": True,
                "reasoning_budget": 4096,
                "force_nonempty_content": True,
            },
        )

    def test_super_teacher_parse_response_prefers_content_over_reasoning_trace(self):
        teacher = load_script("nano_stage2_super_teacher")
        payload = {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "<analysis>hidden scratchpad should not train</analysis>",
                        "content": "<analysis>\nA\nB\nC\n</analysis>",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        parsed = teacher.parse_chat_completion(payload)

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.text, "A\n\nB\n\nC")
        self.assertEqual(parsed.source, "content")


if __name__ == "__main__":
    unittest.main()
