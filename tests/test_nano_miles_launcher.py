import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoMilesLauncherTests(unittest.TestCase):
    def test_rl_launcher_renders_configurable_kl_type(self):
        script = (
            ROOT
            / "external"
            / "natural_language_autoencoders"
            / "configs"
            / "rl.sh"
        )
        text = script.read_text()

        self.assertIn('KL_LOSS_TYPE="${KL_LOSS_TYPE:-k1}"', text)
        self.assertIn('k1 | k2 | k3 | low_var_kl', text)
        self.assertIn('--kl-loss-type "$KL_LOSS_TYPE"', text)
        self.assertIn("KL_LOSS_TYPE", text[text.index("NLA_TRAIN_ENV_KEYS=") :])

    def test_rl_launcher_renders_optional_train_guard(self):
        script = (
            ROOT
            / "external"
            / "natural_language_autoencoders"
            / "configs"
            / "rl.sh"
        )
        text = script.read_text()

        self.assertIn("NLA_CUSTOM_TRAIN_GUARD_FUNCTION", text)
        self.assertIn("--custom-train-guard-function-path", text)
        self.assertIn("NLA_TRAIN_GUARD_MAX_LOGPROB_ABS_DIFF", text)
        self.assertIn("NLA_TRAIN_GUARD_CONSECUTIVE_STEPS", text)
        patch = (
            ROOT
            / "external"
            / "natural_language_autoencoders"
            / "nla"
            / "miles_patches"
            / "0017_train_guard_hook.patch"
        )
        patch_text = patch.read_text()
        self.assertIn("--custom-train-guard-function-path", patch_text)
        self.assertIn("load_function(args.custom_train_guard_function_path)", patch_text)

    def test_rl_launcher_forwards_runtime_retention_and_guard_contracts(self):
        script = (
            ROOT
            / "external"
            / "natural_language_autoencoders"
            / "configs"
            / "rl.sh"
        )
        text = script.read_text()
        train_env_keys = text[text.index("NLA_TRAIN_ENV_KEYS=") :]

        for key in (
            "NLA_MIN_CRITIC_RETAINED_FRACTION",
            "NLA_TRAIN_GUARD_RULES_JSON",
            "NLA_ROLLOUT_GUARD_RULES_JSON",
        ):
            with self.subTest(key=key):
                self.assertIn(key, train_env_keys)

    def test_miles_fsdp2_launcher_contains_required_nla_sft_args(self):
        script = ROOT / "scripts" / "run_nano_av_miles_fsdp2_sft.sh"
        text = script.read_text()

        required = [
            "--train-backend fsdp",
            "--custom-actor-cls-path nla.train_actor.NLAFSDPActor",
            "--loss-type sft_loss",
            "--debug-train-only",
            "--disable-compute-advantages-and-returns",
            "--rollout-function-path nla.rollout.sft_actor.generate_rollout",
            "--data-source-path nla.data_source.NLADataSource",
            "--input-key prompt",
            "--nla-injection-scale",
            "ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-eager}",
            "--save-interval",
            "NO_SAVE_OPTIM",
            "--no-save-optim",
            "ADAM_FOREACH=${ADAM_FOREACH:-0}",
            "DISABLE_CHECKPOINT_SAVE=${DISABLE_CHECKPOINT_SAVE:-0}",
            "complete-performance cannot disable checkpoint saves",
            "cmd+=(--save \"$SAVE_DIR\" --save-interval \"$SAVE_INTERVAL\")",
            "NLA_TIMING_DEBUG=${NLA_TIMING_DEBUG:-0}",
            "--nla-timing-debug",
            "NLA_SKIP_GRAD_NORM=${NLA_SKIP_GRAD_NORM:-0}",
            "--nla-skip-grad-norm",
            "NLA_LOCAL_GRAD_NORM=${NLA_LOCAL_GRAD_NORM:-1}",
            "--no-nla-local-grad-norm",
            "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}",
            "LOAD_CHECKPOINT=${LOAD_CHECKPOINT:-}",
            "LOAD_CHECKPOINT must point to the checkpoint root containing latest_checkpointed_iteration.txt",
            '[[ "$load_leaf" =~ ^iter_[0-9]+$ ]]',
            "cmd+=(--load \"$LOAD_CHECKPOINT\")",
            "--use-wandb",
            "--wandb-mode offline",
            "NLA_KEEP_LOCAL=${NLA_KEEP_LOCAL:-1}",
        ]
        for needle in required:
            with self.subTest(needle=needle):
                self.assertIn(needle, text)

    def test_miles_patch_disables_adam_foreach_by_default(self):
        patch = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches" / "0003_fsdp_sft_import_fallbacks.patch"
        text = patch.read_text()

        self.assertIn("adam_foreach: bool = False", text)
        self.assertIn('foreach=getattr(args, "adam_foreach", False)', text)

    def test_rl_launcher_exposes_padded_qkv_format(self):
        script = ROOT / "external" / "natural_language_autoencoders" / "configs" / "rl.sh"
        text = script.read_text()

        self.assertIn("QKV_FORMAT=${QKV_FORMAT:-thd}", text)
        self.assertIn('--qkv-format "$QKV_FORMAT"', text)
        actor = (
            ROOT
            / "external"
            / "natural_language_autoencoders"
            / "nla"
            / "train_actor.py"
        ).read_text()
        self.assertIn("build_bshd_max_seq_lens", actor)
        self.assertIn("bshd_pad_size=bshd_pad_size", actor)
        self.assertIn(
            "[NLA SAVE SCHEDULE] explicit_iterations=${NLA_SAVE_ITERATIONS} "
            "miles_save_interval=disabled",
            text,
        )

    def test_miles_patch_enables_bshd_for_fsdp(self):
        patch = (
            ROOT
            / "external"
            / "natural_language_autoencoders"
            / "nla"
            / "miles_patches"
            / "0020_fsdp_bshd_support.patch"
        ).read_text()

        self.assertIn('args.train_backend in (', patch)
        self.assertIn('"megatron", "fsdp"', patch)

    def test_miles_patches_have_valid_hunk_headers(self):
        checker = load_script("check_miles_patches")
        patch_dir = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches"

        failures = checker.check_hunk_counts(patch_dir)

        self.assertEqual(failures, [])

    def test_miles_patch_keeps_router_fp32_without_auto_patching_remote_code(self):
        patch = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches" / "0003_fsdp_sft_import_fallbacks.patch"
        text = patch.read_text()

        self.assertIn("keep_router_fp32", text)
        self.assertNotIn("patch_nemotron_h_checkpoint_dir", text)
        self.assertNotIn("NLA_PATCH_NEMOTRON_REMOTE_CODE", text)

    def test_legacy_miles_launcher_disables_remote_code_auto_patch_by_default(self):
        script = ROOT / "scripts" / "run_nano_av_miles_fsdp2_sft.sh"
        text = script.read_text()

        self.assertIn("NLA_PATCH_NEMOTRON_REMOTE_CODE=${NLA_PATCH_NEMOTRON_REMOTE_CODE:-0}", text)

    def test_miles_patch_skips_empty_optimizer_checkpoint_dirs(self):
        patch = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches" / "0003_fsdp_sft_import_fallbacks.patch"
        text = patch.read_text()

        self.assertIn('optimizer_metadata = optimizer_dir / ".metadata"', text)
        self.assertIn("optimizer_metadata.exists()", text)
        self.assertIn('lr_scheduler_metadata = lr_scheduler_dir / ".metadata"', text)
        self.assertIn("lr_scheduler_metadata.exists()", text)

    def test_miles_patch_does_not_neuter_decay_schedules(self):
        patch = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches" / "0001_miles_nla_integration.patch"
        text = patch.read_text()

        self.assertNotIn("self.lr_scheduler.max_lr = float(args.lr)", text)
        self.assertNotIn("self.lr_scheduler.min_lr = float(args.lr)", text)
        self.assertNotIn('pg.pop("max_lr", None)', text)
        self.assertIn("not getattr(actor.args, \"finetune\", False)", text)

    def test_miles_patch_exposes_no_load_optim_for_fresh_rl_phases(self):
        patch = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches" / "0001_miles_nla_integration.patch"
        text = patch.read_text()

        self.assertIn('"--no-load-optim"', text)
        self.assertIn("Do not load optimizer state from --load", text)
        self.assertIn('args.no_load_optim = True', text)

    def test_fsdp_patch_preserves_decay_for_fresh_and_resumed_sft(self):
        patch = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches" / "0018_fsdp_resume_lr_policy.patch"
        text = patch.read_text()

        self.assertIn("from nla.lr_policy import apply_fsdp_live_lr_policy", text)
        self.assertIn("lr_policy = apply_fsdp_live_lr_policy(", text)
        self.assertIn('lr_policy["last_epoch"]', text)
        self.assertIn('lr_policy["live_lrs"]', text)
        added_lines = "\n".join(
            line[1:] for line in text.splitlines() if line.startswith("+") and not line.startswith("+++")
        )
        self.assertNotIn("stale_resume", added_lines)

    def test_megatron_actor_preserves_decay_for_fresh_sft(self):
        actor = ROOT / "external" / "natural_language_autoencoders" / "nla" / "megatron" / "train_actor.py"
        text = actor.read_text()

        self.assertIn("def _apply_live_lr_policy", text)
        self.assertIn("NLA_FORCE_CONSTANT_LR", text)
        self.assertIn("NLA_PRESERVE_LR_SCHEDULE", text)
        self.assertIn('getattr(args, "finetune", False) or getattr(args, "no_load_optim", False)', text)
        self.assertIn("_apply_live_lr_policy(self.opt_param_scheduler, self.optimizer, args)", text)
        self.assertNotIn('self.opt_param_scheduler.lr_decay_style = "constant"', text)

    def test_miles_patch_adds_nla_timing_debug_breakdown(self):
        patch = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches" / "0004_fsdp_timing_debug.patch"
        text = patch.read_text()

        required = [
            "nla_timing_debug: bool = False",
            "_nla_timing_context",
            "[NLA TIMING]",
            "nla_timing_get_batch",
            "nla_timing_train_step_forward",
            "nla_timing_train_step_backward",
            "nla_timing_clip_grad_norm_raw",
            "nla_timing_clip_grad_norm_full_tensor",
            "nla_timing_optimizer_step",
            "nla_timing_lr_scheduler_step",
            "nla_timing_log_train_step",
        ]
        for needle in required:
            with self.subTest(needle=needle):
                self.assertIn(needle, text)

    def test_miles_patch_adds_explicit_grad_norm_skip_gate(self):
        patch = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches" / "0005_fsdp_skip_grad_norm_debug.patch"
        text = patch.read_text()

        required = [
            "nla_skip_grad_norm: bool = False",
            "nla_local_grad_norm: bool = True",
            'getattr(self.args, "nla_skip_grad_norm", False)',
            'getattr(self.args, "nla_local_grad_norm", True)',
            "clip_grad_norm_local_shards",
            "nla_timing_clip_grad_norm_skipped",
            "nla_timing_clip_grad_norm_local_shards",
            "nla_timing_clip_grad_norm_raw",
            "nla_timing_clip_grad_norm_full_tensor",
        ]
        for needle in required:
            with self.subTest(needle=needle):
                self.assertIn(needle, text)

    def test_miles_patch_routes_dcp_metadata_collectives_through_gloo(self):
        patch = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches" / "0006_fsdp_checkpoint_gloo_pg.patch"
        text = patch.read_text()

        required = [
            "_get_checkpoint_process_group",
            'dist.new_group(backend="gloo")',
            "process_group=checkpoint_process_group",
            "dcp.save(state_dict, checkpoint_id=str(model_dir), process_group=checkpoint_process_group)",
            "dcp.load(state_dict=state_dict, checkpoint_id=str(model_dir), process_group=checkpoint_process_group)",
        ]
        for needle in required:
            with self.subTest(needle=needle):
                self.assertIn(needle, text)

    def test_miles_process_group_patch_uses_torch_helper_signature(self):
        patch = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches" / "0009_process_group_helper_signature_compat.patch"
        text = patch.read_text()

        self.assertIn("inspect.signature(_new_process_group_helper)", text)
        self.assertIn('if "backend_options" in helper_params:', text)
        self.assertIn('elif "pg_options" in helper_params:', text)
        added_lines = "\n".join(
            line[1:] for line in text.splitlines() if line.startswith("+") and not line.startswith("+++")
        )
        self.assertNotIn('str(torch.__version__) >= "2.6"', added_lines)

    def test_miles_weight_update_patch_adds_configured_progress_logging(self):
        patch = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches" / "0010_weight_update_progress_logging.patch"
        text = patch.read_text()

        self.assertIn("NLA_WEIGHT_UPDATE_LOG_EVERY", text)
        self.assertIn("[NLA weight_update] distributed_bucket_start", text)
        self.assertIn("[NLA weight_update] broadcast_enqueued", text)
        self.assertIn("[NLA weight_update] broadcast_waited", text)
        self.assertIn("[NLA weight_update] distributed_bucket_done", text)

    def test_miles_weight_update_patch_keeps_actor_ranks_bucket_synchronized(self):
        patch = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches" / "0010_weight_update_progress_logging.patch"
        text = patch.read_text()

        self.assertIn("NLA weight_update] actor_rank_bucket_barrier", text)
        self.assertIn("dist.barrier()", text)

    def test_miles_train_patch_can_skip_rollout_weight_sync_for_preloaded_smoke(self):
        patch = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches" / "0011_skip_rollout_weight_sync_gate.patch"
        text = patch.read_text()

        self.assertIn("NLA_SKIP_ROLLOUT_WEIGHT_SYNC", text)
        self.assertIn("[NLA] Skipping rollout weight sync", text)
        self.assertIn("return", text)

    def test_miles_abort_patch_uses_external_engine_addrs(self):
        patch = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches" / "0012_external_sglang_abort_addrs.patch"
        text = patch.read_text()

        self.assertIn("rollout_external_engine_addrs", text)
        self.assertIn("_external_sglang_engine_urls", text)
        self.assertIn("[NLA] Abort request using external engine args", text)
        self.assertIn("/abort_request", text)

    def test_miles_tokenizer_patch_falls_back_for_tokenizers_backend_checkpoint(self):
        patch = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches" / "0013_tokenizers_backend_compat.patch"
        text = patch.read_text()

        self.assertIn("PreTrainedTokenizerFast", text)
        self.assertIn("Tokenizer class TokenizersBackend does not exist", text)
        self.assertIn("[NLA tokenizer] Falling back to PreTrainedTokenizerFast", text)
        self.assertIn("raise", text)

    def test_miles_patch_adds_configurable_fsdp_reduce_dtype(self):
        patch = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches" / "0014_fsdp_reduce_dtype_config.patch"
        text = patch.read_text()

        required = [
            "fsdp_reduce_dtype: str = \"float32\"",
            "reduce_dtype_name = str(getattr(args, \"fsdp_reduce_dtype\", \"float32\"))",
            "\"bfloat16\": torch.bfloat16",
            "\"bf16\": torch.bfloat16",
            "Unsupported --fsdp-reduce-dtype",
            "reduce_dtype = reduce_dtype_by_name[reduce_dtype_name]",
        ]
        for needle in required:
            with self.subTest(needle=needle):
                self.assertIn(needle, text)

    def test_miles_patch_can_disable_fsdp_backward_prefetch(self):
        patch = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches" / "0015_fsdp_disable_backward_prefetch.patch"
        text = patch.read_text()

        required = [
            "fsdp_disable_backward_prefetch: bool = False",
            'getattr(args, "fsdp_disable_backward_prefetch", False)',
            "original_pre_backward = param_group.pre_backward",
            "pre_backward_no_default_prefetch",
            "return _original(False, *unused)",
            "_nla_backward_prefetch_disabled",
            "Disabled FSDP2 default backward prefetch",
        ]
        for needle in required:
            with self.subTest(needle=needle):
                self.assertIn(needle, text)

    def test_nla_actor_prunes_local_checkpoints_without_remote_backup(self):
        actor = ROOT / "external" / "natural_language_autoencoders" / "nla" / "train_actor.py"
        text = actor.read_text()

        self.assertIn('os.environ.get("NLA_KEEP_LOCAL", "1")', text)
        self.assertIn('subprocess.run(["bash", "-c", prune], check=False)', text)
        self.assertNotIn('if os.environ.get("NLA_BACKUP_REMOTE"):\n                    prune =', text)

    def test_nla_ref_model_normalizes_dtype_before_fsdp_state_capture(self):
        actor = ROOT / "external" / "natural_language_autoencoders" / "nla" / "train_actor.py"
        text = actor.read_text()

        self.assertIn("normalize_floating_module_dtype", text)
        self.assertLess(
            text.index("normalize_floating_module_dtype(ref"),
            text.index("full_state = ref.state_dict()"),
        )

    def test_miles_patch_gives_offline_secondary_runs_distinct_role_ids(self):
        patch = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches" / "0016_wandb_offline_role_runs.patch"
        text = patch.read_text()

        required = [
            "def init_tracking(args, primary: bool = True, role: str | None = None, **kwargs):",
            "init_wandb_secondary(args, role=role, **kwargs)",
            'init_tracking(args, primary=False, role="rollout"',
            "init_tracking(args, primary=False, role=role)",
            "def init_wandb_secondary(args, router_addr=None, role=None):",
            'offline_role = role or "secondary"',
            'offline_run_id = f"{wandb_run_id}-{offline_role}"',
            '"id": offline_run_id',
            'init_kwargs["name"] = f"{args.wandb_group}-{offline_role}"',
            'init_kwargs["id"] = args.wandb_run_id',
        ]
        for needle in required:
            with self.subTest(needle=needle):
                self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
