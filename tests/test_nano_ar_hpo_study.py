import importlib.util
import json
import pathlib
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoARHPOStudyTests(unittest.TestCase):
    def test_test_only_metrics_never_define_an_hpo_objective(self):
        study = load_script("nano_ar_hpo_study")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            ar_report = root / "ar.json"
            ar_report.write_text(
                json.dumps(
                    {
                        "splits": {
                            "test": {
                                "controls": {
                                    "teacher": {"normalized_mse": 0.01}
                                }
                            }
                        }
                    }
                )
            )
            av_report = root / "av.json"
            av_report.write_text(
                json.dumps(
                    {
                        "loss_summary": {
                            "real": {"test": {"loss": 0.01, "count": 512}}
                        }
                    }
                )
            )
            roundtrip_report = root / "roundtrip.json"
            roundtrip_report.write_text(
                json.dumps(
                    {
                        "splits": {
                            "test": {
                                "variants": {
                                    "av_real": {"normalized_mse": 0.01}
                                }
                            }
                        }
                    }
                )
            )

            ar_metrics = study.metrics_from_eval(ar_report, task="ar")
            av_metrics = study.metrics_from_eval(av_report, task="av")
            roundtrip_metrics = study.metrics_from_roundtrip_report(
                roundtrip_report
            )

        self.assertNotIn("objective_nmse", ar_metrics)
        self.assertNotIn("objective_nll", av_metrics)
        self.assertNotIn("objective_roundtrip_nmse", roundtrip_metrics)

    def test_build_trial_record_extracts_eval_objective_and_train_metrics(self):
        study = load_script("nano_ar_hpo_study")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config = root / "config.yaml"
            config.write_text(
                textwrap.dedent(
                    """
                    run:
                      name: nano-ar-lr1e5
                    training:
                      lr: 1e-5
                      min_lr: 1e-6
                      lr_decay_style: cosine
                      lr_warmup_iters: 25
                      resume_steps: 256
                      global_batch_size: 192
                      micro_batch_size: 8
                      rollout_batch_size: 192
                    """
                )
            )
            eval_report = root / "eval.json"
            eval_report.write_text(
                json.dumps(
                    {
                        "checkpoint_dir": "/ckpt/iter_0001803",
                        "splits": {
                            "validation": {
                                "controls": {
                                    "teacher": {
                                        "normalized_mse": 0.4,
                                        "cosine_mean": 0.8,
                                        "fve_nrm": 0.5,
                                    },
                                    "mean": {"normalized_mse": 0.9},
                                    "source_context": {"normalized_mse": 0.5},
                                },
                                "rowwise_win_rates": {
                                    "teacher_vs_mean": {"teacher_better_fraction": 0.98},
                                    "teacher_vs_source_context": {"teacher_better_fraction": 0.65},
                                },
                            },
                            "test": {
                                "controls": {
                                    "teacher": {
                                        "normalized_mse": 0.5,
                                        "cosine_mean": 0.75,
                                        "fve_nrm": 0.45,
                                    },
                                    "mean": {"normalized_mse": 0.88},
                                    "source_context": {"normalized_mse": 0.52},
                                },
                                "rowwise_win_rates": {
                                    "teacher_vs_mean": {"teacher_better_fraction": 0.97},
                                    "teacher_vs_source_context": {"teacher_better_fraction": 0.6},
                                },
                            },
                        },
                    }
                )
            )
            train_log = root / "train.log"
            train_log.write_text(
                "[x] step 1: {'train/loss': 0.42, 'train/fve_nrm': 0.37, 'train/grad_norm': 0.7, 'train/step': 1}\n"
            )

            record = study.build_trial_record(
                trial_name="nano-ar-lr1e5",
                config_path=config,
                eval_report_path=eval_report,
                train_log_path=train_log,
                run_dir=pathlib.Path("/runs/nano-ar-lr1e5"),
                status="complete",
                notes="unit test",
            )

        self.assertEqual(record["schema_version"], "nano_ar_hpo_trial.v1")
        self.assertEqual(record["params"]["lr"], 1e-5)
        self.assertEqual(record["params"]["lr_decay_style"], "cosine")
        self.assertEqual(record["params"]["resume_steps"], 256)
        self.assertAlmostEqual(record["metrics"]["objective_nmse"], 0.4)
        self.assertAlmostEqual(record["metrics"]["validation_teacher_nmse"], 0.4)
        self.assertAlmostEqual(record["metrics"]["test_teacher_nmse"], 0.5)
        self.assertAlmostEqual(record["metrics"]["validation_teacher_beats_mean"], 0.98)
        self.assertAlmostEqual(record["metrics"]["final_train_loss"], 0.42)
        self.assertEqual(record["artifacts"]["checkpoint_dir"], "/ckpt/iter_0001803")

    def test_lr_decay_canary_flags_flat_cosine_schedule(self):
        study = load_script("nano_ar_hpo_study")

        flat = study.lr_decay_canary(
            {"lr": 2e-5, "min_lr": 2e-6, "lr_decay_style": "cosine"},
            {"final_train_lr": 2e-5},
        )
        decayed = study.lr_decay_canary(
            {"lr": 2e-5, "min_lr": 2e-6, "lr_decay_style": "cosine"},
            {"final_train_lr": 2e-6},
        )

        self.assertFalse(flat["passed"])
        self.assertIn("flat", flat["message"])
        self.assertTrue(decayed["passed"])

    def test_lr_decay_canary_fails_missing_final_lr_on_decay_schedule(self):
        study = load_script("nano_ar_hpo_study")

        missing = study.lr_decay_canary(
            {"lr": 2e-5, "min_lr": 2e-6, "lr_decay_style": "cosine"},
            {},
        )
        invalid_min = study.lr_decay_canary(
            {"lr": 2e-5, "min_lr": 2e-5, "lr_decay_style": "cosine"},
            {"final_train_lr": 2e-5},
        )

        self.assertTrue(missing["applicable"])
        self.assertFalse(missing["passed"])
        self.assertIn("missing final_train_lr", missing["message"])
        self.assertTrue(invalid_min["applicable"])
        self.assertFalse(invalid_min["passed"])
        self.assertIn("min_lr", invalid_min["message"])

    def test_assert_lr_decay_canary_passed_raises_for_failed_cosine(self):
        study = load_script("nano_ar_hpo_study")

        with self.assertRaisesRegex(ValueError, "LR decay canary failed"):
            study.assert_lr_decay_canary_passed(
                {"lr": 2e-5, "min_lr": 2e-6, "lr_decay_style": "cosine"},
                {"final_train_lr": 2e-5},
            )

    def test_suggest_next_trials_skips_completed_parameter_sets(self):
        study = load_script("nano_ar_hpo_study")
        completed = [
            {
                "trial_name": "fullscan",
                "status": "complete",
                "params": {
                    "lr": 1e-5,
                    "min_lr_ratio": 0.1,
                    "lr_decay_style": "constant",
                    "lr_warmup_iters": 0,
                    "resume_steps": 1291,
                    "global_batch_size": 192,
                },
                "metrics": {"objective_nmse": 0.508706},
            },
            {
                "trial_name": "qwen-lr2e5",
                "status": "complete",
                "params": {
                    "lr": 2e-5,
                    "min_lr_ratio": 0.1,
                    "lr_decay_style": "cosine",
                    "lr_warmup_iters": 50,
                    "resume_steps": 1547,
                    "global_batch_size": 192,
                },
                "metrics": {"objective_nmse": 0.443697},
            },
        ]

        suggestions = study.suggest_next_trials(completed, top_n=3)
        completed_keys = {study.param_signature(trial["params"]) for trial in completed}

        self.assertEqual(len(suggestions), 3)
        for suggestion in suggestions:
            self.assertNotIn(study.param_signature(suggestion["params"]), completed_keys)
            self.assertIn("reason", suggestion)
            self.assertIn("objective_hint", suggestion)

        markdown = study.render_suggestions_markdown(suggestions, completed)
        self.assertIn("| Rank | lr | min_lr_ratio | schedule | warmup | steps | batch | reason |", markdown)
        self.assertIn("Best completed trial", markdown)

    def test_export_optuna_payload_uses_objective_values_for_complete_trials(self):
        study = load_script("nano_ar_hpo_study")
        trials = [
            {
                "trial_name": "complete",
                "status": "complete",
                "params": {"lr": 1e-5, "resume_steps": 256},
                "metrics": {"objective_nmse": 0.42},
                "artifacts": {"eval_report": "/tmp/eval.json"},
            },
            {
                "trial_name": "running",
                "status": "running",
                "params": {"lr": 5e-6, "resume_steps": 256},
                "metrics": {},
                "artifacts": {},
            },
        ]

        payload = study.export_optuna_payload(trials)

        self.assertEqual(payload["direction"], "minimize")
        self.assertEqual(payload["objective"], "objective_nmse")
        self.assertEqual(payload["trials"][0]["state"], "COMPLETE")
        self.assertEqual(payload["trials"][0]["value"], 0.42)
        self.assertEqual(payload["trials"][1]["state"], "RUNNING")
        self.assertIsNone(payload["trials"][1]["value"])

    def test_build_av_trial_record_extracts_real_nll_and_control_gaps(self):
        study = load_script("nano_ar_hpo_study")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config = root / "av_config.yaml"
            config.write_text(
                textwrap.dedent(
                    """
                    training:
                      objective: av_sft
                      lr: 1e-5
                      injection_scale: 75
                      epochs: 1
                      global_batch_size: 192
                      micro_batch_size: 8
                      rollout_batch_size: 192
                    """
                )
            )
            eval_report = root / "av_eval.json"
            eval_report.write_text(
                json.dumps(
                    {
                        "format": "nano_av_miles_checkpoint_eval.v1",
                        "hf_checkpoint": "/hf/iter_0467",
                        "loss_summary": {
                            "real": {
                                "validation": {"loss": 0.9, "count": 64},
                                "test": {"loss": 1.0, "count": 64},
                                "validation_loss_gap_vs_mean": 0.3,
                                "test_loss_gap_vs_mean": 0.25,
                                "validation_loss_gap_vs_shuffled": 0.4,
                                "test_loss_gap_vs_shuffled": 0.35,
                                "validation_loss_gap_vs_zero": 0.2,
                                "test_loss_gap_vs_zero": 0.15,
                                "validation_loss_gap_vs_none": 0.5,
                                "test_loss_gap_vs_none": 0.45,
                            },
                            "mean": {"validation": {"loss": 1.2}, "test": {"loss": 1.25}},
                            "shuffled": {"validation": {"loss": 1.3}, "test": {"loss": 1.35}},
                            "zero": {"validation": {"loss": 1.1}, "test": {"loss": 1.15}},
                            "none": {"validation": {"loss": 1.4}, "test": {"loss": 1.45}},
                        },
                    }
                )
            )

            record = study.build_trial_record(
                trial_name="av-hero",
                config_path=config,
                eval_report_path=eval_report,
                train_log_path=None,
                run_dir=pathlib.Path("/runs/av-hero"),
                status="complete",
                task="av",
            )

        self.assertEqual(record["task"], "av")
        self.assertEqual(record["metrics"]["objective_key"], "objective_nll")
        self.assertAlmostEqual(record["metrics"]["objective_nll"], 0.9)
        self.assertAlmostEqual(record["metrics"]["validation_real_nll"], 0.9)
        self.assertAlmostEqual(record["metrics"]["test_real_nll"], 1.0)
        self.assertAlmostEqual(record["metrics"]["validation_gap_vs_mean"], 0.3)
        self.assertEqual(record["params"]["injection_scale"], 75.0)

    def test_build_av_roundtrip_trial_record_uses_roundtrip_nmse_objective(self):
        study = load_script("nano_ar_hpo_study")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config = root / "av_config.yaml"
            config.write_text(
                textwrap.dedent(
                    """
                    training:
                      objective: av_sft
                      lr: 1e-4
                      injection_scale: 75
                      epochs: 1
                      global_batch_size: 192
                      micro_batch_size: 1
                      rollout_batch_size: 192
                    """
                )
            )
            eval_report = root / "av_eval.json"
            eval_report.write_text(
                json.dumps(
                    {
                        "hf_checkpoint": "/hf/iter_0096",
                        "loss_summary": {
                            "real": {
                                "validation": {"loss": 0.9, "count": 8},
                                "test": {"loss": 1.0, "count": 8},
                            }
                        },
                    }
                )
            )
            roundtrip_report = root / "roundtrip.json"
            roundtrip_report.write_text(
                json.dumps(
                    {
                        "splits": {
                            "validation": {
                                "variants": {
                                    "av_real": {"normalized_mse": 0.41},
                                    "teacher": {"normalized_mse": 0.32},
                                    "mean": {"normalized_mse": 0.7},
                                },
                                "generation_parse": {"real": {"closed_fraction": 0.875, "empty_fraction": 0.0}},
                            },
                            "test": {
                                "variants": {
                                    "av_real": {"normalized_mse": 0.45},
                                    "teacher": {"normalized_mse": 0.35},
                                    "mean": {"normalized_mse": 0.72},
                                },
                                "generation_parse": {"real": {"closed_fraction": 1.0, "empty_fraction": 0.0}},
                            },
                        },
                        "gate": {
                            "passed": False,
                            "splits": {
                                "validation": {"beats_all_controls": True},
                                "test": {"beats_all_controls": False},
                            },
                        },
                    }
                )
            )

            record = study.build_trial_record(
                trial_name="av-rt",
                config_path=config,
                eval_report_path=eval_report,
                roundtrip_report_path=roundtrip_report,
                train_log_path=None,
                run_dir=pathlib.Path("/runs/av-rt"),
                status="complete",
                task="av_roundtrip",
            )

        self.assertEqual(record["task"], "av_roundtrip")
        self.assertEqual(record["metrics"]["objective_key"], "objective_roundtrip_nmse")
        self.assertAlmostEqual(record["metrics"]["objective_roundtrip_nmse"], 0.41)
        self.assertAlmostEqual(record["metrics"]["validation_roundtrip_av_real_nmse"], 0.41)
        self.assertAlmostEqual(record["metrics"]["test_roundtrip_teacher_nmse"], 0.35)
        self.assertAlmostEqual(record["metrics"]["validation_roundtrip_parse_closed_fraction"], 0.875)
        self.assertFalse(record["metrics"]["roundtrip_gate_passed"])
        self.assertEqual(record["artifacts"]["roundtrip_report"], str(roundtrip_report))

    def test_av_suggestions_use_objective_nll_and_injection_scale(self):
        study = load_script("nano_ar_hpo_study")
        trials = [
            {
                "trial_name": "phase1",
                "task": "av",
                "status": "complete",
                "params": {
                    "lr": 1e-4,
                    "injection_scale": 75.0,
                    "global_batch_size": 8,
                    "resume_steps": 10000,
                },
                "metrics": {"objective_key": "objective_nll", "objective_nll": 1.6051},
            },
            {
                "trial_name": "hero",
                "task": "av",
                "status": "complete",
                "params": {
                    "lr": 1e-5,
                    "injection_scale": 75.0,
                    "global_batch_size": 192,
                    "resume_steps": 467,
                },
                "metrics": {"objective_key": "objective_nll", "objective_nll": 0.930576},
            },
        ]

        suggestions = study.suggest_next_trials(trials, top_n=3, task="av")
        markdown = study.render_suggestions_markdown(suggestions, trials, task="av")
        payload = study.export_optuna_payload(trials, task="av")

        self.assertEqual(len(suggestions), 3)
        self.assertEqual(suggestions[0]["params"]["injection_scale"], 75.0)
        self.assertIn("objective NLL", markdown)
        self.assertEqual(payload["objective"], "objective_nll")
        self.assertEqual(payload["trials"][0]["value"], 1.6051)

    def test_av_roundtrip_suggestions_use_roundtrip_nmse_objective(self):
        study = load_script("nano_ar_hpo_study")
        trials = [
            {
                "trial_name": "rt-a",
                "task": "av_roundtrip",
                "status": "complete",
                "params": {
                    "lr": 1e-4,
                    "injection_scale": 75.0,
                    "global_batch_size": 192,
                    "resume_steps": 96,
                },
                "metrics": {
                    "objective_key": "objective_roundtrip_nmse",
                    "objective_roundtrip_nmse": 0.43,
                },
            }
        ]

        suggestions = study.suggest_next_trials(trials, top_n=2, task="av_roundtrip")
        markdown = study.render_suggestions_markdown(suggestions, trials, task="av_roundtrip")
        payload = study.export_optuna_payload(trials, task="av_roundtrip")

        self.assertEqual(len(suggestions), 2)
        self.assertIn("round-trip objective NMSE", markdown)
        self.assertEqual(payload["objective"], "objective_roundtrip_nmse")
        self.assertEqual(payload["trials"][0]["value"], 0.43)


if __name__ == "__main__":
    unittest.main()
