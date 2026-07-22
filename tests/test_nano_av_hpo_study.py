import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "nano_av_hpo_study.py"
    spec = importlib.util.spec_from_file_location("nano_av_hpo_study", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoAVHPOStudyTests(unittest.TestCase):
    def test_export_defaults_to_validation_only_av_objective(self):
        study = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            trials = root / "trials.jsonl"
            trials.write_text(
                json.dumps(
                    {
                        "trial_name": "av-probe",
                        "task": "av",
                        "status": "complete",
                        "metrics": {
                            "objective_key": "objective_nll",
                            "objective_nll": 0.9,
                            "objective_split": "validation",
                            "test_real_nll": 0.1,
                        },
                    }
                )
                + "\n"
            )
            output = root / "optuna.json"

            result = study.main(
                [
                    "export-optuna",
                    "--study-jsonl",
                    str(trials),
                    "--out-json",
                    str(output),
                ]
            )
            payload = json.loads(output.read_text())

        self.assertEqual(result, 0)
        self.assertEqual(payload["objective"], "objective_nll")
        self.assertEqual(payload["trials"][0]["value"], 0.9)


if __name__ == "__main__":
    unittest.main()
