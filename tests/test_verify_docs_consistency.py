import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_docs_consistency.py"


def load_script():
    spec = importlib.util.spec_from_file_location("verify_docs_consistency", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class VerifyDocsConsistencyTests(unittest.TestCase):
    def test_rejects_legacy_headline_without_invalidation_marker(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "current_state.md"
            path.write_text("The hero improved by 30.97% / 32.34%.\n")

            issues = module.validate_paths([path])

        self.assertEqual(len(issues), 1)
        self.assertIn("missing publication invalidation marker", issues[0])

    def test_accepts_historical_headline_with_invalidation_marker(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "experiment_logbook.md"
            path.write_text(
                f"{module.PUBLICATION_INVALIDATION_MARKER}\n"
                "Historical result: 30.97% / 32.34%.\n"
            )

            issues = module.validate_paths([path])

        self.assertEqual(issues, [])

    def test_registry_requires_structured_publication_invalidation(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "experiments.yaml"
            path.write_text(
                "experiments:\n"
                "  - id: r33-corrected-k3-hero-lr1e5-update342-resume228-retry3\n"
                "    status: passed_selected\n"
            )

            issues = module.validate_paths([path])

        self.assertEqual(len(issues), 1)
        self.assertIn("structured publication invalidation", issues[0])

    def test_registry_rejects_superseded_protocol_only_invalidation(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "experiments.yaml"
            path.write_text(
                "experiments:\n"
                "  - id: r33-corrected-k3-hero-lr1e5-update342-resume228-retry3\n"
                "    status: exploratory_selected\n"
                "    publication_valid: false\n"
                "    publication_invalid_reason: mixed_generation_protocol_sft_baseline\n"
                "    corrected_effect_pending: true\n"
            )

            issues = module.validate_paths([path])

        self.assertEqual(len(issues), 1)
        self.assertIn("structured publication invalidation", issues[0])

    def test_registry_accepts_corrected_salvage_with_activation_invalidation(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "experiments.yaml"
            path.write_text(
                "experiments:\n"
                "  - id: r33-corrected-k3-hero-lr1e5-update342-resume228-retry3\n"
                "    status: exploratory_corrected_salvage_passed_activation_invalid\n"
                "    publication_valid: false\n"
                "    publication_invalid_reason: stored_activation_identity_failure_and_exploratory_test_exposure\n"
                "    corrected_effect_pending: false\n"
                "    metrics:\n"
                "      corrected_salvage:\n"
                "        cross_critic_gate_passed: true\n"
                "    provenance:\n"
                "      activation_fidelity_publication_ready: false\n"
            )

            issues = module.validate_paths([path])

        self.assertEqual(issues, [])

    def test_repository_contract_rejects_missing_canonical_claims(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for relative in module.CANONICAL_REQUIREMENTS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("placeholder\n")
            registry = root / "runs/registry/experiments.yaml"
            registry.parent.mkdir(parents=True, exist_ok=True)
            registry.write_text("experiments: []\n")

            issues = module.validate_repository_contract(root)

        self.assertTrue(any("missing canonical statement" in issue for issue in issues))
        self.assertTrue(any("missing clean lineage" in issue for issue in issues))

    def test_repository_contract_accepts_current_clean_lineage(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for relative, fragments in module.CANONICAL_REQUIREMENTS.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(fragments) + "\n")
            registry = root / "runs/registry/experiments.yaml"
            registry.parent.mkdir(parents=True, exist_ok=True)
            registry.write_text(
                yaml.safe_dump(
                    {
                        "experiments": [
                            {"id": run_id, "status": status}
                            for run_id, status in module.CLEAN_REGISTRY_STATUSES.items()
                        ]
                    }
                )
            )

            issues = module.validate_repository_contract(root)

        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
