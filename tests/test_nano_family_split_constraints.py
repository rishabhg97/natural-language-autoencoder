import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "nano_functional_eval_data.py"
    spec = importlib.util.spec_from_file_location("nano_functional_eval_data_constraints", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoFamilySplitConstraintTests(unittest.TestCase):
    def test_previously_evaluated_families_are_forbidden_from_test(self):
        module = load_script()
        family_manifest = {
            "families": [
                {"content_family_id": f"family-{index}", "row_count": 10}
                for index in range(30)
            ]
        }
        forbidden = {f"family-{index}": {"test"} for index in range(12)}

        assigned = module.assign_family_splits(
            family_manifest,
            split_weights={"train": 0.8, "validation": 0.1, "test": 0.1},
            seed=20260709,
            forbidden_splits_by_family=forbidden,
        )

        test_families = {
            family_id
            for family_id, split in assigned["family_splits"].items()
            if split == "test"
        }
        self.assertFalse(test_families & set(forbidden))
        self.assertEqual(assigned["split_assignment"]["constraint_family_count"], 12)
        self.assertEqual(set(assigned["family_splits"].values()), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
