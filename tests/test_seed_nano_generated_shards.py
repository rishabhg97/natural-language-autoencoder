import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "seed_nano_generated_shards.py"
    spec = importlib.util.spec_from_file_location("seed_nano_generated_shards", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SeedNanoGeneratedShardsTests(unittest.TestCase):
    def test_shards_existing_rows_by_target_eval_position(self):
        module = load_script()
        records = [
            {"split": "validation", "row_index": 100},
            {"split": "validation", "row_index": 101},
            {"split": "test", "row_index": 200},
            {"split": "test", "row_index": 201},
        ]

        shards = module.shard_generated_records(
            records,
            split_starts={"validation": 100, "test": 200},
            validation_limit=4,
            test_limit=4,
            shard_count=2,
        )

        self.assertEqual(
            [[record["row_index"] for record in shard] for shard in shards],
            [[100, 200], [101, 201]],
        )

    def test_rejects_duplicate_or_out_of_target_rows(self):
        module = load_script()
        with self.assertRaisesRegex(module.ShardSeedError, "duplicate"):
            module.shard_generated_records(
                [
                    {"split": "validation", "row_index": 100},
                    {"split": "validation", "row_index": 100},
                ],
                split_starts={"validation": 100, "test": 200},
                validation_limit=4,
                test_limit=4,
                shard_count=2,
            )
        with self.assertRaisesRegex(module.ShardSeedError, "outside target"):
            module.shard_generated_records(
                [{"split": "test", "row_index": 205}],
                split_starts={"validation": 100, "test": 200},
                validation_limit=4,
                test_limit=4,
                shard_count=2,
            )


if __name__ == "__main__":
    unittest.main()
