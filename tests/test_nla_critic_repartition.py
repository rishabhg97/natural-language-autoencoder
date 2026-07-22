import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "external"
    / "natural_language_autoencoders"
    / "nla"
    / "critic_repartition.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("critic_repartition", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def actor_partitions(*, missing=()):
    missing = set(missing)
    total_lengths = [100 + i for i in range(24)]
    partitions = []
    for actor_rank in range(4):
        indices = list(range(actor_rank * 6, (actor_rank + 1) * 6))
        multimodal = []
        for index in indices:
            row = {"activation": f"h{index}"}
            if index not in missing:
                row["critic_tokens"] = f"tokens{index}"
            multimodal.append(row)
        partitions.append(
            {
                "partition": indices,
                "total_lengths": total_lengths,
                "tokens": [f"actor{index}" for index in indices],
                "rewards": [float(index) for index in indices],
                "raw_reward": [float(index) / 10.0 for index in range(24)],
                "multimodal_train_inputs": multimodal,
                "scalar_metadata": "same",
            }
        )
    return partitions


class CriticRepartitionTests(unittest.TestCase):
    def test_balances_all_rows_across_asymmetric_dp(self):
        module = load_module()
        shards = []
        reports = []
        for rank in range(3):
            shard, report = module.balance_critic_partition(
                actor_partitions(),
                critic_rank=rank,
                critic_dp=3,
                alignment=2,
                required_multimodal_key="critic_tokens",
            )
            shards.append(shard)
            reports.append(report)

        self.assertEqual([len(shard["partition"]) for shard in shards], [8, 8, 8])
        self.assertEqual(
            sorted(index for shard in shards for index in shard["partition"]),
            list(range(24)),
        )
        self.assertTrue(all(report["retained_samples"] == 24 for report in reports))
        self.assertTrue(
            all(report["retained_fraction_of_usable"] == 1.0 for report in reports)
        )
        self.assertTrue(
            all(
                shard["raw_reward"] == [float(index) / 10.0 for index in range(24)]
                for shard in shards
            )
        )
        self.assertTrue(
            all(report["replicated_global_fields"] == "raw_reward" for report in reports)
        )

    def test_filters_invalid_rows_then_aligns_once_globally(self):
        module = load_module()
        shards = []
        report = None
        for rank in range(3):
            shard, report = module.balance_critic_partition(
                actor_partitions(missing={0, 7, 14}),
                critic_rank=rank,
                critic_dp=3,
                alignment=2,
                required_multimodal_key="critic_tokens",
            )
            shards.append(shard)

        assert report is not None
        self.assertEqual([len(shard["partition"]) for shard in shards], [6, 6, 6])
        self.assertEqual(report["usable_samples"], 21)
        self.assertEqual(report["retained_samples"], 18)
        self.assertEqual(report["dropped_unusable_samples"], 3)
        self.assertEqual(report["dropped_alignment_samples"], 3)
        self.assertAlmostEqual(report["retained_fraction_of_usable"], 18 / 21)

    def test_rejects_inconsistent_sample_field_lengths(self):
        module = load_module()
        partitions = actor_partitions()
        partitions[2]["tokens"].pop()

        with self.assertRaisesRegex(module.CriticRepartitionError, "field 'tokens'"):
            module.balance_critic_partition(
                partitions,
                critic_rank=0,
                critic_dp=3,
                alignment=2,
                required_multimodal_key="critic_tokens",
            )

    def test_rejects_inconsistent_replicated_global_field(self):
        module = load_module()
        partitions = actor_partitions()
        partitions[2]["raw_reward"][5] = -123.0

        with self.assertRaisesRegex(
            module.CriticRepartitionError,
            "inconsistent replicated global field 'raw_reward'",
        ):
            module.balance_critic_partition(
                partitions,
                critic_rank=0,
                critic_dp=3,
                alignment=2,
                required_multimodal_key="critic_tokens",
            )

    def test_minimum_retention_gate_fails_closed(self):
        module = load_module()
        _, report = module.balance_critic_partition(
            actor_partitions(missing={0, 7, 14}),
            critic_rank=0,
            critic_dp=3,
            alignment=2,
            required_multimodal_key="critic_tokens",
        )

        with self.assertRaisesRegex(
            module.CriticRepartitionError, "below the configured minimum"
        ):
            module.require_minimum_retained_fraction(report, 0.95)
        module.require_minimum_retained_fraction(report, 0.85)


if __name__ == "__main__":
    unittest.main()
