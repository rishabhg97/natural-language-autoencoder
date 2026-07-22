import math
import unittest
from types import SimpleNamespace

from nla.lr_policy import apply_fsdp_live_lr_policy


class _FakeScheduler:
    def __init__(self, optimizer, *, last_epoch: int) -> None:
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.max_lr = 5e-5
        self.min_lr = 5e-6
        self.lr_warmup_steps = 25
        self.lr_decay_steps = 1289
        self.lr_decay_style = "cosine"

    def get_lr(self):
        lrs = []
        for group in self.optimizer.param_groups:
            max_lr = group.get("max_lr", self.max_lr)
            min_lr = group.get("min_lr", self.min_lr)
            if self.lr_warmup_steps and self.last_epoch <= self.lr_warmup_steps:
                lr = max_lr * self.last_epoch / self.lr_warmup_steps
            elif self.lr_decay_style == "constant":
                lr = max_lr
            else:
                ratio = (self.last_epoch - self.lr_warmup_steps) / (
                    self.lr_decay_steps - self.lr_warmup_steps
                )
                lr = min_lr + 0.5 * (math.cos(math.pi * ratio) + 1.0) * (
                    max_lr - min_lr
                )
            lrs.append(lr)
        return lrs


class FSDPLiveLRPolicyTests(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "lr": 5e-5,
            "min_lr": 5e-6,
            "lr_decay_style": "cosine",
            "load": "/checkpoint",
            "finetune": False,
            "no_load_optim": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_cosine_resume_recomputes_lr_at_restored_epoch(self):
        optimizer = SimpleNamespace(param_groups=[{"lr": 5e-5}])
        scheduler = _FakeScheduler(optimizer, last_epoch=393)

        policy = apply_fsdp_live_lr_policy(
            scheduler,
            optimizer,
            self._args(),
            environ={},
        )

        self.assertEqual(policy["style"], "cosine")
        self.assertTrue(policy["resumed_optimizer"])
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 4.12266907852368e-5)

    def test_explicit_force_constant_is_the_only_resume_safety_override(self):
        optimizer = SimpleNamespace(param_groups=[{"lr": 4.1e-5}])
        scheduler = _FakeScheduler(optimizer, last_epoch=393)

        policy = apply_fsdp_live_lr_policy(
            scheduler,
            optimizer,
            self._args(),
            environ={"NLA_FORCE_CONSTANT_LR": "1"},
        )

        self.assertEqual(policy["style"], "constant")
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-5)

    def test_fresh_warmup_keeps_scheduler_initial_lr(self):
        optimizer = SimpleNamespace(param_groups=[{"lr": 5e-5}])
        scheduler = _FakeScheduler(optimizer, last_epoch=0)

        policy = apply_fsdp_live_lr_policy(
            scheduler,
            optimizer,
            self._args(load=None, no_load_optim=True),
            environ={},
        )

        self.assertFalse(policy["resumed_optimizer"])
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)


if __name__ == "__main__":
    unittest.main()
