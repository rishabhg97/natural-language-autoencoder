import os
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
NLA_ROOT = ROOT / "external" / "natural_language_autoencoders"
if str(NLA_ROOT) not in sys.path:
    sys.path.insert(0, str(NLA_ROOT))


class EnvFlagTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("NLA_TEST_FLAG", None)
        os.environ.pop("NLA_TEST_FLOAT", None)

    def test_env_flag_uses_default_when_unset(self):
        from nla.runtime_flags import env_flag

        self.assertTrue(env_flag("NLA_TEST_FLAG", True))
        self.assertFalse(env_flag("NLA_TEST_FLAG", False))

    def test_env_flag_parses_true_false_values(self):
        from nla.runtime_flags import env_flag

        os.environ["NLA_TEST_FLAG"] = "yes"
        self.assertTrue(env_flag("NLA_TEST_FLAG"))

        os.environ["NLA_TEST_FLAG"] = "0"
        self.assertFalse(env_flag("NLA_TEST_FLAG", True))

    def test_env_flag_rejects_ambiguous_values(self):
        from nla.runtime_flags import env_flag

        os.environ["NLA_TEST_FLAG"] = "maybe"
        with self.assertRaises(ValueError):
            env_flag("NLA_TEST_FLAG")

    def test_env_float_uses_default_and_rejects_nonfinite_values(self):
        from nla.runtime_flags import env_float

        self.assertEqual(env_float("NLA_TEST_FLOAT", 0.25), 0.25)

        os.environ["NLA_TEST_FLOAT"] = "0.125"
        self.assertEqual(env_float("NLA_TEST_FLOAT", 0.25), 0.125)

        for invalid in ("nope", "nan", "inf"):
            with self.subTest(invalid=invalid):
                os.environ["NLA_TEST_FLOAT"] = invalid
                with self.assertRaises(ValueError):
                    env_float("NLA_TEST_FLOAT", 0.25)


if __name__ == "__main__":
    unittest.main()
