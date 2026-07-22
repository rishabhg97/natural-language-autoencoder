import importlib
import pathlib
import sys
import types
import unittest

import pytest

pytest.importorskip("torch")


ROOT = pathlib.Path(__file__).resolve().parents[1]
NLA_ROOT = ROOT / "external" / "natural_language_autoencoders"
if str(NLA_ROOT) not in sys.path:
    sys.path.insert(0, str(NLA_ROOT))


class ExplanationParsingTests(unittest.TestCase):
    def test_extract_explanation_accepts_body_with_close_tag_only(self):
        from nla.schema import extract_explanation

        response = (
            "Syntax/continuation feature: expects a verb phrase.\n\n"
            "Final-token constraint: adverb often expects a following verb.\n"
            "</explanation><|im_end|>"
        )

        self.assertEqual(
            extract_explanation(response),
            "Syntax/continuation feature: expects a verb phrase.\n\n"
            "Final-token constraint: adverb often expects a following verb.",
        )

    def test_extract_explanation_still_rejects_unclosed_text(self):
        from nla.schema import extract_explanation

        self.assertIsNone(
            extract_explanation("Syntax/continuation feature: no closing tag here")
        )


def _install_reward_import_stubs() -> None:
    modules = {
        "ray": types.ModuleType("ray"),
        "miles": types.ModuleType("miles"),
        "miles.utils": types.ModuleType("miles.utils"),
        "miles.utils.processing_utils": types.ModuleType("miles.utils.processing_utils"),
        "miles.utils.types": types.ModuleType("miles.utils.types"),
    }
    modules["miles.utils.processing_utils"].load_tokenizer = lambda *args, **kwargs: None
    modules["miles.utils.types"].Sample = object
    sys.modules.update(modules)


class RewardTokenizerTests(unittest.TestCase):
    def setUp(self):
        _install_reward_import_stubs()
        sys.modules.pop("nla.reward", None)
        self.reward = importlib.import_module("nla.reward")

    def test_ensure_padding_token_uses_eos_when_missing(self):
        class Tokenizer:
            pad_token = None
            pad_token_id = None
            eos_token = "<eos>"
            eos_token_id = 42
            unk_token = "<unk>"
            unk_token_id = 7

        tokenizer = Tokenizer()

        self.reward._ensure_padding_token(tokenizer)

        self.assertEqual(tokenizer.pad_token, "<eos>")
        self.assertEqual(tokenizer.pad_token_id, 42)


if __name__ == "__main__":
    unittest.main()
