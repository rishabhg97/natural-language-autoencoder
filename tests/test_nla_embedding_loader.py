import pathlib
import sys
import unittest

import pytest

pytest.importorskip("torch")


ROOT = pathlib.Path(__file__).resolve().parents[1]
NLA_ROOT = ROOT / "external" / "natural_language_autoencoders"
if str(NLA_ROOT) not in sys.path:
    sys.path.insert(0, str(NLA_ROOT))


class NLAEmbeddingLoaderTests(unittest.TestCase):
    def test_find_embed_key_accepts_nemotron_backbone_embeddings(self):
        from nla.models import _find_embed_key

        self.assertEqual(
            _find_embed_key(
                [
                    "backbone.layers.0.mixer.in_proj.weight",
                    "backbone.embeddings.weight",
                    "backbone.norm_f.weight",
                ],
                "fake-index",
            ),
            "backbone.embeddings.weight",
        )

    def test_find_embed_key_rejects_ambiguous_input_embeddings(self):
        from nla.models import _find_embed_key

        with self.assertRaisesRegex(AssertionError, "expected exactly one"):
            _find_embed_key(
                [
                    "model.embed_tokens.weight",
                    "backbone.embeddings.weight",
                ],
                "fake-index",
            )


if __name__ == "__main__":
    unittest.main()
