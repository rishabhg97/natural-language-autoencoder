import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoRoundtripTransformTests(unittest.TestCase):
    def test_length_and_canonicalization_transforms_are_explicit(self):
        module = load_script("nano_roundtrip_transforms")
        source = "- **First point** has useful detail.\n- Second point adds context."

        canonical = module.transform_generated_text(
            source, transform="surface_canonicalized", seed=0
        )
        truncated = module.transform_generated_text(
            source, transform="truncate_words_50", seed=0
        )
        dropped = module.transform_generated_text(
            source, transform="drop_last_unit", seed=0
        )

        self.assertEqual(
            canonical,
            "First point has useful detail. Second point adds context.",
        )
        self.assertGreater(len(source.split()), len(truncated.split()))
        self.assertIn("First point", dropped)
        self.assertNotIn("Second point", dropped)

    def test_reorder_units_is_seeded_and_preserves_units(self):
        module = load_script("nano_roundtrip_transforms")
        text = "- first point\n- second point\n- third point"

        one = module.reorder_units(text, seed=19)
        two = module.reorder_units(text, seed=19)

        self.assertEqual(one, two)
        self.assertEqual(sorted(one.splitlines()), sorted(text.splitlines()))
        self.assertNotEqual(one, text)

    def test_transform_generated_text_preserves_explanation_envelope(self):
        module = load_script("nano_roundtrip_transforms")
        generated = "prefix<explanation>First. Second. Third.</explanation>suffix"

        transformed = module.transform_generated_text(
            generated,
            transform="unit_reordered",
            seed=4,
        )

        self.assertTrue(transformed.startswith("prefix<explanation>"))
        self.assertTrue(transformed.endswith("</explanation>suffix"))
        self.assertNotEqual(transformed, generated)

    def test_transform_record_hashes_source(self):
        module = load_script("nano_roundtrip_transforms")
        source = "A sentence.  Another sentence."
        transformed = module.normalize_formatting(source)

        record = module.build_transform_record(
            row_key="doc-1:20",
            source=source,
            transform="format_normalized",
            transformed=transformed,
            seed=0,
        )

        self.assertEqual(record["schema_version"], "nano_roundtrip_transform.v1")
        self.assertEqual(len(record["source_sha256"]), 64)
        self.assertEqual(record["transformed_text"], "A sentence. Another sentence.")

    def test_apply_transform_records_fails_on_missing_or_stale_transform(self):
        module = load_script("nano_roundtrip_transforms")
        generated = "<explanation>First. Second.</explanation>"
        records = [
            {
                "split": "validation",
                "row_index": 3,
                "controls": {"real": {"generated": generated}},
            }
        ]

        with self.assertRaisesRegex(module.TransformError, "missing"):
            module.apply_transform_records(
                records,
                {},
                transform="unit_reordered",
            )

        stale = module.build_transform_record(
            row_key="validation:3",
            source="different",
            transform="unit_reordered",
            transformed="changed",
            seed=0,
        )
        with self.assertRaisesRegex(module.TransformError, "source hash"):
            module.apply_transform_records(
                records,
                {("validation:3", "unit_reordered"): stale},
                transform="unit_reordered",
            )


if __name__ == "__main__":
    unittest.main()
