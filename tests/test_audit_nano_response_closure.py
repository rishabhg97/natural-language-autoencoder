import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "audit_nano_response_closure.py"
    spec = importlib.util.spec_from_file_location("audit_nano_response_closure", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


def record(split: str, row_index: int, body_tokens: int, *, closed: bool = True) -> dict:
    close = " </explanation>" if closed else ""
    generated = "<explanation> " + " ".join(["token"] * body_tokens) + close
    return {
        "split": split,
        "row_index": row_index,
        "controls": {
            "real": {
                "generated": generated,
                "parsed": {"closed": closed, "usable": closed},
            }
        },
    }


class AuditNanoResponseClosureTests(unittest.TestCase):
    def test_selects_150_when_at_least_95_percent_close_by_150_on_each_split(self):
        module = load_script()
        rows = []
        for split in ("validation", "test"):
            rows.extend(record(split, index, 100) for index in range(19))
            rows.append(record(split, 19, 170))

        report = module.audit_response_closure(
            rows,
            tokenizer=WhitespaceTokenizer(),
            split_limits={"validation": 20, "test": 20},
            candidate_caps=(150, 192),
            required_fraction=0.95,
        )

        self.assertEqual(report["selected_cap"], 150)
        self.assertEqual(report["splits"]["validation"]["closed_by_cap"]["150"], 0.95)

    def test_falls_back_to_192_and_counts_unclosed_rows_against_fraction(self):
        module = load_script()
        rows = []
        for split in ("validation", "test"):
            rows.extend(record(split, index, 100) for index in range(18))
            rows.append(record(split, 18, 170))
            rows.append(record(split, 19, 100, closed=False))

        report = module.audit_response_closure(
            rows,
            tokenizer=WhitespaceTokenizer(),
            split_limits={"validation": 20, "test": 20},
            candidate_caps=(150, 192),
            required_fraction=0.95,
        )

        self.assertEqual(report["selected_cap"], 192)
        self.assertEqual(report["splits"]["test"]["closed_fraction"], 0.95)
        self.assertEqual(report["splits"]["test"]["closed_by_cap"]["150"], 0.9)


if __name__ == "__main__":
    unittest.main()
