import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "nano_functional_eval_data.py"
    spec = importlib.util.spec_from_file_location("nano_functional_eval_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_deterministic_closure_catches_pairs_from_oversized_buckets():
    module = load_module()
    common = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    rows = [
        {"doc_id": "doc-a", "source_text": common},
        {"doc_id": "doc-b", "source_text": common + " mu"},
        {"doc_id": "doc-c", "source_text": common + " nu"},
        {"doc_id": "doc-d", "source_text": "unrelated words for another topic"},
    ]

    manifest = module.build_content_families(
        rows,
        text_field="source_text",
        shingle_width=2,
        similarity_threshold=0.8,
        signature_size=32,
        candidate_min_shared=4,
        max_signature_bucket_size=2,
    )

    assignments = manifest["doc_assignments"]
    assert assignments["doc-a"] == assignments["doc-b"]
    assert assignments["doc-a"] == assignments["doc-c"]
    assert assignments["doc-a"] != assignments["doc-d"]
    assert manifest["stats"]["skipped_oversized_signature_buckets"] > 0
    assert manifest["stats"]["deterministic_threshold_pairs_evaluated"] > 0
    assert (
        manifest["algorithm"]["exact_threshold_closure"]
        == "deterministic_prefix_filter"
    )
