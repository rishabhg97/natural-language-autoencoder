import pathlib
import sys
import unittest

import pytest

torch = pytest.importorskip("torch")


ROOT = pathlib.Path(__file__).resolve().parents[1]
NLA_ROOT = ROOT / "external" / "natural_language_autoencoders"
if str(NLA_ROOT) not in sys.path:
    sys.path.insert(0, str(NLA_ROOT))

from nla.injection import inject_at_marked_positions  # noqa: E402


class NanoMilesInjectionTests(unittest.TestCase):
    def test_single_row_injection_overwrites_marker_embedding(self):
        input_ids = torch.tensor([[10, 20, 30, 40]])
        embeddings = torch.zeros(1, 4, 3)
        vector = torch.tensor([[1.0, 2.0, 3.0]])

        out = inject_at_marked_positions(input_ids, embeddings, vector, inj_id=20, left_id=10, right_id=30)

        torch.testing.assert_close(out[0, 1], vector[0])
        torch.testing.assert_close(out[0, 0], embeddings[0, 0])

    def test_batched_injection_preserves_row_specific_vectors(self):
        input_ids = torch.tensor(
            [
                [10, 20, 30, 0],
                [1, 10, 20, 30],
                [10, 20, 30, 2],
                [3, 10, 20, 30],
            ]
        )
        embeddings = torch.zeros(4, 4, 2)
        vectors = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])

        out = inject_at_marked_positions(input_ids, embeddings, vectors, inj_id=20, left_id=10, right_id=30)

        torch.testing.assert_close(out[0, 1], vectors[0])
        torch.testing.assert_close(out[1, 2], vectors[1])
        torch.testing.assert_close(out[2, 1], vectors[2])
        torch.testing.assert_close(out[3, 2], vectors[3])

    def test_shuffling_vectors_changes_corresponding_injected_rows(self):
        input_ids = torch.tensor([[10, 20, 30], [10, 20, 30]])
        embeddings = torch.zeros(2, 3, 2)
        vectors = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
        shuffled = vectors.flip(0)

        real = inject_at_marked_positions(input_ids, embeddings, vectors, inj_id=20, left_id=10, right_id=30)
        permuted = inject_at_marked_positions(input_ids, embeddings, shuffled, inj_id=20, left_id=10, right_id=30)

        torch.testing.assert_close(real[0, 1], vectors[0])
        torch.testing.assert_close(permuted[0, 1], vectors[1])
        self.assertFalse(torch.equal(real[:, 1], permuted[:, 1]))

    def test_marker_with_wrong_neighbors_is_ignored_and_count_mismatch_raises(self):
        input_ids = torch.tensor([[99, 20, 30]])
        embeddings = torch.zeros(1, 3, 2)
        vectors = torch.tensor([[1.0, 2.0]])

        with self.assertRaisesRegex(RuntimeError, "found 0 injection sites"):
            inject_at_marked_positions(input_ids, embeddings, vectors, inj_id=20, left_id=10, right_id=30)

    def test_valid_marker_count_mismatch_raises(self):
        input_ids = torch.tensor([[10, 20, 30]])
        embeddings = torch.zeros(1, 3, 2)
        vectors = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        with self.assertRaisesRegex(RuntimeError, "found 1 injection sites"):
            inject_at_marked_positions(input_ids, embeddings, vectors, inj_id=20, left_id=10, right_id=30)

    def test_seq_slice_writes_only_local_embedding_positions(self):
        input_ids = torch.tensor([[7, 10, 20, 30, 8]])
        embeddings = torch.zeros(1, 2, 2)
        vectors = torch.tensor([[5.0, 6.0]])

        out = inject_at_marked_positions(
            input_ids,
            embeddings,
            vectors,
            inj_id=20,
            left_id=10,
            right_id=30,
            seq_slice=(2, 4),
        )

        torch.testing.assert_close(out[0, 0], vectors[0])
        torch.testing.assert_close(out[0, 1], embeddings[0, 1])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cpu_input_ids_can_inject_cuda_embeddings(self):
        input_ids = torch.tensor([[10, 20, 30]])
        embeddings = torch.zeros(1, 3, 2, device="cuda")
        vectors = torch.tensor([[5.0, 6.0]])

        out = inject_at_marked_positions(input_ids, embeddings, vectors, inj_id=20, left_id=10, right_id=30)

        self.assertEqual(out.device.type, "cuda")
        torch.testing.assert_close(out[0, 1].cpu(), vectors[0])


if __name__ == "__main__":
    unittest.main()
