"""
Unit tests for the attention-from-scratch milestones.

Run from the repository root:

    python -m unittest discover -s tests -v

The tests import the real functions and matrices from src/ and check
mathematical properties (shapes, normalization, masking, invariances) rather
than comparing against hardcoded attention-weight matrices.
"""

import sys
import unittest
from pathlib import Path

import torch

# Make the milestone modules in src/ importable.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import dot_product as m1             # noqa: E402
import scaled_attention as m2        # noqa: E402
import qkv_attention as m3           # noqa: E402
import causal_attention as m4        # noqa: E402
import multi_head_attention as m5    # noqa: E402

TOLERANCE = 1e-6


def make_causal_mask(seq_len):
    """Same construction used in M4 and M5."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


# ---------------------------------------------------------------------------
# A. Dot-product similarity (M1)
# ---------------------------------------------------------------------------
class TestDotProductSimilarity(unittest.TestCase):

    def test_single_dot_product_matches_definition(self):
        manual_scores = m1.manual_dot_product_scores(m1.embeddings)
        for i in range(len(m1.tokens)):
            for j in range(len(m1.tokens)):
                # a · b = sum of element-wise products
                expected = (m1.embeddings[i] * m1.embeddings[j]).sum().item()
                self.assertAlmostEqual(manual_scores[i][j], expected, places=6)

    def test_score_matrix_shape(self):
        scores = m1.embeddings @ m1.embeddings.T
        num_tokens = m1.embeddings.shape[0]
        self.assertEqual(scores.shape, (num_tokens, num_tokens))
        self.assertEqual(len(m1.manual_dot_product_scores(m1.embeddings)), num_tokens)

    def test_manual_matches_matrix_multiplication(self):
        manual = torch.tensor(m1.manual_dot_product_scores(m1.embeddings))
        self.assertTrue(torch.allclose(manual, m1.embeddings @ m1.embeddings.T, atol=TOLERANCE))

    def test_manual_matches_matrix_multiplication_on_other_input(self):
        # A different deterministic input: 4 tokens, 3 dimensions.
        embeddings = torch.arange(12, dtype=torch.float32).reshape(4, 3) - 5.0
        manual = torch.tensor(m1.manual_dot_product_scores(embeddings))
        self.assertEqual(manual.shape, (4, 4))
        self.assertTrue(torch.allclose(manual, embeddings @ embeddings.T, atol=TOLERANCE))

    def test_self_similarity_score_matrix_is_symmetric(self):
        scores = m1.embeddings @ m1.embeddings.T
        self.assertTrue(torch.allclose(scores, scores.T))


# ---------------------------------------------------------------------------
# B. Scaled dot-product attention (M2, using M3's reusable function)
# ---------------------------------------------------------------------------
class TestScaledDotProductAttention(unittest.TestCase):

    def setUp(self):
        self.X = m2.embeddings
        self.raw, self.scaled, self.weights, self.output = \
            m3.scaled_dot_product_attention(self.X, self.X, self.X)

    def test_scaling_by_sqrt_dk(self):
        d_k = self.X.shape[-1]
        self.assertTrue(torch.allclose(self.raw, self.X @ self.X.T))
        self.assertTrue(torch.allclose(self.scaled, self.raw / d_k ** 0.5))

    def test_scaling_uses_key_dimension(self):
        # d_k = 4 here, so the divisor must be 2.
        Q = torch.eye(4)
        K = torch.ones(4, 4)
        raw, scaled, _, _ = m3.scaled_dot_product_attention(Q, K, K)
        self.assertTrue(torch.allclose(scaled, raw / 2.0))

    def test_softmax_rows_sum_to_one(self):
        row_sums = self.weights.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=TOLERANCE))

    def test_weights_are_nonnegative(self):
        self.assertTrue((self.weights >= 0).all())

    def test_manual_softmax_matches_torch_softmax(self):
        self.assertTrue(torch.allclose(m2.manual_softmax(self.scaled),
                                       torch.softmax(self.scaled, dim=-1), atol=TOLERANCE))

    def test_output_is_weighted_sum_of_values(self):
        for i in range(self.X.shape[0]):
            weighted_sum = sum(self.weights[i, j] * self.X[j] for j in range(self.X.shape[0]))
            self.assertTrue(torch.allclose(self.output[i], weighted_sum, atol=TOLERANCE))

    def test_output_dimensions_follow_values(self):
        # Output has one row per query and one column per value feature.
        V = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        _, _, weights, output = m3.scaled_dot_product_attention(self.X, self.X, V)
        self.assertEqual(weights.shape, (3, 3))
        self.assertEqual(output.shape, (3, 3))
        self.assertEqual(self.output.shape, self.X.shape)


# ---------------------------------------------------------------------------
# C. Q/K/V projections (M3)
# ---------------------------------------------------------------------------
class TestQKVProjections(unittest.TestCase):

    def setUp(self):
        self.Q = m3.X @ m3.W_Q
        self.K = m3.X @ m3.W_K
        self.V = m3.X @ m3.W_V

    def test_projection_dimensions(self):
        num_tokens, model_dim = m3.X.shape
        for W in [m3.W_Q, m3.W_K, m3.W_V]:
            self.assertEqual(W.shape[0], model_dim)
        for tensor, W in [(self.Q, m3.W_Q), (self.K, m3.W_K), (self.V, m3.W_V)]:
            self.assertEqual(tensor.shape, (num_tokens, W.shape[1]))

    def test_projection_rows_are_linear_combinations(self):
        # Row i of X @ W equals sum_d X[i, d] * W[d].
        for i in range(m3.X.shape[0]):
            expected = sum(m3.X[i, d] * m3.W_Q[d] for d in range(m3.X.shape[1]))
            self.assertTrue(torch.allclose(self.Q[i], expected))

    def test_separate_projections_produce_distinct_tensors(self):
        self.assertFalse(torch.allclose(self.Q, self.K))
        self.assertFalse(torch.allclose(self.Q, self.V))
        self.assertFalse(torch.allclose(self.K, self.V))

    def test_separate_projections_break_score_symmetry(self):
        raw, _, _, _ = m3.scaled_dot_product_attention(self.Q, self.K, self.V)
        self.assertFalse(torch.allclose(raw, raw.T))

    def test_changing_value_projection_keeps_weights(self):
        W_V_other = m3.W_V * 3.0 + 1.0
        _, _, weights, _ = m3.scaled_dot_product_attention(self.Q, self.K, self.V)
        _, _, weights_other, _ = m3.scaled_dot_product_attention(self.Q, self.K, m3.X @ W_V_other)
        self.assertTrue(torch.allclose(weights, weights_other))

    def test_changing_value_projection_changes_output(self):
        W_V_other = m3.W_V * 3.0 + 1.0
        _, _, _, output = m3.scaled_dot_product_attention(self.Q, self.K, self.V)
        _, _, _, output_other = m3.scaled_dot_product_attention(self.Q, self.K, m3.X @ W_V_other)
        self.assertFalse(torch.allclose(output, output_other))

    def test_changing_query_projection_changes_weights(self):
        Q_other = m3.X @ torch.eye(2)
        _, _, weights, _ = m3.scaled_dot_product_attention(self.Q, self.K, self.V)
        _, _, weights_other, _ = m3.scaled_dot_product_attention(Q_other, self.K, self.V)
        self.assertFalse(torch.allclose(weights, weights_other))


# ---------------------------------------------------------------------------
# D. Causal masking (M4)
# ---------------------------------------------------------------------------
class TestCausalMasking(unittest.TestCase):

    def setUp(self):
        self.seq_len = m4.X.shape[0]
        self.mask = make_causal_mask(self.seq_len)
        self.Q, self.K, self.V = m4.project(m4.X)
        _, self.masked_scores, self.weights, self.output = m4.attention(self.Q, self.K, self.V, mask=self.mask)

    def test_mask_is_lower_triangular_with_queries_as_rows(self):
        for query in range(self.seq_len):
            for key in range(self.seq_len):
                # Allowed exactly when the key is at the same or an earlier position.
                self.assertEqual(bool(self.mask[query, key]), key <= query)

    def test_project_mask_matches_expected_construction(self):
        self.assertTrue(torch.equal(m5.causal_mask, make_causal_mask(m5.X.shape[0])))

    def test_scores_are_scaled_by_sqrt_dk(self):
        scaled_scores, _, _, _ = m4.attention(self.Q, self.K, self.V, mask=self.mask)
        d_k = self.K.shape[-1]
        self.assertTrue(torch.allclose(scaled_scores, (self.Q @ self.K.T) / d_k ** 0.5))

    def test_future_scores_are_negative_infinity(self):
        future = ~self.mask
        self.assertTrue(torch.isneginf(self.masked_scores[future]).all())
        self.assertTrue(torch.isfinite(self.masked_scores[self.mask]).all())

    def test_future_weights_are_exactly_zero(self):
        self.assertTrue((self.weights[~self.mask] == 0).all())

    def test_rows_sum_to_one(self):
        self.assertTrue(torch.allclose(self.weights.sum(dim=-1), torch.ones(self.seq_len), atol=TOLERANCE))

    def test_first_token_attends_only_to_itself(self):
        self.assertTrue(torch.allclose(self.weights[0, 0], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(self.output[0], self.V[0]))

    def test_last_token_is_same_with_or_without_mask(self):
        # The last row has nothing in its future, so masking cannot change it.
        _, _, unmasked_weights, _ = m4.attention(self.Q, self.K, self.V)
        self.assertTrue(torch.allclose(self.weights[-1], unmasked_weights[-1]))

    def test_changing_any_token_never_affects_earlier_outputs(self):
        # For each position p, change that token and check all rows before p.
        for position in range(1, self.seq_len):
            X_changed = m4.X.clone()
            X_changed[position] = torch.tensor([3.0, -2.0])
            _, _, _, output_changed = m4.attention(*m4.project(X_changed), mask=self.mask)
            self.assertTrue(torch.allclose(self.output[:position], output_changed[:position]),
                            f"Changing position {position} leaked into earlier outputs")

    def test_without_mask_future_changes_do_leak(self):
        _, _, _, unmasked = m4.attention(self.Q, self.K, self.V)
        X_changed = m4.X.clone()
        X_changed[-1] = torch.tensor([3.0, 0.0])
        _, _, _, unmasked_changed = m4.attention(*m4.project(X_changed))
        self.assertFalse(torch.allclose(unmasked[:-1], unmasked_changed[:-1]))


# ---------------------------------------------------------------------------
# E. Multi-head attention (M5)
# ---------------------------------------------------------------------------
class TestMultiHeadAttention(unittest.TestCase):

    def setUp(self):
        self.heads, self.concatenated, self.final_output = m5.multi_head_attention(
            m5.X, m5.head_matrices, m5.W_O, m5.causal_mask)
        self.seq_len = m5.X.shape[0]

    def test_heads_are_computed_independently(self):
        # Running a head on its own gives exactly the same result.
        for head, (W_Q, W_K, W_V) in zip(self.heads, m5.head_matrices):
            alone = m5.attention_head(m5.X, W_Q, W_K, W_V, m5.causal_mask)
            self.assertTrue(torch.allclose(head["output"], alone["output"]))
            self.assertTrue(torch.allclose(head["weights"], alone["weights"]))

    def test_each_head_scales_scores_by_sqrt_dk(self):
        for head in self.heads:
            d_k = head["K"].shape[-1]
            expected = (head["Q"] @ head["K"].T) / d_k ** 0.5
            self.assertTrue(torch.allclose(head["scaled_scores"], expected))

    def test_per_head_shapes(self):
        for head, (W_Q, _, W_V) in zip(self.heads, m5.head_matrices):
            self.assertEqual(head["Q"].shape, (self.seq_len, W_Q.shape[1]))
            self.assertEqual(head["scaled_scores"].shape, (self.seq_len, self.seq_len))
            self.assertEqual(head["weights"].shape, (self.seq_len, self.seq_len))
            self.assertEqual(head["output"].shape, (self.seq_len, W_V.shape[1]))

    def test_concatenation_dimensions_and_layout(self):
        total_value_dim = sum(W_V.shape[1] for _, _, W_V in m5.head_matrices)
        self.assertEqual(self.concatenated.shape, (self.seq_len, total_value_dim))
        # Head outputs sit side by side, in order, along the feature axis.
        column = 0
        for head in self.heads:
            width = head["output"].shape[1]
            self.assertTrue(torch.allclose(self.concatenated[:, column:column + width], head["output"]))
            column += width

    def test_output_projection(self):
        self.assertEqual(self.final_output.shape, (self.seq_len, m5.W_O.shape[1]))
        self.assertTrue(torch.allclose(self.final_output, self.concatenated @ m5.W_O))

    def test_heads_have_different_attention_distributions(self):
        self.assertFalse(torch.allclose(self.heads[0]["weights"], self.heads[1]["weights"], atol=1e-2))

    def test_every_head_is_causal_and_normalized(self):
        for head in self.heads:
            weights = head["weights"]
            self.assertTrue((weights[~m5.causal_mask] == 0).all())
            self.assertTrue((weights >= 0).all())
            self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(self.seq_len), atol=TOLERANCE))

    def test_changing_one_heads_values_leaves_other_head_unchanged(self):
        (W_Q1, W_K1, W_V1), head2_matrices = m5.head_matrices
        changed = [(W_Q1, W_K1, W_V1 + 1.0), head2_matrices]
        heads_changed, concatenated_changed, final_changed = m5.multi_head_attention(
            m5.X, changed, m5.W_O, m5.causal_mask)
        # Head 1: same weights, different output.
        self.assertTrue(torch.allclose(self.heads[0]["weights"], heads_changed[0]["weights"]))
        self.assertFalse(torch.allclose(self.heads[0]["output"], heads_changed[0]["output"]))
        # Head 2: completely unaffected.
        self.assertTrue(torch.allclose(self.heads[1]["weights"], heads_changed[1]["weights"]))
        self.assertTrue(torch.allclose(self.heads[1]["output"], heads_changed[1]["output"]))
        # The change reaches the combined representation.
        self.assertFalse(torch.allclose(self.final_output, final_changed))

    def test_output_projection_acts_on_each_row_independently(self):
        # Changing one row of the concatenated output changes only that row of the result.
        modified = self.concatenated.clone()
        modified[2] += 5.0
        result = modified @ m5.W_O
        unchanged_rows = [0, 1, 3]
        self.assertTrue(torch.allclose(result[unchanged_rows], self.final_output[unchanged_rows]))

    def test_masking_survives_concatenation_and_projection(self):
        for position in range(1, self.seq_len):
            X_changed = m5.X.clone()
            X_changed[position] = torch.tensor([1.0, -1.0, 2.0, 0.5])
            heads_changed, concatenated_changed, final_changed = m5.multi_head_attention(
                X_changed, m5.head_matrices, m5.W_O, m5.causal_mask)
            earlier = slice(0, position)
            for head, head_changed in zip(self.heads, heads_changed):
                self.assertTrue(torch.allclose(head["output"][earlier], head_changed["output"][earlier]))
            self.assertTrue(torch.allclose(self.concatenated[earlier], concatenated_changed[earlier]))
            self.assertTrue(torch.allclose(self.final_output[earlier], final_changed[earlier]))


# ---------------------------------------------------------------------------
# F. Numerical stability
# ---------------------------------------------------------------------------
class TestNumericalStability(unittest.TestCase):

    # Large enough that a naive exp() would overflow float32 (max exp input ~88).
    large_scores = torch.tensor([
        [1000.0, 999.0, 998.0],
        [-500.0, 500.0, 0.0],
        [250.0, 250.0, 250.0],
    ])

    def test_manual_and_torch_softmax_stay_finite_and_normalized(self):
        for softmax in [m2.manual_softmax, lambda s: torch.softmax(s, dim=-1)]:
            weights = softmax(self.large_scores)
            self.assertTrue(torch.isfinite(weights).all())
            self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=TOLERANCE))
            self.assertTrue((weights >= 0).all())

    def test_manual_softmax_matches_torch_on_large_scores(self):
        self.assertTrue(torch.allclose(m2.manual_softmax(self.large_scores),
                                       torch.softmax(self.large_scores, dim=-1), atol=TOLERANCE))

    def test_softmax_is_shift_invariant(self):
        # Subtracting a constant from a row doesn't change softmax. This is why
        # subtracting the row maximum is safe.
        scores = torch.tensor([[1.0, 2.0, 3.0]])
        self.assertTrue(torch.allclose(torch.softmax(scores, dim=-1),
                                       torch.softmax(scores + 900.0, dim=-1), atol=TOLERANCE))

    def test_equal_large_scores_give_uniform_weights(self):
        weights = torch.softmax(self.large_scores[2:3], dim=-1)
        self.assertTrue(torch.allclose(weights, torch.full((1, 3), 1 / 3), atol=TOLERANCE))

    def test_causal_attention_with_large_scores(self):
        # Scale up M4's queries and keys so scores reach the hundreds.
        Q, K, V = m4.project(m4.X)
        mask = make_causal_mask(Q.shape[0])
        scaled_scores, _, weights, output = m4.attention(Q * 20.0, K * 20.0, V, mask=mask)
        self.assertGreater(scaled_scores.abs().max().item(), 100.0)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(Q.shape[0]), atol=TOLERANCE))
        self.assertTrue((weights[~mask] == 0).all())


if __name__ == "__main__":
    unittest.main()
