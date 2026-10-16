"""
Milestone 4: Causal (masked) self-attention.

GPT-style models generate text one token at a time, left to right. When
predicting the next token, a position must only use itself and the tokens
before it, never tokens that come later. A causal mask enforces this by
blocking every "future" position before softmax.

    scores         = Q @ K.T / sqrt(d_k)
    masked_scores  = scores with future positions set to -inf
    causal_weights = softmax(masked_scores)
    causal_output  = causal_weights @ V
"""

import torch

# Print tensors with 4 decimal places so small differences are visible.
torch.set_printoptions(precision=4)

TOLERANCE = 1e-5

# ---------------------------------------------------------------------------
# 1. Tokens and embeddings
# ---------------------------------------------------------------------------
# Hand-picked educational embeddings, not trained representations.
# "cat", "eats" and "fish" reuse the vectors from M1-M3. "The" gets a small,
# neutral vector because it carries little meaning on its own.
# Shape (4, 2): 4 tokens in sequence order, each a 2-dimensional vector.
tokens = ["The", "cat", "eats", "fish"]
X = torch.tensor([
    [0.5, 0.5],  # "The"   (position 0)
    [1.0, 0.0],  # "cat"   (position 1)
    [0.0, 1.0],  # "eats"  (position 2)
    [1.0, 1.0],  # "fish"  (position 3)
], dtype=torch.float32)

# ---------------------------------------------------------------------------
# 2. Projection matrices (same as M3, fixed, not trained)
# ---------------------------------------------------------------------------
# Query projection: [a, b] -> [b, a]
W_Q = torch.tensor([
    [0.0, 1.0],
    [1.0, 0.0],
])
# Key projection: [a, b] -> [-b, a + 2b]
W_K = torch.tensor([
    [0.0, 1.0],
    [-1.0, 2.0],
])
# Value projection: [a, b] -> [2a, 0.5b]
W_V = torch.tensor([
    [2.0, 0.0],
    [0.0, 0.5],
])


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def project(embeddings):
    """Create queries, keys and values. (4, 2) @ (2, 2) -> (4, 2)."""
    return embeddings @ W_Q, embeddings @ W_K, embeddings @ W_V


def attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention, optionally with a boolean mask.
    mask[i][j] = True means query i may attend to key j.
    Returns (scaled_scores, masked_scores, attention_weights, output).
    """
    # Attention SCORES: how well each query matches each key. (4, 4)
    d_k = K.shape[-1]
    scaled_scores = (Q @ K.T) / (d_k ** 0.5)

    # Masking happens BEFORE softmax. Blocked positions get -inf, and
    # exp(-inf) = 0, so softmax gives them exactly zero weight while the
    # remaining allowed weights in the row still sum to 1.
    if mask is None:
        masked_scores = scaled_scores
    else:
        masked_scores = scaled_scores.masked_fill(~mask, float("-inf"))

    # Attention WEIGHTS: each row is a probability distribution over keys. (4, 4)
    attention_weights = torch.softmax(masked_scores, dim=-1)

    # Contextual OUTPUT: weighted sum of value vectors for each token. (4, 2)
    output = attention_weights @ V
    return scaled_scores, masked_scores, attention_weights, output


def print_tensor(label, tensor):
    print(f"--- {label} | shape {tuple(tensor.shape)} ---")
    print(tensor)


def print_matrix_table(label, matrix):
    """Row = query token, column = key token."""
    column_width = 8
    print(f"--- {label} (row = query, column = key) ---")
    print(" " * column_width + "".join(f"{token:>{column_width}}" for token in tokens))
    for row_index, row_token in enumerate(tokens):
        cells = []
        for value in matrix[row_index]:
            if isinstance(value.item(), bool):
                cells.append(f"{str(value.item()):>{column_width}}")
            else:
                cells.append(f"{value.item():>{column_width}.4f}")
        print(f"{row_token:<{column_width}}" + "".join(cells))


def print_output_comparison(label_a, output_a, label_b, output_b):
    print(f"{'token':<6} {label_a:>22} {label_b:>22}   changed?")
    for index, token in enumerate(tokens):
        changed = not torch.allclose(output_a[index], output_b[index], atol=TOLERANCE)
        print(f"{token:<6} {str([round(v, 4) for v in output_a[index].tolist()]):>22} "
              f"{str([round(v, 4) for v in output_b[index].tolist()]):>22}   {'YES' if changed else 'no'}")


def main():
    # -----------------------------------------------------------------------
    # 3. Q, K, V
    # -----------------------------------------------------------------------
    Q, K, V = project(X)

    print("=================== 1. Inputs and projections ===================")
    print("Tokens (in order):", tokens)
    print_tensor("Embeddings X", X)
    print_tensor("Q = X @ W_Q", Q)
    print_tensor("K = X @ W_K", K)
    print_tensor("V = X @ W_V", V)
    print()

    # -----------------------------------------------------------------------
    # 4. Unmasked attention baseline
    # -----------------------------------------------------------------------
    # Without a mask, every token attends to every token, including tokens that
    # come LATER in the sequence (the future).
    scaled_scores, _, unmasked_weights, unmasked_output = attention(Q, K, V)

    print("=================== 2. Unmasked attention (can see the future) ===================")
    print_tensor("Raw scores Q @ K.T", Q @ K.T)
    print_tensor("Scaled scores", scaled_scores)
    print_matrix_table("Unmasked attention WEIGHTS", unmasked_weights)
    print_tensor("Unmasked contextual OUTPUT", unmasked_output)
    print()

    # -----------------------------------------------------------------------
    # 5. The causal mask
    # -----------------------------------------------------------------------
    # torch.tril keeps the lower triangle (on and below the diagonal).
    # Rows = query positions, columns = key positions.
    # True  = attention allowed (key is at the same or an earlier position).
    # False = attention blocked (key is in the future).
    seq_len = X.shape[0]
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    print("=================== 3. Causal mask ===================")
    print_tensor("causal_mask", causal_mask)
    print_matrix_table("Causal mask (True = allowed)", causal_mask)
    print("Allowed positions per token:")
    for row_index, row_token in enumerate(tokens):
        allowed_tokens = [tokens[j] for j in range(seq_len) if causal_mask[row_index][j]]
        print(f"  {row_token:<5} -> {', '.join(allowed_tokens)}")
    print()

    # -----------------------------------------------------------------------
    # 6. Causal (masked) attention
    # -----------------------------------------------------------------------
    _, masked_scores, causal_weights, causal_output = attention(Q, K, V, mask=causal_mask)

    print("=================== 4. Causal attention ===================")
    print_tensor("Masked SCORES (future = -inf)", masked_scores)
    print_matrix_table("Causal attention WEIGHTS", causal_weights)
    print_tensor("Causal contextual OUTPUT", causal_output)
    print()

    # -----------------------------------------------------------------------
    # 7. Masked vs unmasked comparison
    # -----------------------------------------------------------------------
    print("=================== 5. Unmasked vs causal ===================")
    print_matrix_table("Unmasked WEIGHTS", unmasked_weights)
    print_matrix_table("Causal WEIGHTS", causal_weights)
    print("Contextual OUTPUT comparison:")
    print_output_comparison("unmasked", unmasked_output, "causal", causal_output)
    print("(Only the last token sees the full sequence in both cases, so its output is identical.)")
    print()

    # -----------------------------------------------------------------------
    # 8. Future-token leakage experiment
    # -----------------------------------------------------------------------
    # Change ONLY the last token's embedding. Projections, the mask, and the
    # earlier embeddings stay the same. If causal masking works, the outputs of
    # "The", "cat" and "eats" must not move, because they never look at "fish".
    X_modified = X.clone()
    X_modified[3] = torch.tensor([3.0, 0.0])  # a very different "fish"

    Q_modified, K_modified, V_modified = project(X_modified)
    _, _, causal_weights_modified, causal_output_modified = attention(
        Q_modified, K_modified, V_modified, mask=causal_mask)
    _, _, unmasked_weights_modified, unmasked_output_modified = attention(
        Q_modified, K_modified, V_modified)

    print("=================== 6. Future-token leakage experiment ===================")
    print("Original 'fish' embedding:", X[3].tolist(), "-> modified:", X_modified[3].tolist())
    print_tensor("Modified K (only the last row changes)", K_modified)
    print_tensor("Modified V (only the last row changes)", V_modified)
    print()
    print("CAUSAL output before vs after changing 'fish':")
    print_output_comparison("before", causal_output, "after", causal_output_modified)
    print()
    print("UNMASKED output before vs after changing 'fish':")
    print_output_comparison("before", unmasked_output, "after", unmasked_output_modified)
    print()
    print_matrix_table("Unmasked WEIGHTS after changing 'fish'", unmasked_weights_modified)
    print_matrix_table("Causal WEIGHTS after changing 'fish'", causal_weights_modified)
    print()

    # -----------------------------------------------------------------------
    # 9. Verification
    # -----------------------------------------------------------------------
    future_positions = ~causal_mask
    earlier_tokens = slice(0, 3)  # "The", "cat", "eats"

    # 1-5. Shapes
    assert X.shape == (4, 2), "Input should be (4, 2)"
    for name, matrix in [("Q", Q), ("K", K), ("V", V)]:
        assert matrix.shape == (4, 2), f"{name} should be (4, 2)"
    assert scaled_scores.shape == (4, 4), "Scores should be (4, 4)"
    assert causal_mask.shape == (4, 4), "Mask should be (4, 4)"
    assert causal_output.shape == (4, 2) and unmasked_output.shape == (4, 2), "Output should be (4, 2)"

    # 6-8. Mask structure
    assert torch.equal(causal_mask, torch.tril(causal_mask)), "Mask should be lower triangular"
    assert causal_mask.diagonal().all(), "Every diagonal entry should be True"
    assert not causal_mask[torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)].any(), \
        "Every future position should be False"

    # 9-12. Causal weights
    for weights in [causal_weights, causal_weights_modified]:
        assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=TOLERANCE), "Rows should sum to 1"
        assert (weights >= 0).all(), "Weights should be nonnegative"
        assert (weights[future_positions] == 0).all(), "Future positions should get exactly zero weight"
        assert torch.allclose(weights[0, 0], torch.tensor(1.0)), "First token should attend only to itself"

    # 13. No leakage under causal masking
    assert torch.allclose(causal_output[earlier_tokens], causal_output_modified[earlier_tokens], atol=TOLERANCE), \
        "Changing the last token must not change earlier causal outputs"

    # 14. Leakage without a mask
    assert not torch.allclose(unmasked_output[earlier_tokens], unmasked_output_modified[earlier_tokens], atol=TOLERANCE), \
        "Changing the last token should change some earlier unmasked output"

    # 15. Masking makes a difference
    assert not torch.allclose(causal_output, unmasked_output, atol=TOLERANCE), \
        "Causal and unmasked outputs should differ"

    print("=================== Verification ===================")
    print("All checks passed.")


if __name__ == "__main__":
    main()
