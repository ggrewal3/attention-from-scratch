"""
Milestone 5: Multi-head causal self-attention.

Instead of one attention calculation, we run several "heads" in parallel.
Each head has its own Q/K/V projections, so it can compute a different
attention pattern over the same input. The head outputs are concatenated and
mixed by a final output projection W_O.

    head_h       = softmax(Q_h @ K_h.T / sqrt(d_k) + M) @ V_h     (M = causal mask)
    concatenated = [head_1, head_2]
    final_output = concatenated @ W_O

All matrices are hand-picked. In a trained transformer they would be learned.
"""

import torch

# Print tensors with 4 decimal places so small differences are visible.
torch.set_printoptions(precision=4)

TOLERANCE = 1e-5

# ---------------------------------------------------------------------------
# 1. Tokens and input embeddings
# ---------------------------------------------------------------------------
# Simple one-hot educational embeddings, not trained semantic representations.
# Shape (4, 4): 4 tokens (sequence length) x 4 features (model dimension).
# Because X is the identity matrix, X @ W simply returns W, so row i of each
# projection matrix below IS the query/key/value vector for token i.
tokens = ["The", "cat", "eats", "fish"]
X = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],  # "The"
    [0.0, 1.0, 0.0, 0.0],  # "cat"
    [0.0, 0.0, 1.0, 0.0],  # "eats"
    [0.0, 0.0, 0.0, 1.0],  # "fish"
], dtype=torch.float32)

seq_len, model_dim = X.shape
num_heads = 2
head_dim = 2  # d_k = d_v = model_dim / num_heads

# ---------------------------------------------------------------------------
# 2. Projection matrices for each head, shape (4, 2)
# ---------------------------------------------------------------------------
# Each head projects the SAME 4-dim input into its OWN 2-dim query, key and
# value spaces. Different projections = different notions of "what matches"
# (Q, K) and "what information to pass along" (V).
#
# The patterns below are built in by hand to be easy to see. They are not
# learned, and they carry no real linguistic meaning.

# Head 1: each query points in the same direction as its own key, so each
# token's best match among the allowed positions is ITSELF.
W_Q1 = torch.tensor([
    [2.0, 0.0],   # The
    [0.0, 2.0],   # cat
    [-2.0, 0.0],  # eats
    [0.0, -2.0],  # fish
])
W_K1 = torch.tensor([
    [1.0, 0.0],   # The
    [0.0, 1.0],   # cat
    [-1.0, 0.0],  # eats
    [0.0, -1.0],  # fish
])
W_V1 = torch.tensor([
    [1.0, 0.0],   # The
    [0.0, 1.0],   # cat
    [1.0, 1.0],   # eats
    [0.0, 2.0],   # fish
])

# Head 2: each query points in the direction of the PREVIOUS token's key,
# so each token's best match is the token right before it.
# ("The" has no previous token. The causal mask leaves it only itself.)
W_Q2 = torch.tensor([
    [0.0, 2.0],   # The   (any value works: only "The" is visible)
    [0.0, 2.0],   # cat   -> matches key of "The"
    [-2.0, 0.0],  # eats  -> matches key of "cat"
    [0.0, -2.0],  # fish  -> matches key of "eats"
])
W_K2 = torch.tensor([
    [0.0, 1.0],   # The
    [-1.0, 0.0],  # cat
    [0.0, -1.0],  # eats
    [1.0, 0.0],   # fish
])
W_V2 = torch.tensor([
    [0.0, 2.0],   # The
    [1.0, 0.0],   # cat
    [2.0, 0.0],   # eats
    [1.0, 1.0],   # fish
])

# Output projection W_O, shape (4, 4). Concatenated features are
# [head1_a, head1_b, head2_a, head2_b]. This W_O produces
#   [head1_a + head2_a,  head1_b + head2_b,  head1_a - head2_a,  head1_b - head2_b]
# so every output feature mixes information from BOTH heads.
W_O = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, -1.0],
])

# Causal mask, shared by both heads. True = allowed (same or earlier position).
causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

# Each head's (W_Q, W_K, W_V). Importable by other scripts (e.g. M6 visualization).
head_matrices = [(W_Q1, W_K1, W_V1), (W_Q2, W_K2, W_V2)]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def attention_head(X, W_Q, W_K, W_V, mask):
    """
    One independent causal attention head. Returns a dict of every step.
    """
    # Q/K/V projections: (4, 4) @ (4, 2) -> (4, 2)
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    # Scaled attention SCORES: (4, 2) @ (2, 4) -> (4, 4), divided by sqrt(d_k)
    d_k = K.shape[-1]
    scaled_scores = (Q @ K.T) / (d_k ** 0.5)

    # Causal masking: future positions get -inf BEFORE softmax.
    # Each head has its own (4, 4) score matrix, so the mask is applied to each
    # one separately. Every head must obey the "no peeking ahead" rule.
    masked_scores = scaled_scores.masked_fill(~mask, float("-inf"))

    # Softmax across keys (dim=-1): exp(-inf) = 0, so future weights are exactly 0.
    weights = torch.softmax(masked_scores, dim=-1)

    # Value aggregation: each token's output is a weighted sum of value vectors.
    # (4, 4) @ (4, 2) -> (4, 2)
    output = weights @ V

    return {"Q": Q, "K": K, "V": V, "scaled_scores": scaled_scores,
            "masked_scores": masked_scores, "weights": weights, "output": output}


def multi_head_attention(X, head_matrices, W_O, mask):
    """
    Run every head independently, concatenate their outputs, then project.
    head_matrices is a list of (W_Q, W_K, W_V) tuples, one per head.
    """
    heads = [attention_head(X, W_Q, W_K, W_V, mask) for W_Q, W_K, W_V in head_matrices]

    # Concatenation: place head outputs side by side along the feature axis.
    # (4, 2) + (4, 2) -> (4, 4). Each head's features keep their own columns.
    concatenated = torch.cat([head["output"] for head in heads], dim=-1)

    # Output projection: mixes the heads' features, one token (row) at a time.
    # (4, 4) @ (4, 4) -> (4, 4). It never mixes different sequence positions.
    final_output = concatenated @ W_O
    return heads, concatenated, final_output


def print_tensor(label, tensor):
    print(f"--- {label} | shape {tuple(tensor.shape)} ---")
    print(tensor)


def print_weight_table(label, weights):
    column_width = 8
    print(f"--- {label} (row = query, column = key) ---")
    print(" " * column_width + "".join(f"{token:>{column_width}}" for token in tokens))
    for row_index, row_token in enumerate(tokens):
        print(f"{row_token:<{column_width}}"
              + "".join(f"{value.item():>{column_width}.4f}" for value in weights[row_index]))


def print_row_comparison(label_a, tensor_a, label_b, tensor_b):
    print(f"{'token':<6} {label_a:>34} {label_b:>34}   changed?")
    for index, token in enumerate(tokens):
        changed = not torch.allclose(tensor_a[index], tensor_b[index], atol=TOLERANCE)
        row_a = str([round(v, 4) for v in tensor_a[index].tolist()])
        row_b = str([round(v, 4) for v in tensor_b[index].tolist()])
        print(f"{token:<6} {row_a:>34} {row_b:>34}   {'YES' if changed else 'no'}")



def main():
    # ---------------------------------------------------------------------------
    # 3. Run multi-head attention
    # ---------------------------------------------------------------------------
    heads, concatenated, final_output = multi_head_attention(X, head_matrices, W_O, causal_mask)
    head1, head2 = heads

    print("=================== 1. Input ===================")
    print("Tokens:", tokens)
    print_tensor("Input embeddings X", X)
    print(f"seq_len = {seq_len}, model_dim = {model_dim}, num_heads = {num_heads}, head_dim = {head_dim}")
    print_tensor("Causal mask (shared by both heads)", causal_mask)
    print()

    for head_number, (head, (W_Q, W_K, W_V)) in enumerate(zip(heads, head_matrices), start=1):
        print(f"=================== {head_number + 1}. Head {head_number} ===================")
        print_tensor(f"W_Q{head_number}", W_Q)
        print_tensor(f"W_K{head_number}", W_K)
        print_tensor(f"W_V{head_number}", W_V)
        print_tensor(f"Q{head_number} = X @ W_Q{head_number}", head["Q"])
        print_tensor(f"K{head_number} = X @ W_K{head_number}", head["K"])
        print_tensor(f"V{head_number} = X @ W_V{head_number}", head["V"])
        print_tensor(f"Scaled SCORES Q{head_number} @ K{head_number}.T / sqrt(2)", head["scaled_scores"])
        print_tensor("Masked SCORES (future = -inf)", head["masked_scores"])
        print_weight_table(f"Head {head_number} attention WEIGHTS", head["weights"])
        print_tensor(f"Head {head_number} OUTPUT = weights @ V{head_number}", head["output"])
        print()

    # ---------------------------------------------------------------------------
    # 4. Compare the heads
    # ---------------------------------------------------------------------------
    print("=================== 4. Head comparison ===================")
    print_weight_table("Head 1 WEIGHTS", head1["weights"])
    print_weight_table("Head 2 WEIGHTS", head2["weights"])
    print("Most-attended token per head:")
    print(f"  {'token':<6} {'head 1':<8} {'head 2':<8}")
    for row_index, token in enumerate(tokens):
        top_1 = tokens[head1["weights"][row_index].argmax().item()]
        top_2 = tokens[head2["weights"][row_index].argmax().item()]
        print(f"  {token:<6} {top_1:<8} {top_2:<8}")
    print("(Head 1 was built to favor the token itself, head 2 the previous token.)")
    print()

    # ---------------------------------------------------------------------------
    # 5. Concatenation and output projection
    # ---------------------------------------------------------------------------
    print("=================== 5. Concatenate and project ===================")
    print("Head 1 output shape:", tuple(head1["output"].shape))
    print("Head 2 output shape:", tuple(head2["output"].shape))
    print_tensor("Concatenated [head1 | head2]", concatenated)
    print_tensor("W_O (output projection)", W_O)
    print_tensor("Final output = concatenated @ W_O", final_output)
    print()

    # ---------------------------------------------------------------------------
    # 6. Experiment A: change only W_V1
    # ---------------------------------------------------------------------------
    # Swap the two columns of W_V1. Q1, K1, all of head 2, W_O and the mask stay
    # the same, so head 1's scores and weights cannot change. Only the values it
    # mixes change.
    W_V1_modified = W_V1[:, [1, 0]]
    heads_A, concatenated_A, final_output_A = multi_head_attention(
        X, [(W_Q1, W_K1, W_V1_modified), (W_Q2, W_K2, W_V2)], W_O, causal_mask)
    head1_A, head2_A = heads_A

    print("=================== 6. Experiment A: change W_V1 only ===================")
    print_tensor("Original W_V1", W_V1)
    print_tensor("Modified W_V1 (columns swapped)", W_V1_modified)
    print("Head 1 scores unchanged: ", torch.allclose(head1["scaled_scores"], head1_A["scaled_scores"]))
    print("Head 1 weights unchanged:", torch.allclose(head1["weights"], head1_A["weights"]))
    print("Head 2 weights unchanged:", torch.allclose(head2["weights"], head2_A["weights"]))
    print("Head 2 output unchanged: ", torch.allclose(head2["output"], head2_A["output"]))
    print("\nHead 1 OUTPUT:")
    print_row_comparison("before", head1["output"], "after", head1_A["output"])
    print("\nConcatenated:")
    print_row_comparison("before", concatenated, "after", concatenated_A)
    print("\nFinal output:")
    print_row_comparison("before", final_output, "after", final_output_A)
    print()

    # ---------------------------------------------------------------------------
    # 7. Experiment B: future-token leakage
    # ---------------------------------------------------------------------------
    # Change ONLY the last token's embedding. All matrices and the mask stay fixed.
    X_modified = X.clone()
    X_modified[3] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    heads_B, concatenated_B, final_output_B = multi_head_attention(X_modified, head_matrices, W_O, causal_mask)
    head1_B, head2_B = heads_B
    earlier_tokens = slice(0, 3)  # "The", "cat", "eats"

    print("=================== 7. Experiment B: change the last token only ===================")
    print("Original 'fish' embedding:", X[3].tolist(), "-> modified:", X_modified[3].tolist())
    print("\nHead 1 OUTPUT:")
    print_row_comparison("before", head1["output"], "after", head1_B["output"])
    print("\nHead 2 OUTPUT:")
    print_row_comparison("before", head2["output"], "after", head2_B["output"])
    print("\nConcatenated:")
    print_row_comparison("before", concatenated, "after", concatenated_B)
    print("\nFinal output:")
    print_row_comparison("before", final_output, "after", final_output_B)
    print("(W_O multiplies each row separately, so it cannot carry 'fish' into earlier rows.)")
    print()

    # ---------------------------------------------------------------------------
    # 8. Verification
    # ---------------------------------------------------------------------------
    future_positions = ~causal_mask
    all_heads = heads + heads_A + heads_B

    # 1-9. Shapes
    assert X.shape == (4, 4), "X should be (4, 4)"
    for W in [W_Q1, W_K1, W_V1, W_Q2, W_K2, W_V2, W_V1_modified]:
        assert W.shape == (4, 2), "Each projection matrix should be (4, 2)"
    for head in all_heads:
        for name in ["Q", "K", "V"]:
            assert head[name].shape == (4, 2), f"{name} should be (4, 2)"
        assert head["scaled_scores"].shape == (4, 4), "Scores should be (4, 4)"
        assert head["weights"].shape == (4, 4), "Weights should be (4, 4)"
        assert head["output"].shape == (4, 2), "Head output should be (4, 2)"
    assert concatenated.shape == (4, 4), "Concatenated output should be (4, 4)"
    assert W_O.shape == (4, 4), "W_O should be (4, 4)"
    assert final_output.shape == (4, 4), "Final output should be (4, 4)"

    # 10-13. Attention weights
    for head in all_heads:
        weights = head["weights"]
        assert torch.allclose(weights.sum(dim=-1), torch.ones(seq_len), atol=TOLERANCE), "Rows should sum to 1"
        assert (weights >= 0).all(), "Weights should be nonnegative"
        assert (weights[future_positions] == 0).all(), "Future positions should get exactly zero weight"
        assert torch.allclose(weights[0, 0], torch.tensor(1.0)), "First token should attend only to itself"

    # 14. Heads differ
    assert not torch.allclose(head1["weights"], head2["weights"], atol=1e-2), \
        "The two heads should produce different attention distributions"

    # 15-17. Experiment A
    assert torch.allclose(head1["scaled_scores"], head1_A["scaled_scores"]), "W_V1 must not change head 1 scores"
    assert torch.allclose(head1["weights"], head1_A["weights"]), "W_V1 must not change head 1 weights"
    assert not torch.allclose(head1["output"], head1_A["output"], atol=TOLERANCE), "W_V1 should change head 1 output"
    assert torch.allclose(head2["weights"], head2_A["weights"]), "W_V1 must not change head 2 weights"
    assert torch.allclose(head2["output"], head2_A["output"]), "W_V1 must not change head 2 output"
    assert not torch.allclose(concatenated, concatenated_A, atol=TOLERANCE), "Concatenated output should change"
    assert not torch.allclose(final_output, final_output_A, atol=TOLERANCE), "Final output should change"

    # 18-19. Experiment B
    for before, after in [(head1["output"], head1_B["output"]), (head2["output"], head2_B["output"]),
                          (concatenated, concatenated_B), (final_output, final_output_B)]:
        assert torch.allclose(before[earlier_tokens], after[earlier_tokens], atol=TOLERANCE), \
            "Changing the last token must not change earlier rows"

    print("=================== Verification ===================")
    print("All checks passed.")


if __name__ == "__main__":
    main()
