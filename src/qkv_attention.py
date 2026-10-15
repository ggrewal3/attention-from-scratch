"""
Milestone 3: Query, Key, and Value projections.

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V

In M2 we used Q = K = V = X. Here each of Q, K and V gets its own projection
matrix, so a token can "ask" (query), "advertise" (key) and "share" (value)
different things.

The projection matrices are hand-picked for learning. In a trained transformer
they are learned parameters adjusted during training. There is no training here.
"""

import torch

# Print tensors with 4 decimal places so small differences are visible.
torch.set_printoptions(precision=4)

TOLERANCE = 1e-4

# ---------------------------------------------------------------------------
# 1. Input embeddings
# ---------------------------------------------------------------------------
# X has shape (3, 2): 3 tokens, each a 2-dimensional embedding.
tokens = ["cat", "eats", "fish"]
X = torch.tensor([
    [1.0, 0.0],  # "cat"
    [0.0, 1.0],  # "eats"
    [1.0, 1.0],  # "fish"
], dtype=torch.float32)

# ---------------------------------------------------------------------------
# 2. Projection matrices (hand-picked, normally learned)
# ---------------------------------------------------------------------------
# Each matrix has shape (2, 2): it maps a 2-dim embedding to a 2-dim vector.
# Multiplying a row vector [a, b] by a 2x2 matrix W gives
#   [a * W[0][0] + b * W[1][0],  a * W[0][1] + b * W[1][1]]

# Query projection: what each token is looking for.
# Swaps the two dimensions: [a, b] -> [b, a]
# So a token strong in feature 1 now "looks for" feature 2, and vice versa.
W_Q = torch.tensor([
    [0.0, 1.0],
    [1.0, 0.0],
])

# Key projection: what each token offers to be matched against.
# [a, b] -> [-b, a + 2b]  (the negative entry lets some keys score below zero)
W_K = torch.tensor([
    [0.0, 1.0],
    [-1.0, 2.0],
])

# Value projection: the information each token passes along when attended to.
# Stretches dimension 1 and shrinks dimension 2: [a, b] -> [2a, 0.5b]
W_V = torch.tensor([
    [2.0, 0.0],
    [0.0, 0.5],
])


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(Q, K, V):
    """Return raw scores, scaled scores, attention weights and output."""
    # Attention scores: dot product of every query with every key.
    # (3, 2) @ (2, 3) -> (3, 3). Row i = query token, column j = key token.
    raw_scores = Q @ K.T

    # Scaling: divide by sqrt(d_k) so scores don't grow with the key size.
    d_k = K.shape[-1]
    scaled_scores = raw_scores / (d_k ** 0.5)

    # Softmax across each row (dim=-1): positive weights that sum to 1.
    attention_weights = torch.softmax(scaled_scores, dim=-1)

    # Contextual output: each row is a weighted sum of all value vectors.
    # (3, 3) @ (3, 2) -> (3, 2).
    output = attention_weights @ V
    return raw_scores, scaled_scores, attention_weights, output


def print_tensor(label, tensor):
    print(f"--- {label} | shape {tuple(tensor.shape)} ---")
    print(tensor)


def print_weight_table(attention_weights):
    """Row = query token, column = how much it attends to that key token."""
    column_width = 8
    print(" " * column_width + "".join(f"{token:>{column_width}}" for token in tokens))
    for row_index, row_token in enumerate(tokens):
        row_values = "".join(
            f"{attention_weights[row_index][column_index].item():>{column_width}.4f}"
            for column_index in range(len(tokens))
        )
        print(f"{row_token:<{column_width}}" + row_values)


def print_most_attended(attention_weights):
    for row_index, row_token in enumerate(tokens):
        top_index = attention_weights[row_index].argmax().item()
        print(f"  {row_token:<5} attends most to {tokens[top_index]}")


# ---------------------------------------------------------------------------
# 3. Calculate Q, K and V
# ---------------------------------------------------------------------------
# (3, 2) @ (2, 2) -> (3, 2): the inner dims (2 and 2) match, and the result
# keeps the outer dims, so we still get one 2-dim vector per token.
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("=================== 1. Projections ===================")
print_tensor("Input embeddings X", X)
print_tensor("W_Q (query projection)", W_Q)
print_tensor("W_K (key projection)", W_K)
print_tensor("W_V (value projection)", W_V)
print_tensor("Q = X @ W_Q", Q)
print_tensor("K = X @ W_K", K)
print_tensor("V = X @ W_V", V)
print()

# ---------------------------------------------------------------------------
# 4. Scaled dot-product attention with projections
# ---------------------------------------------------------------------------
raw_scores, scaled_scores, attention_weights, output = scaled_dot_product_attention(Q, K, V)

print("=================== 2. Attention with projections ===================")
print_tensor("Raw scores Q @ K.T", raw_scores)
print_tensor(f"Scaled scores (/ sqrt({K.shape[-1]}))", scaled_scores)
print_tensor("Attention weights softmax(scaled, dim=-1)", attention_weights)
print_weight_table(attention_weights)
print_most_attended(attention_weights)
print_tensor("Output = attention_weights @ V", output)

# Each output row is sum_j weight[i][j] * V[j]. Show it for "cat".
manual_cat_output = sum(attention_weights[0][j] * V[j] for j in range(len(tokens)))
print("Manual weighted sum for 'cat':",
      " + ".join(f"{attention_weights[0][j].item():.4f}*{V[j].tolist()}" for j in range(len(tokens))),
      "=", manual_cat_output)
print()

# ---------------------------------------------------------------------------
# 5. Comparison against the M2 baseline (Q = K = V = X)
# ---------------------------------------------------------------------------
baseline_raw, baseline_scaled, baseline_weights, baseline_output = scaled_dot_product_attention(X, X, X)

print("=================== 3. Comparison with M2 baseline (Q = K = V = X) ===================")
print_tensor("Baseline raw scores", baseline_raw)
print_tensor("Projected raw scores", raw_scores)
print("Baseline attention weights:")
print_weight_table(baseline_weights)
print("Projected attention weights:")
print_weight_table(attention_weights)
print_tensor("Baseline output", baseline_output)
print_tensor("Projected output", output)
print("Baseline scores symmetric? ", torch.allclose(baseline_raw, baseline_raw.T))
print("Projected scores symmetric?", torch.allclose(raw_scores, raw_scores.T))
print()

# ---------------------------------------------------------------------------
# 6. Experiment A: change only W_Q
# ---------------------------------------------------------------------------
# New query projection is the identity: [a, b] -> [a, b], so queries are
# just the raw embeddings (no swap). W_K and W_V are unchanged.
W_Q_modified = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
])
Q_modified = X @ W_Q_modified
raw_scores_A, scaled_scores_A, attention_weights_A, output_A = scaled_dot_product_attention(Q_modified, K, V)

print("=================== 4. Experiment A: change W_Q only ===================")
print_tensor("Modified W_Q", W_Q_modified)
print_tensor("Original Q", Q)
print_tensor("Modified Q", Q_modified)
print_tensor("Original raw scores", raw_scores)
print_tensor("Modified raw scores", raw_scores_A)
print("Original attention weights:")
print_weight_table(attention_weights)
print_most_attended(attention_weights)
print("Modified attention weights:")
print_weight_table(attention_weights_A)
print_most_attended(attention_weights_A)
print_tensor("Original output", output)
print_tensor("Modified output", output_A)
print()

# ---------------------------------------------------------------------------
# 7. Experiment B: change only W_V (original W_Q restored)
# ---------------------------------------------------------------------------
# New value projection reverses the emphasis: [a, b] -> [0.5a, 2b].
# Q and K are unchanged, so scores and weights must be identical.
W_V_modified = torch.tensor([
    [0.5, 0.0],
    [0.0, 2.0],
])
V_modified = X @ W_V_modified
raw_scores_B, scaled_scores_B, attention_weights_B, output_B = scaled_dot_product_attention(Q, K, V_modified)

print("=================== 5. Experiment B: change W_V only ===================")
print_tensor("Modified W_V", W_V_modified)
print_tensor("Original V", V)
print_tensor("Modified V", V_modified)
print("Raw scores unchanged:       ", torch.allclose(raw_scores, raw_scores_B))
print("Attention weights unchanged:", torch.allclose(attention_weights, attention_weights_B))
print_tensor("Original output", output)
print_tensor("Modified output", output_B)
print("Output changed:             ", not torch.allclose(output, output_B, atol=TOLERANCE))
print()

# ---------------------------------------------------------------------------
# 8. Verification
# ---------------------------------------------------------------------------
# M2 results (Q = K = V = X), recorded from src/scaled_attention.py.
m2_expected_weights = torch.tensor([
    [0.4011, 0.1978, 0.4011],
    [0.1978, 0.4011, 0.4011],
    [0.2483, 0.2483, 0.5035],
])
m2_expected_output = torch.tensor([
    [0.8022, 0.5989],
    [0.5989, 0.8022],
    [0.7517, 0.7517],
])

assert X.shape == (3, 2), "X should be (3, 2)"
for name, matrix in [("W_Q", W_Q), ("W_K", W_K), ("W_V", W_V),
                     ("W_Q_modified", W_Q_modified), ("W_V_modified", W_V_modified)]:
    assert matrix.shape == (2, 2), f"{name} should be (2, 2)"
for name, matrix in [("Q", Q), ("K", K), ("V", V)]:
    assert matrix.shape == (3, 2), f"{name} should be (3, 2)"
assert raw_scores.shape == (3, 3), "Raw scores should be (3, 3)"
assert scaled_scores.shape == (3, 3), "Scaled scores should be (3, 3)"
assert attention_weights.shape == (3, 3), "Attention weights should be (3, 3)"
assert output.shape == (3, 2), "Output should be (3, 2)"
for weights in [attention_weights, attention_weights_A, attention_weights_B, baseline_weights]:
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=TOLERANCE), "Rows should sum to 1"
    assert (weights >= 0).all(), "Attention weights should be nonnegative"
assert torch.allclose(manual_cat_output, output[0], atol=TOLERANCE), "Output should be a weighted sum of V"
assert not torch.allclose(attention_weights, attention_weights_A, atol=TOLERANCE), \
    "Changing W_Q should change attention weights"
assert torch.allclose(raw_scores, raw_scores_B, atol=TOLERANCE), \
    "Changing W_V should not change attention scores"
assert torch.allclose(attention_weights, attention_weights_B, atol=TOLERANCE), \
    "Changing W_V should not change attention weights"
assert not torch.allclose(output, output_B, atol=TOLERANCE), \
    "Changing W_V should change the contextual output"
assert torch.allclose(baseline_weights, m2_expected_weights, atol=TOLERANCE), \
    "Baseline weights should match M2"
assert torch.allclose(baseline_output, m2_expected_output, atol=TOLERANCE), \
    "Baseline output should match M2"

print("=================== Verification ===================")
print("All checks passed.")
