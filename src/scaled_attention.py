"""
Milestone 2: Scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V

Raw similarity scores (from M1) are scaled, turned into attention weights with
softmax, and then used to mix value vectors into contextual representations.

For learning purposes, Q = K = V = embeddings. Learned projection matrices
that produce different Q, K and V come in M3.
"""

import torch

# Print tensors with 4 decimal places so small differences are visible.
torch.set_printoptions(precision=4)

# ---------------------------------------------------------------------------
# 1. Tokens, embeddings, and Q / K / V
# ---------------------------------------------------------------------------
# Hand-picked embeddings (not from a trained model).
# Shape (3, 2): 3 tokens, each a 2-dimensional vector.
tokens = ["cat", "eats", "fish"]
embeddings = torch.tensor([
    [1.0, 0.0],  # "cat"
    [0.0, 1.0],  # "eats"
    [1.0, 1.0],  # "fish"
], dtype=torch.float32)

# Query (Q): what each token is "looking for".
# Key (K):   what each token "offers" to be matched against.
# Value (V): the information each token passes along once it is attended to.
# In a real transformer these come from separate learned projections (M3).
# Here we use the raw embeddings for all three.
Q = embeddings
K = embeddings
V = embeddings

print("=== Q, K, V (all equal to the embeddings in M2) ===")
print(embeddings)
print("Q shape:", tuple(Q.shape), "| K shape:", tuple(K.shape), "| V shape:", tuple(V.shape))
print()

# ---------------------------------------------------------------------------
# 2. Raw attention scores: Q @ K.T
# ---------------------------------------------------------------------------
# Each entry raw_scores[i][j] is the dot product of query i with key j:
# multiply matching positions and add them up.
# (3, 2) @ (2, 3) -> (3, 3): inner dims (2) match, outer dims (3, 3) remain.
# Row i = one query token; column j = one key token.
raw_scores = Q @ K.T

print("=== Raw scores (Q @ K.T) ===")
print(raw_scores)
print("Raw score shape:", tuple(raw_scores.shape))
print()

# ---------------------------------------------------------------------------
# 3. Scaling by sqrt(d_k)
# ---------------------------------------------------------------------------
# d_k is the length of each key vector (last dimension of K).
# Dot products of d_k-dimensional vectors tend to grow with d_k. Dividing by
# sqrt(d_k) keeps scores in a moderate range so softmax doesn't saturate.
# Dividing every element by one number changes values, never the shape.
d_k = K.shape[-1]
scaled_scores = raw_scores / (d_k ** 0.5)

print(f"=== Scaled scores (raw / sqrt(d_k)), d_k = {d_k}, sqrt(d_k) = {d_k ** 0.5:.4f} ===")
print(scaled_scores)
print("Scaled score shape:", tuple(scaled_scores.shape))
print()

# ---------------------------------------------------------------------------
# 4. Manual softmax (numerically stable)
# ---------------------------------------------------------------------------
# Softmax turns a row of arbitrary numbers into positive weights summing to 1:
#   softmax(x)_j = exp(x_j) / sum_k exp(x_k)
#
# Numerical stability: exp() of a large number can overflow to infinity.
# Subtracting the row maximum first makes the largest value 0, so every
# exp() is in (0, 1]. The result is unchanged because
#   exp(x_j - m) / sum_k exp(x_k - m) = exp(x_j) / sum_k exp(x_k)
# (the common factor exp(-m) cancels in the numerator and denominator).
def manual_softmax(scores):
    row_max = scores.max(dim=-1, keepdim=True).values      # shape (3, 1)
    shifted_scores = scores - row_max                       # largest per row is 0
    exponentials = torch.exp(shifted_scores)                # all in (0, 1]
    row_sums = exponentials.sum(dim=-1, keepdim=True)       # shape (3, 1)
    return exponentials / row_sums                          # each row sums to 1

manual_attention_weights = manual_softmax(scaled_scores)

print("=== Attention weights (manual softmax) ===")
print(manual_attention_weights)
print()

# ---------------------------------------------------------------------------
# 5. PyTorch softmax
# ---------------------------------------------------------------------------
# dim=-1 means "the last dimension", i.e. across the columns of each row.
# So each query's scores over all keys are normalized independently.
attention_weights = torch.softmax(scaled_scores, dim=-1)

print("=== Attention weights (torch.softmax, dim=-1) ===")
print(attention_weights)
print("Attention weight shape:", tuple(attention_weights.shape))
print("Row sums:", attention_weights.sum(dim=-1))
print("Manual softmax matches torch.softmax:",
      torch.allclose(manual_attention_weights, attention_weights, atol=1e-6))
print()

# Readable table: row = query token, column = how much it attends to that key.
column_width = 8
print("=== Attention weight table (row attends to column) ===")
print(" " * column_width + "".join(f"{token:>{column_width}}" for token in tokens))
for row_index, row_token in enumerate(tokens):
    row_values = "".join(
        f"{attention_weights[row_index][column_index].item():>{column_width}.4f}"
        for column_index in range(len(tokens))
    )
    print(f"{row_token:<{column_width}}" + row_values)
print()

# ---------------------------------------------------------------------------
# 6. Contextual output: attention_weights @ V
# ---------------------------------------------------------------------------
# (3, 3) @ (3, 2) -> (3, 2)
#   attention_weights: one row of 3 weights per query token.
#   V:                 one 2-dim value vector per token.
#   output:            one new 2-dim vector per token.
#
# Each output row is a weighted sum of ALL value vectors:
#   output[i] = w[i][0] * V[0] + w[i][1] * V[1] + w[i][2] * V[2]
# So every token's new representation blends in information from the other
# tokens, in proportion to how much it attends to them. That blend is a
# "contextual representation".
output = attention_weights @ V

print("=== Contextual output (attention_weights @ V) ===")
print(output)
print("Output shape:", tuple(output.shape))
print()

# Show the weighted sum for the first token step by step.
first_token_index = 0
first_token_weights = attention_weights[first_token_index]
manual_first_output = torch.zeros(V.shape[-1])
print(f"=== Manual weighted sum for '{tokens[first_token_index]}' ===")
for value_index, value_token in enumerate(tokens):
    weight = first_token_weights[value_index]
    manual_first_output += weight * V[value_index]
    print(f"  {weight.item():.4f} * V[{value_token}] {V[value_index].tolist()}")
print("  sum =", manual_first_output)
print("  matches output row:", torch.allclose(manual_first_output, output[first_token_index], atol=1e-6))
print()

# ---------------------------------------------------------------------------
# 7. Scaling experiment
# ---------------------------------------------------------------------------
# Compare softmax on raw scores vs. scaled scores. Larger logits make softmax
# "peakier" (more weight on the top score). Dividing by sqrt(d_k) softens this.
# With d_k = 2 the effect is mild; it matters much more when d_k is large
# (e.g. 64 or 128), where raw dot products can be big.
unscaled_attention_weights = torch.softmax(raw_scores, dim=-1)

print("=== Scaling experiment ===")
print("Without scaling: softmax(Q @ K.T)")
print(unscaled_attention_weights)
print(f"With scaling:    softmax(Q @ K.T / sqrt({d_k}))")
print(attention_weights)
print("Largest weight per row (unscaled):", unscaled_attention_weights.max(dim=-1).values)
print("Largest weight per row (scaled):  ", attention_weights.max(dim=-1).values)
print()

# ---------------------------------------------------------------------------
# 8. Verification
# ---------------------------------------------------------------------------
tolerance = 1e-6
num_tokens = len(tokens)

assert Q.shape == (3, 2), "Q should be (3, 2)"
assert K.shape == (3, 2), "K should be (3, 2)"
assert V.shape == (3, 2), "V should be (3, 2)"
assert raw_scores.shape == (3, 3), "Raw scores should be (3, 3)"
assert scaled_scores.shape == (3, 3), "Scaled scores should be (3, 3)"
assert attention_weights.shape == (3, 3), "Attention weights should be (3, 3)"
assert output.shape == (3, 2), "Output should be (3, 2)"
assert torch.allclose(manual_attention_weights, attention_weights, atol=tolerance), \
    "Manual softmax does not match torch.softmax"
assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(num_tokens), atol=tolerance), \
    "Each attention row should sum to 1"
assert (attention_weights >= 0).all(), "Attention weights should be nonnegative"
assert torch.allclose(manual_first_output, output[first_token_index], atol=tolerance), \
    "Output row does not match manual weighted sum"

print("=== Verification ===")
print("All checks passed.")
