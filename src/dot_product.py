"""
Milestone 1: Token-to-token dot-product similarity.

We compute how "similar" every token is to every other token using the dot
product, first with plain Python loops and then with a single PyTorch matrix
multiplication. Both approaches must give identical results.
"""

import torch

# ---------------------------------------------------------------------------
# 1. Tokens and embeddings
# ---------------------------------------------------------------------------
# A tensor is a multi-dimensional array of numbers (a generalization of
# vectors and matrices). Here it is a 2D tensor, i.e. a matrix.
#
# Each row is the embedding (a vector of numbers) for one token.
# Shape is (3, 2) because we have 3 tokens and each embedding has 2 numbers
# (2 dimensions): (num_tokens, embedding_dim).
#
# These values are hand-picked for learning; they do not come from a trained model.
tokens = ["cat", "eats", "fish"]
embeddings = torch.tensor([
    [1.0, 0.0],  # "cat"
    [0.0, 1.0],  # "eats"
    [1.0, 1.0],  # "fish"
])


def manual_dot_product_scores(embeddings):
    """
    Token-to-token dot products using plain Python loops.
    Returns a list of lists: scores[i][j] = embeddings[i] · embeddings[j].
    """
    num_tokens, embedding_dim = embeddings.shape
    manual_scores = []
    for row_index in range(num_tokens):
        row_scores = []
        for column_index in range(num_tokens):
            dot_product = 0.0
            for dimension in range(embedding_dim):
                dot_product += (
                    embeddings[row_index][dimension].item()
                    * embeddings[column_index][dimension].item()
                )
            row_scores.append(dot_product)
        manual_scores.append(row_scores)
    return manual_scores


def main():
    num_tokens = embeddings.shape[0]

    print("=== Embeddings ===")
    print(embeddings)
    print("Embedding shape:", tuple(embeddings.shape))
    print()

    # -----------------------------------------------------------------------
    # 2. Transpose
    # -----------------------------------------------------------------------
    # .T swaps rows and columns. Each token's embedding becomes a column instead
    # of a row, so shape (3, 2) becomes (2, 3).
    embeddings_transposed = embeddings.T

    print("=== Transposed embeddings (embeddings.T) ===")
    print(embeddings_transposed)
    print("Transposed shape:", tuple(embeddings_transposed.shape))
    print()

    # -----------------------------------------------------------------------
    # 3. Manual dot-product similarity with nested loops
    # -----------------------------------------------------------------------
    # The dot product of two vectors multiplies matching positions and sums them:
    #   a · b = a[0]*b[0] + a[1]*b[1] + ...
    # Example: "fish" · "fish" = 1*1 + 1*1 = 2
    #
    # We compute it for every (row token, column token) pair.
    manual_scores = manual_dot_product_scores(embeddings)

    print("=== Manual scores (nested Python loops) ===")
    for row_scores in manual_scores:
        print(row_scores)
    print()

    # -----------------------------------------------------------------------
    # 4. Same calculation with PyTorch matrix multiplication
    # -----------------------------------------------------------------------
    # (3, 2) @ (2, 3) -> (3, 3)
    # Matrix multiplication requires the inner dimensions to match (2 and 2).
    # The result takes the outer dimensions (3 and 3). Entry [i][j] is the dot
    # product of row i of the left matrix with column j of the right matrix,
    # which is exactly token i's embedding · token j's embedding.
    scores = embeddings @ embeddings.T

    print("=== PyTorch scores (embeddings @ embeddings.T) ===")
    print(scores)
    print("Score shape:", tuple(scores.shape))
    print()

    # -----------------------------------------------------------------------
    # 5. Readable token-to-token table
    # -----------------------------------------------------------------------
    # Row i    = the token doing the comparing.
    # Column j = the token being compared against.
    # scores[i][j] = similarity between tokens[i] and tokens[j].
    # The matrix is symmetric because a · b == b · a.
    column_width = 8
    print("=== Token-to-token similarity table ===")
    print(" " * column_width + "".join(f"{token:>{column_width}}" for token in tokens))
    for row_index, row_token in enumerate(tokens):
        row_values = "".join(
            f"{scores[row_index][column_index].item():>{column_width}.1f}"
            for column_index in range(num_tokens)
        )
        print(f"{row_token:<{column_width}}" + row_values)
    print()

    # -----------------------------------------------------------------------
    # 6. Verification
    # -----------------------------------------------------------------------
    manual_scores_tensor = torch.tensor(manual_scores)
    expected_scores = torch.tensor([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 2.0],
    ])

    assert embeddings.shape == (3, 2), "Embedding shape should be (3, 2)"
    assert embeddings_transposed.shape == (2, 3), "Transposed shape should be (2, 3)"
    assert scores.shape == (3, 3), "Score shape should be (3, 3)"
    assert torch.allclose(manual_scores_tensor, scores), "Manual and PyTorch scores differ"
    assert torch.allclose(scores, expected_scores), "Scores do not match expected values"

    print("=== Verification ===")
    print("Manual and PyTorch results match:", torch.allclose(manual_scores_tensor, scores))
    print("Results match expected values:  ", torch.allclose(scores, expected_scores))
    print("All checks passed.")


if __name__ == "__main__":
    main()
