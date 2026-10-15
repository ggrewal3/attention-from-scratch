# attention-from-scratch
Implementing transformer attention from scratch using Python and PyTorch, from dot products to multi-head attention.

## Project purpose

An educational project to understand transformer attention both mathematically
and in code. Each milestone adds one small idea on top of the previous one.

## Milestone 1 — Dot-product similarity

**Objective:** compute token-to-token similarity using the dot product in two ways:

1. Manually, with nested Python loops.
2. With PyTorch matrix multiplication: `embeddings @ embeddings.T`.

Both approaches must produce the same score matrix.

Input (hand-picked embeddings for learning, not from a trained model):

```python
tokens = ["cat", "eats", "fish"]
embeddings = torch.tensor([
    [1.0, 0.0],  # cat
    [0.0, 1.0],  # eats
    [1.0, 1.0],  # fish
])
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python src/dot_product.py
```

## Expected output

```
=== Embeddings ===
tensor([[1., 0.],
        [0., 1.],
        [1., 1.]])
Embedding shape: (3, 2)

=== Transposed embeddings (embeddings.T) ===
tensor([[1., 0., 1.],
        [0., 1., 1.]])
Transposed shape: (2, 3)

=== Manual scores (nested Python loops) ===
[1.0, 0.0, 1.0]
[0.0, 1.0, 1.0]
[1.0, 1.0, 2.0]

=== PyTorch scores (embeddings @ embeddings.T) ===
tensor([[1., 0., 1.],
        [0., 1., 1.],
        [1., 1., 2.]])
Score shape: (3, 3)

=== Token-to-token similarity table ===
             cat    eats    fish
cat          1.0     0.0     1.0
eats         0.0     1.0     1.0
fish         1.0     1.0     2.0

=== Verification ===
Manual and PyTorch results match: True
Results match expected values:   True
All checks passed.
```

## Matrix shapes explained

| Tensor | Shape | Meaning |
|---|---|---|
| `embeddings` | (3, 2) | 3 tokens, each a 2-dimensional vector |
| `embeddings.T` | (2, 3) | Transpose: rows and columns swapped, so each token is now a column |
| `scores` | (3, 3) | One similarity score for every (token, token) pair |

Matrix multiplication `(3, 2) @ (2, 3)` works because the inner dimensions match
(2 = 2). The result takes the outer dimensions, giving `(3, 3)`.

Each entry `scores[i][j]` is the dot product of token `i`'s embedding with token
`j`'s embedding. For example:

```
fish · fish = (1 * 1) + (1 * 1) = 2
cat  · eats = (1 * 0) + (0 * 1) = 0
```

- **Row i:** how token `i` compares to every token.
- **Column j:** how every token compares to token `j`.
- The matrix is symmetric because `a · b == b · a`.

## Raw dot-product similarity vs. transformer attention

This milestone computes **raw** similarity only. Real transformer attention adds
several steps on top of it, which later milestones will cover:

- **Q, K, V projections:** the embeddings are first multiplied by learned weight
  matrices to produce queries, keys and values. Scores are `Q @ K.T`, not
  `embeddings @ embeddings.T`, so the matrix is generally not symmetric.
- **Scaling:** scores are divided by `sqrt(d_k)` to keep them from growing too large.
- **Softmax:** each row is turned into weights that are positive and sum to 1.
- **Weighted sum of values:** those weights are used to mix the value vectors
  into a new, context-aware representation of each token.
- **Masking and multiple heads:** decoders hide future tokens, and multi-head
  attention runs several attention computations in parallel.

Raw dot products are unnormalized. Longer vectors produce bigger scores, which is
why "fish" scores 2 with itself while "cat" scores 1.

---

## Milestone 2 — Scaled dot-product attention

**Objective:** understand how raw similarity scores become attention weights,
and how those weights combine value vectors into contextual representations.

```
Attention(Q, K, V) = softmax( Q @ K.T / sqrt(d_k) ) @ V
```

> **Note:** M2 uses `Q = K = V = embeddings` to keep the focus on the attention
> math. In a real transformer, Q, K and V come from separate learned projection
> matrices. Those are introduced in M3.

### Each operation

| Step | Code | What it does |
|---|---|---|
| Q, K, V | `Q = K = V = embeddings` | **Query:** what a token looks for. **Key:** what a token offers to be matched against. **Value:** the information a token passes along. |
| Raw scores | `Q @ K.T` | Dot product of every query with every key: how well they match. |
| Scaling | `raw_scores / sqrt(d_k)` | `d_k = K.shape[-1]` is the key length. Dot products grow with `d_k`, so dividing keeps the logits in a moderate range. Every value changes, but the shape doesn't. |
| Softmax | `torch.softmax(scaled, dim=-1)` | Turns each row into positive weights that sum to 1. `dim=-1` (the last dimension) normalizes across the columns of each row, so each query gets its own distribution over keys. |
| Weighted sum | `attention_weights @ V` | Each output row is a blend of all value vectors, weighted by attention. |

**Numerically stable softmax.** The script also implements softmax by hand.
It subtracts each row's maximum before calling `exp()`, so the largest exponent
is `exp(0) = 1` and nothing can overflow. The result is unchanged because
`exp(x - m) / Σ exp(x - m) = exp(x) / Σ exp(x)`: the common factor `exp(-m)` cancels.

### Tensor shapes

| Tensor | Shape | Meaning |
|---|---|---|
| `Q`, `K`, `V` | (3, 2) | 3 tokens × 2 dimensions |
| `K.T` | (2, 3) | Keys as columns |
| `raw_scores`, `scaled_scores` | (3, 3) | One score per (query, key) pair |
| `attention_weights` | (3, 3) | Row i = how query i distributes its attention over the 3 keys |
| `output` | (3, 2) | `(3, 3) @ (3, 2)`: one new 2-dim vector per token |

### Run

```bash
python src/scaled_attention.py
```

### Expected results

Scaled scores (`d_k = 2`, `sqrt(d_k) ≈ 1.4142`):

```
[[0.7071, 0.0000, 0.7071],
 [0.0000, 0.7071, 0.7071],
 [0.7071, 0.7071, 1.4142]]
```

Attention weights (each row sums to 1; manual and PyTorch softmax match):

```
             cat    eats    fish
cat       0.4011  0.1978  0.4011
eats      0.1978  0.4011  0.4011
fish      0.2483  0.2483  0.5035
```

Contextual output:

```
[[0.8022, 0.5989],   # cat
 [0.5989, 0.8022],   # eats
 [0.7517, 0.7517]]   # fish
```

For example, the new vector for "cat" is

```
0.4011 * [1, 0] + 0.1978 * [0, 1] + 0.4011 * [1, 1] = [0.8022, 0.5989]
```

"cat" started as `[1, 0]`. After attention it has picked up some of the second
dimension from "eats" and "fish". That blending is what makes the representation
*contextual*.

### Scaling experiment

| | Largest weight per row (cat, eats, fish) |
|---|---|
| `softmax(Q @ K.T)` (unscaled) | 0.4223, 0.4223, 0.5761 |
| `softmax(Q @ K.T / sqrt(2))` (scaled) | 0.4011, 0.4011, 0.5035 |

Without scaling, the distributions are somewhat more concentrated on the top
score. With `d_k = 2` the difference is mild. Scaling matters much more when
`d_k` is large (e.g. 64 or 128). There, raw dot products can get big enough to
push softmax toward a near one-hot output with very small gradients.

### What M2 does and does not implement

**Does:** raw scores, scaling by `sqrt(d_k)`, manual stable softmax, PyTorch
softmax, weighted sum of values, scaling comparison, and shape and value assertions.

**Does not (future milestones):** learned Q/K/V projections (M3), multi-head
attention, causal masking, positional encoding, training, transformer blocks,
visualization.
