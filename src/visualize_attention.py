"""
Milestone 6: Attention visualization.

Draws heatmaps of the two-head causal self-attention from M5:
    1. Head 1 attention weights
    2. Head 2 attention weights
    3. Head 1 minus Head 2
    4. The causal mask

All values are computed by M5's own functions and matrices. Nothing is copied
in by hand.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend: write PNG files, open no windows.
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import torch

# Reuse M5's inputs and attention implementation.
from multi_head_attention import X, tokens, head_matrices, W_O, causal_mask, multi_head_attention

TOLERANCE = 1e-5
DPI = 160
FIGURE_SIZE = (5.6, 4.6)
IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold"})


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def label_axes(axis):
    """Rows = query tokens (who is attending), columns = key tokens (attended to)."""
    positions = range(len(tokens))
    axis.set_xticks(positions, tokens)
    axis.set_yticks(positions, tokens)
    axis.set_xlabel("Key token (being attended to)")
    axis.set_ylabel("Query token (attending)")
    # Thin white grid lines between cells.
    axis.set_xticks([p - 0.5 for p in range(1, len(tokens))], minor=True)
    axis.set_yticks([p - 0.5 for p in range(1, len(tokens))], minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", length=0)


def hatch_blocked_cells(axis, mask):
    """Overlay a light hatch on positions the causal mask blocks."""
    for row in range(mask.shape[0]):
        for column in range(mask.shape[1]):
            if not mask[row, column]:
                axis.add_patch(plt.Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False,
                                             hatch="///", edgecolor="#b0b0b0", linewidth=0))


def add_hatch_note(figure):
    """Footnote placed below the x-axis label (bbox_inches="tight" keeps it in frame)."""
    figure.text(0.45, -0.04, "Hatched cells = future positions blocked by the causal mask",
                ha="center", fontsize=9, color="#555555")


def save_attention_heatmap(weights, title, path):
    """Heatmap of one head's attention weights, on a fixed 0-1 color scale."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    image = axis.imshow(weights.numpy(), cmap="Blues", vmin=0.0, vmax=1.0)
    hatch_blocked_cells(axis, causal_mask)
    for row in range(weights.shape[0]):
        for column in range(weights.shape[1]):
            value = weights[row, column].item()
            text_color = "white" if value > 0.55 else "#1a1a1a"
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", color=text_color)
    label_axes(axis)
    axis.set_title(title, pad=12)
    figure.colorbar(image, ax=axis, label="Attention weight")
    add_hatch_note(figure)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def save_difference_heatmap(difference, path):
    """Diverging heatmap centered at zero: red = Head 1 stronger, blue = Head 2 stronger."""
    limit = max(difference.abs().max().item(), 1e-6)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    image = axis.imshow(difference.numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit)
    hatch_blocked_cells(axis, causal_mask)
    for row in range(difference.shape[0]):
        for column in range(difference.shape[1]):
            value = difference[row, column].item()
            text_color = "white" if abs(value) > 0.6 * limit else "#1a1a1a"
            axis.text(column, row, f"{value:+.2f}", ha="center", va="center", color=text_color)
    label_axes(axis)
    axis.set_title("Head 1 − Head 2 Attention", pad=12)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("← Head 2 stronger   |   Head 1 stronger →")
    add_hatch_note(figure)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def save_mask_heatmap(mask, path):
    """Two-color grid: Allowed (green) vs Blocked (gray)."""
    allowed_color, blocked_color = "#4c9a6a", "#d9d9d9"
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    # 1 = allowed, 0 = blocked. Color index 0 -> blocked, 1 -> allowed.
    axis.imshow(mask.int().numpy(), cmap=ListedColormap([blocked_color, allowed_color]), vmin=0, vmax=1)
    for row in range(mask.shape[0]):
        for column in range(mask.shape[1]):
            allowed = bool(mask[row, column])
            axis.text(column, row, "Allowed" if allowed else "Blocked", ha="center", va="center",
                      color="white" if allowed else "#555555", fontsize=10)
    label_axes(axis)
    axis.set_title("Causal Attention Mask", pad=12)
    axis.legend(handles=[Patch(color=allowed_color, label="Allowed (same or earlier)"),
                         Patch(color=blocked_color, label="Blocked (future, -inf before softmax)")],
                loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False, fontsize=9)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def main():
    # -----------------------------------------------------------------------
    # 1. Compute attention with M5's code
    # -----------------------------------------------------------------------
    heads, _, _ = multi_head_attention(X, head_matrices, W_O, causal_mask)
    head1_weights = heads[0]["weights"]
    head2_weights = heads[1]["weights"]

    # Signed difference: positive = Head 1 attends more, negative = Head 2 attends more.
    difference = head1_weights - head2_weights

    print("Head 1 weights (from M5):\n", head1_weights)
    print("Head 2 weights (from M5):\n", head2_weights)
    print("Head 1 - Head 2:\n", difference)
    print()

    # -----------------------------------------------------------------------
    # 2. Draw and save figures
    # -----------------------------------------------------------------------
    IMAGES_DIR.mkdir(exist_ok=True)
    output_paths = {
        "head1": IMAGES_DIR / "head1_attention.png",
        "head2": IMAGES_DIR / "head2_attention.png",
        "difference": IMAGES_DIR / "head_difference.png",
        "mask": IMAGES_DIR / "causal_mask.png",
    }

    save_attention_heatmap(head1_weights, "Head 1 — Causal Self-Attention", output_paths["head1"])
    save_attention_heatmap(head2_weights, "Head 2 — Causal Self-Attention", output_paths["head2"])
    save_difference_heatmap(difference, output_paths["difference"])
    save_mask_heatmap(causal_mask, output_paths["mask"])

    # -----------------------------------------------------------------------
    # 3. Verification
    # -----------------------------------------------------------------------
    future_positions = ~causal_mask

    assert head1_weights.shape == (4, 4) and head2_weights.shape == (4, 4), "Weights should be (4, 4)"
    assert causal_mask.shape == (4, 4), "Mask should be (4, 4)"
    for weights in [head1_weights, head2_weights]:
        assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=TOLERANCE), "Rows should sum to 1"
        assert (weights >= 0).all(), "Weights should be nonnegative"
        assert (weights[future_positions] == 0).all(), "Future positions should be exactly zero"
    assert torch.allclose(difference, head1_weights - head2_weights), "Difference should be head1 - head2"
    assert not torch.allclose(head1_weights, head2_weights, atol=1e-2), "Heads should differ"
    for path in output_paths.values():
        assert path.is_file(), f"{path} was not created"
        assert path.stat().st_size > 0, f"{path} is empty"

    print("Saved figures:")
    for path in output_paths.values():
        print(f"  {path.relative_to(IMAGES_DIR.parent)}  ({path.stat().st_size // 1024} KB)")
    print("All checks passed.")


if __name__ == "__main__":
    main()
