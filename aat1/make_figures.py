"""Figures for AAT 1 (basic tensor operations). Values are computed with TensorFlow."""
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "report/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
NAVY, BLUE, LIGHT, ORANGE, GREY = "#1f3b63", "#3b6ea8", "#dce8f5", "#e07b39", "#555555"

a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])


def draw_matrix(ax, m, x, y, cell=0.5, fc=LIGHT, fmt="{:g}", fs=12):
    m = np.atleast_2d(np.asarray(m))
    r, c = m.shape
    for i in range(r):
        for j in range(c):
            ax.add_patch(plt.Rectangle((x + j * cell, y - (i + 1) * cell), cell, cell, fc=fc, ec=NAVY, lw=1.2))
            ax.text(x + (j + .5) * cell, y - (i + .5) * cell, fmt.format(m[i, j]), ha="center", va="center", fontsize=fs, color=NAVY)
    return c * cell, r * cell


# ---------- Fig: tensor ranks ----------
fig, ax = plt.subplots(figsize=(11, 2.9), dpi=200)
ax.set_xlim(0, 11); ax.axis("off")
items = [("Rank 0: Scalar", "shape ()", [[7]]), ("Rank 1: Vector", "shape (3,)", [[1, 2, 3]]),
         ("Rank 2: Matrix", "shape (2, 2)", [[1, 2], [3, 4]])]
xs = [0.9, 2.9, 5.6]
for (title, shape, m), x in zip(items, xs):
    w, h = draw_matrix(ax, m, x, 2.45)
    ax.text(x + w / 2, 2.95, title, ha="center", fontsize=12, weight="bold", color=NAVY)
    ax.text(x + w / 2, 2.45 - h - 0.35, shape, ha="center", fontsize=10.5, color=GREY)
# rank 3 as two 2x2 slices
draw_matrix(ax, [[1, 2], [3, 4]], 7.9, 2.45)
draw_matrix(ax, [[5, 6], [7, 8]], 9.2, 2.45, fc="#c9daef")
ax.text(8.4, 1.25 + 0.0, "slice 0", ha="center", fontsize=9, color=GREY)
ax.text(9.7, 1.25, "slice 1", ha="center", fontsize=9, color=GREY)
ax.text(9.05, 2.95, "Rank 3: 3D Tensor", ha="center", fontsize=12, weight="bold", color=NAVY)
ax.text(9.05, 0.85, "shape (2, 2, 2)", ha="center", fontsize=10.5, color=GREY)
ax.set_ylim(0.6, 3.3)
fig.savefig(f"{OUT}/aat1_tensor_ranks.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig: element-wise operations ----------
ops = [("tf.add(a, b)", tf.add(a, b), "{:g}"), ("tf.subtract(a, b)", tf.subtract(a, b), "{:g}"),
       ("tf.multiply(a, b)", tf.multiply(a, b), "{:g}"), ("tf.divide(a, b)", tf.divide(a, b), "{:.2f}")]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), dpi=200)
for ax, (name, res, fmt) in zip(axes, ops):
    ax.set_xlim(0, 2.3); ax.set_ylim(0, 2.4); ax.axis("off")
    draw_matrix(ax, res.numpy(), 0.45, 1.75, cell=0.7, fc="#fde7d6", fmt=fmt, fs=13)
    ax.text(1.15, 2.1, name, ha="center", fontsize=12, family="monospace", color=NAVY, weight="bold")
fig.suptitle("Element-wise operations on a = [[1, 2], [3, 4]] and b = [[5, 6], [7, 8]]", fontsize=13, color=NAVY, y=1.0)
fig.savefig(f"{OUT}/aat1_elementwise.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig: element-wise multiply vs matmul ----------
fig, ax = plt.subplots(figsize=(11, 3.2), dpi=200)
ax.set_xlim(0, 11); ax.set_ylim(1.2, 4.2); ax.axis("off")
ax.text(2.6, 3.9, "Element-wise: tf.multiply(a, b)", ha="center", fontsize=12.5, weight="bold", color=NAVY)
draw_matrix(ax, a.numpy(), 0.3, 3.3, cell=0.6); ax.text(1.75, 2.7, "x", fontsize=16, ha="center", va="center")
draw_matrix(ax, b.numpy(), 2.0, 3.3, cell=0.6); ax.text(3.45, 2.7, "=", fontsize=16, ha="center", va="center")
draw_matrix(ax, tf.multiply(a, b).numpy(), 3.7, 3.3, cell=0.6, fc="#fde7d6")
ax.text(2.6, 1.6, "Each cell is multiplied by the cell\nin the same position: 1x5, 2x6, 3x7, 4x8", ha="center", fontsize=10.5, color=GREY)
ax.plot([5.5, 5.5], [1.3, 4.0], color="#bbbbbb", lw=1)
ax.text(8.3, 3.9, "Matrix product: tf.matmul(a, b)", ha="center", fontsize=12.5, weight="bold", color=NAVY)
draw_matrix(ax, a.numpy(), 6.0, 3.3, cell=0.6)
ax.add_patch(plt.Rectangle((6.0, 2.7), 1.2, 0.6, fill=False, ec=ORANGE, lw=2.5))
ax.text(7.45, 2.7, "x", fontsize=16, ha="center", va="center")
draw_matrix(ax, b.numpy(), 7.7, 3.3, cell=0.6)
ax.add_patch(plt.Rectangle((7.7, 2.1), 0.6, 1.2, fill=False, ec=ORANGE, lw=2.5))
ax.text(9.15, 2.7, "=", fontsize=16, ha="center", va="center")
draw_matrix(ax, tf.matmul(a, b).numpy(), 9.4, 3.3, cell=0.6, fc="#fde7d6")
ax.add_patch(plt.Rectangle((9.4, 2.7), 0.6, 0.6, fill=False, ec=ORANGE, lw=2.5))
ax.text(8.3, 1.6, "Row of a times column of b, then summed:\n(1x5) + (2x7) = 19", ha="center", fontsize=10.5, color=GREY)
fig.savefig(f"{OUT}/aat1_mul_vs_matmul.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig: shape operations and reductions ----------
fig, ax = plt.subplots(figsize=(12, 3.2), dpi=200)
ax.set_xlim(0, 12); ax.set_ylim(0.8, 3.6); ax.axis("off")
draw_matrix(ax, a.numpy(), 0.3, 2.6, cell=0.6); ax.text(0.9, 3.0, "a", ha="center", fontsize=13, weight="bold", color=NAVY)
def arrow(x0, x1, y, label):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=16, color=BLUE, lw=1.6))
    ax.text((x0 + x1) / 2, y + 0.18, label, ha="center", fontsize=9.5, family="monospace", color=NAVY)
arrow(1.6, 3.4, 2.0, "tf.transpose(a)")
draw_matrix(ax, tf.transpose(a).numpy(), 3.6, 2.6, cell=0.6, fc="#fde7d6")
arrow(4.9, 7.0, 2.0, "tf.reshape(a, [4, 1])")
ax.text(5.95, 1.7, "(applied to the original a)", ha="center", fontsize=8.5, color=GREY)
draw_matrix(ax, tf.reshape(a, [4, 1]).numpy(), 7.2, 3.2, cell=0.55, fc="#fde7d6")
red = [("tf.reduce_sum(a)", tf.reduce_sum(a).numpy()), ("tf.reduce_mean(a)", tf.reduce_mean(a).numpy()),
       ("tf.reduce_max(a)", tf.reduce_max(a).numpy()), ("tf.argmax(a, axis=1)", str(tf.argmax(a, axis=1).numpy().tolist()))]
ax.text(10.2, 3.25, "Reductions on a", ha="center", fontsize=12, weight="bold", color=NAVY)
for k, (n, v) in enumerate(red):
    ax.text(8.4, 2.75 - k * 0.55, n, fontsize=10, family="monospace", color=NAVY, va="center")
    ax.text(11.9, 2.75 - k * 0.55, str(v), fontsize=11, weight="bold", color=ORANGE, va="center", ha="right")
fig.savefig(f"{OUT}/aat1_shape_reduce.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig: workflow of the experiment ----------
def flow(path, steps, title=None, colors=None, w=1.9, h=0.95, figw=12):
    n = len(steps)
    fig, ax = plt.subplots(figsize=(figw, 2.1), dpi=200)
    gap = (figw - 0.4 - n * w) / (n - 1)
    ax.set_xlim(0, figw); ax.set_ylim(0, 2.1); ax.axis("off")
    for i, (head, sub) in enumerate(steps):
        x = 0.2 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.55), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=(colors[i] if colors else NAVY), ec="none"))
        ax.text(x + w / 2, 0.55 + h * 0.66, head, ha="center", va="center", color="white", fontsize=10.5, weight="bold")
        ax.text(x + w / 2, 0.55 + h * 0.3, sub, ha="center", va="center", color="white", fontsize=8.3)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, 0.55 + h / 2), (x + w + gap - 0.03, 0.55 + h / 2),
                                         arrowstyle="-|>", mutation_scale=14, color=GREY, lw=1.4))
    if title:
        ax.text(figw / 2, 1.85, title, ha="center", fontsize=13, weight="bold", color=NAVY)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)

flow(f"{OUT}/aat1_workflow.png",
     [("Import", "TensorFlow 2.20"), ("Create", "constant, zeros,\nones, random"),
      ("Arithmetic", "add, subtract,\nmultiply, divide"), ("Matrix ops", "matmul, transpose,\nreshape, square"),
      ("Reduce", "sum, mean,\nmax, argmax"), ("Variables", "tf.Variable\n.assign()")],
     title="Workflow of the tensor operations experiment",
     colors=[NAVY, BLUE, BLUE, BLUE, BLUE, ORANGE], w=1.65)

if __name__ == "__main__":
    print("AAT 1 figures written to", OUT)
