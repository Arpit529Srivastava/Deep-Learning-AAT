"""System diagrams for AAT 2 (block diagram, CNN architecture, workflow, dataset samples)."""
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "report/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans"})
NAVY, BLUE, TEAL, ORANGE, RED, GREEN, GREY = "#1f3b63", "#3b6ea8", "#2a8c8c", "#e07b39", "#c0392b", "#2e8b57", "#555555"


def box(ax, x, y, w, h, head, sub="", fc=NAVY, fs=10.5, sfs=8.5, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec="none"))
    if sub:
        ax.text(x + w / 2, y + h * 0.68, head, ha="center", va="center", color=tc, fontsize=fs, weight="bold")
        ax.text(x + w / 2, y + h * 0.32, sub, ha="center", va="center", color=tc, fontsize=sfs)
    else:
        ax.text(x + w / 2, y + h / 2, head, ha="center", va="center", color=tc, fontsize=fs, weight="bold")


def arrow(ax, p0, p1, color=GREY, label=None, lpos=None):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, color=color, lw=1.5))
    if label:
        lx, ly = lpos if lpos else ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=9, color=color, weight="bold",
                bbox=dict(fc="white", ec="none", pad=1))


# ---------- 1. Block diagram ----------
fig, ax = plt.subplots(figsize=(12, 2.6), dpi=220)
ax.set_xlim(0, 12); ax.set_ylim(0, 2.6); ax.axis("off")
steps = [("Camera", "driver video\nframe", NAVY), ("Face Detection", "Haar cascade\nface box", BLUE),
         ("Eye Extraction", "two 64x64\ngrayscale crops", BLUE), ("Eye-state CNN", "P(eyes closed)\nper frame", BLUE),
         ("Frame Counter", "closed for 15\nframes in a row?", TEAL), ("Alert", "alarm and\nwarning text", ORANGE)]
w, gap = 1.65, 0.36
for i, (h, s, c) in enumerate(steps):
    x = 0.15 + i * (w + gap)
    box(ax, x, 0.55, w, 1.15, h, s, fc=c)
    if i < len(steps) - 1:
        arrow(ax, (x + w + 0.03, 1.12), (x + w + gap - 0.03, 1.12))
ax.text(6, 2.2, "Block diagram of the driver drowsiness detection system", ha="center", fontsize=13, weight="bold", color=NAVY)
fig.savefig(f"{OUT}/aat2_block_diagram.png", bbox_inches="tight"); plt.close(fig)

# ---------- 2. CNN architecture ----------
fig, ax = plt.subplots(figsize=(12.5, 4.4), dpi=220)
ax.set_xlim(0, 12.5); ax.set_ylim(0, 4.4); ax.axis("off")
layers_ = [("Input\neye crop", "64x64x1", 2.6, "#7f8c8d"),
           ("Block 1\n2x Conv 32\nPool", "32x32x32", 2.3, BLUE), ("Block 2\n2x Conv 64\nPool", "16x16x64", 1.9, BLUE),
           ("Block 3\n2x Conv 128\nPool", "8x8x128", 1.5, BLUE),
           ("Global\nAvg Pool", "128", 1.0, TEAL), ("Dense 128\nReLU", "128", 1.0, TEAL), ("Dropout\n0.5", "128", 0.9, TEAL),
           ("Dense 1\nSigmoid", "P(closed)", 0.8, ORANGE)]
x = 0.15
cx = []
for i, (name, shape, h, c) in enumerate(layers_):
    wbox = 1.15
    y0 = 2.1 - h / 2
    # draw a 3D-looking slab
    d = 0.18
    ax.add_patch(plt.Polygon([[x, y0 + h], [x + d, y0 + h + d], [x + wbox + d, y0 + h + d], [x + wbox, y0 + h]], fc=c, ec="white", alpha=0.55))
    ax.add_patch(plt.Polygon([[x + wbox, y0], [x + wbox + d, y0 + d], [x + wbox + d, y0 + h + d], [x + wbox, y0 + h]], fc=c, ec="white", alpha=0.75))
    ax.add_patch(plt.Rectangle((x, y0), wbox, h, fc=c, ec="white"))
    ax.text(x + wbox / 2, y0 + h / 2, name, ha="center", va="center", color="white", fontsize=8.6, weight="bold")
    ax.text(x + wbox / 2, 0.45, shape, ha="center", va="center", color=NAVY, fontsize=9)
    cx.append(x)
    if i < len(layers_) - 1:
        arrow(ax, (x + wbox + d + 0.02, 2.1), (x + wbox + 0.42, 2.1))
    x += wbox + 0.4
ax.text(6.25, 4.05, "Architecture of the eye-state CNN (304,417 parameters)", ha="center", fontsize=13, weight="bold", color=NAVY)
ax.text(6.25, 0.08, "Output shape after each stage. Each block: two (3x3 Conv2D, BatchNorm, ReLU), then 2x2 MaxPool and Dropout 0.25",
        ha="center", fontsize=9, color=GREY)
fig.savefig(f"{OUT}/aat2_cnn_architecture.png", bbox_inches="tight"); plt.close(fig)

# ---------- 3. Workflow flowchart (training and real-time use) ----------
fig, ax = plt.subplots(figsize=(11, 7.4), dpi=220)
ax.set_xlim(0, 11); ax.set_ylim(0, 7.4); ax.axis("off")
ax.text(2.75, 7.1, "Training phase", ha="center", fontsize=12.5, weight="bold", color=NAVY)
ax.text(8.25, 7.1, "Real-time detection phase", ha="center", fontsize=12.5, weight="bold", color=NAVY)
ax.plot([5.5, 5.5], [0.2, 7.2], color="#cccccc", lw=1, ls="--")
L = [("Load 21,099 face images", "open / closed eyes dataset"), ("Crop 42,131 eyes", "64x64 grayscale, scale 0-1"),
     ("Split by face", "70% train / 15% val / 15% test"), ("Augment training eyes", "flip, rotate, blur, noise"),
     ("Train eye-state CNN", "Adam, binary cross-entropy"), ("Evaluate", "held-out test + 3,486 real eyes"),
     ("Save model", "eye_state_cnn.keras")]
bw, bh, top, step = 3.6, 0.62, 6.25, 0.92
for i, (h, s) in enumerate(L):
    y = top - i * step
    box(ax, 0.95, y, bw, bh, h, s, fc=NAVY if i in (0, 6) else BLUE, fs=9.8, sfs=8)
    if i < len(L) - 1:
        arrow(ax, (2.75, y - 0.02), (2.75, y - step + bh + 0.02))
R = [("Capture webcam frame", "OpenCV VideoCapture"), ("Detect face", "Haar cascade"),
     ("Crop both eyes", "64x64 grayscale"), ("CNN prediction", "mean P(closed) of two eyes")]
for i, (h, s) in enumerate(R):
    y = top - i * step
    box(ax, 6.45, y, bw, bh, h, s, fc=TEAL, fs=9.8, sfs=8)
    if i < len(R) - 1:
        arrow(ax, (8.25, y - 0.02), (8.25, y - step + bh + 0.02))
# decision diamond
dy = top - 4 * step - 0.15
diamond = plt.Polygon([[8.25, dy + 0.75], [9.65, dy + 0.3], [8.25, dy - 0.15], [6.85, dy + 0.3]], fc="#f6d7c3", ec=ORANGE, lw=1.6)
ax.add_patch(diamond)
ax.text(8.25, dy + 0.3, "Closed for\n15 frames in a row?", ha="center", va="center", fontsize=8.8, color=NAVY, weight="bold")
arrow(ax, (8.25, top - 3 * step - 0.02), (8.25, dy + 0.77))
ay = dy - 1.3
box(ax, 6.45, ay, bw, bh, "Sound alarm", "show DROWSY warning", fc=RED, fs=9.8, sfs=8)
arrow(ax, (8.25, dy - 0.17), (8.25, ay + bh + 0.02), color=RED, label="Yes", lpos=(8.6, dy - 0.5))
# No loop back to capture
ax.add_patch(FancyArrowPatch((9.65, dy + 0.3), (10.45, dy + 0.3), arrowstyle="-", color=GREEN, lw=1.5))
ax.add_patch(FancyArrowPatch((10.45, dy + 0.3), (10.45, top + bh / 2), arrowstyle="-", color=GREEN, lw=1.5))
ax.add_patch(FancyArrowPatch((10.45, top + bh / 2), (6.45 + bw + 0.02, top + bh / 2), arrowstyle="-|>", mutation_scale=14, color=GREEN, lw=1.5))
ax.text(10.05, dy + 0.48, "No", fontsize=9, color=GREEN, weight="bold")
ax.add_patch(FancyArrowPatch((6.45, ay + bh / 2), (6.05, ay + bh / 2), arrowstyle="-", color=GREY, lw=1.3))
ax.add_patch(FancyArrowPatch((6.05, ay + bh / 2), (6.05, top + bh / 2), arrowstyle="-", color=GREY, lw=1.3))
ax.add_patch(FancyArrowPatch((6.05, top + bh / 2), (6.43, top + bh / 2), arrowstyle="-|>", mutation_scale=14, color=GREY, lw=1.3))
ax.text(5.85, (ay + top) / 2 + 0.3, "next frame", rotation=90, ha="center", va="center", fontsize=8.5, color=GREY)
# model link
ax.add_patch(FancyArrowPatch((0.95 + bw + 0.02, top - 6 * step + bh / 2), (6.45 - 0.02, top - 3 * step + bh / 2),
                             arrowstyle="-|>", mutation_scale=14, color=ORANGE, lw=1.4, ls="--", connectionstyle="arc3,rad=0.25"))
ax.text(4.95, top - 6 * step + 1.05, "trained\nmodel", ha="center", fontsize=8.5, color=ORANGE, weight="bold")
fig.savefig(f"{OUT}/aat2_workflow.png", bbox_inches="tight"); plt.close(fig)

# ---------- 4. Dataset samples: source faces, eye crops, real eyes ----------
syn = np.load("aat2/eyes/synth_eyes_64.npz"); real = np.load("aat2/eyes/real_eyes_64.npz")
rng = np.random.default_rng(11)
fig, axes = plt.subplots(4, 8, figsize=(12, 6.6), dpi=220)
rows = [(syn, 0, "Training: open"), (syn, 1, "Training: closed"), (real, 0, "Real test: open"), (real, 1, "Real test: closed")]
for r, (data, lab, name) in enumerate(rows):
    idx = rng.choice(np.where(data["y"] == lab)[0], 8, replace=False)
    for c, i in enumerate(idx):
        ax = axes[r, c]; ax.imshow(data["x"][i], cmap="gray", vmin=0, vmax=255); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(GREEN if lab == 0 else RED); sp.set_linewidth(2.2)
    axes[r, 0].set_ylabel(name, fontsize=10.5, weight="bold", color=GREEN if lab == 0 else RED)
fig.suptitle("Eye crops used for training (rows 1-2) and the separate real-world test set (rows 3-4)",
             fontsize=12.5, weight="bold", color=NAVY)
fig.tight_layout(); fig.savefig(f"{OUT}/aat2_dataset_samples.png", bbox_inches="tight"); plt.close(fig)

# ---------- 5. Source face with the two eye boxes ----------
from PIL import Image
face = Image.open("aat2/eyes/probe/m_0.jpg")
fig, ax = plt.subplots(1, 3, figsize=(10, 3.6), dpi=220, gridspec_kw={"width_ratios": [2.2, 1, 1]})
ax[0].imshow(face)
for (x, y, w, h) in [(329, 219, 29, 9), (230, 222, 49, 10)]:
    side = max(1.6 * w, 24); cx, cy = x + w / 2, y + h / 2
    ax[0].add_patch(plt.Rectangle((cx - side / 2, cy - side / 2), side, side, fill=False, ec="#f1c40f", lw=2))
ax[0].set_title("Source image with eye boxes", fontsize=10.5, color=NAVY)
for k, (x, y, w, h) in enumerate([(230, 222, 49, 10), (329, 219, 29, 9)]):
    side = max(1.6 * w, 24); cx, cy = x + w / 2, y + h / 2
    crop = face.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))).convert("L").resize((64, 64))
    ax[k + 1].imshow(crop, cmap="gray"); ax[k + 1].set_title(f"{'Right' if k == 0 else 'Left'} eye, 64x64", fontsize=10.5, color=NAVY)
for a in ax:
    a.axis("off")
fig.suptitle("Eye extraction: a square crop of 1.6 times the eye width around each eye", fontsize=12, weight="bold", color=NAVY)
fig.tight_layout(); fig.savefig(f"{OUT}/aat2_eye_extraction.png", bbox_inches="tight"); plt.close(fig)

# ---------- 6. Data split ----------
fig, ax = plt.subplots(figsize=(11, 1.9), dpi=220)
ax.set_xlim(0, 11); ax.set_ylim(0, 1.9); ax.axis("off")
parts = [("Train: 29,495 eyes", 29495, BLUE), ("Val: 6,317", 6317, TEAL), ("Test: 6,319", 6319, ORANGE), ("Real: 3,486", 3486, RED)]
x = 0.2; total = sum(p[1] for p in parts); W = 10.6
for lab, n, c in parts:
    w = W * n / total
    ax.add_patch(plt.Rectangle((x, 0.55), w, 0.7, fc=c, ec="white", lw=2))
    ax.text(x + w / 2, 0.9, lab, ha="center", va="center", color="white", fontsize=10, weight="bold")
    x += w
ax.text(5.5, 1.55, "Data split: both eyes of a face stay in the same set; the real set comes from a different source",
        ha="center", fontsize=11.5, weight="bold", color=NAVY)
ax.text(5.5, 0.25, "Train, validation and test come from 21,099 faces; the real-world set is 3,486 photographs never used in training",
        ha="center", fontsize=9, color=GREY)
fig.savefig(f"{OUT}/aat2_split.png", bbox_inches="tight"); plt.close(fig)
print("diagrams written")
