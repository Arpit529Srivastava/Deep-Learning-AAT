"""Result figures for AAT 2: training curves, confusion matrices, ROC, sample predictions and Grad-CAM."""
import json, os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT, FIG = "aat2/outputs", "report/figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
NAVY, BLUE, ORANGE, RED, GREEN = "#1f3b63", "#3b6ea8", "#e07b39", "#c0392b", "#2e8b57"

hist = json.load(open(f"{OUT}/history.json"))
m = json.load(open(f"{OUT}/metrics.json"))
ep = np.arange(1, len(hist["loss"]) + 1)

# ---------- training curves ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=220)
for ax, key, name in [(axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")]:
    ax.plot(ep, hist[key], "-o", ms=3.5, color=BLUE, label=f"Train {name.lower()}")
    ax.plot(ep, hist["val_" + key], "-o", ms=3.5, color=ORANGE, label=f"Validation {name.lower()}")
    ax.set_title(name, fontsize=13, color=NAVY, weight="bold"); ax.set_xlabel("Epoch")
    ax.set_xticks(ep); ax.grid(alpha=0.3); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/aat2_training_curves.png", bbox_inches="tight"); plt.close(fig)

# ---------- confusion matrices ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=220)
for ax, key, name in [(axes[0], "test", f"Held-out test set ({m['test']['n']:,} eyes)"),
                      (axes[1], "real", f"Real-world eye photos ({m['real']['n']:,} eyes)")]:
    r = m[key]
    cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Open", "Closed"]); ax.set_yticks([0, 1], ["Open", "Closed"])
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum() * 100:.1f}%)", ha="center", va="center", fontsize=13,
                    weight="bold", color="white" if cm[i, j] > cm.max() / 2 else NAVY)
    ax.set_title(name, fontsize=12, color=NAVY, weight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/aat2_confusion_matrix.png", bbox_inches="tight"); plt.close(fig)

# ---------- ROC ----------
fig, ax = plt.subplots(figsize=(6, 4.6), dpi=220)
for key, name, col in [("test", "Held-out test", BLUE), ("real", "Real-world photos", ORANGE)]:
    p = np.load(f"{OUT}/pred_{key}.npz")
    ax.plot(p["fpr"], p["tpr"], color=col, lw=2.2, label=f"{name} (AUC = {m[key]['auc']:.3f})")
ax.plot([0, 1], [0, 1], "--", color="#999999", label="Random guess")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
ax.set_title("ROC curves of the eye-state CNN", fontsize=12, color=NAVY, weight="bold")
ax.legend(frameon=False, loc="lower right"); ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/aat2_roc.png", bbox_inches="tight"); plt.close(fig)

# ---------- predictions on real photos ----------
real = np.load("aat2/eyes/real_eyes_64.npz")
pr = np.load(f"{OUT}/pred_real.npz")
prob, y = pr["prob"], pr["y"].astype(int)
pred = (prob >= 0.5).astype(int)
rng = np.random.default_rng(3)
picks = []
for mask, k in [((y == 0) & (pred == 0), 4), ((y == 1) & (pred == 1), 4), (y != pred, 4)]:
    idx = np.where(mask)[0]
    picks += list(rng.choice(idx, min(k, len(idx)), replace=False))
fig, axes = plt.subplots(2, 6, figsize=(12, 4.8), dpi=220)
for k, i in enumerate(picks[:12]):
    ax = axes.flat[k]
    ax.imshow(real["x"][i], cmap="gray", vmin=0, vmax=255); ax.set_xticks([]); ax.set_yticks([])
    ok = pred[i] == y[i]
    ax.set_title(f"pred: {'Closed' if pred[i] else 'Open'} ({prob[i]:.2f})\ntrue: {'Closed' if y[i] else 'Open'}",
                 fontsize=9, color=GREEN if ok else RED)
    for sp in ax.spines.values():
        sp.set_edgecolor(GREEN if ok else RED); sp.set_linewidth(2.5)
fig.suptitle("Predictions on real eye photographs (green = correct, red = wrong, value = P(closed))",
             fontsize=12, color=NAVY, weight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/aat2_predictions.png", bbox_inches="tight"); plt.close(fig)

# ---------- Grad-CAM ----------
model = tf.keras.models.load_model(f"{OUT}/eye_state_cnn.keras")
last_conv = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1]
inp = tf.keras.Input((64, 64, 1)); h = inp; conv_out = None
for layer in model.layers:
    h = layer(h)
    if layer is last_conv:
        conv_out = h
grad_model = tf.keras.Model(inp, [conv_out, h])


def gradcam(img, closed):
    with tf.GradientTape() as tape:
        conv, out = grad_model(img[None, ..., None].astype("float32") / 255.0)
        score = out[:, 0] if closed else 1 - out[:, 0]
    g = tape.gradient(score, conv)[0]
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * tf.reduce_mean(g, axis=(0, 1)), axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    return tf.image.resize(cam[..., None], (64, 64)).numpy()[..., 0]

cam_ids = picks[:2] + picks[4:6]
fig, axes = plt.subplots(2, 4, figsize=(10, 5.3), dpi=220)
for c, i in enumerate(cam_ids):
    axes[0, c].imshow(real["x"][i], cmap="gray"); axes[0, c].set_title(f"{'Closed' if y[i] else 'Open'} eye, P(closed) = {prob[i]:.2f}", fontsize=9.5)
    axes[1, c].imshow(real["x"][i], cmap="gray"); axes[1, c].imshow(gradcam(real["x"][i], bool(pred[i])), cmap="jet", alpha=0.45)
    axes[0, c].axis("off"); axes[1, c].axis("off")
fig.suptitle("Grad-CAM: image regions the CNN uses for its decision (red = strongest)", fontsize=12, color=NAVY, weight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/aat2_gradcam.png", bbox_inches="tight"); plt.close(fig)
print("result figures written")
