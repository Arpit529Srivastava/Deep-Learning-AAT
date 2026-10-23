"""Train the eye-state CNN (Open / Closed) used for driver drowsiness detection.

Training data: eye crops from the open/closed eyes dataset, split by face so both eyes of
one face stay in the same set. External test: real eye photographs from a different source.
"""
import json, os, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
OUT = "aat2/outputs"
os.makedirs(OUT, exist_ok=True)

s = np.load("aat2/eyes/synth_eyes_64.npz")
r = np.load("aat2/eyes/real_eyes_64.npz")
x, y, face = s["x"][..., None], s["y"].astype("float32"), s["face"]

faces = np.unique(face)
rng = np.random.default_rng(SEED)
rng.shuffle(faces)
n = len(faces)
test_f, val_f = set(faces[: int(0.15 * n)]), set(faces[int(0.15 * n): int(0.30 * n)])
in_test = np.array([f in test_f for f in face]); in_val = np.array([f in val_f for f in face])
in_train = ~(in_test | in_val)
sets = {"train": in_train, "val": in_val, "test": in_test}
for k, m in sets.items():
    print(f"{k}: {m.sum()} eyes, closed={int(y[m].sum())}")

geo_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"), layers.RandomRotation(0.08), layers.RandomZoom(0.15),
    layers.RandomTranslation(0.08, 0.08), layers.RandomContrast(0.3),
    layers.RandomBrightness(0.2, value_range=(0, 1)),
], name="augmentation")


def photo_aug(img):
    # random blur and sensor noise on both classes, so sharpness alone cannot decide the label
    if tf.random.uniform(()) < 0.35:
        img = tf.nn.avg_pool2d(img[None], 3, 1, "SAME")[0]
    if tf.random.uniform(()) < 0.35:
        img = img + tf.random.normal(tf.shape(img), stddev=0.05)
    return tf.clip_by_value(img, 0.0, 1.0)


BATCH = 128
def make_ds(xx, yy, training):
    ds = tf.data.Dataset.from_tensor_slices((xx, yy))
    if training:
        ds = ds.shuffle(len(yy), seed=SEED)
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, b), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(lambda a, b: (photo_aug(geo_aug(a, training=True)), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(x[in_train], y[in_train], True)
val_ds = make_ds(x[in_val], y[in_val], False)


def conv_block(filters):
    return [layers.Conv2D(filters, 3, padding="same", use_bias=False), layers.BatchNormalization(momentum=0.9), layers.ReLU(),
            layers.Conv2D(filters, 3, padding="same", use_bias=False), layers.BatchNormalization(momentum=0.9), layers.ReLU(),
            layers.MaxPooling2D(), layers.Dropout(0.25)]

model = models.Sequential(
    [layers.Input((64, 64, 1))] + conv_block(32) + conv_block(64) + conv_block(128)
    + [layers.GlobalAveragePooling2D(), layers.Dense(128, activation="relu"), layers.Dropout(0.5),
       layers.Dense(1, activation="sigmoid")],
    name="eye_state_cnn")
model.summary()
with open(f"{OUT}/model_summary.txt", "w") as f:
    model.summary(print_fn=lambda t, **k: f.write(t + "\n"))

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])
EPOCHS = int(os.environ.get("EPOCHS", 20))
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2, callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)])
model.save(f"{OUT}/eye_state_cnn.keras")
json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, open(f"{OUT}/history.json", "w"), indent=1)


def evaluate(xx, yy):
    prob = model.predict(make_ds(xx, yy, False), verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(yy, prob)
    return prob, fpr, tpr, {
        "n": int(len(yy)), "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "accuracy": float((tp + tn) / len(yy)), "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)), "auc": float(roc_auc_score(yy, prob))}

metrics = {"counts": {k: int(m.sum()) for k, m in sets.items()}, "params": int(model.count_params()),
           "epochs_run": len(history.history["loss"])}
for name, (xx, yy) in {"test": (x[in_test], y[in_test]), "real": (r["x"][..., None], r["y"].astype("float32"))}.items():
    prob, fpr, tpr, m = evaluate(xx, yy)
    metrics[name] = m
    np.savez(f"{OUT}/pred_{name}.npz", prob=prob, y=yy, fpr=fpr, tpr=tpr, idx=np.where(in_test)[0] if name == "test" else np.arange(len(yy)))
print(json.dumps(metrics, indent=1))
json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=1)
