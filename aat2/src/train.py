"""Train a CNN that classifies a driver's face as Drowsy or Non Drowsy.

The split is done by person, so no face in the test set is seen during training.
"""
import json, os, random, re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
OUT = "aat2/outputs_face_cnn"
os.makedirs(OUT, exist_ok=True)

# ---------------- data ----------------
d = np.load("aat2/data/ddd_96.npz")
x, y_raw, subject = d["x"], d["y"], d["subject"]
y = (y_raw == 0).astype(np.float32)           # 1 = Drowsy (positive class), 0 = Non Drowsy
# file names look like A0001.png or ZB0001.png: the letters identify the person,
# upper case is a drowsy clip and lower case an alert clip of the same person
names = d["name"].astype(str)
person = np.array([re.match(r"[A-Za-z]+", n).group().lower() for n in names])

people = sorted(set(person))
rng = np.random.default_rng(SEED)
rng.shuffle(people)
test_people, val_people, train_people = people[:6], people[6:10], people[10:]
split = {"train": sorted(map(str, train_people)), "val": sorted(map(str, val_people)), "test": sorted(map(str, test_people))}
# consecutive video frames are almost identical, so every 2nd frame is kept
frame_no = np.array([int(re.search(r"\d+", n).group()) for n in names])
keep = frame_no % 2 == 0
masks = {k: np.isin(person, v) & keep for k, v in split.items()}
for k, m in masks.items():
    print(f"{k}: {m.sum()} images, {len(split[k])} people {[str(p) for p in split[k]]}, drowsy={int(y[m].sum())}")

BATCH = 64
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.15, value_range=(0, 1)),
], name="augmentation")

def make_ds(mask, training):
    ds = tf.data.Dataset.from_tensor_slices((x[mask], y[mask]))
    if training:
        ds = ds.shuffle(len(y[mask]), seed=SEED)
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, b), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(lambda a, b: (augment(a, training=True), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds, val_ds, test_ds = make_ds(masks["train"], True), make_ds(masks["val"], False), make_ds(masks["test"], False)

# ---------------- model ----------------
def conv_block(filters):
    return [layers.Conv2D(filters, 3, padding="same", use_bias=False),
            layers.BatchNormalization(momentum=0.9), layers.ReLU(),
            layers.MaxPooling2D(), layers.Dropout(0.25)]

model = models.Sequential(
    [layers.Input((96, 96, 3))]
    + conv_block(32) + conv_block(64) + conv_block(128) + conv_block(256)
    + [layers.GlobalAveragePooling2D(),
       layers.Dense(128, activation="relu"),
       layers.Dropout(0.5),
       layers.Dense(1, activation="sigmoid")],
    name="drowsiness_cnn")
model.summary()
with open(f"{OUT}/model_summary.txt", "w") as f:
    model.summary(print_fn=lambda s: f.write(s + "\n"))

model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
              loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
]
EPOCHS = int(os.environ.get("EPOCHS", 25))
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)
model.save(f"{OUT}/drowsiness_cnn.keras")
json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, open(f"{OUT}/history.json", "w"), indent=1)

# ---------------- evaluation ----------------
prob = model.predict(test_ds, verbose=0).ravel()
y_test = y[masks["test"]]
pred = (prob >= 0.5).astype(int)
cm = confusion_matrix(y_test, pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
metrics = {
    "split": split,
    "counts": {k: int(m.sum()) for k, m in masks.items()},
    "epochs_run": len(history.history["loss"]),
    "params": int(model.count_params()),
    "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    "accuracy": float((tp + tn) / len(y_test)),
    "precision": float(tp / (tp + fp)),
    "recall": float(tp / (tp + fn)),
    "specificity": float(tn / (tn + fp)),
    "f1": float(2 * tp / (2 * tp + fp + fn)),
    "auc": float(roc_auc_score(y_test, prob)),
}
print(json.dumps(metrics, indent=1))
print(classification_report(y_test, pred, target_names=["Non Drowsy", "Drowsy"], digits=4))
json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=1)
fpr, tpr, _ = roc_curve(y_test, prob)
np.savez(f"{OUT}/test_predictions.npz", prob=prob, y=y_test, fpr=fpr, tpr=tpr,
         idx=np.where(masks["test"])[0])
