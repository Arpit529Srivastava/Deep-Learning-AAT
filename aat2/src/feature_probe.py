"""Quick probe: how well do frozen ImageNet CNN features separate drowsy from alert faces of unseen people?"""
import re, time
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

d = np.load("aat2/data/ddd_96.npz")
names = d["name"].astype(str)
person = np.array([re.match(r"[A-Za-z]+", n).group().lower() for n in names])
frame = np.array([int(re.search(r"\d+", n).group()) for n in names])
keep = frame % 4 == 0                       # every 4th frame is enough for a probe
x, y, g = d["x"][keep], (d["y"][keep] == 0).astype(int), person[keep]
print(len(y), "images")

base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(128, 128, 3), pooling="avg")
prep = tf.keras.applications.mobilenet_v2.preprocess_input

def feats(imgs):
    out = []
    for i in range(0, len(imgs), 256):
        b = tf.image.resize(imgs[i:i + 256].astype("float32"), (128, 128))
        out.append(base(prep(b), training=False).numpy())
    return np.concatenate(out)

variants = {"full face": x, "eye region": x[:, 18:58, :, :]}   # rows 18-58 of the 96 px face hold the eyes
for name, imgs in variants.items():
    t = time.time(); f = feats(imgs)
    accs = []
    for tr, te in GroupKFold(5).split(f, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(f[tr], y[tr])
        accs.append(clf.score(f[te], y[te]))
    print(f"{name}: person-wise CV accuracy {np.mean(accs):.3f} +- {np.std(accs):.3f}  ({time.time()-t:.0f}s)")
