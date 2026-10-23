"""Crop left and right eyes from the open/closed eyes dataset and build a real-world test set.

Run with a Python that has pyarrow (kept out of the TensorFlow environment, where it clashes).
"""
import glob, io, os
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

SIZE = 64


def crop_eye(img, box):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    side = max(1.6 * w, 24)                 # square crop sized by eye width, so eye height does not leak the label
    l, t = int(cx - side / 2), int(cy - side / 2)
    return img.crop((l, t, l + int(side), t + int(side))).convert("L").resize((SIZE, SIZE), Image.BILINEAR)


xs, ys, face_ids = [], [], []
for path in sorted(glob.glob("aat2/eyes/synth/*.parquet")):
    for r in pq.read_table(path).to_pylist():
        img = Image.open(io.BytesIO(r["Image_data"]["file"])).convert("RGB")
        label = 1 if r["Label"] == "closed_eyes" else 0          # 1 = closed
        for box in (r["Left_eye_react"], r["Right_eye_react"]):
            if box and box[2] > 8:
                xs.append(np.asarray(crop_eye(img, box), np.uint8)); ys.append(label)
                face_ids.append(f"{os.path.basename(path)}:{r['Image_id']}")
np.savez("aat2/eyes/synth_eyes_64.npz", x=np.stack(xs), y=np.array(ys), face=np.array(face_ids))
print("synthetic eyes:", len(ys), "closed:", int(np.sum(ys)))

rx, ry = [], []
for label, folder in [(1, "Closed"), (0, "Open")]:
    for f in sorted(glob.glob(f"aat2/eyes/real/{folder}/*.jpg")):
        rx.append(np.asarray(Image.open(f).convert("L").resize((SIZE, SIZE), Image.BILINEAR), np.uint8)); ry.append(label)
np.savez("aat2/eyes/real_eyes_64.npz", x=np.stack(rx), y=np.array(ry))
print("real eyes:", len(ry), "closed:", int(np.sum(ry)))
