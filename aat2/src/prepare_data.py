"""Decode the Driver Drowsiness Dataset parquet files into one compact NumPy archive."""
import glob, io
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

IMG_SIZE = 96
images, labels, subjects, names = [], [], [], []
for path in sorted(glob.glob("aat2/data/*.parquet")):
    table = pq.read_table(path)
    for row in table.to_pylist():
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        images.append(np.asarray(img, dtype=np.uint8))
        labels.append(row["label"])                 # 0 = Drowsy, 1 = Non Drowsy
        name = row["image"]["path"]
        names.append(name)
        subjects.append(name[0])                    # first letter identifies the person
    print(path, len(images))

np.savez("aat2/data/ddd_96.npz", x=np.stack(images), y=np.array(labels), subject=np.array(subjects), name=np.array(names))
print("saved", len(images), "images")
