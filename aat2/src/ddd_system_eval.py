"""System-level test on real driver faces: does the eye-state CNN give a higher PERCLOS
(share of frames with closed eyes) for each person's drowsy clip than for their alert clip?"""
import json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline import model, eye_boxes

d = np.load("aat2/data/ddd_96.npz")
names = d["name"].astype(str)
person = np.array([re.match(r"[A-Za-z]+", n).group().lower() for n in names])
drowsy = d["y"] == 0
frame = np.array([int(re.search(r"\d+", n).group()) for n in names])
keep = frame % 3 == 0
gray = (d["x"][keep] @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
person, drowsy = person[keep], drowsy[keep]

# DDD images are already tight face crops, so the whole image is the face box
boxes = eye_boxes(0, 0, 96, 96)
crops = np.concatenate([np.stack([np.array(__import__("cv2").resize(g[ey:ey + s, ex:ex + s], (64, 64))) for g in gray])[..., None]
                        for ex, ey, s, _ in boxes], axis=0).astype("float32") / 255.0
p = model().predict(crops, batch_size=512, verbose=0).ravel().reshape(2, -1).mean(0)
closed = p >= 0.5

rows = []
for who in sorted(set(person)):
    m_d, m_a = (person == who) & drowsy, (person == who) & ~drowsy
    if m_d.sum() and m_a.sum():
        rows.append((who, float(closed[m_d].mean()), float(closed[m_a].mean())))
higher = sum(1 for _, a, b in rows if a > b)
res = {"people_with_both_clips": len(rows), "drowsy_clip_higher": higher,
       "mean_perclos_drowsy": float(np.mean([r[1] for r in rows])), "mean_perclos_alert": float(np.mean([r[2] for r in rows])),
       "frames": int(len(p)), "per_person": rows}
print(json.dumps({k: v for k, v in res.items() if k != "per_person"}, indent=1))
json.dump(res, open("aat2/outputs/ddd_system_eval.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(12, 3.9), dpi=220)
xs = np.arange(len(rows))
ax.bar(xs - 0.2, [r[1] * 100 for r in rows], 0.4, color="#c0392b", label="Drowsy clip")
ax.bar(xs + 0.2, [r[2] * 100 for r in rows], 0.4, color="#2e8b57", label="Alert clip")
ax.set_xticks(xs, [r[0].upper() for r in rows]); ax.set_xlabel("Person")
ax.set_ylabel("Frames with eyes closed (%)")
ax.set_title(f"PERCLOS per person on real driver videos: drowsy clip higher for {higher} of {len(rows)} people",
             fontsize=12, color="#1f3b63", weight="bold")
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig("report/figures/aat2_perclos.png", bbox_inches="tight")
