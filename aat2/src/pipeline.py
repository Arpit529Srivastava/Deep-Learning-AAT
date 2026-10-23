"""Shared drowsiness pipeline: face -> two eye crops -> eye-state CNN -> P(eyes closed)."""
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "aat2/outputs/eye_state_cnn.keras"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_model = None


def model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def detect_face(gray):
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def eye_boxes(x, y, w, h, eye_y=0.40):
    """Square eye regions from face geometry: eyes sit about 40% down the face, 32% and 68% across.
    The side is 0.36 of the face width, which matches the scale of the training crops."""
    side = int(0.36 * w)
    boxes = []
    for cx in (0.32, 0.68):
        ex, ey = int(x + cx * w - side / 2), int(y + eye_y * h - side / 2)
        boxes.append((max(ex, 0), max(ey, 0), side, side))
    return boxes


def eye_inputs(gray, boxes):
    crops = [cv2.resize(gray[ey:ey + s, ex:ex + s], (64, 64)) for ex, ey, s, _ in boxes]
    return np.stack(crops)[..., None].astype("float32") / 255.0


def closed_probability(gray, boxes):
    p = model().predict(eye_inputs(gray, boxes), verbose=0).ravel()
    return float(p.mean()), p


def analyse_frame(rgb):
    """Returns (P(closed) averaged over both eyes, face box, eye boxes) or (None, None, None) if no face."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    f = detect_face(gray)
    if f is None:
        return None, None, None
    boxes = eye_boxes(*f)
    p, _ = closed_probability(gray, boxes)
    return p, f, boxes
