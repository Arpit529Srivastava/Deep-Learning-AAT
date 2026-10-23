"""Gradio web interface: upload a photo of a driver and get the eye state and drowsiness status."""
import json
import cv2
import gradio as gr
from pipeline import analyse_frame

metrics = json.load(open("aat2/outputs/metrics.json"))


def predict(image):
    if image is None:
        return None, "Please upload an image.", ""
    p, face, eyes = analyse_frame(image)
    if p is None:
        return image, "No face found. Please upload a clear frontal photo.", ""
    drowsy = p >= 0.5
    shown = image.copy()
    color = (220, 40, 40) if drowsy else (40, 170, 80)
    x, y, w, h = face
    cv2.rectangle(shown, (x, y), (x + w, y + h), color, 3)
    for ex, ey, s, _ in eyes:
        cv2.rectangle(shown, (ex, ey), (ex + s, ey + s), (255, 200, 0), 2)
    status = (f"EYES CLOSED: DROWSY  (P(closed) = {p:.2f})" if drowsy
              else f"EYES OPEN: ALERT  (P(closed) = {p:.2f})")
    t, r = metrics["test"], metrics["real"]
    info = (f"Eye-state CNN, held-out test set: accuracy {t['accuracy']:.4f}, F1 {t['f1']:.4f}\n"
            f"Real-world eye photos: accuracy {r['accuracy']:.4f}, F1 {r['f1']:.4f}")
    return shown, status, info


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Driver image"),
    outputs=[gr.Image(label="Detected face and eyes"), gr.Textbox(label="Driver status"),
             gr.Textbox(label="Model performance")],
    title="Driver Drowsiness Detection using CNN",
    description="Upload a photo of the driver. The system finds the face and both eyes, and the CNN decides "
                "whether the eyes are open or closed.",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(server_port=7860)
