"""Real-time drowsiness monitor: webcam -> face -> eyes -> CNN. Alarm when the eyes stay closed for 15 frames."""
import cv2
from pipeline import analyse_frame

ALERT_FRAMES = 15          # about half a second at 30 fps
cap = cv2.VideoCapture(0)
closed_run = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    p, face, eyes = analyse_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if p is not None:
        closed_run = closed_run + 1 if p >= 0.5 else 0
        color = (0, 0, 255) if p >= 0.5 else (0, 200, 0)
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        for ex, ey, s, _ in eyes:
            cv2.rectangle(frame, (ex, ey), (ex + s, ey + s), (0, 200, 255), 1)
        cv2.putText(frame, f"P(closed) = {p:.2f}  closed frames = {closed_run}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if closed_run >= ALERT_FRAMES:
        cv2.putText(frame, "DROWSINESS ALERT!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        print("\a", end="", flush=True)      # terminal bell as the alarm
    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
