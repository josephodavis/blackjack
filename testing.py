import cv2
from ultralytics import YOLO

# trained model
model = YOLO("best.pt")

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # prediction
    results = model(frame)

    # display prediction
    annotated_frame = results[0].plot()

    cv2.imshow("cards", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()