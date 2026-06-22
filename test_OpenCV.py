# OpenCV로 동영상 재생

import cv2

cap = cv2.VideoCapture("tokyo.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Tokyo", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()