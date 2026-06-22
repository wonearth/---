# OpenCV와 YOLO 모델을 사용하여 동영상에서 객체 탐지

import cv2
from ultralytics import YOLO

# YOLO 모델 불러오기
model = YOLO("yolo11n.pt")

# 영상 파일 이름
video_path = "tokyo.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("영상 파일을 열 수 없습니다. 파일 이름/경로를 확인하세요.")
    exit()

# 탐지할 클래스: person, bicycle, car, motorcycle, bus, truck
target_classes = [0, 1, 2, 3, 5, 7]

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        frame,
        conf=0.4,
        classes=target_classes,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()