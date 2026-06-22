# OpenCV와 YOLO 모델, Segmentation 모델을 사용하여 동영상에서 객체 탐지 및 도로/인도 구분

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image


# =========================
# 설정
# =========================
VIDEO_PATH = "tokyo.mp4"   # 네 영상 파일명으로 수정
YOLO_MODEL = "yolo11n.pt"

# YOLO COCO 클래스
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]
# 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck

# Cityscapes 라벨
ROAD_ID = 0
SIDEWALK_ID = 1

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 모델 로드
# =========================
print("YOLO 모델 로딩 중...")
yolo = YOLO(YOLO_MODEL)

print("Segmentation 모델 로딩 중...")
processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
)
seg_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
).to(device)

seg_model.eval()


# =========================
# Segmentation 함수
# =========================
def get_road_sidewalk_mask(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    inputs = processor(images=pil_img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = seg_model(**inputs)

    logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=frame.shape[:2],
        mode="bilinear",
        align_corners=False
    )

    pred = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    road_mask = pred == ROAD_ID
    sidewalk_mask = pred == SIDEWALK_ID

    return road_mask, sidewalk_mask


# =========================
# 영상 실행
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("영상 파일을 열 수 없습니다:", VIDEO_PATH)
    exit()

frame_count = 0
road_mask = None
sidewalk_mask = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # 너무 느리면 10프레임마다 segmentation
    if frame_count % 10 == 1:
        road_mask, sidewalk_mask = get_road_sidewalk_mask(frame)

    # YOLO 객체 탐지
    results = yolo.predict(
        frame,
        conf=0.4,
        classes=TARGET_CLASSES,
        verbose=False
    )

    boxes = results[0].boxes

    # road 영역 시각화
    if road_mask is not None:
        overlay = frame.copy()
        overlay[road_mask] = (0, 0, 255)        # road 빨강
        overlay[sidewalk_mask] = (0, 255, 0)    # sidewalk 초록
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # YOLO 결과 처리
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        class_name = yolo.names[cls_id]

        # 사람일 때만 road/sidewalk 판단
        location = ""

        if class_name == "person" and road_mask is not None:
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)

            foot_x = max(0, min(foot_x, frame.shape[1] - 1))
            foot_y = max(0, min(foot_y, frame.shape[0] - 1))

            if road_mask[foot_y, foot_x]:
                location = "ROAD"
            elif sidewalk_mask[foot_y, foot_x]:
                location = "SIDEWALK"
            else:
                location = "UNKNOWN"

            cv2.circle(frame, (foot_x, foot_y), 5, (255, 255, 255), -1)

        label = f"{class_name} {conf:.2f}"
        if location:
            label += f" | {location}"

        # 박스 그리기
        color = (255, 0, 0)

        if location == "ROAD":
            color = (0, 0, 255)
        elif location == "SIDEWALK":
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("YOLO + Road/Sidewalk Segmentation", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()