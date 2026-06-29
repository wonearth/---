# OpenCV + YOLO + SegFormer를 이용한 킥보드 시점 위험 탐지
# - YOLO: person, bicycle, car, motorcycle, bus, truck 탐지
# - SegFormer: road / sidewalk segmentation
# - 객체 위치, 거리, 진행 경로 여부를 바탕으로 SAFE / WARNING / DANGER 판단


# OpenCV + YOLO + Segmentation/ROI 기반 킥보드 위험 탐지
# 실시간 속도 우선 버전

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


# ==============================
# 기본 설정
# ==============================
VIDEO_PATH = "tokyo.mp4"

# 속도 우선: nano 모델 사용
YOLO_MODEL = "runs/detect/yolo11n_coco8/weights/best.pt"


TARGET_CLASSES = [0, 1, 2, 3, 5, 7]                      # 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck

# Cityscapes class id
ROAD_ID = 0
SIDEWALK_ID = 1

# 속도/성능 조절값
YOLO_CONF = 0.4
YOLO_IMGSZ = 320                        # CPU: 속도 개선
SEG_INTERVAL = 90                       # segmentation 더 드물게 실행
SEG_WIDTH = 320                         # segmentation 입력 크기 축소

PATH_RATIO = 0.30                           # 화면 중앙 30%를 킥보드 진행 경로로 판단

# 실시간 우선이면 True
USE_SEGMENTATION = True


# ── 핀홀 카메라 거리 추정 설정 ──
# 라즈베리파이 Camera Module v2 기준 (1280×720)  ->  v3 사용 시 ~1650으로 변경
FOCAL_LENGTH_PX = 1058

# 객체별 실제 높이 (미터)
KNOWN_HEIGHTS_M = {
    "person":     1.7,
    "car":        1.5,
    "bicycle":    1.0,
    "motorcycle": 1.1,
    "bus":        3.0,
    "truck":      2.5,
}

# 거리 기준 (미터)
DIST_CLOSE_M = 5.0
DIST_MID_M   = 10.0

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_device = 0 if torch.cuda.is_available() else "cpu"

print("=" * 50)
print("사용 장치:", device)
print("YOLO 모델:", YOLO_MODEL)
print("YOLO 이미지 크기:", YOLO_IMGSZ)
print("Segmentation interval:", SEG_INTERVAL)
print("=" * 50)


# ==============================
# 모델 로딩
# ==============================
print("YOLO 모델 로딩 중...")
yolo = YOLO(YOLO_MODEL)

processor = None
seg_model = None

if USE_SEGMENTATION:
    print("Segmentation 모델 로딩 중...")
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    )

    seg_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    ).to(device)

    seg_model.eval()


# ==============================
# Segmentation 함수
# ==============================

# 속도 개선용 segmentation 함수
def get_road_sidewalk_mask_fast(frame):
    original_h, original_w = frame.shape[:2]

    scale = SEG_WIDTH / original_w
    seg_h = int(original_h * scale)

    small_frame = cv2.resize(frame, (SEG_WIDTH, seg_h))

    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = seg_model(**inputs)

    logits = outputs.logits

    upsampled_logits = F.interpolate(
        logits,
        size=(seg_h, SEG_WIDTH),
        mode="bilinear",
        align_corners=False
    )

    pred_small = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    road_mask_small = (pred_small == ROAD_ID).astype("uint8")
    sidewalk_mask_small = (pred_small == SIDEWALK_ID).astype("uint8")

    road_mask = cv2.resize(
        road_mask_small,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    sidewalk_mask = cv2.resize(
        sidewalk_mask_small,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    return road_mask, sidewalk_mask


# ==============================
# 위치 판단 함수
# ==============================

# 사람의 발 위치 주변 영역을 보고 ROAD / SIDEWALK 판단
def get_location_from_mask(x1, y1, x2, y2, road_mask, sidewalk_mask, frame_shape):
    h, w = frame_shape[:2]

    foot_x = int((x1 + x2) / 2)
    foot_y = int(y2)

    foot_x = max(0, min(foot_x, w - 1))
    foot_y = max(0, min(foot_y, h - 1))

    r = 8

    x_min = max(0, foot_x - r)
    x_max = min(w, foot_x + r)
    y_min = max(0, foot_y - r)
    y_max = min(h, foot_y + r)

    road_count = road_mask[y_min:y_max, x_min:x_max].sum()
    sidewalk_count = sidewalk_mask[y_min:y_max, x_min:x_max].sum()

    if road_count > sidewalk_count and road_count > 5:
        return "ROAD", foot_x, foot_y
    elif sidewalk_count > road_count and sidewalk_count > 5:
        return "SIDEWALK", foot_x, foot_y
    else:
        return "UNKNOWN", foot_x, foot_y


# 화면 아래쪽 중앙은 ROAD, 양옆 하단은 SIDEWALK로 대략 판단
def get_location_from_roi(x1, y1, x2, y2, frame_shape):
    h, w = frame_shape[:2]

    foot_x = int((x1 + x2) / 2)
    foot_y = int(y2)

    # 화면 아래 45%만 도로/인도 판단에 사용
    if foot_y < h * 0.55:
        return "UNKNOWN", foot_x, foot_y

    # 중앙 영역은 차도/주행 영역으로 가정
    road_left = int(w * 0.25)
    road_right = int(w * 0.75)

    if road_left <= foot_x <= road_right:
        return "ROAD", foot_x, foot_y
    else:
        return "SIDEWALK", foot_x, foot_y


# ==============================
# 위험 판단 함수
# ==============================

# 핀홀 카메라 모델로 실제 거리(미터) 추정
def estimate_distance_m(class_name, y1, y2):
    bbox_h = y2 - y1
    if bbox_h <= 0:
        return 99.0
    real_h = KNOWN_HEIGHTS_M.get(class_name, 1.5)
    return (real_h * FOCAL_LENGTH_PX) / bbox_h


def get_distance_status(dist_m):
    if dist_m <= DIST_CLOSE_M:
        return "CLOSE"
    elif dist_m <= DIST_MID_M:
        return "MID"
    else:
        return "FAR"

# 객체 박스가 킥보드 진행 경로와 겹치는지 판단
def is_in_kickboard_path(x1, x2, frame_width):
    frame_center_x = frame_width // 2
    path_width = frame_width * PATH_RATIO

    path_left = int(frame_center_x - path_width / 2)
    path_right = int(frame_center_x + path_width / 2)

    return x2 >= path_left and x1 <= path_right

# 위험도 판단
def judge_risk(class_name, location, distance_status, in_path):
    if class_name == "person":
        if location == "ROAD":
            if distance_status == "CLOSE" and in_path:
                return "DANGER"
            elif distance_status == "MID" and in_path:
                return "WARNING"
            elif distance_status == "CLOSE":
                return "WARNING"
            else:
                return "SAFE"

        elif location == "SIDEWALK":
            if distance_status == "CLOSE" and in_path:
                return "WARNING"
            else:
                return "SAFE"

        else:
            if distance_status == "CLOSE" and in_path:
                return "WARNING"

    elif class_name in ["car", "bus", "truck", "motorcycle", "bicycle"]:
        if distance_status == "CLOSE" and in_path:
            return "DANGER"
        elif distance_status == "MID" and in_path:
            return "WARNING"
        elif distance_status == "CLOSE":
            return "WARNING"

    return "SAFE"


def get_color_by_risk(risk):
    if risk == "DANGER":
        return (0, 0, 255)
    elif risk == "WARNING":
        return (0, 255, 255)
    else:
        return (0, 255, 0)


# ==============================
# 시각화 함수
# ==============================

def draw_kickboard_path(frame):
    h, w = frame.shape[:2]

    center_x = w // 2
    path_width = int(w * PATH_RATIO)

    path_left = int(center_x - path_width / 2)
    path_right = int(center_x + path_width / 2)

    cv2.rectangle(frame, (path_left, 0), (path_right, h), (255, 255, 255), 2)

    cv2.putText(
        frame,
        "KICKBOARD PATH",
        (path_left + 10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return frame


def draw_warning_banner(frame, max_risk, danger_reason):
    h, w = frame.shape[:2]

    if max_risk == "DANGER":
        color = (0, 0, 255)
        text = "DANGER"
        text_color = (255, 255, 255)
    elif max_risk == "WARNING":
        color = (0, 255, 255)
        text = "WARNING"
        text_color = (0, 0, 0)
    else:
        return frame

    cv2.rectangle(frame, (0, 0), (w, 90), color, -1)
    cv2.putText(
        frame,
        text,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        text_color,
        3
    )

    cv2.putText(
        frame,
        danger_reason,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        text_color,
        2
    )

    return frame


# ==============================
# 영상 처리 시작
# ==============================

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

    max_risk = "SAFE"
    danger_reason = "No immediate risk"

    # segmentation은 매우 느리므로 일정 프레임마다만 실행
    if USE_SEGMENTATION and frame_count % SEG_INTERVAL == 1:
        road_mask, sidewalk_mask = get_road_sidewalk_mask_fast(frame)

    # YOLO 객체 탐지
    results = yolo.predict(
        frame,
        conf=YOLO_CONF,
        classes=TARGET_CLASSES,
        imgsz=YOLO_IMGSZ,
        device=yolo_device,
        verbose=False
    )

    boxes = results[0].boxes

    # segmentation overlay는 연하게 표시
    if road_mask is not None and sidewalk_mask is not None:
        overlay = frame.copy()
        overlay[road_mask] = (0, 0, 255)
        overlay[sidewalk_mask] = (0, 255, 0)
        frame = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)

    frame = draw_kickboard_path(frame)

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = yolo.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        location = "UNKNOWN"

        # 사람만 ROAD/SIDEWALK 판단
        if class_name == "person":
            if road_mask is not None and sidewalk_mask is not None:
                location, foot_x, foot_y = get_location_from_mask(
                    x1, y1, x2, y2,
                    road_mask,
                    sidewalk_mask,
                    frame.shape
                )
            else:
                location, foot_x, foot_y = get_location_from_roi(
                    x1, y1, x2, y2,
                    frame.shape
                )

            cv2.circle(frame, (foot_x, foot_y), 5, (255, 255, 255), -1)

        dist_m = estimate_distance_m(class_name, y1, y2)
        distance_status = get_distance_status(dist_m)
        in_path = is_in_kickboard_path(x1, x2, frame.shape[1])
        risk = judge_risk(class_name, location, distance_status, in_path)

        if risk == "DANGER":
            max_risk = "DANGER"
            if class_name == "person" and location == "ROAD":
                danger_reason = "Pedestrian detected on road"
            else:
                danger_reason = f"Close {class_name} detected"

        elif risk == "WARNING" and max_risk != "DANGER":
            max_risk = "WARNING"
            if class_name == "person":
                danger_reason = "Pedestrian nearby"
            else:
                danger_reason = f"{class_name} nearby"

        color = get_color_by_risk(risk)

        label = f"{class_name} {dist_m:.1f}m"
        if risk != "SAFE":
            label += f" {risk}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    frame = draw_warning_banner(frame, max_risk, danger_reason)

    # 출력 크기 제한
    display_frame = cv2.resize(frame, (960, 540))
    cv2.imshow("Kickboard Real-Time Risk Detection", display_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
