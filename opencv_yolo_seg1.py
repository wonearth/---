import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


VIDEO_PATH = "tokyo.mp4"
YOLO_MODEL = "yolo11n.pt"

TARGET_CLASSES = [0, 1, 2, 3, 5, 7]
# 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck

ROAD_ID = 0
SIDEWALK_ID = 1

device = "cuda" if torch.cuda.is_available() else "cpu"


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


def get_road_sidewalk_mask(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    inputs = processor(images=image, return_tensors="pt").to(device)

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


def estimate_distance_status(y1, y2):
    box_height = y2 - y1

    if box_height > 350:
        return "CLOSE"
    elif box_height > 220:
        return "MID"
    else:
        return "FAR"


def is_in_kickboard_path(x1, x2, frame_width):
    object_center_x = (x1 + x2) // 2
    frame_center_x = frame_width // 2

    path_width = frame_width * 0.30  # 화면 중앙 30%

    return abs(object_center_x - frame_center_x) < path_width / 2

def judge_risk(class_name, location, distance_status, in_path):
    if class_name == "person":
        if location == "ROAD":
            if distance_status == "CLOSE" and in_path:
                return "DANGER"
            elif distance_status in ["CLOSE", "MID"]:
                return "WARNING"
            else:
                return "SAFE"

        elif location == "SIDEWALK":
            if distance_status == "CLOSE" and in_path:
                return "WARNING"
            else:
                return "SAFE"

    elif class_name in ["car", "bus", "truck", "motorcycle", "bicycle"]:
        if distance_status == "CLOSE" and in_path:
            return "DANGER"
        elif distance_status in ["CLOSE", "MID"]:
            return "WARNING"

    return "SAFE"


def get_color_by_risk(risk):
    if risk == "DANGER":
        return (0, 0, 255)
    elif risk == "WARNING":
        return (0, 255, 255)
    else:
        return (0, 255, 0)


def draw_warning_banner(frame, max_risk, danger_reason):
    h, w = frame.shape[:2]

    if max_risk == "DANGER":
        color = (0, 0, 255)
        text = "DANGER"
        sub_text = danger_reason

    elif max_risk == "WARNING":
        color = (0, 255, 255)
        text = "WARNING"
        sub_text = danger_reason

    else:
        return frame

    cv2.rectangle(frame, (0, 0), (w, 90), color, -1)

    text_color = (255, 255, 255) if max_risk == "DANGER" else (0, 0, 0)

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
        sub_text,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        text_color,
        2
    )

    return frame


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

    if frame_count % 10 == 1:
        road_mask, sidewalk_mask = get_road_sidewalk_mask(frame)

    results = yolo.predict(
        frame,
        conf=0.4,
        classes=TARGET_CLASSES,
        verbose=False
    )

    boxes = results[0].boxes

    if road_mask is not None:
        overlay = frame.copy()
        overlay[road_mask] = (0, 0, 255)
        overlay[sidewalk_mask] = (0, 255, 0)
        frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = yolo.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        location = "UNKNOWN"

        if class_name == "person" and road_mask is not None:
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)

            foot_x = max(0, min(foot_x, frame.shape[1] - 1))
            foot_y = max(0, min(foot_y, frame.shape[0] - 1))

            if road_mask[foot_y, foot_x]:
                location = "ROAD"
            elif sidewalk_mask[foot_y, foot_x]:
                location = "SIDEWALK"

            cv2.circle(frame, (foot_x, foot_y), 5, (255, 255, 255), -1)

        distance_status = estimate_distance_status(y1, y2)
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

        label = f"{class_name} {conf:.2f} | {distance_status}"

        if class_name == "person":
            label += f" | {location}"

        if in_path:
            label += " | IN PATH"
        
        label += f" | {risk}"

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

    cv2.imshow("Kickboard Risk Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()