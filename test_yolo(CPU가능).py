# CPU로 학습 (GPU가 없는 경우)

from ultralytics import YOLO
import torch


def main():
    print("=" * 40)

    # GPU 사용 가능 여부 확인
    if torch.cuda.is_available():
        print("★ AI 학습에 사용할 GPU:", torch.cuda.get_device_name(0))
        device = 0
    else:
        print("★ CUDA 사용 불가: CPU로 학습합니다.")
        device = "cpu"

    print("=" * 40)

    # YOLO 모델 불러오기
    model = YOLO("yolo11n.pt")

    # 학습
    model.train(
        data="coco8.yaml",            # Ultralytics가 제공하는 아주 작은 예제 데이터셋
        epochs=10,              # 데이터셋을 몇 번 반복해서 학습(50,100)
        imgsz=640,
        batch=4,                # 한 번에 몇 장씩 학습
        device=device,
        workers=0,
        name="yolo11n_coco8"
    )


if __name__ == "__main__":
    main()