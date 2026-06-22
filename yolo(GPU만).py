# GPU로 학습

from ultralytics import YOLO
import torch

def main():
    # 1. 내 게이밍 노트북 GPU가 잘 작동하는지 터미널에 출력
    print("\n" + "="*40)
    print("★ AI 학습에 사용할 GPU:", torch.cuda.get_device_name(0))
    print("="*40 + "\n")

    # 2. 전 세계 사물을 이미 공부한 기본 YOLOv11 나노 모델 데려오기
    model = YOLO("yolo11n.pt")

    # 3. 내 컴퓨터 그래픽카드로 진짜 보행자 데이터셋 학습(train) 시작!
    model.train(
        data="C:\pedestrian detection using yolo.v1i.yolov11",    # ◀ 본인이 압축을 푼 실제 data.yaml 경로로 적어주세요!
        epochs=30,                        # 우선 가볍게 30번만 기출문제 풀이 돌리기
        imgsz=640,                        # 이미지 해상도 표준 설정
        device=0                          # 내 노트북 외장 그래픽카드(GPU) 강제 지정
    )

if __name__ == '__main__':
    main()