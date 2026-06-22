# 실행 방법

## 1. YOLO 기본 테스트

### yolo(GPU만).py

* GPU 환경에서 YOLO 학습 실행
* GPU가 없는 경우 실행 중 오류 발생 가능

### test_yolo(CPU가능).py

* CPU 환경에서도 실행 가능
* 실행 시 기본 모델 `yolo11n.pt` 자동 다운로드 및 사용

---

## 2. OpenCV 영상 테스트

실행 전 테스트 영상을 프로젝트 폴더에 추가해야 합니다.

예시:

```text
project/
└── tokyo.mp4
```

---

## 3. YOLO + OpenCV

### opencv_yolo.py

기능:

* OpenCV를 이용하여 영상 읽기
* YOLO를 이용하여 객체 탐지

탐지 클래스:

* Person
* Bicycle
* Car
* Motorcycle
* Bus
* Truck

실행:

```bash
python opencv_yolo.py
```

---

## 4. YOLO + Segmentation

### opencv_yolo_seg.py

기능추가:

* Segmentation 모델을 이용하여 Road / Sidewalk 영역 분리
* 사람의 위치가 차도(Road)인지 인도(Sidewalk)인지 판별

### opencv_yolo_seg1.py

```text
판단기준:
person + ROAD + CLOSE + 화면 중앙 → DANGER
person + ROAD + MID → WARNING
person + ROAD + FAR → SAFE
person + SIDEWALK → 대부분 SAFE
차량/자전거/오토바이 + CLOSE + 화면 중앙 → DANGER
```

실행:

```bash
python opencv_yolo_seg.py
```

---

## 생성 파일

YOLO 학습 완료 후:

```text
runs/
└── detect/
    └── train/
        └── weights/
            ├── best.pt
            └── last.pt
└── yolo11n.pt
```

* best.pt : 최고 성능 모델
* last.pt : 마지막 학습 모델
* yolo11n.pt로 일단 학습
