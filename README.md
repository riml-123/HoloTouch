# HoloTouch - Gesture Recognition & AR UI System

제스처 인식 모델을 통해 AR 환경 UI를 직관적으로 제어하는 인터페이스 시스템.

Jester 데이터셋을 학습한 경량화된 SE-TCN 모델을 사용하여, 저사양 환경에서도 평균 2ms의 지연속도로 사용자의 의도를 파악하고 상호작용합니다.

## 특징
- **실시간 제스처 인식**: 카메라를 통해 실시간으로 제스처를 인식
- **SE-TCN 모델 사용**: 시간적 컨볼루션 네트워크를 기반으로 한 고성능 모델
- **GUI 애플리케이션**: PySide6을 사용한 사용자 친화적인 인터페이스
- **ONNX 지원**: 모델을 ONNX 형식으로 변환하여 다양한 플랫폼에서 실행 가능

## 지원하는 제스처
- No gesture
- Doing other things
- Zooming In With Two Fingers
- Zooming Out With Two Fingers
- Zooming In With Full Hand
- Zooming Out With Full Hand
- Thumb Up
- Thumb Down

## 요구사항
- Python 3.8 이상
- PyTorch
- OpenCV
- MediaPipe
- PySide6
- ONNX Runtime
- NumPy
- Pandas
- tqdm

## 설치

1. 저장소를 클론합니다:
   ```
   git clone https://github.com/your-username/jester-gesture-recognition.git
   cd jester-gesture-recognition
   ```

2. 의존성을 설치합니다:
   ```
   pip install torch torchvision opencv-python mediapipe PySide6 onnxruntime numpy pandas tqdm
   ```

3. 데이터셋을 다운로드합니다 (Jester v1):
   - [Jester 데이터셋](https://www.kaggle.com/datasets/toxicmender/20bn-jester)에서 다운로드하여 `jester_dataset_v1/` 폴더에 배치합니다.

## 사용법
```
├── app/                    # GUI 애플리케이션(별도 다운로드)
│   ├── main_app.py         # 메인 애플리케이션
│   ├── worker.py           # 추론 워커
│   └── HoloTouch_SE_TCN.onnx  # ONNX 모델
├── code/                   # 모델 및 추론 코드
│   ├── model.py            # SE-TCN 모델 정의
│   ├── train.ipynb         # 훈련 노트북
│   ├── inference.py        # 추론 스크립트
│   └── extract_mediapipe_features.py  # 특징 추출
├── jester_dataset_v1/      # Jester 데이터셋(별도 생성)
├── jester_mediapipe_csv/   # 전처리된 데이터(별도 다운로드)
└── README.md              
```

**app과 jester_mediapipe_csv 폴더는 용량 이슈로 인해 [이 드라이브 링크](https://drive.google.com/drive/folders/162QUNoFs08auC40fTd_GvwbtwobN8ePO?usp=sharing)에서 확인 가능합니다**

## 모델 아키텍처

SE-TCN 모델은 다음과 같은 구성 요소를 포함합니다:

- Temporal Convolutional Blocks with Dilations
- Squeeze-and-Excitation (SE) Blocks
- Residual Connections
- Dropout for Regularization


## 참고

- Jester 데이터셋: 20BN에서 제공
- MediaPipe: Google의 손 추적 라이브러리
- PyTorch: 딥러닝 프레임워크</content>
<parameter name="filePath">d:\학교\3-2\파이썬기반 딥러닝\3-2_jester_recognation\README.md
