# Jester Gesture Recognition

이 프로젝트는 Jester 데이터셋을 기반으로 한 제스처 인식 시스템입니다. MediaPipe를 사용하여 손의 특징을 추출하고, Squeeze-and-Excitation Temporal Convolutional Network (SE-TCN) 모델을 사용하여 제스처를 분류합니다. PySide6 기반의 GUI 애플리케이션을 통해 실시간 제스처 인식을 제공합니다.

## 특징

- **실시간 제스처 인식**: 카메라를 통해 실시간으로 제스처를 인식합니다.
- **SE-TCN 모델**: 시간적 컨볼루션 네트워크를 기반으로 한 고성능 모델.
- **MediaPipe 통합**: 손의 랜드마크를 정확하게 추출.
- **GUI 애플리케이션**: PySide6을 사용한 사용자 친화적인 인터페이스.
- **ONNX 지원**: 모델을 ONNX 형식으로 변환하여 다양한 플랫폼에서 실행 가능.

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
   - [Jester 데이터셋](https://20bn.com/datasets/jester)에서 다운로드하여 `jester_dataset_v1/` 폴더에 배치합니다.

## 사용법

### 1. 데이터 전처리

MediaPipe를 사용하여 특징을 추출합니다:

```bash
python code/extract_mediapipe_features.py
```

### 2. 모델 훈련

Jupyter 노트북을 사용하여 모델을 훈련합니다:

```bash
jupyter notebook code/train.ipynb
```

### 3. 추론

훈련된 모델로 제스처를 인식합니다:

```bash
python code/inference.py
```

### 4. GUI 애플리케이션 실행

실시간 제스처 인식 애플리케이션을 실행합니다:

```bash
python app/main_app.py
```

## 프로젝트 구조

```
├── app/                    # GUI 애플리케이션
│   ├── main_app.py         # 메인 애플리케이션
│   ├── worker.py           # 추론 워커
│   └── HoloTouch_SE_TCN.onnx  # ONNX 모델
├── code/                   # 모델 및 추론 코드
│   ├── model.py            # SE-TCN 모델 정의
│   ├── train.ipynb         # 훈련 노트북
│   ├── inference.py        # 추론 스크립트
│   └── extract_mediapipe_features.py  # 특징 추출
├── jester_dataset_v1/      # Jester 데이터셋
├── jester_mediapipe_csv/   # 전처리된 데이터
└── README.md               # 이 파일
```

## 모델 아키텍처

SE-TCN 모델은 다음과 같은 구성 요소를 포함합니다:

- Temporal Convolutional Blocks with Dilations
- Squeeze-and-Excitation (SE) Blocks
- Residual Connections
- Dropout for Regularization

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.

## 기여

기여를 환영합니다! 이슈를 열거나 풀 리퀘스트를 제출해 주세요.

## 참고

- Jester 데이터셋: 20BN에서 제공
- MediaPipe: Google의 손 추적 라이브러리
- PyTorch: 딥러닝 프레임워크</content>
<parameter name="filePath">d:\학교\3-2\파이썬기반 딥러닝\3-2_jester_recognation\README.md