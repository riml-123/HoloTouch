import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import time
from collections import deque
import math

# ================= ⚙️ 설정 (Configuration) =================
# 모델 경로는 실제 환경에 맞게 수정하세요.
MODEL_PATH = r"code\HoloTouch_SE_TCN.onnx" 

INPUT_CHANNELS = 126
MAX_SEQ_LEN = 40
NUM_CLASSES = 8

TARGET_LABELS = [
    "No gesture", "Doing other things",
    "Zooming In With Two Fingers", "Zooming Out With Two Fingers",
    "Zooming In With Full Hand", "Zooming Out With Full Hand",
    "Thumb Up", "Thumb Down" 
]

CONFIDENCE_THRESHOLD = 0.8
ACTION_COOLDOWN = 0.5
Z_TOUCH_THRESHOLD = -0.05
MOVE_SCALE = 1.5 # 이동 스케일 증가 (민감도 상승)
# SMOOTHING_FACTOR 대신 OneEuro Filter 파라미터 사용

# ================= 📈 OneEuro Filter (신호 처리 고도화) =================
class OneEuroFilter:
    """ 
    속도에 따라 적응적으로 필터링을 조절하여 떨림 방지(Jitter)와 
    반응 지연(Lag)을 최소화하는 고급 필터 
    """
    def __init__(self, t0, x0, min_cutoff=0.01, beta=0.5, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = 0.0
        self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: 
            return self.x_prev

        # 1. 속도 추정 (1차 미분)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(self.smoothing_factor(t_e, self.d_cutoff), dx, self.dx_prev)

        # 2. cutoff 주파수 동적 조절
        # 속도가 빠르면 cutoff를 높여 필터링을 약하게 함 (반응성 증가)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # 3. 최종 필터링
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # 상태 업데이트
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

# ================= 🧮 유틸리티 & 전처리 =================

def normalize_skeleton(data):
    """ TCN 모델 입력에 맞는 손목 기준 상대 좌표 및 스케일 정규화 """
    T = data.shape[0]
    skeleton = data.reshape(T, 21, 3)
    wrist = skeleton[:, 0:1, :]
    skeleton = skeleton - wrist
    # 손목-중지 손가락 끝 (9번) 거리로 스케일 정규화
    dist = np.linalg.norm(skeleton[:, 9, :] - skeleton[:, 0, :], axis=1, keepdims=True) + 1e-6
    skeleton = skeleton / dist[:, :, np.newaxis]
    return skeleton.reshape(T, -1)

def preprocess_buffer(buffer):
    """ 버퍼 데이터를 전처리하여 TCN 모델 입력 형식으로 변환 """
    features = np.array(buffer, dtype=np.float32)
    features = normalize_skeleton(features)
    
    # 속도(Velocity) 특징 추가
    velocity = np.zeros_like(features)
    if features.shape[0] > 1:
        velocity[1:] = features[1:] - features[:-1]
        
    combined = np.concatenate([features, velocity], axis=1) # 63 * 2 = 126 채널
    
    # 패딩/트리밍으로 시퀀스 길이 맞추기
    seq_len = combined.shape[0]
    if seq_len < MAX_SEQ_LEN:
        pad_len = MAX_SEQ_LEN - seq_len
        padding = np.zeros((pad_len, INPUT_CHANNELS), dtype=np.float32)
        combined = np.vstack([combined, padding])
    else:
        combined = combined[-MAX_SEQ_LEN:, :]
        
    # TCN 입력 형식 (N, C, T)로 변환
    combined = combined.transpose(1, 0) 
    input_data = np.expand_dims(combined, axis=0).astype(np.float32)
    return input_data

def softmax(x):
    """ Softmax 함수 구현 """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def draw_dashboard(image, label, conf, fps):
    """ 화면 상단에 AI 추론 결과 및 FPS 대시보드 그리기 """
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (380, 90), (0, 0, 0), -1) 
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    is_confident = conf > CONFIDENCE_THRESHOLD
    color = (0, 255, 0) if is_confident else (180, 180, 180)
    display_label = label if is_confident else "Waiting..."
    
    cv2.putText(image, f"AI: {display_label}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    bar_width = 200
    filled_width = int(bar_width * conf)
    cv2.rectangle(image, (15, 50), (15 + bar_width, 65), (50, 50, 50), -1)
    cv2.rectangle(image, (15, 50), (15 + filled_width, 65), color, -1)
    cv2.rectangle(image, (15, 50), (15 + bar_width, 65), (200, 200, 200), 1)
    
    cv2.putText(image, f"{conf*100:.1f}%", (230, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, f"FPS: {int(fps)}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

# ================= 🎮 인터랙션 관리자 (InteractionManager) =================
class InteractionManager:
    def __init__(self, screen_w, screen_h):
        self.w, self.h = screen_w, screen_h
        
        # UI 상태
        self.ui_x, self.ui_y = screen_w // 2, screen_h // 2
        self.ui_size = 150
        self.ui_color = (255, 0, 0)
        self.is_selected = False
        
        # Grab 상태
        self.is_grabbing = False
        self.grab_anchor_wrist = None 
        self.grab_anchor_ui = None
        self.release_counter = 0      
        
        # 스와이핑 상태
        self.last_action_time = 0
        self.prev_hand_pos = None # 손 중심 (9번) 좌표

        # 좌표 필터링 (OneEuro Filter)
        now = time.time()
        # 포인터(검지) 필터: 민감도 높음 (beta=1.0)
        self.filter_idx_x = OneEuroFilter(now, self.ui_x, min_cutoff=0.01, beta=1.0)
        self.filter_idx_y = OneEuroFilter(now, self.ui_y, min_cutoff=0.01, beta=1.0)
        # 손목 필터: 안정성 중요 (beta=0.5)
        self.filter_wrist_x = OneEuroFilter(now, self.ui_x, min_cutoff=0.05, beta=0.5)
        self.filter_wrist_y = OneEuroFilter(now, self.ui_y, min_cutoff=0.05, beta=0.5)

    def _get_pinch_center(self, landmarks):
        """ 앵커 포인트 (엄지-검지 중점) """
        thumb = landmarks.landmark[4]
        index = landmarks.landmark[8]
        avg_x = (thumb.x + index.x) / 2 * self.w
        avg_y = (thumb.y + index.y) / 2 * self.h
        return (avg_x, avg_y)

    def is_point_in_ui(self, x, y):
        """ 좌표가 UI 박스 안에 있는지 확인 """
        half = self.ui_size // 2
        return (self.ui_x - half < x < self.ui_x + half) and \
               (self.ui_y - half < y < self.ui_y + half)

    def process(self, ai_label, ai_conf, landmarks, image):
        current_time = time.time()
        
        # 1. 랜드마크 추출 및 OneEuro Filter 적용
        idx_tip = landmarks.landmark[8]
        wrist = landmarks.landmark[0] 
        hand_center = landmarks.landmark[9]
        
        # raw 좌표
        raw_ix, raw_iy = int(idx_tip.x * self.w), int(idx_tip.y * self.h)
        raw_wx, raw_wy = int(wrist.x * self.w), int(wrist.y * self.h)
        cx, cy = int(hand_center.x * self.w), int(hand_center.y * self.h)
        
        # OneEuro Filter 적용 (포인터)
        ix = int(self.filter_idx_x(current_time, raw_ix))
        iy = int(self.filter_idx_y(current_time, raw_iy))
        
        # OneEuro Filter 적용 (손목 - Grab 이동용)
        wx = int(self.filter_wrist_x(current_time, raw_wx))
        wy = int(self.filter_wrist_y(current_time, raw_wy))
        
        # 판정용 앵커 (엄지-검지 중점)
        grab_x, grab_y = self._get_pinch_center(landmarks)
        
        # 포인터 그리기 (검지 끝 - 파란 점)
        cv2.circle(image, (ix, iy), 8, (255, 0, 0), -1) 

        # =========================================================
        # [우선순위 1] Grab & Move (손목 기준 안정화)
        # =========================================================
        
        # 1-1. Grab 시작: 'Full Hand Zoom In' 제스처, UI 위에서
        confident = ai_conf > CONFIDENCE_THRESHOLD

        # 1-1. Grab 시작: 'Full Hand Zoom In' + 검지가 UI 위
        if ai_label == "Zooming In With Full Hand" and confident:
            # === 핵심 수정: 판정 포인트를 ix, iy (검지 끝)로 변경 ===
            if not self.is_grabbing and self.is_point_in_ui(ix, iy): 
                self.is_grabbing = True
                self.grab_anchor_wrist = (wx, wy) 
                self.grab_anchor_ui = (self.ui_x, self.ui_y)
                self.release_counter = 0 
                # Grab 활성화 시 UI는 무조건 노란색 (Grab 중)
                self.ui_color = (0, 255, 255) 
                print("✊ Grab Started! (Anchored at Filtered Wrist)")
                
            # Grab 상태를 유지할 때, Release 카운터를 리셋 (Grab 제스처 유지 중)
            elif self.is_grabbing:
                 self.release_counter = 0

        # 1-2. Release 체크: 'Full Hand Zoom Out' (디바운스 적용)
        elif ai_label == "Zooming Out With Full Hand" and confident:
            if self.is_grabbing:
                self.release_counter += 1
                if self.release_counter > 3: # 연속 3프레임 이상 '놓기' 감지
                    self.is_grabbing = False
                    self.grab_anchor_wrist = None
                    # Release 후 UI 색상은 선택 상태에 따라 복원
                    self.ui_color = (0, 255, 0) if self.is_selected else (255, 0, 0)
                    self.release_counter = 0
                    print("🖐 Released (Confirmed)")
        
        else:
            # 다른 제스처이거나 제스처가 없을 때:
            # Grab 중일 경우, Release 제스처가 아니므로 카운터 리셋하고 Grab 유지
            if self.is_grabbing:
                self.release_counter = 0
                
        # 1-3. 이동 로직 (Grab 상태)
        if self.is_grabbing:
            # UI 색상은 Grab 시작 시 이미 (0, 255, 255)로 설정됨
            if self.grab_anchor_wrist:
                dx = wx - self.grab_anchor_wrist[0]
                dy = wy - self.grab_anchor_wrist[1]
                
                self.ui_x = int(self.grab_anchor_ui[0] + dx * MOVE_SCALE)
                self.ui_y = int(self.grab_anchor_ui[1] + dy * MOVE_SCALE)
                
                # 화면 이탈 방지
                self.ui_x = max(50, min(self.w-50, self.ui_x))
                self.ui_y = max(50, min(self.h-50, self.ui_y))
            
            self.prev_hand_pos = (cx, cy)
            return # Grab 중에는 하위 로직 차단 (선택/줌/스와이프 금지)

        # =========================================================
        # [우선순위 2] UI 선택/해제 (Z축 터치)
        # =========================================================
        # Z축 깊이 정보를 활용한 가상 터치 판정
        is_touching = idx_tip.z < Z_TOUCH_THRESHOLD
        
        if current_time - self.last_action_time > 1.0: # 디바운스 1초
            if is_touching:
                if self.is_point_in_ui(ix, iy):
                    if not self.is_selected:
                        self.is_selected = True
                        self.ui_color = (0, 255, 0)
                        self.last_action_time = current_time
                        print("👆 Selected (Z-Touch)")
                else:
                    if self.is_selected:
                        self.is_selected = False
                        self.ui_color = (255, 0, 0)
                        self.last_action_time = current_time
                        print("🚫 Deselected (Z-Touch)")
        
        # 선택된 상태에서 엄지 제스처 (확인/취소)
        if self.is_selected and ai_conf > CONFIDENCE_THRESHOLD:
            if ai_label == "Thumb Up":
                cv2.putText(image, "CONFIRMED", (self.ui_x - 60, self.ui_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            elif ai_label == "Thumb Down":
                self.is_selected = False
                self.ui_color = (255, 0, 0)
                self.last_action_time = current_time

        # =========================================================
        # [우선순위 3] 줌 & 스와이프 (선택 상태에서만)
        # =========================================================
        if self.is_selected:
            # 줌 제스처 (두 손가락)
            if ai_conf > CONFIDENCE_THRESHOLD:
                if ai_label == "Zooming In With Two Fingers":
                    self.ui_size = min(400, self.ui_size + 8)
                elif ai_label == "Zooming Out With Two Fingers":
                    self.ui_size = max(100, self.ui_size - 8)

            # 스와이프 (손 중심의 빠른 이동)
            if current_time - self.last_action_time > ACTION_COOLDOWN:
                if self.prev_hand_pos is not None:
                    # 손 중심 (9번)의 이동 벡터
                    vx = cx - self.prev_hand_pos[0]
                    vy = cy - self.prev_hand_pos[1]
                    SWIPE_THRESH = 40 # 임계값 상향 조정 (오작동 방지)
                    
                    if abs(vx) > SWIPE_THRESH or abs(vy) > SWIPE_THRESH:
                        # 더 큰 이동 축으로 스와이프 판정
                        if abs(vx) > abs(vy):
                            self.ui_x += int(vx * 1.5)
                        else:
                            self.ui_y += int(vy * 1.5)
                            
                        # 화면 이탈 방지
                        self.ui_x = max(50, min(self.w-50, self.ui_x))
                        self.ui_y = max(50, min(self.h-50, self.ui_y))
                            
                        self.last_action_time = current_time
                        print("💨 Swiped (Hand Center Movement)")

        # 다음 프레임을 위해 현재 손 중심 좌표 저장
        self.prev_hand_pos = (cx, cy)

    def draw(self, image):
        """ UI 박스 및 상태 표시 그리기 """
        half = self.ui_size // 2
        top_left = (self.ui_x - half, self.ui_y - half)
        bottom_right = (self.ui_x + half, self.ui_y + half)
        
        cv2.rectangle(image, top_left, bottom_right, self.ui_color, -1)
        cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 3)
        
        status = "Grabbing" if self.is_grabbing else ("Selected" if self.is_selected else "Idle")
        cv2.putText(image, status, (self.ui_x - 40, self.ui_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# ================= 🎥 메인 루프 =================
def run():
    print(f"Loading Model: {MODEL_PATH}...")
    try:
        # GPU 사용을 선호하는 경우 CUDAExecutionProvider를 먼저 시도
        session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        print("✅ Model Loaded!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # MediaPipe 초기화
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # 웹캠 초기화 (1280x720 해상도 설정)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    history_buffer = deque(maxlen=MAX_SEQ_LEN)
    manager = InteractionManager(1280, 720)
    prev_time = 0
    
    # MediaPipe Hands 모델 실행
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            
            # 좌우 반전 및 RGB 변환
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            landmarks_flat = None
            curr_label = "No gesture"
            curr_conf = 0.0
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # 랜드마크 추출 및 버퍼 추가
                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y, lm.z])
                    landmarks_flat = np.array(coords, dtype=np.float32)
                    history_buffer.append(landmarks_flat)
                    
                    # 최소 길이(15프레임) 도달 시 추론 시작
                    if len(history_buffer) >= 15:
                        input_data = preprocess_buffer(history_buffer)
                        outputs = session.run(None, {input_name: input_data})[0]
                        probs = softmax(outputs)
                        idx = np.argmax(probs)
                        curr_label = TARGET_LABELS[idx]
                        curr_conf = probs[0][idx]
                        
                        # 인터랙션 관리자 프로세스 실행
                        manager.process(curr_label, curr_conf, hand_landmarks, image)
                    break # 한 손만 처리
            else:
                # 손이 감지되지 않으면 빈 데이터 추가 (시퀀스 유지)
                if landmarks_flat is None: landmarks_flat = np.zeros(63, dtype=np.float32)
                history_buffer.append(landmarks_flat)
                manager.prev_hand_pos = None # 스와이프 방지

            # UI 및 대시보드 그리기
            manager.draw(image)
            
            # FPS 계산
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            
            draw_dashboard(image, curr_label, curr_conf, fps)

            # 화면 출력 및 종료 조건
            cv2.imshow('HoloTouch Final Improved (Industrial Grade)', image)
            if cv2.waitKey(1) & 0xFF == 27: break # ESC 키로 종료
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()