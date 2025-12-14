import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ==========================================
# 1. 설정
# ==========================================
DATASET_ROOT = r'D:\학교\3-2\파이썬기반 딥러닝\3-2_jester_recognation\jester_final_dataset'
OUTPUT_DIR = r'D:\학교\3-2\파이썬기반 딥러닝\3-2_jester_recognation\jester_mediapipe_csv'

SPLITS = ['train', 'val']
MAX_WORKERS = 8  # 안전하게 코어 수 조절

mp_hands = mp.solutions.hands

FEATURE_COLS = []
for i in range(21):
    FEATURE_COLS.extend([f'joint_{i}_x', f'joint_{i}_y', f'joint_{i}_z', f'joint_{i}_v'])

# ==========================================
# ★ 핵심 수정: 한글 경로 이미지 읽기 함수
# ==========================================
def imread_korean(path):
    """
    OpenCV의 cv2.imread는 윈도우 한글 경로를 못 읽습니다.
    따라서 numpy로 파일을 바이너리로 읽은 후 디코딩해야 합니다.
    """
    try:
        stream = open(path.encode("utf-8"), "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        return cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        return None

# ==========================================
# 2. Worker 함수
# ==========================================
def process_video(args):
    video_path, label, video_id = args
    
    # 이미지 파일 찾기
    image_files = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
    
    # jpg가 없으면 JPG도 찾아봄 (혹시 모를 대소문자 이슈)
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(video_path, "*.JPG")))
    
    if not image_files:
        return None

    data_rows = []
    
    # MediaPipe 초기화
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        for frame_idx, img_file in enumerate(image_files):
            # [수정] 한글 경로 호환 읽기 함수 사용
            image = imread_korean(img_file)
            
            if image is None:
                # 이미지를 못 읽었으면 건너뜀 (여기서 문제 해결됨)
                continue

            # BGR -> RGB 변환
            # (이미지가 흑백이거나 채널이 다를 경우 예외처리)
            if len(image.shape) == 2: # 흑백인 경우
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4: # PNG 투명도 있는 경우
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = hands.process(image)

            row = {
                'video_id': video_id,
                'label_name': label,
                'frame_idx': frame_idx
            }

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                for i, lm in enumerate(hand_landmarks.landmark):
                    row[f'joint_{i}_x'] = lm.x
                    row[f'joint_{i}_y'] = lm.y
                    row[f'joint_{i}_z'] = lm.z
                    row[f'joint_{i}_v'] = 1.0 
            else:
                # 손을 못 찾으면 0.0으로 패딩
                for i in range(21):
                    row[f'joint_{i}_x'] = 0.0
                    row[f'joint_{i}_y'] = 0.0
                    row[f'joint_{i}_z'] = 0.0
                    row[f'joint_{i}_v'] = 0.0
            
            data_rows.append(row)

    return data_rows

# ==========================================
# 3. Main 함수
# ==========================================
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for split in SPLITS:
        split_dir = os.path.join(DATASET_ROOT, split)
        if not os.path.exists(split_dir):
            print(f"❌ 경로 없음: {split_dir}")
            continue

        print(f"Processing {split} set...")
        
        tasks = []
        labels = os.listdir(split_dir)
        
        for label in labels:
            label_dir = os.path.join(split_dir, label)
            if not os.path.isdir(label_dir):
                continue
                
            video_ids = os.listdir(label_dir)
            for vid in video_ids:
                video_path = os.path.join(label_dir, vid)
                if os.path.isdir(video_path):
                    tasks.append((video_path, label, vid))

        print(f"-> 총 {len(tasks)}개 비디오 처리 시작")

        results = []
        # 병렬 처리
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for res in tqdm(executor.map(process_video, tasks), total=len(tasks)):
                if res:
                    results.extend(res)
        
        # [방어 코드] 결과가 비어있으면 에러 대신 메시지 출력
        if not results:
            print(f"⚠️ {split} 데이터셋에서 추출된 데이터가 없습니다. 한글 경로 문제였을 가능성이 높습니다. (이번엔 해결되었길 바랍니다)")
            continue

        # DataFrame 저장
        print(f"Saving {split} CSV ({len(results)} rows)...")
        df = pd.DataFrame(results)
        
        cols = ['video_id', 'label_name', 'frame_idx'] + FEATURE_COLS
        df = df[cols]
        
        save_path = os.path.join(OUTPUT_DIR, f"{split}_data.csv")
        df.to_csv(save_path, index=False)
        print(f"✅ 저장 완료: {save_path}")

if __name__ == '__main__':
    main()     