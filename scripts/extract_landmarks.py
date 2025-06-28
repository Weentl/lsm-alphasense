# scripts/extract_landmarks.py
import cv2, mediapipe as mp, os, numpy as np
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw_frames' / 'other'      # o 'other'
OUT_DIR = ROOT / 'data' / 'landmarks' / 'other'      # o 'other'
OUT_DIR.mkdir(parents=True, exist_ok=True)

mp_hands = mp.solutions.hands
hands     = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

for img_path in sorted(RAW_DIR.glob('*.jpg')):
    img = cv2.imread(str(img_path))
    if img is None: continue

    res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        continue

    lm = res.multi_hand_landmarks[0].landmark
    # Aplana (x,y,z) → [x1,y1,z1, x2,y2,z2, …]
    feat = np.array([[p.x, p.y, p.z] for p in lm]).flatten()
    np.save(OUT_DIR / f"{img_path.stem}.npy", feat)
