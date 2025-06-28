# scripts/extract_landmarks_all.py
import cv2, mediapipe as mp, numpy as np, os
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
RAW_DIR   = ROOT / 'data' / 'raw_frames'
LM_DIR    = ROOT / 'data' / 'landmarks_all'
# Limpia / crea carpeta de salida
if LM_DIR.exists():
    for f in LM_DIR.glob('*/*.npy'): f.unlink()
else:
    LM_DIR.mkdir(parents=True)

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Para cada letra (subcarpeta) en raw_frames
for letter_dir in sorted(RAW_DIR.iterdir()):
    if not letter_dir.is_dir(): continue
    letter = letter_dir.name
    out_dir = LM_DIR / letter
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(letter_dir.glob('*.[jp][pn]g')):
        img = cv2.imread(str(img_path))
        if img is None: continue

        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            continue

        lm = res.multi_hand_landmarks[0].landmark
        feat = np.array([[p.x, p.y, p.z] for p in lm]).flatten()
        np.save(out_dir / f"{img_path.stem}.npy", feat)

    print(f"✅ Extraídos {len(list(out_dir.glob('*.npy')))} landmarks para '{letter}'")
