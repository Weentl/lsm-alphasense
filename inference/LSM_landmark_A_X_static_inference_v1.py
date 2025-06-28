# scripts/run_inference_multiclass.py
import cv2, mediapipe as mp, numpy as np, tensorflow as tf
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'LSM-Landmark_A_X_static_v1.keras'
STREAM_URL = 0

# Carga modelo y clases
model       = tf.keras.models.load_model(MODEL_PATH)
classes_dir = ROOT / 'data' / 'landmarks_all'
class_names = sorted([d.name for d in classes_dir.iterdir() if d.is_dir()])

# MediaPipe
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
hands       = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

cap = cv2.VideoCapture(STREAM_URL)
while True:
    ret, frame = cap.read()
    if not ret: break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        feat = np.array([[p.x,p.y,p.z] for p in lm]).flatten()[None,:]
        preds = model.predict(feat, verbose=0)[0]
        idx   = int(preds.argmax()); conf = preds[idx]
        label = f"{class_names[idx]} ({conf*100:.1f}%)"

        # Dibuja landmarks
        mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        # Escribe etiqueta
        cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('LSM AlphaSense – Abecedario', frame)
    if cv2.waitKey(1)&0xFF==ord('q'): break

cap.release(); cv2.destroyAllWindows()
