# scripts/run_inference_landmarks_with_drawing.py
import cv2, mediapipe as mp, numpy as np, tensorflow as tf
from pathlib import Path

# ── Configuración ───────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'landmark_A_vs_other.keras'
STREAM_URL = 0  # o 0 para webcam local

# ── Carga del modelo ───────────────────────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_PATH)

# ── MediaPipe Hands y Drawing ──────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_style   = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# ── Captura de vídeo ───────────────────────────────────────────────────────────
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir el stream")

print("▶️  Inferencia con landmarks + visualización. (q para salir)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # 1) Dibuja los landmarks y conexiones sobre el frame
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_style.get_default_hand_landmarks_style(),
            mp_style.get_default_hand_connections_style())

        # 2) Prepara el vector de características para clasificar
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()[None, :]

        # 3) Inferencia
        pred = model.predict(pts, verbose=0)[0]
        cls  = 'A' if pred[0] > pred[1] else 'other'
        conf = pred.max()

        # 4) Muestra la etiqueta
        color = (0,255,0) if cls=='A' else (0,0,255)
        text  = f"{cls} ({conf*100:.1f}%)"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Landmark Inference + Drawing', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
