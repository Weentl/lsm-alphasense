# scripts/run_inference.py
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

# ── Configuración ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          'models', 'cnn_A_vs_other_hands_augmented.keras')

# Unit V2 vía USB-ethernet (si lo cambiaste por la webcam interna, pon 0)
# STREAM_URL = 'http://10.254.239.1:8080/?action=stream'
STREAM_URL = 0   # ← descomenta esta línea si quieres usar la webcam local

IMG_SIZE   = (224, 224)
THRESHOLD  = 0.70        # umbral mínimo de confianza para “A”

# ── Modelo y clases ───────────────────────────────────────────────────────────
print(f"🔄 Cargando modelo: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

data_labeled = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'data', 'labeled')
class_names = sorted([d for d in os.listdir(data_labeled)
                      if os.path.isdir(os.path.join(data_labeled, d))])
print("🗂️  Clases:", class_names)   # debería mostrar ['A', 'other']

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# ── Captura ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir la cámara/stream: {STREAM_URL}")

print("▶️  Iniciando inferencia.  (q para salir)")
while True:
    ok, frame = cap.read()
    if not ok:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        xs = [p.x for p in lm]; ys = [p.y for p in lm]
        h, w, _ = frame.shape
        x_min, x_max = int(min(xs)*w), int(max(xs)*w)
        y_min, y_max = int(min(ys)*h), int(max(ys)*h)

        pad = int(0.35 * max(x_max-x_min, y_max-y_min))
        x1, y1 = max(x_min-pad, 0), max(y_min-pad, 0)
        x2, y2 = min(x_max+pad, w), min(y_max+pad, h)

        crop = frame[y1:y2, x1:x2]
        if crop.size:
            crop_r = cv2.resize(crop, IMG_SIZE).astype('float32') / 255.0
            pred   = model.predict(crop_r[None, ...], verbose=0)[0]
            idx    = int(np.argmax(pred))
            conf   = float(pred[idx])

            # Decisión usando THRESHOLD
            if class_names[idx] == 'A' and conf >= THRESHOLD:
                label = f"A  ({conf*100:.1f}%)"
                color = (0, 255, 0)
            else:
                label = f"other ({conf*100:.1f}%)"
                color = (0,   0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('LSM AlphaSense – Inference', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

