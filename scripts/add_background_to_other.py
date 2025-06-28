# scripts/add_background_to_other.py
"""
Genera datos 'other' para el modelo:
  • Recorta cualquier mano (gestos que NO son letra A).
  • Extrae también un recorte de fondo por imagen.
Los recortes se guardan en: data/labeled/other/
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import random

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(__file__))
RAW_DIR   = os.path.join(ROOT, 'data', 'raw_frames', 'other')      # imágenes sin procesar
SAVE_DIR  = os.path.join(ROOT, 'data', 'labeled', 'other')         # destino recortes
os.makedirs(SAVE_DIR, exist_ok=True)

# ── MediaPipe Hands ────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands_det = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5)

# ── Utilidades ────────────────────────────────────────────────────────────────
def clamp(val, lo, hi):
    return max(lo, min(val, hi))

def crop_square(img, x1, y1, x2, y2, scale=1.2, out_size=224):
    """Recorta un cuadro centrado en la caja [x1,y1,x2,y2] y lo redimensiona."""
    h, w = img.shape[:2]
    cx, cy = (x1+x2)/2, (y1+y2)/2
    side = max(x2-x1, y2-y1) * scale
    x1n = int(clamp(cx - side/2, 0, w))
    y1n = int(clamp(cy - side/2, 0, h))
    x2n = int(clamp(cx + side/2, 0, w))
    y2n = int(clamp(cy + side/2, 0, h))
    crop = img[y1n:y2n, x1n:x2n]
    return cv2.resize(crop, (out_size, out_size))

def random_bg_crop(img, hand_boxes, out_size=224, max_tries=20):
    """Devuelve un recorte aleatorio que NO solape manos (>30 % solape)."""
    h, w = img.shape[:2]
    side = min(h, w) // 2
    for _ in range(max_tries):
        x1 = random.randint(0, w - side)
        y1 = random.randint(0, h - side)
        x2, y2 = x1 + side, y1 + side
        box_area = side * side
        too_close = False
        for (hx1, hy1, hx2, hy2) in hand_boxes:
            # IoU aproximado
            ix1, iy1 = max(x1, hx1), max(y1, hy1)
            ix2, iy2 = min(x2, hx2), min(y2, hy2)
            iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
            inter = iw * ih
            if inter / box_area > 0.3:
                too_close = True
                break
        if not too_close:
            crop = img[y1:y2, x1:x2]
            return cv2.resize(crop, (out_size, out_size))
    # Si no encuentra zona limpia, usa recorte central
    mid = min(h, w)
    return cv2.resize(img[(h-mid)//2:(h+mid)//2, (w-mid)//2:(w+mid)//2], (out_size, out_size))

# ── Índices para nombres únicos ───────────────────────────────────────────────
existing = [f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg')]
hand_idx = len([f for f in existing if f.startswith('other_hand')])
bg_idx   = len([f for f in existing if f.startswith('other_bg')])

# ── Procesado principal ───────────────────────────────────────────────────────
for fname in sorted(os.listdir(RAW_DIR)):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img_path = os.path.join(RAW_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands_det.process(img_rgb)

    hand_boxes = []
    if res.multi_hand_landmarks:
        # Guardar cada mano detectada
        for hand_lm in res.multi_hand_landmarks:
            xs = [lm.x for lm in hand_lm.landmark]
            ys = [lm.y for lm in hand_lm.landmark]
            x1 = int(min(xs) * w); x2 = int(max(xs) * w)
            y1 = int(min(ys) * h); y2 = int(max(ys) * h)
            hand_boxes.append((x1,y1,x2,y2))

            crop = crop_square(img, x1, y1, x2, y2)
            out_name = f"other_hand_{hand_idx:04d}.jpg"
            cv2.imwrite(os.path.join(SAVE_DIR, out_name), crop)
            hand_idx += 1

    # Recorte de fondo
    bg_crop = random_bg_crop(img, hand_boxes)
    out_bg = f"other_bg_{bg_idx:04d}.jpg"
    cv2.imwrite(os.path.join(SAVE_DIR, out_bg), bg_crop)
    bg_idx += 1

print(f"✅ Proceso finalizado.\n   Manos guardadas: {hand_idx}\n   Fondos guardados: {bg_idx}")

