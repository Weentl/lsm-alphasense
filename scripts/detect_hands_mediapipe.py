# scripts/detect_hands_mediapipe.py
# Detecta la mano con MediaPipe, recorta la caja y guarda en data/labeled/<letra>/
import cv2, mediapipe as mp, os, sys

# La letra que vamos a procesar, p.ej. python detect_hands_mediapipe.py B
LETTER = sys.argv[1].upper() if len(sys.argv) > 1 else 'A'

# Rutas base
root = os.path.dirname(os.path.dirname(__file__))
raw_dir   = os.path.join(root, 'data', 'raw_frames', LETTER)  # ahora por letra
save_dir  = os.path.join(root, 'data', 'labeled', LETTER)
os.makedirs(save_dir, exist_ok=True)

# Inicializamos MediaPipe Hands
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def crop_bbox(img, bbox, scale=1.2):
    h, w, _ = img.shape
    x, y, w_box, h_box = bbox
    cx, cy = x + w_box/2, y + h_box/2
    side = int(max(w_box, h_box) * scale)
    x1 = int(max(cx - side/2, 0)); y1 = int(max(cy - side/2, 0))
    x2 = int(min(cx + side/2, w)); y2 = int(min(cy + side/2, h))
    return img[y1:y2, x1:x2]

# Procesamos cada imagen de raw_frames/<LETTER>/
count = 0
for fname in sorted(os.listdir(raw_dir)):

    path = os.path.join(raw_dir, fname)
    img  = cv2.imread(path)
    if img is None:
        continue

    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        continue

    # Calculamos bounding box a partir de los landmarks
    lm = results.multi_hand_landmarks[0].landmark
    xs = [p.x for p in lm]; ys = [p.y for p in lm]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bbox = (
        x_min * img.shape[1],
        y_min * img.shape[0],
        (x_max - x_min) * img.shape[1],
        (y_max - y_min) * img.shape[0]
    )

    crop = crop_bbox(img, bbox)
    if crop.size == 0:
        continue

    out_name = f"{LETTER}_{count:04d}.jpg"
    cv2.imwrite(os.path.join(save_dir, out_name), crop)
    count += 1

print(f"✅ Guardados {count} recortes de letra '{LETTER}' en: {save_dir}")

