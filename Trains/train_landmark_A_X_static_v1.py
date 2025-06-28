# scripts/train_landmark_multiclass.py
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT   = Path(__file__).resolve().parents[1]
LM_DIR = ROOT / 'data' / 'landmarks_all'
MODEL_OUT = ROOT / 'models' / 'landmark_multiclass_ABC...Z.keras'

# 1) Cargar vectores y etiquetas
classes = sorted([d.name for d in LM_DIR.iterdir() if d.is_dir()])
X, y = [], []
for idx, cls in enumerate(classes):
    for f in (LM_DIR/cls).glob('*.npy'):
        feat = np.load(f)
        X.append(feat)
        y.append(idx)
X = np.stack(X)   # shape (N, 63)
y = np.array(y)   # shape (N,)

print("Clases:", classes)
print("Total ejemplos:", X.shape[0])

# 2) Split y conversión de dtypes
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Forzar tipos adecuados
X_train = X_train.astype('float32')
X_val   = X_val.astype('float32')
y_train = y_train.astype('int32')
y_val   = y_val.astype('int32')

# 3) Crear tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_ds   = tf.data.Dataset.from_tensor_slices((X_val,   y_val))

# Batching y prefetch
BATCH = 32
train_ds = train_ds.shuffle(buffer_size=len(X_train)).batch(BATCH).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

# 4) Construcción del MLP multiclas
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5) Entrenamiento con EarlyStopping
es = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[es]
)

# 6) Guardar en formato .keras
model.save(MODEL_OUT, save_format='keras')
print("✅ Modelo multiclas guardado en", MODEL_OUT)
