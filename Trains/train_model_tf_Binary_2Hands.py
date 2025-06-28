# scripts/train_model_tf.py
"""
Entrenamiento mejorado para el clasificador binario A vs other,
ahora con datos de mano derecha e izquierda.
Incluye:
 • Data augmentation
 • Class weights
 • EarlyStopping + ReduceLROnPlateau
 • Dropout
 • Fine-tuning de últimas 40 capas
"""

import tensorflow as tf
from pathlib import Path
import numpy as np
from collections import Counter

# ── Configuración ────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
DATA_DIR     = ROOT / 'data' / 'labeled'
IMG_SIZE     = (224, 224)
BATCH        = 16
EPOCHS_HEAD  = 12
EPOCHS_FINE  = 8
UNFREEZE_LAY = 40     # últimas 40 capas a descongelar
MODEL_NAME   = 'cnn_A_vs_other_hands_augmented.keras'

# ── Carga de datos ────────────────────────────────────────────────────────────
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH)

class_names = train_ds.class_names  # ['A','other']
print("Clases:", class_names)

# ── Calcular class weights ────────────────────────────────────────────────────
y_train = np.concatenate([y.numpy() for _, y in train_ds])
counts  = Counter(y_train)
total   = sum(counts.values())
class_weight = {i: total/(len(counts)*counts[i]) for i in counts}
print("Class weights:", class_weight)

# ── Data augmentation ─────────────────────────────────────────────────────────
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.1),
])

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (train_ds
            .map(lambda x, y: (augment(x, training=True), y))
            .map(lambda x, y: ((x / 127.5) - 1.0, y))
            .prefetch(AUTOTUNE))
val_ds = (val_ds
          .map(lambda x, y: ((x / 127.5) - 1.0, y))
          .prefetch(AUTOTUNE))

# ── Construcción del modelo ───────────────────────────────────────────────────
base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet')
base.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = base(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)                 # help regularize
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ── Callbacks ────────────────────────────────────────────────────────────────
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=4, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# ── Entrenamiento cabeza (solo la parte superior) ─────────────────────────────
model.fit(train_ds,
          epochs=EPOCHS_HEAD,
          validation_data=val_ds,
          class_weight=class_weight,
          callbacks=[early_stop, reduce_lr])

# ── Fine-tuning de últimas capas ──────────────────────────────────────────────
base.trainable = True
# congelar todas menos las últimas UNFREEZE_LAY capas
for layer in base.layers[:-UNFREEZE_LAY]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          epochs=EPOCHS_FINE,
          validation_data=val_ds,
          class_weight=class_weight,
          callbacks=[early_stop, reduce_lr])

# ── Guardar modelo ────────────────────────────────────────────────────────────
out_path = ROOT / 'models' / MODEL_NAME
model.save(out_path, save_format='keras')
print(f"✅ Modelo entrenado y guardado en {out_path}")


