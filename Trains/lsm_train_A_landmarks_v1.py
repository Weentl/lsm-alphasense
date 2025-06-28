# scripts/train_landmark_classifier.py
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT       = Path(__file__).resolve().parents[1]
LM_DIR     = ROOT / 'data' / 'landmarks'
classes    = ['A', 'other']
X, y = [], []

for idx, cls in enumerate(classes):
    for npy in (LM_DIR / cls).glob('*.npy'):
        feat = np.load(npy)
        X.append(feat)
        y.append(idx)

X = np.vstack(X)
y = np.array(y)

# Dividir
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Modelo MLP
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar
model.fit(X_train, y_train, epochs=20, batch_size=16,
          validation_data=(X_val, y_val),
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

# Guardar
model.save(ROOT / 'models' / 'landmark_A_vs_other.keras')
print("✅ Entrenado landmark-based classifier")
