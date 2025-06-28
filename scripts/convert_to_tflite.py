import tensorflow as tf
from pathlib import Path

root        = Path(__file__).resolve().parents[1]
model_keras = root / 'models' / 'LSM-Landmark_A_X_static_v1.keras'
model_tflite= root / 'models' / 'LSM-Landmark_A_X_static_v1.tflite'

# Carga y convierte
keras_model = tf.keras.models.load_model(model_keras)
converter   = tf.lite.TFLiteConverter.from_keras_model(keras_model)
# Usa optimización por defecto para reducir tamaño
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_buf  = converter.convert()

# Guarda el TFLite
model_tflite.write_bytes(tflite_buf)
print("✅ TFLite generado en", model_tflite)
