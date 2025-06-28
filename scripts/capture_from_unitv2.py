# scripts/capture_from_android_by_letter.py

import cv2
import os
import sys

# 1. Verificar el argumento de la línea de comandos (la letra)
if len(sys.argv) < 2:
    print("Uso: python capture_from_android_by_letter.py <LETRA>")
    sys.exit(1)

LETTER = sys.argv[1].upper()

# 2. Definir rutas de guardado
project_root = os.path.dirname(os.path.dirname(__file__))
# Cambiamos 'raw_frames' a 'raw_frames_android' para diferenciar los datos
save_dir = os.path.join(project_root, 'data', 'raw_frames', LETTER)
os.makedirs(save_dir, exist_ok=True)
print(f"📁 Las imágenes de la letra '{LETTER}' se intentarán guardar en: {save_dir}")

# 3. Conexión a la cámara del teléfono Android (como webcam USB)
# **IMPORTANTE**:
# - Asegúrate de que tu teléfono Android está configurado como webcam USB (ej. con DroidCam).
# - El índice '0' es la primera cámara detectada. Si tienes varias, prueba con 1, 2, etc.
# - El 'stream_url' original se mantiene como comentario solo para referencia, ya no se usa.
cap = cv2.VideoCapture(0) # ¡Aquí está el cambio clave!

# 4. Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara del teléfono.")
    print("💡 Posibles soluciones:")
    print("   - Asegúrate de que tu teléfono esté conectado por USB y la aplicación de webcam esté activa.")
    print("   - Prueba diferentes índices numéricos en cv2.VideoCapture() (ej. 1, 2, 3...) si tienes otras cámaras.")
    print("   - Reinicia la aplicación de webcam en el teléfono o el teléfono mismo.")
    sys.exit(1)

print(f"🖼️ Capturando letra '{LETTER}'. Presiona 'c' para foto, 'q' para salir.")
idx = len(os.listdir(save_dir))

# 5. Bucle principal de captura y visualización
while True:
    ret, frame = cap.read()
    if not ret or frame is None: # Añadimos 'frame is None' para mayor robustez
        print("⚠️ Frame no disponible o error de lectura. Intentando reconectar…")
        cap.release()
        # Intentar reconectar con el mismo índice
        cap = cv2.VideoCapture(0) # Vuelve a usar el índice numérico
        if not cap.isOpened():
            print("❌ No se pudo reconectar al stream. Saliendo.")
            break
        continue

    # Mostrar el frame en una ventana
    cv2.imshow(f'Captura para la letra {LETTER}', frame) # Nombre de la ventana más descriptivo
    
    # Capturar la pulsación de tecla
    key = cv2.waitKey(1) & 0xFF

    # 6. Lógica para guardar la imagen al presionar 'c'
    if key == ord('c'):
        print(f"✅ 'c' presionado. Estado del frame: ret={ret}, frame is None={frame is None}") # Mensaje de depuración
        if frame is not None: # Asegurarse de que el frame no es None antes de guardar
            fname = f"{LETTER}_{idx:04d}.jpg"
            full_path = os.path.join(save_dir, fname)
            try:
                cv2.imwrite(full_path, frame)
                print(f"✅ Guardado {fname}")
                idx += 1
            except Exception as e:
                print(f"❌ ERROR al guardar la imagen {fname}: {e}")
                print(f"Verifica permisos de escritura en {save_dir}")
        else:
            print("⚠️ No se pudo guardar: el frame es None en este momento.")

    # 7. Lógica para salir al presionar 'q'
    elif key == ord('q'):
        print("👋 Saliendo de la aplicación.")
        break

# 8. Liberar recursos
cap.release()
cv2.destroyAllWindows()
