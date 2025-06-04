import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =======================
# CONFIGURACIÓN
# =======================
IMG_SIZE = (224, 224)
TEST_DIR = "dataset/test"
MODEL_PATH = "waste_model_vgg16.h5"

# Mapeo de etiquetas legibles
label_map = {"R": "Recycled", "O": "Organic"}

# =======================
# CARGA DEL MODELO
# =======================
model = load_model(MODEL_PATH)

# =======================
# RECORRIDO DEL DATASET
# =======================
all_images = []
class_names = []

for label_folder in sorted(os.listdir(TEST_DIR)):
    full_path = os.path.join(TEST_DIR, label_folder)
    if os.path.isdir(full_path):
        for fname in os.listdir(full_path):
            all_images.append((os.path.join(full_path, fname), label_folder))
            if label_folder not in class_names:
                class_names.append(label_folder)

# Mostrar total
total = len(all_images)
print(f"Total de imágenes en test: {total}")

# =======================
# SELECCIÓN DE ÍNDICE MANUAL
# =======================
while True:
    try:
        idx = int(input(f"Introduce un número entre 0 y {total - 1}: "))
        if 0 <= idx < total:
            break
        else:
            print("Número fuera de rango. Intenta otra vez.")
    except ValueError:
        print("Entrada no válida. Introduce un número entero.")

img_path, true_label = all_images[idx]
true_label_readable = label_map.get(true_label, true_label)

print(f"\n Imagen seleccionada: {img_path}")
print(f" Etiqueta real: {true_label_readable}")

# =======================
# PREPROCESAMIENTO
# =======================
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# =======================
# PREDICCIÓN
# =======================
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
predicted_label = label_map.get(predicted_class, predicted_class)

# =======================
# MOSTRAR RESULTADOS
# =======================
plt.imshow(img)
plt.title(f"Predicción: {predicted_label}\nEtiqueta real: {true_label_readable}")
plt.axis('off')
plt.show()
