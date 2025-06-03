"""
Proyecto: Clasificación de Residuos con Transfer Learning (VGG16)
Este script carga automáticamente el dataset desde un archivo ZIP ubicado en la raíz del proyecto.
"""

# =======================
#  IMPORTACIÓN DE LIBRERÍAS
# =======================
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image

# =======================
#  EXTRACCIÓN DEL DATASET
# =======================
ZIP_FILENAME = "o-vs-r-split-reduced-1200.zip"
EXTRACTED_FOLDER = "dataset"

if not os.path.exists(EXTRACTED_FOLDER):
    print(f"📦 Extrayendo {ZIP_FILENAME}...")
    with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
        zip_ref.extractall(EXTRACTED_FOLDER)
    print("✅ Extracción completada.")
else:
    print("✅ Dataset ya extraído.")

# =======================
# CONFIGURACIÓN INICIAL
# =======================
# Definir rutas dentro de la carpeta extraída
TRAIN_DIR = os.path.join(EXTRACTED_FOLDER, "train")
VAL_DIR = os.path.join(EXTRACTED_FOLDER, "val")
TEST_DIR = os.path.join(EXTRACTED_FOLDER, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# =======================
#  GENERADORES DE IMÁGENES
# =======================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
).flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

# =======================
#  CONSTRUCCIÓN DEL MODELO
# =======================
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

# =======================
#  COMPILACIÓN Y ENTRENAMIENTO
# =======================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=2, verbose=1)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =======================
# EVALUACIÓN DEL MODELO
# =======================
loss, acc = model.evaluate(test_gen)
print(f" Precisión en test: {acc:.2f}")

# =======================
#  PREDICCIÓN SOBRE IMAGEN INDIVIDUAL
# =======================
import random

class_names = list(test_gen.class_indices.keys())
folder = random.choice(class_names)
path = os.path.join(TEST_DIR, folder)
img_name = random.choice(os.listdir(path))
img_path = os.path.join(path, img_name)

img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

plt.imshow(img)
plt.title(f"Predicción: {predicted_class}")
plt.axis('off')
plt.show()
