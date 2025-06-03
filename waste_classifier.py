"""
Proyecto: Clasificación de Residuos con Transfer Learning (VGG16)
Descripción: Clasifica imágenes de residuos en orgánico y reciclable utilizando un modelo VGG16 preentrenado.
"""

# =======================
#  IMPORTACIÓN DE LIBRERÍAS
# =======================
import os
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
#  CONFIGURACIÓN INICIAL
# =======================
# Rutas a las carpetas con imágenes
BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Tamaño de entrada (VGG16 espera 224x224)
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
# Cargar modelo base VGG16 sin las capas superiores
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
base_model.trainable = False  # congelar capas convolucionales

# Añadir una cabeza personalizada para clasificación binaria o multiclase
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')  # salida multiclase
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
predicted_class = class_names[np.argmax(pred)]

plt.imshow(img)
plt.title(f"Predicción: {predicted_class}")
plt.axis('off')
plt.show()
