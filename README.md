#  Waste Classification with Transfer Learning (VGG16)

Este proyecto aplica **transfer learning** para clasificar imágenes de residuos como **orgánicos** o **reciclables**. Se utiliza el modelo preentrenado **VGG16** sobre ImageNet como extractor de características, y se entrena una cabeza personalizada para ajustarse al nuevo conjunto de datos.

---

##  Objetivos del Proyecto

- Aplicar transferencia de aprendizaje para clasificación de imágenes.
- Preprocesar imágenes usando `ImageDataGenerator`.
- Entrenar un modelo basado en VGG16 con nuevas capas densas.
- Evaluar el modelo y visualizar predicciones sobre imágenes reales.
- Dejar una estructura preparada para adaptarse a otros proyectos reales de clasificación de imágenes.

---

##  ¿Cómo funciona este script?

El script `waste_classifier.py` sigue una estructura clara y modular que puede adaptarse fácilmente a otros conjuntos de datos. Aquí explicamos cada bloque:

### 1️⃣ Importación de librerías
Se importan librerías estándar de Machine Learning (TensorFlow, Keras, NumPy, Matplotlib) para construir y entrenar la red neuronal.

### 2️⃣ Configuración inicial
Se definen las rutas locales a los datos (`train/`, `val/`, `test/`) y parámetros como tamaño de imagen, batch size y número de épocas.

### 3️⃣ Generadores de imágenes
Usamos `ImageDataGenerator` para cargar las imágenes en tiempo real desde disco, aplicar **rescalado y aumentación** y alimentar el modelo con batches.

### 4️⃣ Modelo VGG16
Se carga el modelo `VGG16` sin su cabeza final (`include_top=False`) y se congela para evitar que se actualicen sus pesos.

### 5️⃣ Capas personalizadas
Se añade un `Flatten`, una capa `Dense(256)` con `ReLU`, un `Dropout`, y una capa final `Dense` con softmax, adaptada al número de clases del dataset.

### 6️⃣ Entrenamiento
El modelo se entrena con `categorical_crossentropy`, `adam` como optimizador y callbacks (`EarlyStopping`, `ReduceLROnPlateau`) para detener el entrenamiento si no mejora.

### 7️⃣ Evaluación
Se evalúa el modelo sobre el conjunto de test y se imprime la precisión.

### 8️⃣ Predicción visual
Se selecciona aleatoriamente una imagen de test y se muestra junto con la clase predicha.

---

##  Cómo usar este proyecto

1. **Organiza tu dataset** así:

```
dataset/
├── train/
│   ├── reciclable/
│   └── organico/
├── val/
│   ├── reciclable/
│   └── organico/
└── test/
    ├── reciclable/
    └── organico/
```

2. **Instala las dependencias**:

```bash
pip install -r requirements.txt
```

3. **Ejecuta el script**:

```bash
python waste_classifier.py
```

---

##  Cómo adaptarlo a un proyecto real

Este flujo de trabajo es fácilmente adaptable a otras tareas de clasificación de imágenes. Para ello:

- Cambia las carpetas `train/`, `val/` y `test/` por otras categorías (por ejemplo: `metal`, `plástico`, `papel`, `vidrio`, etc.).
- Puedes usar otro modelo preentrenado como `InceptionV3`, `ResNet50`, etc., cambiando `VGG16` por otro en la línea:

```python
from tensorflow.keras.applications import InceptionV3
base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
```

- Si tienes imágenes de mayor tamaño o diferentes proporciones, ajusta `IMG_SIZE` y considera usar `GlobalAveragePooling2D` en lugar de `Flatten()`.

- Para aplicaciones industriales o municipales, puedes:
  - Integrar la inferencia en una API (Flask, FastAPI).
  - Conectar el modelo con una cámara que tome imágenes en tiempo real.
  - Exportar el modelo con `model.save("modelo_final.h5")` y cargarlo en un sistema embebido o dispositivo edge.

---

