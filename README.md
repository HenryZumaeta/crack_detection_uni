# Detección y Clasificación de Grietas en Pavimentos

Este repositorio contiene código Python para detectar y clasificar grietas en pavimentos usando Deep Learning, enfocado en el mantenimiento preventivo en Perú. Utiliza SegFormer (nvidia/mit-b5) para segmentación semántica y esqueletización con análisis geométrico para clasificar grietas (longitudinales, transversales, ramificadas).

**Datos**: Dataset CrackIPN (400 imágenes) + 50 imágenes locales de La Molina, Lima (julio-agosto 2025).

## Requisitos
- Python 3.8+
- Librerías: `torch`, `transformers`, `albumentations`, `scikit-image`, `opencv-python`, `numpy`, `matplotlib`, `pillow`, `sklearn`, `scipy`

Instalación:
```
pip install torch transformers albumentations scikit-image opencv-python numpy matplotlib pillow scikit-learn scipy
```

## Uso

### Segmentación (`carck_segmentation.py`)
1. Preparar dataset CrackIPN en `data/CrackIPN/images` y `masks`.
2. Ejecutar:
   ```
   python carck_segmentation.py
   ```
   Entrena el modelo (150 épocas) y evalúa el F1-score.

### Clasificación (`carck_classification.py`)
1. Cargar modelo SegFormer entrenado.
2. Actualizar `image_path` con la imagen de prueba.
3. Ejecutar:
   ```
   python carck_classification.py
   ```
   Clasifica la grieta (e.g., "Transversal").

## Contexto
Parte de la tesis: "Detección de Grietas en Pavimentos para Mejorar el Mantenimiento Preventivo en Perú usando Deep Learning".

## Licencia
MIT License.
