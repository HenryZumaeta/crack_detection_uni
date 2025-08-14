import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Configuración
TRAIN_SIZE = (480, 320)
BASE_BUFFER_MARGIN = 70

# Predicción con redimensionamiento
def predict_mask(image_path, model, feature_extractor):
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size
    inputs = feature_extractor(images=image, return_tensors="pt", size=TRAIN_SIZE[::-1]).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = torch.nn.functional.interpolate(
        outputs.logits, size=orig_size[::-1], mode="bilinear", align_corners=False
    )
    pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    return pred_mask, np.array(image), orig_size

# Clasificación por morfología y geometría
def classify_crack(pred_mask, orig_size):
    scale = min(TRAIN_SIZE[0]/orig_size[0], TRAIN_SIZE[1]/orig_size[1])
    new_size = (int(orig_size[0]*scale), int(orig_size[1]*scale))
    mask = cv2.resize(pred_mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST)
    
    # Esqueletización
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(mask*255, cv2.MORPH_CLOSE, kernel) / 255
    skeleton = skeletonize(cleaned).astype(np.uint8)
    labeled = label(skeleton)
    regions = [r for r in regionprops(labeled) if r.area >= 15]
    
    if not regions:
        return "No detectada"
    
    skel_clean = np.zeros_like(skeleton)
    for r in regions:
        skel_clean[labeled == r.label] = 1
    
    y_coords, x_coords = np.where(skel_clean)
    if len(x_coords) == 0:
        return "No clara"
    
    # Buffers
    margin = int(BASE_BUFFER_MARGIN * (TRAIN_SIZE[0]/480))
    h_buffer = {
        'y1': int((y_coords.min() + y_coords.max()) / 2) - margin,
        'y2': int((y_coords.min() + y_coords.max()) / 2) + margin
    }
    v_buffer = {
        'x1': int((x_coords.min() + x_coords.max()) / 2) - margin,
        'x2': int((x_coords.min() + x_coords.max()) / 2) + margin
    }
    
    in_h = np.all((y_coords >= h_buffer['y1']) & (y_coords <= h_buffer['y2']))
    in_v = np.all((x_coords >= v_buffer['x1']) & (x_coords <= v_buffer['x2']))
    
    if in_h:
        return "Transversal"
    elif in_v:
        return "Longitudinal"
    
    # Geometría
    neighbors = convolve2d(skel_clean, np.array([[1,1,1],[1,0,1],[1,1,1]]), mode='same')
    bifurcations = np.sum(neighbors >= 3)
    
    if bifurcations >= 2:
        return "Ramificada"
    
    angles = []
    for y, x in zip(y_coords, x_coords):
        if neighbors[y, x] == 2:
            window = skel_clean[y-1:y+2, x-1:x+2]
            if window.sum() >= 3:
                dy, dx = np.gradient(window.astype(float))
                angle = np.arctan2(dy.mean(), dx.mean()) * 180 / np.pi % 180
                angles.append(angle)
    
    if not angles:
        return "Ramificada"
    
    angles = np.array(angles)
    is_vert = np.all((90-30 <= angles) & (angles <= 90+30))
    is_horiz = np.all((angles <= 30) | (angles >= 150))
    
    return "Longitudinal" if is_vert else "Transversal" if is_horiz else "Ramificada"

# Ejecución
image_path = "data/local/test_image.jpg"  # Ruta genérica
pred_mask, image, size = predict_mask(image_path, model, feature_extractor)
classification = classify_crack(pred_mask, size)

print(f"Clasificación: {classification}")
