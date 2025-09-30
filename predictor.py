# predictor.py
import os
from ultralytics import YOLO
from config import MODEL_SAVE_PATH, IMAGE_SIZE

def predict_on_image(image_path):
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_SAVE_PATH}")
    
    model = YOLO(MODEL_SAVE_PATH)
    results = model.predict(source=image_path, imgsz=IMAGE_SIZE, conf=0.25)
    return results