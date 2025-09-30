# trainer.py
from ultralytics import YOLO
from config import DATA_YAML_PATH, IMAGE_SIZE

def train_yolov8_obb(epochs=50):
    model = YOLO('yolov8m-obb.pt')  
    model.train(data=DATA_YAML_PATH, imgsz=IMAGE_SIZE, epochs=epochs)
    print("✅ Training Complete.")

if __name__ == "__main__":
    train_yolov8_obb()