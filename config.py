# config.py
DATA_YAML_PATH = "dataset/data.yaml"
MODEL_SAVE_PATH = "runs/obb/train17/weights/best.pt"
IMAGE_SIZE = 640

import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()