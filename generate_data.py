import cv2
import numpy as np
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

NUM_IMAGES = 100
IMG_SIZE = 640
TRAIN_RATIO = 0.8
CLASS_ID = 0

BASE_DIR = Path("dataset")
IMG_DIR = BASE_DIR / "images"
LBL_DIR = BASE_DIR / "labels"

def add_starfield(img):
    for _ in range(random.randint(300, 600)):
        x = random.randint(0, IMG_SIZE - 1)
        y = random.randint(0, IMG_SIZE - 1)
        img[y, x] = 255
    return img

def draw_satellite(img):
    center = (
        random.randint(100, IMG_SIZE - 100),
        random.randint(100, IMG_SIZE - 100)
    )

    size = (
        random.randint(40, 120),
        random.randint(20, 60)
    )

    