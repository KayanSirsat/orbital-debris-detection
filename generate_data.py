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

    angle = random.uniform(0, 180)

    rect = (center, size, angle)
    box = cv2.boxPoints(rect).astype(int)

    cv2.drawContours(img, [box], 0, 255, -1)

    return box

def box_to_yolo(box):
    x_coords = box[:,0]
    y_coords = box[:,1]

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    x_center = ((x_min + x_max) / 2) / IMG_SIZE
    y_center = ((y_min + y_max) / 2) / IMG_SIZE
    width = (x_max - x_min) / IMG_SIZE
    height = (y_max - y_min) / IMG_SIZE
    return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

indices = list(range(NUM_IMAGES))
train_idx, val_idx = train_test_split(indices, train_size=TRAIN_RATIO, random_state = 42)

def generate(split, idxs):
    
    for i in idxs:
        img = np.zeros((IMG_SIZE, IMG_SIZE), dtype = np.uint8)
        img = add_starfield(img)
        box = draw_satellite(img)
        
        img_path = IMG_DIR / split / f"img_{i}.jpg"
        lbl_path = LBL_DIR / split / f"img_{i}.txt"

        cv2.imwrite(str(img_path), img)
        with open(lbl_path, "w") as f:
            f.write(box_to_yolo(box))

generate("train", train_idx)
generate("val", val_idx)

print("Synthetic dataset generated succesfully.")