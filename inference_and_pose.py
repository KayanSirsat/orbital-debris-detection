import cv2
import numpy as np
from ultralytics import YOLO

def add_guassian_noise(img, mean=0, std=15):
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def denoise(img):
    return cv2.meadianBlur(img, 5)

def estimate_orientation(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    
    cnt = max(contours, key = cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(int)
    angle = rect[-1]
    if angle < -45:
        angle += 90
    return angle, box

model = YOLO("runs/detect/orbital_detector/weights/best.pt")

