import os
import sys

import cv2
import joblib
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Adjust this import to wherever hog_extractor.py lives in your project
from src.features.hog_extractor import HOGExtractor

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "svm_ocr.joblib"))

# ASSUMPTION: 62 classes ordered 0-9, A-Z, a-z. Verify against your CSV's labels.
LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

# Must match the settings used when the SVM was trained
HOG = HOGExtractor(orientations=9, pixels_per_cell=(3, 3), cells_per_block=(2, 2), block_norm="L2-Hys")


def load_model(path):
    return joblib.load(path)


def load_grayscale(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img


def detect_edges(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # Dilate so broken strokes join into connected contours
    return cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)


def find_character_box(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in image")
    # Union of all contours, so letters like 'i' and 'j' keep their dots
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 10]
    if not boxes:
        raise ValueError("No contours large enough to be a character")
    x1 = min(x for x, y, w, h in boxes)
    y1 = min(y for x, y, w, h in boxes)
    x2 = max(x + w for x, y, w, h in boxes)
    y2 = max(y + h for x, y, w, h in boxes)
    return x1, y1, x2, y2


def crop_character(gray, box):
    x1, y1, x2, y2 = box
    return gray[y1:y2, x1:x2]


def pad_to_square(char_img, margin=4):
    h, w = char_img.shape
    size = max(h, w) + 2 * margin
    bg = int(np.median(char_img))  # background shade estimate
    canvas = np.full((size, size), bg, dtype=np.uint8)
    y0, x0 = (size - h) // 2, (size - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = char_img
    return canvas


def to_feature_vector(square_img, invert=False):
    # HOGExtractor expects 28x28 images with values in [0, 1]
    resized = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)
    if invert:
        resized = 255 - resized
    img = resized.astype(np.float32) / 255.0
    return HOG.extract(img[np.newaxis])  # (1, 28, 28) -> (1, D)


def predict_character(model, features):
    idx = int(model.predict(features)[0])
    return LABELS[idx]


def main():
    image_path = input("Path to letter image: ").strip()
    invert = input("Invert colors? (y/n, use 'y' if your training data is light-on-dark): ").strip().lower() == "y"

    model = load_model(MODEL_PATH)
    gray = load_grayscale(image_path)
    edges = detect_edges(gray)
    box = find_character_box(edges)
    char_img = crop_character(gray, box)
    square = pad_to_square(char_img)
    features = to_feature_vector(square, invert=invert)

    print(f"Prediction: {predict_character(model, features)}")


if __name__ == "__main__":
    main()