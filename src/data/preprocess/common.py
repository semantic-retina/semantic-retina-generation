from pathlib import Path

import cv2
import numpy as np

BLACK = 0
WHITE = 255


GRAY_CLASS = {
    "RETINA": 0,
    "OD": 1,
    "MA": 2,
    "HE": 3,
    "EX": 4,
    "SE": 5,
    "NV": 6,
    "IRMA": 7,
}


def open_binary_mask(path: Path) -> np.ndarray:
    try:
        return open_image(path, cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        return np.zeros((2848, 4288))


def open_colour_image(path: Path) -> np.ndarray:
    return open_image(path, cv2.IMREAD_COLOR)


# Note: mask if modified in-place.
def overlay_label(mask: np.ndarray, img: np.ndarray, label: np.ndarray):
    img_idx = np.where(img > 0)
    mask[img_idx] = label[img_idx]


def write_image(img: np.ndarray, path: Path):
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise IOError(f"Failed to write image to {path}")


def open_image(path: Path, flags: int) -> np.ndarray:
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"Could not open {path}")
    return img


# Contours are drawn on the original image.
def fill_contours(image, contours, color):
    for i in range(0, len(contours)):
        cv2.drawContours(image, contours, i, color, cv2.FILLED)


def find_eye(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # A threshold value of 10 appears to work well.
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours[0]
