from pathlib import Path

import cv2
import numpy as np

BLACK = 0
WHITE = 255


GRAY_CLASS = {
    "BG": 0,
    "OD": 1,
    "MA": 2,
    "HE": 3,
    "EX": 4,
    "SE": 5,
}


def open_binary_mask(path: Path) -> np.ndarray:
    try:
        return open_image(path, cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        print(f"Could not open {path}, continuing...")
        return np.zeros((2848, 4288))


def open_colour_image(path: Path) -> np.ndarray:
    try:
        return open_image(path, cv2.IMREAD_COLOR)
    except FileNotFoundError:
        print(f"Could not open {path}, continuing...")
        return np.zeros((2848, 4288))


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


def create_mask(rows, cols, hull):
    # black image
    mask = np.zeros((rows, cols), dtype=np.uint8)
    # blit our contours onto it in white color
    cv2.drawContours(mask, [hull], 0, 255, -1)
    return mask


# Contours are drawn on the original image.
def fill_contours(image, contours, color):
    for i in range(0, len(contours)):
        cv2.drawContours(image, contours, i, color, cv2.FILLED)
