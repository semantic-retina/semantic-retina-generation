from pathlib import Path
from typing import Tuple

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


def open_binary_mask(
    path: Path, img_size: Tuple[int, int] = (2848, 4288)
) -> np.ndarray:
    try:
        return open_image(path, cv2.IMREAD_GRAYSCALE)
    except FileNotFoundError:
        return np.zeros(img_size)


def change_suffix(image_name: str, suffix: str):
    return image_name[:-4] + suffix


def open_colour_image(path: Path) -> np.ndarray:
    return open_image(path, cv2.IMREAD_COLOR)


# Note: mask if modified in-place.
def overlay_label(
    mask: np.ndarray, img: np.ndarray, label: np.ndarray, thresh: int = 0
):
    img_idx = np.where(img > thresh)
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
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # A threshold value of 10 appears to work well.
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours[0]


def pad_to_square(image, w, h, colour):
    target = max(h, w)
    top_bottom = (target - h) // 2
    left_right = (target - w) // 2

    return cv2.copyMakeBorder(
        image,
        top_bottom,
        top_bottom,
        left_right,
        left_right,
        cv2.BORDER_CONSTANT,
        value=colour,
    )


def transform(image):
    scale = image.shape[0]
    image = cv2.addWeighted(
        image, 4, cv2.GaussianBlur(image, (0, 0), scale / 30), -4, 128
    )

    # Remove outer 10% boundary effects
    mask = np.zeros(image.shape)
    mask = cv2.circle(
        mask,
        (image.shape[1] // 2, image.shape[0] // 2),
        int(scale * 0.9) // 2,
        (1, 1, 1),
        -1,
        8,
        0,
    )
    image = image * mask + 128 * (1 - mask)
    return image
