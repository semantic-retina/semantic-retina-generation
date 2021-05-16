import json
from pathlib import Path

import cv2
import numpy as np
from tqdm.contrib.concurrent import thread_map

from src.data.preprocess.common import (
    BLACK,
    GRAY_CLASS,
    WHITE,
    find_eye,
    open_binary_mask,
    open_colour_image,
    overlay_label,
    pad_to_square,
    transform,
    write_image,
)
from src.utils.sample import colour_labels_numpy

# This is the max confidence level as a raw pixel value from the images.
MAX_VALUE = 252
CONFIDENCE_THRESHOLD = 0.75


def draw_od(label: np.ndarray, inst: np.ndarray, image_name: str, file_path: str):
    """Writes optic disc annotations onto both the label and inst images."""
    with open(file_path) as json_file:
        data = json.load(json_file)

    if image_name not in data:
        return

    x = data[image_name]["regions"]["0"]["shape_attributes"]["all_points_x"]
    y = data[image_name]["regions"]["0"]["shape_attributes"]["all_points_y"]

    # Draw optic disc.
    pts = np.array((np.int32(x), np.int32(y))).T
    pts = pts.reshape((-1, 1, 2))
    ellipse = cv2.fitEllipse(pts)

    cv2.ellipse(label, ellipse, GRAY_CLASS["OD"], cv2.FILLED, 0)
    cv2.ellipse(inst, ellipse, GRAY_CLASS["OD"], cv2.FILLED, 0)


def process_image(
    image_name: str,
    retina_path: Path,
    ex_path: Path,
    he_path: Path,
    ma_path: Path,
    se_path: Path,
    retina_mask: np.ndarray,
    retina_label: np.ndarray,
    ex_label: np.ndarray,
    he_label: np.ndarray,
    ma_label: np.ndarray,
    se_label: np.ndarray,
    label_output_path: Path,
    inst_output_path: Path,
    img_output_path: Path,
    img_transformed_output_path: Path,
    od_file_path: str,
    colour: bool,
):
    retina_img = open_colour_image(retina_path / image_name)
    contour = find_eye(retina_mask)

    ex_img = open_binary_mask(ex_path / image_name)
    he_img = open_binary_mask(he_path / image_name)
    ma_img = open_binary_mask(ma_path / image_name)
    se_img = open_binary_mask(se_path / image_name)

    img_size = (1152, 1500)
    mask = np.ones(img_size, dtype="uint8") * WHITE
    inst = np.ones(img_size, dtype="uint8") * WHITE

    thresh = int(MAX_VALUE * CONFIDENCE_THRESHOLD)
    overlay_label(mask, retina_mask, retina_label, thresh)
    draw_od(mask, inst, image_name, od_file_path)
    overlay_label(mask, ex_img, ex_label, thresh=thresh)
    overlay_label(mask, he_img, he_label, thresh=thresh)
    overlay_label(mask, ma_img, ma_label, thresh=thresh)
    overlay_label(mask, se_img, se_label, thresh=thresh)

    # Find bounding box.
    x, y, w, h = cv2.boundingRect(contour)

    # Crop around bounding box.
    mask = mask[y : y + h, x : x + w]
    inst = inst[y : y + h, x : x + w]
    retina_img = retina_img[y : y + h, x : x + w]

    # Pad to square.
    mask = pad_to_square(mask, w, h, WHITE)
    inst = pad_to_square(inst, w, h, WHITE)
    retina_img = pad_to_square(retina_img, w, h, [BLACK, BLACK, BLACK])

    # Resize to 1280 x 1280.
    mask = cv2.resize(mask, (1280, 1280), interpolation=cv2.INTER_NEAREST)
    inst = cv2.resize(inst, (1280, 1280), interpolation=cv2.INTER_NEAREST)
    retina_img = cv2.resize(retina_img, (1280, 1280), interpolation=cv2.INTER_NEAREST)
    transformed_retina_img = transform(retina_img)

    if colour:
        mask = colour_labels_numpy(mask)

    write_image(mask, label_output_path / image_name)
    write_image(inst, inst_output_path / image_name)
    write_image(retina_img, img_output_path / image_name)
    write_image(transformed_retina_img, img_transformed_output_path / image_name)


def preprocess_diaretdb1(
    root_dir: str,
    output_dir: str,
    n_workers: int,
    od_file_path: str,
    colour: bool,
):
    root_path = Path(root_dir) / "resources" / "images"

    output_path = Path(output_dir) / "diaretdb1"

    label_output_path = output_path / "label"
    label_output_path.mkdir(parents=True, exist_ok=True)

    inst_output_path = output_path / "inst"
    inst_output_path.mkdir(parents=True, exist_ok=True)

    img_output_path = output_path / "img"
    img_output_path.mkdir(parents=True, exist_ok=True)

    img_transformed_output_path = output_path / "transformed"
    img_transformed_output_path.mkdir(parents=True, exist_ok=True)

    retina_path = root_path / "ddb1_fundusimages"

    label_path = root_path / "ddb1_groundtruth"
    ex_path = label_path / "hardexudates"
    he_path = label_path / "hemorrhages"
    ma_path = label_path / "redsmalldots"
    se_path = label_path / "softexudates"

    # Labels should never be modified.
    img_size = (1152, 1500)
    retina_label = np.ones(img_size, dtype="uint8") * GRAY_CLASS["RETINA"]
    ma_label = np.ones(img_size, dtype="uint8") * GRAY_CLASS["MA"]
    se_label = np.ones(img_size, dtype="uint8") * GRAY_CLASS["SE"]
    he_label = np.ones(img_size, dtype="uint8") * GRAY_CLASS["HE"]
    ex_label = np.ones(img_size, dtype="uint8") * GRAY_CLASS["EX"]

    retina_mask = open_binary_mask(root_path / "ddb1_fundusmask" / "fmask.tif")

    image_names = [f.name for f in retina_path.glob("**/*")]

    # Worker function that wraps the image processing function.
    def worker(image_name: str):
        process_image(
            image_name=image_name,
            retina_path=retina_path,
            ex_path=ex_path,
            he_path=he_path,
            ma_path=ma_path,
            se_path=se_path,
            retina_mask=retina_mask,
            retina_label=retina_label,
            ex_label=ex_label,
            he_label=he_label,
            ma_label=ma_label,
            se_label=se_label,
            label_output_path=label_output_path,
            inst_output_path=inst_output_path,
            img_output_path=img_output_path,
            img_transformed_output_path=img_transformed_output_path,
            od_file_path=od_file_path,
            colour=colour,
        )

    print(f"Preprocessing DIARETDB1 with {n_workers} workers...")
    thread_map(worker, image_names, max_workers=n_workers)
