import json
from pathlib import Path

import cv2
import numpy as np
from tqdm.contrib.concurrent import thread_map

from src.data.preprocess.common import (
    GRAY_CLASS,
    WHITE,
    create_mask,
    fill_contours,
    open_binary_mask,
    open_colour_image,
    overlay_label,
    write_image,
)
from src.utils.sample import colour_labels_numpy


def find_eye(image, thresh=4):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = np.array([])
    edges = cv2.Canny(gray, thresh, thresh * 3, edges)

    # Find contours; second output is hierarchy - we are not interested in it.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Now let's get only what we need out of it.
    hull_contours = cv2.convexHull(np.vstack(np.array(contours)))
    hull = np.vstack(hull_contours)

    mask = create_mask(*image.shape[0:2], hull)

    return mask, hull


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
    ex_label: np.ndarray,
    he_label: np.ndarray,
    ma_label: np.ndarray,
    se_label: np.ndarray,
    label_output_path: Path,
    inst_output_path: Path,
    od_file_path: str,
    colour: bool,
):
    retina_img = open_colour_image(retina_path / image_name)
    retina_mask, hull = find_eye(retina_img)

    ex_img = open_binary_mask(ex_path / image_name)
    he_img = open_binary_mask(he_path / image_name)
    ma_img = open_binary_mask(ma_path / image_name)
    se_img = open_binary_mask(se_path / image_name)

    label = np.ones((1280, 1280), dtype="uint8") * WHITE
    inst = np.ones((1280, 1280), dtype="uint8") * WHITE

    fill_contours(label, [hull], GRAY_CLASS["BG"])

    draw_od(label, inst, image_name, od_file_path)
    overlay_label(label, ex_img, ex_label)
    overlay_label(label, he_img, he_label)
    overlay_label(label, ma_img, ma_label)
    overlay_label(label, se_img, se_label)

    if colour:
        label = colour_labels_numpy(label)

    write_image(label, label_output_path / image_name)
    write_image(inst, inst_output_path / image_name)


def preprocess_fgadr(
    root_dir: str,
    output_dir: str,
    n_workers: int,
    od_file_path: str,
    colour: bool,
):
    root_path = Path(root_dir)

    output_path = Path(output_dir) / "fgadr"

    label_output_path = output_path / "label"
    label_output_path.mkdir(parents=True, exist_ok=True)

    inst_output_path = output_path / "inst"
    inst_output_path.mkdir(parents=True, exist_ok=True)

    retina_path = root_path / "Original_Images"
    ex_path = root_path / "HardExudate_Masks"
    # Note: there is a typo in the folder name, the spelling here is intentional.
    he_path = root_path / "Hemohedge_Masks"
    ma_path = root_path / "Microaneurysms_Masks"
    se_path = root_path / "SoftExudate_Masks"

    ma_label = np.ones((1280, 1280), dtype="uint8") * GRAY_CLASS["MA"]
    se_label = np.ones((1280, 1280), dtype="uint8") * GRAY_CLASS["SE"]
    he_label = np.ones((1280, 1280), dtype="uint8") * GRAY_CLASS["HE"]
    ex_label = np.ones((1280, 1280), dtype="uint8") * GRAY_CLASS["EX"]

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
            ex_label=ex_label,
            he_label=he_label,
            ma_label=ma_label,
            se_label=se_label,
            label_output_path=label_output_path,
            inst_output_path=inst_output_path,
            od_file_path=od_file_path,
            colour=colour,
        )

    print(f"Preprocessing FGADR with {n_workers} workers...")
    thread_map(worker, image_names, max_workers=n_workers)
