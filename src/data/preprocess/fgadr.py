import json
from pathlib import Path

import cv2
import numpy as np
from tqdm.contrib.concurrent import thread_map

from src.data.preprocess.common import (
    GRAY_CLASS,
    WHITE,
    fill_contours,
    find_eye,
    open_binary_mask,
    open_colour_image,
    overlay_label,
    write_image,
)
from src.utils.sample import colour_labels_numpy


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
    nv_path: Path,
    irma_path: Path,
    ex_label: np.ndarray,
    he_label: np.ndarray,
    ma_label: np.ndarray,
    se_label: np.ndarray,
    nv_label: np.ndarray,
    irma_label: np.ndarray,
    label_output_path: Path,
    inst_output_path: Path,
    od_file_path: str,
    colour: bool,
):
    retina_img = open_colour_image(retina_path / image_name)
    contour = find_eye(retina_img)

    ex_img = open_binary_mask(ex_path / image_name)
    he_img = open_binary_mask(he_path / image_name)
    ma_img = open_binary_mask(ma_path / image_name)
    se_img = open_binary_mask(se_path / image_name)
    nv_img = open_binary_mask(nv_path / image_name)
    irma_img = open_binary_mask(irma_path / image_name)

    mask = np.ones((1280, 1280), dtype="uint8") * WHITE
    inst = np.ones((1280, 1280), dtype="uint8") * WHITE

    fill_contours(mask, [contour], GRAY_CLASS["RETINA"])
    draw_od(mask, inst, image_name, od_file_path)
    overlay_label(mask, ex_img, ex_label)
    overlay_label(mask, he_img, he_label)
    overlay_label(mask, ma_img, ma_label)
    overlay_label(mask, se_img, se_label)
    overlay_label(mask, nv_img, nv_label)
    overlay_label(mask, irma_img, irma_label)

    if colour:
        mask = colour_labels_numpy(mask)

    write_image(mask, label_output_path / image_name)
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
    nv_path = root_path / "Neovascularization_Masks"
    irma_path = root_path / "IRMA_Masks"

    # Labels should never be modified.
    ma_label = np.ones((1280, 1280), dtype="uint8") * GRAY_CLASS["MA"]
    se_label = np.ones((1280, 1280), dtype="uint8") * GRAY_CLASS["SE"]
    he_label = np.ones((1280, 1280), dtype="uint8") * GRAY_CLASS["HE"]
    ex_label = np.ones((1280, 1280), dtype="uint8") * GRAY_CLASS["EX"]
    nv_label = np.ones((1280, 1280), dtype="uint8") * GRAY_CLASS["NV"]
    irma_label = np.ones((1280, 1280), dtype="uint8") * GRAY_CLASS["IRMA"]

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
            nv_path=nv_path,
            irma_path=irma_path,
            ex_label=ex_label,
            he_label=he_label,
            ma_label=ma_label,
            se_label=se_label,
            nv_label=nv_label,
            irma_label=irma_label,
            label_output_path=label_output_path,
            inst_output_path=inst_output_path,
            od_file_path=od_file_path,
            colour=colour,
        )

    print(f"Preprocessing FGADR with {n_workers} workers...")
    thread_map(worker, image_names, max_workers=n_workers)
