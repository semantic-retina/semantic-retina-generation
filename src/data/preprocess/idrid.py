from pathlib import Path

import cv2
import numpy as np
from tqdm.contrib.concurrent import thread_map

from src.data.preprocess.common import (
    BLACK,
    GRAY_CLASS,
    WHITE,
    fill_contours,
    find_eye,
    open_binary_mask,
    open_colour_image,
    overlay_label,
    pad_to_square,
    transform,
    write_image,
)
from src.utils.sample import colour_labels_numpy


def change_suffix(image_name: str, suffix: str):
    return image_name[:-4] + suffix


def process_image(
    image_name: str,
    retina_path: Path,
    ex_path: Path,
    he_path: Path,
    ma_path: Path,
    se_path: Path,
    od_path: Path,
    ex_label: np.ndarray,
    he_label: np.ndarray,
    ma_label: np.ndarray,
    se_label: np.ndarray,
    od_label: np.ndarray,
    bg_label: np.ndarray,
    label_output_path: Path,
    instance_output_path: Path,
    img_output_path: Path,
    img_transformed_output_path: Path,
    colour: bool,
):
    retina_img = open_colour_image(retina_path / image_name)
    contour = find_eye(retina_img)

    ex_img = open_binary_mask(ex_path / change_suffix(image_name, "_EX.tif"))
    he_img = open_binary_mask(he_path / change_suffix(image_name, "_HE.tif"))
    ma_img = open_binary_mask(ma_path / change_suffix(image_name, "_MA.tif"))
    se_img = open_binary_mask(se_path / change_suffix(image_name, "_SE.tif"))
    od_img = open_binary_mask(od_path / change_suffix(image_name, "_OD.tif"))

    mask = np.ones((2848, 4288), dtype="uint8") * WHITE

    fill_contours(mask, [contour], GRAY_CLASS["RETINA"])
    overlay_label(mask, od_img, od_label)
    overlay_label(mask, ex_img, ex_label)
    overlay_label(mask, he_img, he_label)
    overlay_label(mask, ma_img, ma_label)
    overlay_label(mask, se_img, se_label)

    inst = np.ones((2848, 4288), dtype="uint8") * WHITE
    overlay_label(inst, od_img, bg_label)

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

    new_name = change_suffix(image_name, ".png")

    if colour:
        mask = colour_labels_numpy(mask)

    write_image(mask, label_output_path / new_name)
    write_image(inst, instance_output_path / new_name)
    write_image(retina_img, img_output_path / new_name)
    write_image(transformed_retina_img, img_transformed_output_path / new_name)


def preprocess_idrid(
    root_dir: str, output_dir: str, n_workers: int, train: bool, colour: bool
):
    root_path = Path(root_dir)

    output_path = Path(output_dir) / "idrid"

    label_output_path = output_path / "label"
    label_output_path.mkdir(parents=True, exist_ok=True)

    inst_output_path = output_path / "inst"
    inst_output_path.mkdir(parents=True, exist_ok=True)

    img_output_path = output_path / "img"
    img_output_path.mkdir(parents=True, exist_ok=True)

    img_transformed_output_path = output_path / "transformed"
    img_transformed_output_path.mkdir(parents=True, exist_ok=True)

    train_test_path = "train" if train else "test"

    retina_path = root_path / "images" / train_test_path
    ex_path = root_path / "masks" / train_test_path / "3_Hard_Exudates"
    he_path = root_path / "masks" / train_test_path / "2_Haemorrhages"
    ma_path = root_path / "masks" / train_test_path / "1_Microaneurysms"
    se_path = root_path / "masks" / train_test_path / "4_Soft_Exudates"
    od_path = root_path / "masks" / train_test_path / "5_Optic_Disc"

    od_label = np.ones((2848, 4288), dtype="uint8") * GRAY_CLASS["OD"]
    ma_label = np.ones((2848, 4288), dtype="uint8") * GRAY_CLASS["MA"]
    se_label = np.ones((2848, 4288), dtype="uint8") * GRAY_CLASS["SE"]
    he_label = np.ones((2848, 4288), dtype="uint8") * GRAY_CLASS["HE"]
    ex_label = np.ones((2848, 4288), dtype="uint8") * GRAY_CLASS["EX"]

    bg_label = np.ones((2848, 4288), dtype="uint8") * GRAY_CLASS["RETINA"]

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
            od_path=od_path,
            ex_label=ex_label,
            he_label=he_label,
            ma_label=ma_label,
            se_label=se_label,
            od_label=od_label,
            bg_label=bg_label,
            label_output_path=label_output_path,
            instance_output_path=inst_output_path,
            img_output_path=img_output_path,
            img_transformed_output_path=img_transformed_output_path,
            colour=colour,
        )

    print(f"Preprocessing IDRiD ({train_test_path}) with {n_workers} workers...")
    thread_map(worker, image_names, max_workers=n_workers)
