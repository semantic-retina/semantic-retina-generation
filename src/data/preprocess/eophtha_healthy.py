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
from src.data.preprocess.idrid import change_suffix
from src.utils.sample import colour_labels_numpy


def process_image(
    image_path: Path,
    retina_path: Path,
    od_path: Path,
    label_output_path: Path,
    instance_output_path: Path,
    img_output_path: Path,
    img_transformed_output_path: Path,
    colour: bool,
):
    retina_img = open_colour_image(retina_path / image_path)

    height, width, _ = retina_img.shape

    contour = find_eye(retina_img)

    bg_label = np.ones((height, width), dtype="uint8") * GRAY_CLASS["RETINA"]

    mask = np.ones((height, width), dtype="uint8") * WHITE

    fill_contours(mask, [contour], GRAY_CLASS["RETINA"])

    inst = np.ones((height, width), dtype="uint8") * WHITE

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

    new_name = "_".join(image_path.parts)
    new_name = change_suffix(new_name, ".png")

    od_img = open_binary_mask(od_path / new_name)
    od_label = np.ones((1280, 1280), dtype="uint8") * GRAY_CLASS["OD"]
    overlay_label(mask, od_img, od_label)

    overlay_label(inst, od_img, bg_label)

    if colour:
        mask = colour_labels_numpy(mask)

    write_image(mask, label_output_path / new_name)
    write_image(inst, instance_output_path / new_name)
    write_image(retina_img, img_output_path / new_name)
    write_image(transformed_retina_img, img_transformed_output_path / new_name)


def preprocess_eophtha_ma(
    root_dir: str, od_dir: str, output_dir: str, n_workers: int, colour: bool
):
    root_path = Path(root_dir)
    od_path = Path(od_dir)

    output_path = Path(output_dir) / "eophtha"

    label_output_path = output_path / "label"
    label_output_path.mkdir(parents=True, exist_ok=True)

    inst_output_path = output_path / "inst"
    inst_output_path.mkdir(parents=True, exist_ok=True)

    img_output_path = output_path / "img"
    img_output_path.mkdir(parents=True, exist_ok=True)

    img_transformed_output_path = output_path / "transformed"
    img_transformed_output_path.mkdir(parents=True, exist_ok=True)

    retina_path = root_path / "e_optha_MA" / "MA"

    suffixes = [".JPG", ".jpg", ".png", ".PNG"]
    image_names = [
        Path(*f.parts[-2:])
        for f in retina_path.glob("**/*")
        if f.is_file() and f.suffix in suffixes
    ]

    # Worker function that wraps the image processing function.
    def worker(image_name: str):
        process_image(
            image_path=image_name,
            retina_path=retina_path,
            od_path=od_path,
            label_output_path=label_output_path,
            instance_output_path=inst_output_path,
            img_output_path=img_output_path,
            img_transformed_output_path=img_transformed_output_path,
            colour=colour,
        )

    print(f"Preprocessing e-ophtha_MA) with {n_workers} workers...")
    thread_map(worker, image_names, max_workers=n_workers)
