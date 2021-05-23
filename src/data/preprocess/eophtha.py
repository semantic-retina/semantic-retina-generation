from pathlib import Path

import cv2
import numpy as np
from tqdm.contrib.concurrent import thread_map

from src.data.preprocess.common import (
    BLACK,
    GRAY_CLASS,
    WHITE,
    change_suffix,
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


def process_image(
    retina_path: Path,
    ex_path: Path,
    ma_path: Path,
    od_path: Path,
    label_output_path: Path,
    instance_output_path: Path,
    img_output_path: Path,
    img_transformed_output_path: Path,
    colour: bool,
):
    retina_img = open_colour_image(retina_path)
    image_path = Path(*retina_path.parts[-2:])

    height, width, _ = retina_img.shape
    img_size = (height, width)

    contour = find_eye(retina_img)

    ex_label = np.ones(img_size, dtype="uint8") * GRAY_CLASS["EX"]
    ma_label = np.ones(img_size, dtype="uint8") * GRAY_CLASS["MA"]

    ex_img = open_binary_mask(
        ex_path / change_suffix(str(image_path), f"_EX.png"), img_size=img_size
    )
    ma_img = open_binary_mask(
        ma_path / change_suffix(str(image_path), f".png"), img_size=img_size
    )

    mask = np.ones(img_size, dtype="uint8") * WHITE

    fill_contours(mask, [contour], GRAY_CLASS["RETINA"])
    overlay_label(mask, ex_img, ex_label)
    overlay_label(mask, ma_img, ma_label)

    inst = np.ones(img_size, dtype="uint8") * WHITE

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

    bg_label = np.ones((1280, 1280), dtype="uint8") * GRAY_CLASS["RETINA"]
    overlay_label(inst, od_img, bg_label)

    if colour:
        mask = colour_labels_numpy(mask)

    write_image(mask, label_output_path / new_name)
    write_image(inst, instance_output_path / new_name)
    write_image(retina_img, img_output_path / new_name)
    write_image(transformed_retina_img, img_transformed_output_path / new_name)


def preprocess_eophtha(
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

    retina_path_ex_healthy = root_path / "e_optha_EX" / "healthy"
    retina_path_ma_healthy = root_path / "e_optha_MA" / "healthy"
    retina_path_ex = root_path / "e_optha_EX" / "EX"
    retina_path_ma = root_path / "e_optha_MA" / "MA"

    ex_path = root_path / "e_optha_EX" / "Annotation_EX"
    ma_path = root_path / "e_optha_MA" / "Annotation_MA"

    suffixes = [".JPG", ".jpg", ".png", ".PNG"]
    retina_paths = []
    # There might be duplicates between e.g. healthy EX and annotation MA, but since
    # healthy files are processed first they will be overwritten.
    retina_paths += [
        f
        for f in retina_path_ex_healthy.glob("**/*")
        if f.is_file() and f.suffix in suffixes
    ]
    retina_paths += [
        f
        for f in retina_path_ma_healthy.glob("**/*")
        if f.is_file() and f.suffix in suffixes
    ]
    retina_paths += [
        f for f in retina_path_ex.glob("**/*") if f.is_file() and f.suffix in suffixes
    ]
    retina_paths += [
        f for f in retina_path_ma.glob("**/*") if f.is_file() and f.suffix in suffixes
    ]

    # Worker function that wraps the image processing function.
    def worker(retina_path: Path):
        process_image(
            retina_path=retina_path,
            ex_path=ex_path,
            ma_path=ma_path,
            od_path=od_path,
            label_output_path=label_output_path,
            instance_output_path=inst_output_path,
            img_output_path=img_output_path,
            img_transformed_output_path=img_transformed_output_path,
            colour=colour,
        )

    print(f"Preprocessing e-ophtha (EX, MA) with {n_workers} workers...")
    thread_map(worker, retina_paths, max_workers=n_workers)
