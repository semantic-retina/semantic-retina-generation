from pathlib import Path

import cv2
from tqdm.contrib.concurrent import thread_map

from src.data.preprocess.common import (
    BLACK,
    change_suffix,
    find_eye,
    open_colour_image,
    pad_to_square,
    transform,
    write_image,
)


def process_image(
    image_name: str,
    retina_path: Path,
    img_output_path: Path,
    img_transformed_output_path: Path,
):
    retina_img = open_colour_image(retina_path / image_name)
    contour = find_eye(retina_img)

    # Find bounding box.
    x, y, w, h = cv2.boundingRect(contour)

    # Crop around bounding box.
    retina_img = retina_img[y : y + h, x : x + w]

    # Pad to square.
    retina_img = pad_to_square(retina_img, w, h, [BLACK, BLACK, BLACK])

    # Resize to 1280 x 1280.
    retina_img = cv2.resize(retina_img, (1280, 1280), interpolation=cv2.INTER_NEAREST)
    transformed_retina_img = transform(retina_img)

    new_name = change_suffix(image_name, ".png")

    write_image(retina_img, img_output_path / new_name)
    write_image(transformed_retina_img, img_transformed_output_path / new_name)


def preprocess_idrid_grade(root_dir: str, output_dir: str, n_workers: int, train: bool):
    root_path = Path(root_dir)

    output_path = Path(output_dir) / "idrid_grade"

    img_output_path = output_path / "img"
    img_output_path.mkdir(parents=True, exist_ok=True)

    img_transformed_output_path = output_path / "transformed"
    img_transformed_output_path.mkdir(parents=True, exist_ok=True)

    train_test_path = "a. Training Set" if train else "b. Testing Set"

    retina_path = root_path / "1. Original Images" / train_test_path
    print(retina_path)

    image_names = [f.name for f in retina_path.glob("**/*")]

    # Worker function that wraps the image processing function.
    def worker(image_name: str):
        process_image(
            image_name=image_name,
            retina_path=retina_path,
            img_output_path=img_output_path,
            img_transformed_output_path=img_transformed_output_path,
        )

    print(f"Preprocessing IDRiD Grade ({train_test_path}) with {n_workers} workers...")
    thread_map(worker, image_names, max_workers=n_workers)
