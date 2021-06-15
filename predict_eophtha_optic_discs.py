from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch import nn
from tqdm import tqdm

from src.data.preprocess.common import (
    BLACK,
    find_eye,
    open_colour_image,
    pad_to_square,
    transform,
)
from src.data.preprocess.eophtha import change_suffix
from src.models.unet.common import load_binary_segmentation_model


def predict_od(model: nn.Module, transformed_image: np.ndarray):
    transformed_image = TF.to_tensor(transformed_image)
    transformed_image = TF.resize(transformed_image, (512, 512))
    transformed_image = transformed_image.unsqueeze(0).float().cuda()
    output = model(transformed_image)
    output = torch.argmax(output, dim=1)
    output = TF.resize(output, 1280)
    output = output.squeeze(0)
    return output.cpu().detach().numpy()


def main():
    name = "od_transformed"
    out_dir = Path("data/eophtha/od")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_binary_segmentation_model(name)
    root_path = Path("/vol/bitbucket/js6317/individual-project/data/e_optha")
    path_1 = root_path / "e_optha_MA" / "healthy"
    path_2 = root_path / "e_optha_MA" / "MA"
    path_3 = root_path / "e_optha_EX" / "EX"
    path_4 = root_path / "e_optha_EX" / "healthy"

    suffixes = [".JPG", ".jpg", ".PNG", ".png", ".JPEG", ".jpeg"]
    files_1 = [f for f in path_1.glob("**/*") if f.is_file() and f.suffix in suffixes]
    files_2 = [f for f in path_2.glob("**/*") if f.is_file() and f.suffix in suffixes]
    files_3 = [f for f in path_3.glob("**/*") if f.is_file() and f.suffix in suffixes]
    files_4 = [f for f in path_4.glob("**/*") if f.is_file() and f.suffix in suffixes]
    files = files_1 + files_2 + files_3 + files_4
    for file in tqdm(files):
        img = open_colour_image(file)

        contour = find_eye(img)
        # Find bounding box.
        x, y, w, h = cv2.boundingRect(contour)

        # Crop around bounding box.
        img = img[y : y + h, x : x + w]

        # Pad to square.
        img = pad_to_square(img, w, h, [BLACK, BLACK, BLACK])

        # Resize to 1280 x 1280.
        img = cv2.resize(img, (1280, 1280), interpolation=cv2.INTER_NEAREST)
        transformed_image = transform(img)

        predicted = predict_od(model, transformed_image)

        image_path = Path(*file.parts[-2:])
        new_name = "_".join(image_path.parts)
        new_name = change_suffix(new_name, ".png")
        cv2.imwrite(f"data/eophtha/od/{new_name}", predicted * 255.0)


if __name__ == "__main__":
    main()
