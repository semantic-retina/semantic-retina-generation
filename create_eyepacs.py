from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import thread_map

from src.data.preprocess.common import BLACK, find_eye, open_colour_image, pad_to_square


def process_df(df, f, group_name):
    # We save as [C, H, W]
    train_shape = (len(df), 3, 512, 512)
    train_labels = df["level"].to_numpy()

    grp = f.create_group(group_name)
    grp.create_dataset(
        "images",
        shape=train_shape,
        dtype=np.uint8,
        # compression="lzf",
    )
    grp.create_dataset("labels", shape=train_labels.shape, dtype=np.uint8)
    grp["labels"][...] = train_labels

    grp_images = grp["images"]

    def process_image(i):
        path = df["path"].iloc[i]
        img = open_colour_image(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        contour = find_eye(img)
        x, y, w, h = cv2.boundingRect(contour)
        img = img[y : y + h, x : x + w]
        img = pad_to_square(img, w, h, [BLACK, BLACK, BLACK])

        img = cv2.resize(img, (512, 512))

        scale = 512
        img = cv2.addWeighted(
            img, 4, cv2.GaussianBlur(img, (0, 0), scale / 30), -4, 128
        )

        # Remove outer 10% boundary effects
        b = np.zeros(img.shape)
        b = cv2.circle(
            b,
            (img.shape[1] // 2, img.shape[0] // 2),
            int(scale * 0.9) // 2,
            (1, 1, 1),
            -1,
            8,
            0,
        )
        img = img * b + 128 * (1 - b)

        img = np.transpose(img, (2, 0, 1))
        grp_images[i, ...] = img[None]

    thread_map(process_image, range(len(df)), max_workers=8)


def main():
    path = Path("/data/js6317") / "hdf5"
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / "eyepacs.hdf5"

    data_path = Path("data/eyepacs")

    f = h5py.File(file_path, "w", libver="latest")

    # CSV columns are: "PatientId, name, eye, level, level_binary,
    # level_hot, path, path_preprocess, exists".
    train_df_path = data_path / "train.csv"
    train_df = pd.read_csv(train_df_path)

    test_df_path = data_path / "test.csv"
    test_df = pd.read_csv(test_df_path)

    print("Train:", len(train_df))
    print("Test:", len(test_df))
    process_df(train_df, f, "train")
    process_df(test_df, f, "test")

    f.close()


if __name__ == "__main__":
    main()
