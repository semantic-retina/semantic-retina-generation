import argparse
from pathlib import Path

from tqdm import tqdm

from src.data.preprocess.common import open_colour_image, transform, write_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        type=str,
    )
    opt = parser.parse_args()

    if opt.name == "copypaste":
        generated_retina_path = Path("results") / opt.name / "img"
        out_path = Path("results") / opt.name / "transformed"
    else:
        generated_retina_path = Path("results") / opt.name / "test" / "img"
        out_path = Path("results") / opt.name / "test" / "transformed"
    out_path.mkdir(parents=True, exist_ok=True)

    files = list(generated_retina_path.glob("**/*"))
    for path in tqdm(files):
        retina_img = open_colour_image(str(path))
        transformed_retina_img = transform(retina_img)
        write_image(transformed_retina_img, out_path / path.name)


if __name__ == "__main__":
    main()
