"""
This script preprocesses the FGADR and IDRiD datasets. It generates the following
directories:
- `fgadr/inst`: Instance maps of the optic disc for the FGADR dataset.
- `fgadr/label`: Semantic labels for the FGADR dataset.
- `idrid/img`: Square-cropped retina images for the IDRiD dataset.
- `idrid/inst`: Instance maps of the optic disc for the IDRiD dataset.
- `idrid/label`: Semantic labels for the IDRiD dataset.

All images are of size 1280x1280. FGADR retina images not cropped, since they are
already of the desired shape, unlike the IDRiD images which are of varying shapes.
The semantic labels are encoded as gray values with:
- Retina: 0
- Optic disc (OD): 1
- Microaneurysm (MA): 2
- Hemorrhages (HE): 3
- Hard exudates (EX): 4
- Soft exudates (SE): 5
- Background: 255
"""

import argparse

from src.data.preprocess.fgadr import preprocess_fgadr
from src.data.preprocess.idrid import preprocess_idrid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--idrid_root_dir",
        type=str,
        default="/vol/vipdata/data/retina/IDRID/a_segmentation/",
        help="Path to the IDRiD dataset",
    )
    parser.add_argument(
        "--fgadr_root_dir",
        type=str,
        default="/vol/vipdata/data/retina/FGADR-Seg/Seg-set",
        help="Path to the FGADR dataset",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="data/total.json",
        help="JSON file containing annotations for the FGADR dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/",
    )
    parser.add_argument("--n_workers", type=int, default=4)

    opt = parser.parse_args()
    preprocess_idrid(opt.idrid_root_dir, opt.output_dir, opt.n_workers, True)
    preprocess_idrid(opt.idrid_root_dir, opt.output_dir, opt.n_workers, False)
    preprocess_fgadr(
        opt.fgadr_root_dir, opt.output_dir, opt.n_workers, opt.annotation_file
    )


if __name__ == "__main__":
    main()
