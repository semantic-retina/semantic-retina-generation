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
- Neovascularization (NV): 6
- Intraretinal microvascular abnormalities (IRMA): 7
- Background: 255
"""
from src.data.preprocess.diaretdb import preprocess_diaretdb1
from src.data.preprocess.eophtha import preprocess_eophtha
from src.data.preprocess.fgadr import preprocess_fgadr
from src.data.preprocess.idrid import preprocess_idrid
from src.data.preprocess.idrid_grade import preprocess_idrid_grade
from src.options.preprocess import get_args


def main():
    opt = get_args()
    if opt.idrid:
        preprocess_idrid(
            opt.idrid_root_dir,
            opt.output_dir,
            opt.n_workers,
            True,
            opt.colour,
        )
        preprocess_idrid(
            opt.idrid_root_dir,
            opt.output_dir,
            opt.n_workers,
            False,
            opt.colour,
        )
    if opt.idrid_grade:
        preprocess_idrid_grade(
            opt.idrid_grade_root_dir,
            opt.output_dir,
            opt.n_workers,
            True,
        )
        preprocess_idrid_grade(
            opt.idrid_grade_root_dir,
            opt.output_dir,
            opt.n_workers,
            False,
        )
    if opt.fgadr:
        preprocess_fgadr(
            opt.fgadr_root_dir,
            opt.output_dir,
            opt.n_workers,
            opt.fgadr_annotation_file,
            opt.colour,
        )
    if opt.diaretdb1:
        preprocess_diaretdb1(
            opt.diaretdb1_root_dir,
            opt.output_dir,
            opt.n_workers,
            opt.diaretdb1_annotation_file,
            opt.colour,
        )
    if opt.eophtha:
        preprocess_eophtha(
            opt.eophtha_root_dir,
            opt.eophtha_od_dir,
            opt.output_dir,
            opt.n_workers,
            opt.colour,
        )


if __name__ == "__main__":
    main()
