import argparse


def get_args():
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
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers for concurrent processing",
    )
    parser.add_argument(
        "--colour",
        dest="colour",
        action="store_true",
        help="Whether to colour the labels",
    )
    parser.add_argument(
        "--nocolour",
        action="store_false",
        dest="colour",
    )
    parser.set_defaults(colour=False)

    return parser.parse_args()
