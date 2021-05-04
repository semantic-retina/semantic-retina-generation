import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
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
