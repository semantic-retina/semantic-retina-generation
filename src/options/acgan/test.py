import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        type=str,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--n_samples", type=int, default=128, help="Number of samples to generate"
    )
    parser.add_argument(
        "--upsample_factor",
        type=int,
        default=1,
        help="Factor by which upsampling is applied to generated images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--epoch",
        type=str,
        default="",
        help="Which checkpoint to load. Uses the final checkpoint if left unset",
    )
    parser.add_argument(
        "--colour",
        action="store_true",
        dest="colour",
    )
    parser.add_argument(
        "--nocolour",
        action="store_false",
        dest="colour",
    )
    parser.set_defaults(colour=False)
    return parser.parse_args()
