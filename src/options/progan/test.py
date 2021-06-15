import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        type=str,
    )
    parser.add_argument("--out_dir", type=str, default="results/")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
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
    parser.add_argument(
        "--mask_retina",
        action="store_true",
        dest="mask_retina",
    )
    parser.add_argument(
        "--nomask_retina",
        action="store_false",
        dest="mask_retina",
    )
    parser.set_defaults(colour=False, mask_retina=False)
    return parser.parse_args()
