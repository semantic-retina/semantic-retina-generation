import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("--dataset", type=str, default="test")
    parser.add_argument(
        "--lesions",
        type=str,
        nargs="+",
        default=["MA", "HE", "EX", "SE", "IRMA", "NV"],
        help="Which lesions to generate, as specified by the `Labels` enum",
    )
    return parser.parse_args()
