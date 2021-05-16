import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        type=str,
    )
    parser.add_argument(
        "--n_epochs",
        metavar="E",
        type=int,
        default=40,
        help="Number of epochs",
        dest="epochs",
    )
    parser.add_argument(
        "--batch_size",
        metavar="B",
        type=int,
        nargs="?",
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        metavar="LR",
        type=float,
        nargs="?",
        default=0.0001,
        help="Learning rate",
    )
    parser.add_argument(
        "--val_proportion",
        type=float,
        default=0.2,
        help="Proportion of the data to use as validation",
    )
    parser.add_argument(
        "--n_synthetic",
        type=int,
        default=0,
        help="Whether or not to use synthetic data during training",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
    )

    return parser.parse_args()
