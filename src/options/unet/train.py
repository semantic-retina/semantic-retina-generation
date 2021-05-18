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
        default=50,
        help="Number of epochs",
        dest="epochs",
    )
    parser.add_argument(
        "--batch_size",
        metavar="B",
        type=int,
        nargs="?",
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        metavar="LR",
        type=float,
        nargs="?",
        default=0.001,
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
        help="How many synthetic samples to use",
    )
    parser.add_argument(
        "--n_real",
        type=int,
        default=-1,
        help="How many real samples to use",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--load_name",
        type=str,
        default="",
        help="If specified, loads the model with this name",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1000,
    )
    return parser.parse_args()
