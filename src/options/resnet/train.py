import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--val_interval", type=int, default=1000)
    parser.add_argument("--synthetic_name", type=str, default="")
    parser.add_argument(
        "--load_name",
        type=str,
        default="",
        help="If specified, loads the model with this name",
    )
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--subsample_size", type=int, default=-1)
    parser.add_argument("--n_synthetic", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--tensorboard", action="store_true", dest="tensorboard")
    parser.add_argument("--notensorboard", action="store_false", dest="tensorboard")
    parser.add_argument("--use_hdf5", action="store_true", dest="use_hdf5")
    parser.add_argument("--nouse_hdf5", action="store_false", dest="use_hdf5")
    parser.add_argument("--use_real", action="store_true", dest="use_real")
    parser.add_argument("--nouse_real", action="store_false", dest="use_real")
    parser.set_defaults(tensorboard=True, use_hdf5=True, use_real=True)
    return parser.parse_args()
