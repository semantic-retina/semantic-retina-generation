import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--val_interval", type=int, default=5000)
    parser.add_argument("--use_synthetic", type=str, default=False)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--tensorboard", action="store_true", dest="tensorboard")
    parser.add_argument("--notensorboard", action="store_false", dest="tensorboard")
    parser.add_argument("--use_hdf5", action="store_true", dest="use_hdf5")
    parser.add_argument("--nouse_hdf5", action="store_false", dest="use_hdf5")
    parser.set_defaults(tensorboard=True, use_hdf5=True)
    return parser.parse_args()
