import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("--n_epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Optimizer learning rate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="Output directory"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=100, help="Dimensionality of the latent space"
    )
    parser.add_argument("--n_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument(
        "--n_channels", type=int, default=3, help="Number of output image channels"
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=10,
        help="Interval between image samples",
    )
    parser.add_argument(
        "--chkpt_interval", type=int, default=100, help="Interval between checkpoints"
    )
    parser.add_argument(
        "--n_critic",
        type=int,
        default=1,
        help="Discriminator training iterations for each batch",
    )
    parser.add_argument(
        "--n_gen",
        type=int,
        default=2,
        help="Generator training iterations for each batch",
    )
    parser.add_argument(
        "--label_smoothing",
        type=bool,
        default=True,
        help="Label smoothing",
    )
    parser.add_argument(
        "--clip_gradient",
        type=bool,
        default=True,
        help="Whether to clip the discriminator gradient",
    )
    return parser.parse_args()
