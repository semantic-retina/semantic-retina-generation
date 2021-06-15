import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        type=str,
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=1000,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--lr_g",
        type=float,
        default=0.0005,
        help="Generator learning rate",
    )
    parser.add_argument(
        "--lr_d",
        type=float,
        default=0.0001,
        help="Discriminator learning rate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Output directory",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=128,
        help="Dimensionality of the latent space",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--chkpt_interval",
        type=int,
        default=500,
        help="Interval between checkpoints, in terms of epochs",
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
        "--sample_interval",
        type=int,
        default=50,
        help="Interval between image samples, in terms of epochs",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=50,
        help="Interval between validation, in terms of epochs",
    )
    parser.add_argument(
        "--log_step",
        type=int,
        default=200,
        help="Interval between logging, in terms of batches",
    )
    parser.add_argument(
        "--lesions",
        type=str,
        nargs="+",
        default=["RETINA", "OD", "MA", "HE", "EX", "SE", "NV", "IRMA"],
        help="Which lesions to generate, as specified by the `Labels` enum",
    )
    parser.add_argument(
        "--filter_dataset",
        type=str,
        default="",
    )
    parser.add_argument(
        "--label_smoothing",
        dest="label_smoothing",
        action="store_true",
        help="Label smoothing",
    )
    parser.add_argument(
        "--nolabel_smoothing",
        dest="label_smoothing",
        action="store_false",
    )
    parser.add_argument(
        "--clip_gradient",
        dest="clip_gradient",
        action="store_true",
        help="Clip the discriminator gradient",
    )
    parser.add_argument(
        "--noclip_gradient",
        action="store_false",
        dest="clip_gradient",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        dest="tensorboard",
    )
    parser.add_argument(
        "--notensorboard",
        action="store_false",
        dest="tensorboard",
    )
    parser.add_argument(
        "--use_copypaste",
        action="store_true",
        dest="use_copypaste",
    )
    parser.add_argument(
        "--nouse_copypaste",
        action="store_false",
        dest="use_copypaste",
    )
    parser.add_argument(
        "--use_ada",
        action="store_true",
        dest="use_ada",
    )
    parser.add_argument(
        "--nouse_ada",
        action="store_false",
        dest="use_ada",
    )
    parser.set_defaults(
        label_smoothing=True,
        clip_gradient=True,
        tensorboard=True,
        use_copypaste=False,
        use_ada=True,
    )
    return parser.parse_args()
