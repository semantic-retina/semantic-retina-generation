import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fgadr_original_dir",
        type=str,
        default="/vol/vipdata/data/retina/FGADR-Seg/Seg-set/",
    )
    parser.add_argument(
        "--fgadr_processed_dir",
        type=str,
        default="/vol/bitbucket/js6317/individual-project/semantic-dr-gan/data/fgadr/",
    )
    parser.add_argument(
        "--idrid_processed_dir",
        type=str,
        default="/vol/bitbucket/js6317/individual-project/semantic-dr-gan/data/idrid/",
    )
    parser.add_argument(
        "--diaretdb1_processed_dir",
        type=str,
        default="/vol/bitbucket/js6317/individual-project/semantic-dr-gan/data/diaretdb1/",
    )
    parser.add_argument(
        "--eophtha_processed_dir",
        type=str,
        default="/vol/bitbucket/js6317/individual-project/semantic-dr-gan/data/eophtha/",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--grade_inference_strategy",
        type=str,
        default="image",
    )
    parser.add_argument(
        "--grade_inference_model",
        type=str,
        default="eyepacs_transformed",
    )
    parser.add_argument(
        "--denylist", type=str, dest="denylist", default="data/fgadr_denylist.csv"
    )
    parser.set_defaults(predict_grades=True)

    return parser.parse_args()
