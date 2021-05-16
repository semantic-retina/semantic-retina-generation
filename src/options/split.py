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
        default="random",
    )
    parser.add_argument(
        "--grade_inference_model",
        type=str,
        default="",
    )
    parser.add_argument(
        "--exclude_grader_1",
        dest="exclude_grader_1",
        action="store_true",
    )
    parser.add_argument(
        "--noexclude_grader_1",
        action="store_false",
        dest="exclude_grader_1",
    )
    parser.set_defaults(exclude_grader_1=False, predict_grades=True)

    return parser.parse_args()
