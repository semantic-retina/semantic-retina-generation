import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--idrid_root_dir",
        type=str,
        default="/vol/vipdata/data/retina/IDRID/a_segmentation/",
        help="Path to the IDRiD dataset",
    )
    parser.add_argument(
        "--fgadr_root_dir",
        type=str,
        default="/vol/vipdata/data/retina/FGADR-Seg/Seg-set",
        help="Path to the FGADR dataset",
    )
    parser.add_argument(
        "--diaretdb1_root_dir",
        type=str,
        default="/vol/bitbucket/js6317/individual-project/data/diaretdb1_v_1_1/",
        help="Path to the DIARETDB1 dataset",
    )
    parser.add_argument(
        "--diaretdb1_annotation_file",
        type=str,
        default="data/diaretdb1_od.json",
        help="JSON file containing annotations for the DIARETDB1 dataset",
    )
    parser.add_argument(
        "--fgadr_annotation_file",
        type=str,
        default="data/total.json",
        help="JSON file containing annotations for the FGADR dataset",
    )
    parser.add_argument(
        "--eophtha_root_dir",
        type=str,
        default="/vol/bitbucket/js6317/individual-project/data/e_optha/",
        help="Path to the e-ophtha dataset",
    )
    parser.add_argument(
        "--eophtha_od_dir",
        type=str,
        default="data/eophtha/od",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers for concurrent processing",
    )
    parser.add_argument(
        "--colour",
        dest="colour",
        action="store_true",
        help="Whether to colour the labels",
    )
    parser.add_argument(
        "--nocolour",
        action="store_false",
        dest="colour",
    )
    parser.add_argument(
        "--fgadr",
        action="store_true",
        dest="fgadr",
    )
    parser.add_argument(
        "--nofgadr",
        action="store_false",
        dest="fgadr",
    )
    parser.add_argument(
        "--idrid",
        action="store_true",
        dest="idrid",
    )
    parser.add_argument(
        "--noidrid",
        action="store_false",
        dest="idrid",
    )
    parser.add_argument(
        "--diaretdb1",
        action="store_true",
        dest="diaretdb1",
    )
    parser.add_argument(
        "--nodiaretdb1",
        action="store_false",
        dest="diaretdb1",
    )
    parser.add_argument(
        "--eophtha",
        action="store_true",
        dest="eophtha",
    )
    parser.add_argument(
        "--noeophtha",
        action="store_false",
        dest="eophtha",
    )
    parser.set_defaults(
        colour=False, fgadr=True, idrid=True, diaretdb1=True, eophtha=True
    )

    return parser.parse_args()
