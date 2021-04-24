import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="",
                        help="config yaml path")
    parser.add_argument("--load", type=str, default="",
                        help="path to model weight")
    parser.add_argument("--mode", type=str, default="train",
                        help="model running mode (train/valid/test)")
    parser.add_argument("--reset", action="store_true",
                        help="reset epoch")
    parser.add_argument("--clear", action="store_true",
                        help="clear best metric")
    parser.add_argument("opts", default=None,
                        help="Modify config options using the command-line",
                        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    return args
