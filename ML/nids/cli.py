"""Command line entry point:  python -m nids <command>"""
import argparse

from . import evaluate, predict, prepare, train
from .config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nids", description="CICIDS network intrusion detection pipeline")
    ap.add_argument("--config", default=None, help="path to config.yaml (default: project config.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="clean raw CSVs, split, select features")
    sub.add_parser("train", help="train detector + attack classifier")
    sub.add_parser("evaluate", help="evaluate on the held-out test split")
    sub.add_parser("run-all", help="prepare -> train -> evaluate")
    pr = sub.add_parser("predict", help="classify a new flow CSV")
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", default="predictions.csv")
    pr.add_argument("--summary", default=None, help="write a JSON summary (counts, warnings)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.cmd == "prepare":
        prepare.run(cfg)
    elif args.cmd == "train":
        train.run(cfg)
    elif args.cmd == "evaluate":
        evaluate.run(cfg)
    elif args.cmd == "run-all":
        prepare.run(cfg)
        train.run(cfg)
        evaluate.run(cfg)
    elif args.cmd == "predict":
        predict.run(cfg, args.input, args.output, summary_path=args.summary)


if __name__ == "__main__":
    main()
