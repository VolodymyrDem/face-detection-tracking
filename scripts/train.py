"""Train the YOLO face detector.

Examples:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml --set epochs=1 fraction=0.05   # smoke test
    python scripts/train.py --resume /kaggle/working/runs/yolo26n_wider_face/weights/last.pt
"""

from __future__ import annotations

import argparse

import yaml
from ultralytics import YOLO

from face_tracker.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train face detector")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume", default=None, help="Continue from a last.pt checkpoint")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override train settings from the config, e.g. --set epochs=1 batch=16",
    )
    return parser.parse_args()


def parse_overrides(pairs: list[str]) -> dict:
    overrides = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"Expected KEY=VALUE, got {pair!r}")
        overrides[key] = yaml.safe_load(value)
    return overrides


def pick_device():
    """All visible GPUs (Kaggle offers 2x T4), else Apple MPS, else CPU."""
    import torch

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        return list(range(count)) if count > 1 else 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()

    if args.resume:
        YOLO(args.resume).train(resume=True)
        return

    config = load_config(args.config)
    settings = {**config["train"], **parse_overrides(args.set)}
    settings.setdefault("device", pick_device())
    print(f"Model: {config['model']}\nSettings: {settings}")
    YOLO(config["model"]).train(**settings)


if __name__ == "__main__":
    main()
