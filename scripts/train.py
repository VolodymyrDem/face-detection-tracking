"""Train the face detector on WIDER FACE.

Example:
    python scripts/train.py --config configs/train.yaml
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train face detector")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError("TODO: read the config, build the dataset and the training loop")


if __name__ == "__main__":
    main()
