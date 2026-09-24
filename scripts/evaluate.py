"""Evaluate model quality and speed.

Example:
    python scripts/evaluate.py --model models/face_detector.onnx --dataset pascal_face
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate face detector")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=["wider_face", "pascal_face"], required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--benchmark-fps", action="store_true", help="Measure FPS on video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "TODO: compute AP@0.5 (easy/medium/hard for WIDER FACE), Precision/Recall and FPS"
    )


if __name__ == "__main__":
    main()
