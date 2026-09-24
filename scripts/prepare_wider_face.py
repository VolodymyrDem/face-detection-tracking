"""Convert WIDER FACE into the YOLO format used for training.

Kaggle (dataset mksaad/wider-face-a-face-detection-benchmark attached):
    python scripts/prepare_wider_face.py --root /kaggle/input --output /kaggle/working/wider_yolo

Local:
    python scripts/prepare_wider_face.py --root data/raw/wider_face --output data/processed/wider_yolo
"""

from __future__ import annotations

import argparse

from face_tracker.wider_face import convert_split, find_wider_face, write_data_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WIDER FACE -> YOLO converter")
    parser.add_argument("--root", required=True, help="Folder that contains the WIDER FACE files")
    parser.add_argument("--output", required=True, help="Where to write the YOLO dataset")
    parser.add_argument(
        "--min-size", type=float, default=2.0, help="Drop faces smaller than this (pixels)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split, (images_dir, annotations) in find_wider_face(args.root).items():
        print(f"[{split}] images: {images_dir}\n[{split}] labels: {annotations}")
        stats = convert_split(images_dir, annotations, args.output, split, args.min_size)
        print(
            f"[{split}] {stats.images} images, {stats.faces} faces "
            f"({stats.skipped_faces} invalid/tiny skipped, {stats.missing_images} images missing)"
        )
    print(f"Dataset config: {write_data_yaml(args.output)}")


if __name__ == "__main__":
    main()
