"""Download and extract the WIDER FACE and PASCAL Face datasets.

Example:
    python scripts/download_data.py --dataset wider_face --output data/raw
"""

from __future__ import annotations

import argparse

DATASETS = {
    "wider_face": "http://shuoyang1213.me/WIDERFACE/",
    "pascal_face": "http://host.robots.ox.ac.uk/pascal/VOC/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset downloader")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--output", default="data/raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"TODO: download {args.dataset} from {DATASETS[args.dataset]} into {args.output} "
        "and convert the annotations into the shared format (data/processed)"
    )


if __name__ == "__main__":
    main()
