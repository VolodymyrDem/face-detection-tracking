"""Conversion of WIDER FACE annotations to the YOLO format used by ultralytics.

WIDER FACE annotation file (``wider_face_{train,val}_bbx_gt.txt``) layout::

    0--Parade/0_Parade_marchingband_1_849.jpg     <- image path relative to images/
    1                                             <- number of faces
    449 330 122 149 0 0 0 0 0 0                   <- x1 y1 w h blur expression illumination invalid occlusion pose

Images without faces have a count of 0 followed by one line of zeros.

YOLO label file (one per image, same name with .txt)::

    0 cx cy w h                                   <- class 0 = face, values normalized to [0, 1]
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

SPLITS = ("train", "val")
CLASS_NAMES = {0: "face"}


@dataclass(frozen=True)
class FaceBox:
    x: float
    y: float
    width: float
    height: float
    invalid: bool = False


def parse_annotations(path: str | Path) -> Iterator[tuple[str, list[FaceBox]]]:
    """Yield (relative image path, faces) for every image in a WIDER FACE annotation file."""
    with open(path, encoding="utf-8") as file:
        lines = (line.strip() for line in file)
        lines = iter([line for line in lines if line])

    for image_path in lines:
        count = int(next(lines))
        boxes = []
        # Images without faces still have one placeholder line of zeros.
        for _ in range(max(count, 1)):
            values = next(lines).split()
            if count == 0:
                continue
            x, y, w, h = (float(v) for v in values[:4])
            invalid = len(values) > 7 and values[7] == "1"
            boxes.append(FaceBox(x, y, w, h, invalid))
        yield image_path, boxes


def to_yolo_lines(
    boxes: list[FaceBox], image_width: int, image_height: int, min_size: float = 2.0
) -> list[str]:
    """Convert pixel boxes to normalized YOLO lines, dropping invalid and tiny faces."""
    lines = []
    for box in boxes:
        if box.invalid:
            continue
        x1 = min(max(box.x, 0.0), image_width)
        y1 = min(max(box.y, 0.0), image_height)
        x2 = min(max(box.x + box.width, 0.0), image_width)
        y2 = min(max(box.y + box.height, 0.0), image_height)
        width, height = x2 - x1, y2 - y1
        if width < min_size or height < min_size:
            continue
        cx = (x1 + width / 2) / image_width
        cy = (y1 + height / 2) / image_height
        lines.append(f"0 {cx:.6f} {cy:.6f} {width / image_width:.6f} {height / image_height:.6f}")
    return lines


@dataclass
class SplitStats:
    images: int = 0
    faces: int = 0
    skipped_faces: int = 0
    missing_images: int = 0


def convert_split(
    images_dir: str | Path,
    annotations: str | Path,
    output_dir: str | Path,
    split: str,
    min_size: float = 2.0,
) -> SplitStats:
    """Write YOLO labels for one split and symlink its images next to them.

    ultralytics finds a label by replacing ``/images/`` with ``/labels/`` in the image
    path, so images are symlinked (not copied) into ``output_dir/images/<split>``.
    """
    images_dir = Path(images_dir).resolve()
    output_dir = Path(output_dir)
    stats = SplitStats()

    for relative_path, boxes in parse_annotations(annotations):
        source = images_dir / relative_path
        if not source.is_file():
            stats.missing_images += 1
            continue
        with Image.open(source) as image:
            width, height = image.size

        lines = to_yolo_lines(boxes, width, height, min_size)
        stats.images += 1
        stats.faces += len(lines)
        stats.skipped_faces += len(boxes) - len(lines)

        image_link = output_dir / "images" / split / relative_path
        label_file = (output_dir / "labels" / split / relative_path).with_suffix(".txt")
        image_link.parent.mkdir(parents=True, exist_ok=True)
        label_file.parent.mkdir(parents=True, exist_ok=True)
        if image_link.is_symlink() or image_link.exists():
            image_link.unlink()
        os.symlink(source, image_link)
        label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return stats


def write_data_yaml(output_dir: str | Path) -> Path:
    """Write the ultralytics dataset description for the converted dataset."""
    output_dir = Path(output_dir).resolve()
    path = output_dir / "data.yaml"
    names = "\n".join(f"  {index}: {name}" for index, name in CLASS_NAMES.items())
    path.write_text(
        f"path: {output_dir}\ntrain: images/train\nval: images/val\nnames:\n{names}\n",
        encoding="utf-8",
    )
    return path


def find_wider_face(root: str | Path) -> dict[str, tuple[Path, Path]]:
    """Locate images/ dirs and annotation files for each split under ``root``."""
    root = Path(root)
    found = {}
    for split in SPLITS:
        annotations = sorted(root.rglob(f"wider_face_{split}_bbx_gt.txt"))
        images = sorted(
            path for path in root.rglob("images")
            if path.is_dir() and path.parent.name == f"WIDER_{split}"
        )
        if not annotations or not images:
            raise FileNotFoundError(f"WIDER FACE {split} split not found under {root}")
        found[split] = (images[0], annotations[0])
    return found
