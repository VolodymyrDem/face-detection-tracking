from pathlib import Path

import pytest
import yaml
from PIL import Image

from face_tracker.wider_face import (
    FaceBox,
    convert_split,
    find_wider_face,
    parse_annotations,
    to_yolo_lines,
    write_data_yaml,
)

ANNOTATIONS = """\
0--Parade/a.jpg
2
10 20 30 40 0 0 0 0 0 0
0 0 5 5 0 0 0 1 0 0
1--Handshaking/b.jpg
0
0 0 0 0 0 0 0 0 0 0
0--Parade/c.jpg
1
90 90 20 20 1 0 0 0 0 0
"""


def make_wider(root: Path, split: str = "train") -> tuple[Path, Path]:
    images = root / f"WIDER_{split}" / f"WIDER_{split}" / "images"
    for name in ("0--Parade/a.jpg", "1--Handshaking/b.jpg", "0--Parade/c.jpg"):
        (images / name).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100)).save(images / name)
    annotations = root / "wider_face_split" / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
    annotations.parent.mkdir(parents=True, exist_ok=True)
    annotations.write_text(ANNOTATIONS)
    return images, annotations


def test_parses_faces_and_invalid_flag(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text(ANNOTATIONS)
    items = dict(parse_annotations(path))
    assert list(items) == ["0--Parade/a.jpg", "1--Handshaking/b.jpg", "0--Parade/c.jpg"]
    assert items["0--Parade/a.jpg"] == [FaceBox(10, 20, 30, 40, False), FaceBox(0, 0, 5, 5, True)]


def test_image_without_faces_skips_placeholder_line(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text(ANNOTATIONS)
    items = dict(parse_annotations(path))
    assert items["1--Handshaking/b.jpg"] == []
    assert items["0--Parade/c.jpg"] == [FaceBox(90, 90, 20, 20, False)]


def test_yolo_lines_are_normalized_centers():
    (line,) = to_yolo_lines([FaceBox(10, 20, 30, 40)], 100, 200)
    assert line == "0 0.250000 0.200000 0.300000 0.200000"


def test_invalid_and_tiny_faces_are_dropped():
    boxes = [FaceBox(0, 0, 50, 50, invalid=True), FaceBox(0, 0, 1, 1), FaceBox(0, 0, 0, 10)]
    assert to_yolo_lines(boxes, 100, 100) == []


def test_boxes_are_clipped_to_image():
    (line,) = to_yolo_lines([FaceBox(90, 90, 20, 20)], 100, 100)
    assert [float(v) for v in line.split()[1:]] == pytest.approx([0.95, 0.95, 0.1, 0.1])


def test_convert_split_writes_labels_and_links_images(tmp_path):
    images, annotations = make_wider(tmp_path / "raw")
    out = tmp_path / "yolo"
    stats = convert_split(images, annotations, out, "train")

    assert (stats.images, stats.faces, stats.skipped_faces, stats.missing_images) == (3, 2, 1, 0)
    link = out / "images" / "train" / "0--Parade" / "a.jpg"
    assert link.is_symlink() and link.resolve() == (images / "0--Parade" / "a.jpg").resolve()
    assert (out / "labels" / "train" / "0--Parade" / "a.txt").read_text().count("\n") == 1
    assert (out / "labels" / "train" / "1--Handshaking" / "b.txt").read_text() == ""


def test_convert_split_is_rerunnable(tmp_path):
    images, annotations = make_wider(tmp_path / "raw")
    out = tmp_path / "yolo"
    convert_split(images, annotations, out, "train")
    assert convert_split(images, annotations, out, "train").images == 3


def test_missing_images_are_counted(tmp_path):
    images, annotations = make_wider(tmp_path / "raw")
    (images / "0--Parade" / "c.jpg").unlink()
    stats = convert_split(images, annotations, tmp_path / "yolo", "train")
    assert (stats.images, stats.missing_images) == (2, 1)


def test_data_yaml_points_to_splits(tmp_path):
    data = yaml.safe_load(write_data_yaml(tmp_path).read_text())
    assert data == {
        "path": str(tmp_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "face"},
    }


def test_find_wider_face_uses_kaggle_layout(tmp_path):
    for split in ("train", "val"):
        make_wider(tmp_path, split)
    found = find_wider_face(tmp_path)
    assert found["val"][0] == tmp_path / "WIDER_val" / "WIDER_val" / "images"
    assert found["train"][1].name == "wider_face_train_bbx_gt.txt"


def test_find_wider_face_reports_missing_split(tmp_path):
    make_wider(tmp_path, "train")
    with pytest.raises(FileNotFoundError, match="val"):
        find_wider_face(tmp_path)
