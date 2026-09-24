import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from face_tracker import detector as detector_module
from face_tracker.detector import (
    YUNET_MODEL_PATH,
    Detection,
    YuNetFaceDetector,
    clip_detection,
    create_detector,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / YUNET_MODEL_PATH


class FakeYuNet:
    def __init__(self, rows):
        self.rows = rows
        self.input_sizes = []

    def setInputSize(self, size):
        self.input_sizes.append(size)

    def detect(self, image):
        return 1, self.rows


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    holder = {}

    def create(path, config, size, score, nms, top_k):
        holder["args"] = (score, nms)
        return holder["model"]

    model_file = tmp_path / "yunet.onnx"
    model_file.write_bytes(b"x")
    monkeypatch.setattr(cv2.FaceDetectorYN, "create", create)

    def build(rows, **kwargs):
        holder["model"] = FakeYuNet(None if rows is None else np.array(rows, dtype=np.float32))
        return YuNetFaceDetector(model_file, **kwargs), holder

    return build


def row(x, y, w, h, score):
    return [x, y, w, h] + [0] * 10 + [score]


def test_converts_rows_and_scales_back_to_frame(fake_model):
    detector, holder = fake_model([row(10, 20, 30, 40, 0.9)], input_width=640)
    (face,) = detector.detect(np.zeros((1080, 1920, 3), np.uint8))
    assert holder["model"].input_sizes == [(640, 360)]
    assert face == Detection(30, 60, 90, 120, pytest.approx(0.9))


def test_small_frames_are_not_resized(fake_model):
    detector, holder = fake_model([row(10, 20, 30, 40, 0.9)], input_width=640)
    (face,) = detector.detect(np.zeros((240, 320, 3), np.uint8))
    assert holder["model"].input_sizes == [(320, 240)]
    assert face.xyxy == (10, 20, 40, 60)


def test_input_size_is_set_only_when_frame_size_changes(fake_model):
    detector, holder = fake_model(None)
    for _ in range(3):
        detector.detect(np.zeros((480, 640, 3), np.uint8))
    assert holder["model"].input_sizes == [(640, 480)]


def test_no_faces_returns_empty_list(fake_model):
    detector, _ = fake_model(None)
    assert detector.detect(np.zeros((480, 640, 3), np.uint8)) == []


def test_thresholds_are_passed_to_opencv(fake_model):
    _, holder = fake_model(None, min_confidence=0.8, nms_threshold=0.2)
    assert holder["args"] == (0.8, 0.2)


def test_clip_detection():
    assert clip_detection(-10, -10, 50, 50, 0.5, 100, 100) == Detection(0, 0, 40, 40, 0.5)
    assert clip_detection(90, 90, 50, 50, 0.5, 100, 100) == Detection(90, 90, 10, 10, 0.5)
    assert clip_detection(200, 200, 50, 50, 0.5, 100, 100) is None


def test_unknown_detector_type_raises():
    with pytest.raises(ValueError):
        create_detector({"type": "nope"})


@pytest.mark.skipif(not MODEL.is_file(), reason="YuNet model not downloaded yet")
def test_real_model_finds_faces():
    import ultralytics

    image = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "zidane.jpg"))
    faces = YuNetFaceDetector(MODEL, min_confidence=0.8).detect(image)
    assert len(faces) == 2
    assert all(0.8 <= face.score <= 1 for face in faces)
