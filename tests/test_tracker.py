import pytest

from face_tracker.detector import Detection
from face_tracker.metrics import MIN_REQUIRED_FPS, FpsMeter
from face_tracker.tracker import iou


def test_detection_xyxy():
    det = Detection(x=10, y=20, width=30, height=40, score=0.9)
    assert det.xyxy == (10, 20, 40, 60)


def test_iou_identical_boxes():
    a = Detection(0, 0, 10, 10, 1.0)
    assert iou(a, a) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    a = Detection(0, 0, 10, 10, 1.0)
    b = Detection(100, 100, 10, 10, 1.0)
    assert iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = Detection(0, 0, 10, 10, 1.0)
    b = Detection(5, 0, 10, 10, 1.0)
    assert iou(a, b) == pytest.approx(50 / 150)


def test_fps_meter_starts_empty():
    meter = FpsMeter()
    assert meter.fps == 0.0
    assert not meter.meets_requirement


def test_min_required_fps():
    assert MIN_REQUIRED_FPS == 20.0
