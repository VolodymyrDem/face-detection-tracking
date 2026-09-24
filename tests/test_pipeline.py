import numpy as np
import pytest

from face_tracker.detector import Detection
from face_tracker.pipeline import FaceTrackingPipeline


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [Detection(10, 10, 50, 50, 0.9)]


def frame():
    return np.zeros((240, 320, 3), dtype=np.uint8)


def test_process_returns_tracked_faces():
    pipeline = FaceTrackingPipeline(FakeDetector())
    (track,) = pipeline.process(frame())
    assert track.track_id == 1
    assert pipeline.detector_ms is not None


def test_detects_every_n_frames_and_reuses_tracks():
    detector = FakeDetector()
    pipeline = FaceTrackingPipeline(detector, detect_every_n_frames=3)
    results = [pipeline.process(frame()) for _ in range(7)]
    assert detector.calls == 3
    assert all(len(tracks) == 1 for tracks in results)


def test_annotate_draws_on_frame():
    pipeline = FaceTrackingPipeline(FakeDetector())
    image = frame()
    assert pipeline.annotate(image) is image
    assert image.any()


def test_rejects_invalid_detect_interval():
    with pytest.raises(ValueError):
        FaceTrackingPipeline(FakeDetector(), detect_every_n_frames=0)
