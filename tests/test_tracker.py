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


from face_tracker.tracker import IouTracker, smooth  # noqa: E402


def det(x, y, size=100, score=0.9):
    return Detection(x, y, size, size, score)


def test_new_faces_get_increasing_ids():
    tracks = IouTracker().update([det(0, 0), det(300, 0)])
    assert [t.track_id for t in tracks] == [1, 2]


def test_moving_face_keeps_its_id():
    tracker = IouTracker(smoothing=0)
    (first,) = tracker.update([det(0, 0)])
    (moved,) = tracker.update([det(10, 5)])
    assert moved.track_id == first.track_id
    assert moved.detection == det(10, 5)


def test_ids_follow_faces_not_list_order():
    tracker = IouTracker(smoothing=0)
    tracker.update([det(0, 0), det(300, 0)])
    tracks = tracker.update([det(305, 0), det(5, 0)])
    ids = {t.detection.x: t.track_id for t in tracks}
    assert ids == {5: 1, 305: 2}


def test_far_jump_starts_new_track():
    tracker = IouTracker()
    tracker.update([det(0, 0)])
    tracks = tracker.update([det(500, 500)])
    assert {t.track_id: t.misses for t in tracks} == {1: 1, 2: 0}


def test_briefly_missed_face_stays_visible_then_hides():
    tracker = IouTracker(keep_visible=2, max_misses=5)
    tracker.update([det(0, 0)])
    assert len(tracker.update([])) == 1
    assert len(tracker.update([])) == 1
    assert tracker.update([]) == []
    assert len(tracker.tracks) == 1


def test_face_back_within_max_misses_keeps_id():
    tracker = IouTracker(keep_visible=0, max_misses=5)
    tracker.update([det(0, 0)])
    for _ in range(4):
        tracker.update([])
    (track,) = tracker.update([det(0, 0)])
    assert track.track_id == 1


def test_track_is_dropped_after_max_misses():
    tracker = IouTracker(max_misses=2)
    tracker.update([det(0, 0)])
    for _ in range(3):
        tracker.update([])
    assert tracker.tracks == []
    (track,) = tracker.update([det(0, 0)])
    assert track.track_id == 2


def test_smoothing_blends_boxes():
    assert smooth(det(0, 0), det(100, 100), 0.5) == Detection(50, 50, 100, 100, 0.9)
    assert smooth(det(0, 0), det(100, 100), 0.0) == det(100, 100)


def test_reset_clears_tracks_and_ids():
    tracker = IouTracker()
    tracker.update([det(0, 0)])
    tracker.reset()
    assert tracker.tracks == []
    assert tracker.update([det(0, 0)])[0].track_id == 1
