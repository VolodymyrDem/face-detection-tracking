import numpy as np

from face_tracker.detector import Detection
from face_tracker.tracker import Track
from face_tracker.visualize import BAD_COLOR, BOX_COLOR, OK_COLOR, draw_fps, draw_tracks


def blank(width=640, height=480):
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_draws_box_on_face_edges_only():
    frame = blank()
    result = draw_tracks(frame, [Track(1, Detection(200, 200, 100, 100, 0.9))])
    assert result is frame
    assert tuple(frame[250, 200]) == BOX_COLOR
    assert tuple(frame[250, 250]) == (0, 0, 0)


def test_label_fits_when_face_touches_top():
    frame = blank()
    draw_tracks(frame, [Track(1, Detection(10, 0, 100, 100, 0.9))])
    assert frame[:, :, 1].sum() > 0


def test_no_tracks_leaves_frame_untouched():
    frame = blank()
    draw_tracks(frame, [])
    assert not frame.any()


def test_fps_color_shows_requirement():
    good, bad = blank(), blank()
    draw_fps(good, 30.0, 7.0)
    draw_fps(bad, 10.0)
    assert tuple(good[15, 15]) == OK_COLOR
    assert tuple(bad[15, 15]) == BAD_COLOR
