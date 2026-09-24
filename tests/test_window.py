import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from face_tracker import window as window_module
from face_tracker.window import VideoWindow, fit_rect


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def locks(monkeypatch):
    calls = []
    monkeypatch.setattr(window_module, "lock_aspect_ratio", lambda w, width, height: calls.append((width, height)))
    return calls


def frame(width, height):
    return np.zeros((height, width, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    ("frame_size", "area", "expected"),
    [
        ((1920, 1080), (1920, 1080), (0, 0, 1920, 1080)),
        ((1920, 1080), (960, 540), (0, 0, 960, 540)),
        ((1920, 1080), (1000, 1000), (0, 219, 1000, 562)),
        ((640, 480), (1000, 480), (180, 0, 640, 480)),
    ],
)
def test_fit_rect_keeps_whole_frame_centered(frame_size, area, expected):
    rect = fit_rect(*frame_size, *area)
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == expected


def test_first_frame_sizes_window_to_frame_ratio(app, locks):
    win = VideoWindow("t")
    win.set_frame(frame(1920, 1080))
    assert locks == [(1920, 1080)]
    assert win.width() / win.height() == pytest.approx(1920 / 1080, abs=0.01)


def test_same_size_frames_do_not_relock(app, locks):
    win = VideoWindow("t")
    win.set_frame(frame(1280, 720))
    win.set_frame(frame(1280, 720))
    assert locks == [(1280, 720)]


def test_new_frame_size_relocks(app, locks):
    win = VideoWindow("t")
    win.set_frame(frame(1280, 720))
    win.set_frame(frame(640, 480))
    assert locks == [(1280, 720), (640, 480)]


def test_paints_without_errors(app, locks):
    win = VideoWindow("t")
    win.set_frame(np.full((480, 640, 3), 200, dtype=np.uint8))
    image = win.grab().toImage()
    center = image.pixelColor(image.width() // 2, image.height() // 2)
    assert (center.red(), center.green(), center.blue()) == (200, 200, 200)


@pytest.mark.parametrize("key", [Qt.Key.Key_Q, Qt.Key.Key_Escape])
def test_quit_keys_close_window(app, locks, key):
    win = VideoWindow("t")
    closed = []
    win.closed.connect(lambda: closed.append(True))
    win.show()
    win.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))
    assert closed == [True]
    assert not win.isVisible()
