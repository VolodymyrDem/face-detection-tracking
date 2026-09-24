import cv2
import numpy as np
import pytest

from face_tracker.camera import Camera


class FakeCapture:
    """Stands in for cv2.VideoCapture so tests don't need a real webcam."""

    instances: list["FakeCapture"] = []

    def __init__(self, source, num_frames=3, opened=True):
        self.source = source
        self._frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(num_frames)]
        self._opened = opened
        self.released = False
        self.props = {}
        FakeCapture.instances.append(self)

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self):
        self.released = True


@pytest.fixture
def fake_capture(monkeypatch):
    FakeCapture.instances.clear()
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    return FakeCapture


@pytest.fixture
def broken_capture(monkeypatch):
    FakeCapture.instances.clear()
    monkeypatch.setattr(cv2, "VideoCapture", lambda source: FakeCapture(source, opened=False))
    return FakeCapture


def test_enter_returns_camera(fake_capture):
    with Camera(0) as cam:
        assert isinstance(cam, Camera)


def test_opens_requested_source(fake_capture):
    with Camera(1):
        pass
    assert fake_capture.instances[0].source == 1


def test_sets_resolution(fake_capture):
    with Camera(0, width=320, height=240):
        props = fake_capture.instances[0].props
        assert props[cv2.CAP_PROP_FRAME_WIDTH] == 320
        assert props[cv2.CAP_PROP_FRAME_HEIGHT] == 240


def test_keeps_native_resolution_by_default(fake_capture):
    with Camera(0):
        assert fake_capture.instances[0].props == {}


def test_frames_yields_every_frame_then_stops(fake_capture):
    with Camera(0) as cam:
        frames = list(cam.frames())
    assert len(frames) == 3
    assert all(frame.shape == (480, 640, 3) for frame in frames)


def test_releases_camera_on_exit(fake_capture):
    with Camera(0):
        pass
    assert fake_capture.instances[0].released


def test_releases_camera_even_on_error(fake_capture):
    with pytest.raises(ValueError):
        with Camera(0):
            raise ValueError("boom")
    assert fake_capture.instances[0].released


def test_raises_if_camera_cannot_be_opened(broken_capture):
    with pytest.raises(RuntimeError) as error:
        with Camera(0):
            pass
    assert not isinstance(error.value, NotImplementedError)


def test_frames_without_opening_raises():
    with pytest.raises(RuntimeError) as error:
        next(Camera(0).frames())
    assert not isinstance(error.value, NotImplementedError)


def _single_frame_capture(frame):
    class SingleFrameCapture(FakeCapture):
        def __init__(self, source):
            super().__init__(source, num_frames=0)
            self._frames = [frame.copy()]

    return SingleFrameCapture


def _frame_with_white_left_column():
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[:, 0] = 255
    return frame


def test_mirror_is_enabled_by_default(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", _single_frame_capture(_frame_with_white_left_column()))
    with Camera(0) as cam:
        (frame,) = list(cam.frames())
    assert (frame[:, -1] == 255).all()
    assert (frame[:, 0] == 0).all()


def test_mirror_can_be_disabled(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", _single_frame_capture(_frame_with_white_left_column()))
    with Camera(0, mirror=False) as cam:
        (frame,) = list(cam.frames())
    assert (frame[:, 0] == 255).all()
    assert (frame[:, -1] == 0).all()
