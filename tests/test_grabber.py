import threading

import cv2
import numpy as np
import pytest

from face_tracker.camera import Camera
from face_tracker.grabber import FrameGrabber


class FakeCapture:
    def __init__(self, source, num_frames=5, fail_after=None):
        self._frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(num_frames)]
        self._fail_after = fail_after
        self._reads = 0

    def isOpened(self):
        return True

    def set(self, prop, value):
        return True

    def read(self):
        self._reads += 1
        if self._fail_after is not None and self._reads > self._fail_after:
            raise OSError("camera unplugged")
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self):
        pass


def run(grabber):
    grabber.start()
    grabber.join(timeout=2)
    assert not grabber.running


def test_delivers_all_frames_in_order(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    received = []
    with Camera(0, mirror=False) as cam:
        run(FrameGrabber(cam, on_frame=lambda f: received.append(int(f[0, 0, 0]))))
    assert received == [0, 1, 2, 3, 4]


def test_runs_processor_before_callback(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    received = []
    with Camera(0, mirror=False) as cam:
        run(FrameGrabber(cam, on_frame=lambda f: received.append(int(f[0, 0, 0])), process=lambda f: f + 10))
    assert received == [10, 11, 12, 13, 14]


def test_reads_in_background_thread(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    threads = set()
    with Camera(0) as cam:
        run(FrameGrabber(cam, on_frame=lambda f: threads.add(threading.current_thread())))
    assert threading.main_thread() not in threads


def test_stop_ends_stream_early(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    received = []
    with Camera(0) as cam:
        grabber = FrameGrabber(cam, on_frame=lambda f: (received.append(f), grabber.stop()))
        run(grabber)
    assert len(received) == 1


def test_reports_stop_and_error(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", lambda s: FakeCapture(s, fail_after=2))
    stops = []
    with Camera(0) as cam:
        grabber = FrameGrabber(cam, on_frame=lambda f: None, on_stop=stops.append)
        run(grabber)
    assert isinstance(grabber.error, OSError)
    assert stops == [grabber.error]


def test_reports_clean_stop(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    stops = []
    with Camera(0) as cam:
        grabber = FrameGrabber(cam, on_frame=lambda f: None, on_stop=stops.append)
        run(grabber)
    assert grabber.error is None
    assert stops == [None]
