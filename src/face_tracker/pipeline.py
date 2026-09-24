"""Full pipeline: frame -> detection -> tracking -> visualization."""

from __future__ import annotations

import time

import numpy as np

from .detector import FaceDetector
from .metrics import FpsMeter
from .tracker import IouTracker, Track
from .visualize import draw_fps, draw_tracks


class FaceTrackingPipeline:
    """Combines the detector and the tracker into a single per-frame call.

    With ``detect_every_n_frames > 1`` the detector runs only on every N-th frame and
    the last known boxes are reused in between (faster, slightly laggier boxes).
    """

    def __init__(
        self,
        detector: FaceDetector,
        tracker: IouTracker | None = None,
        detect_every_n_frames: int = 1,
    ) -> None:
        if detect_every_n_frames < 1:
            raise ValueError("detect_every_n_frames must be >= 1")
        self.detector = detector
        self.tracker = tracker or IouTracker()
        self.detect_every_n_frames = detect_every_n_frames
        self.fps_meter = FpsMeter()
        self.detector_ms: float | None = None
        self._tracks: list[Track] = []
        self._frame_index = 0

    def process(self, frame: np.ndarray) -> list[Track]:
        """Process a single frame and return the tracks to display."""
        if self._frame_index % self.detect_every_n_frames == 0:
            start = time.perf_counter()
            detections = self.detector.detect(frame)
            self.detector_ms = (time.perf_counter() - start) * 1000
            self._tracks = self.tracker.update(detections)
        self._frame_index += 1
        self.fps_meter.tick()
        return self._tracks

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame and draw boxes, IDs and FPS on it."""
        tracks = self.process(frame)
        draw_tracks(frame, tracks)
        draw_fps(frame, self.fps_meter.fps, self.detector_ms)
        return frame
