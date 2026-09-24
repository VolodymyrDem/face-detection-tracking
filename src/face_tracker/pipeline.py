"""Full pipeline: frame -> detection -> tracking -> visualization."""

from __future__ import annotations

import numpy as np

from .detector import FaceDetector
from .metrics import FpsMeter
from .tracker import IouTracker, Track


class FaceTrackingPipeline:
    """Combines the detector and the tracker into a single per-frame call."""

    def __init__(
        self,
        detector: FaceDetector,
        tracker: IouTracker | None = None,
        detect_every_n_frames: int = 1,
    ) -> None:
        self.detector = detector
        self.tracker = tracker or IouTracker()
        self.detect_every_n_frames = detect_every_n_frames
        self.fps_meter = FpsMeter()
        self._frame_index = 0

    def process(self, frame: np.ndarray) -> list[Track]:
        """Process a single frame and return the active tracks."""
        raise NotImplementedError(
            "TODO: run the detector every N frames and update only the tracker in between"
        )
