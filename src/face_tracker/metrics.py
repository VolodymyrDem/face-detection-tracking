"""Quality and performance metrics."""

from __future__ import annotations

import time
from collections import deque

MIN_REQUIRED_FPS = 20.0


class FpsMeter:
    """Rolling average FPS over the last N frames."""

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    @property
    def meets_requirement(self) -> bool:
        return self.fps >= MIN_REQUIRED_FPS


def mean_average_precision(predictions, ground_truth, iou_threshold: float = 0.5) -> float:
    """mAP@IoU across a set of images."""
    raise NotImplementedError("TODO: implement mAP computation for WIDER FACE / PASCAL Face")
