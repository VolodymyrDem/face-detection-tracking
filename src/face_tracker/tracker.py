"""Face tracking across frames."""

from __future__ import annotations

from dataclasses import dataclass, field

from .detector import Detection


@dataclass
class Track:
    """A stable trajectory for a single face."""

    track_id: int
    detection: Detection
    age: int = 0
    misses: int = 0
    history: list[Detection] = field(default_factory=list)


def iou(a: Detection, b: Detection) -> float:
    """Intersection over union of two bounding boxes."""
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    union = a.width * a.height + b.width * b.height - inter
    return inter / union


class IouTracker:
    """Simple tracker based on greedy IoU association."""

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 15) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self._tracks: list[Track] = []
        self._next_id = 0

    def update(self, detections: list[Detection]) -> list[Track]:
        """Update tracks with new detections and return the active ones."""
        raise NotImplementedError(
            "TODO: match detections to tracks by IoU, spawn new tracks, drop stale ones"
        )

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0
