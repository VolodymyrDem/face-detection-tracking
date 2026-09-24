"""Face tracking across frames."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .detector import Detection

HISTORY_LENGTH = 30


@dataclass
class Track:
    """A stable trajectory for a single face."""

    track_id: int
    detection: Detection
    age: int = 0
    hits: int = 1
    misses: int = 0
    history: deque[Detection] = field(default_factory=lambda: deque(maxlen=HISTORY_LENGTH))


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


def smooth(previous: Detection, current: Detection, factor: float) -> Detection:
    """Blend the new box with the previous one to reduce jitter (factor 0 = no smoothing)."""
    if factor <= 0:
        return current

    def blend(old: int, new: int) -> int:
        return round(factor * old + (1 - factor) * new)

    return Detection(
        blend(previous.x, current.x),
        blend(previous.y, current.y),
        blend(previous.width, current.width),
        blend(previous.height, current.height),
        current.score,
    )


class IouTracker:
    """Simple tracker based on greedy IoU association.

    A track survives up to ``max_misses`` frames without a matching detection (so a
    face that is missed for a moment keeps its ID) and stays visible for the first
    ``keep_visible`` of them (so boxes do not flicker).
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_misses: int = 15,
        keep_visible: int = 3,
        smoothing: float = 0.4,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.keep_visible = keep_visible
        self.smoothing = smoothing
        self._tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        """Update tracks with new detections and return the ones to display."""
        pairs = sorted(
            (
                (iou(track.detection, detection), t, d)
                for t, track in enumerate(self._tracks)
                for d, detection in enumerate(detections)
            ),
            reverse=True,
        )
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for overlap, t, d in pairs:
            if overlap < self.iou_threshold:
                break
            if t in matched_tracks or d in matched_detections:
                continue
            matched_tracks.add(t)
            matched_detections.add(d)
            track = self._tracks[t]
            track.detection = smooth(track.detection, detections[d], self.smoothing)
            track.history.append(track.detection)
            track.hits += 1
            track.misses = 0

        for t, track in enumerate(self._tracks):
            track.age += 1
            if t not in matched_tracks:
                track.misses += 1
        self._tracks = [track for track in self._tracks if track.misses <= self.max_misses]

        for d, detection in enumerate(detections):
            if d not in matched_detections:
                track = Track(self._next_id, detection)
                track.history.append(detection)
                self._tracks.append(track)
                self._next_id += 1

        return [track for track in self._tracks if track.misses <= self.keep_visible]

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
