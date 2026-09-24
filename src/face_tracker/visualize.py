"""Rendering of detection and tracking results."""

from __future__ import annotations

import cv2
import numpy as np

from .metrics import MIN_REQUIRED_FPS
from .tracker import Track

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
OK_COLOR = (0, 200, 0)
BAD_COLOR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _scale(frame: np.ndarray) -> float:
    """Line/text scale so overlays look the same on 480p and 1080p frames."""
    return max(frame.shape[0], 1) / 720


def _label(frame: np.ndarray, text: str, x: int, y: int, background: tuple[int, int, int]) -> None:
    """Text with a filled background; (x, y) is the top-left corner of the label."""
    font_scale = 0.7 * _scale(frame)
    thickness = max(1, round(2 * _scale(frame)))
    (width, height), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    pad = max(2, round(4 * _scale(frame)))
    y = max(0, y)
    cv2.rectangle(frame, (x, y), (x + width + 2 * pad, y + height + baseline + 2 * pad), background, -1)
    cv2.putText(frame, text, (x + pad, y + pad + height), FONT, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    """Draw a bounding box and ID for every tracked face (in place)."""
    thickness = max(2, round(3 * _scale(frame)))
    for track in tracks:
        x1, y1, x2, y2 = track.detection.xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, thickness)
        label_height = round(34 * _scale(frame))
        label_y = y1 - label_height if y1 >= label_height else y2
        _label(frame, f"Face #{track.track_id} {track.detection.score:.2f}", x1, label_y, (0, 140, 0))
    return frame


def draw_fps(frame: np.ndarray, fps: float, detector_ms: float | None = None) -> np.ndarray:
    """Render FPS (green if it meets the requirement) and detector time in the corner (in place)."""
    text = f"FPS {fps:.1f}"
    if detector_ms is not None:
        text += f" | model {detector_ms:.1f} ms"
    color = OK_COLOR if fps >= MIN_REQUIRED_FPS else BAD_COLOR
    margin = round(10 * _scale(frame))
    _label(frame, text, margin, margin, color)
    return frame
