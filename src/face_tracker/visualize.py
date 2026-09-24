"""Rendering of detection and tracking results."""

from __future__ import annotations

import numpy as np

from .tracker import Track

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    """Draw a bounding box and ID for every tracked face."""
    raise NotImplementedError("TODO: cv2.rectangle + cv2.putText for each track")


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Render the current FPS value in the corner of the frame."""
    raise NotImplementedError("TODO: draw the FPS value")
