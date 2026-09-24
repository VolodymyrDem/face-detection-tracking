"""Webcam frame capture."""

from __future__ import annotations

from collections.abc import Iterator
from types import TracebackType

import cv2
import numpy as np


class Camera:
    """Context manager wrapping cv2.VideoCapture.

    If width/height are None, the camera's own default resolution is used.
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int | None = None,
        height: int | None = None,
        mirror: bool = True,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.mirror = mirror
        self._capture = None

    def __enter__(self) -> "Camera":
        self._capture = cv2.VideoCapture(self.source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Cannot open camera {self.source}")
        if self.width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames in BGR format."""
        if self._capture is None:
            raise RuntimeError("Camera is not opened")
        while True:
            ret, frame = self._capture.read()
            if not ret:
                break
            if self.mirror:
                frame = cv2.flip(frame, 1)
            yield frame
