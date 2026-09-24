"""Webcam frame capture."""

from __future__ import annotations

from collections.abc import Iterator
from types import TracebackType

import numpy as np


class Camera:
    """Context manager wrapping cv2.VideoCapture."""

    def __init__(self, source: int | str = 0, width: int = 640, height: int = 480) -> None:
        self.source = source
        self.width = width
        self.height = height
        self._capture = None

    def __enter__(self) -> "Camera":
        raise NotImplementedError("TODO: open cv2.VideoCapture and set the capture resolution")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        raise NotImplementedError("TODO: release the camera resources")

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames in BGR format."""
        raise NotImplementedError("TODO: read frames while the camera stays open")
