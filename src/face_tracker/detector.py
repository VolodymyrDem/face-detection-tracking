"""Face detection on images."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

YUNET_MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


@dataclass(frozen=True)
class Detection:
    """A single detected face."""

    x: int
    y: int
    width: int
    height: int
    score: float

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height


class FaceDetector(Protocol):
    """Common interface shared by all detector implementations."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        ...


def ensure_model(path: str | Path, url: str) -> Path:
    """Download the model file on first use."""
    path = Path(path)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        print(f"Downloading {url} -> {path}")
        urllib.request.urlretrieve(url, partial)
        partial.rename(path)
    return path


def clip_detection(
    x: float, y: float, width: float, height: float, score: float, frame_width: int, frame_height: int
) -> Detection | None:
    """Clip a box to the frame; None if nothing of it is left inside."""
    x1, y1 = max(0, round(x)), max(0, round(y))
    x2, y2 = min(frame_width, round(x + width)), min(frame_height, round(y + height))
    if x2 <= x1 or y2 <= y1:
        return None
    return Detection(x1, y1, x2 - x1, y2 - y1, float(score))


class YuNetFaceDetector:
    """Stand-in detector: OpenCV YuNet, pretrained on WIDER FACE.

    Used to build the app until our own model is trained, and as the baseline to
    compare against. Frames wider than ``input_width`` are downscaled before
    detection (much faster) and the boxes are scaled back to the original frame.
    """

    def __init__(
        self,
        model_path: str | Path = YUNET_MODEL_PATH,
        min_confidence: float = 0.6,
        nms_threshold: float = 0.3,
        input_width: int = 640,
    ) -> None:
        path = ensure_model(model_path, YUNET_MODEL_URL)
        self.input_width = input_width
        self._input_size: tuple[int, int] | None = None
        self._model = cv2.FaceDetectorYN.create(
            str(path), "", (320, 320), min_confidence, nms_threshold, 5000
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        frame_height, frame_width = frame.shape[:2]
        scale = min(1.0, self.input_width / frame_width)
        if scale < 1.0:
            size = (round(frame_width * scale), round(frame_height * scale))
            image = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        else:
            size = (frame_width, frame_height)
            image = frame

        if self._input_size != size:
            self._model.setInputSize(size)
            self._input_size = size

        _, faces = self._model.detect(image)
        if faces is None:
            return []

        detections = []
        for row in faces:
            x, y, width, height = (float(v) / scale for v in row[:4])
            detection = clip_detection(x, y, width, height, row[-1], frame_width, frame_height)
            if detection is not None:
                detections.append(detection)
        return detections


def create_detector(config: dict[str, Any]) -> FaceDetector:
    """Build a detector from the ``detector`` section of a config."""
    settings = dict(config)
    kind = settings.pop("type", "yunet")
    if kind == "yunet":
        return YuNetFaceDetector(**settings)
    raise ValueError(f"Unknown detector type: {kind!r}")
