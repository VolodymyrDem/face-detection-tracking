"""Face detection on images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


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


class MediaPipeFaceDetector:
    """Baseline detector built on MediaPipe Face Detector."""

    def __init__(self, min_confidence: float = 0.5, model_selection: int = 0) -> None:
        self.min_confidence = min_confidence
        self.model_selection = model_selection
        raise NotImplementedError("TODO: initialize mediapipe.solutions.face_detection")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError("TODO: run the frame through MediaPipe and collect Detection objects")


class OnnxFaceDetector:
    """Custom trained model exported to ONNX for fast inference."""

    def __init__(self, model_path: str, input_size: tuple[int, int] = (320, 320)) -> None:
        self.model_path = model_path
        self.input_size = input_size
        raise NotImplementedError("TODO: load the model via onnxruntime.InferenceSession")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError("TODO: preprocess, run inference, apply NMS, postprocess")
