"""Face localization and tracking on webcam images."""

from .detector import Detection, FaceDetector
from .metrics import MIN_REQUIRED_FPS, FpsMeter
from .pipeline import FaceTrackingPipeline
from .tracker import IouTracker, Track

__all__ = [
    "MIN_REQUIRED_FPS",
    "Detection",
    "FaceDetector",
    "FaceTrackingPipeline",
    "FpsMeter",
    "IouTracker",
    "Track",
]
__version__ = "0.1.0"
