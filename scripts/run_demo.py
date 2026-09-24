"""Demo application: webcam feed with a box, ID and score around every face, plus FPS.

Run from the project root:
    python scripts/run_demo.py
    python scripts/run_demo.py --config configs/demo.yaml
"""

from __future__ import annotations

import argparse

from face_tracker.camera import Camera
from face_tracker.config import load_config
from face_tracker.detector import create_detector
from face_tracker.pipeline import FaceTrackingPipeline
from face_tracker.tracker import IouTracker
from face_tracker.window import run_video_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face detection & tracking demo")
    parser.add_argument("--config", default="configs/demo.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_config(parse_args().config)
    tracking = dict(config.get("tracking", {}))
    detect_every = tracking.pop("detect_every_n_frames", 1)

    pipeline = FaceTrackingPipeline(
        create_detector(config.get("detector", {})),
        IouTracker(**tracking),
        detect_every_n_frames=detect_every,
    )
    camera = Camera(**config.get("camera", {}))
    run_video_app(camera, "Face tracking (q / Esc to quit)", process=pipeline.annotate)


if __name__ == "__main__":
    main()
