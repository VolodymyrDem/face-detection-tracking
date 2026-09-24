"""Demo application: display the webcam feed with bounding boxes around faces.

Example:
    python scripts/run_demo.py --camera 0 --detector mediapipe
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face detection & tracking demo")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument(
        "--detector",
        choices=["mediapipe", "onnx"],
        default="mediapipe",
        help="Which detector implementation to use",
    )
    parser.add_argument("--model", type=str, default="models/face_detector.onnx")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--show-fps", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "TODO: wire up Camera + FaceTrackingPipeline + draw_tracks and show a cv2.imshow window"
    )


if __name__ == "__main__":
    main()
