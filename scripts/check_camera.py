"""Show the webcam feed to check that the camera works.

Run from the project root:
    python scripts/check_camera.py
"""

from face_tracker.camera import Camera
from face_tracker.config import load_config
from face_tracker.window import run_video_app

camera_cfg = load_config("configs/demo.yaml")["camera"]
camera = Camera(**camera_cfg)

printed = False


def report_resolution(frame):
    global printed
    if not printed:
        height, width = frame.shape[:2]
        print(f"Camera resolution: {width}x{height}")
        printed = True
    return frame


run_video_app(camera, "Camera check (q / Esc to quit)", process=report_resolution)
