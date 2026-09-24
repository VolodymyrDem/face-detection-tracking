# Face Localization and Tracking

> Task #1

## Problem Statement

Train a model that locates and tracks the region containing a human face (if present)
in images captured from a webcam.

Build a test application that displays the webcam feed on screen and draws a bounding
box around every face present in the frame.

**Performance requirement:** the model must run at **at least 20 FPS**.

Expected result example:
https://developers.google.com/static/mediapipe/images/solutions/examples/face_detector.png

## Datasets

| Dataset | Description | Link |
|---|---|---|
| **WIDER FACE** | 32,203 images with 393,703 labeled faces; high variability in scale, pose, lighting and occlusion | http://shuoyang1213.me/WIDERFACE/ |
| **PASCAL Face** | 851 images from the PASCAL VOC subset with 1,335 faces; used as an evaluation benchmark | http://host.robots.ox.ac.uk/pascal/VOC/ |

## Repository Layout

```
.
├── configs/                # YAML configs for training and inference
├── data/                   # Datasets (not versioned)
│   ├── raw/                # Downloaded WIDER FACE / PASCAL Face archives
│   └── processed/          # Prepared annotations and images
├── docs/                   # Documentation, reports, metrics
├── models/                 # Model weights (not versioned)
├── notebooks/              # Research notebooks (EDA, visualization)
├── scripts/                # CLI scripts: data download, training, evaluation
├── src/face_tracker/       # Main package
│   ├── detector.py         # Face detector
│   ├── tracker.py          # Tracker (detection association across frames)
│   ├── pipeline.py         # Detection + tracking pipeline
│   ├── camera.py           # Webcam video capture (native resolution, optional mirroring)
│   ├── config.py           # YAML config loading
│   ├── grabber.py          # Background thread that reads (and processes) camera frames
│   ├── window.py           # Qt video window locked to the camera aspect ratio
│   ├── visualize.py        # Bounding box and label rendering
│   └── metrics.py          # FPS, mAP, IoU computation
└── tests/                  # Unit tests
```

## Roadmap

1. **Data** — download and convert WIDER FACE / PASCAL Face into a unified annotation format.
2. **Baseline** — baseline detector (MediaPipe / SSD / YOLO-face) to measure the accuracy and speed ceiling.
3. **Training** — train a custom model on WIDER FACE train, validate on WIDER FACE val, test on PASCAL Face.
4. **Tracking** — add cross-frame detection association (IoU tracker / SORT) for stable IDs.
5. **Demo app** — render the webcam feed with real-time bounding boxes around faces.
6. **Optimization** — reach ≥20 FPS (quantization, ONNX Runtime, reduced input resolution).

## Metrics

- **Detection quality:** AP@0.5 (IoU ≥ 0.5), Precision and Recall.
  - WIDER FACE validation set, reported separately for the official easy / medium / hard subsets.
  - PASCAL Face as an independent test set (not used for training).
- **Performance (required):** average and minimum FPS of the full pipeline
  (capture → detection → tracking → rendering) on the target hardware. Must be **≥20 FPS**,
  i.e. under 50 ms per frame.
- **Tracking:** evaluated qualitatively on live webcam video (stable IDs, no flickering boxes).
  WIDER FACE and PASCAL Face contain only still images without identity annotations
  across frames, so quantitative tracking metrics (MOTA, IDF1) cannot be computed on them.

The detector is trained on still images only. Webcam video is processed frame by frame,
and tracking is a non-learned algorithm (IoU association) that links detections between
consecutive frames.

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the datasets
python scripts/download_data.py --dataset wider_face
python scripts/download_data.py --dataset pascal_face

# 4. Check the webcam (settings in configs/demo.yaml)
python scripts/check_camera.py

# 5. Run the webcam demo
python scripts/run_demo.py --camera 0
```

## License

MIT — see [LICENSE](LICENSE).
