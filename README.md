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
│   ├── detector.py         # Face detectors (YuNet baseline; our trained model later)
│   ├── tracker.py          # Tracker (detection association across frames)
│   ├── pipeline.py         # Detection + tracking pipeline
│   ├── camera.py           # Webcam video capture (native resolution, optional mirroring)
│   ├── config.py           # YAML config loading
│   ├── grabber.py          # Background thread that reads (and processes) camera frames
│   ├── window.py           # Qt video window locked to the camera aspect ratio
│   ├── visualize.py        # Bounding box and label rendering
│   ├── metrics.py          # FPS, mAP, IoU computation
│   └── wider_face.py       # WIDER FACE -> YOLO label conversion
└── tests/                  # Unit tests
```

## Roadmap

1. **Data** — download and convert WIDER FACE / PASCAL Face into a unified annotation format.
2. **Baseline** — pretrained OpenCV YuNet detector as a stand-in to build the app and as a reference point for accuracy and speed.
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

# 3. Check the webcam (settings in configs/demo.yaml)
python scripts/check_camera.py

# 4. Run the webcam demo: boxes, face IDs and FPS (settings in configs/demo.yaml;
#    the YuNet model is downloaded to models/ on the first run)
python scripts/run_demo.py
```

## Training (Colab or Kaggle)

The detector is trained in the cloud on a free T4 GPU; only the resulting weights
(`best.pt`, ~6 MB) are brought back to the laptop.

**Google Colab** — [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb):

1. Open the notebook in Colab (**File → Upload notebook** or **Open from GitHub**), select a `T4 GPU` runtime.
2. Private repo: add a `GITHUB_TOKEN` secret (🔑 in the sidebar) with notebook access.
3. **Run all** with `SMOKE_TEST = True`, then set it to `False` and **Run all** again.
   WIDER FACE is downloaded from Hugging Face; results go to Google Drive
   (`MyDrive/face-detection-tracking/runs`). After a disconnect, **Run all** resumes from `last.pt`.
4. Download `best.pt` from Drive into `models/`.

**Kaggle** — [`notebooks/train_kaggle.ipynb`](notebooks/train_kaggle.ipynb) (needs a phone-verified account):

1. Import the notebook, attach the dataset `mksaad/wider-face-a-face-detection-benchmark`,
   enable `GPU T4 x2` and Internet.
2. Run with `SMOKE_TEST = True`, then set it to `False` and **Save Version → Save & Run All**.
3. Download `best.pt` from the notebook output into `models/`.

The notebook runs these scripts, which also work locally:

```bash
# WIDER FACE -> YOLO labels (images are symlinked, not copied)
python scripts/prepare_wider_face.py --root data/raw/wider_face --output data/processed/wider_yolo

# Train (settings in configs/train.yaml; --set overrides any of them)
python scripts/train.py --config configs/train.yaml --set data=data/processed/wider_yolo/data.yaml
```

## License

MIT — see [LICENSE](LICENSE).
