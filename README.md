# SkillCorner Test

Video decoding + inference pipeline: decodes a video frame by frame, resizes
frames to 720x1280 RGB, samples them at 10 inferences per second, and runs
YOLOv8n (pretrained on COCO) object detection on each sampled frame.

## Requirements

- Python 3.14
- Dependencies listed in `requirements.txt` (OpenCV, NumPy, PyTorch,
  torchvision, Ultralytics)

## Setup

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The model weights (`yolov8n.pt`) are downloaded automatically on first run
if not already present locally, so no separate download step is required.

## Usage

Run inference on a video:

```bash
source .venv/bin/activate
python infer.py <path-to-video>
```

Or without activating, using the venv's Python directly:

```bash
.venv/bin/python3 infer.py <path-to-video>
```

Optional flags:

- `--output-dir <dir>`: change the output directory (default: `output`)
- `--model <path-or-name>`: use a different YOLO model instead of the
  default `yolov8n.pt` (auto-downloaded if not found locally)

## Output

Running the script writes to `output/` (created automatically):

- `output/output.json` — per-frame inference results plus video metadata
  and run summary
- `output/samples/*.jpg` — a handful of annotated frames illustrating
  detections
- `output/run.log` — execution log (progress every 10 processed frames,
  total execution time, video metadata)
