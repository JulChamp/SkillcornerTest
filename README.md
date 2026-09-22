# SkillCorner Test

Video decoding + inference pipeline: decodes a video frame by frame, resizes
frames to 720x1280 RGB, samples them at 10 inferences per second, and runs
YOLOv8n (pretrained on COCO) object detection on each sampled frame.

Two independent ways to run it — pick whichever you prefer, neither
requires the other: a local virtual environment (**Setup** / **Usage**
below), or a container (**Docker** below, for a reproducible environment
without installing Python 3.14 locally).

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

## Docker

An alternative to the **Setup**/**Usage** above, not a replacement — both
run the same `infer.py` unmodified, so use whichever fits (native tends to
be faster on Apple Silicon, see the note below; Docker needs no local
Python install). A `Dockerfile` is provided for a reproducible environment
(pinned Python
version, system libraries `ffmpeg`/`libgl1` that `requirements.txt` alone
doesn't cover). PyTorch is installed from the CPU-only wheel index —
installing it from the default PyPI index on Linux otherwise pulls in
several GB of unused CUDA packages.

Build the image:

```bash
docker build -t skillcorner-test .
```

Run it, mounting the video, the model weights (to skip re-downloading),
and an output directory:

```bash
docker run --rm \
  -v "$(pwd)/cut.mp4:/data/cut.mp4:ro" \
  -v "$(pwd)/yolov8n.pt:/app/yolov8n.pt:ro" \
  -v "$(pwd)/output:/app/output" \
  skillcorner-test /data/cut.mp4 --output-dir /app/output
```

**Note:** on macOS, Docker Desktop runs containers in a Linux VM, so this
loses the native CPU optimizations PyTorch's macOS build has (and CoreML
is unavailable inside the container entirely — it's macOS-only).

The VM also reports all of the host's logical CPUs (10 on this Apple M4),
but PyTorch on macOS auto-detects and sticks to the 4 performance cores,
avoiding the efficiency cores for compute; that distinction doesn't exist
inside the Linux VM, so PyTorch defaults to 10 threads there and
over-subscribes. Explicitly capping it with `OMP_NUM_THREADS` (set to `4`
by default in the `Dockerfile`, matching what macOS picks natively) closes
most of that gap:

| Config | Total time (3000 frames) | vs. native |
|---|---|---|
| Native (macOS, PyTorch auto: 4 threads) | 61.6s | — |
| Docker, default threads (10, over-subscribed) | 213.7s | 3.5x slower |
| Docker, `OMP_NUM_THREADS=4` | 117.9s | 1.9x slower |

The remaining ~1.9x gap is the Accelerate-framework CPU kernels PyTorch's
macOS build has and the generic Linux ARM build doesn't — there's no way
to recover that from inside a Linux container. Docker here is for
environment reproducibility, not representative performance numbers;
always benchmark on the real deployment target, and re-tune
`OMP_NUM_THREADS` for its actual core layout (e.g. via `docker run -e
OMP_NUM_THREADS=<n>`, matching physical/performance core count, not
`nproc`).

## Performance analysis

The pipeline's main cost is the inference call, not video decoding or
resizing, so the optimization work here focused on the inference backend:
plain PyTorch (`.pt`) vs. exported ONNX Runtime (`.onnx`) vs. exported
CoreML (`.mlpackage`), on two model sizes.

**Test setup:** full pipeline run (decode + resize + RGB conversion +
inference) over the complete `cut.mp4` (300s @ 25 FPS source, 7500 source
frames, 3000 sampled at 10 inferences/sec), on an Apple M4 (10 cores),
macOS, Python 3.14, PyTorch 2.14 CPU, ONNX Runtime 1.30
(`CPUExecutionProvider`), all inference on CPU (no GPU/MPS device
requested).

| Model | Backend | Total time (3000 frames) | ms / sampled frame |
|---|---|---|---|
| YOLOv8n (3.1M params) | PyTorch `.pt` | 61.6s | 20.5ms |
| YOLOv8n | ONNX Runtime `.onnx` | 71.6–90.0s* | 23.9–30.0ms |
| YOLOv8n | CoreML `.mlpackage` (Neural Engine) | 67.7s | 22.6ms |
| YOLOv8m (25.9M params) | PyTorch `.pt` | 255.7s | 85.2ms |
| YOLOv8m | ONNX Runtime `.onnx` | 411.2s | 137.1ms |

\* ONNX showed run-to-run variance; explicit `intra_op_num_threads` tuning
(2/4/6/8/10) was tested and never beat ONNX Runtime's own default
auto-threading.

**Findings:**

- PyTorch's native CPU backend beats both ONNX Runtime and CoreML on this
  chip, on both model sizes. The gap *widens* with model size (YOLOv8n:
  +15–45%, YOLOv8m: +61%) rather than narrowing, which argues against the
  "small model, per-call overhead dominates" explanation — PyTorch's
  ARM-optimized CPU kernels appear to simply scale better here than ONNX
  Runtime's generic CPU execution provider.
- Detections agree closely across backends (bounding boxes and confidences
  within ~0.01–0.03); the only divergence is a handful of borderline
  detections near the `conf=0.35` threshold flipping in or out — expected
  floating-point variance between backends, not a correctness bug.
- `cap.grab()`/`cap.retrieve()` (decode only sampled frames, skip decoding
  the rest) was tested as a decode-side optimization but produced a
  negligible (<1%) gain, confirming inference — not decoding — is the
  bottleneck.
- No football-specific pretrained detector exists off the shelf; YOLOv8m
  (still COCO-pretrained) was used as a stand-in for "more accurate,"
  since the relevant class (`person`) benefits from a larger backbone on
  small/distant players. A real accuracy gain would require fine-tuning on
  football footage (e.g. SoccerNet), which was out of scope here.

**Conclusion:** on this machine, plain PyTorch inference is both the
fastest and the simplest option — no export step, no extra dependencies.
The ONNX/CoreML results here are specific to Apple Silicon CPU inference
and shouldn't be assumed to generalize to other targets (e.g. an x86
server with AVX-512/oneDNN might favor ONNX Runtime); re-benchmark on the
actual deployment hardware before switching backends.
