FROM python:3.14-slim

# System libraries required by opencv-python at runtime (video decoding,
# libGL for headless image ops) — not covered by requirements.txt.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# torch/torchvision from PyPI pull in multi-GB CUDA packages by default even
# for CPU-only use; install the CPU-only builds first so requirements.txt
# below finds them already satisfied.
RUN pip install --no-cache-dir torch==2.14.0 torchvision==0.29.0 \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY infer.py .

# Docker Desktop's Linux VM reports all logical CPUs (10 on this Apple M4),
# but PyTorch on macOS auto-detects the 4 performance cores and avoids the
# efficiency cores for compute; inside the VM that distinction is lost, so
# PyTorch defaults to 10 threads and over-subscribes, ~40% slower than
# capping it at the same count native macOS picks. Override with `-e
# OMP_NUM_THREADS=<n>` to tune for a different host.
ENV OMP_NUM_THREADS=4

ENTRYPOINT ["python3", "infer.py"]
