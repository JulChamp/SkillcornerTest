#!/usr/bin/env python3
"""
Video decoding + inference pipeline.

Usage:
    python3 infer.py cut.mp4

Pipeline:
  1. Decode the video frame by frame.
  2. Resize every frame to 720 (height) x 1280 (width).
  3. Ensure every frame is RGB (3-channel) before inference.
  4. Sample frames at 10 inferences per second of video (independent of the
     source video's native frame rate) and run object detection (YOLOv8n,
     pretrained on COCO) on each sampled frame.
  5. Log execution time every 10 processed frames, the total execution time,
     and video metadata.
  6. Write:
       - output/output.json   -> per-frame inference results
       - output/samples/*.jpg -> a few annotated frames illustrating results
       - output/run.log       -> the requested log file
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

TARGET_HEIGHT = 720
TARGET_WIDTH = 1280
TARGET_INFERENCE_FPS = 10
LOG_EVERY_N_FRAMES = 10
NUM_SAMPLE_IMAGES = 8
MODEL_WEIGHTS = "yolov8n.pt"
CONF_THRESHOLD = 0.35


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("video_inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


def get_video_metadata(cap: cv2.VideoCapture, video_path: Path) -> dict:
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = (
        chr(fourcc_int & 0xFF)
        + chr((fourcc_int >> 8) & 0xFF)
        + chr((fourcc_int >> 16) & 0xFF)
        + chr((fourcc_int >> 24) & 0xFF)
    )
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps > 0 else 0.0

    return {
        "file_name": video_path.name,
        "file_size_bytes": video_path.stat().st_size,
        "fourcc": fourcc,
        "source_fps": round(fps, 3),
        "source_frame_count": frame_count,
        "source_width": width,
        "source_height": height,
        "duration_sec": round(duration_sec, 3),
    }


def ensure_rgb(frame: np.ndarray) -> np.ndarray:
    """Guarantee a 3-channel RGB frame regardless of the source color layout."""
    if frame.ndim == 2:
        # Grayscale -> RGB
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if frame.ndim == 3 and frame.shape[2] == 4:
        # BGRA -> RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    if frame.ndim == 3 and frame.shape[2] == 3:
        # OpenCV decodes color frames as BGR -> convert to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unexpected frame shape: {frame.shape}")


def resize_frame(frame: np.ndarray) -> np.ndarray:
    # cv2.resize expects (width, height)
    return cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)


def run_inference(model: YOLO, frame_rgb: np.ndarray) -> list:
    """Run YOLO detection on an RGB frame and return a list of detections."""
    results = cast(list, model.predict(frame_rgb, conf=CONF_THRESHOLD, verbose=False, stream=False))[0]
    results = cast(Results, results)
    detections = []
    boxes = results.boxes if results.boxes is not None else []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            {
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            }
        )
    return detections


def draw_detections(frame_rgb: np.ndarray, detections: list) -> np.ndarray:
    annotated = frame_rgb.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        color = (255, 0, 0) if det["class_name"] == "person" else (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def main():
    parser = argparse.ArgumentParser(description="Decode a video and run inference on its frames.")
    parser.add_argument("video", type=str, help="Path to the input video file (e.g. cut.mp4)")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for outputs")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_WEIGHTS,
        help=f"Path to YOLO model weights (default: {MODEL_WEIGHTS}, auto-downloaded if not found locally)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "run.log"
    logger = setup_logger(log_path)

    if not video_path.is_file():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        sys.exit(1)

    metadata = get_video_metadata(cap, video_path)
    logger.info("Video metadata: " + json.dumps(metadata))

    source_fps = metadata["source_fps"] or TARGET_INFERENCE_FPS
    sample_interval_sec = 1.0 / TARGET_INFERENCE_FPS
    logger.info(
        f"Sampling for inference at {TARGET_INFERENCE_FPS} FPS "
        f"(1 sampled frame every {sample_interval_sec:.3f}s of video, "
        f"source video runs at {source_fps} FPS)."
    )

    logger.info(f"Loading model weights: {args.model}")
    model = YOLO(args.model)

    results_out = []
    next_sample_time = 0.0
    source_frame_idx = -1
    processed_count = 0

    batch_start_time = time.time()
    total_start_time = batch_start_time

    # Save a spread of sample visualization frames across the whole run
    total_expected_samples = max(1, int(metadata["duration_sec"] * TARGET_INFERENCE_FPS))
    viz_stride = max(1, total_expected_samples // NUM_SAMPLE_IMAGES)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        source_frame_idx += 1
        frame_time_sec = source_frame_idx / source_fps

        if frame_time_sec + 1e-9 < next_sample_time:
            continue
        next_sample_time += sample_interval_sec

        # 1) Resize to 720x1280
        resized = resize_frame(frame_bgr)
        # 2) Ensure RGB color
        frame_rgb = ensure_rgb(resized)

        # 3) Run inference
        detections = run_inference(model, frame_rgb)

        results_out.append(
            {
                "sample_index": processed_count,
                "source_frame_index": source_frame_idx,
                "timestamp_sec": round(frame_time_sec, 3),
                "frame_shape_hwc": list(frame_rgb.shape),
                "num_detections": len(detections),
                "detections": detections,
            }
        )

        # 4) Save a subset of annotated frames for visualization
        if processed_count % viz_stride == 0 and len(list(samples_dir.glob("*.jpg"))) < NUM_SAMPLE_IMAGES:
            annotated_rgb = draw_detections(frame_rgb, detections)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            out_path = samples_dir / f"sample_{processed_count:04d}_t{frame_time_sec:.2f}s.jpg"
            cv2.imwrite(str(out_path), annotated_bgr)

        processed_count += 1

        # 5) Log execution time every 10 processed frames
        if processed_count % LOG_EVERY_N_FRAMES == 0:
            now = time.time()
            batch_elapsed = now - batch_start_time
            total_elapsed = now - total_start_time
            logger.info(
                f"Processed {processed_count} sampled frames "
                f"(video t={frame_time_sec:.2f}s) | "
                f"last {LOG_EVERY_N_FRAMES} frames: {batch_elapsed:.3f}s | "
                f"total elapsed: {total_elapsed:.3f}s"
            )
            batch_start_time = now

    cap.release()

    total_time = time.time() - total_start_time
    logger.info(
        f"Done. Processed {processed_count} sampled frames from "
        f"{metadata['source_frame_count']} source frames. "
        f"Total execution time: {total_time:.3f}s"
    )

    output_json_path = output_dir / "output.json"
    output_payload = {
        "video_metadata": metadata,
        "inference_config": {
            "model": args.model,
            "task": "object_detection",
            "classes": model.names,
            "confidence_threshold": CONF_THRESHOLD,
            "target_inference_fps": TARGET_INFERENCE_FPS,
            "resized_height": TARGET_HEIGHT,
            "resized_width": TARGET_WIDTH,
        },
        "summary": {
            "num_sampled_frames": processed_count,
            "total_execution_time_sec": round(total_time, 3),
        },
        "frames": results_out,
    }
    with open(output_json_path, "w") as f:
        json.dump(output_payload, f, indent=2)

    logger.info(f"Wrote inference results to {output_json_path}")
    logger.info(f"Wrote {len(list(samples_dir.glob('*.jpg')))} sample visualization images to {samples_dir}")
    logger.info(f"Log file written to {log_path}")


if __name__ == "__main__":
    main()
