# DoorbellAIVision

Real-time person detection and **smart zoom** for a UniFi Protect doorbell camera, running on an NVIDIA Jetson. The pipeline ingests two RTSP streams (doorbell + package camera), runs YOLO11 detection, and produces a broadcast-style output: it automatically frames and smoothly tracks the subject, overlays the package camera picture-in-picture, and serves a live dashboard over the LAN.

## How it works

```
RTSP (GStreamer, NVDEC hw decode)
  → YOLO11 person detection            scripts/yolo11.py, scripts/vision/
  → focus selection + smart zoom        scripts/analyze.py
  → package-cam PiP + weather overlay
  → MJPEG web dashboard                 scripts/web_server.py
```

- **Smart zoom** — the crop lerps toward the detected subject with configurable padding, smoothing, and zoom-out hold; aspect-corrected expansion is anchored to avoid clipping heads on a high doorbell view.
- **FAR → CLOSE power saving** — the detector starts on a fixed porch/walk-up ROI crop (distant people are larger in NN input, and inference is cheaper); once a subject is confirmed close for N frames, it switches to full-frame detection with gating disabled.
- **Deployment** — one `docker compose up` on a TensorRT base image (`nvcr.io/nvidia/tensorrt`), with an optional systemd unit ([scripts/install_services.sh](scripts/install_services.sh)) for run-on-boot.

## Quick start

```bash
cp config/camera.env.example config/camera.env   # then set your RTSP URLs
./scripts/setup_models.sh                        # download YOLO11 weights into ./models
docker compose up
```

Dashboard: `http://<jetson>:8080`. Detections can be saved as an H.264 MP4 (`SAVE_DETECTIONS=true`); the writer re-encodes with `libx264 + faststart` so the file previews in browsers, not just VLC.

Replay a recorded clip instead of live RTSP (for tuning):

```bash
INPUT_VIDEO=/output/test.mp4 ./scripts/replay_output.sh
```

## Configuration

Everything is environment-driven via `config/camera.env` — see [config/camera.env.example](config/camera.env.example) for the full annotated list (detection thresholds, zoom behavior, ROI crops, FAR/CLOSE switching, overlays, web dashboard).

ROI values are produced with the included browser-based labeling tool: [tools/roi_label_tool.html](tools/roi_label_tool.html) ([docs](tools/README_roi_label_tool.md)).

## About this public mirror

This is a curated mirror of a private working repository. The **subject-tracking evaluation suite** — keyframe-based ground-truth labeling, optical-flow subject tracking, A/B scoring of zoom behavior against "time on subject," and the Optuna-style parameter tuner that produced the shipped defaults — is withheld. The docs it produced ([docs/MOTION_ZOOM.md](docs/MOTION_ZOOM.md), [docs/ROI_RECOMMENDATIONS.md](docs/ROI_RECOMMENDATIONS.md)) are included. Happy to walk through the withheld tooling on request.
