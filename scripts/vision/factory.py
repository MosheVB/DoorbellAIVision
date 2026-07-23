"""Select detector implementation from environment (YOLO11 + stub)."""

from __future__ import annotations

import os


def _is_yolo11_detector_name(name: str) -> bool:
    """True for Ultralytics YOLO11 / YOLOv8-style model keys."""
    n = (name or "").lower().strip()
    if not n:
        return False
    if n in ("yolo", "yolo11", "yolov11"):
        return True
    if n.startswith("yolo11"):
        return True
    if n.startswith("yolov"):
        return True
    return False


def create_detector():
    """Return an object with Detect(bgr), GetClassDesc(id), GetNetworkFPS()."""
    from vision.backends.stub import StubDetector

    det_model = (os.environ.get("DETECTION_MODEL") or "yolo11").lower().strip()
    threshold = float(os.environ.get("DETECTION_THRESHOLD", "0.5") or "0.5")

    if det_model in ("stub", "none", "off"):
        return StubDetector()

    if det_model in ("bytetrack",):
        print(
            "[vision] DETECTION_MODEL=bytetrack — multi-object tracker is not wired yet; "
            "use YOLO11 for detection. Using stub (no detections)."
        )
        return StubDetector()

    if _is_yolo11_detector_name(det_model):
        from yolo11 import YOLO11Detector

        return YOLO11Detector(model_key=det_model, threshold=threshold)

    raise RuntimeError(
        f"Unknown DETECTION_MODEL={det_model!r}. Use yolo11 (default), stub, or bytetrack (stub)."
    )
