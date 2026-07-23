#!/usr/bin/env python3
"""
YOLO11 detector (Ultralytics) — primary path for DoorbellAIVision.

Implements the pipeline detector interface:
  Detect(bgr) -> list of Detection-like objects
  GetClassDesc(class_id) -> str
  GetNetworkFPS() -> float
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Detection:
    """Bounding box + score in the shape expected by analyze.py."""

    ClassID: int
    Confidence: float
    Left: float
    Top: float
    Right: float
    Bottom: float


class YOLO11Detector:
    """Ultralytics YOLO11 (or compatible .pt weights)."""

    def __init__(self, model_key: str = "yolo11", threshold: float = 0.5):
        self.threshold = float(threshold)
        self.model_key = (model_key or "yolo11").lower().strip()
        self._weights = self._resolve_weights(self.model_key)
        self._imgsz = int(os.environ.get("YOLO11_IMGSZ", "640") or "640")
        self._device = (os.environ.get("YOLO11_DEVICE", "0") or "0").strip()

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "Ultralytics is required for YOLO11. Install with: pip install ultralytics"
            ) from e

        self._model = YOLO(self._weights)
        self._t_prev = time.time()
        self._frames = 0
        self._fps = 0.0
        print(
            f"[YOLO11] Ready  weights={self._weights!r}  threshold={threshold}  "
            f"imgsz={self._imgsz}  device={self._device!r}"
        )

    @staticmethod
    def _hub_weights_filename(model_key: str) -> str:
        """Filename passed to Ultralytics (hub download) for a given DETECTION_MODEL key."""
        k = model_key.lower().strip()
        if k in ("yolo", "yolo11", "yolov11"):
            return "yolo11s.pt"
        if k.startswith("yolo11") and (
            k.endswith(".pt") or k.endswith(".onnx") or k.endswith(".engine")
        ):
            return k
        if k.startswith("yolo11"):
            return f"{k}.pt"
        if k.startswith("yolov") and not k.endswith((".pt", ".onnx", ".engine")):
            return f"{k}.pt"
        return "yolo11s.pt"

    @staticmethod
    def _resolve_weights(model_key: str) -> str:
        """Prefer YOLO11_WEIGHTS, then a file under YOLO11_WEIGHTS_DIR, else hub name."""
        explicit = (os.environ.get("YOLO11_WEIGHTS") or "").strip()
        if explicit:
            return explicit
        hub = YOLO11Detector._hub_weights_filename(model_key)
        wdir = (os.environ.get("YOLO11_WEIGHTS_DIR") or "/models/yolo11").strip()
        local = os.path.join(wdir, os.path.basename(hub))
        if os.path.isfile(local) and os.path.getsize(local) > 0:
            return local
        return hub

    def Detect(self, bgr: np.ndarray, overlay: str = "") -> list[Detection]:
        del overlay  # unused; signature compatibility
        if bgr is None or bgr.size == 0:
            return []

        r0 = self._model.predict(
            source=bgr,
            conf=self.threshold,
            verbose=False,
            imgsz=self._imgsz,
            device=self._device,
        )[0]

        out: list[Detection] = []
        if r0.boxes is None or len(r0.boxes) == 0:
            self._frames += 1
            if self._frames % 30 == 0:
                now = time.time()
                self._fps = 30.0 / max(now - self._t_prev, 1e-6)
                self._t_prev = now
            return out

        h, w = bgr.shape[:2]
        boxes = r0.boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].detach().cpu().numpy().astype(np.float64)
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            x1 = max(0.0, min(float(w - 1), x1))
            x2 = max(0.0, min(float(w - 1), x2))
            y1 = max(0.0, min(float(h - 1), y1))
            y2 = max(0.0, min(float(h - 1), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(Detection(cls_id, conf, x1, y1, x2, y2))

        self._frames += 1
        if self._frames % 30 == 0:
            now = time.time()
            self._fps = 30.0 / max(now - self._t_prev, 1e-6)
            self._t_prev = now
        return out

    def GetClassDesc(self, class_id: int) -> str:
        names: Any = getattr(self._model, "names", None)
        if isinstance(names, dict):
            return str(names.get(int(class_id), f"class_{int(class_id)}"))
        if isinstance(names, (list, tuple)) and 0 <= int(class_id) < len(names):
            return str(names[int(class_id)])
        return f"class_{int(class_id)}"

    def GetNetworkFPS(self) -> float:
        return float(self._fps)
