#!/usr/bin/env python3
"""
Optional: export YOLO11 .pt weights to ONNX (e.g. for TensorRT / ONNX Runtime later).

The live pipeline uses Ultralytics with .pt directly — ONNX is NOT required to run analyze.py.

Example:

  docker compose run --rm --entrypoint python3 doorbell-ai \\
    /app/scripts/export_yolo11_onnx.py --weights /models/yolo11/yolo11s.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    wdir = (os.environ.get("YOLO11_WEIGHTS_DIR") or "/models/yolo11").strip()
    ap.add_argument(
        "--weights",
        default=os.path.join(wdir, "yolo11s.pt"),
        help="Input .pt path (default: YOLO11_WEIGHTS_DIR/yolo11s.pt)",
    )
    ap.add_argument(
        "--out-dir",
        default=wdir,
        help="Directory for the .onnx file (default: same dir as weights)",
    )
    ap.add_argument("--imgsz", type=int, default=640, help="Export image size")
    ap.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset (12 is widely compatible)",
    )
    args = ap.parse_args()

    w = Path(args.weights)
    if not w.is_file():
        raise SystemExit(f"[export_yolo11_onnx] Missing weights: {w}")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("pip install ultralytics") from e

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(w))
    out = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
    )
    # ultralytics returns path to exported file
    p = Path(out)
    print(f"[export_yolo11_onnx] wrote {p} ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
