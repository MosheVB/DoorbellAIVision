#!/usr/bin/env python3
"""
Download YOLO11 .pt weights into models/yolo11/ so they appear under the repo's models/ tree.

Ultralytics normally caches under ~/.cache/ultralytics or downloads to CWD when you pass
only a filename — this script copies the official asset into YOLO11_WEIGHTS_DIR.

Usage (host, project root, with Docker):

  docker compose run --rm --entrypoint python3 doorbell-ai /app/scripts/ensure_yolo11_weights.py

  docker compose run --rm --entrypoint python3 doorbell-ai \\
    /app/scripts/ensure_yolo11_weights.py --name yolo11n.pt

Usage (host with ultralytics installed):

  python3 scripts/ensure_yolo11_weights.py --dest-dir ./models/yolo11
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--name",
        default="yolo11s.pt",
        help="Weights file as published on ultralytics/assets (default: yolo11s.pt)",
    )
    ap.add_argument(
        "--dest-dir",
        default=(os.environ.get("YOLO11_WEIGHTS_DIR") or "/models/yolo11").strip(),
        help="Target directory (Docker: /models/yolo11 → host ./models/yolo11)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite even if dest already exists",
    )
    args = ap.parse_args()

    from ultralytics.utils.downloads import attempt_download_asset

    dest = Path(args.dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / args.name

    if out_path.is_file() and out_path.stat().st_size > 0 and not args.force:
        print(f"[ensure_yolo11_weights] OK (existing): {out_path} ({out_path.stat().st_size} bytes)")
        return 0

    src = Path(attempt_download_asset(args.name)).resolve()
    if not src.is_file():
        raise SystemExit(f"[ensure_yolo11_weights] Download did not produce a file: {src}")

    shutil.copy2(src, out_path)
    print(f"[ensure_yolo11_weights] wrote {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
