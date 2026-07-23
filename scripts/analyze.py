#!/usr/bin/env python3
"""
DoorbellAIVision - RTSP video pipeline with object detection and smart zoom.

Capture: OpenCV + GStreamer (RTSP, hardware H.264 decode).
Inference: pluggable detector (see `vision.factory` — YOLO11 via Ultralytics; ByteTrack TBD).
Features:
  - Smart zoom toward a chosen detection (priority: face > person > bag when labels exist)
  - Optional attention / inference ROIs for porch layout tuning
  - Package camera overlay, weather overlay, web dashboard
"""

import atexit
import json
import os
import math
import shutil
import subprocess
import csv
import cv2
import numpy as np
import time
import datetime
import threading
import requests
from collections import deque
from types import SimpleNamespace

# ── Config ─────────────────────────────────────────────────────────────────────
# If set, run from this local video file instead of RTSP (e.g. /output/rtsp_raw.mp4).
INPUT_VIDEO      = (os.environ.get("INPUT_VIDEO", "") or "").strip()
# Max frames to process then exit gracefully (0 = unlimited).
# - Local file: also combined with OpenCV/ffprobe EOF limits (whichever is smaller).
# - Live RTSP: this is the ONLY auto-stop unless you Ctrl+C.
FILE_MAX_FRAMES  = int(os.environ.get("FILE_MAX_FRAMES", "0") or "0")
# Start processing at this frame index (local file only; 0 = from beginning).
# Frame indexing is 0-based, but this is implemented by skipping writes for
# capture frames with `frame_count <= FILE_START_FRAME`.
FILE_START_FRAME = int(os.environ.get("FILE_START_FRAME", "0") or "0")
# Alternatively start processing at this wall-clock time (seconds) from the
# beginning of the local file. Takes precedence over FILE_START_FRAME.
# Example: FILE_START_TIME_SEC=30 starts at the frame just after 30 seconds.
FILE_START_TIME_SEC = float(os.environ.get("FILE_START_TIME_SEC", "-1") or "-1")
RTSP_URL         = os.environ.get("RTSP_URL",     "")
PKG_RTSP_URL     = os.environ.get("PKG_RTSP_URL", "")
TLS_SKIP_VERIFY  = os.environ.get("RTSP_TLS_SKIP_VERIFY", "true").lower() == "true"
CODEC            = os.environ.get("RTSP_CODEC",     "h264")
LATENCY_MS       = int(os.environ.get("RTSP_LATENCY_MS", "200"))
PROTOCOLS        = os.environ.get("RTSP_PROTOCOLS", "tcp")
# nvh264dec  = NVDEC hardware decode (RTX GPU, recommended)
# avdec_h264 = software fallback if nvh264dec is unavailable
RTSP_DECODER     = os.environ.get("RTSP_DECODER",   "nvh264dec")
THRESHOLD         = float(os.environ.get("DETECTION_THRESHOLD", "0.50"))
DETECTION_MODEL   = os.environ.get("DETECTION_MODEL",  "yolo11")
SAVE_DETECTIONS  = os.environ.get("SAVE_DETECTIONS", "true").lower() == "true"
RECORD_RTSP       = os.environ.get("RECORD_RTSP", "true").lower() == "true"
RECORD_RTSP_FPS   = float(os.environ.get("RECORD_RTSP_FPS", "30"))
RECORD_RTSP_NAME  = os.environ.get("RECORD_RTSP_NAME", "rtsp_raw.mp4")
# If 0, record until the process is stopped.
RECORD_RTSP_SECS  = int(os.environ.get("RECORD_RTSP_SECS", "0"))
# When using RECORD_RTSP_SECS > 0, optionally exit once the raw recording
# writer is closed. This prevents MP4 corruption caused by abrupt container
# termination before the muxer writes the final moov atom.
EXIT_AFTER_RECORD = os.environ.get("EXIT_AFTER_RECORD", "false").lower() == "true"
OUTPUT_DIR       = os.environ.get("OUTPUT_DIR", "/output")
DETECTIONS_VIDEO_NAME = os.environ.get("DETECTIONS_VIDEO_NAME", "detections.mp4").strip() or "detections.mp4"
ENABLE_WEB       = os.environ.get("ENABLE_WEB", "false").lower() == "true"
WEB_PORT         = int(os.environ.get("WEB_PORT", "8080"))
SHOW_BOXES       = os.environ.get("SHOW_BOXES", "false").lower() == "true"
# When running from INPUT_VIDEO, you typically don't want to connect to the
# live package RTSP feed after the fact.
DISABLE_PKG_CAM  = os.environ.get("DISABLE_PKG_CAM", "false").lower() == "true"

# Debug tracing (CSV under OUTPUT_DIR).
DEBUG_TRACE = os.environ.get("DEBUG_TRACE", "false").lower() == "true"
DEBUG_TRACE_EVERY = int(os.environ.get("DEBUG_TRACE_EVERY", "5") or "5")
DEBUG_TRACE_CSV_NAME = (os.environ.get("DEBUG_TRACE_CSV_NAME", "zoom_trace.csv") or "zoom_trace.csv").strip()

# Far-lock evaluation trace (CSV under OUTPUT_DIR).
# This writes per-frame focus bbox + derived far-flag so we can compare
# Path A vs Path B detection backends.
DEBUG_FOCUS_TRACE = os.environ.get("DEBUG_FOCUS_TRACE", "false").lower() == "true"
DEBUG_FOCUS_TRACE_CSV_NAME = (
    os.environ.get("DEBUG_FOCUS_TRACE_CSV_NAME", "focus_trace.csv") or "focus_trace.csv"
).strip()
# Define "far" by bbox height fraction (focus bbox height / frame height).
FOCUS_FAR_FRAC_THRESHOLD = float(os.environ.get("FOCUS_FAR_FRAC_THRESHOLD", "0.12"))
# "Stability" metric for focus center continuity (center distance normalized
# by frame height).
FOCUS_STABLE_DIST_NORM_THRESHOLD = float(os.environ.get("FOCUS_STABLE_DIST_NORM_THRESHOLD", "0.06"))

# Far → close detection mode.
# We optimize for distance detection using a far ROI (r3) and only switch
# to full-frame inference once we have strong evidence the person is close.
#
# Transition is monotonic: FAR -> CLOSE and stays CLOSE (we do not switch back).
CLOSE_FRAC_THRESHOLD = float(os.environ.get("CLOSE_FRAC_THRESHOLD", "0.15"))
# Require the close condition for N consecutive frames to avoid momentary spikes.
CLOSE_SUSTAIN_FRAMES = int(os.environ.get("CLOSE_SUSTAIN_FRAMES", "30"))

# Optional occasional low-res full-frame snapshot while in FAR mode, so we
# don't miss people who enter outside the far ROI.
# Set to 0 to disable snapshots entirely.
FULL_SNAPSHOT_EVERY_N_FRAMES = int(os.environ.get("FULL_SNAPSHOT_EVERY_N_FRAMES", "90"))
# Resize factor for the snapshot full-frame inference input.
# (This is applied before calling net.Detect; coordinates are mapped back.)
SNAPSHOT_FULL_SCALE = float(os.environ.get("SNAPSHOT_FULL_SCALE", "0.5"))

# ── Optional motion priming (FAR mode only, easy to disable) ─────────────────
# When MOTION_PRIME_FAR_SNAPSHOT=1 and ZOOM_ATTENTION_ROI_NORM is set, a spike in
# mean abs-diff inside the attention ROI triggers an extra low-res full-frame
# snapshot (same path as FULL_SNAPSHOT_EVERY_N_FRAMES). Undo: set to 0 or remove.
# Does not change YOLO or any model — env flag only.
MOTION_PRIME_FAR_SNAPSHOT = os.environ.get("MOTION_PRIME_FAR_SNAPSHOT", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MOTION_PRIME_MEAN_DIFF = float(os.environ.get("MOTION_PRIME_MEAN_DIFF", "5.0") or "5.0")
MOTION_PRIME_COOLDOWN_FRAMES = int(os.environ.get("MOTION_PRIME_COOLDOWN_FRAMES", "12") or "12")
MOTION_PRIME_GRAY_WIDTH = int(os.environ.get("MOTION_PRIME_GRAY_WIDTH", "96") or "96")

# Optional: crop fed to the detector (normalized x1,y1,x2,y2 — same convention as tools/ROI docs).
INFERENCE_ROI_NORM = os.environ.get("INFERENCE_ROI_NORM", "").strip()
# Optional: smart-zoom considers only detections whose center lies inside this ROI.
# Aliases honor older configs: MOTION_STAGE1_ROI_NORM / FOVEATED_ROI_NORM.
_zattn = (
    os.environ.get("ZOOM_ATTENTION_ROI_NORM", "").strip()
    or os.environ.get("MOTION_STAGE1_ROI_NORM", "").strip()
    or os.environ.get("FOVEATED_ROI_NORM", "").strip()
)
ZOOM_ATTENTION_ROI_NORM = _zattn

# Comma-separated labels (lowercase), e.g. "person" or "person,face". Empty = all classes.
_raw_cf = (os.environ.get("DETECT_CLASS_FILTER", "") or "").strip().lower()
DETECT_CLASS_FILTER: frozenset[str] | None = None
if _raw_cf:
    DETECT_CLASS_FILTER = frozenset(x.strip() for x in _raw_cf.split(",") if x.strip())

# Min IoU vs previous focus bbox to keep prior focus (reduces label/priority hopping).
ZOOM_FOCUS_IOU_MIN = float(os.environ.get("ZOOM_FOCUS_IOU_MIN", "0.35"))
# EMA on target crop center/size (0 = off).
ZOOM_TARGET_EMA = float(os.environ.get("ZOOM_TARGET_EMA", "0") or "0")
# EMA blend full-frame↔zoom when gate flips (0 = instant).
ZOOM_GATE_EMA = float(os.environ.get("ZOOM_GATE_EMA", "0.15") or "0")

# When both face and person are detected, frame using their union bbox for zoom.
# PeopleNet often keeps face+person as separate boxes when the person box is huge
# (IoU(face,person) can fall below merge threshold), which makes focus pick the tiny
# face box and tight aspect-ratio crops can clip the forehead/hair.
FACE_PERSON_UNION_FRAMING = os.environ.get("FACE_PERSON_UNION_FRAMING", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# When fixing crop aspect ratio, the crop is often "too wide" and we add vertical pixels.
# "center" expands symmetrically (default) and can clip the forehead when the subject is
# high in frame and y1 clamps to 0. "bottom" keeps the lower edge of the crop and grows
# upward — better for doorbell views where the head is near the top of the image.
_arh = (os.environ.get("ZOOM_AR_HEIGHT_ANCHOR", "center") or "center").strip().lower()
ZOOM_AR_HEIGHT_ANCHOR = _arh if _arh in ("center", "bottom", "top") else "center"

# Extra padding above the top of a *person* box (fraction of bbox height). Helps when
# there is no face class (YOLO11 person-only) so the crop leaves room for hair/forehead.
ZOOM_PERSON_TOP_PAD_FRAC = max(
    0.0,
    float(os.environ.get("ZOOM_PERSON_TOP_PAD_FRAC", "0") or "0"),
)


def _parse_norm_roi(s: str) -> tuple[float, float, float, float] | None:
    if not (s or "").strip():
        return None
    try:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 4:
            return None
        return tuple(map(float, parts))  # type: ignore[return-value]
    except Exception:
        return None


_INFERENCE_ROI_PARSED = _parse_norm_roi(INFERENCE_ROI_NORM)
_ATTENTION_ROI_PARSED = _parse_norm_roi(ZOOM_ATTENTION_ROI_NORM)


def _attention_point_inside(px: float, py: float, img_w: int, img_h: int) -> bool:
    if _ATTENTION_ROI_PARSED is None:
        return True
    x1n, y1n, x2n, y2n = _ATTENTION_ROI_PARSED
    x1p = int(round(x1n * (img_w - 1)))
    x2p = int(round(x2n * (img_w - 1)))
    y1p = int(round(y1n * (img_h - 1)))
    y2p = int(round(y2n * (img_h - 1)))
    if px < x1p or px > x2p:
        return False
    if py < y1p or py > y2p:
        return False
    return True


def _attention_roi_gray_prime(bgr: np.ndarray, img_w: int, img_h: int) -> np.ndarray | None:
    """Grayscale patch of attention ROI for motion priming (mean abs-diff)."""
    if _ATTENTION_ROI_PARSED is None:
        return None
    x1n, y1n, x2n, y2n = _ATTENTION_ROI_PARSED
    x1 = int(round(x1n * (img_w - 1)))
    x2 = int(round(x2n * (img_w - 1)))
    y1 = int(round(y1n * (img_h - 1)))
    y2 = int(round(y2n * (img_h - 1)))
    x1 = max(0, min(img_w - 1, min(x1, x2)))
    x2 = max(0, min(img_w, max(x1, x2) + 1))
    y1 = max(0, min(img_h - 1, min(y1, y2)))
    y2 = max(0, min(img_h, max(y1, y2) + 1))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gw = max(8, min(320, int(MOTION_PRIME_GRAY_WIDTH)))
    gh = max(1, int(round(gray.shape[0] * (gw / float(max(1, gray.shape[1]))))))
    return cv2.resize(gray, (gw, gh), interpolation=cv2.INTER_AREA)


def _mean_absdiff_gray_prime(prev: np.ndarray | None, cur: np.ndarray) -> float:
    if prev is None or prev.shape != cur.shape:
        return 0.0
    return float(np.mean(cv2.absdiff(prev, cur)))


def inference_crop_for_frame(img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """Pixel crop (x1,y1,x2,y2) with x2/y2 suitable for numpy slice [y1:y2, x1:x2]."""
    if _INFERENCE_ROI_PARSED is None:
        return None
    x1n, y1n, x2n, y2n = _INFERENCE_ROI_PARSED
    a = int(round(x1n * (img_w - 1)))
    b = int(round(y1n * (img_h - 1)))
    c = int(round(x2n * (img_w - 1)))
    d = int(round(y2n * (img_h - 1)))
    xi1 = max(0, min(img_w - 1, min(a, c)))
    yi1 = max(0, min(img_h - 1, min(b, d)))
    # exclusive end for slicing
    xi2 = max(0, min(img_w, max(a, c) + 1))
    yi2 = max(0, min(img_h, max(b, d) + 1))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    return (xi1, yi1, xi2, yi2)

# Web stream JPEG width.  Resizes before encoding to cut encode time in half
# compared to the native 1600×1200 feed.  Set to 0 to disable downscaling.
WEB_JPEG_WIDTH   = int(os.environ.get("WEB_JPEG_WIDTH", "1280"))

# PeopleNet zoom priority (label-string based — robust across model versions)
# Face first — for a doorbell camera the face is the most useful target.
# Falls back to person (body) when no face is detected, then bag.
_ZOOM_PRIORITY = {
    "face":   0,
    "person": 1,
    "bag":     2,
    # Scaffolding for future multi-class models.
    "car":    2,
    "truck":  2,
    "bus":    2,
}

# Smart zoom
ZOOM_PADDING      = float(os.environ.get("ZOOM_PADDING",      "0.35"))
ZOOM_SMOOTH       = float(os.environ.get("ZOOM_SMOOTH",       "0.10"))
ZOOM_OUT_HOLD     = int(os.environ.get("ZOOM_OUT_HOLD",       "90"))
# Distance-based padding (tighter zoom when person is far away).
# depth_frac = bbox_height / frame_height
ZOOM_PADDING_DIST_FAR_FRAC = float(os.environ.get("ZOOM_PADDING_DIST_FAR_FRAC", "0.05"))
ZOOM_PADDING_DIST_NEAR_FRAC = float(os.environ.get("ZOOM_PADDING_DIST_NEAR_FRAC", "0.20"))
ZOOM_PADDING_DIST_FAR_MUL = float(os.environ.get("ZOOM_PADDING_DIST_FAR_MUL", "0.75"))
ZOOM_PADDING_DIST_NEAR_MUL = float(os.environ.get("ZOOM_PADDING_DIST_NEAR_MUL", "1.00"))
# Minimum fraction of the frame each crop dimension must cover.
# Lower = tighter zoom allowed on small/distant subjects.
ZOOM_MIN_FRACTION = float(os.environ.get("ZOOM_MIN_FRACTION", "0.25"))
# Minimum bounding-box height as a fraction of frame height.
# Objects shorter than this are considered too far away and are discarded.
# 0.0 = disabled (detect everything); 0.10 = ignore objects < 10 % of frame height.
MIN_DET_FRACTION  = float(os.environ.get("MIN_DET_FRACTION",  "0.0"))


def _effective_class_filter() -> frozenset[str] | None:
    """Optional label filter; YOLO11 defaults to person-only when unset (COCO noise)."""
    if DETECT_CLASS_FILTER is not None:
        return DETECT_CLASS_FILTER
    m = (DETECTION_MODEL or "").lower().strip()
    if m in ("yolo", "yolo11", "yolov11") or m.startswith("yolo11") or m.startswith("yolov"):
        return frozenset(["person"])
    return None


# Package cam overlay
OVERLAY_FRACTION = float(os.environ.get("OVERLAY_FRACTION", "0.25"))
OVERLAY_MARGIN   = 12

# Weather
WEATHER_LAT      = os.environ.get("WEATHER_LAT", "")   # blank = auto-detect from IP
WEATHER_LON      = os.environ.get("WEATHER_LON", "")
WEATHER_UNIT     = os.environ.get("WEATHER_UNIT", "fahrenheit")  # or celsius
WEATHER_REFRESH  = int(os.environ.get("WEATHER_REFRESH_MINUTES", "30")) * 60

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Shared state ───────────────────────────────────────────────────────────────
_frame_lock     = threading.Lock()
_detection_log  = deque(maxlen=50)
_stats          = {"fps": 0.0, "frames": 0, "latency_ms": 0.0, "last_detection_time": 0.0}

# Inference thread state — keeps inference off the display-loop critical path.
_infer = {
    "lock":     threading.Lock(),
    "ready":    threading.Event(),
    "stop":     threading.Event(),  # set from main shutdown to exit worker cleanly
    "bgr":      None,   # latest frame queued for inference
    "detect_crop": None,  # optional (x1,y1,x2,y2) crop used only for inference
    "infer_scale": 1.0,   # optional resize scale applied to the inference input
    "dets":     [],     # latest completed detections
    "frame_id": 0,      # increments each time inference completes
}

# Runtime-adjustable settings (written by web_server via /api/settings)
_settings = {
    "overlay_scale":      1.0,   # time/weather overlay scale only
    "pkg_overlay_scale":  1.0,   # package camera thumbnail scale only
    "zoom_min_fraction":  ZOOM_MIN_FRACTION,
    "min_det_fraction":   MIN_DET_FRACTION,  # min bbox height / frame height (0 = off)
    "det_threshold":      THRESHOLD,
    "show_pkg_cam":       True,
    # When True, both overlays shift inward to stay inside the 4:3→16:9 crop safe zone.
    "fill_screen":        False,
    "show_boxes":         SHOW_BOXES,  # draw detection bounding boxes (people, etc.) on the feed
    "merge_boxes":       True,    # merge overlapping detections into one box
    # When zooming into a face bbox, we frame with extra padding because
    # PeopleNet face boxes can omit chin/shoulders and may otherwise get cut.
    "face_padding_mul":  1.8,
    # Positive bias shifts crop center downward (include more below face).
    "face_y_bias":       0.15,
    # Layout calibration: normalized vertical boundaries (0..1) for 6 zones.
    # Edges are between zones:
    #   patio [0,e0), steps [e0,e1), driveway [e1,e2), sidewalk [e2,e3),
    #   road [e3,e4), cross-sidewalk [e4,1]
    # Value list has length 5: [e0,e1,e2,e3,e4]
    "zone_edges":        [0.20, 0.33, 0.52, 0.68, 0.82],
}

# Bundle passed to web_server so it can read frames and write settings
_shared = {
    "frame_lock":    _frame_lock,
    "latest_jpeg":   None,
    "detection_log": _detection_log,
    "stats":         _stats,
    "settings":      _settings,
    "det_count":     0,                # number of detections after filtering
    "det_raw_count":0,                # detections after min_det_fraction, before depth-proxy filter
    "det_model_count":0,              # raw count from detector before merge/filter
    "perf_monitor":  None,   # set in main() after SystemMonitor starts
    "overlay":       None,   # { time, date, condition, temp, temp_unit, hours, next_*, ... } for HTML overlay
    "pkg_cam_jpeg":  None,   # latest package camera as JPEG bytes (for /api/pkg_cam)
}

_pkg_lock       = threading.Lock()
_latest_pkg_bgr = None

_weather_lock   = threading.Lock()
_weather_cache  = None   # current_condition, temp_display, temp_unit, hours_until, next_*, timezone


# ── GStreamer pipeline ─────────────────────────────────────────────────────────

def _build_gst_pipeline(url: str) -> str:
    """
    Build a GStreamer pipeline string for cv2.VideoCapture.
    tls-validation-flags=0 accepts UniFi Protect's self-signed certificate.
    drop=1 max-buffers=1 sync=false keeps latency minimal (always latest frame).
    RTSP_DECODER selects the H.264 decoder element:
      nvh264dec  — NVDEC hardware decode via NVIDIA GPU (default, lowest CPU load)
      avdec_h264 — software fallback (gstreamer1.0-libav)
    """
    tls = "0" if TLS_SKIP_VERIFY else "4"
    return (
        f'rtspsrc location="{url}" '
        f'tls-validation-flags={tls} protocols={PROTOCOLS} latency={LATENCY_MS} '
        f'! rtph264depay ! h264parse '
        f'! {RTSP_DECODER} '
        f'! videoconvert '
        f'! video/x-raw,format=BGR '
        f'! appsink drop=1 max-buffers=1 sync=false'
    )


def _is_local_file_source(url: str) -> bool:
    if not url:
        return False
    path = url
    if path.startswith("file://"):
        path = path[7:]
    return os.path.isfile(path)


def _local_video_path(url: str) -> str:
    return url[7:] if url.startswith("file://") else url


def _default_video_writer_fourcc_list() -> str:
    """Prefer mp4v first in Docker: OpenCV/FFmpeg often maps H.264 fourccs to ``h264_v4l2m2m``,
    which is unavailable in typical GPU containers (no V4L2 encoder), causing noisy failures
    before falling back. Hosts with working libx264 can set ``VIDEO_WRITER_FOURCC_LIST`` explicitly.
    """
    if os.path.isfile("/.dockerenv"):
        return "mp4v,avc1,H264,X264"
    if os.environ.get("VIDEO_WRITER_PREFER_SOFTWARE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return "mp4v,avc1,H264,X264"
    return "avc1,H264,X264,mp4v"


def _open_cv_video_writer(path: str, fps: float, frame_wh: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    """Open a VideoWriter, preferring H.264 fourccs so MP4s play in Cursor/browser (not only VLC).

    Default ``mp4v`` (MPEG-4 Part 2) often fails in in-editor HTML5 preview on the *host*; in
    **Docker** the default order puts ``mp4v`` first to avoid V4L2 hardware-encoder errors.

    Override: ``VIDEO_WRITER_FOURCC_LIST=avc1,H264,X264,mp4v`` or ``VIDEO_WRITER_PREFER_SOFTWARE=1``.
    """
    w, h = frame_wh
    raw = (os.environ.get("VIDEO_WRITER_FOURCC_LIST") or _default_video_writer_fourcc_list()).strip()
    tags = [t.strip() for t in raw.split(",") if len(t.strip()) == 4]
    if not tags:
        tags = ["mp4v"]
    for tag in tags:
        four = cv2.VideoWriter_fourcc(*tag)
        wr = cv2.VideoWriter(path, four, float(fps), (w, h))
        if wr.isOpened():
            return wr, tag
    raise RuntimeError(f"Could not open VideoWriter for {path!r} (tried fourccs: {tags})")


def _finalize_detections_mp4_libx264(path: str) -> None:
    """Re-encode written MP4 to H.264 + yuv420p + faststart for players that reject OpenCV mp4v.

    Many editors and browsers won't play MPEG-4 Part 2 (``mp4v``) in an MP4; VLC often will.
    """
    if not path or not os.path.isfile(path):
        return
    v = os.environ.get("DETECTIONS_MP4_H264_FINALIZE", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return
    if not shutil.which("ffmpeg"):
        return
    tmp = path + ".tmp_h264.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                path,
                "-c:v",
                "libx264",
                "-preset",
                (os.environ.get("DETECTIONS_MP4_H264_PRESET") or "fast").strip(),
                "-crf",
                (os.environ.get("DETECTIONS_MP4_H264_CRF") or "23").strip(),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                tmp,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        os.replace(tmp, path)
        print(
            f"[{_ts()}] Re-encoded detections to H.264 (libx264, yuv420p, faststart) for playback: {path}"
        )
    except subprocess.CalledProcessError as e:
        for p in (tmp,):
            try:
                if os.path.isfile(p):
                    os.unlink(p)
            except OSError:
                pass
        err = (e.stderr or e.stdout or str(e))[:500]
        print(
            f"[{_ts()}] WARNING: ffmpeg finalize failed; leaving OpenCV-encoded file. {err}"
        )


# OpenCV often reports CAP_PROP_FRAME_COUNT=0 or a bogus huge value for MP4s.
_MAX_SANE_OPENCV_FRAME_COUNT = 2_000_000


def _estimated_video_frame_count(path: str) -> int:
    """Frame count from container metadata via ffprobe (no full decode)."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,duration,avg_frame_rate,r_frame_rate",
                "-of",
                "json",
                path,
            ],
            text=True,
            timeout=60,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return 0

    def _parse_rate(r: str) -> float:
        if not r or r == "0/0":
            return 0.0
        if "/" in r:
            a, b = r.split("/", 1)
            try:
                return float(a) / float(b) if float(b) else 0.0
            except ValueError:
                return 0.0
        try:
            return float(r)
        except ValueError:
            return 0.0

    try:
        data = json.loads(out)
        streams = data.get("streams") or []
        if not streams:
            return 0
        s = streams[0]
        nb = s.get("nb_frames")
        if nb not in (None, "", "N/A"):
            try:
                n = int(str(nb).strip())
                if n > 0:
                    return n
            except ValueError:
                pass
        fps = _parse_rate(str(s.get("avg_frame_rate") or s.get("r_frame_rate") or ""))
        dur_s = s.get("duration")
        if dur_s and fps > 1e-6:
            try:
                d = float(dur_s)
                return max(0, int(round(d * fps)))
            except ValueError:
                pass
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    return 0


def _open_capture(url: str) -> cv2.VideoCapture:
    if _is_local_file_source(url):
        path = url[7:] if url.startswith("file://") else url
        # Prefer FFmpeg backend for local MP4s; container OpenCV may default
        # to GStreamer which can fail on some recordings.
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open local video: {path}")
        return cap
    pipeline = _build_gst_pipeline(url)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open stream: {url}\nPipeline: {pipeline}")
    return cap


# ── Detection drawing ──────────────────────────────────────────────────────────

# PeopleNet class colours: person=green, bag=amber, face=cyan
_CLASS_COLORS = {
    "person": (50, 220, 50),
    "bag":    (40, 180, 255),
    "face":   (255, 210, 50),
    # scaffolding (won't be used with PeopleNet)
    "car":     (50, 170, 255),
    "truck":   (50, 170, 255),
    "bus":     (50, 170, 255),
}
_DEFAULT_COLOR = (160, 160, 160)
_FONT          = cv2.FONT_HERSHEY_SIMPLEX


def _class_color(label: str) -> tuple:
    return _CLASS_COLORS.get(label.lower(), _DEFAULT_COLOR)


def draw_detections(frame, detections, net) -> None:
    for d in detections:
        x1, y1 = int(d.Left), int(d.Top)
        x2, y2 = int(d.Right), int(d.Bottom)
        label   = net.GetClassDesc(d.ClassID)
        color   = _class_color(label)
        tid = getattr(d, "track_id", None)
        text = (
            f"{label} #{int(tid)}  {d.Confidence:.0%}"
            if tid is not None
            else f"{label}  {d.Confidence:.0%}"
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background pill
        (tw, th), _ = cv2.getTextSize(text, _FONT, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 4),
                    _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


# ── Smart zoom ─────────────────────────────────────────────────────────────────

def _bbox_area(det) -> float:
    return (det.Right - det.Left) * (det.Bottom - det.Top)

def _bbox_iou_xyxy(a: tuple[float, float, float, float],
                   b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / (a_area + b_area - inter)

def _det_xyxy(det) -> tuple[float, float, float, float]:
    return (float(det.Left), float(det.Top), float(det.Right), float(det.Bottom))


def _merge_detections(detections, net, iou_thr: float = 0.08):
    """Greedy merge of overlapping detections into fewer boxes.

    This reduces the common PeopleNet "multiple boxes across one body" effect
    by clustering boxes whose IoU is above `iou_thr`, then taking the union
    rectangle and a representative class.
    """
    if not detections:
        return []

    # Sort by confidence so the most reliable box seeds clusters.
    ordered = sorted(detections, key=lambda d: float(d.Confidence), reverse=True)
    clusters: list[dict] = []

    def _cluster_iou_with_box(cl, det_xyxy):
        return _bbox_iou_xyxy(det_xyxy, (cl["left"], cl["top"], cl["right"], cl["bottom"]))

    for d in ordered:
        d_xyxy = _det_xyxy(d)
        placed = False
        for cl in clusters:
            if _cluster_iou_with_box(cl, d_xyxy) >= iou_thr:
                cl["members"].append(d)
                cl["left"] = min(cl["left"], d_xyxy[0])
                cl["top"] = min(cl["top"], d_xyxy[1])
                cl["right"] = max(cl["right"], d_xyxy[2])
                cl["bottom"] = max(cl["bottom"], d_xyxy[3])
                placed = True
                break
        if not placed:
            clusters.append({
                "left":   d_xyxy[0],
                "top":    d_xyxy[1],
                "right":  d_xyxy[2],
                "bottom": d_xyxy[3],
                "members":[d],
            })

    merged = []
    for cl in clusters:
        members = cl["members"]
        # Choose representative member: highest priority tier, then highest confidence.
        rep = min(members, key=lambda m: (_zoom_priority(m, net), -float(m.Confidence)))
        merged.append(SimpleNamespace(
            ClassID=rep.ClassID,
            Confidence=max(float(m.Confidence) for m in members),
            Left=cl["left"],
            Top=cl["top"],
            Right=cl["right"],
            Bottom=cl["bottom"],
        ))

    return merged


def _zoom_priority(det, net) -> int:
    """Lower number = higher zoom priority. Unknown classes get priority 99."""
    return _ZOOM_PRIORITY.get(net.GetClassDesc(det.ClassID).lower(), 99)


def select_focus_detection(detections, net):
    """
    Pick the detection to zoom into using PeopleNet priority:
    Face (0) > Person (1) > Bag (2) > other.
    Within the same priority tier, choose the largest bounding box.
    """
    if not detections:
        return None
    return min(detections,
               key=lambda d: (_zoom_priority(d, net), -_bbox_area(d)))


def _face_person_union_for_crop(focus, detections, net):
    """Return a bbox-like namespace (Left/Top/Right/Bottom) unioning face+person if both exist.

    Pairing uses face center inside person, or person center inside face, or IoU ≥ 0.02.
    """
    if not FACE_PERSON_UNION_FRAMING or focus is None or not detections:
        return None

    def _lab(d):
        return net.GetClassDesc(d.ClassID).lower()

    fl = _lab(focus)
    if fl not in ("face", "person"):
        return None
    faces = [d for d in detections if _lab(d) == "face"]
    persons = [d for d in detections if _lab(d) == "person"]
    if not faces or not persons:
        return None

    partner = None
    if fl == "face":
        fx = (float(focus.Left) + float(focus.Right)) / 2.0
        fy = (float(focus.Top) + float(focus.Bottom)) / 2.0
        for p in persons:
            if (
                p.Left <= fx <= p.Right and p.Top <= fy <= p.Bottom
            ) or _bbox_iou_xyxy(_det_xyxy(focus), _det_xyxy(p)) >= 0.02:
                partner = p
                break
    else:
        for f in faces:
            cx = (float(f.Left) + float(f.Right)) / 2.0
            cy = (float(f.Top) + float(f.Bottom)) / 2.0
            if (
                focus.Left <= cx <= focus.Right and focus.Top <= cy <= focus.Bottom
            ) or _bbox_iou_xyxy(_det_xyxy(focus), _det_xyxy(f)) >= 0.02:
                partner = f
                break
    if partner is None:
        return None
    return SimpleNamespace(
        Left=min(float(focus.Left), float(partner.Left)),
        Top=min(float(focus.Top), float(partner.Top)),
        Right=max(float(focus.Right), float(partner.Right)),
        Bottom=max(float(focus.Bottom), float(partner.Bottom)),
    )


def compute_target_crop(det, img_w: int, img_h: int, class_desc: str | None = None) -> tuple:
    bw = det.Right - det.Left
    bh = det.Bottom - det.Top
    # Prefer tighter crops when the subject is far away (small bbox height).
    depth_frac = float(bh) / float(img_h) if img_h else 0.0
    if depth_frac <= ZOOM_PADDING_DIST_FAR_FRAC:
        pad_mul = ZOOM_PADDING_DIST_FAR_MUL
    elif depth_frac >= ZOOM_PADDING_DIST_NEAR_FRAC:
        pad_mul = ZOOM_PADDING_DIST_NEAR_MUL
    else:
        t = (depth_frac - ZOOM_PADDING_DIST_FAR_FRAC) / max(
            1e-6, (ZOOM_PADDING_DIST_NEAR_FRAC - ZOOM_PADDING_DIST_FAR_FRAC)
        )
        pad_mul = ZOOM_PADDING_DIST_FAR_MUL + t * (ZOOM_PADDING_DIST_NEAR_MUL - ZOOM_PADDING_DIST_FAR_MUL)

    px = bw * ZOOM_PADDING * pad_mul
    py = bh * ZOOM_PADDING * pad_mul

    # For face crops: PeopleNet face boxes tend to be tight around the head,
    # which can cut off chin/shoulders when the camera zooms in.
    if class_desc and class_desc.lower() == "face":
        px *= float(_settings.get("face_padding_mul", 1.8))
        py *= float(_settings.get("face_padding_mul", 1.8))

        cy = (det.Top + det.Bottom) / 2.0 + bh * float(_settings.get("face_y_bias", 0.15))
        cx = (det.Left + det.Right) / 2.0

        x1 = max(0,     int(cx - (bw / 2.0 + px)))
        y1 = max(0,     int(cy - (bh / 2.0 + py)))
        x2 = min(img_w, int(cx + (bw / 2.0 + px)))
        y2 = min(img_h, int(cy + (bh / 2.0 + py)))
    else:
        x1 = max(0,     int(det.Left   - px))
        y1 = max(0,     int(det.Top    - py))
        x2 = min(img_w, int(det.Right  + px))
        y2 = min(img_h, int(det.Bottom + py))

    cls_l = (class_desc or "").lower()
    if cls_l == "person" and ZOOM_PERSON_TOP_PAD_FRAC > 0.0:
        y1 = max(0, int(y1 - bh * ZOOM_PERSON_TOP_PAD_FRAC))

    # Enforce minimum crop size so small/distant detections don't over-zoom
    min_w = int(img_w * _settings["zoom_min_fraction"])
    min_h = int(img_h * _settings["zoom_min_fraction"])

    if (x2 - x1) < min_w:
        cx = (x1 + x2) // 2
        x1 = max(0,     cx - min_w // 2)
        x2 = min(img_w, x1 + min_w)
        x1 = max(0,     x2 - min_w)    # re-clamp if x2 was clipped

    if (y2 - y1) < min_h:
        cy = (y1 + y2) // 2
        y1 = max(0,     cy - min_h // 2)
        y2 = min(img_h, y1 + min_h)
        y1 = max(0,     y2 - min_h)    # re-clamp if y2 was clipped

    # Match the frame's aspect ratio so the stretched output isn't distorted
    frame_ar = img_w / img_h
    cw, ch   = x2 - x1, y2 - y1
    if cw / ch < frame_ar:
        # Crop is too tall — expand width
        new_w = int(ch * frame_ar)
        cx    = (x1 + x2) // 2
        x1    = max(0,     cx - new_w // 2)
        x2    = min(img_w, x1 + new_w)
        x1    = max(0,     x2 - new_w)
    else:
        # Crop is too wide — expand height
        new_h = int(cw / frame_ar)
        if ZOOM_AR_HEIGHT_ANCHOR == "bottom":
            y2 = min(img_h, y2)
            y1 = max(0, int(y2 - new_h))
            y2 = min(img_h, y1 + new_h)
            if y2 - y1 < new_h:
                y1 = max(0, img_h - new_h)
                y2 = img_h
        elif ZOOM_AR_HEIGHT_ANCHOR == "top":
            y1 = max(0, y1)
            y2 = min(img_h, int(y1 + new_h))
            if y2 - y1 < new_h:
                y2 = img_h
                y1 = max(0, y2 - new_h)
        else:
            cy = (y1 + y2) // 2
            y1 = max(0, cy - new_h // 2)
            y2 = min(img_h, y1 + new_h)
            y1 = max(0, y2 - new_h)

    return (x1, y1, x2, y2)


def lerp_crop(current: tuple | None, target: tuple | None,
              img_w: int, img_h: int) -> tuple | None:
    full = (0, 0, img_w, img_h)
    src  = current if current is not None else full
    dst  = target  if target  is not None else full
    nxt  = tuple(int(s + ZOOM_SMOOTH * (d - s)) for s, d in zip(src, dst))
    if target is None and all(abs(n - f) <= 4 for n, f in zip(nxt, full)):
        return None
    return nxt


def apply_zoom(bgr, crop: tuple | None, out_w: int, out_h: int):
    if crop is None:
        return bgr
    x1, y1, x2, y2 = crop
    region = bgr[y1:y2, x1:x2]
    if region.size == 0:
        return bgr
    return cv2.resize(region, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def _lerp_rect_f(
    full: tuple[int, int, int, int],
    zoom: tuple[int, int, int, int],
    t: float,
) -> tuple[int, int, int, int]:
    """Linear blend between two axis-aligned rects; t=0 full frame, t=1 zoom rect."""
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(round(full[i] * (1.0 - t) + zoom[i] * t)) for i in range(4))


# ── Scene-relative overlay position ───────────────────────────────────────────
# When the view is zoomed (cropped), map a scene position (full-frame coords) to
# the current frame (zoomed display, possibly downscaled). Clamp so overlay stays on-screen.
def _scene_to_frame(sx: float, sy: float, overlay_w: int, overlay_h: int,
                    crop: tuple | None, full_w: int, full_h: int,
                    frame_w: int, frame_h: int) -> tuple[int, int]:
    if crop is None:
        return (int(sx * frame_w / full_w), int(sy * frame_h / full_h))
    x1, y1, x2, y2 = crop
    cw, ch = x2 - x1, y2 - y1
    dx = (sx - x1) / cw * frame_w
    dy = (sy - y1) / ch * frame_h
    dx = max(0, min(frame_w - overlay_w, dx))
    dy = max(0, min(frame_h - overlay_h, dy))
    return (int(dx), int(dy))


# ── Package cam overlay ────────────────────────────────────────────────────────

def apply_pkg_overlay(frame, fill_screen: bool = False, scale: float = 1.0,
                     crop: tuple | None = None, full_size: tuple | None = None):
    """Draw package camera thumbnail at bottom-right. scale (0.5–4) sizes it like the info overlay."""
    with _pkg_lock:
        pkg = _latest_pkg_bgr
    if pkg is None:
        return frame

    h, w = frame.shape[:2]
    s    = max(0.5, min(4.0, scale))
    frac = min(0.5, OVERLAY_FRACTION * s)   # cap so thumbnail doesn’t dominate
    tw   = max(1, int(w * frac))
    th   = max(1, int(pkg.shape[0] * tw / pkg.shape[1]))
    thumb = cv2.resize(pkg, (tw, th), interpolation=cv2.INTER_LINEAR)

    margin = max(OVERLAY_MARGIN, int(OVERLAY_MARGIN * s))

    if crop is not None and full_size is not None:
        fw, fh = full_size
        sx = fw - tw - margin
        sy = fh - th - margin
        x0, y0 = _scene_to_frame(sx, sy, tw, th, crop, fw, fh, w, h)
    else:
        x0 = w - tw - margin
        y0 = h - th - margin

    border = max(1, int(2 * s))
    cv2.rectangle(frame, (x0 - border, y0 - border), (x0 + tw + border, y0 + th + border), (30, 30, 30), border)
    frame[y0:y0 + th, x0:x0 + tw] = thumb
    return frame


def _pkg_camera_thread():
    global _latest_pkg_bgr
    while True:
        try:
            cap = _open_capture(PKG_RTSP_URL)
            print("[pkg-cam] Stream open")
            while True:
                ret, bgr = cap.read()
                if not ret or bgr is None:
                    break
                with _pkg_lock:
                    _latest_pkg_bgr = bgr.copy()
            cap.release()
        except Exception as e:
            print(f"[pkg-cam] ERROR: {e} — retrying in 5 s")
        time.sleep(5)


# ── Weather ────────────────────────────────────────────────────────────────────

_WMO_CODES = {
    0: "Clear",        1: "Mostly Clear",  2: "Partly Cloudy", 3: "Overcast",
    45: "Foggy",       48: "Icy Fog",
    51: "Drizzle",     53: "Drizzle",      55: "Drizzle",
    61: "Rain",        63: "Rain",         65: "Rain",
    71: "Snow",        73: "Snow",         75: "Snow",         77: "Snow",
    80: "Showers",     81: "Showers",      82: "Showers",
    85: "Snow Shower", 86: "Snow Shower",
    95: "T-Storm",     96: "T-Storm",      99: "T-Storm",
}

# ── Weather icon drawing ───────────────────────────────────────────────────────
# All colors in BGR (OpenCV convention).
_C_SUN   = (0,   200, 255)   # warm golden yellow
_C_CLOUD = (210, 210, 210)   # clean light gray
_C_RAIN  = (200, 130,  60)   # sky blue
_C_SNOW  = (240, 238, 225)   # near-white ice
_C_STORM = (20,  190, 255)   # bright golden (lightning)
_C_FOG   = (160, 160, 160)   # muted gray


def _cloud(img, cx, cy, r, color=_C_CLOUD):
    """Three-bump cloud silhouette drawn with filled circles + ellipse."""
    cv2.ellipse(img, (cx, cy + r // 5), (int(r * .70), int(r * .40)), 0, 0, 360, color, -1)
    cv2.circle(img, (cx - r // 3,  cy - r // 10), r // 3,          color, -1)
    cv2.circle(img, (cx - r // 9,  cy - r // 4),  int(r * .38),    color, -1)
    cv2.circle(img, (cx + r // 4,  cy - r // 8),  int(r * .28),    color, -1)


def _wx_clear(img, cx, cy, r):
    disc = max(3, r * 5 // 12)
    cv2.circle(img, (cx, cy), disc, _C_SUN, -1)
    for i in range(8):
        a = math.radians(i * 45)
        ca, sa = math.cos(a), math.sin(a)
        cv2.line(img,
                 (cx + int((disc + 2) * ca), cy + int((disc + 2) * sa)),
                 (cx + int((r - 1)   * ca), cy + int((r - 1)   * sa)),
                 _C_SUN, max(1, r // 9))


def _wx_partly(img, cx, cy, r):
    _wx_clear(img, cx - r // 5, cy - r // 5, max(3, int(r * .65)))
    _cloud(img, cx + r // 9, cy + r // 9, int(r * .78))


def _wx_cloud(img, cx, cy, r):
    _cloud(img, cx, cy, r)


def _wx_rain(img, cx, cy, r):
    _cloud(img, cx, cy - r // 5, int(r * .72))
    rl = int(r * .30)
    for i in range(3):
        dx = (i - 1) * (r // 3)
        cv2.line(img, (cx + dx, cy + r // 4),
                 (cx + dx - rl // 2, cy + r // 4 + rl), _C_RAIN, max(1, r // 8))


def _wx_snow(img, cx, cy, r):
    _cloud(img, cx, cy - r // 5, int(r * .72))
    dr = max(1, r // 9)
    for i in range(3):
        cv2.circle(img, (cx + (i - 1) * (r // 3), cy + int(r * .42)), dr, _C_SNOW, -1)


def _wx_storm(img, cx, cy, r):
    _cloud(img, cx, cy - r // 5, int(r * .72))
    pts = [
        (cx + r // 6,  cy - r // 10),
        (cx - r // 10, cy + r // 8),
        (cx + r // 12, cy + r // 8),
        (cx - r // 8,  cy + r // 2),
    ]
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], _C_STORM, max(1, r // 7))


def _wx_fog(img, cx, cy, r):
    lw = max(1, r // 8)
    for i in range(3):
        dy = (i - 1) * (r // 3)
        cv2.line(img, (cx - int(r * .70), cy + dy), (cx + int(r * .70), cy + dy), _C_FOG, lw)


_WX_ICON_FN = {
    0: _wx_clear,  1: _wx_clear,
    2: _wx_partly,
    3: _wx_cloud,
    45: _wx_fog,   48: _wx_fog,
    51: _wx_rain,  53: _wx_rain,  55: _wx_rain,
    61: _wx_rain,  63: _wx_rain,  65: _wx_rain,
    71: _wx_snow,  73: _wx_snow,  75: _wx_snow,  77: _wx_snow,
    80: _wx_rain,  81: _wx_rain,  82: _wx_rain,
    85: _wx_snow,  86: _wx_snow,
    95: _wx_storm, 96: _wx_storm, 99: _wx_storm,
}


def _draw_wx_icon(img, cx, cy, r, wmo_code):
    _WX_ICON_FN.get(wmo_code, _wx_cloud)(img, cx, cy, r)


# ── Weather fetch ──────────────────────────────────────────────────────────────

def _fetch_location() -> tuple[float, float]:
    if WEATHER_LAT and WEATHER_LON:
        return float(WEATHER_LAT), float(WEATHER_LON)
    r   = requests.get("https://ipinfo.io/json", timeout=5)
    loc = r.json()["loc"].split(",")
    return float(loc[0]), float(loc[1])


def _fetch_weather(lat: float, lon: float) -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=weathercode,temperature_2m"
        f"&forecast_days=1&timezone=auto"
    )
    resp   = requests.get(url, timeout=8).json()
    tz_str = resp.get("timezone", "")
    codes  = resp["hourly"]["weathercode"]   # 24 values, index = local hour
    temps  = resp["hourly"].get("temperature_2m", [])  # °C from API

    # Use the API-provided timezone for the current local hour
    now_h = datetime.datetime.now().hour
    try:
        from zoneinfo import ZoneInfo
        now_h = datetime.datetime.now(ZoneInfo(tz_str)).hour
    except Exception:
        pass

    curr_code = int(codes[now_h]) if now_h < len(codes) else 0
    temp_c    = float(temps[now_h]) if now_h < len(temps) and temps[now_h] is not None else None
    if temp_c is not None and WEATHER_UNIT.lower() == "fahrenheit":
        temp_display = round(temp_c * 9 / 5 + 32)
        temp_unit    = " F"   # plain ASCII; OpenCV putText often fails on °
    elif temp_c is not None:
        temp_display = round(temp_c)
        temp_unit    = " C"
    else:
        temp_display = None
        temp_unit    = " C"

    # Find when the condition next changes (scan remaining hours today)
    hours_until = None
    next_code   = None
    next_time   = None
    for h in range(now_h + 1, min(24, len(codes))):
        if int(codes[h]) != curr_code:
            hours_until = h - now_h
            next_code   = int(codes[h])
            nh12 = h % 12 or 12
            next_time = f"{nh12} {'AM' if h < 12 else 'PM'}"
            break

    return {
        "current_code":      curr_code,
        "current_condition": _WMO_CODES.get(curr_code, "Unknown"),
        "temp_display":      temp_display,   # int or None
        "temp_unit":        temp_unit,       # " F" or " C"
        "hours_until":       hours_until,   # None = same rest of day
        "next_code":         next_code,
        "next_condition":    _WMO_CODES.get(next_code, "") if next_code is not None else None,
        "next_time":         next_time,
        "timezone":          tz_str,
    }


def _weather_thread():
    global _weather_cache
    while True:
        try:
            lat, lon = _fetch_location()
            data     = _fetch_weather(lat, lon)
            with _weather_lock:
                _weather_cache = data
            nxt = f"→ {data['next_condition']} {data['next_time']}" if data["next_condition"] else "same all day"
            print(f"[weather] {data['current_condition']}  ({data['hours_until']}h)  {nxt}  tz={data['timezone']}")
        except Exception as e:
            print(f"[weather] ERROR: {e}")
        time.sleep(WEATHER_REFRESH)


# ── Overlay data for HTML (time/weather) ───────────────────────────────────────
def _build_overlay_data() -> dict | None:
    """Build dict for frontend clock/weather overlay. Uses _weather_cache + local time."""
    with _weather_lock:
        wx = _weather_cache
    tz = None
    if wx and wx.get("timezone"):
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(wx["timezone"])
        except Exception:
            pass
    now = datetime.datetime.now(tz) if tz else datetime.datetime.now()
    hour = now.hour % 12 or 12
    time_str = f"{hour}:{now.strftime('%M')} {'AM' if now.hour < 12 else 'PM'}"
    date_str = f"{now.strftime('%a %b')} {now.day}"
    out = {"time": time_str, "date": date_str}
    if wx:
        out["condition"] = wx.get("current_condition", "")
        out["current_code"] = wx.get("current_code")
        out["temp"] = wx.get("temp_display")
        out["temp_unit"] = wx.get("temp_unit", " C")
        out["hours"] = wx.get("hours_until")
        out["next_condition"] = wx.get("next_condition")
        out["next_time"] = wx.get("next_time")
        out["next_code"] = wx.get("next_code")
    else:
        out["condition"] = ""
        out["current_code"] = None
        out["temp"] = None
        out["temp_unit"] = " C"
        out["hours"] = None
        out["next_condition"] = None
        out["next_time"] = None
        out["next_code"] = None
    return out


# ── Date / time / weather overlay (top-right) ─────────────────────────────────

def render_info_overlay(frame, scale: float = 1.0, fill_screen: bool = False,
                        crop: tuple | None = None, full_size: tuple | None = None) -> None:
    """Render clock, date, and weather (condition + outdoor temp) at top-right, in-place.
    If crop and full_size are set, position is scene-relative so it moves with zoom.

    Layout (Apple-inspired):
        9:45 AM
        Sat Mar 14
        [icon] Clear  72 F  3h
        [icon] Cloudy  5 PM   ← only if condition changes today
    """
    with _weather_lock:
        wx = _weather_cache

    fh, fw = frame.shape[:2]
    s      = max(0.5, min(4.0, scale))
    pad    = max(4, int(9  * s))
    lh     = max(8, int(26 * s))
    icon_r = max(4, int(lh * 0.40))       # icon fits inside line height
    icon_w = icon_r * 2 + max(3, int(5 * s))  # icon diameter + gap

    # ── Build time/date using API timezone if available ────────────────────
    tz = None
    if wx and wx.get("timezone"):
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(wx["timezone"])
        except Exception:
            pass
    now  = datetime.datetime.now(tz) if tz else datetime.datetime.now()
    hour = now.hour % 12 or 12          # 12-hour, no leading zero
    time_str = f"{hour}:{now.strftime('%M')} {'AM' if now.hour < 12 else 'PM'}"
    date_str = f"{now.strftime('%a %b')} {now.day}"   # "Sat Mar 14" — int avoids leading zero

    # ── Text-only lines (time, date) ───────────────────────────────────────
    text_lines = [
        (time_str, 0.80 * s, 2),
        (date_str, 0.50 * s, 1),
    ]

    # ── Weather icon lines ─────────────────────────────────────────────────
    wx_lines = []   # (wmo_code, label_str, font_scale, thickness)
    if wx:
        cond  = wx["current_condition"]
        hours = wx["hours_until"]
        temp  = wx.get("temp_display")
        temp_u = wx.get("temp_unit", " C")
        # e.g. "Clear  72 F  3h" or "Clear  72 F"
        temp_str = f"  {temp}{temp_u}" if temp is not None else ""
        label = f"{cond}{temp_str}  {hours}h" if hours else (f"{cond}{temp_str}" if temp_str else cond)
        wx_lines.append((wx["current_code"], label, 0.48 * s, 1))

        if wx.get("next_condition"):
            nxt_label = f"{wx['next_condition']}   {wx['next_time']}"
            wx_lines.append((wx["next_code"], nxt_label, 0.46 * s, 1))
    else:
        text_lines.append(("Weather...", 0.46 * s, 1))

    # ── Measure box dimensions ─────────────────────────────────────────────
    all_text_w = [cv2.getTextSize(t, _FONT, fs, th)[0][0]
                  for t, fs, th in text_lines]
    wx_text_w  = [cv2.getTextSize(t, _FONT, fs, th)[0][0] + icon_w
                  for _, t, fs, th in wx_lines]
    max_w  = max(all_text_w + wx_text_w) if (all_text_w or wx_text_w) else 80
    n_rows = len(text_lines) + len(wx_lines)
    box_w  = max_w + pad * 2
    box_h  = n_rows * lh + pad * 2
    margin = 10
    top    = margin
    if crop is not None and full_size is not None:
        full_w, full_h = full_size
        sx = full_w - box_w - margin
        sy = top
        x0, y0 = _scene_to_frame(sx, sy, box_w, box_h, crop, full_w, full_h, fw, fh)
    else:
        x0 = fw - box_w - margin
        y0 = top

    if x0 < 0 or y0 + box_h > fh:
        return

    # ── Background ─────────────────────────────────────────────────────────
    roi = frame[y0:y0 + box_h, x0:x0 + box_w]
    bg  = roi.copy()
    cv2.rectangle(roi, (0, 0), (box_w, box_h), (10, 10, 10), -1)
    cv2.addWeighted(bg, 0.35, roi, 0.65, 0, roi)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (70, 70, 70), 1)

    # ── Text-only rows ─────────────────────────────────────────────────────
    for i, (text, fs, th) in enumerate(text_lines):
        ty = y0 + pad + (i + 1) * lh - 4
        cv2.putText(frame, text, (x0 + pad, ty), _FONT, fs, (230, 230, 230), th, cv2.LINE_AA)

    # ── Weather icon rows ──────────────────────────────────────────────────
    for j, (code, text, fs, th) in enumerate(wx_lines):
        row = len(text_lines) + j
        ty  = y0 + pad + (row + 1) * lh - 4
        icx = x0 + pad + icon_r
        icy = y0 + pad + row * lh + lh // 2
        _draw_wx_icon(frame, icx, icy, icon_r, code)
        cv2.putText(frame, text, (x0 + pad + icon_w, ty),
                    _FONT, fs, (230, 230, 230), th, cv2.LINE_AA)


# ── Web server (implementation lives in web_server.py) ────────────────────────
# Imported lazily in main() so the heavy Flask import only runs when needed.
_HTML = """<!-- placeholder — replaced by web_server.py -->
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DoorbellAIVision</title>
<style>
  :root {
    --bg:      #0b0d0f;
    --surface: #111518;
    --border:  #1e2328;
    --blue:    #4d9cf8;
    --green:   #3fb950;
    --red:     #f85149;
    --muted:   #6e7681;
    --text:    #cdd9e5;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:'Courier New',monospace; height:100vh; display:flex; flex-direction:column; }
  header {
    display:flex; align-items:center; gap:12px;
    padding:10px 20px; background:var(--surface);
    border-bottom:1px solid var(--border); flex-shrink:0;
  }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--green); animation:pulse 2s infinite; }
  .dot.offline { background:var(--red); animation:none; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  header h1 { font-size:.95rem; letter-spacing:.15em; color:var(--blue); }
  .hstat { margin-left:auto; font-size:.75rem; color:var(--muted); }
  .hstat span { color:var(--text); }
  .body { display:flex; flex:1; overflow:hidden; }
  .video-wrap {
    flex:1; background:#000;
    display:flex; align-items:center; justify-content:center; overflow:hidden;
  }
  .video-wrap img { max-width:100%; max-height:100%; object-fit:contain; display:block; }
  .no-signal { display:none; flex-direction:column; align-items:center; gap:12px; color:var(--muted); font-size:.85rem; }
  .no-signal svg { opacity:.3; }
  aside { width:300px; flex-shrink:0; border-left:1px solid var(--border); display:flex; flex-direction:column; background:var(--surface); }
  .aside-title { padding:10px 14px; font-size:.7rem; text-transform:uppercase; letter-spacing:.12em; color:var(--muted); border-bottom:1px solid var(--border); flex-shrink:0; }
  .log { flex:1; overflow-y:auto; }
  .log::-webkit-scrollbar { width:4px; }
  .log::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }
  .det { padding:9px 14px; border-bottom:1px solid var(--border); }
  .det:hover { background:rgba(255,255,255,.02); }
  .det-label { font-size:.85rem; color:var(--blue); }
  .det-conf  { font-size:.75rem; color:var(--green); margin-top:1px; }
  .det-meta  { font-size:.68rem; color:var(--muted); margin-top:2px; }
  .empty     { padding:20px 14px; font-size:.78rem; color:var(--muted); }
  footer { padding:8px 14px; border-top:1px solid var(--border); font-size:.7rem; color:var(--muted); flex-shrink:0; }
  footer span { color:var(--text); }
</style>
</head>
<body>
<header>
  <div class="dot" id="dot"></div>
  <h1>DOORBELLAIVISION</h1>
  <div class="hstat">inference &nbsp;<span id="h-fps">--</span> fps</div>
</header>
<div class="body">
  <div class="video-wrap">
    <img id="feed" src="/video_feed" onload="setOnline(true)" onerror="setOnline(false)">
    <div class="no-signal" id="nosig">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path d="M3 3l18 18M10.584 10.587a2 2 0 002.828 2.83M6.343 6.346A8 8 0 0017.66 17.658M3.515 3.515A12 12 0 0020.485 20.485"/>
      </svg>No signal
    </div>
  </div>
  <aside>
    <div class="aside-title">Detections</div>
    <div class="log" id="log"><div class="empty">Waiting for detections…</div></div>
    <footer>Frames: <span id="frames">0</span></footer>
  </aside>
</div>
<script>
  function setOnline(on) {
    document.getElementById('dot').className        = 'dot' + (on ? '' : ' offline');
    document.getElementById('feed').style.display   = on ? 'block' : 'none';
    document.getElementById('nosig').style.display  = on ? 'none'  : 'flex';
  }
  function poll() {
    fetch('/api/state').then(r => r.json()).then(d => {
      document.getElementById('h-fps').textContent  = d.fps.toFixed(1);
      document.getElementById('frames').textContent = d.frames;
      const log = document.getElementById('log');
      if (!d.detections.length) { log.innerHTML = '<div class="empty">No detections yet…</div>'; return; }
      log.innerHTML = d.detections.map(det => `
        <div class="det">
          <div class="det-label">${det.label}</div>
          <div class="det-conf">${(det.confidence*100).toFixed(0)}% confidence</div>
          <div class="det-meta">${det.time} &nbsp;·&nbsp; frame ${det.frame}</div>
        </div>`).join('');
    }).catch(() => setOnline(false));
  }
  setInterval(poll, 750);
  poll();
</script>
</body>
</html>"""


def _start_web_server():
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import web_server
    web_server.start(_shared, port=WEB_PORT)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Inference worker ────────────────────────────────────────────────────────────

def _inference_worker(net) -> None:
    """Dedicated inference thread.

    Runs as fast as the model allows (YOLO11 on GPU: often 30–60+ fps).
    The main capture loop submits the latest frame (non-blocking) and reads back
    whatever detections are available, so it is never stalled waiting for the GPU.
    Detections are at most one inference cycle stale, imperceptible given the
    zoom's smooth lerp.
    """
    ev = _infer["ready"]
    stop = _infer["stop"]
    while True:
        if stop.is_set():
            break
        # Wake periodically so we observe ``stop`` without requiring another frame.
        if not ev.wait(timeout=1.0):
            continue
        ev.clear()
        if stop.is_set():
            break
        with _infer["lock"]:
            bgr = _infer["bgr"]
            detect_crop = _infer.get("detect_crop")
            infer_scale = float(_infer.get("infer_scale", 1.0) or 1.0)
        if bgr is None:
            continue
        try:
            # Allow live tuning of confidence threshold for far detections.
            # Live tuning: YOLO11 uses `self.threshold` as predict conf=.
            if hasattr(net, "threshold"):
                net.threshold = float(_settings.get("det_threshold", THRESHOLD))
            # Optionally run detection on a cropped view (faster + more accurate
            # because the subject is larger). Coordinates are mapped back to
            # the full-frame space below.
            offset_x = 0
            offset_y = 0
            det_input = bgr
            if detect_crop is not None:
                x1, y1, x2, y2 = detect_crop
                x1 = max(0, min(bgr.shape[1] - 1, int(x1)))
                y1 = max(0, min(bgr.shape[0] - 1, int(y1)))
                x2 = max(0, min(bgr.shape[1], int(x2)))
                y2 = max(0, min(bgr.shape[0], int(y2)))
                if x2 > x1 and y2 > y1:
                    offset_x = x1
                    offset_y = y1
                    det_input = bgr[y1:y2, x1:x2]

            # Optional resize (used by low-res snapshot schedule).
            orig_h, orig_w = det_input.shape[:2]
            if infer_scale != 1.0 and orig_w > 0 and orig_h > 0:
                new_w = max(1, int(round(float(orig_w) * float(infer_scale))))
                new_h = max(1, int(round(float(orig_h) * float(infer_scale))))
                sx = float(orig_w) / float(new_w)
                sy = float(orig_h) / float(new_h)
                det_input = cv2.resize(det_input, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                sx = sy = 1.0

            dets = net.Detect(det_input)

            if offset_x != 0 or offset_y != 0 or sx != 1.0 or sy != 1.0:
                mapped = []
                for d in dets:
                    mapped.append(SimpleNamespace(
                        ClassID=d.ClassID,
                        Confidence=float(d.Confidence),
                        Left=float(d.Left) * float(sx) + float(offset_x),
                        Top=float(d.Top) * float(sy) + float(offset_y),
                        Right=float(d.Right) * float(sx) + float(offset_x),
                        Bottom=float(d.Bottom) * float(sy) + float(offset_y),
                    ))
                dets = mapped
            with _infer["lock"]:
                _infer["dets"]     = dets
                _infer["frame_id"] += 1
        except Exception as e:
            print(f"[{_ts()}] [infer] ERROR: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    source_url = INPUT_VIDEO or RTSP_URL
    is_file_source = _is_local_file_source(source_url)
    print(f"[{_ts()}] Starting DoorbellAIVision")
    _iv_raw = (os.environ.get("INPUT_VIDEO") or "").strip()
    if _iv_raw and not is_file_source:
        print(
            f"[{_ts()}] WARNING: INPUT_VIDEO={_iv_raw!r} is set but is not a readable file inside "
            f"this container — using live RTSP. For Docker: pass the var (see docker-compose.yml "
            f"passthrough or use `docker compose run -e INPUT_VIDEO=...`)."
        )
    if is_file_source:
        print(f"  Input (local): {source_url}")
    else:
        print(f"  Main stream  : {RTSP_URL}")
        if not _iv_raw:
            print(
                f"  (No INPUT_VIDEO — reading live RTSP. RECORD_RTSP may write "
                f"{os.path.join(OUTPUT_DIR, RECORD_RTSP_NAME)}.)"
            )
    print(f"  Pkg stream   : {PKG_RTSP_URL}")
    print(f"  Web dashboard: {'enabled → :%d' % WEB_PORT if ENABLE_WEB else 'disabled'}")
    print(f"  Threshold    : {THRESHOLD}")
    if INFERENCE_ROI_NORM:
        print(f"  Inference ROI: {INFERENCE_ROI_NORM!r}")
    if ZOOM_ATTENTION_ROI_NORM:
        print(f"  Zoom focus ROI: {ZOOM_ATTENTION_ROI_NORM!r}")
    if DETECT_CLASS_FILTER:
        print(f"  Class filter : {', '.join(sorted(DETECT_CLASS_FILTER))}")
    if ZOOM_TARGET_EMA > 0.0:
        print(f"  Zoom smooth  : target EMA α={ZOOM_TARGET_EMA} (ZOOM_TARGET_EMA)")
    if ZOOM_FOCUS_IOU_MIN != 0.35:
        print(f"  Zoom hysteresis IoU ≥ {ZOOM_FOCUS_IOU_MIN} (ZOOM_FOCUS_IOU_MIN)")
    print(
        f"  Face+person union framing: "
        f"{'on' if FACE_PERSON_UNION_FRAMING else 'off'} (FACE_PERSON_UNION_FRAMING)"
    )
    print(
        f"  Motion prime (FAR snapshot): "
        f"{'on' if MOTION_PRIME_FAR_SNAPSHOT else 'off'} — "
        f"undo: MOTION_PRIME_FAR_SNAPSHOT=0"
    )
    if ZOOM_AR_HEIGHT_ANCHOR != "center":
        print(f"  AR height anchor: {ZOOM_AR_HEIGHT_ANCHOR!r} (ZOOM_AR_HEIGHT_ANCHOR)")
    if ZOOM_PERSON_TOP_PAD_FRAC > 0.0:
        print(
            f"  Person top pad: {ZOOM_PERSON_TOP_PAD_FRAC} (ZOOM_PERSON_TOP_PAD_FRAC)"
        )

    if ENABLE_WEB:
        threading.Thread(target=_start_web_server, daemon=True).start()

    # Only connect to the live package camera when we're running the live main stream.
    if not is_file_source and not DISABLE_PKG_CAM:
        threading.Thread(target=_pkg_camera_thread, daemon=True).start()
    threading.Thread(target=_weather_thread,    daemon=True).start()

    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

    from system_monitor import SystemMonitor
    _shared["perf_monitor"] = SystemMonitor()

    from vision import create_detector

    net = create_detector()
    print(f"[{_ts()}] Detector     : {(DETECTION_MODEL or 'stub')!r}")
    _cf = _effective_class_filter()
    if FACE_PERSON_UNION_FRAMING and _cf is not None and "face" not in _cf:
        print(
            f"[{_ts()}] NOTE: FACE_PERSON_UNION_FRAMING needs `face` in detections; "
            f"effective DETECT_CLASS_FILTER is {_cf!r} — union framing has no effect. "
            f"Use a detector with face+person classes, or add person top pad / AR anchor "
            f"(ZOOM_PERSON_TOP_PAD_FRAC, ZOOM_AR_HEIGHT_ANCHOR)."
        )

    cap = _open_capture(source_url)
    print(f"[{_ts()}] {'Local video' if is_file_source else 'Main stream'} open — running inference…\n")

    det_out = os.path.join(OUTPUT_DIR, DETECTIONS_VIDEO_NAME)
    if SAVE_DETECTIONS:
        print(f"[{_ts()}] SAVE_DETECTIONS=true → will write {det_out}")
    else:
        print(
            f"[{_ts()}] SAVE_DETECTIONS=false → no annotated video "
            f"(set SAVE_DETECTIONS=true; file is {det_out})."
        )

    # Stop after N frames for INPUT_VIDEO (OpenCV often repeats last frame forever at EOF).
    file_frame_limit = 0
    process_frame_limit = 0
    if is_file_source:
        vid_path = _local_video_path(source_url)
        fc_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps_raw = cap.get(cv2.CAP_PROP_FPS)
        try:
            fc_i = int(round(float(fc_raw)))
        except (TypeError, ValueError):
            fc_i = 0
        if fc_i < 0:
            fc_i = 0
        if fc_i > _MAX_SANE_OPENCV_FRAME_COUNT:
            print(
                f"[{_ts()}] Local file: ignoring suspicious CAP_PROP_FRAME_COUNT={fc_i} "
                f"(>{_MAX_SANE_OPENCV_FRAME_COUNT}); will try ffprobe / FILE_MAX_FRAMES."
            )
            fc_i = 0

        file_frame_limit = fc_i
        ff_est = 0
        if file_frame_limit <= 0:
            ff_est = _estimated_video_frame_count(vid_path)
            if ff_est > 0:
                file_frame_limit = ff_est

        # If FILE_START_TIME_SEC is set, override FILE_START_FRAME for this run.
        # This happens before frame-limit math and output gating.
        global FILE_START_FRAME
        if FILE_START_TIME_SEC > 0.0:
            try:
                fps_eff = float(fps_raw)
            except (TypeError, ValueError):
                fps_eff = 0.0
            if fps_eff <= 1e-6:
                fps_eff = 30.0  # conservative fallback
            # "Just after" means floor(t*fps) + 1 (next frame after boundary).
            new_start = int(math.floor(FILE_START_TIME_SEC * fps_eff)) + 1
            new_start = max(0, new_start)
            if fc_i > 0 and new_start >= fc_i:
                print(
                    f"[{_ts()}] WARNING: FILE_START_TIME_SEC={FILE_START_TIME_SEC} "
                    f"maps to FILE_START_FRAME={new_start}, which is past EOF "
                    f"(CAP_PROP_FRAME_COUNT={fc_i}). Nothing to process."
                )
                return
            FILE_START_FRAME = new_start
            print(
                f"[{_ts()}] FILE_START_TIME_SEC={FILE_START_TIME_SEC}s "
                f"(FPS≈{fps_eff:.3g}) → FILE_START_FRAME={FILE_START_FRAME}"
            )

        file_frame_limit_total = file_frame_limit
        # FILE_MAX_FRAMES is the chunk length; FILE_START_FRAME is the offset.
        if file_frame_limit_total <= 0:
            # If we can't reliably determine total frames, fall back to
            # processing at most FILE_MAX_FRAMES starting at FILE_START_FRAME.
            if FILE_MAX_FRAMES > 0:
                process_frame_limit = FILE_START_FRAME + FILE_MAX_FRAMES
            else:
                process_frame_limit = 0  # unlimited; we'll stop by EOF
        else:
            chunk_end = None
            if FILE_MAX_FRAMES > 0:
                chunk_end = FILE_START_FRAME + FILE_MAX_FRAMES
            else:
                chunk_end = file_frame_limit_total

            # Clamp end within the file's known frame count.
            process_frame_limit = min(file_frame_limit_total, chunk_end)

            if FILE_START_FRAME >= file_frame_limit_total:
                print(
                    f"[{_ts()}] FILE_START_FRAME={FILE_START_FRAME} is past EOF. "
                    f"Nothing to process."
                )
                return
        try:
            fps_f = float(fps_raw)
            fps_msg = f"{fps_f:.3g}" if fps_f > 1e-6 else "?"
        except (TypeError, ValueError):
            fps_msg = "?"
        if file_frame_limit > 0:
            parts = []
            if fc_i > 0 and fc_i <= _MAX_SANE_OPENCV_FRAME_COUNT:
                parts.append("OpenCV CAP_PROP_FRAME_COUNT")
            if ff_est > 0:
                parts.append("ffprobe metadata")
            if FILE_MAX_FRAMES > 0:
                parts.append("FILE_MAX_FRAMES cap")
            src = " + ".join(parts) if parts else "heuristic"
            print(
                f"[{_ts()}] Local file frame limit: {file_frame_limit} ({src}), "
                f"OpenCV FPS≈{fps_msg}"
            )
        else:
            print(
                f"[{_ts()}] Local file: no frame limit (CAP_PROP_FRAME_COUNT={fc_raw!r}, "
                f"ffprobe={ff_est}). Playback may never stop — set FILE_MAX_FRAMES=N or "
                f"fix MP4 metadata. OpenCV FPS≈{fps_msg}"
            )

        # process_frame_limit set above
    elif FILE_MAX_FRAMES > 0:
        process_frame_limit = FILE_MAX_FRAMES
        print(
            f"[{_ts()}] Live stream: will stop after {process_frame_limit} frames "
            f"(FILE_MAX_FRAMES) and finalize writers."
        )
    elif not is_file_source:
        print(
            f"[{_ts()}] WARNING: Live stream with FILE_MAX_FRAMES=0 → runs until Ctrl+C. "
            f"Set FILE_MAX_FRAMES=N in config/camera.env or pass "
            f"-e FILE_MAX_FRAMES=N to docker compose run."
        )

    # Ensure recordings are finalized on exit (Ctrl+C, SIGTERM, end-of-file) so MP4 moov atom is written.
    _cleanup: dict = {"cap": cap, "output": None, "raw_output": None, "detections_path": None, "raw_path": None}

    def _release_recordings():
        released = []
        for key in ("output", "raw_output", "cap"):
            obj = _cleanup.get(key)
            if obj is not None:
                try:
                    obj.release()
                    released.append(key)
                except Exception:
                    pass
                _cleanup[key] = None
        if released:
            dp = _cleanup.get("detections_path")
            rp = _cleanup.get("raw_path")
            if dp:
                print(f"[{_ts()}] Finalized detections video: {dp}")
                _finalize_detections_mp4_libx264(dp)
            if rp:
                print(f"[{_ts()}] Finalized raw recording: {rp}")
            print(f"[{_ts()}] Released capture/writers: {', '.join(released)}")

    atexit.register(_release_recordings)

    # Non-daemon + explicit shutdown: if this thread is still inside native Detect() when the
    # interpreter exits, some stacks (Ultralytics/OpenCV) can hit std::terminate — seen as
    # "terminate called without an active exception" and Docker exit 139.
    infer_thread = threading.Thread(target=_inference_worker, args=(net,), daemon=False)
    infer_thread.start()

    # cv2.VideoWriter is lazy-initialized after the first frame reveals resolution.
    output = None
    raw_output = None
    record_start_ts = None

    frame_count    = 0
    error_streak   = 0
    no_det_frames  = 0
    current_crop   = None
    prev_focus_xyxy = None     # last chosen focus bbox (full-res coords)
    prev_focus_pr   = None     # last chosen focus priority rank
    prev_focus_miss = 0       # consecutive frames without matching focus
    fps_time       = time.time()
    img_w = img_h  = None
    last_infer_fid = -1   # tracks which inference cycle we last logged

    far_detect_crop = None

    # FAR -> CLOSE mode state machine.
    # - FAR: run detector on far ROI (r3 via INFERENCE_ROI_NORM), and only allow
    #        detections whose center is inside the far attention ROI (r3 via
    #        ZOOM_ATTENTION_ROI_NORM) to become focus/zoom target.
    # - CLOSE: once focus bbox is "close" for CLOSE_SUSTAIN_FRAMES, disable
    #          attention gating so the full frame can take over.
    far_mode = True
    close_frames = 0

    prev_no_zoom_eligible = True
    prev_attn_gray_prime: np.ndarray | None = None
    last_motion_prime_frame = -10**9

    focus_target_ema: tuple[int, int, int, int] | None = None
    zoom_gate_smooth = 0.0
    last_target_zoom: tuple[int, int, int, int] | None = None
    class_filter = _effective_class_filter()

    # CSV trace (optional).
    trace_fp = None
    trace_writer = None
    prev_zoom_trigger = None
    if DEBUG_TRACE:
        trace_path = os.path.join(OUTPUT_DIR, DEBUG_TRACE_CSV_NAME)
        trace_fp = open(trace_path, "w", newline="")
        trace_writer = csv.writer(trace_fp)
        trace_writer.writerow([
            "frame",
            "det_count",
            "zoom_trigger",
            "target_x1", "target_y1", "target_x2", "target_y2",
            "current_x1", "current_y1", "current_x2", "current_y2",
        ])

    # Focus trace for far-lock evaluation.
    focus_trace_fp = None
    focus_trace_writer = None
    if DEBUG_FOCUS_TRACE:
        focus_trace_path = os.path.join(OUTPUT_DIR, DEBUG_FOCUS_TRACE_CSV_NAME)
        focus_trace_fp = open(focus_trace_path, "w", newline="")
        focus_trace_writer = csv.writer(focus_trace_fp)
        focus_trace_writer.writerow([
            "frame",
            "has_focus",
            "focus_left", "focus_top", "focus_right", "focus_bottom",
            "focus_cx", "focus_cy",
            # Normalize centers by frame height so stability thresholds are
            # resolution-independent (distance in "analysis units of 1=img_h").
            "focus_cx_by_h", "focus_cy_by_h",
            "focus_h_frac",
            "is_far",
        ])

    try:
        while True:
            ret, bgr = cap.read()

            if not ret or bgr is None:
                if is_file_source:
                    print(f"[{_ts()}] End of file — stopping.")
                    break
                error_streak += 1
                print(f"[{_ts()}] WARNING: dropped frame ({error_streak}/30)")
                if error_streak >= 30:
                    print(f"[{_ts()}] ERROR: too many dropped frames, attempting reconnect…")
                    cap.release()
                    time.sleep(3)
                    try:
                        cap = _open_capture(source_url)
                        _cleanup["cap"] = cap
                        error_streak = 0
                    except Exception as e:
                        print(f"[{_ts()}] Reconnect failed: {e}")
                continue

            error_streak = 0
            frame_count += 1
            t_frame = time.perf_counter()   # start latency clock

            if img_w is None:
                img_h, img_w = bgr.shape[:2]
                print(f"[{_ts()}] Stream resolution: {img_w}×{img_h}")
                far_detect_crop = inference_crop_for_frame(img_w, img_h) if img_w else None
                if SAVE_DETECTIONS:
                    out_path = os.path.join(OUTPUT_DIR, DETECTIONS_VIDEO_NAME)
                    if is_file_source:
                        in_abs = os.path.abspath(_local_video_path(source_url))
                        if in_abs == os.path.abspath(out_path):
                            out_path = os.path.join(OUTPUT_DIR, "detections_annotated.mp4")
                            print(
                                f"[{_ts()}] WARNING: INPUT_VIDEO is same path as detection output — "
                                f"writing annotated video to {out_path} instead."
                            )
                    output, _det_fourcc = _open_cv_video_writer(out_path, 30.0, (img_w, img_h))
                    _cleanup["output"] = output
                    _cleanup["detections_path"] = out_path
                    print(f"  Saving to   : {out_path}  (fourcc={_det_fourcc})")
                if RECORD_RTSP and not is_file_source:
                    out_path = os.path.join(OUTPUT_DIR, RECORD_RTSP_NAME)
                    raw_output, _raw_fourcc = _open_cv_video_writer(
                        out_path, RECORD_RTSP_FPS, (img_w, img_h)
                    )
                    _cleanup["raw_output"] = raw_output
                    _cleanup["raw_path"] = out_path
                    record_start_ts = time.time()
                    print(f"  Recording RTSP raw to: {out_path}  (fourcc={_raw_fourcc})")

            # Skip early frames when replaying from a file offset.
            if is_file_source and FILE_START_FRAME > 0 and frame_count <= FILE_START_FRAME:
                continue

            # Reset zoom state right after the warm-up window.
            if is_file_source and FILE_START_FRAME > 0 and frame_count == (FILE_START_FRAME + 1):
                current_crop = None
                focus_target_ema = None
                zoom_gate_smooth = 0.0
                last_target_zoom = None
                prev_focus_xyxy = None
                prev_focus_pr = None
                prev_focus_miss = 0
                no_det_frames = 0
                far_mode = True
                close_frames = 0
                prev_no_zoom_eligible = True
                prev_attn_gray_prime = None
                last_motion_prime_frame = -10**9

            # ── Submit frame to inference thread (non-blocking) ───────────────────
            # Decide detector input mode:
            #  - FAR mode: crop to far ROI (optional) + enable far-only focus eligibility
            #  - CLOSE mode: full frame (attention gating disabled)
            #  - Optional snapshots in FAR mode: periodically run a low-res full-frame
            #    pass to catch far bodies outside the ROI.
            #  - Optional MOTION_PRIME_FAR_SNAPSHOT: extra snapshot on motion in attention ROI
            periodic_snapshot = (
                far_mode
                and FULL_SNAPSHOT_EVERY_N_FRAMES > 0
                and (frame_count % FULL_SNAPSHOT_EVERY_N_FRAMES == 0)
            )
            motion_prime_snapshot = False
            if (
                MOTION_PRIME_FAR_SNAPSHOT
                and far_mode
                and _ATTENTION_ROI_PARSED is not None
                and img_w is not None
                and img_h is not None
            ):
                cur = _attention_roi_gray_prime(bgr, img_w, img_h)
                if cur is not None:
                    prev_b = prev_attn_gray_prime
                    md = _mean_absdiff_gray_prime(prev_b, cur)
                    prev_attn_gray_prime = cur
                    if (
                        prev_no_zoom_eligible
                        and prev_b is not None
                        and md >= MOTION_PRIME_MEAN_DIFF
                        and (frame_count - last_motion_prime_frame) >= MOTION_PRIME_COOLDOWN_FRAMES
                    ):
                        motion_prime_snapshot = True
                        last_motion_prime_frame = frame_count
            elif not far_mode or _ATTENTION_ROI_PARSED is None:
                prev_attn_gray_prime = None

            should_snapshot = periodic_snapshot or motion_prime_snapshot
            detect_crop_for_infer = None
            infer_scale = 1.0
            if should_snapshot:
                detect_crop_for_infer = None
                infer_scale = float(SNAPSHOT_FULL_SCALE)
            elif far_mode:
                detect_crop_for_infer = far_detect_crop
            else:
                detect_crop_for_infer = None

            with _infer["lock"]:
                _infer["bgr"] = bgr.copy()
                _infer["detect_crop"] = detect_crop_for_infer
                _infer["infer_scale"] = infer_scale
            _infer["ready"].set()

            # ── Use latest available detections (may be 1-2 frames stale) ─────────
            with _infer["lock"]:
                detections = list(_infer["dets"])
                infer_fid  = _infer["frame_id"]
            _shared["det_model_count"] = len(detections)

            if class_filter is not None:
                detections = [
                    d for d in detections if net.GetClassDesc(d.ClassID).lower() in class_filter
                ]

            # ── Max-distance filter: discard detections whose bbox height is below
            #    the configured fraction of the frame (small bbox = far away object).
            min_frac = _settings["min_det_fraction"]
            if min_frac > 0.0 and img_h:
                detections = [d for d in detections
                              if (d.Bottom - d.Top) / img_h >= min_frac]
            _shared["det_raw_count"] = len(detections)

            # Depth proxy (optional): approximate distance from bbox height.
            # NOTE: For now we keep it non-strict (no near-threshold increase)
            # because if this proxy is off it will hide otherwise-valid detections.
            if img_h:
                far_thr = float(_settings.get("det_threshold", THRESHOLD))
                def _accept(d) -> bool:
                    # Keep threshold equal to detector threshold for stability.
                    thr = far_thr
                    return float(d.Confidence) >= thr
                detections = [d for d in detections if _accept(d)]

            _shared["det_count"] = len(detections)

            # Optionally merge overlapping detections so we draw/zoom using
            # fewer boxes (PeopleNet often emits multiple boxes across one body).
            if _settings.get("merge_boxes", True):
                zoom_detections = _merge_detections(detections, net)
            else:
                zoom_detections = detections

            # ── Draw boxes on numpy frame ─────────────────────────────────────────
            if _settings["show_boxes"]:
                draw_detections(bgr, zoom_detections, net)

            zoom_detections_for_focus = zoom_detections
            if far_mode and _ATTENTION_ROI_PARSED is not None and zoom_detections:
                zoom_detections_for_focus = [
                    d
                    for d in zoom_detections
                    if _attention_point_inside(
                        (float(d.Left) + float(d.Right)) / 2.0,
                        (float(d.Top) + float(d.Bottom)) / 2.0,
                        img_w,
                        img_h,
                    )
                ]
            zoom_trigger = len(zoom_detections_for_focus) > 0
            prev_no_zoom_eligible = len(zoom_detections_for_focus) == 0

            # ── Smart zoom (detection-driven) ────────────────────────────────────
            target_zoom_opt: tuple[int, int, int, int] | None = None
            focus = None
            if zoom_detections_for_focus:
                cand = select_focus_detection(zoom_detections_for_focus, net)
                if cand is not None:
                    focus = cand
                if prev_focus_xyxy is not None and zoom_detections_for_focus:
                    prev_pr = (
                        prev_focus_pr
                        if prev_focus_pr is not None
                        else _zoom_priority(zoom_detections_for_focus[0], net)
                    )
                    same_pr = [
                        d
                        for d in zoom_detections_for_focus
                        if _zoom_priority(d, net) == prev_pr
                    ]
                    if not same_pr:
                        same_pr = zoom_detections_for_focus
                    best_match = None
                    best_iou = 0.0
                    for d in same_pr:
                        iou = _bbox_iou_xyxy(prev_focus_xyxy, _det_xyxy(d))
                        if iou > best_iou:
                            best_iou = iou
                            best_match = d
                    if best_match is not None and best_iou >= ZOOM_FOCUS_IOU_MIN:
                        focus = best_match

            if focus is not None:
                no_det_frames = 0
                cls = net.GetClassDesc(focus.ClassID).lower()
                union_crop = _face_person_union_for_crop(focus, detections, net)
                crop_det = union_crop if union_crop is not None else focus
                if union_crop is not None:
                    # Union is much larger than a face-only box; use person-style padding
                    # so we do not apply the tight face branch to a body-sized region.
                    cls = "person"
                target_zoom_opt = compute_target_crop(crop_det, img_w, img_h, class_desc=cls)
                if ZOOM_TARGET_EMA > 0.0:
                    a = min(1.0, max(0.05, ZOOM_TARGET_EMA))
                    if focus_target_ema is None:
                        focus_target_ema = target_zoom_opt
                    else:
                        focus_target_ema = tuple(
                            int(round(a * n + (1.0 - a) * o))
                            for n, o in zip(target_zoom_opt, focus_target_ema)
                        )
                    target_zoom_opt = focus_target_ema
                else:
                    focus_target_ema = None
                prev_focus_xyxy = _det_xyxy(focus)
                prev_focus_pr = _zoom_priority(focus, net)
                prev_focus_miss = 0
            else:
                focus_target_ema = None
                no_det_frames += 1
                prev_focus_miss = prev_focus_miss + 1
                if prev_focus_miss > ZOOM_OUT_HOLD:
                    prev_focus_xyxy = None
                    prev_focus_pr = None
                target_zoom_opt = current_crop if no_det_frames < ZOOM_OUT_HOLD else None

            # FAR -> CLOSE switch based on focus bbox height staying high long
            # enough. This is monotonic: once CLOSE, we stay CLOSE.
            if far_mode:
                if focus is not None:
                    fh = float(focus.Bottom - focus.Top)
                    fh_frac = fh / float(img_h) if img_h else 0.0
                    if fh_frac >= float(CLOSE_FRAC_THRESHOLD):
                        close_frames += 1
                    else:
                        close_frames = 0
                else:
                    close_frames = 0
                if close_frames >= int(CLOSE_SUSTAIN_FRAMES):
                    far_mode = False

            zoom_gate_on = (focus is not None) or (no_det_frames < ZOOM_OUT_HOLD)

            if DEBUG_FOCUS_TRACE and focus_trace_writer is not None:
                if focus is not None:
                    fh = float(focus.Bottom - focus.Top)
                    focus_h_frac = fh / float(img_h) if img_h else 0.0
                    is_far = 1 if focus_h_frac <= FOCUS_FAR_FRAC_THRESHOLD else 0
                    cx = float((focus.Left + focus.Right) / 2.0)
                    cy = float((focus.Top + focus.Bottom) / 2.0)
                    cx_by_h = cx / float(img_h) if img_h else 0.0
                    cy_by_h = cy / float(img_h) if img_h else 0.0
                    focus_trace_writer.writerow([
                        frame_count,
                        1,
                        int(focus.Left), int(focus.Top), int(focus.Right), int(focus.Bottom),
                        int(cx), int(cy),
                        round(float(cx_by_h), 6),
                        round(float(cy_by_h), 6),
                        round(float(focus_h_frac), 6),
                        is_far,
                    ])
                else:
                    focus_trace_writer.writerow([
                        frame_count,
                        0,
                        -1, -1, -1, -1,
                        -1, -1,
                        -1.0, -1.0,
                        -1.0,
                        0,
                    ])

            if ZOOM_GATE_EMA > 0.0 and img_w and img_h:
                full_rect = (0, 0, img_w, img_h)
                ga = min(1.0, max(0.02, ZOOM_GATE_EMA))
                want_gate = 1.0 if zoom_gate_on else 0.0
                zoom_gate_smooth = zoom_gate_smooth * (1.0 - ga) + want_gate * ga
                if target_zoom_opt is not None:
                    last_target_zoom = target_zoom_opt
                if last_target_zoom is not None and zoom_gate_smooth > 0.002:
                    target_crop = _lerp_rect_f(full_rect, last_target_zoom, zoom_gate_smooth)
                else:
                    target_crop = None
                    if zoom_gate_smooth <= 0.002:
                        last_target_zoom = None
            else:
                target_crop = target_zoom_opt

            current_crop = lerp_crop(current_crop, target_crop, img_w, img_h)
            display      = apply_zoom(bgr, current_crop, img_w, img_h)

            if DEBUG_TRACE and trace_writer is not None:
                due = (frame_count % max(1, DEBUG_TRACE_EVERY) == 0) or (
                    prev_zoom_trigger is None
                ) or (zoom_trigger != prev_zoom_trigger)
                if due:
                    def _bbox_or_m1(bb):
                        if bb is None:
                            return (-1, -1, -1, -1)
                        return (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))

                    t_x1, t_y1, t_x2, t_y2 = _bbox_or_m1(target_crop)
                    c_x1, c_y1, c_x2, c_y2 = _bbox_or_m1(current_crop)
                    trace_writer.writerow([
                        frame_count,
                        len(zoom_detections),
                        int(1 if zoom_trigger else 0),
                        t_x1, t_y1, t_x2, t_y2,
                        c_x1, c_y1, c_x2, c_y2,
                    ])
                    prev_zoom_trigger = zoom_trigger

            # ── Web frame (downscale first, then overlay at correct resolution) ──────
            # Overlays are drawn on web_frame rather than on the full-res display
            # frame so their pixel sizes match the stream resolution.  Drawing on
            # the full-res frame then downscaling shrinks text by 0.8×, making the
            # overlay appear too small on screen.
            if ENABLE_WEB:
                if WEB_JPEG_WIDTH and WEB_JPEG_WIDTH < display.shape[1]:
                    wh = int(display.shape[0] * WEB_JPEG_WIDTH / display.shape[1])
                    web_frame = cv2.resize(display, (WEB_JPEG_WIDTH, wh),
                                           interpolation=cv2.INTER_LINEAR)
                else:
                    web_frame = display
                # Do not draw time/weather or package overlay on the JPEG; the frontend
                # renders them as HTML overlays (like the perf panel) so they stay fixed
                # when fill screen or zoom changes.
                _shared["overlay"] = _build_overlay_data()
                if _settings["show_pkg_cam"]:
                    with _pkg_lock:
                        pkg = _latest_pkg_bgr
                    if pkg is not None:
                        _, pkg_buf = cv2.imencode(".jpg", pkg, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        _shared["pkg_cam_jpeg"] = pkg_buf.tobytes()
                    else:
                        _shared["pkg_cam_jpeg"] = None
                else:
                    _shared["pkg_cam_jpeg"] = None
                _, buf = cv2.imencode(".jpg", web_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                with _frame_lock:
                    _shared["latest_jpeg"] = buf.tobytes()

            # ── Latency (capture → encode) ────────────────────────────────────────
            _stats["latency_ms"] = round((time.perf_counter() - t_frame) * 1000, 1)

            # ── File output ───────────────────────────────────────────────────────
            if output is not None:
                if frame_count > FILE_START_FRAME:
                    output.write(display)
            if raw_output is not None:
                if RECORD_RTSP_SECS > 0 and record_start_ts is not None:
                    if (time.time() - record_start_ts) >= RECORD_RTSP_SECS:
                        raw_output.release()
                        raw_output = None
                        _cleanup["raw_output"] = None
                        if EXIT_AFTER_RECORD:
                            print(f"[{_ts()}] Raw recording done (RECORD_RTSP_SECS). Exiting.")
                            break
                    else:
                        if frame_count > FILE_START_FRAME:
                            raw_output.write(bgr)
                else:
                    if frame_count > FILE_START_FRAME:
                        raw_output.write(bgr)

            # ── Log (only when inference produced a new result) ───────────────────
            if infer_fid > last_infer_fid:
                last_infer_fid = infer_fid
                if detections:
                    _stats["last_detection_time"] = time.time()
                for d in detections:
                    label = net.GetClassDesc(d.ClassID)
                    _detection_log.appendleft({
                        "label":      label,
                        "confidence": round(float(d.Confidence), 3),
                        "bbox":       [int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)],
                        "frame":      frame_count,
                        "time":       _ts(),
                        "fps":        round(_stats["fps"], 1),
                    })
                    print(f"[{_ts()}] frame={frame_count:06d}  "
                          f"{label:<20s}  conf={d.Confidence:.2f}  "
                          f"bbox=({int(d.Left)},{int(d.Top)})-({int(d.Right)},{int(d.Bottom)})")

            if frame_count % 30 == 0:
                now = time.time()
                _stats.update(fps=30.0 / max(now - fps_time, 0.001), frames=frame_count)
                fps_time = now
                print(f"[{_ts()}] ── {frame_count} frames  "
                      f"pipeline={_stats['fps']:.1f} fps  "
                      f"infer={net.GetNetworkFPS():.1f} fps  "
                      f"zoom={'on' if current_crop else 'off'} ──")

            if process_frame_limit > 0 and frame_count >= process_frame_limit:
                print(
                    f"[{_ts()}] Stopping after {process_frame_limit} frames "
                    f"(FILE_MAX_FRAMES and/or file metadata / EOF cap)."
                )
                break

    finally:
        try:
            if trace_fp is not None:
                trace_fp.close()
            if focus_trace_fp is not None:
                focus_trace_fp.close()
        except Exception:
            pass
        _infer["stop"].set()
        _infer["ready"].set()
        try:
            infer_thread.join(timeout=8.0)
        except Exception:
            pass
        _release_recordings()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
