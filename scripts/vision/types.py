from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    """Bounding-box detection; optional track id for multi-object trackers (e.g. ByteTrack)."""

    ClassID: int
    Confidence: float
    Left: float
    Top: float
    Right: float
    Bottom: float
    track_id: int | None = None
