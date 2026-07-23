"""Placeholder backend until YOLO + ByteTrack are wired in."""

from __future__ import annotations


class StubDetector:
    def Detect(self, bgr, overlay: str = "") -> list:
        return []

    def GetClassDesc(self, class_id: int) -> str:
        return "none"

    def GetNetworkFPS(self) -> float:
        return 0.0
