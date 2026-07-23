#!/usr/bin/env python3
"""
SystemMonitor — lightweight GPU + CPU health polling thread.

Collects every INTERVAL seconds (default 2 s):
  GPU  — name, utilization %, VRAM used/total MB, temperature °C   (pynvml)
  CPU  — utilization %, RAM used/total MB, temperature °C           (psutil)

Temperature thresholds for the UI:
  < 70 °C  → green (nominal)
  70–85 °C → amber (warm)
  > 85 °C  → red   (hot — consider airflow)

Usage:
    from system_monitor import SystemMonitor
    mon = SystemMonitor()
    stats = mon.get()   # {"gpu": {...} | None, "cpu": {...} | None}
"""

import threading
import time


class SystemMonitor:
    INTERVAL = 2.0   # seconds between each poll

    def __init__(self):
        self._lock  = threading.Lock()
        self._stats: dict = {"gpu": None, "cpu": None}

        self._nvml   = self._init_nvml()
        self._psutil = self._init_psutil()

        t = threading.Thread(target=self._run, daemon=True, name="sys-monitor")
        t.start()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self) -> dict:
        """Return a copy of the latest snapshot."""
        with self._lock:
            return {
                "gpu": dict(self._stats["gpu"]) if self._stats["gpu"] else None,
                "cpu": dict(self._stats["cpu"]) if self._stats["cpu"] else None,
            }

    # ── Init helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _init_nvml():
        try:
            import pynvml  # nvidia-ml-py ships this module under the same name
            pynvml.nvmlInit()
            return pynvml
        except Exception as exc:
            print(f"[sysmon] pynvml unavailable ({exc}) — GPU stats disabled")
            return None

    @staticmethod
    def _init_psutil():
        try:
            import psutil
            psutil.cpu_percent(interval=None)   # warm-up; first call is always 0
            return psutil
        except Exception as exc:
            print(f"[sysmon] psutil unavailable ({exc}) — CPU stats disabled")
            return None

    # ── Poll loop ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        handle   = None
        gpu_name = "GPU"

        if self._nvml:
            try:
                handle   = self._nvml.nvmlDeviceGetHandleByIndex(0)
                raw      = self._nvml.nvmlDeviceGetName(handle)
                gpu_name = raw.decode() if isinstance(raw, bytes) else raw
                print(f"[sysmon] GPU detected: {gpu_name}")
            except Exception as exc:
                print(f"[sysmon] Could not open GPU handle: {exc}")

        while True:
            gpu = self._poll_gpu(handle, gpu_name)
            cpu = self._poll_cpu()
            with self._lock:
                self._stats["gpu"] = gpu
                self._stats["cpu"] = cpu
            time.sleep(self.INTERVAL)

    def _poll_gpu(self, handle, name: str) -> dict | None:
        if self._nvml is None or handle is None:
            return None
        try:
            util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
            mem  = self._nvml.nvmlDeviceGetMemoryInfo(handle)
            temp = self._nvml.nvmlDeviceGetTemperature(
                handle, self._nvml.NVML_TEMPERATURE_GPU
            )
            return {
                "name":         name,
                "util_pct":     util.gpu,
                "mem_used_mb":  mem.used  // (1024 * 1024),
                "mem_total_mb": mem.total // (1024 * 1024),
                "temp_c":       temp,
            }
        except Exception as exc:
            print(f"[sysmon] GPU poll error: {exc}")
            return None

    def _poll_cpu(self) -> dict | None:
        if self._psutil is None:
            return None
        try:
            cpu_pct = self._psutil.cpu_percent(interval=None)
            vmem    = self._psutil.virtual_memory()
            return {
                "util_pct":     round(cpu_pct, 1),
                "mem_used_mb":  vmem.used  // (1024 * 1024),
                "mem_total_mb": vmem.total // (1024 * 1024),
                "mem_pct":      round(vmem.percent, 1),
                "temp_c":       self._cpu_temp(),
            }
        except Exception as exc:
            print(f"[sysmon] CPU poll error: {exc}")
            return None

    def _cpu_temp(self) -> float | None:
        """Best-effort CPU temperature from lm-sensors / hwmon (Linux only)."""
        try:
            sensor_map = self._psutil.sensors_temperatures()
        except Exception:
            return None

        for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz", "cpu-thermal"):
            entries = sensor_map.get(key, [])
            if entries:
                return round(sum(e.current for e in entries) / len(entries), 1)
        return None
