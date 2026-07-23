#!/usr/bin/env python3
"""
DoorbellAIVision — web dashboard server.

Expects a `shared` dict from analyze.py containing:
  frame_lock    threading.Lock
  latest_jpeg   bytes | None       (updated each frame)
  stats         {"fps": float, "frames": int, "latency_ms": float}
  settings      {
      "overlay_scale":    float,
      "zoom_min_fraction": float,
      "min_det_fraction": float,
      "det_threshold":   float,
      "show_pkg_cam":     bool,
      "fill_screen":      bool,
      "show_boxes":       bool,
      "merge_boxes":      bool,
  }
  perf_monitor  SystemMonitor | None
"""

import time
import logging
from flask import Flask, Response, jsonify, request

# ── HTML ──────────────────────────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DoorbellAI</title>
<style>
:root {
  --bg:      #080909;
  --surface: #111315;
  --border:  #1f2225;
  --blue:    #3b82f6;
  --green:   #22c55e;
  --red:     #ef4444;
  --amber:   #f59e0b;
  --muted:   #5a6470;
  --text:    #dde3ea;
  --panel:   rgba(10,11,12,0.93);
  --ease:    cubic-bezier(.4,0,.2,1);
}
*,*::before,*::after { margin:0; padding:0; box-sizing:border-box; }
html,body { width:100%; height:100%; overflow:hidden; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }

/* ── Layout ── */
#app { display:flex; width:100vw; height:100vh; position:relative; }

/* Video column */
#vsection { flex:1; min-width:0; display:flex; flex-direction:column; position:relative; overflow:hidden; }
#vwrap { flex:1; background:#000; display:flex; align-items:center; justify-content:center; overflow:hidden; position:relative; }
#feed { width:100%; height:100%; object-fit:contain; display:block; cursor:none; user-select:none; -webkit-user-select:none; position:relative; z-index:0; }
#feed.fill { object-fit:cover; }

/* Overlays inside video area so they sit on top of feed; viewport-fixed when fill/zoom change */
.overlay-box { display:block !important; }
#clockoverlay, #pkgoverlay {
  position:absolute; pointer-events:none; z-index:10;
}
#clockoverlay, #pkgoverlay { z-index:25; }
#clockoverlay { top:56px; right:12px; transform-origin: top right; --overlay-scale: 1; transform: scale(var(--overlay-scale)); }
#pkgoverlay  { bottom:56px; right:12px; transform-origin: bottom right; --pkg-scale: 1; transform: scale(var(--pkg-scale)); }
#clockoverlay .co-inner {
  background:rgba(8,9,10,.72); border:1px solid var(--border); border-radius:10px;
  padding:12px 14px; min-width:140px;
  font-size:.9rem; color:var(--text); line-height:1.4;
  box-shadow:0 2px 12px rgba(0,0,0,.35);
}
#clockoverlay .co-time { font-size:1.18rem; font-weight:600; color:#fff; letter-spacing:.02em; }
#clockoverlay .co-date { font-size:.78rem; color:var(--muted); margin-top:3px; }
#clockoverlay .co-wx, #clockoverlay .co-wx-next {
  font-size:.8rem; color:var(--text); margin-top:8px; display:flex; align-items:center; gap:6px;
}
#clockoverlay .co-wx-next { margin-top:4px; font-size:.76rem; color:var(--muted); }
#clockoverlay .co-wx-icon { font-size:1rem; line-height:1; }
#pkgoverlay .po-inner {
  background:rgba(10,10,10,.9); border:1px solid var(--border); border-radius:6px;
  overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,.4);
}
#pkgoverlay img { display:block; vertical-align:bottom; max-width:min(42vw, 240px); max-height:min(28vh, 160px); }

/* ── Layout calibration zones ──────────────────────────────────────────── */
#zoneLayer {
  position:absolute; inset:0;
  z-index:12;
  pointer-events:none;
}
#vwrap.calib-on #zoneLayer { pointer-events:auto; }
.zone-rect {
  position:absolute; left:0; right:0;
  background:rgba(255,255,255,0.0);
  pointer-events:none;
}
.zone-line {
  position:absolute; left:0; right:0;
  height:3px;
  background:rgba(255,255,255,0.85);
  box-shadow:0 0 0 1px rgba(0,0,0,0.25);
  cursor:row-resize;
  pointer-events:auto;
}
.zone-line.dragging {
  background:rgba(59,130,246,0.95);
}
.calib-hint {
  position:absolute; top:14px; left:14px; z-index:13;
  background:rgba(0,0,0,0.55); border:1px solid var(--border);
  padding:8px 10px; border-radius:8px;
  font-size:.72rem; color:var(--muted);
  pointer-events:none;
  max-width:60%;
}

/* No-signal */
#nosig { position:absolute; inset:0; display:none; flex-direction:column; align-items:center; justify-content:center; gap:14px; color:var(--muted); font-size:.85rem; pointer-events:none; }
#nosig.show { display:flex; }
#nosig svg { opacity:.25; }

/* ── Control bars (top + bottom) ── */
.bar {
  position:absolute; left:0; right:0; height:44px;
  display:flex; align-items:center; padding:0 12px; gap:10px;
  opacity:0; pointer-events:none;
  transition:opacity 240ms var(--ease), transform 240ms var(--ease);
  z-index:20;
}
.bar.top    { top:0;    background:linear-gradient(to bottom,rgba(0,0,0,.72) 0%,transparent 100%); transform:translateY(-6px); }
.bar.bottom { bottom:0; background:linear-gradient(to top,   rgba(0,0,0,.72) 0%,transparent 100%); transform:translateY(6px); }
.bar.vis    { opacity:1; pointer-events:auto; transform:translateY(0); }

/* Status */
#dot { width:7px; height:7px; border-radius:50%; background:var(--green); animation:pulse 2.4s infinite; flex-shrink:0; }
#dot.off { background:var(--red); animation:none; }
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(34,197,94,.45)} 60%{box-shadow:0 0 0 5px rgba(34,197,94,0)} }
#title { font-size:.68rem; font-weight:700; letter-spacing:.2em; color:#fff; text-transform:uppercase; }
#fpsbadge { font-size:.66rem; color:var(--muted); }
#fpsbadge span { color:var(--text); font-variant-numeric:tabular-nums; }
.spacer { flex:1; }

/* Icon buttons */
.ibtn {
  width:32px; height:32px; border:none; border-radius:7px;
  background:rgba(255,255,255,.08); color:var(--text);
  display:flex; align-items:center; justify-content:center;
  cursor:pointer; flex-shrink:0;
  transition:background 140ms, transform 100ms;
}
.ibtn:hover  { background:rgba(255,255,255,.16); }
.ibtn:active { transform:scale(.93); }
.ibtn svg    { width:15px; height:15px; pointer-events:none; }
.ibtn.active { background:rgba(59,130,246,.25); color:var(--blue); }

/* ── Settings panel ── */
#spanel {
  position:absolute; top:0; right:0; bottom:0; width:268px;
  background:var(--panel); border-left:1px solid var(--border);
  backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  transform:translateX(100%); transition:transform 280ms var(--ease);
  z-index:30; display:flex; flex-direction:column;
  padding-top:50px;
}
#spanel.open { transform:translateX(0); }
#spanel-hdr {
  position:absolute; top:0; left:0; right:0; height:50px;
  display:flex; align-items:center; padding:0 16px;
  border-bottom:1px solid var(--border);
}
#spanel-hdr span { font-size:.68rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }
#spanel-close {
  margin-left:auto; background:none; border:none; color:var(--muted);
  cursor:pointer; padding:5px 6px; border-radius:5px; font-size:1rem; line-height:1;
  transition:color 140ms, background 140ms;
}
#spanel-close:hover { color:var(--text); background:rgba(255,255,255,.07); }
.sscroll { flex:1; overflow-y:auto; padding:18px 16px; display:flex; flex-direction:column; gap:22px; scrollbar-width:thin; scrollbar-color:var(--border) transparent; }
.sscroll::-webkit-scrollbar { width:3px; }
.sscroll::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }

/* Setting groups */
.sgroup { display:flex; flex-direction:column; gap:10px; }
.slabel { font-size:.64rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); }
.srow { display:flex; align-items:center; justify-content:space-between; }
.srow-label { font-size:.8rem; color:var(--text); }
.sdivider { height:1px; background:var(--border); }

/* Slider */
.slider-wrap { display:flex; align-items:center; gap:10px; }
.slider-wrap input[type=range] {
  flex:1; -webkit-appearance:none; height:3px;
  background:var(--border); border-radius:2px; outline:none; cursor:pointer;
}
.slider-wrap input[type=range]::-webkit-slider-thumb {
  -webkit-appearance:none; width:14px; height:14px; border-radius:50%;
  background:var(--blue); cursor:pointer; transition:transform 140ms;
}
.slider-wrap input[type=range]::-webkit-slider-thumb:hover { transform:scale(1.25); }
.sval { font-size:.7rem; color:var(--text); min-width:32px; text-align:right; font-variant-numeric:tabular-nums; }

/* Toggle */
.toggle { position:relative; width:38px; height:21px; flex-shrink:0; }
.toggle input { position:absolute; opacity:0; width:0; height:0; }
.ttrack {
  position:absolute; inset:0; border-radius:11px;
  background:var(--border); cursor:pointer;
  transition:background 200ms;
}
.ttrack::after {
  content:''; position:absolute; top:3px; left:3px; width:15px; height:15px;
  border-radius:50%; background:#fff; transition:transform 200ms;
}
.toggle input:checked + .ttrack { background:var(--blue); }
.toggle input:checked + .ttrack::after { transform:translateX(17px); }

/* ── Performance panel (floating, bottom-left of video) ── */
#perfpanel {
  position:absolute; bottom:56px; left:12px; width:236px;
  background:rgba(8,9,9,.92); border:1px solid var(--border); border-radius:10px;
  backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
  padding:12px 13px;
  display:flex; flex-direction:column; gap:10px;
  z-index:18;
  opacity:0; pointer-events:none;
  transform-origin: bottom left;
  --perf-scale: 1;
  transform: scale(var(--perf-scale)) translateY(6px);
  transition:opacity 240ms var(--ease), transform 240ms var(--ease);
}
#perfpanel.show { opacity:1; pointer-events:auto; transform: scale(var(--perf-scale)) translateY(0); }

/* chip block */
.pp-chip { display:flex; flex-direction:column; gap:5px; }
.pp-chip-hdr {
  display:flex; justify-content:space-between; align-items:baseline; gap:8px;
  margin-bottom:2px;
}
.pp-chip-name {
  font-size:.59rem; font-weight:700; letter-spacing:.09em;
  text-transform:uppercase; color:var(--muted);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0; flex:1;
}
.pp-chip-temp {
  font-size:.72rem; font-weight:600; font-variant-numeric:tabular-nums;
  color:var(--green); transition:color 400ms;
  flex-shrink:0;
}

/* metric rows */
.pp-row { display:flex; align-items:center; gap:7px; }
.pp-lbl { font-size:.62rem; color:var(--muted); width:30px; flex-shrink:0; }
.pp-bar { flex:1; height:4px; background:var(--border); border-radius:2px; overflow:hidden; }
.pp-fill {
  height:100%; width:0%; border-radius:2px; background:var(--green);
  transition:width 700ms ease, background 400ms ease;
}
.pp-val {
  font-size:.64rem; color:var(--green);
  min-width:76px; text-align:right;
  font-variant-numeric:tabular-nums;
  transition:color 400ms;
}
.pp-sep { height:1px; background:var(--border); }

/* Cursor management on video */
#vwrap.c-show #feed { cursor:default; }
</style>
</head>
<body>
<div id="app">

  <!-- ── Video section ── -->
  <div id="vsection">

    <!-- Top bar -->
    <div class="bar top" id="topbar">
      <div id="dot"></div>
      <span id="title">DoorbellAI</span>
      <span id="statsbadge">&nbsp;·&nbsp;<span id="fps">--</span>&nbsp;fps&nbsp;&nbsp;<span id="lat" style="color:var(--muted)">--&nbsp;ms</span>&nbsp;&nbsp;<span id="detbadge" style="color:var(--muted)">M:-- Raw:-- Det:--</span></span>
      <div class="spacer"></div>
    </div>

    <!-- Video -->
    <div id="vwrap">
      <img id="feed" src="/video_feed"
           onload="setOnline(true)" onerror="handleFeedError()"
           ondblclick="toggleFS()">
      <div id="nosig">
        <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4">
          <path d="M3 3l18 18M10.584 10.587a2 2 0 002.828 2.83
                   M6.343 6.346A8 8 0 0017.66 17.658
                   M3.515 3.515A12 12 0 0020.485 20.485"/>
        </svg>
        No signal
      </div>
      <div id="clockoverlay" class="overlay-box"><div class="co-inner"><div class="co-time" id="co-time">--:--</div><div class="co-date" id="co-date">--</div><div class="co-wx" id="co-wx"><span class="co-wx-icon" id="co-wx-icon"></span><span id="co-wx-text">Weather...</span></div><div class="co-wx-next" id="co-wx-next-wrap" style="display:none"><span class="co-wx-icon" id="co-wx-next-icon"></span><span id="co-wx-next-text"></span></div></div></div>
      <div id="pkgoverlay" class="overlay-box" style="display:none"><div class="po-inner"><img id="pkgcam" src="" alt="Package cam"></div></div>
      <div id="zoneLayer"></div>
    </div>

    <!-- Bottom bar -->
    <div class="bar bottom" id="botbar">
      <!-- Fullscreen -->
      <button class="ibtn" id="fsbtn" title="Fullscreen (or double-click video)" onclick="toggleFS()">
        <svg id="fsico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M8 3H5a2 2 0 00-2 2v3m18 0V5a2 2 0 00-2-2h-3
                   M3 16v3a2 2 0 002 2h3m8 0h3a2 2 0 002-2v-3"/>
        </svg>
      </button>
      <div class="spacer"></div>
      <!-- System Monitor -->
      <button class="ibtn" id="perfbtn" title="System Monitor" onclick="togglePerf()">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="4" y="4" width="16" height="16" rx="2"/>
          <rect x="9" y="9" width="6" height="6"/>
          <line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/>
          <line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/>
          <line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/>
          <line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>
        </svg>
      </button>
      <!-- Settings -->
      <button class="ibtn" id="sbtn" title="Settings" onclick="toggleSettings()">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="3"/>
          <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83
                   2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33
                   1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09
                   A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33
                   l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15
                   a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09
                   A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82
                   l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68
                   a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09
                   a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33
                   l.06-.06a2 2 0 012.83 2.83l-.06.06
                   A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1
                   H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/>
        </svg>
      </button>
    </div>

    <!-- Settings panel -->
    <div id="spanel">
      <div id="spanel-hdr">
        <span>Settings</span>
        <button id="spanel-close" onclick="toggleSettings()">&#x2715;</button>
      </div>
      <div class="sscroll">

        <div class="sgroup">
          <div class="slabel">Time / Weather Overlay</div>
          <div class="slider-wrap">
            <input type="range" id="sl-overlay" min="50" max="500" value="100" step="5"
                   oninput="onOverlaySlider(this.value)">
            <span class="sval" id="sv-overlay">1.0x</span>
          </div>
        </div>

        <div class="sgroup">
          <div class="slabel">Package Camera Overlay</div>
          <div class="slider-wrap">
            <input type="range" id="sl-pkg" min="50" max="400" value="100" step="5"
                   oninput="onPkgOverlaySlider(this.value)">
            <span class="sval" id="sv-pkg">1.0x</span>
          </div>
        </div>

        <div class="sgroup">
          <div class="slabel">System Monitor (GPU/CPU)</div>
          <div class="slider-wrap">
            <input type="range" id="sl-perf" min="50" max="400" value="100" step="5"
                   oninput="onPerfScaleSlider(this.value)">
            <span class="sval" id="sv-perf">1.0x</span>
          </div>
        </div>

        <div class="sgroup">
          <div class="slabel">Max Zoom</div>
          <div class="slider-wrap">
            <input type="range" id="sl-zoom" min="10" max="80" value="40" step="5"
                   oninput="onZoomSlider(this.value)">
            <span class="sval" id="sv-zoom">4.0x</span>
          </div>
        </div>

        <div class="sgroup">
          <div class="slabel">Detection Threshold</div>
          <div class="slider-wrap">
            <input type="range" id="sl-detth" min="10" max="80" value="50" step="1"
                   oninput="onDetThreshSlider(this.value)">
            <span class="sval" id="sv-detth">0.50</span>
          </div>
        </div>

        <div class="sgroup">
          <div class="slabel">Max Detection Distance</div>
          <div class="slider-wrap">
            <input type="range" id="sl-mindet" min="0" max="40" value="0" step="2"
                   oninput="onMinDetSlider(this.value)">
            <span class="sval" id="sv-mindet">Off</span>
          </div>
        </div>

        <div class="sdivider"></div>

        <div class="sgroup">
          <div class="slabel">Display</div>
          <div class="srow">
            <span class="srow-label">Detection boxes</span>
            <label class="toggle">
              <input type="checkbox" id="tog-boxes"
                     onchange="onToggle('show_boxes', this.checked)">
              <div class="ttrack"></div>
            </label>
          </div>
          <div class="srow">
            <span class="srow-label">Merge boxes</span>
            <label class="toggle">
              <input type="checkbox" id="tog-merge"
                     onchange="onToggle('merge_boxes', this.checked)">
              <div class="ttrack"></div>
            </label>
          </div>
          <div class="srow">
            <span class="srow-label">Performance Stats</span>
            <label class="toggle">
              <input type="checkbox" id="tog-stats" checked
                     onchange="onToggle('show_stats', this.checked)">
              <div class="ttrack"></div>
            </label>
          </div>
          <div class="srow">
            <span class="srow-label">System Monitor</span>
            <label class="toggle">
              <input type="checkbox" id="tog-perf"
                     onchange="onPerfToggle(this.checked)">
              <div class="ttrack"></div>
            </label>
          </div>
          <div class="srow">
            <span class="srow-label">Package Camera</span>
            <label class="toggle">
              <input type="checkbox" id="tog-pkg" checked
                     onchange="onToggle('show_pkg_cam', this.checked)">
              <div class="ttrack"></div>
            </label>
          </div>
          <div class="srow">
            <span class="srow-label">Fill Screen</span>
            <label class="toggle">
              <input type="checkbox" id="tog-fill"
                     onchange="onFillToggle(this.checked)">
              <div class="ttrack"></div>
            </label>
          </div>
          <div class="srow">
            <span class="srow-label">Layout Calibration</span>
            <label class="toggle">
              <input type="checkbox" id="tog-calib"
                     onchange="onCalibToggle(this.checked)">
              <div class="ttrack"></div>
            </label>
          </div>
          <div class="srow">
            <span class="srow-label">Zone edges</span>
            <span class="sval" id="zoneedges">[--]</span>
          </div>
        </div>

      </div>
    </div>

    <!-- Performance panel (floating, bottom-left) -->
    <div id="perfpanel">
      <!-- GPU -->
      <div class="pp-chip">
        <div class="pp-chip-hdr">
          <span class="pp-chip-name" id="pp-gpu-name">GPU</span>
          <span class="pp-chip-temp" id="pp-gpu-temp">—</span>
        </div>
        <div class="pp-row">
          <span class="pp-lbl">Util</span>
          <div class="pp-bar"><div class="pp-fill" id="pp-gpu-util-bar"></div></div>
          <span class="pp-val" id="pp-gpu-util-val">—</span>
        </div>
        <div class="pp-row">
          <span class="pp-lbl">VRAM</span>
          <div class="pp-bar"><div class="pp-fill" id="pp-gpu-vram-bar"></div></div>
          <span class="pp-val" id="pp-gpu-vram-val">—</span>
        </div>
      </div>
      <div class="pp-sep"></div>
      <!-- CPU -->
      <div class="pp-chip">
        <div class="pp-chip-hdr">
          <span class="pp-chip-name">CPU</span>
          <span class="pp-chip-temp" id="pp-cpu-temp">—</span>
        </div>
        <div class="pp-row">
          <span class="pp-lbl">Util</span>
          <div class="pp-bar"><div class="pp-fill" id="pp-cpu-util-bar"></div></div>
          <span class="pp-val" id="pp-cpu-util-val">—</span>
        </div>
        <div class="pp-row">
          <span class="pp-lbl">RAM</span>
          <div class="pp-bar"><div class="pp-fill" id="pp-cpu-ram-bar"></div></div>
          <span class="pp-val" id="pp-cpu-ram-val">—</span>
        </div>
      </div>
    </div>

  </div><!-- /vsection -->


</div><!-- /app -->
<script>
// ── Defaults & state ──────────────────────────────────────────────────────────
const DEFS = {
  overlay_scale: 100, pkg_overlay_scale: 100, perf_scale: 100,
  max_zoom_x10: 40, min_det_pct: 0,
  det_thresh_pct: 50,
  show_pkg_cam: true, show_stats: true,
  fill_screen: false, show_perf: false,
  show_boxes: false,
  merge_boxes: true,
  calib_on: false,
  // zone edges between zones in the *visible video content* coordinates (0..1)
  // length 5 => [e0,e1,e2,e3,e4]
  zone_edges: [0.20, 0.33, 0.52, 0.68, 0.82],
};
function wmoIcon(code) {
  if (code == null) return '\u2601';
  const m = { 0:'\u2600', 1:'\u2600', 2:'\u26C5', 3:'\u2601', 45:'\u2592', 48:'\u2592',
    51:'\u2614', 53:'\u2614', 55:'\u2614', 61:'\u2614', 63:'\u2614', 65:'\u2614',
    71:'\u2744', 73:'\u2744', 75:'\u2744', 77:'\u2744', 80:'\u2614', 81:'\u2614', 82:'\u2614',
    85:'\u2744', 86:'\u2744', 95:'\u26C8', 96:'\u26C8', 99:'\u26C8' };
  return m[code] || '\u2601';
}
let cfg = Object.assign({}, DEFS, JSON.parse(localStorage.getItem('dav') || '{}'));
let settingsOpen = false;
let ctrlTimer    = null;
let ctrlVis      = false;
let isFS         = false;
let feedOk       = false;
let retryTimer   = null;
let perfTimer    = null;

// ── Boot ──────────────────────────────────────────────────────────────────────
(function init() {
  try {
    // Restore overlay sliders
    const sl = document.getElementById('sl-overlay');
    if (sl) sl.value = cfg.overlay_scale;
    const svOverlay = document.getElementById('sv-overlay');
    if (svOverlay) svOverlay.textContent = (cfg.overlay_scale/100).toFixed(1)+'x';

    const slPkg = document.getElementById('sl-pkg');
    if (slPkg) slPkg.value = cfg.pkg_overlay_scale;
    const svPkg = document.getElementById('sv-pkg');
    if (svPkg) svPkg.textContent = (cfg.pkg_overlay_scale/100).toFixed(1)+'x';

    const slPerf = document.getElementById('sl-perf');
    if (slPerf) slPerf.value = cfg.perf_scale;
    const svPerf = document.getElementById('sv-perf');
    if (svPerf) svPerf.textContent = (cfg.perf_scale/100).toFixed(1)+'x';
    applyPerfScale(cfg.perf_scale);

    // Restore zoom slider
    const slz = document.getElementById('sl-zoom');
    if (slz) slz.value = cfg.max_zoom_x10;
    const svZoom = document.getElementById('sv-zoom');
    if (svZoom) svZoom.textContent = (cfg.max_zoom_x10/10).toFixed(1)+'x';

    // Restore min-detection slider
    const slmd = document.getElementById('sl-mindet');
    if (slmd) slmd.value = cfg.min_det_pct;
    const svMindet = document.getElementById('sv-mindet');
    if (svMindet) svMindet.textContent = cfg.min_det_pct === 0 ? 'Off' : '≥'+cfg.min_det_pct+'%';

    const sldet = document.getElementById('sl-detth');
    if (sldet) sldet.value = cfg.det_thresh_pct;
    const svDetth = document.getElementById('sv-detth');
    if (svDetth) svDetth.textContent = (cfg.det_thresh_pct/100).toFixed(2);

    // Restore toggles
    const tStats = document.getElementById('tog-stats'); if (tStats) tStats.checked = cfg.show_stats;
    const tPerf  = document.getElementById('tog-perf');  if (tPerf)  tPerf.checked  = cfg.show_perf;
    const tPkg   = document.getElementById('tog-pkg');   if (tPkg)   tPkg.checked   = cfg.show_pkg_cam;
    const tFill  = document.getElementById('tog-fill');  if (tFill)  tFill.checked  = cfg.fill_screen;
    const tBoxes = document.getElementById('tog-boxes'); if (tBoxes) tBoxes.checked = cfg.show_boxes;
    const tMerge = document.getElementById('tog-merge'); if (tMerge) tMerge.checked = cfg.merge_boxes;
    const tCalib = document.getElementById('tog-calib'); if (tCalib) tCalib.checked = cfg.calib_on;

    applyStatsBadge(cfg.show_stats);
    applyFillScreen(cfg.fill_screen);
    if (cfg.show_perf) openPerf();

    // Push current settings to server on load
    pushSettings({ overlay_scale:     cfg.overlay_scale/100,
                   pkg_overlay_scale:  cfg.pkg_overlay_scale/100,
                   zoom_min_fraction:  1.0 / (cfg.max_zoom_x10/10),
                   min_det_fraction:   cfg.min_det_pct / 100,
                   det_threshold:      cfg.det_thresh_pct / 100,
                   show_pkg_cam:       cfg.show_pkg_cam,
                   fill_screen:        cfg.fill_screen,
                   show_boxes:         cfg.show_boxes,
                   merge_boxes:       cfg.merge_boxes,
                   zone_edges:        cfg.zone_edges });

    applyOverlayScales();
    const pkgEl = document.getElementById('pkgoverlay');
    const pkgImg = document.getElementById('pkgcam');
    if (pkgEl) pkgEl.style.display = cfg.show_pkg_cam ? '' : 'none';
    if (cfg.show_pkg_cam && pkgImg) pkgImg.src = '/api/pkg_cam?t=' + Date.now();
    const zl = document.getElementById('zoneLayer');
    if (zl) zl.style.display = cfg.calib_on ? '' : 'none';
    if (cfg.calib_on) onCalibToggle(true);
  } catch(e) {
    console.warn('[ui] init error', e);
  }

  // Activity listeners → show controls
  ['mousemove','mousedown','keydown','touchstart'].forEach(ev =>
    document.addEventListener(ev, onActivity, { passive: true }));

  // Fullscreen change
  ['fullscreenchange','webkitfullscreenchange'].forEach(ev =>
    document.addEventListener(ev, onFSChange));

  showCtrls();
  setInterval(poll, 800);
  poll();
})();

// ── Controls auto-hide ────────────────────────────────────────────────────────
function onActivity() { showCtrls(); resetHideTimer(); }

function showCtrls() {
  if (ctrlVis) return;
  ctrlVis = true;
  document.getElementById('topbar').classList.add('vis');
  document.getElementById('botbar').classList.add('vis');
  document.getElementById('vwrap').classList.add('c-show');
}
function hideCtrls() {
  if (settingsOpen) return;
  ctrlVis = false;
  document.getElementById('topbar').classList.remove('vis');
  document.getElementById('botbar').classList.remove('vis');
  document.getElementById('vwrap').classList.remove('c-show');
}
function resetHideTimer() {
  clearTimeout(ctrlTimer);
  if (!settingsOpen) ctrlTimer = setTimeout(hideCtrls, 3200);
}

// ── Settings panel ────────────────────────────────────────────────────────────
function toggleSettings() {
  settingsOpen = !settingsOpen;
  document.getElementById('spanel').classList.toggle('open', settingsOpen);
  document.getElementById('sbtn').classList.toggle('active', settingsOpen);
  if (settingsOpen) { showCtrls(); clearTimeout(ctrlTimer); }
  else              { resetHideTimer(); }
}

// ── Time/weather overlay slider ───────────────────────────────────────────────
let overlayDebounce;
function applyOverlayScales() {
  const co = document.getElementById('clockoverlay');
  const po = document.getElementById('pkgoverlay');
  if (co) co.style.setProperty('--overlay-scale', (cfg.overlay_scale/100).toString());
  if (po) po.style.setProperty('--pkg-scale', (cfg.pkg_overlay_scale/100).toString());
}
function onOverlaySlider(rawVal) {
  const v = parseInt(rawVal);
  document.getElementById('sv-overlay').textContent = (v/100).toFixed(1)+'x';
  cfg.overlay_scale = v;
  save();
  applyOverlayScales();
  clearTimeout(overlayDebounce);
  overlayDebounce = setTimeout(() => pushSettings({ overlay_scale: v/100 }), 180);
}

// ── Package camera overlay slider ─────────────────────────────────────────────
let pkgOverlayDebounce;
function onPkgOverlaySlider(rawVal) {
  const v = parseInt(rawVal);
  document.getElementById('sv-pkg').textContent = (v/100).toFixed(1)+'x';
  cfg.pkg_overlay_scale = v;
  save();
  applyOverlayScales();
  clearTimeout(pkgOverlayDebounce);
  pkgOverlayDebounce = setTimeout(() => pushSettings({ pkg_overlay_scale: v/100 }), 180);
}

// ── System Monitor (GPU/CPU) scale slider ─────────────────────────────────────
function onPerfScaleSlider(rawVal) {
  const v = parseInt(rawVal);
  document.getElementById('sv-perf').textContent = (v/100).toFixed(1)+'x';
  cfg.perf_scale = v;
  save();
  applyPerfScale(v);
}
function applyPerfScale(pct) {
  const scale = pct / 100;
  const el = document.getElementById('perfpanel');
  if (el) el.style.setProperty('--perf-scale', String(scale));
}

// ── Zoom slider ───────────────────────────────────────────────────────────────
let zoomDebounce;
function onZoomSlider(rawVal) {
  const v = parseInt(rawVal);
  const zoom = v / 10;
  document.getElementById('sv-zoom').textContent = zoom.toFixed(1)+'x';
  cfg.max_zoom_x10 = v;
  save();
  clearTimeout(zoomDebounce);
  zoomDebounce = setTimeout(() =>
    pushSettings({ zoom_min_fraction: 1.0 / zoom }), 180);
}

// ── Min detection size (max distance) ────────────────────────────────────────
let minDetDebounce;
function onMinDetSlider(rawVal) {
  const v = parseInt(rawVal);
  document.getElementById('sv-mindet').textContent = v === 0 ? 'Off' : '≥'+v+'%';
  cfg.min_det_pct = v;
  save();
  clearTimeout(minDetDebounce);
  minDetDebounce = setTimeout(() =>
    pushSettings({ min_det_fraction: v / 100 }), 180);
}

// ── Detection confidence threshold ───────────────────────────────────────
let detThreshDebounce;
function onDetThreshSlider(rawVal) {
  const v = parseInt(rawVal);
  document.getElementById('sv-detth').textContent = (v/100).toFixed(2);
  cfg.det_thresh_pct = v;
  save();
  clearTimeout(detThreshDebounce);
  detThreshDebounce = setTimeout(() =>
    pushSettings({ det_threshold: v / 100 }), 180);
}

// ── Toggles ───────────────────────────────────────────────────────────────────
function onToggle(key, val) {
  cfg[key] = val;
  save();
  if (key === 'show_stats') applyStatsBadge(val);
  if (key === 'show_pkg_cam') {
    const pkgEl = document.getElementById('pkgoverlay');
    const pkgImg = document.getElementById('pkgcam');
    if (pkgEl) pkgEl.style.display = val ? '' : 'none';
    if (val && pkgImg) pkgImg.src = '/api/pkg_cam?t=' + Date.now();
  }
  pushSettings({ [key]: val });
}
function applyStatsBadge(show) {
  document.getElementById('statsbadge').style.display = show ? '' : 'none';
}
function applyFillScreen(fill) {
  document.getElementById('feed').classList.toggle('fill', fill);
}

// ── Fill Screen toggle ────────────────────────────────────────────────────────
function onFillToggle(val) {
  cfg.fill_screen = val;
  save();
  applyFillScreen(val);
  pushSettings({ fill_screen: val });
}

// ── Layout calibration (draggable horizontal boundaries) ─────────────────────
let calibEls = null; // { lines:[], rects:[] }
let calibDrag = { dragging: false, edgeIdx: -1 };

function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

function getVideoContentRect() {
  const vwrap = document.getElementById('vwrap');
  const feed  = document.getElementById('feed');
  if (!vwrap || !feed) return null;
  const vr = vwrap.getBoundingClientRect();
  const fr = feed.getBoundingClientRect();
  return {
    // zoneLayer is positioned relative to vwrap (inset:0), so we use top offset.
    contentTop: fr.top - vr.top,
    contentHeight: fr.height,
    feedTop: fr.top,
  };
}

function ensureCalibElements() {
  const zl = document.getElementById('zoneLayer');
  if (!zl) return;
  if (calibEls) return;

  const zoneColors = [
    'rgba(34,197,94,0.16)',  // patio (green)
    'rgba(245,158,11,0.16)', // steps (amber)
    'rgba(124,58,237,0.16)', // driveway (purple)
    'rgba(59,130,246,0.14)', // sidewalk (blue)
    'rgba(239,68,68,0.14)', // road (red)
    'rgba(245,158,11,0.12)'  // cross-sidewalk (amber-lite)
  ];

  const rects = [];
  for (let i = 0; i < 6; i++) {
    const d = document.createElement('div');
    d.className = 'zone-rect';
    d.style.background = zoneColors[i] || 'rgba(255,255,255,0.08)';
    zl.appendChild(d);
    rects.push(d);
  }

  const lines = [];
  for (let i = 0; i < 5; i++) {
    const l = document.createElement('div');
    l.className = 'zone-line';
    l.dataset.edgeIndex = String(i);
    l.addEventListener('mousedown', startCalibDrag);
    zl.appendChild(l);
    lines.push(l);
  }

  calibEls = { rects, lines };
}

function updateCalibElements() {
  const zl = document.getElementById('zoneLayer');
  if (!zl || !calibEls) return;
  const r = getVideoContentRect();
  if (!r || r.contentHeight <= 4) return;

  const edges = (cfg.zone_edges && cfg.zone_edges.length === 5)
    ? cfg.zone_edges
    : [0.20, 0.33, 0.52, 0.68, 0.82];

  const boundaries = [0].concat(edges).concat([1]); // length 7

  // Rects: 6 zones between boundaries[i] and boundaries[i+1]
  for (let z = 0; z < 6; z++) {
    const topFrac = boundaries[z];
    const botFrac = boundaries[z + 1];
    const topPx = r.contentTop + topFrac * r.contentHeight;
    const hPx   = Math.max(2, (botFrac - topFrac) * r.contentHeight);
    calibEls.rects[z].style.top = topPx + 'px';
    calibEls.rects[z].style.height = hPx + 'px';
  }

  // Lines: edges[0..4]
  for (let i = 0; i < 5; i++) {
    const topPx = r.contentTop + edges[i] * r.contentHeight - 1.5;
    calibEls.lines[i].style.top = topPx + 'px';
  }

  const ze = document.getElementById('zoneedges');
  if (ze && edges && edges.length === 5) {
    ze.textContent = '[' + edges.map(x => x.toFixed(3)).join(', ') + ']';
  }
}

function onCalibToggle(val) {
  cfg.calib_on = !!val;
  save();
  document.getElementById('vwrap').classList.toggle('calib-on', cfg.calib_on);

  const zl = document.getElementById('zoneLayer');
  if (zl) zl.style.display = cfg.calib_on ? '' : 'none';

  if (cfg.calib_on) {
    ensureCalibElements();
    updateCalibElements();
  } else {
    calibDrag.dragging = false;
    calibDrag.edgeIdx = -1;
  }
}

// Start drag on mousedown over a boundary line.
function startCalibDrag(e) {
  const t = e.target;
  if (!t || !t.dataset || t.dataset.edgeIndex == null) return;
  const idx = parseInt(t.dataset.edgeIndex);
  if (Number.isNaN(idx)) return;

  calibDrag.dragging = true;
  calibDrag.edgeIdx = idx;
  e.preventDefault();

  window.addEventListener('mousemove', onCalibMouseMove, { passive: false });
  window.addEventListener('mouseup', onCalibMouseUp);
}

function onCalibMouseMove(e) {
  if (!calibDrag.dragging) return;
  const r = getVideoContentRect();
  if (!r || r.contentHeight <= 4) return;

  const edges = (cfg.zone_edges && cfg.zone_edges.length === 5)
    ? cfg.zone_edges
    : [0.20, 0.33, 0.52, 0.68, 0.82];

  const idx = calibDrag.edgeIdx;
  let frac = (e.clientY - r.feedTop) / r.contentHeight;
  frac = clamp(frac, 0.0, 1.0);

  const eps = 0.01;
  const minV = (idx > 0) ? (edges[idx - 1] + eps) : 0.0;
  const maxV = (idx < 4) ? (edges[idx + 1] - eps) : 1.0;
  edges[idx] = clamp(frac, minV, maxV);
  cfg.zone_edges = edges;

  updateCalibElements();
}

async function onCalibMouseUp() {
  if (!calibDrag.dragging) return;
  calibDrag.dragging = false;
  calibDrag.edgeIdx = -1;
  window.removeEventListener('mousemove', onCalibMouseMove);
  window.removeEventListener('mouseup', onCalibMouseUp);
  save();
  // Push to backend so you can use these zones later for zoom/motion logic.
  pushSettings({ zone_edges: cfg.zone_edges });
}

// ── Persistence ───────────────────────────────────────────────────────────────
function save() { localStorage.setItem('dav', JSON.stringify(cfg)); }

async function pushSettings(payload) {
  try {
    await fetch('/api/settings', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
  } catch(_) {}
}

// ── Fullscreen ────────────────────────────────────────────────────────────────
function toggleFS() {
  const el = document.getElementById('app');
  if (!document.fullscreenElement && !document.webkitFullscreenElement) {
    (el.requestFullscreen || el.webkitRequestFullscreen).call(el);
  } else {
    (document.exitFullscreen || document.webkitExitFullscreen).call(document);
  }
}
function onFSChange() {
  isFS = !!(document.fullscreenElement || document.webkitFullscreenElement);
  const ico = document.getElementById('fsico');
  ico.innerHTML = isFS
    ? '<path d="M8 3v3a2 2 0 01-2 2H3m18 0h-3a2 2 0 01-2-2V3m0 18v-3a2 2 0 012-2h3M3 16h3a2 2 0 012 2v3"/>'
    : '<path d="M8 3H5a2 2 0 00-2 2v3m18 0V5a2 2 0 00-2-2h-3M3 16v3a2 2 0 002 2h3m8 0h3a2 2 0 002-2v-3"/>';
  resetHideTimer();
}

// ── Feed status ───────────────────────────────────────────────────────────────
function setOnline(on) {
  feedOk = on;
  document.getElementById('dot').className      = on ? '' : 'off';
  document.getElementById('feed').style.display = on ? 'block' : 'none';
  document.getElementById('nosig').classList.toggle('show', !on);
}
function handleFeedError() {
  setOnline(false);
  clearTimeout(retryTimer);
  retryTimer = setTimeout(() => {
    document.getElementById('feed').src = '/video_feed?t=' + Date.now();
  }, 4000);
}

// ── Poll (FPS + latency + overlay) ─────────────────────────────────────────────
async function poll() {
  try {
    const d = await (await fetch('/api/state')).json();
    document.getElementById('fps').textContent = d.fps.toFixed(1);
    document.getElementById('lat').textContent = d.latency_ms.toFixed(0) + ' ms';
    const detCount = d.det_count ?? 0;
    const detRaw = d.det_raw_count ?? 0;
    const detModel = d.det_model_count ?? 0;
    const detBadge = document.getElementById('detbadge');
    if (detBadge) detBadge.textContent = 'M:' + detModel + ' Raw:' + detRaw + ' Det:' + detCount;
    const o = d.overlay || {};
    const coTime = document.getElementById('co-time');
    const coDate = document.getElementById('co-date');
    const coWxIcon = document.getElementById('co-wx-icon');
    const coWxText = document.getElementById('co-wx-text');
    const coWxNextWrap = document.getElementById('co-wx-next-wrap');
    const coWxNextIcon = document.getElementById('co-wx-next-icon');
    const coWxNextText = document.getElementById('co-wx-next-text');
    if (coTime) coTime.textContent = o.time || '--:--';
    if (coDate) coDate.textContent = o.date || '--';
    if (coWxIcon) coWxIcon.textContent = wmoIcon(o.current_code);
    let wxLine = (o.condition || '') + (o.temp != null ? '  ' + o.temp + (o.temp_unit || '') : '') + (o.hours != null ? '  ' + o.hours + 'h' : '');
    if (!wxLine.trim()) wxLine = 'Weather...';
    if (coWxText) coWxText.textContent = wxLine;
    if (coWxNextWrap && coWxNextIcon && coWxNextText) {
      if (o.next_condition && o.next_time) {
        coWxNextWrap.style.display = 'flex';
        coWxNextIcon.textContent = wmoIcon(o.next_code);
        coWxNextText.textContent = o.next_condition + '   ' + o.next_time;
      } else {
        coWxNextWrap.style.display = 'none';
      }
    }
    const pkgEl = document.getElementById('pkgoverlay');
    const pkgImg = document.getElementById('pkgcam');
    if (pkgEl) pkgEl.style.display = cfg.show_pkg_cam ? '' : 'none';
    if (cfg.show_pkg_cam && pkgImg) pkgImg.src = '/api/pkg_cam?t=' + Date.now();
    if (!feedOk) {
      setOnline(true);
      document.getElementById('feed').src = '/video_feed?t=' + Date.now();
    }
  } catch(_) { setOnline(false); }
}

// ── System Monitor ────────────────────────────────────────────────────────────
function togglePerf() {
  cfg.show_perf = !cfg.show_perf;
  save();
  document.getElementById('tog-perf').checked = cfg.show_perf;
  document.getElementById('perfbtn').classList.toggle('active', cfg.show_perf);
  cfg.show_perf ? openPerf() : closePerf();
}

function onPerfToggle(val) {
  cfg.show_perf = val;
  save();
  document.getElementById('perfbtn').classList.toggle('active', val);
  val ? openPerf() : closePerf();
}

function openPerf() {
  document.getElementById('perfpanel').classList.add('show');
  document.getElementById('perfbtn').classList.add('active');
  document.getElementById('tog-perf').checked = true;
  pollPerf();
  clearInterval(perfTimer);
  perfTimer = setInterval(pollPerf, 2000);
}

function closePerf() {
  document.getElementById('perfpanel').classList.remove('show');
  document.getElementById('perfbtn').classList.remove('active');
  document.getElementById('tog-perf').checked = false;
  clearInterval(perfTimer);
}

async function pollPerf() {
  try {
    const d = await (await fetch('/api/perf')).json();
    renderPerf(d);
  } catch(_) {}
}

function renderPerf(d) {
  const g = d.gpu;
  const c = d.cpu;

  if (g) {
    document.getElementById('pp-gpu-name').textContent = g.name || 'GPU';
    const gTemp = document.getElementById('pp-gpu-temp');
    gTemp.textContent = g.temp_c != null ? g.temp_c + ' °C' : '—';
    gTemp.style.color = heatColor(g.temp_c);

    setBar('pp-gpu-util-bar', 'pp-gpu-util-val',
           g.util_pct, g.util_pct + '%', threshColor(g.util_pct, 75, 90));

    const vp  = g.mem_total_mb > 0 ? Math.round(g.mem_used_mb / g.mem_total_mb * 100) : 0;
    const vLbl = toGB(g.mem_used_mb) + ' / ' + toGB(g.mem_total_mb) + ' GB';
    setBar('pp-gpu-vram-bar', 'pp-gpu-vram-val', vp, vLbl, threshColor(vp, 75, 90));
  }

  if (c) {
    const cTemp = document.getElementById('pp-cpu-temp');
    cTemp.textContent = c.temp_c != null ? c.temp_c + ' °C' : '—';
    cTemp.style.color = heatColor(c.temp_c);

    setBar('pp-cpu-util-bar', 'pp-cpu-util-val',
           c.util_pct, c.util_pct + '%', threshColor(c.util_pct, 75, 90));

    const rLbl = toGB(c.mem_used_mb) + ' / ' + toGB(c.mem_total_mb) + ' GB';
    setBar('pp-cpu-ram-bar', 'pp-cpu-ram-val', c.mem_pct, rLbl, threshColor(c.mem_pct, 75, 90));
  }
}

function setBar(barId, valId, pct, label, color) {
  const bar = document.getElementById(barId);
  const val = document.getElementById(valId);
  bar.style.width      = Math.min(100, Math.max(0, pct)) + '%';
  bar.style.background = color;
  val.textContent      = label;
  val.style.color      = color;
}

// Returns CSS color for utilization / memory percentage thresholds
function threshColor(v, warn, crit) {
  return v >= crit ? 'var(--red)' : v >= warn ? 'var(--amber)' : 'var(--green)';
}

// Returns CSS color for temperature (warn ≥70 °C, crit ≥85 °C)
function heatColor(temp) {
  if (temp == null) return 'var(--muted)';
  return threshColor(temp, 70, 85);
}

// Convert MB → GB string with one decimal place
function toGB(mb) { return (mb / 1024).toFixed(1); }
</script>
</body>
</html>"""


# ── Flask app ─────────────────────────────────────────────────────────────────

def create_app(shared: dict) -> Flask:
    app = Flask(__name__)
    # Silence werkzeug request logs (they flood the container log)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.logger.setLevel(logging.ERROR)

    def _stream():
        try:
            while True:
                with shared["frame_lock"]:
                    frame = shared["latest_jpeg"]
                if frame is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Cache-Control: no-cache\r\n\r\n"
                        + frame + b"\r\n"
                    )
                time.sleep(1 / 30)
        except GeneratorExit:
            pass  # client disconnected cleanly

    @app.route("/")
    def index():
        return _HTML, 200, {
            "Content-Type": "text/html; charset=utf-8",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }

    @app.route("/video_feed")
    def video_feed():
        r = Response(_stream(),
                     mimetype="multipart/x-mixed-replace; boundary=frame")
        r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        r.headers["Pragma"]        = "no-cache"
        r.headers["Expires"]       = "0"
        return r

    @app.route("/api/state")
    def state():
        return jsonify({
            "fps":       round(shared["stats"]["fps"], 1),
            "latency_ms": shared["stats"].get("latency_ms", 0.0),
            "overlay":   shared.get("overlay") or {},
            "det_count": shared.get("det_count", 0),
            "det_raw_count": shared.get("det_raw_count", 0),
            "det_model_count": shared.get("det_model_count", 0),
        })

    @app.route("/api/pkg_cam")
    def pkg_cam():
        data = shared.get("pkg_cam_jpeg")
        if not data:
            return "", 204
        r = Response(data, mimetype="image/jpeg")
        r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return r

    @app.route("/api/perf")
    def perf():
        mon = shared.get("perf_monitor")
        return jsonify(mon.get() if mon else {"gpu": None, "cpu": None})

    @app.route("/api/settings", methods=["GET", "POST"])
    def settings():
        s = shared["settings"]
        if request.method == "POST":
            data = request.get_json(force=True, silent=True) or {}
            if "overlay_scale" in data:
                s["overlay_scale"] = max(0.5, min(5.0, float(data["overlay_scale"])))
            if "pkg_overlay_scale" in data:
                s["pkg_overlay_scale"] = max(0.5, min(4.0, float(data["pkg_overlay_scale"])))
            if "zoom_min_fraction" in data:
                s["zoom_min_fraction"] = max(0.05, min(1.0, float(data["zoom_min_fraction"])))
            if "min_det_fraction" in data:
                s["min_det_fraction"] = max(0.0, min(0.5, float(data["min_det_fraction"])))
            if "det_threshold" in data:
                s["det_threshold"] = max(0.05, min(0.8, float(data["det_threshold"])))
            if "zone_edges" in data:
                # Expected: array length 5 => [e0,e1,e2,e3,e4] with 0..1
                arr = data["zone_edges"]
                if isinstance(arr, list) and len(arr) == 5:
                    vals = []
                    for v in arr:
                        try:
                            vals.append(float(v))
                        except Exception:
                            vals.append(0.0)
                    # clamp + sort + enforce spacing
                    vals = [max(0.0, min(1.0, x)) for x in vals]
                    vals.sort()
                    # enforce monotonic with small epsilon
                    eps = 0.01
                    for i in range(1, 5):
                        if vals[i] < vals[i-1] + eps:
                            vals[i] = vals[i-1] + eps
                    # clamp again to max 1.0 and back-propagate if needed
                    if vals[-1] > 1.0:
                        vals[-1] = 1.0
                        for i in range(3, -1, -1):
                            if vals[i] > vals[i+1] - eps:
                                vals[i] = vals[i+1] - eps
                    # final clamp
                    vals = [max(0.0, min(1.0, x)) for x in vals]
                    s["zone_edges"] = vals
            if "show_pkg_cam" in data:
                s["show_pkg_cam"] = bool(data["show_pkg_cam"])
            if "fill_screen" in data:
                s["fill_screen"] = bool(data["fill_screen"])
            if "show_boxes" in data:
                s["show_boxes"] = bool(data["show_boxes"])
            if "merge_boxes" in data:
                s["merge_boxes"] = bool(data["merge_boxes"])
        return jsonify(s)

    return app


def start(shared: dict, port: int = 8080) -> None:
    app = create_app(shared)
    print(f"[web] Dashboard → http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)
