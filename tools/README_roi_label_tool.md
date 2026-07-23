# ROI label tool

Use this when you want to **draw rectangles on a still frame** and send the assistant (or your `camera.env`) exact normalized ROIs plus human-readable notes.

## Run it

1. Export a frame from your recording (e.g. a PNG from VLC/ffmpeg), or use an existing debug still.
2. Open **`roi_label_tool.html`** in a normal desktop browser (Chrome/Firefox):  
   **File → Open** the file, or from the repo root:  
   `xdg-open tools/roi_label_tool.html`
3. **Load image** → drag on the canvas to draw a box → set **label** + **description** → **Add last drawn rect**.
4. Repeat for each zone, then **Copy JSON to clipboard**.

## Labels → env vars

| Label in tool | Purpose |
|---------------|---------|
| `MOTION_STAGE1_ROI_NORM` | Red “stage 1” motion / approach gate |
| `FOVEATED_ROI_NORM` | Blue optional ROI (alias for attention / foveation tuning) |
| `CENTER_CROP_ROI_NORM` | Green **fallback zoom** when there is no person bbox (your hand-drawn “door” crop) |

The JSON includes an `env_snippets` object you can paste into `config/camera.env` (or pass via `docker compose` / `-e`).

## Coordinate convention

Normalized values are **`pixel / (width - 1)`** and **`pixel / (height - 1)`**, clamped to 0…1 — same convention as `scripts/analyze.py`.

## Semantic export → env mapping

If you paste a JSON export for the assistant, see **`docs/ROI_RECOMMENDATIONS.md`**. Put normalized values in **`config/camera.env`** as `INFERENCE_ROI_NORM` / `ZOOM_ATTENTION_ROI_NORM` (or legacy `MOTION_STAGE1_ROI_NORM` / `FOVEATED_ROI_NORM`).

## Check overlays

Run **`analyze.py`** with `SAVE_DETECTIONS=true` on a short clip and inspect **`output/detections.mp4`**, or use the live web preview with your ROI env vars set.

## Smaller “stage 2” foveation

There is only one foveated ROI today (`FOVEATED_ROI_NORM`). If the blue area feels too large for early detection, **shrink that rectangle** in the tool (or split concepts: keep stage-1 tight for motion, draw a smaller blue box for foveation) and update `FOVEATED_ROI_NORM` accordingly.
