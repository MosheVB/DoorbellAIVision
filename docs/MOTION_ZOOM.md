# Motion-driven zoom (historical)

> **Note:** The live `scripts/analyze.py` pipeline **no longer** uses motion-energy gating or optical-flow lock. Zoom is **detection-driven**, with optional `INFERENCE_ROI_NORM` (run the detector on a fixed crop) and `ZOOM_ATTENTION_ROI_NORM` (only subjects inside this ROI can become the zoom target). Variables below apply only to older revisions or forked logic.

---

Instead of training a small network for “how much motion should start zoom,” the pipeline uses **motion energy**: the mean value of the existing frame-difference mask, **restricted to stage‑1 ROI** (`MOTION_STAGE1_ROI_NORM`), then **EMA-smoothed** and compared to `MOTION_ENERGY_THRESHOLD`.

That gives a single scalar “how busy is the porch” signal that does not depend on contour geometry or sustain-frame heuristics alone.

## Environment variables

| Variable | Role |
|----------|------|
| `MOTION_ENERGY_ZOOM` | `true`/`false` — enable energy as a zoom gate (default `true`). |
| `MOTION_ENERGY_THRESHOLD` | Mean mask activation (0–1) after EMA; above → contributes to `zoom_trigger`. Tune with your scene (default `0.012`). |
| `MOTION_ENERGY_EMA` | Smoothing of the raw energy (default `0.35`). Higher = faster response. |
| `ZOOM_GATE_EMA` | Smooth **full-frame ↔ zoom** blend when the boolean trigger flips (default `0.15`). Set `0` for instant on/off. |
| `MOTION_CROP_EMA` | Smooth motion **bounding box** before `compute_target_crop` (default `0.30`). Set `0` to follow raw motion box. |

Legacy counters (`MOTION_STAGE1_SUSTAIN_FRAMES`, `ZOOM_ON_MOTION_SUSTAIN_FRAMES`, approach heuristics) still apply if enabled; energy is **OR**’d into the same `zoom_trigger` so you can rely on it when geometry is unreliable.

## Optical-flow lock ("try B")

If zoom jitter comes from detector bbox center changes, enable an additional pixel-following mode:
- `FLOW_LOCK_ENABLED=true` initializes a locked crop from the current detector bbox.
- While zoom is active, it uses Farneback optical flow inside a neighborhood around the lock to shift the crop center.
- Detector bboxes only "correct" the lock when overlap is high (`FLOW_LOCK_DET_IOU_MIN`), which prevents continuous re-centering on jittery detections.
- To reduce "above-head" drift and up-close wobble, the flow ROI can be biased toward the bottom of the locked crop via `FLOW_LOCK_FOLLOW_BOTTOM_FRAC` (default `1.0`).
- If the locked crop starts too high up close, shift it downward on init (and re-corrections) via `FLOW_LOCK_INIT_Y_SHIFT_FRAC` (default `0.10`).
- To prevent detector corrections from causing big snaps, detector re-corrections now blend the lock center via `FLOW_LOCK_DET_CENTER_EMA` (default `0.15`) and ignore corrections implying a center jump larger than `FLOW_LOCK_CORR_MAX_SHIFT_FRAC` (default `0.20`). Optionally keep lock crop size via `FLOW_LOCK_KEEP_SIZE_ON_DET_CORR` (default `true`).

## Tuning

1. Run on a clip with `INPUT_VIDEO` and watch when zoom engages.
2. If zoom is too eager (trees / shadows): raise `MOTION_ENERGY_THRESHOLD` or set `MOTION_ENERGY_ZOOM=false` and use sustain/approach only.
3. If zoom snaps: increase `ZOOM_GATE_EMA` slightly or increase `ZOOM_SMOOTH` (crop lerp).
4. If the motion crop jitters: increase `MOTION_CROP_EMA`.

## Distance-based tighter zoom

Zoom padding can be reduced when the subject is far away (small bbox height):
- `ZOOM_PADDING_DIST_FAR_FRAC` (default `0.05`)
- `ZOOM_PADDING_DIST_NEAR_FRAC` (default `0.20`)
- `ZOOM_PADDING_DIST_FAR_MUL` (default `0.75`)
- `ZOOM_PADDING_DIST_NEAR_MUL` (default `1.00`)

For a data-driven threshold without labeling, you could log `motion_energy_ema` to CSV and pick a percentile; that is optional and does not require a new detector model.
