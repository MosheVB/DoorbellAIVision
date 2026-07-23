# ROI mapping from semantic export (`redbox_debug_shifted.png`)

Interpretation from **descriptions** (not arbitrary JSON labels):

| Region | Meaning |
|--------|---------|
| r0, r1, r3 | Porch wall, floor, ceiling — little value for *distant* approach; excluded by choosing a stage-1 rect that starts past the wall. |
| r2 | Sky + far houses — useful mainly once someone is already relevant; don’t let this dominate early motion. |
| r4–r7 | Actual “outside world” sightlines; r4 is the narrow gap at the walk-up; r6/r7 are patio / side views (lower priority). |
| r8, r9 | Railing slats — motion is fragmented until someone is near the rail. |
| **r5** | Primary walk-up / sidewalk-to-door channel — **default zoom** when there is no person bbox. |
| **r10** | Single rectangle that mixes near + far without the left wall — good **motion / approach gate** superset. |

## Suggested environment values

```bash
CENTER_CROP_ROI_NORM=0.438557,0.459300,0.616664,0.679984
MOTION_STAGE1_ROI_NORM=0.141412,0.295874,0.653360,0.895897
FOVEATED_ROI_NORM=0.394622,0.434529,0.616635,0.679816
```

**FOVEATED_ROI_NORM** = union of r4 + r5 + r8 in pixels → (631,521)–(986,815) → normalized with `(w-1),(h-1)`.

## Verify

Draw and copy normalized ROIs with **`tools/roi_label_tool.html`** (see **`tools/README_roi_label_tool.md`**). Pipeline env names: `INFERENCE_ROI_NORM`, `ZOOM_ATTENTION_ROI_NORM` (and legacy aliases `MOTION_STAGE1_ROI_NORM` / `FOVEATED_ROI_NORM`).
