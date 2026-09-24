"""
Task 0.2c -- Prepare ONE in-band (40-65k) mid-density frame with an editable
pseudo-mask overlay, for hand-correction.

This is the gate: the synthetic_v3 coverage target and the pseudo-mask
correction factor must come from a hand-corrected IN-BAND frame. The 80k GT
correction is explicitly NOT extrapolated to 40-65k.

Frame choice: the median-coverage frame of the 18 in-band 50k-normalized
frames (mid-density by construction), databar cropped to 1536x1024. The
ranking deliberately uses the ORIGINAL v1 criterion (sato thr 0.10, no
closing) so the selected gate frame cannot drift from the published one
(SF251213M_50000x_50) when the mask pipeline evolves.

v3 (2026-07-24): masks come from the texture+ridge RF classifier
(wire_classifier.py, cross-frame validated). The overlay handed to the user
is the RECALL-ORIENTED variant -- dirclose(p > 0.70), recall 0.780 /
junction recall 0.909 vs GT: most of the manual work is ERASING excess
magenta, far faster than hand-painting missed wires. The production mask
(p > 0.70, no closing; recall 0.617 / precision 0.550) is also computed and
recorded in the meta JSON -- the correction factor is
    corrected_coverage / production_pseudo_coverage.
Since the CycleGAN track was dropped (2026-07-24), the corrected frame's
main purpose is now in-band training/validation data for the classifier.

Writes, into results/handcorrect/:
  <name>_raw.png          the frame, databar cropped (the base layer)
  <name>_pseudo.png       the recall-oriented mask being handed out, 0/255
  <name>_overlay.png      RGB: raw in gray + recall mask in magenta (edit this)
  handcorrect_README.txt  how to correct it and what to hand back
  handcorrect_meta.json   coverages of both variants + parameters
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.filters import sato

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "real" / "normalized_cache"          # the 40-65k in-band population
OUT = ROOT / "results" / "handcorrect"
OUT.mkdir(parents=True, exist_ok=True)

KEEP_ROWS = 1024        # databar is the bottom 70 rows of the 1094-row frames
SIGMAS = (1, 2, 3)      # v1 sato scales -- kept ONLY for the frame-selection ranking
THR_V1 = 0.10           # v1 threshold -- kept ONLY for the frame-selection ranking
CLOSE_LEN = 9           # line-kernel length (px @50k) for the edit-variant closing
CLOSE_ANGLES = (0, 60, 123)     # measured orientation families (real_stats)


def line_kernel(length: int, ang_deg: float) -> np.ndarray:
    """Binary line structuring element at the given orientation."""
    L = length if length % 2 else length + 1
    ker = np.zeros((L, L), np.uint8)
    cc = (L - 1) / 2
    rad = np.deg2rad(ang_deg)
    for t in np.linspace(-cc, cc, 4 * L):
        ker[int(round(cc - t * np.sin(rad))), int(round(cc + t * np.cos(rad)))] = 1
    return ker


KERNELS = [line_kernel(CLOSE_LEN, a) for a in CLOSE_ANGLES]


def dir_close(mask: np.ndarray) -> np.ndarray:
    """Union of closings along the family orientations: bridges junction gaps."""
    u8 = mask.astype(np.uint8)
    out = mask.copy()
    for ker in KERNELS:
        out |= cv2.morphologyEx(u8, cv2.MORPH_CLOSE, ker).astype(bool)
    return out


def sato_response(gray: np.ndarray) -> np.ndarray:
    """Per-frame min-max normalised sato response. Wires are DARK bodies."""
    img01 = gray.astype(np.float32) / 255.0
    r = sato(img01, sigmas=SIGMAS, black_ridges=True)
    return (r - r.min()) / (r.max() - r.min() + 1e-12)


def crop_databar(a: np.ndarray) -> np.ndarray:
    return a[:KEEP_ROWS] if a.shape[0] > KEEP_ROWS else a


def main() -> None:
    paths = sorted(glob.glob(str(CACHE / "*.npy")))
    if not paths:
        raise SystemExit(f"No in-band frames in {CACHE}")

    rows = []
    for p in paths:
        g = crop_databar(np.load(p))
        cov_v1 = float((sato_response(g) > THR_V1).mean())
        rows.append((cov_v1, p))
    rows.sort()
    for cov, p in rows:
        print(f"  {Path(p).stem:34s} v1 pseudo coverage {cov*100:5.2f}%")

    # mid-density = median coverage frame (v1 ranking, pinned -- see docstring)
    _, path = rows[len(rows) // 2]
    name = Path(path).stem
    gray = crop_databar(np.load(path))
    from wire_classifier import PROD_THR, load_classifier, wire_probability
    prob = wire_probability(load_classifier(), gray.astype(np.float32) / 255.0)
    mask_prod = prob > PROD_THR
    mask_edit = dir_close(mask_prod)
    print(f"\nSELECTED mid-density in-band frame: {name}")
    print(f"  shape {gray.shape}")
    print(f"  edit-variant coverage (dirclose(p>{PROD_THR})) {mask_edit.mean()*100:.2f}%")
    print(f"  production coverage  (p>{PROD_THR}) {mask_prod.mean()*100:.2f}%")

    Image.fromarray(gray.astype(np.uint8), mode="L").save(OUT / f"{name}_raw.png")
    Image.fromarray((mask_edit * 255).astype(np.uint8), mode="L").save(
        OUT / f"{name}_pseudo.png")

    rgb = np.stack([gray] * 3, axis=2).astype(np.uint8)
    rgb[mask_edit] = (255, 0, 255)
    Image.fromarray(rgb, mode="RGB").save(OUT / f"{name}_overlay.png")

    readme = f"""HAND-CORRECTION REQUEST -- in-band coverage target  (v3 overlay, 2026-07-24)
=============================================================================
Frame        : {name}
Shape        : {gray.shape[0]}x{gray.shape[1]}  (databar already cropped: kept rows 0-{KEEP_ROWS})
Scale        : 50k-equivalent (this frame comes from the 40-65k in-band cache)
Mask handed out: texture+ridge RF classifier (wire_classifier.py), p > {PROD_THR},
               plus directional closing (9 px line kernels at {CLOSE_ANGLES} deg,
               the measured orientation families). RECALL-ORIENTED variant.
Edit coverage: {mask_edit.mean()*100:.2f}%   (deliberately OVER-complete)
Prod coverage: {mask_prod.mean()*100:.2f}%   (p > {PROD_THR}, no closing -- the calibrated
               production mask; the correction factor is defined against THIS)

WHY A CLASSIFIER: the classical ridge pipeline hit its accuracy ceiling.
The RF keys mostly on local texture (wires are SMOOTH dark bodies against
mottled background) and, cross-frame validated on the 80k GT, beats it on
every metric. This overlay variant reaches recall 0.78 / junction recall
0.91 -- most of the correction work is ERASING excess magenta, much faster
than painting missed wires. Since the CycleGAN track was dropped, this
corrected frame's main purpose is in-band TRAINING DATA for the classifier.

WHAT TO DO:
  Edit  {name}_overlay.png  (magenta = current pseudo-mask, gray = the SEM frame).
  - ERASE magenta where it is NOT a wire (expect a fair amount of this).
  - Paint magenta where a wire was MISSED (should be rare now).
  Keep the image size unchanged. Any pure-magenta-ish pixel is read as wire.
  Alternatively edit {name}_pseudo.png directly (white = wire, black = background).

HAND BACK: the corrected file, same directory, named
  {name}_corrected.png

WHAT IT UNBLOCKS: the true in-band coverage, the real correction factor
(corrected_coverage / {mask_prod.mean()*100:.2f}% production coverage), and therefore the
synthetic_v3 coverage target and the Phase 5 A/B run. Everything else proceeds
meanwhile.
"""
    (OUT / "handcorrect_README.txt").write_text(readme)
    with open(OUT / "handcorrect_meta.json", "w") as fh:
        json.dump({"frame": name, "shape": list(gray.shape),
                   "pipeline": "v3 texture+ridge RF (2026-07-24)",
                   "model": "results/wire_classifier/rf_v3.joblib",
                   "dirclose": {"length_px": CLOSE_LEN,
                                "angles_deg": list(CLOSE_ANGLES)},
                   "edit_variant": {"rule": f"dirclose(p>{PROD_THR})",
                                    "coverage": float(mask_edit.mean()),
                                    "gt_recall": 0.780, "gt_precision": 0.525,
                                    "gt_junction_recall": 0.909},
                   "production_variant": {"rule": f"p>{PROD_THR}",
                                          "coverage": float(mask_prod.mean()),
                                          "gt_recall": 0.617, "gt_precision": 0.550,
                                          "gt_coverage_ratio": 0.93,
                                          "gt_junction_recall": 0.729},
                   "correction_factor_definition":
                       "corrected_coverage / production_variant.coverage",
                   "expected_corrected_file": f"{name}_corrected.png",
                   "v1_all_frame_pseudo_coverage":
                       {Path(p).stem: c for c, p in rows}}, fh, indent=2)
    print(f"\nwrote -> {OUT}")
    print(readme)


if __name__ == "__main__":
    main()
