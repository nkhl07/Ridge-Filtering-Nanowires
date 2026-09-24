"""
Shared parsing of the hand-labeled GT overlays (OneDrive_1_7-17-2026).

The *_Labeled.tif files are RGBA, anti-aliased, HOLLOW OUTLINE strokes (~4 px
pen) drawn per orientation family over the raw SEM frame -- they are NOT binary
masks and NOT filled bodies. Parsing convention (validated against the user's
own counts of 231/179 instances and ~12.3%/~9.5% filled coverage):

  1. Crop rows >= LABEL_ROWS (1024): the SEM databar occupies the bottom 70
     rows of the 1094-row frame, in BOTH the raw and labeled files.
  2. Wire stroke = channel spread (max-min RGB) > SPREAD_THR.
  3. Family = dominant RGB channel  (red / green / blue).
  4. binary_fill_holes PER FAMILY -> filled bodies (required before ANY
     coverage / area / thickness statistic).
  5. Instances = connected components (4-conn) per filled family, >= MIN_PX.

Verified orientation families (consistent across both frames):
    red ~122 deg, green ~60 deg, blue ~0/180 deg
Family weights must come from INSTANCE COUNTS, not pixel mass (a few thick
wires inflate area).

Scale: the GT frames are 80000x. HFW read from FEI TIFF tag 34682 gives
5.18 um (80k) vs 8.288 um (50k) -> 50k-equivalent rescale factor 0.625.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

LABEL_ROWS = 1024      # keep rows [0, 1024); bottom 70 rows are the databar
SPREAD_THR = 20        # RGB channel spread above which a pixel is a label stroke
MIN_PX = 20            # min filled-component area (reproduces the user's counts)
FAMILIES = ("red", "green", "blue")

GT_DIR = Path(__file__).resolve().parents[2] / "data" / "real" / "OneDrive_1_7-17-2026"
TARGET_HFW_50K = 8.288e-6      # HFW of a 50000x frame (FEI tag 34682)


def read_hfw(tif_path: str | Path) -> float | None:
    """Half-frame width in metres from the FEI metadata block (TIFF tag 34682)."""
    im = Image.open(tif_path)
    blob = im.tag_v2.get(34682) if hasattr(im, "tag_v2") else None
    if not blob:
        return None
    m = re.search(r"HFW\s*=\s*([0-9.eE+-]+)", blob if isinstance(blob, str) else str(blob))
    return float(m.group(1)) if m else None


def rescale_factor_to_50k(tif_path: str | Path) -> float:
    """Linear factor converting this frame's pixels to 50k-equivalent pixels."""
    hfw = read_hfw(tif_path)
    if hfw is None:
        raise ValueError(f"No HFW in metadata: {tif_path}")
    return hfw / TARGET_HFW_50K          # 5.18/8.288 = 0.625 for the 80k GT frames


def parse_labeled(labeled_path: str | Path,
                  spread_thr: int = SPREAD_THR,
                  min_px: int = MIN_PX) -> dict:
    """Parse one *_Labeled.tif into filled per-family masks and instances.

    Returns dict with:
        filled[fam]  bool mask, holes filled, specks removed
        labels[fam]  int label image of kept instances
        counts[fam]  instance count
        union        bool mask, union over families
        coverage     filled union coverage fraction
    """
    a = np.array(Image.open(labeled_path))[:LABEL_ROWS, :, :3].astype(int)
    spread = a.max(2) - a.min(2)
    stroke = spread > spread_thr
    dom = np.argmax(a, axis=2)

    filled, labels, counts = {}, {}, {}
    union = np.zeros(stroke.shape, dtype=bool)
    for i, fam in enumerate(FAMILIES):
        body = ndi.binary_fill_holes(stroke & (dom == i))
        lbl, _ = ndi.label(body)                      # 4-connectivity
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        keep = sizes >= min_px
        mask = keep[lbl]
        # renumber kept instances compactly
        lbl2, n2 = ndi.label(mask)
        filled[fam] = mask
        labels[fam] = lbl2
        counts[fam] = int(n2)
        union |= mask

    return {"filled": filled, "labels": labels, "counts": counts,
            "union": union, "coverage": float(union.mean())}


def load_raw(raw_path: str | Path) -> np.ndarray:
    """Raw SEM frame, databar cropped, min-max scaled to float32 [0,255]."""
    a = np.array(Image.open(raw_path))[:LABEL_ROWS].astype(np.float32)
    return (a - a.min()) / (a.max() - a.min() + 1e-9) * 255.0


def gt_frames() -> list[dict]:
    """The available GT frames: raw path, labeled path, and 50k rescale factor."""
    out = []
    for n in ("1", "2"):
        raw = GT_DIR / f"TiS3_80000×_{n}.tif"
        lab = GT_DIR / f"TiS3_80000×_{n} Labeled.tif"
        if raw.exists() and lab.exists():
            out.append({"name": f"TiS3_80000x_{n}", "raw": raw, "labeled": lab,
                        "scale_to_50k": rescale_factor_to_50k(raw)})
    return out
