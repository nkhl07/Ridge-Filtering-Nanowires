"""Label-free (unsupervised) ridge pipeline: sato + per-image auto threshold +
directional closing at orientations measured from the image itself.

Why this exists: v2 (ridge_nanowire_images2.pseudo_mask) fixes THR=0.14 and the
closing directions CLOSE_ANGLES=(0,60,123), and both were chosen by scoring
against the 2 hand-labelled GT frames. Once the GT frames are reserved for
EVALUATION ONLY, those two numbers are leaked labels. This module replaces each
of them with a per-image, label-free estimate, so the GT frames can score the
result honestly. It deliberately imports neither gt_labels nor THR/CLOSE_ANGLES.

Declared PRIORS (carried from the label era, fixed in advance, NOT tuned on the
image being processed and never re-tuned on GT results):
  - input is at 50k-equivalent pixel scale (ridge_nanowire_images2.load_frame_50k);
  - sato sigmas SIGMAS = (1, 2, 3) px, black_ridges=True (wires are DARK bodies);
  - closing line-kernel length CLOSE_LEN = 9 px;
  - k = 3 orientation families (material prior: TiS3 wires grow in 3 families);
  - DEFAULT_METHOD = "otsu" for the auto threshold, PRE-DECLARED before any GT
    evaluation. Do not change it because another method scores better on GT --
    that would re-introduce label tuning through the back door.
  - estimator constants below (structure-tensor scales, histogram smoothing,
    peak separation / prominence) are generic image-analysis defaults, chosen on
    synthetic images only.

Per image and label-free:
  - threshold: skimage.filters.threshold_<method> on the min-max normalised
    sato response;
  - closing angles: peaks of a coherence-weighted structure-tensor orientation
    histogram over the above-threshold pixels.

Angle convention (CRITICAL): an angle a returned here is in degrees in [0, 180)
and means the same thing as in ridge_nanowire_images2.line_kernel(L, a), which
draws pixels at (row = cc - t*sin(a), col = cc + t*cos(a)): counter-clockwise
from +col with rows pointing UP. So line_kernel(L, a) lies along the detected
wires. test_labelfree.py checks this against drawn synthetic lines.

Bounded cost on huge frames (the 50k-equivalent set has frames well above
16 Mpx): auto_threshold runs on a deterministic strided subsample
resp[::s, ::s], and estimate_family_angles on a deterministic central crop of
at most 4096x4096. Both are reproducible run-to-run.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage import filters as skf
from skimage.filters import sato

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ridge_nanowire_images2 import CLOSE_LEN, SIGMAS, line_kernel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

LF_METHODS = ("otsu", "li", "triangle", "yen", "isodata")
DEFAULT_METHOD = "otsu"     # PRE-DECLARED before any GT evaluation; do not change on GT results
K_FAMILIES = 3              # material prior: TiS3 grows in 3 orientation families

MAX_THR_PIXELS = 16_000_000     # above this, threshold on a strided subsample
MAX_EST_SIDE = 4096             # above 16 Mpx, estimate angles on a central crop this size
GRAD_SIGMA = 1.5                # structure tensor: Gaussian-derivative scale (px)
WINDOW_SIGMA = 3.0              # structure tensor: integration window (px)
HIST_SMOOTH_DEG = 3.0           # circular Gaussian smoothing of the 1-deg histogram
MIN_SEP_DEG = 20.0              # circular min separation between family peaks
MIN_PEAK_REL = 0.20             # peak must rise >= this fraction of the top peak above the floor

_THRESHOLD_FNS = {
    "otsu": skf.threshold_otsu,
    "li": skf.threshold_li,
    "triangle": skf.threshold_triangle,
    "yen": skf.threshold_yen,
    "isodata": skf.threshold_isodata,
}


def sato_response(gray: np.ndarray) -> np.ndarray:
    """Min-max normalised sato response, exactly as v2 pseudo_mask computes r.

    gray is float 0-255 at 50k-equivalent scale. A constant image gives all 0.
    """
    img01 = gray.astype(np.float32) / 255.0
    r = sato(img01, sigmas=SIGMAS, black_ridges=True)
    r = (r - r.min()) / (r.max() - r.min() + 1e-12)
    return r.astype(np.float32, copy=False)


def _stride_for(n_pixels: int, max_pixels: int) -> int:
    return max(1, math.ceil(math.sqrt(n_pixels / max_pixels)))


def auto_threshold(resp: np.ndarray, method: str = DEFAULT_METHOD) -> float:
    """Per-image threshold on the normalised response; no labels involved.

    Arrays above MAX_THR_PIXELS are thresholded on resp[::s, ::s] with the
    smallest integer stride s that brings them under the cap (deterministic;
    histogram methods are insensitive to this uniform subsample). A
    (near-)constant response returns resp.max(), so `resp > thr` is empty.
    """
    if method not in _THRESHOLD_FNS:
        raise ValueError(f"unknown method {method!r}; choose from {LF_METHODS}")
    if resp.size == 0:
        return 0.0
    lo, hi = float(resp.min()), float(resp.max())
    if not hi - lo > 1e-9:
        return hi
    s = _stride_for(resp.size, MAX_THR_PIXELS) if resp.ndim == 2 else 1
    sub = resp[::s, ::s] if s > 1 else resp
    sub = np.ascontiguousarray(sub, dtype=np.float32)
    if not float(sub.max()) - float(sub.min()) > 1e-9:    # subsample missed all structure
        sub = np.ascontiguousarray(resp, dtype=np.float32)
    thr = float(_THRESHOLD_FNS[method](sub))
    return thr if math.isfinite(thr) else hi


def _central_crop(a: np.ndarray, side: int) -> np.ndarray:
    h, w = a.shape
    r0, c0 = max(0, (h - side) // 2), max(0, (w - side) // 2)
    return a[r0:r0 + side, c0:c0 + side]


def orientation_histogram(resp: np.ndarray, thr: float) -> np.ndarray:
    """Smoothed 180-bin (1 deg) histogram of RIDGE orientations, line_kernel convention.

    Structure tensor of resp with Gaussian derivatives (GRAD_SIGMA) integrated
    over WINDOW_SIGMA. Its dominant eigenvector is the gradient direction,
    i.e. ACROSS the ridge; the ridge runs perpendicular to it. Each pixel with
    resp > thr votes with weight resp * coherence, so strong, clearly oriented
    wire bodies dominate and isotropic junction blobs / mottle barely count.
    Bin i covers [i, i+1) deg.
    """
    if resp.size > MAX_THR_PIXELS:
        resp = _central_crop(resp, MAX_EST_SIDE)
    r = resp.astype(np.float32, copy=False)
    gr = ndi.gaussian_filter(r, GRAD_SIGMA, order=(1, 0))   # d/drow (row axis points down)
    gc = ndi.gaussian_filter(r, GRAD_SIGMA, order=(0, 1))   # d/dcol
    jrr = ndi.gaussian_filter(gr * gr, WINDOW_SIGMA)
    jcc = ndi.gaussian_filter(gc * gc, WINDOW_SIGMA)
    jrc = ndi.gaussian_filter(gr * gc, WINDOW_SIGMA)
    del gr, gc

    sel = r > thr
    hist = np.zeros(180, np.float64)
    if not sel.any():
        return hist
    jrr, jcc, jrc, w = jrr[sel], jcc[sel], jrc[sel], r[sel]
    # gradient angle in image coords (x = col, y = row DOWN), then flip y to
    # line_kernel's rows-UP convention and rotate 90 deg to the ridge direction
    theta_g = 0.5 * np.arctan2(2.0 * jrc, jcc - jrr)
    ang = np.mod(np.rad2deg(-theta_g) + 90.0, 180.0)
    coh = np.sqrt((jcc - jrr) ** 2 + 4.0 * jrc ** 2) / (jcc + jrr + 1e-12)
    hist = np.bincount(np.minimum(ang.astype(np.int64), 179), weights=w * coh,
                       minlength=180).astype(np.float64)
    return ndi.gaussian_filter1d(hist, HIST_SMOOTH_DEG, mode="wrap")


def _circ_dist(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def estimate_family_angles(resp: np.ndarray, thr: float, k: int = K_FAMILIES) -> list[float]:
    """Up to k wire-family orientations (deg, [0,180), sorted, line_kernel convention).

    Peaks are taken greedily, strongest first, from the circular orientation
    histogram, enforcing MIN_SEP_DEG circular separation. A candidate is kept
    only if it is a local maximum of the smoothed histogram AND rises at least
    MIN_PEAK_REL of the top peak's height above the floor (histogram median).
    So an image with fewer visible families returns FEWER than k angles rather
    than inventing families from the isotropic mottle floor; an image with no
    pixels above thr returns []. Peak positions are refined to sub-degree by
    parabolic interpolation on the circular histogram.
    """
    hist = orientation_histogram(resp, thr)
    if not hist.max() > 0:
        return []
    floor = float(np.median(hist))
    top = float(hist.max()) - floor
    if not top > 0:
        return []
    is_max = (hist >= np.roll(hist, 1)) & (hist >= np.roll(hist, -1))
    peaks: list[float] = []
    for i in np.argsort(hist)[::-1]:
        if len(peaks) >= k:
            break
        h = float(hist[i])
        if h - floor < MIN_PEAK_REL * top:
            break                       # everything after is weaker still
        if not is_max[i]:
            continue
        hm, hp = float(hist[(i - 1) % 180]), float(hist[(i + 1) % 180])
        den = hm - 2.0 * h + hp
        off = 0.5 * (hm - hp) / den if den < 0 else 0.0
        a = float(np.mod(i + 0.5 + off, 180.0))      # bin centre + sub-bin offset
        a = 0.0 if a >= 180.0 else a                 # float mod of a tiny negative -> 180.0
        if all(_circ_dist(a, p) >= MIN_SEP_DEG for p in peaks):
            peaks.append(a)
    return sorted(peaks)


def dir_close_angles(mask: np.ndarray, angles, length: int = CLOSE_LEN) -> np.ndarray:
    """Union of closings along arbitrary orientations (v2 dir_close semantics).

    With angles (0, 60, 123) and length CLOSE_LEN this is bitwise identical to
    ridge_nanowire_images2.dir_close. Empty angles -> plain copy of the mask.
    """
    u8 = mask.astype(np.uint8)
    out = mask.astype(bool, copy=True)
    for a in angles:
        out |= cv2.morphologyEx(u8, cv2.MORPH_CLOSE, line_kernel(length, a)).astype(bool)
    return out


def labelfree_mask(gray: np.ndarray, method: str = DEFAULT_METHOD, angles=None,
                   close: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """Label-free v2 analogue: sato -> auto threshold -> closing at image-estimated angles.

    angles=None estimates the families from this image; passing angles uses
    them as given (e.g. a fixed set for ablations). Returns (mask, resp, info).
    """
    resp = sato_response(gray)
    thr = auto_threshold(resp, method)
    if angles is None:
        ang = estimate_family_angles(resp, thr)
        source = "image"
    else:
        ang = [float(a) for a in angles]
        source = "given"
    mask = resp > thr
    if close:
        mask = dir_close_angles(mask, ang)
    info = {"method": method, "thr": float(thr), "angles_deg": [float(a) for a in ang],
            "angles_source": source, "close": bool(close)}
    return mask, resp, info
