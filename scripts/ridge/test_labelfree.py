"""Synthetic-only tests for labelfree.py (plain asserts, no pytest).

Why synthetic: the GT frames are reserved for evaluation, and real frames must
not be processed on the login node. These tests pin down the things that can
silently go wrong without any labels: the angle CONVENTION shared with
line_kernel (a sign flip would close along mirrored directions and still look
plausible), peak-count behaviour, threshold sanity, the bitwise regression link
to v2's dir_close, and degenerate inputs.

Run: OMP_NUM_THREADS=2 venv/bin/python scripts/ridge/test_labelfree.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labelfree as lf  # noqa: E402
import ridge_nanowire_images2 as v2  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SIZE = 512


def circ_dist(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def line_endpoints(rc: float, cc: float, ang_deg: float, half: float):
    """Endpoints (x=col, y=row) of a line through (rc, cc) in line_kernel's convention."""
    t = np.deg2rad(ang_deg)
    dc, dr = np.cos(t), -np.sin(t)          # row = cc - t*sin(a), col = cc + t*cos(a)
    p0 = (int(round(cc - half * dc)), int(round(rc - half * dr)))
    p1 = (int(round(cc + half * dc)), int(round(rc + half * dr)))
    return p0, p1


def synth(angles, n_per_family: int = 6, seed: int = 0, size: int = SIZE):
    """Bright mottled noisy background with DARK 3-4 px lines; returns (gray 0-255, truth)."""
    rng = np.random.default_rng(seed)
    mottle = ndi.gaussian_filter(rng.standard_normal((size, size)), 12)
    mottle *= 18.0 / (mottle.std() + 1e-12)
    truth = np.zeros((size, size), np.uint8)
    for a in angles:
        for _ in range(n_per_family):
            rc, cc = rng.uniform(0.1 * size, 0.9 * size, 2)
            p0, p1 = line_endpoints(rc, cc, a, 2 * size)
            cv2.line(truth, p0, p1, 1, thickness=int(rng.integers(3, 5)))
    img = 185.0 + mottle
    img = np.where(truth > 0, 85.0, img)
    img = ndi.gaussian_filter(img, 0.8) + rng.normal(0, 14, (size, size))
    return np.clip(img, 0, 255).astype(np.float32), truth.astype(bool)


def test_convention() -> None:
    """The synthetic line drawer agrees with line_kernel's orientation."""
    L = 41
    for a in (0, 15, 37, 60, 90, 123, 140, 171):
        ker = v2.line_kernel(L, a).astype(bool)
        img = np.zeros((L, L), np.uint8)
        p0, p1 = line_endpoints((L - 1) / 2, (L - 1) / 2, a, (L - 1) / 2)
        cv2.line(img, p0, p1, 1, thickness=1)
        iou = (ker & (img > 0)).sum() / (ker | (img > 0)).sum()
        assert iou > 0.6, f"drawer vs line_kernel mismatch at {a} deg: IoU {iou:.2f}"
    print("PASS test_convention")


def test_angle_recovery() -> None:
    for truth_angles, seed in (((0, 60, 123), 1), ((15, 75, 140), 2)):
        gray, _ = synth(truth_angles, seed=seed)
        resp = lf.sato_response(gray)
        thr = lf.auto_threshold(resp)
        est = lf.estimate_family_angles(resp, thr)
        assert len(est) == 3, f"expected 3 families for {truth_angles}, got {est}"
        assert all(0.0 <= a < 180.0 for a in est) and est == sorted(est)
        for t in truth_angles:
            d = min(circ_dist(t, e) for e in est)
            assert d <= 3.0, f"truth {t} not recovered: est {est} (err {d:.2f})"
        errs = [min(circ_dist(t, e) for e in est) for t in truth_angles]
        print(f"PASS test_angle_recovery truth={truth_angles} "
              f"est={[round(a, 2) for a in est]} max_err={max(errs):.2f} deg")


def test_single_family() -> None:
    """One family -> one angle, near truth; no invented families from the mottle floor."""
    gray, _ = synth((35,), n_per_family=8, seed=3)
    resp = lf.sato_response(gray)
    est = lf.estimate_family_angles(resp, lf.auto_threshold(resp))
    assert 1 <= len(est) <= 2, f"single family produced {est}"
    assert min(circ_dist(35, e) for e in est) <= 3.0, est
    assert len(est) == 1, f"single family invented extra family: {est}"
    print(f"PASS test_single_family est={[round(a, 2) for a in est]}")


def test_threshold_and_mask() -> None:
    gray, truth = synth((0, 60, 123), seed=1)
    resp = lf.sato_response(gray)
    lo, hi = float(resp.min()), float(resp.max())
    for m in lf.LF_METHODS:
        t = lf.auto_threshold(resp, m)
        assert lo < t < hi, f"{m}: thr {t} not inside ({lo}, {hi})"
    mask, r2, info = lf.labelfree_mask(gray)
    assert mask.dtype == bool and mask.shape == gray.shape
    assert np.array_equal(r2, resp) and r2.dtype == np.float32
    assert info["method"] == lf.DEFAULT_METHOD == "otsu"
    assert info["angles_source"] == "image" and info["close"] is True
    assert len(info["angles_deg"]) == 3
    rec = (mask & truth).sum() / truth.sum()
    cov_ratio = mask.mean() / truth.mean()
    prec = (mask & truth).sum() / max(mask.sum(), 1)
    assert rec > 0.6, f"recall {rec:.3f}"
    assert 0.5 < cov_ratio < 2.5, f"coverage ratio {cov_ratio:.2f}"

    # given angles bypass estimation; close=False is the raw threshold
    m_raw, _, info_raw = lf.labelfree_mask(gray, angles=(0, 60, 123), close=False)
    assert info_raw["angles_source"] == "given" and not info_raw["close"]
    assert np.array_equal(m_raw, resp > info["thr"])
    assert (mask >= m_raw).all(), "closing must only add pixels"
    thr_line = " ".join(f"{m}={lf.auto_threshold(resp, m):.3f}" for m in lf.LF_METHODS)
    print(f"PASS test_threshold_and_mask {thr_line} | otsu+close: recall {rec:.3f} "
          f"precision {prec:.3f} coverage {mask.mean() * 100:.1f}% "
          f"(truth {truth.mean() * 100:.1f}%)")


def test_large_threshold_subsample() -> None:
    """Strided subsample path is deterministic and close to the full answer.

    The cap is lowered temporarily instead of building a > 16 Mpx array (tests
    stay <= 1024x1024 on the login node). Same for the central-crop angle path.
    """
    gray, _ = synth((0, 60, 123), seed=1)
    resp = lf.sato_response(gray)
    t_full = lf.auto_threshold(resp)
    est_full = lf.estimate_family_angles(resp, t_full)
    caps = (lf.MAX_THR_PIXELS, lf.MAX_EST_SIDE)
    try:
        lf.MAX_THR_PIXELS, lf.MAX_EST_SIDE = 60_000, 384    # 512^2 -> stride 3, 384 crop
        t_sub = lf.auto_threshold(resp)
        assert t_sub == lf.auto_threshold(resp)
        est_crop = lf.estimate_family_angles(resp, t_sub)
    finally:
        lf.MAX_THR_PIXELS, lf.MAX_EST_SIDE = caps
    assert abs(t_sub - t_full) < 0.02, (t_sub, t_full)
    assert len(est_crop) == 3 and all(
        min(circ_dist(a, b) for b in est_full) <= 3.0 for a in est_crop), (est_crop, est_full)
    print(f"PASS test_large_threshold_subsample thr_sub {t_sub:.4f} vs full {t_full:.4f}; "
          f"crop angles {[round(a, 2) for a in est_crop]}")


def test_dir_close_regression() -> None:
    rng = np.random.default_rng(7)
    mask = rng.random((300, 400)) < 0.04
    a = lf.dir_close_angles(mask, (0, 60, 123))
    b = v2.dir_close(mask)
    assert a.dtype == b.dtype == bool and np.array_equal(a, b), "differs from v2 dir_close"
    assert np.array_equal(lf.dir_close_angles(mask, []), mask)
    print(f"PASS test_dir_close_regression (bitwise identical, {int(a.sum())} px)")


def test_degenerate() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for shape in ((64, 64), (5, 5), (1, 1), (2, 7)):
            gray = np.full(shape, 128.0, np.float32)
            for m in lf.LF_METHODS:
                mask, resp, info = lf.labelfree_mask(gray, method=m)
                assert mask.shape == shape and not mask.any(), (shape, m)
                assert info["angles_deg"] == []
        for shape in ((5, 5), (3, 9)):                  # tiny but non-constant
            gray = np.random.default_rng(0).uniform(0, 255, shape).astype(np.float32)
            mask, resp, info = lf.labelfree_mask(gray)
            assert mask.shape == shape and len(info["angles_deg"]) <= 3
    assert lf.estimate_family_angles(np.zeros((32, 32), np.float32), 0.5) == []
    print("PASS test_degenerate")


def main() -> None:
    test_convention()
    test_angle_recovery()
    test_single_family()
    test_threshold_and_mask()
    test_large_threshold_subsample()
    test_dir_close_regression()
    test_degenerate()
    print("ALL PASS")


if __name__ == "__main__":
    main()
