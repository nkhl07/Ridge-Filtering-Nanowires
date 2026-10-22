"""Synthetic-only tests for unet/pseudo_labels.py (plain asserts, no pytest).

Small synthetic arrays only -- safe on the login node; real frames go through
slurm/unet_pseudo_labels.sbatch.

Run from the repo root: OMP_NUM_THREADS=2 $PYTHON -m unet.test_pseudo_labels
"""
from __future__ import annotations

import cv2
import numpy as np

from unet.common import BG, IGNORE, WIRE
from unet import pseudo_labels as pl


def bar(img, p0, p1, w):
    cv2.line(img, p0, p1, 1, w)
    return img


def test_confidence_split() -> None:
    resp = np.array([[0.00, 0.05, 0.10, 0.20, 0.30]], np.float32)
    teacher = np.array([[False, True, True, True, True]])
    lab = pl.confidence_split(resp, thr=0.1, teacher=teacher, lo_mult=0.5, hi_mult=1.5)
    # 0.05 <= lo but the teacher marked it (closing) -> ignore, not background
    assert lab.tolist() == [[BG, IGNORE, IGNORE, WIRE, WIRE]], lab


def test_hole_fill() -> None:
    w = np.zeros((200, 200), np.uint8)
    # hollow wire: 12 px wide body with a 4 px slot down the middle
    cv2.rectangle(w, (20, 20), (120, 31), 1, -1)
    w[24:28, 30:110] = 0
    # 60x60 background box enclosed by 4 crossing wires
    for r in (100, 170):
        bar(w, (90, r), (190, r), 5)
    for c in (100, 170):
        bar(w, (c, 90), (c, 190), 5)
    wire = w.astype(bool)
    filled, n_filled, n_holes = pl.narrow_holes(wire, max_halfwidth=4)
    assert n_holes == 2 and n_filled == 1, (n_holes, n_filled)
    assert filled[24:28, 30:110].all()
    assert not filled[120:150, 120:150].any()


def test_particles() -> None:
    w = np.zeros((200, 200), bool)
    w[10:16, 10:16] = True               # 6x6 round blob -> removed
    w[50:54, 20:100] = True              # 4x80 wire -> kept (elongated)
    w[100:120, 100:120] = True           # 20x20 blob, area 400 >= 220 -> kept (too big)
    w[150:153, 150:153] = True           # 3x3 speck -> removed
    rem, n = pl.round_particles(w, max_ratio=3.0, max_area=220)
    assert n == 2, n
    assert rem[10:16, 10:16].all() and rem[150:153, 150:153].all()
    assert not rem[50:54, 20:100].any() and not rem[100:120, 100:120].any()


def test_graph_junction_and_qc() -> None:
    H = W = 300
    w = np.zeros((H, W), np.uint8)
    bar(w, (20, 150), (280, 150), 5)                      # horizontal
    bar(w, (150, 20), (150, 280), 5)                      # vertical -> one X crossing
    # one chain with a 40 deg kink (wrong-merge signature), clear of the cross.
    # (A sharp V is NOT used: wire_extract splits it at the apex into two
    # straight chains, which is its behaviour, not this module's.)
    bent = np.zeros((H, W), np.uint8)
    bar(bent, (20, 40), (80, 40), 5)
    bar(bent, (80, 40), (126, 78), 5)
    w |= bent
    wire = w.astype(bool)
    p = pl.wire_extract_params(junction_close=False)
    g = pl.graph_qc(wire, wire, angles=[], p=p, fail_flags=["high_curvature", "high_turn_rate"])
    assert len(g["nodes3"]) == 1, g["nodes3"]
    r, c = g["nodes3"][0]
    assert abs(r - 150) <= 4 and abs(c - 150) <= 4, (r, c)
    assert 4 <= g["median_width_px"] <= 9, g["median_width_px"]    # 2*EDT convention
    assert g["qc_px"][bent.astype(bool)].mean() > 0.9, "kinked chain should fail curvature QC"
    assert not g["qc_px"][148:153, 200:280].any(), "straight arm must not be ignored"
    assert g["qc_px"].sum() <= wire.sum()
    band = pl.junction_band(wire.shape, g["nodes3"], radius=10)
    assert band[150, 150] and not band[150, 175]


def test_clean_frame_smoke() -> None:
    rng = np.random.default_rng(0)
    gray = 180 + rng.normal(0, 6, (256, 256))
    # two separate 4-way crossings (a 6-way crossing makes wire_extract pair arms
    # across a 45 deg turn, which correctly fails curvature QC)
    for (p0, p1) in (((10, 128), (246, 128)), ((128, 10), (128, 246)), ((20, 60), (200, 240))):
        cv2.line(gray, p0, p1, 60, 5)
    gray = np.clip(gray, 0, 255).astype(np.float32)
    cfg = {"confidence": {"enabled": True, "lo_mult": 0.5, "hi_mult": 1.5},
           "hole_fill": {"enabled": True, "max_hole_halfwidth_px": 4},
           "particles": {"enabled": True, "max_axis_ratio": 3.0, "max_particle_area_px": 220},
           "qc_chains": {"enabled": True, "fail_flags": ["high_curvature", "high_turn_rate"],
                         "graph_junction_close": False},
           "junction_band": {"enabled": True, "radius_px": "auto", "radius_width_mult": 2.0}}
    label, resp, m, st = pl.clean_frame(gray, cfg)
    assert set(np.unique(label)) <= {BG, WIRE, IGNORE}
    assert abs(sum(st["fractions"].values()) - 1) < 1e-9
    assert label[128, 30] == WIRE and label[60, 200] == BG, (label[128, 30], label[60, 200])
    assert st["junction_nodes_deg3"] >= 1 and label[128, 128] == IGNORE
    ov = pl.qc_overlay(gray, label, m, "smoke")
    assert ov.ndim == 3 and ov.shape[1] == 2 * 256 + 8

    # every step disabled -> plain teacher mask, nothing ignored
    off = {k: dict(v, enabled=False) for k, v in cfg.items()}
    label0, _, _, st0 = pl.clean_frame(gray, off)
    assert st0["fractions"]["ignore"] == 0.0


def main() -> None:
    test_confidence_split()
    test_hole_fill()
    test_particles()
    test_graph_junction_and_qc()
    test_clean_frame_smoke()
    print("ALL PASS")


if __name__ == "__main__":
    main()
