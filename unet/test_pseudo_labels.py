"""Synthetic-only tests for unet/pseudo_labels.py (plain asserts, no pytest).

Small synthetic arrays only -- safe on the login node; real frames go through
slurm/unet_pseudo_labels.sbatch.

Run from the repo root: OMP_NUM_THREADS=2 $PYTHON -m unet.test_pseudo_labels
"""
from __future__ import annotations

import cv2
import numpy as np

from unet.common import BG, IGNORE, WIRE   # first: puts scripts/ridge on sys.path
from unet import pseudo_labels as pl
import wire_extract as wx  # noqa: E402

CFG = {"confidence": {"enabled": True, "lo_mult": 0.5, "hi_mult": 1.0, "band_px": 0},
       "hole_fill": {"enabled": True, "max_hole_halfwidth_px": 4, "local_window_px": 64},
       "particles": {"enabled": True, "max_axis_ratio": 3.0, "max_particle_area_px": 600,
                     "audit_prev_max_area_px": 220},
       "saturated": {"enabled": True, "level": 250, "min_blob_px": 4, "radius_width_mult": 2.0},
       "dark_region": {"enabled": True, "sigma_px": 8, "dark_frac": 0.5,
                       "min_area_frac": 0.01, "margin_px": 8},
       "qc_chains": {"enabled": True, "fail_flags": [], "graph_junction_close": False},
       "junction_band": {"enabled": True, "radius_px": "auto", "radius_width_mult": 2.0}}


def bar(img, p0, p1, w, val=1):
    cv2.line(img, p0, p1, val, w)
    return img


def test_confidence_split() -> None:
    resp = np.zeros((1, 9), np.float32)
    resp[0, 2:7] = 0.3                    # 5 px core at thr 0.1
    resp[0, 8] = 0.07                     # halo between lo and thr
    teacher = resp > 0.1
    lab = pl.confidence_split(resp, 0.1, teacher, lo_mult=0.5, hi_mult=1.0, band_px=1)
    # bg | band(out) | band(in) wire wire wire band(in) | band(out) | halo
    assert lab.tolist() == [[BG, IGNORE, IGNORE, WIRE, WIRE, WIRE, IGNORE, IGNORE, IGNORE]], lab
    lab0 = pl.confidence_split(resp, 0.1, teacher, 0.5, 1.0, band_px=0)
    assert lab0.tolist() == [[BG, BG, WIRE, WIRE, WIRE, WIRE, WIRE, BG, IGNORE]], lab0
    # closing-added px (teacher, low resp) -> ignore, not background
    t2 = teacher.copy()
    t2[0, 0] = True
    assert pl.confidence_split(resp, 0.1, t2, 0.5, 1.0, 0)[0, 0] == IGNORE


def test_hole_fill() -> None:
    gray = np.full((200, 200), 150, np.float32)
    w = np.zeros((200, 200), np.uint8)
    # hollow DARK wire: 12 px body, 4 px slot (slot gray dark too) -> filled
    cv2.rectangle(w, (20, 20), (120, 31), 1, -1)
    w[24:28, 30:110] = 0
    gray[20:32, 20:121] = 60
    # dark rim around BRIGHT debris -> narrow hole but bright -> not filled
    cv2.circle(w, (60, 150), 4, 1, 2)
    cv2.circle(gray, (60, 150), 3, 255, -1)
    # 60x60 background box enclosed by 4 crossing wires -> wide -> not filled
    for r in (100, 170):
        bar(w, (110, r), (190, r), 5)
    for c in (120, 180):
        bar(w, (c, 90), (c, 190), 5)
    filled, n = pl.narrow_holes(w.astype(bool), 4, gray, 64)
    assert n["holes_total"] == 3 and n["holes_narrow"] == 2 and n["holes_filled"] == 1, n
    assert filled[24:28, 30:110].all()
    assert not filled[145:156, 55:66].any() and not filled[120:150, 130:170].any()


def test_particles() -> None:
    w = np.zeros((200, 200), bool)
    w[10:16, 10:16] = True               # 6x6 round blob -> removed
    w[50:54, 20:100] = True              # 4x80 wire -> kept (elongated)
    w[100:120, 100:120] = True           # 20x20 blob, area 400 >= 220 -> kept (too big)
    w[150:153, 150:153] = True           # 3x3 speck -> removed
    rem, n, aud, na = pl.round_particles(w, max_ratio=3.0, max_area=220, audit_min_area=100)
    assert (n, na) == (2, 0), (n, na)
    assert rem[10:16, 10:16].all() and rem[150:153, 150:153].all()
    assert not rem[50:54, 20:100].any() and not rem[100:120, 100:120].any()
    # raised limit: the 20x20 blob (area 400) is now removed AND flagged for audit
    rem, n, aud, na = pl.round_particles(w, max_ratio=3.0, max_area=600, audit_min_area=220)
    assert (n, na) == (3, 1), (n, na)
    assert aud[100:120, 100:120].all() and not aud[10:16, 10:16].any()


def test_saturated() -> None:
    gray = np.full((100, 100), 120, np.float32)
    gray[20:26, 20:26] = 255             # bright debris
    gray[80, 80] = 255                   # single hot pixel: below min_blob_px
    mask = np.zeros((100, 100), bool)
    mask[18:20, 15:60] = True            # rim wire running away from the debris
    mask[60:63, 10:90] = True            # wire far away
    mask[78:79, 70:90] = True            # wire next to the hot pixel only
    hit, nblob = pl.saturated_near(mask, gray, 250, 4, radius=6)
    assert nblob == 1, nblob
    assert hit[18:20, 15:31].all()                       # within 6 px of the blob edge
    assert not hit[18:20, 33:60].any()                   # rest of the same wire kept (local)
    assert not hit[60:63].any() and not hit[78].any()
    assert not hit[~mask].any()                          # only mask px are touched


def test_dark_region() -> None:
    rng = np.random.default_rng(0)
    gray = (150 + rng.normal(0, 5, (300, 300))).astype(np.float32)
    gray[120:180, :] = 10                # charging band across the frame
    gray[40:44, 40:44] = 10              # small dark pit: below min_area_frac
    edge, n, frac = pl.dark_region_edges(gray, 8, 0.5, 0.01, 8)
    assert n == 1 and 0.15 < frac < 0.25, (n, frac)
    assert edge[120, 150] and edge[179, 150] and edge[113, 150]
    assert not edge[150, 150] and not edge[60, 150] and not edge[42, 42]


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
    g0 = pl.graph_qc(wire, wire, angles=[], p=p, fail_flags=[])
    assert not g0["qc_px"].any() and len(g0["nodes3"]) == 1     # v1b default: QC off, bands on
    band = pl.junction_band(wire.shape, g["nodes3"], radius=10)
    assert band[150, 150] and not band[150, 175]


def test_known_wire_extract_terminal_junction_issue() -> None:
    """Pins a KNOWN wire_extract behaviour (documented in results/unet_v1/README.md):
    a perfectly straight diagonal wire that ENDS at a junction gets
    max_local_turning ~29 deg > curv_local_max 25 -> high_curvature, because the
    skeleton hooks into the junction node and turn_junction_excl only excises
    pixels around junctions a chain passes THROUGH (joins), not terminal ones.
    If wire_extract is fixed this test fails -- update the README then.

    Input: the TEACHER mask (labelfree_mask, direction-closed) of a synthetic
    frame with three straight dark lines; the closing bulges each crossing,
    which is where the terminal hook comes from (a plain binary line mask
    crosses cleanly)."""
    rng = np.random.default_rng(0)
    gray = 180 + rng.normal(0, 6, (256, 256))
    for p0, p1 in (((10, 128), (246, 128)), ((128, 10), (128, 246)), ((20, 60), (200, 240))):
        cv2.line(gray, p0, p1, 60, 5)
    teacher, _, _ = pl.lf.labelfree_mask(np.clip(gray, 0, 255).astype(np.float32))
    recs, _, _ = wx.extract_from_mask(teacher, 5.4, pl.wire_extract_params(False))
    diag = [r for r in recs if 30 < r["orientation_deg"] % 180 < 60
            or 120 < r["orientation_deg"] % 180 < 150]
    assert diag, "expected diagonal chains"
    bad = [r for r in diag if "high_curvature" in r["qc_flags"] and r["n_junctions_traversed"] == 0]
    assert bad, [(r["orientation_deg"], r["max_local_turning_deg"], r["qc_flags"]) for r in diag]
    assert all(25 < r["max_local_turning_deg"] < 35 for r in bad), \
        [r["max_local_turning_deg"] for r in bad]


def test_clean_frame_smoke() -> None:
    rng = np.random.default_rng(0)
    gray = 180 + rng.normal(0, 6, (256, 256))
    # two separate 4-way crossings (a 6-way crossing makes wire_extract pair arms
    # across a 45 deg turn, which correctly fails curvature QC)
    for (p0, p1) in (((10, 128), (246, 128)), ((128, 10), (128, 246)), ((20, 60), (200, 240))):
        cv2.line(gray, p0, p1, 60, 5)
    gray = np.clip(gray, 0, 255).astype(np.float32)
    label, resp, m, st = pl.clean_frame(gray, CFG)
    assert set(np.unique(label)) <= {BG, WIRE, IGNORE}
    assert abs(sum(st["fractions"].values()) - 1) < 1e-9
    assert label[128, 30] == WIRE and label[60, 200] == BG, (label[128, 30], label[60, 200])
    assert st["junction_nodes_deg3"] >= 1 and label[128, 128] == IGNORE
    assert np.isfinite(st["label_wire_width_nm"])
    ov = pl.qc_overlay(gray, label, m, "smoke")
    assert ov.ndim == 3 and ov.shape[1] == 2 * 256 + 8

    # every step disabled -> plain teacher mask, nothing ignored
    off = {k: dict(v, enabled=False) for k, v in CFG.items()}
    _, _, _, st0 = pl.clean_frame(gray, off)
    assert st0["fractions"]["ignore"] == 0.0


def main() -> None:
    test_confidence_split()
    test_hole_fill()
    test_particles()
    test_saturated()
    test_dark_region()
    test_graph_junction_and_qc()
    test_known_wire_extract_terminal_junction_issue()
    test_clean_frame_smoke()
    print("ALL PASS")


if __name__ == "__main__":
    main()
