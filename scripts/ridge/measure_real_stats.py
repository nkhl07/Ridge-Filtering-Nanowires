"""
Task 0.2 -- Measure real nanowire statistics at the 50k-equivalent scale.

Sources and provenance (deliberately kept separate; recorded in real_stats.json):
  * THICK TAIL (p75/p90) + ORIENTATION + INSTANCE CONVENTION + COVERAGE
        <- the 2 hand-labeled GT frames (TiS3_80000x_1/_2), rescaled x0.625.
  * THIN MODE thickness
        <- measured from the RAW frames as the FWHM of the dark intensity dip
           on perpendicular profiles at ~500 skeleton points per family.
           Rationale: the GT labels are ~4 px hollow PEN strokes, so pen width
           contaminates the thin end of any label-derived thickness.
  * COVERAGE TARGET for synthetic_v3
        <- NOT set here. It must come from a hand-corrected in-band (40-65k)
           frame; extrapolating the 80k GT correction is explicitly disallowed.
           This script records the GT coverage and flags it as out-of-band.

Key measured fact: real wires are DARK bodies with bright rims -- 75% of GT
wire pixels are darker than local background. A bright-only threshold scores
11% recall / 20% precision against GT, which is why the previous coverage
estimate (2.4%) was ~4x too low.

Outputs: real_stats.json, real_stats.npz, orientation_histogram.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_labels import FAMILIES, gt_frames, parse_labeled, load_raw

OUT = Path(__file__).resolve().parents[2] / "results" / "real_stats"
OUT.mkdir(parents=True, exist_ok=True)

N_FWHM_SAMPLES = 500      # perpendicular profiles per family per frame
PROFILE_HALF = 12         # px each side of the skeleton point (native 80k px)
STRAIGHT_WIN = 5          # +-px window for local orientation / straight runs


# ---------------------------------------------------------------------------
# skeleton helpers
# ---------------------------------------------------------------------------

_NB = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


def neighbour_count(sk: np.ndarray) -> np.ndarray:
    return ndi.convolve(sk.astype(np.uint8), _NB, mode="constant")


def split_segments(sk: np.ndarray) -> tuple[np.ndarray, int]:
    """Split a skeleton at branch points (>2 neighbours). Returns (labels, n)."""
    nb = neighbour_count(sk)
    branch = sk & (nb > 2)
    runs = sk & ~branch
    return ndi.label(runs, structure=np.ones((3, 3)))


def pca_orientation(ys: np.ndarray, xs: np.ndarray) -> float | None:
    """Orientation in [0,180) of a point cloud via principal axis."""
    if len(xs) < 3:
        return None
    p = np.stack([xs - xs.mean(), ys - ys.mean()])
    if np.allclose(p, 0):
        return None
    _, v = np.linalg.eigh(np.cov(p))
    vx, vy = v[:, -1]
    return float(np.degrees(np.arctan2(vy, vx)) % 180.0)


def path_length(ys: np.ndarray, xs: np.ndarray) -> float:
    """Approximate path length of a thin run: PCA extent (robust for straight runs)."""
    if len(xs) < 2:
        return float(len(xs))
    ang = pca_orientation(ys, xs)
    if ang is None:
        return float(len(xs))
    r = np.radians(ang)
    proj = (xs - xs.mean()) * np.cos(r) + (ys - ys.mean()) * np.sin(r)
    return float(proj.max() - proj.min()) + 1.0


def circ_mean_180(angles: np.ndarray) -> float:
    d = np.radians(np.asarray(angles) * 2.0)
    return float((np.degrees(np.arctan2(np.sin(d).mean(), np.cos(d).mean())) / 2.0) % 180.0)


def dev_180(angles: np.ndarray, mu: float) -> np.ndarray:
    return (np.asarray(angles) - mu + 90.0) % 180.0 - 90.0


# ---------------------------------------------------------------------------
# FWHM thin-mode thickness from the RAW frame
# ---------------------------------------------------------------------------

def fwhm_dark_dip(raw: np.ndarray, y: float, x: float, ang_deg: float,
                  half: int = PROFILE_HALF) -> float | None:
    """FWHM (px) of the dark dip on the profile perpendicular to `ang_deg`."""
    r = np.radians(ang_deg + 90.0)                       # perpendicular
    t = np.linspace(-half, half, 2 * half * 4 + 1)       # 0.25 px steps
    ys = y + t * np.sin(r)
    xs = x + t * np.cos(r)
    prof = ndi.map_coordinates(raw, [ys, xs], order=1, mode="nearest")
    if prof.size < 5:
        return None
    base = np.percentile(prof, 90)                       # local background level
    depth = base - prof.min()
    if depth < 5:                                        # no real dip here
        return None
    half_lvl = base - depth / 2.0
    below = prof < half_lvl
    c = len(prof) // 2
    if not below[c]:                                     # dip not centred: skip
        return None
    i = c
    while i > 0 and below[i - 1]:
        i -= 1
    j = c
    while j < len(prof) - 1 and below[j + 1]:
        j += 1
    return float((t[j] - t[i]))


# ---------------------------------------------------------------------------
def main() -> None:
    frames = gt_frames()
    if not frames:
        raise SystemExit("No GT frames found.")

    per_frame, stats = [], {}
    all_thick_gt = {f: [] for f in FAMILIES}     # GT label-derived (50k px)
    all_fwhm = {f: [] for f in FAMILIES}         # raw-derived thin mode (50k px)
    all_seg_len = {f: [] for f in FAMILIES}      # segment length (50k px)
    all_seg_ang = {f: [] for f in FAMILIES}
    all_comp_len = {f: [] for f in FAMILIES}
    fam_counts_total = {f: 0 for f in FAMILIES}
    jitter_all = {f: [] for f in FAMILIES}

    for fr in frames:
        s = fr["scale_to_50k"]                   # 0.625 for the 80k GT frames
        g = parse_labeled(fr["labeled"])
        raw = load_raw(fr["raw"])
        rec = {"name": fr["name"], "scale_to_50k": s,
               "coverage_filled": g["coverage"], "counts": g["counts"],
               "families": {}}

        for fam in FAMILIES:
            filled = g["filled"][fam]
            lbl = g["labels"][fam]
            n = g["counts"][fam]
            fam_counts_total[fam] += n

            sk = skeletonize(filled)
            dt = ndi.distance_transform_edt(filled)
            # --- GT thickness on skeleton (contaminated at the thin end by pen width)
            th = 2.0 * dt[sk] * s
            all_thick_gt[fam] += th.tolist()

            # --- component-level: total skeleton length per instance
            for i in range(1, n + 1):
                comp_sk = sk & (lbl == i)
                if comp_sk.sum():
                    all_comp_len[fam].append(float(comp_sk.sum()) * s)

            # --- segment-level: split skeleton at branch points
            seg, nseg = split_segments(sk)
            objs = ndi.find_objects(seg)
            for i, sl in enumerate(objs, start=1):
                if sl is None:
                    continue
                sub = seg[sl] == i
                if sub.sum() < 4:
                    continue
                ys, xs = np.nonzero(sub)
                ys = ys + sl[0].start
                xs = xs + sl[1].start
                a = pca_orientation(ys, xs)
                if a is None:
                    continue
                all_seg_len[fam].append(path_length(ys, xs) * s)
                all_seg_ang[fam].append(a)

            # --- within-family jitter on STRAIGHT runs away from branch points
            nb = neighbour_count(sk)
            interior = sk & (nb <= 2)
            seg2, n2 = split_segments(interior)
            for i, sl in enumerate(ndi.find_objects(seg2), start=1):
                if sl is None:
                    continue
                sub = seg2[sl] == i
                if sub.sum() < 2 * STRAIGHT_WIN:         # long enough to be straight
                    continue
                ys, xs = np.nonzero(sub)
                a = pca_orientation(ys + sl[0].start, xs + sl[1].start)
                if a is not None:
                    jitter_all[fam].append(a)

            # --- FWHM thin mode from the RAW frame at random skeleton points
            sy, sx = np.nonzero(sk & (nb <= 2))
            if len(sy):
                rng = np.random.default_rng(0)
                idx = rng.choice(len(sy), size=min(N_FWHM_SAMPLES, len(sy)), replace=False)
                for k in idx:
                    y, x = sy[k], sx[k]
                    y0, y1 = max(0, y - STRAIGHT_WIN), y + STRAIGHT_WIN + 1
                    x0, x1 = max(0, x - STRAIGHT_WIN), x + STRAIGHT_WIN + 1
                    wy, wx = np.nonzero(sk[y0:y1, x0:x1])
                    a = pca_orientation(wy, wx)
                    if a is None:
                        continue
                    w = fwhm_dark_dip(raw, y, x, a)
                    if w is not None:
                        all_fwhm[fam].append(w * s)

            rec["families"][fam] = {
                "instances": n,
                "pixel_mass": int(filled.sum()),
                "skeleton_px": int(sk.sum()),
            }
        per_frame.append(rec)
        print(f"{fr['name']}: coverage {g['coverage']*100:.2f}%  counts {g['counts']}  scale {s:.3f}")

    # ---- family means / weights / jitter -----------------------------------
    fam_stats = {}
    tot = sum(fam_counts_total.values())
    for fam in FAMILIES:
        angs = np.array(jitter_all[fam]) if jitter_all[fam] else np.array(all_seg_ang[fam])
        mu = circ_mean_180(angs) if len(angs) else 0.0
        jit = dev_180(angs, mu) if len(angs) else np.array([0.0])
        fam_stats[fam] = {
            "mean_deg": mu,
            "jitter_std_deg": float(jit.std()),
            "weight_by_count": fam_counts_total[fam] / tot if tot else 0.0,
            "instances_total": fam_counts_total[fam],
        }

    def summ(v):
        a = np.asarray(v, float)
        if not a.size:
            return {}
        return {"n": int(a.size), "mean": float(a.mean()), "p25": float(np.percentile(a, 25)),
                "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
                "p90": float(np.percentile(a, 90)), "p5": float(np.percentile(a, 5))}

    thick_gt_all = np.concatenate([all_thick_gt[f] for f in FAMILIES]) if any(all_thick_gt.values()) else np.array([])
    fwhm_all = np.concatenate([all_fwhm[f] for f in FAMILIES]) if any(all_fwhm.values()) else np.array([])
    seglen_all = np.concatenate([all_seg_len[f] for f in FAMILIES]) if any(all_seg_len.values()) else np.array([])
    complen_all = np.concatenate([all_comp_len[f] for f in FAMILIES]) if any(all_comp_len.values()) else np.array([])

    stats = {
        "provenance": {
            "gt_frames": [f["name"] for f in frames],
            "gt_magnification": 80000,
            "gt_in_training_band": False,
            "scale_to_50k": frames[0]["scale_to_50k"],
            "hfw_derived": True,
            "databar_rows_cropped": 70,
            "label_parse": {"spread_thr": 20, "min_px": 20, "connectivity": 4,
                            "fill_holes_per_family": True},
            "thin_mode_source": "raw-frame FWHM of dark dip (labels are ~4px hollow pen strokes)",
            "thick_tail_source": "GT filled labels, 2*EDT on skeleton",
            "coverage_target_source": "PENDING hand-corrected in-band frame (do NOT extrapolate 80k GT)",
        },
        "polarity": {
            "gt_wire_px_darker_than_local_bg_frac": 0.753,
            "bright_only_rule_recall": 0.114,
            "bright_only_rule_precision": 0.202,
            "note": "wires are DARK bodies with bright rims; bright-only thresholds fail",
        },
        "coverage_gt_filled": {f["name"]: pf["coverage_filled"]
                               for f, pf in zip(frames, per_frame)},
        "counting_convention": "connected components (4-conn) within one colour "
                               "family on filled labels; splits junctions between "
                               "families, merges same-family contacts",
        "families": fam_stats,
        "thickness_px_50k": {
            "gt_label_derived_ALL": summ(thick_gt_all),
            "gt_thick_tail_USE_p75_p90": {"p75": summ(thick_gt_all).get("p75"),
                                          "p90": summ(thick_gt_all).get("p90")},
            "raw_fwhm_thin_mode_USE": summ(fwhm_all),
            "per_family_fwhm": {f: summ(all_fwhm[f]) for f in FAMILIES},
        },
        "segment_length_px_50k": summ(seglen_all),
        "component_length_px_50k": summ(complen_all),
        "per_frame": per_frame,
    }

    with open(OUT / "real_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    # ---- npz: full histograms with bin edges --------------------------------
    npz = {}
    for name, arr, rng_ in [("segment_length", seglen_all, (0, 200)),
                            ("thickness_fwhm", fwhm_all, (0, 20)),
                            ("thickness_gt", thick_gt_all, (0, 20))]:
        h, e = np.histogram(arr, bins=60, range=rng_)
        npz[f"{name}_hist"] = h
        npz[f"{name}_edges"] = e
        npz[f"{name}_values"] = arr
    for fam in FAMILIES:
        h, e = np.histogram(all_seg_ang[fam], bins=90, range=(0, 180))
        npz[f"orientation_{fam}_hist"] = h
        npz[f"orientation_{fam}_edges"] = e
    ang_all = np.concatenate([all_seg_ang[f] for f in FAMILIES])
    h, e = np.histogram(ang_all, bins=90, range=(0, 180))
    npz["orientation_hist"] = h
    npz["orientation_edges"] = e
    npz["orientation_values"] = ang_all
    np.savez(OUT / "real_stats.npz", **npz)

    # ---- orientation plot ---------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].hist(ang_all, bins=90, range=(0, 180), color="0.4")
    for fam, c in zip(FAMILIES, ["tab:red", "tab:green", "tab:blue"]):
        ax[0].axvline(fam_stats[fam]["mean_deg"], color=c, ls="--",
                      label=f"{fam} {fam_stats[fam]['mean_deg']:.1f}deg")
    ax[0].set_xlabel("segment orientation (deg)"); ax[0].set_ylabel("count")
    ax[0].set_title("Real orientation anisotropy (GT segments)"); ax[0].legend(fontsize=8)
    ax[1].hist(fwhm_all, bins=50, range=(0, 20), alpha=0.6, label="raw FWHM (thin mode)")
    ax[1].hist(thick_gt_all, bins=50, range=(0, 20), alpha=0.5, label="GT label (pen-contaminated)")
    ax[1].set_xlabel("thickness (px @50k)"); ax[1].legend(fontsize=8)
    ax[1].set_title("Thickness: raw FWHM vs GT-label")
    fig.tight_layout()
    fig.savefig(OUT / "orientation_histogram.png", dpi=110)

    # ---- summary table ------------------------------------------------------
    print("\n=== FAMILIES (weights from instance counts) ===")
    print(f"{'family':8s} {'mean deg':>9s} {'jitter sd':>10s} {'weight':>8s} {'n':>6s}")
    for fam in FAMILIES:
        d = fam_stats[fam]
        print(f"{fam:8s} {d['mean_deg']:9.1f} {d['jitter_std_deg']:10.1f} "
              f"{d['weight_by_count']*100:7.1f}% {d['instances_total']:6d}")
    print("\n=== THICKNESS px @50k ===")
    print(f"  raw FWHM (thin mode, USE): {summ(fwhm_all)}")
    print(f"  GT label tail p75/p90    : {summ(thick_gt_all).get('p75'):.2f} / {summ(thick_gt_all).get('p90'):.2f}")
    print("\n=== LENGTH px @50k ===")
    print(f"  segment  : {summ(seglen_all)}")
    print(f"  component: {summ(complen_all)}")
    print(f"\nwrote {OUT}/real_stats.json, real_stats.npz, orientation_histogram.png")


if __name__ == "__main__":
    main()
