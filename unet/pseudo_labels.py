"""unet_v1 Phase 1: cleaned pseudo-labels from the label-free ridge teacher.

Teacher = the implemented production ridge filter, labelfree.labelfree_mask
(sato -> per-frame Otsu -> directional closing at image-estimated angles).
For every in-band non-GT frame (configs/unet_v1.yaml `frames`) this writes,
under results/unet_v1/pseudo_labels/:

  <name>_img.npy     preprocessed image, float32 [0,255] at 50k-equivalent scale
                     (ridge_nanowire_images2.load_frame_50k: databar crop at native
                     scale, min-max, HFW/PixelWidth-derived rescale)
  <name>_prob.npy    teacher score map: min-max normalised sato response (float32);
                     the frame's Otsu threshold is in summary.json
  <name>_label.png   uint8 cleaned label: 0 background, 1 wire, 255 ignore
  <name>_qc.png      [image | overlay] audit panel
  summary.json, config.yaml, provenance.json

Cleaning steps (order fixed, each toggleable in the config):
  1. confidence split on resp / thr   (no probability map exists for this teacher)
  2. narrow-hole fill inside wire components
  3. round small particle removal -> background
  4. wire px of graph chains failing curvature QC -> ignore (graph built on
     the teacher mask restricted to confidently supported components; see
     graph_mask)
  5. band around degree>=3 graph nodes -> ignore

The GT frames are never read here.

Run via slurm/unet_pseudo_labels.sbatch (real frames; not on the login node).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from skimage.measure import regionprops

from unet.common import (BG, IGNORE, ROOT, WIRE, load_config, rel, select_frames,
                         write_provenance)
import labelfree as lf  # noqa: E402  (on sys.path via unet.common)
import wire_extract as wx  # noqa: E402
from ridge_nanowire_images2 import TARGET_PX, load_frame_50k  # noqa: E402

ONES3 = np.ones((3, 3), bool)
NM_PER_PX = TARGET_PX * 1e9         # every frame is resampled to this scale


# --------------------------------------------------------------------------
# cleaning steps (pure functions on arrays; unit-tested in test_pseudo_labels)
# --------------------------------------------------------------------------

def confidence_split(resp: np.ndarray, thr: float, teacher: np.ndarray,
                     lo_mult: float, hi_mult: float) -> np.ndarray:
    """Step 1. wire if resp >= hi*thr; background if resp <= lo*thr and the
    teacher did not mark it (its closing adds low-response px); else ignore."""
    lab = np.full(resp.shape, IGNORE, np.uint8)
    lab[(resp <= lo_mult * thr) & ~teacher] = BG
    lab[resp >= hi_mult * thr] = WIRE
    return lab


def narrow_holes(wire: np.ndarray, max_halfwidth: float) -> tuple[np.ndarray, int, int]:
    """Step 2. Holes enclosed by wire whose max distance-transform value is
    <= max_halfwidth (hollow wire bodies). Wide enclosed regions -- background
    boxed in by crossing wires -- are left alone. Returns (fill mask, n filled,
    n holes)."""
    holes = ndi.binary_fill_holes(wire) & ~wire
    hl, n = ndi.label(holes)         # 4-conn, same as binary_fill_holes' flood
    if n == 0:
        return np.zeros_like(wire), 0, 0
    halfwidth = ndi.maximum(ndi.distance_transform_edt(holes), hl, np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = np.asarray(halfwidth) <= max_halfwidth
    return keep[hl], int(keep.sum()), n


def round_particles(wire: np.ndarray, max_ratio: float, max_area: int) -> tuple[np.ndarray, int]:
    """Step 3. 8-connected wire components that are both round (major/minor
    axis ratio < max_ratio) and small (area < max_area). Returns (mask, count)."""
    lab, _ = ndi.label(wire, structure=ONES3)
    out = np.zeros_like(wire)
    n = 0
    for rp in regionprops(lab):
        if rp.area >= max_area:
            continue
        if rp.axis_major_length / max(rp.axis_minor_length, 1.0) < max_ratio:
            out[rp.coords[:, 0], rp.coords[:, 1]] = True
            n += 1
    return out, n


def wire_extract_params(junction_close: bool):
    """wire_extract's own defaults (its CLI with no arguments)."""
    p = wx.parse_args([])
    p.junction_close = junction_close
    return p


def graph_mask(teacher: np.ndarray, label: np.ndarray, filled: np.ndarray) -> np.ndarray:
    """Mask the graph is built on after steps 1-3.

    The confident-wire core (label == WIRE) alone is the wrong graph input:
    sato dips at crossings, so the core is cut at nearly every junction (no
    junction nodes at all), and closing it again fills crossing corners that
    the skeleton then cuts (false curvature fails). Instead: the teacher mask
    (already direction-closed, which is what wire_extract normally consumes)
    plus the step-2 fills, restricted to the 8-connected components that still
    hold confident wire after step 3 -- so removed particles and components
    with no confident support drop out.
    """
    base = teacher | filled
    comp, _ = ndi.label(base, structure=ONES3)
    keep = np.unique(comp[label == WIRE])
    return np.isin(comp, keep[keep > 0])


def graph_qc(gmask: np.ndarray, wire: np.ndarray, angles, p, fail_flags) -> dict:
    """Steps 4+5 inputs: run the wire_extract graph on `gmask`.

    Returns the `wire` px that belong to QC-failed chains (each px is assigned
    to its nearest chain skeleton px inside the same connected component of
    the graph mask), the degree>=3 node centroids, and the median per-chain
    mean width (px).
    """
    records, summary, render = wx.extract_from_mask(gmask, NM_PER_PX, p, close_angles=angles)
    H, W = wire.shape
    fail_flags = set(fail_flags)
    chain_id = np.zeros((H, W), np.int32)
    failed = np.zeros(len(records) + 1, bool)
    for i, r in enumerate(records, 1):
        pix = np.asarray(r["_pixels"], int)
        chain_id[pix[:, 0], pix[:, 1]] = i
        failed[i] = bool(fail_flags & set(filter(None, r["qc_flags"].split(";"))))

    qc_px = np.zeros_like(wire)
    if (chain_id > 0).any():
        _, (ir, ic) = ndi.distance_transform_edt(chain_id == 0, return_indices=True)
        comp, _ = ndi.label(render["mask_clean"], structure=ONES3)
        qc_px = wire & (comp > 0) & (comp == comp[ir, ic]) & failed[chain_id[ir, ic]]

    # degree = incident edge ends per junction node. render["junction_centroids"]
    # is ordered by sorted real-junction label, and those labels are exactly
    # the junction labels the edges reference.
    deg = Counter(nd[1] for e in render["edges"] for nd in e["nodes"] if nd[0] == "J")
    cents = render["junction_centroids"]
    assert len(deg) == len(cents), (len(deg), len(cents))
    nodes3 = [c for lab_, c in zip(sorted(deg), cents) if deg[lab_] >= 3]

    width = summary["wires_all"]["width_mean_px"].get("median", float("nan"))
    return {"qc_px": qc_px, "n_chains": len(records), "n_failed": int(failed.sum()),
            "nodes3": nodes3, "n_junction_nodes": len(cents),
            "median_width_px": float(width)}


def junction_band(shape, nodes, radius: float) -> np.ndarray:
    """Step 5. All px within `radius` of any node centroid."""
    seed = np.ones(shape, bool)
    for r, c in nodes:
        seed[min(max(int(round(r)), 0), shape[0] - 1), min(max(int(round(c)), 0), shape[1] - 1)] = False
    if seed.all():
        return np.zeros(shape, bool)
    return ndi.distance_transform_edt(seed) <= radius


def clean_frame(gray: np.ndarray, pcfg: dict) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Teacher + all cleaning steps on one preprocessed frame.

    Returns (label, resp, masks for the overlay, stats)."""
    teacher, resp, info = lf.labelfree_mask(gray, method="otsu")
    thr = info["thr"]
    st: dict = {"teacher": {"thr": thr, "angles_deg": info["angles_deg"],
                            "coverage": float(teacher.mean())}}
    m: dict = {}

    c = pcfg["confidence"]
    if c["enabled"]:
        label = confidence_split(resp, thr, teacher, c["lo_mult"], c["hi_mult"])
    else:
        label = np.where(teacher, WIRE, BG).astype(np.uint8)
    st["after_confidence"] = {"wire": float((label == WIRE).mean()),
                              "ignore": float((label == IGNORE).mean())}

    h = pcfg["hole_fill"]
    m["filled"] = np.zeros(label.shape, bool)
    st["holes_total"] = st["holes_filled"] = 0
    if h["enabled"]:
        m["filled"], st["holes_filled"], st["holes_total"] = narrow_holes(
            label == WIRE, h["max_hole_halfwidth_px"])
        label[m["filled"]] = WIRE
    st["holes_filled_px"] = int(m["filled"].sum())

    pc = pcfg["particles"]
    m["removed"] = np.zeros(label.shape, bool)
    st["particles_removed"] = 0
    if pc["enabled"]:
        m["removed"], st["particles_removed"] = round_particles(
            label == WIRE, pc["max_axis_ratio"], pc["max_particle_area_px"])
        label[m["removed"]] = BG
    st["particles_removed_px"] = int(m["removed"].sum())

    q, jb = pcfg["qc_chains"], pcfg["junction_band"]
    m["qc"] = np.zeros(label.shape, bool)
    m["band"] = np.zeros(label.shape, bool)
    m["nodes3"] = []
    if q["enabled"] or jb["enabled"]:
        gm = graph_mask(teacher, label, m["filled"])
        g = graph_qc(gm, label == WIRE, info["angles_deg"],
                     wire_extract_params(q["graph_junction_close"]), q["fail_flags"])
        st.update({"graph_mask_coverage": float(gm.mean()),
                   "chains_total": g["n_chains"], "junction_nodes_all": g["n_junction_nodes"],
                   "junction_nodes_deg3": len(g["nodes3"]),
                   "median_width_px": g["median_width_px"],
                   "median_width_nm": g["median_width_px"] * NM_PER_PX})
        if q["enabled"]:
            m["qc"] = g["qc_px"]
            st["chains_qc_ignored"] = g["n_failed"]
            label[m["qc"]] = IGNORE
        if jb["enabled"]:
            radius = (jb["radius_width_mult"] * g["median_width_px"]
                      if jb["radius_px"] == "auto" else float(jb["radius_px"]))
            st["junction_radius_px"] = radius
            if np.isfinite(radius):
                m["band"] = junction_band(label.shape, g["nodes3"], radius)
                m["nodes3"] = g["nodes3"]
                label[m["band"]] = IGNORE
            st["junction_bands"] = len(m["nodes3"])
    st["qc_ignored_px"] = int(m["qc"].sum())
    st["junction_band_px"] = int(m["band"].sum())

    st["fractions"] = {"wire": float((label == WIRE).mean()),
                       "background": float((label == BG).mean()),
                       "ignore": float((label == IGNORE).mean())}
    return label, resp, m, st


# --------------------------------------------------------------------------
# QC overlay
# --------------------------------------------------------------------------

def _tint(img: np.ndarray, mask: np.ndarray, color, alpha: float) -> None:
    img[mask] = ((1 - alpha) * img[mask] + alpha * np.asarray(color, float)).astype(np.uint8)


LEGEND = [("wire", (0, 200, 0)), ("ignore", (230, 200, 0)), ("QC-failed chain (ignore)", (255, 0, 255)),
          ("junction band", (255, 140, 0)), ("filled hole", (0, 120, 255)),
          ("removed particle", (255, 0, 0))]


def qc_overlay(gray: np.ndarray, label: np.ndarray, m: dict, title: str) -> np.ndarray:
    g8 = np.clip(gray, 0, 255).astype(np.uint8)
    base = np.repeat(g8[..., None], 3, axis=2)
    ov = base.copy()
    _tint(ov, label == WIRE, LEGEND[0][1], 0.35)
    _tint(ov, (label == IGNORE) & ~m["qc"] & ~m["band"], LEGEND[1][1], 0.35)
    _tint(ov, m["qc"], LEGEND[2][1], 0.55)
    _tint(ov, m["band"] & ~m["qc"], LEGEND[1][1], 0.35)
    for r, c in m["nodes3"]:
        cv2.circle(ov, (int(round(c)), int(round(r))), 2, LEGEND[3][1], -1)
    band_edge = m["band"] & ~ndi.binary_erosion(m["band"])
    ov[band_edge] = LEGEND[3][1]
    _tint(ov, m["filled"], LEGEND[4][1], 0.75)
    _tint(ov, m["removed"], LEGEND[5][1], 0.6)
    edge = m["removed"] & ~ndi.binary_erosion(m["removed"])
    ov[ndi.binary_dilation(edge)] = LEGEND[5][1]      # thick outline: removals must be visible

    panel = np.concatenate([base, np.full((g8.shape[0], 8, 3), 255, np.uint8), ov], axis=1)
    bar = np.full((34, panel.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    x = panel.shape[1] // 2 - 200
    for name, col in LEGEND:
        cv2.rectangle(bar, (x, 10), (x + 14, 24), col, -1)
        cv2.putText(bar, name, (x + 18, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        x += 26 + 9 * len(name)
    return np.concatenate([bar, panel], axis=0)


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "unet_v1.yaml")
    ap.add_argument("--only", default="", help="substring filter on frame names (debug)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pcfg = cfg["pseudo_labels"]
    assert pcfg["teacher"] == "lf_otsu", "only the label-free Otsu teacher is implemented"
    np.random.seed(cfg["seed"])                       # nothing here is random; recorded anyway

    out = ROOT / cfg["paths"]["out_root"] / "pseudo_labels"
    frames = [(p, f) for p, f in select_frames(cfg) if args.only in p.name]
    prov = write_provenance(out, args.config, {"n_frames": len(frames)})
    print(f"{len(frames)} frames -> {rel(out)}", flush=True)

    per_frame: dict = {}
    for i, (path, factor) in enumerate(frames, 1):
        t0 = time.time()
        gray, meta = load_frame_50k(path)
        assert abs(meta["rescale_to_50k"] - factor) < 1e-9
        label, resp, m, st = clean_frame(gray, pcfg)
        name = path.stem
        np.save(out / f"{name}_img.npy", gray.astype(np.float32))
        np.save(out / f"{name}_prob.npy", resp.astype(np.float32))
        Image.fromarray(label, mode="L").save(out / f"{name}_label.png")
        st.update({"filename_mag": meta["filename_mag"], "rescale_to_50k": factor,
                   "native_shape": meta["native_shape"], "shape_50k": list(gray.shape),
                   "ignore_flag": st["fractions"]["ignore"] > pcfg["ignore_flag_frac"],
                   "seconds": round(time.time() - t0, 1)})
        fr = st["fractions"]
        title = (f"{name}  x{factor:.3f}  wire {fr['wire']:.1%}  bg {fr['background']:.1%}  "
                 f"ignore {fr['ignore']:.1%}")
        Image.fromarray(qc_overlay(gray, label, m, title)).save(out / f"{name}_qc.png")
        per_frame[name] = st
        print(f"[{i:2d}/{len(frames)}] {title}  holes {st['holes_filled']}  "
              f"particles {st['particles_removed']}  qc-chains {st.get('chains_qc_ignored', 0)}"
              f"/{st.get('chains_total', 0)}  bands {st.get('junction_bands', 0)}  "
              f"r={st.get('junction_radius_px', float('nan')):.1f}px  ({st['seconds']}s)", flush=True)

    def agg(key):
        v = np.array([s["fractions"][key] for s in per_frame.values()])
        return {"median": float(np.median(v)), "mean": float(v.mean()),
                "min": float(v.min()), "max": float(v.max())} if v.size else {}

    summary = {
        "teacher": "labelfree.labelfree_mask(method='otsu') -- implemented production ridge filter",
        "label_values": {"background": BG, "wire": WIRE, "ignore": IGNORE},
        "nm_per_px": NM_PER_PX,
        "config": pcfg,
        "git": prov["git"],
        "n_frames": len(per_frame),
        "aggregate_fractions": {k: agg(k) for k in ("wire", "background", "ignore")},
        "frames_ignore_above_flag": sorted(n for n, s in per_frame.items() if s["ignore_flag"]),
        "per_frame": per_frame,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=wx._jsonable))
    print(json.dumps(summary["aggregate_fractions"], indent=1))
    print("flagged (ignore > {:.0%}): {}".format(pcfg["ignore_flag_frac"],
                                                 summary["frames_ignore_above_flag"]))


if __name__ == "__main__":
    main()
