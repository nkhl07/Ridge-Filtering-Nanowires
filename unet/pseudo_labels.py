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
  1. confidence split on the sato response resp vs the frame's Otsu thr
     (no probability map exists for this teacher), with a boundary ignore band
  2. narrow AND locally-dark hole fill inside wire components
  3. small, round AND compact (filled solidity) teacher components -> background;
     those under tiny_ignore_px -> ignore
  extras (applied after the graph, whose median wire width sets their radius):
          teacher px within 2x median wire width of a saturated bright blob -> ignore;
          edges of large dark regions (charging bands) -> ignore
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
from skimage.morphology import skeletonize

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
                     lo_mult: float, hi_mult: float, band_px: float) -> np.ndarray:
    """Step 1. core = resp > hi_mult*thr (hi_mult 1.0 -> the teacher's own
    pre-closing mask). Pixels within band_px of the core boundary, on either
    side, are ignore; the rest of the core is wire (the core eroded by
    band_px). Background = resp <= lo_mult*thr, outside the teacher mask (its
    closing adds low-response px) and outside the band. Everything else --
    closing-added px and the lo..hi halo -- is ignore. band_px 0 disables the band."""
    core = resp > hi_mult * thr
    band = np.zeros(resp.shape, bool)
    if band_px > 0:
        band = ((core & (ndi.distance_transform_edt(core) <= band_px))
                | (~core & (ndi.distance_transform_edt(~core) <= band_px)))
    lab = np.full(resp.shape, IGNORE, np.uint8)
    lab[(resp <= lo_mult * thr) & ~teacher & ~band] = BG
    lab[core & ~band] = WIRE
    return lab


def narrow_holes(wire: np.ndarray, max_halfwidth: float, gray: np.ndarray,
                 window: int) -> tuple[np.ndarray, dict]:
    """Step 2. Holes enclosed by wire are filled only if (a) narrow: max
    distance-transform value <= max_halfwidth (hollow wire bodies; background
    boxed in by crossing wires is wide and stays), and (b) dark: mean gray of
    the hole < median gray of its local window (hole bbox padded by window/2
    on every side). (b) rejects bright debris whose dark rim the ridge filter
    traces. Returns (fill mask, counts)."""
    holes = ndi.binary_fill_holes(wire) & ~wire
    hl, n = ndi.label(holes)         # 4-conn, same as binary_fill_holes' flood
    counts = {"holes_total": int(n), "holes_narrow": 0, "holes_filled": 0}
    if n == 0:
        return np.zeros_like(wire), counts
    idx = np.arange(1, n + 1)
    narrow = np.asarray(ndi.maximum(ndi.distance_transform_edt(holes), hl, idx)) <= max_halfwidth
    mean = np.asarray(ndi.mean(gray, hl, idx))
    keep = np.zeros(n + 1, bool)
    pad = window // 2
    H, W = gray.shape
    for i, sl in enumerate(ndi.find_objects(hl)):
        if sl is None or not narrow[i]:
            continue
        win = gray[max(sl[0].start - pad, 0):min(sl[0].stop + pad, H),
                   max(sl[1].start - pad, 0):min(sl[1].stop + pad, W)]
        keep[i + 1] = mean[i] < np.median(win)
    counts["holes_narrow"] = int(narrow.sum())
    counts["holes_filled"] = int(keep.sum())
    return keep[hl], counts


def round_particles(mask: np.ndarray, max_ratio: float, max_area: int, min_solidity: float,
                    audit_min_area: int | None = None, tiny_px: int = 0) -> dict:
    """Step 3. 8-connected components of `mask` (whole teacher components, not
    confident-core fragments) are particles if small (area < max_area), round
    (major/minor axis ratio < max_ratio) AND compact (filled solidity =
    hole-filled area / convex-hull area >= min_solidity). The solidity test
    keeps short V/L-shaped wires, whose moments look round but whose hull is
    mostly empty; filling first keeps ring-shaped pits compact.

    Particles under tiny_px px go to `tiny` (caller marks them ignore: a speck
    cannot be told from a wire fragment); the rest go to `removed`
    (background). `audit` = removed particles with area >= audit_min_area.
    """
    lab, _ = ndi.label(mask, structure=ONES3)
    out = {k: np.zeros_like(mask) for k in ("removed", "audit", "tiny")}
    n = {k: 0 for k in ("removed", "audit", "tiny", "kept_by_solidity")}
    for rp in regionprops(lab):
        if rp.area >= max_area:
            continue
        if rp.axis_major_length / max(rp.axis_minor_length, 1.0) >= max_ratio:
            continue
        if rp.area_filled / max(rp.area_convex, 1) < min_solidity:
            n["kept_by_solidity"] += 1
            continue
        rr, cc = rp.coords[:, 0], rp.coords[:, 1]
        if rp.area < tiny_px:
            out["tiny"][rr, cc] = True
            n["tiny"] += 1
            continue
        out["removed"][rr, cc] = True
        n["removed"] += 1
        if audit_min_area is not None and rp.area >= audit_min_area:
            out["audit"][rr, cc] = True
            n["audit"] += 1
    return {"masks": out, "counts": n}


def saturated_near(mask: np.ndarray, gray: np.ndarray, level: float, min_blob_px: int,
                   radius: float) -> tuple[np.ndarray, int]:
    """Extra. Pixels of `mask` within `radius` px of a saturated bright blob
    (gray >= level, >= min_blob_px px), i.e. the blob mask dilated by radius:
    the ridge filter traces the dark rim around bright debris. Local on
    purpose -- whole components would take entire wire networks linked by the
    teacher's closing. Returns (mask of those px, n blobs)."""
    sat = gray >= level
    bl, nb = ndi.label(sat, structure=ONES3)
    if nb == 0 or not np.isfinite(radius):
        return np.zeros_like(mask), 0
    big = np.bincount(bl.ravel()) >= min_blob_px
    big[0] = False
    blobs = big[bl]
    if not blobs.any():
        return np.zeros_like(mask), 0
    return mask & (ndi.distance_transform_edt(~blobs) <= radius), int(big.sum())


def dark_region_edges(gray: np.ndarray, sigma: float, dark_frac: float,
                      min_area_frac: float, margin_px: float) -> tuple[np.ndarray, int, float]:
    """Extra. Edges of large dark regions (charging bands etc.): regions where
    the Gaussian-smoothed gray is below dark_frac x the frame median and that
    cover >= min_area_frac of the frame. Pixels within margin_px of such a
    region's boundary (either side) are returned -- the ridge filter fires on
    that step edge. Returns (edge mask, n regions, region area fraction)."""
    sm = ndi.gaussian_filter(gray.astype(np.float32), sigma)
    dark = sm < dark_frac * float(np.median(gray))
    dl, nd = ndi.label(dark)
    if nd == 0:
        return np.zeros(gray.shape, bool), 0, 0.0
    sizes = np.bincount(dl.ravel())
    big = sizes >= min_area_frac * gray.size
    big[0] = False
    region = big[dl]
    if not region.any():
        return np.zeros(gray.shape, bool), 0, 0.0
    edge = ((region & (ndi.distance_transform_edt(region) <= margin_px))
            | (~region & (ndi.distance_transform_edt(~region) <= margin_px)))
    return edge, int(big.sum()), float(region.mean())


def median_width_px(mask: np.ndarray) -> float:
    """Median of 2*EDT over the skeleton of `mask` (wire_extract's width convention)."""
    if not mask.any():
        return float("nan")
    sk = skeletonize(mask)
    return float(np.median(2.0 * ndi.distance_transform_edt(mask)[sk]))


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
    pre = resp > thr                         # teacher mask before its closing
    st: dict = {"teacher": {"thr": thr, "angles_deg": info["angles_deg"],
                            "coverage": float(teacher.mean()),
                            "coverage_preclose": float(pre.mean()),
                            "width_preclose_nm": median_width_px(pre) * NM_PER_PX}}
    m: dict = {}
    zero = np.zeros(gray.shape, bool)

    c = pcfg["confidence"]
    if c["enabled"]:
        label = confidence_split(resp, thr, teacher, c["lo_mult"], c["hi_mult"], c["band_px"])
    else:
        label = np.where(teacher, WIRE, BG).astype(np.uint8)
    st["after_confidence"] = {"wire": float((label == WIRE).mean()),
                              "ignore": float((label == IGNORE).mean())}

    h = pcfg["hole_fill"]
    m["filled"] = zero
    st.update({"holes_total": 0, "holes_narrow": 0, "holes_filled": 0})
    if h["enabled"]:
        m["filled"], counts = narrow_holes(label == WIRE, h["max_hole_halfwidth_px"],
                                           gray, h["local_window_px"])
        st.update(counts)
        label[m["filled"]] = WIRE
    st["holes_filled_px"] = int(m["filled"].sum())
    comps = teacher | m["filled"]            # whole teacher components (+ fills)

    pc = pcfg["particles"]
    m["removed"] = m["removed_audit"] = m["tiny"] = zero
    st.update({"particles_removed": 0, "particles_removed_audit": 0, "particles_tiny_ignored": 0,
               "particles_kept_by_solidity": 0})
    if pc["enabled"]:
        r = round_particles(comps, pc["max_axis_ratio"], pc["max_particle_area_px"],
                            pc["min_filled_solidity"], pc.get("audit_prev_max_area_px"),
                            pc["tiny_ignore_px"])
        m["removed"], m["removed_audit"], m["tiny"] = (r["masks"][k] for k in ("removed", "audit", "tiny"))
        st.update({"particles_removed": r["counts"]["removed"],
                   "particles_removed_audit": r["counts"]["audit"],
                   "particles_tiny_ignored": r["counts"]["tiny"],
                   "particles_kept_by_solidity": r["counts"]["kept_by_solidity"]})
        label[m["removed"]] = BG
        label[m["tiny"]] = IGNORE
    st["particles_removed_px"] = int(m["removed"].sum())
    st["particles_removed_audit_px"] = int(m["removed_audit"].sum())
    st["particles_tiny_ignored_px"] = int(m["tiny"].sum())

    # graph (steps 4-5) runs before the extras: the saturated-blob radius is
    # defined from its median wire width, the same width the junction bands use
    q, jb = pcfg["qc_chains"], pcfg["junction_band"]
    m["qc"] = zero
    m["band"] = zero
    m["nodes3"] = []
    st["chains_qc_ignored"] = 0
    width = float("nan")
    if q["enabled"] or jb["enabled"]:
        gm = graph_mask(teacher, label, m["filled"])
        g = graph_qc(gm, label == WIRE, info["angles_deg"],
                     wire_extract_params(q["graph_junction_close"]), q["fail_flags"])
        width = g["median_width_px"]
        st.update({"graph_mask_coverage": float(gm.mean()),
                   "chains_total": g["n_chains"], "junction_nodes_all": g["n_junction_nodes"],
                   "junction_nodes_deg3": len(g["nodes3"]),
                   "median_width_px": width, "median_width_nm": width * NM_PER_PX})
        if q["enabled"]:
            m["qc"] = g["qc_px"]
            st["chains_qc_ignored"] = g["n_failed"]
            label[m["qc"]] = IGNORE
        if jb["enabled"]:
            radius = (jb["radius_width_mult"] * width
                      if jb["radius_px"] == "auto" else float(jb["radius_px"]))
            st["junction_radius_px"] = radius
            if np.isfinite(radius):
                m["band"] = junction_band(label.shape, g["nodes3"], radius)
                m["nodes3"] = g["nodes3"]
                label[m["band"]] = IGNORE
            st["junction_bands"] = len(m["nodes3"])
    st["qc_ignored_px"] = int(m["qc"].sum())
    st["junction_band_px"] = int(m["band"].sum())
    if not np.isfinite(width):                   # graph disabled or empty
        width = median_width_px(teacher)

    sc = pcfg["saturated"]
    m["saturated"] = zero
    st["saturated_blobs"] = 0
    if sc["enabled"]:
        r = sc["radius_width_mult"] * width
        st["saturated_radius_px"] = r
        m["saturated"], st["saturated_blobs"] = saturated_near(
            comps & ~m["removed"] & ~m["tiny"], gray, sc["level"], sc["min_blob_px"], r)
        label[m["saturated"]] = IGNORE
    st["saturated_ignored_px"] = int(m["saturated"].sum())

    dc = pcfg["dark_region"]
    m["dark_edge"] = zero
    st["dark_regions"], st["dark_region_frac"] = 0, 0.0
    if dc["enabled"]:
        m["dark_edge"], st["dark_regions"], st["dark_region_frac"] = dark_region_edges(
            gray, dc["sigma_px"], dc["dark_frac"], dc["min_area_frac"], dc["margin_px"])
        label[m["dark_edge"]] = IGNORE
    st["dark_edge_px"] = int(m["dark_edge"].sum())

    w = median_width_px(label == WIRE)
    st["label_wire_width_px"], st["label_wire_width_nm"] = w, w * NM_PER_PX
    st["fractions"] = {"wire": float((label == WIRE).mean()),
                       "background": float((label == BG).mean()),
                       "ignore": float((label == IGNORE).mean())}
    return label, resp, m, st


# --------------------------------------------------------------------------
# QC overlay
# --------------------------------------------------------------------------

def _tint(img: np.ndarray, mask: np.ndarray, color, alpha: float) -> None:
    img[mask] = ((1 - alpha) * img[mask] + alpha * np.asarray(color, float)).astype(np.uint8)


def _outline(img: np.ndarray, mask: np.ndarray, color) -> None:
    edge = mask & ~ndi.binary_erosion(mask)
    img[ndi.binary_dilation(edge)] = color


C_WIRE, C_IGN, C_QC = (0, 200, 0), (230, 200, 0), (255, 0, 255)
C_JUNC, C_HOLE, C_PART = (255, 140, 0), (0, 120, 255), (255, 0, 0)
C_SAT, C_DARK, C_AUDIT, C_TINY = (0, 230, 230), (150, 80, 30), (140, 0, 255), (255, 150, 200)
LEGEND = [("wire", C_WIRE), ("ignore", C_IGN), ("QC-failed chain", C_QC),
          ("junction band", C_JUNC), ("filled hole", C_HOLE), ("removed particle", C_PART),
          ("removed, area>=220", C_AUDIT), ("tiny particle (ignore)", C_TINY),
          ("near saturated blob", C_SAT), ("dark-region edge", C_DARK)]


def qc_overlay(gray: np.ndarray, label: np.ndarray, m: dict, title: str) -> np.ndarray:
    g8 = np.clip(gray, 0, 255).astype(np.uint8)
    base = np.repeat(g8[..., None], 3, axis=2)
    ov = base.copy()
    special = m["qc"] | m["band"] | m["saturated"] | m["dark_edge"] | m["tiny"]
    _tint(ov, label == WIRE, C_WIRE, 0.35)
    _tint(ov, (label == IGNORE) & ~special, C_IGN, 0.35)
    _tint(ov, m["qc"], C_QC, 0.55)
    _tint(ov, m["band"] & ~m["qc"], C_IGN, 0.35)
    for r, c in m["nodes3"]:
        cv2.circle(ov, (int(round(c)), int(round(r))), 2, C_JUNC, -1)
    ov[m["band"] & ~ndi.binary_erosion(m["band"])] = C_JUNC
    _tint(ov, m["saturated"], C_SAT, 0.55)
    _tint(ov, m["dark_edge"], C_DARK, 0.6)
    _tint(ov, m["filled"], C_HOLE, 0.75)
    old = m["removed"] & ~m["removed_audit"]
    _tint(ov, old, C_PART, 0.6)
    _outline(ov, old, C_PART)                    # thick outline: removals must be visible
    _tint(ov, m["removed_audit"], C_AUDIT, 0.6)
    _outline(ov, m["removed_audit"], C_AUDIT)
    _tint(ov, m["tiny"], C_TINY, 0.8)

    panel = np.concatenate([base, np.full((g8.shape[0], 8, 3), 255, np.uint8), ov], axis=1)
    bar = np.full((52, panel.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    x = 6
    for name, col in LEGEND:
        cv2.rectangle(bar, (x, 32), (x + 12, 44), col, -1)
        cv2.putText(bar, name, (x + 16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        x += 26 + 8 * len(name)
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
                   "train_excluded": any(x in name for x in cfg["frames"]["exclude_from_training"]),
                   "seconds": round(time.time() - t0, 1)})
        fr = st["fractions"]
        title = (f"{name}  x{factor:.3f}  wire {fr['wire']:.1%}  bg {fr['background']:.1%}  "
                 f"ignore {fr['ignore']:.1%}  label width {st['label_wire_width_nm']:.0f} nm"
                 + ("  [EXCLUDED FROM TRAINING]" if st["train_excluded"] else ""))
        Image.fromarray(qc_overlay(gray, label, m, title)).save(out / f"{name}_qc.png")
        per_frame[name] = st
        print(f"[{i:2d}/{len(frames)}] {title}  holes {st['holes_filled']}/{st['holes_narrow']}"
              f"  particles {st['particles_removed']} (>=220px {st['particles_removed_audit']},"
              f" tiny->ign {st['particles_tiny_ignored']}, kept-by-solidity {st['particles_kept_by_solidity']})"
              f"  sat-blobs {st['saturated_blobs']} r={st.get('saturated_radius_px', float('nan')):.1f}"
              f"  dark {st['dark_regions']} ({st['dark_region_frac']:.1%})"
              f"  qc-chains {st['chains_qc_ignored']}/{st.get('chains_total', 0)}"
              f"  bands {st.get('junction_bands', 0)}  r={st.get('junction_radius_px', float('nan')):.1f}px"
              f"  ({st['seconds']}s)", flush=True)

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
        "frames_excluded_from_training": sorted(n for n, s in per_frame.items()
                                                if s["train_excluded"]),
        "frames_with_dark_region": sorted(n for n, s in per_frame.items() if s["dark_regions"]),
        "per_frame": per_frame,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=wx._jsonable))
    print(json.dumps(summary["aggregate_fractions"], indent=1))
    print("flagged (ignore > {:.0%}): {}".format(pcfg["ignore_flag_frac"],
                                                 summary["frames_ignore_above_flag"]))


if __name__ == "__main__":
    main()
