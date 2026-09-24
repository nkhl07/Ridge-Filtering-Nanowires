"""Wire extraction stage: skeleton -> graph -> per-wire measurements (Phase 1).

Consumes the v3 pseudo-masks in results/ridge_nanowire_images2/ and converts
per-pixel masks into per-wire instances: counts, lengths, widths, orientation
families, junction statistics, QC flags. Flag-only pass: nothing is split or
deleted automatically; every suspect chain is flagged and kept.

Techniques adapted from fingerprint ridge-feature extraction (Nguyen, Wang &
Li 2013, J. Information Security and Applications, and the methods it builds
on). Wire polarity matches fingerprint ridgelines (dark ridges on bright
background), so no inversion anywhere.

Paper -> implementation map (keep in sync with the manuscript):
  - Rutovitz crossing-number skeleton classification + spurious-feature
    (spur) pruning: minutiae detection/filtering stage
    (Nguyen et al. 2013 Sec. 3; Zhao & Tang 2007).           -> crossing_number(), prune_spurs()
  - Short-bridge contraction between bifurcation clusters: spurious
    minutiae-pair (bridge) filtering (Zhao & Tang 2007).     -> merge_bridged_junctions()
  - Junction stub pairing by collinearity: adapted from ridge-type angle
    classification at bifurcations (Choi et al. 2011, via
    Nguyen et al. Fig. 5).                                   -> pair_stubs_at_junction()
  - Wire network graph (nodes = endpoints/junctions, edges = segments with
    attributes): adapted from K-plet graph storage
    (Chikkerur et al. 2005).                                 -> build_graph()
  - Turning-based curvature QC: adapted from ridge curvature direction,
    eq. 8 (Nguyen et al. 2013).                              -> rdp_indices(), measure_chain()
  - Orientation from path tangents and PCA: stands in for the
    Hong et al. 1998 orientation field.                      -> pca_orientation()
  - Out of scope (future work): direct gray-scale ridgeline tracing
    (Maio & Maltoni 1997) as an alternative to binarize-then-thin.

Scale: masks are at 50k-equivalent (~5.4 nm/px). Per-frame PixelWidth and
rescale come from ridge_coverage.json (FEI tag 34682, never filename mag).
Orientation convention matches measure_real_stats.py: degrees(arctan2(vy, vx))
% 180 on (row, col) point clouds; measured families blue ~0/180, green ~60,
red ~123.5.

Outputs per frame under results/wire_extract/<frame>/:
  wires.csv               one row per extracted wire chain
  summary.json            counts/stats/QC + the full parameter set (reproducibility)
  <frame>_instances.png   instance-colored skeleton over the raw frame,
                          junction (yellow circle) + endpoint (white dot) markers
  <frame>_families.png    orientation overlay in the GT hue convention
                          (blue 0 / green 60 / red 123.5; off-family = white)
Plus results/wire_extract/extract_summary.json across frames, and with
--validate-gt the measurement-level GT comparison gt_validation.json.

Deterministic throughout; no RNG anywhere in Phase 1.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import math
import sys
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from PIL import Image
from scipy.stats import ks_2samp
from skimage.morphology import disk, skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ridge_nanowire_images2 import (TARGET_PX, line_kernel, load_frame_50k,
                                    read_pixel_size)

ROOT = Path(__file__).resolve().parents[2]


def rel(path: Path) -> Path:
    """Path relative to ROOT for logs/JSON; absolute if it lies outside ROOT."""
    path = Path(path).resolve()
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path

TARGET_NM = 5.4                 # nominal 50k-equivalent pixel size for the scale assert
ONES3 = np.ones((3, 3), int)
# circular (CCW) 8-neighborhood order for the Rutovitz crossing number
OFFS_CCW = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
OFFS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _stats(x) -> dict:
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "median": float(np.median(x)), "mean": float(x.mean()),
            "p25": float(np.percentile(x, 25)), "p75": float(np.percentile(x, 75)),
            "min": float(x.min()), "max": float(x.max())}


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")


def pca_orientation(ys: np.ndarray, xs: np.ndarray) -> float:
    """Orientation in [0,180) of a point cloud via its principal axis.

    Same convention as measure_real_stats.py: degrees(arctan2(vy, vx)) % 180
    with y = row, x = col. Stands in for the Hong et al. 1998 orientation
    field (see module docstring map). NaN for degenerate clouds.
    """
    if len(ys) < 2:
        return float("nan")
    pts = np.stack([np.asarray(ys, float), np.asarray(xs, float)], 1)
    pts -= pts.mean(0)
    cov = pts.T @ pts
    w, v = np.linalg.eigh(cov)
    vy, vx = v[:, -1]                       # eigenvector of the larger eigenvalue
    if vy == 0 and vx == 0:
        return float("nan")
    return float(np.degrees(np.arctan2(vy, vx)) % 180.0)


def dev_180(angle: float, center: float) -> float:
    """Signed deviation of an orientation from a family center, in (-90, 90]."""
    return (angle - center + 90.0) % 180.0 - 90.0


def geodesic_len(path: list) -> float:
    """Skeleton path length: 1 per axial step, sqrt(2) per diagonal step."""
    if len(path) < 2:
        return 0.0
    steps = np.abs(np.diff(np.asarray(path, float), axis=0)).sum(1)
    return float(np.where(steps > 1.5, math.sqrt(2), 1.0).sum())


# --------------------------------------------------------------------------
# skeleton classification and decomposition
# --------------------------------------------------------------------------

def crossing_number(sk: np.ndarray) -> np.ndarray:
    """Rutovitz crossing number per skeleton pixel.

    CN = 0.5 * sum(|P_i - P_{i+1}|) over the circular 8-neighborhood
    (P_9 = P_1). CN=1 endpoint, CN=2 path, CN>=3 junction.
    (Nguyen et al. 2013 Sec. 3; Zhao & Tang 2007.)
    """
    p = np.pad(sk.astype(np.int16), 1)
    H, W = sk.shape
    nb = [p[1 + dr:1 + dr + H, 1 + dc:1 + dc + W] for dr, dc in OFFS_CCW]
    s = np.zeros((H, W), np.int16)
    for i in range(8):
        s += np.abs(nb[i] - nb[(i + 1) % 8])
    cn = (s // 2).astype(np.uint8)
    cn[~sk] = 0
    return cn


def trace_component(coords: list) -> tuple[list, int]:
    """Order the pixels of one 8-connected branch component into path(s).

    Returns ([(path, is_cycle), ...], n_incomplete). Branch components are
    simple paths in the normal case (CN<=2 pixels only); thick corners can
    leave leftovers, which are traced as extra paths and counted.
    """
    cset = set(coords)
    nbrs = {pt: [q for dr, dc in OFFS8
                 for q in [(pt[0] + dr, pt[1] + dc)] if q in cset]
            for pt in coords}
    termini = sorted(pt for pt in coords if len(nbrs[pt]) <= 1)
    unvisited = set(coords)

    def walk(start):
        path = [start]
        unvisited.discard(start)
        cur = start
        while True:
            cand = [q for q in nbrs[cur] if q in unvisited]
            if not cand:
                return path
            # prefer 4-connected continuation, then deterministic order
            cand.sort(key=lambda q: (abs(q[0] - cur[0]) + abs(q[1] - cur[1]), q))
            cur = cand[0]
            path.append(cur)
            unvisited.discard(cur)

    paths = []
    first = True
    while unvisited:
        start = next((t for t in termini if t in unvisited), None)
        is_cycle = start is None and not termini and first
        if start is None:
            start = min(unvisited)
        paths.append((walk(start), is_cycle))
        first = False
    # >1 path from one component = the walk could not cover it in one pass
    # (thick corner); the leftovers become extra branches, counted upstream
    return paths, max(0, len(paths) - 1)


def decompose_skeleton(sk: np.ndarray, junction_dilate: int) -> dict:
    """Split a skeleton into junction clusters and ordered branch paths.

    Real crossings often appear as adjacent CN>=3 pixels rather than one
    CN=4 pixel: the CN>=3 set is dilated (disk radius `junction_dilate`),
    connected-labeled, and each cluster becomes one junction node at the
    centroid of its original CN>=3 pixels. Branches are the remaining
    skeleton pixels, traced into ordered paths; each branch end records the
    junction cluster it touches (0 = free end).
    """
    H, W = sk.shape
    cn = crossing_number(sk)
    jmask = cn >= 3
    dil = ndi.binary_dilation(jmask, structure=disk(junction_dilate)) if junction_dilate > 0 else jmask
    jlab, njunc = ndi.label(dil, structure=ONES3)
    territory = np.where(sk & (jlab > 0), jlab, 0)      # skeleton px owned by junction nodes
    branch_mask = sk & (territory == 0)
    blab, nbr = ndi.label(branch_mask, structure=ONES3)

    centroids = {}
    if njunc:
        for lab, com in zip(range(1, njunc + 1),
                            ndi.center_of_mass(jmask.astype(float), jlab,
                                               list(range(1, njunc + 1)))):
            centroids[lab] = com if np.isfinite(com[0]) else \
                ndi.center_of_mass(dil.astype(float), jlab, lab)

    def adjacent_junctions(pix):
        labs = set()
        for dr, dc in OFFS8:
            r, c = pix[0] + dr, pix[1] + dc
            if 0 <= r < H and 0 <= c < W and territory[r, c] > 0:
                labs.add(int(territory[r, c]))
        return labs

    branches, incomplete = [], 0
    if nbr:
        comp_coords = [[] for _ in range(nbr + 1)]
        for r, c in zip(*np.nonzero(branch_mask)):
            comp_coords[blab[r, c]].append((int(r), int(c)))
        for lab in range(1, nbr + 1):
            paths, inc = trace_component(sorted(comp_coords[lab]))
            incomplete += inc
            for path, is_cycle in paths:
                if is_cycle:
                    branches.append({"path": path, "cycle": True, "ends": (0, 0)})
                    continue
                l0, l1 = adjacent_junctions(path[0]), adjacent_junctions(path[-1])
                e0 = min(l0) if l0 else 0
                e1 = 0
                if l1:
                    others = sorted(l1 - {e0})
                    e1 = others[0] if (others and len(path) == 1) else min(l1)
                branches.append({"path": path, "cycle": False, "ends": (e0, e1)})

    return {"cn": cn, "n_junctions": njunc, "centroids": centroids,
            "territory": territory, "branches": branches,
            "trace_incomplete": incomplete,
            "skeleton_px": int(sk.sum())}


def prune_spurs(sk: np.ndarray, p) -> tuple[np.ndarray, dict]:
    """Iteratively delete short spur branches (junction -> free endpoint).

    Spurious-feature filtering per Nguyen et al. 2013 Sec. 3 / Zhao & Tang
    2007: a branch with one end at a junction and the other terminating in
    a free endpoint, geodesic length < p.spur_len, is skeletonization noise.
    CN is recomputed after each pass (max p.spur_passes) so junctions that
    drop to degree 2 dissolve into through-paths.
    """
    sk = sk.copy()
    px0 = int(sk.sum())
    removed_n, removed_len, passes = 0, 0.0, 0
    for _ in range(p.spur_passes):
        dec = decompose_skeleton(sk, p.junction_dilate)
        spurs = [b for b in dec["branches"]
                 if not b["cycle"]
                 and (b["ends"][0] == 0) != (b["ends"][1] == 0)   # exactly one free end
                 and geodesic_len(b["path"]) < p.spur_len]
        if not spurs:
            break
        passes += 1
        for b in spurs:
            removed_len += geodesic_len(b["path"])
            for r, c in b["path"]:
                sk[r, c] = False
        removed_n += len(spurs)
    frac = (px0 - int(sk.sum())) / px0 if px0 else 0.0
    log = {"passes_run": passes, "branches_removed": removed_n,
           "length_removed_px": round(removed_len, 2),
           "removed_frac_of_skeleton_px": round(frac, 4),
           "warning_gt_max_frac": frac > p.spur_warn_frac}
    return sk, log


def merge_bridged_junctions(dec: dict, merge_px: float) -> dict:
    """Contract short junction-to-junction connector branches into one node.

    The mandatory junction closing fattens crossings into elongated blobs
    whose skeletons carry TWO CN=3 points separated by roughly the closing
    kernel length -- beyond the reach of the 3 px cluster dilation. The
    connector segment between them belongs to both crossing wires at once,
    so it cannot be assigned to either chain: clusters joined by a branch
    shorter than `merge_px` are united into one junction node (union-find)
    and the connector is dropped; the euclidean bridge added during chain
    merging replaces its length. Analog of spurious minutiae-pair (bridge)
    filtering, Zhao & Tang 2007. Set --junction-merge-px 0 to disable.
    """
    if merge_px <= 0:
        dec["junction_merges"] = 0
        dec["bridges_dropped"] = 0
        return dec

    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    drop = []
    for i, b in enumerate(dec["branches"]):
        l0, l1 = b["ends"]
        if b["cycle"] or l0 == 0 or l1 == 0:
            continue
        if geodesic_len(b["path"]) < merge_px:
            drop.append(i)                      # short self-loops at a node too
            if l0 != l1:
                parent[find(l0)] = find(l1)

    n_unions = sum(1 for x in parent if find(x) != x)
    if not drop:
        dec["junction_merges"] = 0
        dec["bridges_dropped"] = 0
        return dec

    members: dict = {}
    for lab in list(dec["centroids"]):
        members.setdefault(find(lab) if lab in parent else lab, []).append(lab)
    centroids = {root: (float(np.mean([dec["centroids"][m][0] for m in labs])),
                        float(np.mean([dec["centroids"][m][1] for m in labs])))
                 for root, labs in members.items()}

    branches = []
    dropset = set(drop)
    for i, b in enumerate(dec["branches"]):
        if i in dropset:
            continue
        l0, l1 = b["ends"]
        b = dict(b)
        b["ends"] = (find(l0) if l0 in parent else l0,
                     find(l1) if l1 in parent else l1)
        branches.append(b)

    dec = dict(dec)
    dec["branches"] = branches
    dec["centroids"] = centroids
    dec["junction_merges"] = n_unions
    dec["bridges_dropped"] = len(drop)
    return dec


# --------------------------------------------------------------------------
# graph build + junction stub pairing
# --------------------------------------------------------------------------

def stub_direction(path: list, end: int, k: int) -> np.ndarray | None:
    """Unit tangent of the endmost `k` path pixels, pointing away from the node.

    Least-squares line fit (PCA) of the stub pixels; the sign is aligned with
    the vector from the node-side end pixel into the branch.
    """
    pts = np.asarray(path[:k] if end == 0 else path[-k:][::-1], float)
    if len(pts) < 2:
        return None
    ref = pts[-1] - pts[0]
    n = np.linalg.norm(ref)
    if n == 0:
        return None
    pts_c = pts - pts.mean(0)
    _, v = np.linalg.eigh(pts_c.T @ pts_c)
    d = v[:, -1]
    if np.dot(d, ref) < 0:
        d = -d
    nd = np.linalg.norm(d)
    return d / nd if nd > 0 else ref / n


def pair_stubs_at_junction(dirs: list, p) -> tuple[list, list]:
    """Choose collinear stub pairs at one junction (Choi et al. 2011 via
    Nguyen et al. Fig. 5).

    delta = |180 - angle_between(dir_a, dir_b)| with directions pointing away
    from the node; collinear continuations score near 0. Candidates are pairs
    with delta <= p.pair_max_delta; for degree <= p.pair_enum_max_degree all
    disjoint pair subsets are enumerated (maximize accepted pairs, then
    minimize total delta), above that a greedy fallback is used.
    Returns (accepted [(i, j, delta)], all candidate deltas incl. rejected).
    """
    n = len(dirs)
    all_deltas, cands = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if dirs[i] is None or dirs[j] is None:
                continue
            ang = math.degrees(math.acos(np.clip(np.dot(dirs[i], dirs[j]), -1, 1)))
            delta = abs(180.0 - ang)
            all_deltas.append(delta)
            if delta <= p.pair_max_delta:
                cands.append((i, j, delta))

    if not cands:
        return [], all_deltas

    if n <= p.pair_enum_max_degree:
        best = {"np": -1, "tot": float("inf"), "sel": []}

        def rec(k, used, chosen, total):
            if k == len(cands):
                if (len(chosen), -total) > (best["np"], -best["tot"]):
                    best.update(np=len(chosen), tot=total, sel=list(chosen))
                return
            rec(k + 1, used, chosen, total)                     # skip candidate k
            i, j, d = cands[k]
            if i not in used and j not in used:
                used.update((i, j))
                chosen.append(cands[k])
                rec(k + 1, used, chosen, total + d)
                chosen.pop()
                used.difference_update((i, j))

        rec(0, set(), [], 0.0)
        accepted = best["sel"]
    else:                                                       # rare high-degree fallback
        accepted, used = [], set()
        for i, j, d in sorted(cands, key=lambda t: t[2]):
            if i not in used and j not in used:
                accepted.append((i, j, d))
                used.update((i, j))
    return accepted, all_deltas


def build_graph(dec: dict, edt: np.ndarray, p) -> tuple[nx.MultiGraph, list, list]:
    """networkx graph storage adapted from K-plet graphs (Chikkerur et al. 2005).

    Nodes = junction clusters ('J', label) and free endpoints ('E', idx);
    edges = ordered branch paths with geodesic length, PCA orientation, stub
    tangents, and local width samples (2 * EDT of the mask along the path,
    excluding the p.width_exclude_px pixels nearest each node).
    """
    # junction clusters with <2 incident branches are not real junctions:
    # degree-1 clusters are blunt-end skeleton forks whose tips are shorter
    # than the dilation radius (never emerged as prunable branches) -- their
    # single stub is a free endpoint, not an abutting end
    stub_count: dict = {}
    for b in dec["branches"]:
        if b["cycle"]:
            continue
        for end in (0, 1):
            if b["ends"][end] > 0:
                stub_count[b["ends"][end]] = stub_count.get(b["ends"][end], 0) + 1
    real_junctions = {lab for lab, n in stub_count.items() if n >= 2}
    info = {"phantom_degree1_clusters": sum(1 for n in stub_count.values() if n == 1),
            "clusters_no_branch": dec["n_junctions"] - len(stub_count),
            "real_junction_labels": real_junctions}

    G = nx.MultiGraph()
    for lab in sorted(real_junctions):
        G.add_node(("J", lab), pos=dec["centroids"][lab])
    edges, cycles, e_idx = [], [], 0
    n_end = 0
    for b in dec["branches"]:
        path = b["path"]
        pts = np.asarray(path)
        we = p.width_exclude_px
        core = path[we:len(path) - we] if len(path) > 2 * we else [path[len(path) // 2]]
        widths = [2.0 * float(edt[r, c]) for r, c in core]
        if b["cycle"]:
            cycles.append({"path": path, "length_px": geodesic_len(path),
                           "widths": widths})
            continue
        nodes = []
        for end in (0, 1):
            jl = b["ends"][end]
            if jl in real_junctions:
                nodes.append(("J", jl))
            else:
                nodes.append(("E", n_end))
                G.add_node(("E", n_end), pos=path[0 if end == 0 else -1])
                n_end += 1
        e = {"idx": e_idx, "path": path, "nodes": nodes,
             "length_px": geodesic_len(path),
             "orientation_deg": pca_orientation(pts[:, 0], pts[:, 1]),
             "stub_dirs": [stub_direction(path, end, p.stub_fit_px) for end in (0, 1)],
             "widths": widths}
        G.add_edge(nodes[0], nodes[1], key=e_idx, **{"length_px": e["length_px"]})
        edges.append(e)
        e_idx += 1
    return G, edges, cycles, info


def pair_all_junctions(G: nx.MultiGraph, edges: list, p) -> tuple[dict, dict]:
    """Run stub pairing at every junction; return continuation links + stats."""
    stubs_by_junction: dict = {}
    for e in edges:
        for end in (0, 1):
            node = e["nodes"][end]
            if node[0] == "J":
                stubs_by_junction.setdefault(node, []).append((e["idx"], end))

    links: dict = {}
    accepted_deltas, all_deltas = [], []
    n_stubs = n_paired = 0
    for jnode in sorted(stubs_by_junction):
        stubs = sorted(stubs_by_junction[jnode])
        dirs = [edges[eidx]["stub_dirs"][end] for eidx, end in stubs]
        acc, alld = pair_stubs_at_junction(dirs, p)
        all_deltas.extend(alld)
        n_stubs += len(stubs)
        for i, j, d in acc:
            links[stubs[i]] = stubs[j]
            links[stubs[j]] = stubs[i]
            accepted_deltas.append(d)
            n_paired += 2

    hist_edges = list(np.arange(0, 185, 5))
    hist = np.histogram(all_deltas, bins=hist_edges)[0] if all_deltas else np.zeros(36, int)
    degree_hist: dict = {}
    for j in stubs_by_junction:
        d = int(G.degree(j))
        degree_hist[d] = degree_hist.get(d, 0) + 1
    stats = {"count": len(stubs_by_junction),
             "degree_hist": dict(sorted(degree_hist.items())),
             "n_stubs": n_stubs, "n_paired_stubs": n_paired,
             "pairing_acceptance": round(n_paired / n_stubs, 4) if n_stubs else None,
             "accepted_delta_deg": _stats(accepted_deltas),
             "all_pair_delta_hist_5deg_bins": [int(v) for v in hist]}
    return links, stats


# --------------------------------------------------------------------------
# chains (merged wires) + measurement
# --------------------------------------------------------------------------

def build_chains(edges: list, links: dict) -> list:
    """Merge edges through accepted junction pairs into wire chains.

    Euclidean bridge distance across each traversed junction is added to the
    chain length so crossings do not systematically shorten wires.
    """
    visited = set()
    chains = []

    def walk(start_edge, start_end):
        pixels, widths, joins = [], [], []
        length, n_junc, n_edges = 0.0, 0, 0
        cur, ent = start_edge, start_end
        while True:
            visited.add(cur)
            n_edges += 1
            e = edges[cur]
            pth = e["path"] if ent == 0 else e["path"][::-1]
            if pixels:                                   # bridge across the junction
                length += math.hypot(pixels[-1][0] - pth[0][0],
                                     pixels[-1][1] - pth[0][1])
                joins.append(len(pixels))                # pixel index of the join
            pixels.extend(pth)
            length += e["length_px"]
            widths.extend(e["widths"])
            exit_end = 1 - ent
            nxt = links.get((cur, exit_end))
            if nxt is None:
                return {"pixels": pixels, "length_px": length, "widths": widths,
                        "joins": joins, "n_junctions": n_junc, "n_edges": n_edges,
                        "closed": False,
                        "end_refs": ((start_edge, start_end), (cur, exit_end))}
            n_junc += 1
            if nxt[0] in visited:                        # cycle through junctions closed
                return {"pixels": pixels, "length_px": length, "widths": widths,
                        "joins": joins, "n_junctions": n_junc, "n_edges": n_edges,
                        "closed": True, "end_refs": None}
            cur, ent = nxt

    termini = sorted((e["idx"], end) for e in edges for end in (0, 1)
                     if (e["idx"], end) not in links)
    for eidx, end in termini:
        if eidx not in visited:
            chains.append(walk(eidx, end))
    for e in edges:                                      # pure link-cycles remain
        if e["idx"] not in visited:
            chains.append(walk(e["idx"], 0))
    return chains


def rdp_indices(P: np.ndarray, tol: float) -> np.ndarray:
    """Ramer-Douglas-Peucker vertex indices of an ordered pixel path.

    Turning is measured at the simplified polyline's vertices so that pixel
    digitization noise scores zero on straight wires: a digital straight line
    collapses to its two endpoints, while a wrong-merge kink (families ~60
    deg apart) keeps a vertex with a ~60 deg turn.
    """
    n = len(P)
    if n < 3:
        return np.arange(n)
    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = P[b] - P[a]
        rel = P[a + 1:b] - P[a]
        L = math.hypot(seg[0], seg[1])
        if L == 0:
            d = np.hypot(rel[:, 0], rel[:, 1])
        else:
            d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / L
        i = int(np.argmax(d))
        if d[i] > tol:
            m = a + 1 + i
            keep.add(m)
            stack.extend([(a, m), (m, b)])
    return np.array(sorted(keep))


def polyline_turning(P: np.ndarray, tol: float, window: float) -> tuple[float, float]:
    """(total_abs_turning_deg, max turning summed inside a sliding `window` px
    arc) from the RDP-simplified path. Curvature QC per Nguyen et al. 2013
    eq. 8, adapted to polyline vertex turns."""
    V = P[rdp_indices(P, tol)]
    if len(V) < 3:
        return 0.0, 0.0
    segs = np.diff(V, axis=0)
    head = np.degrees(np.arctan2(segs[:, 0], segs[:, 1]))
    turns = np.abs((np.diff(head) + 180.0) % 360.0 - 180.0)   # at interior vertices
    svert = np.cumsum(np.hypot(segs[:, 0], segs[:, 1]))[:-1]  # arclength of each vertex
    max_local = 0.0
    for i in range(len(turns)):
        inside = np.abs(svert - svert[i]) <= window / 2.0
        max_local = max(max_local, float(turns[inside].sum()))
    return float(turns.sum()), max_local


def measure_chain(ch: dict, edges: list, nm_per_px: float, shape: tuple, p) -> dict:
    """Per-wire record: geometry, orientation family, turning QC (Nguyen et al.
    2013 eq. 8 adaptation), end types, flags. Flags only -- never deletes."""
    H, W = shape
    P = np.asarray(ch["pixels"], float)
    length_px = ch["length_px"]
    orient = pca_orientation(P[:, 0], P[:, 1])

    if np.isfinite(orient):
        devs = [dev_180(orient, c) for c in p.family_centers]
        fi = int(np.argmin(np.abs(devs)))
        family, fdev = p.family_centers[fi], float(devs[fi])
    else:
        fi, family, fdev = -1, float("nan"), float("nan")

    # turning-based curvature QC (TiS3 wires are near-straight). Pixels within
    # p.turn_junction_excl of each junction join are excised first: the
    # skeleton wobbles into the node there (an artifact), while a genuine
    # wrong merge still shows as a direction change ACROSS the junction gap
    Pt = P
    joins = ch.get("joins", [])
    if joins and p.turn_junction_excl > 0:
        keep = np.ones(len(P), bool)
        for j in joins:
            keep[max(0, j - p.turn_junction_excl):j + p.turn_junction_excl] = False
        if keep.sum() >= 3:
            Pt = P[keep]
    total_turn, max_local = polyline_turning(Pt, p.rdp_tol, p.curv_window)

    widths = np.asarray(ch["widths"], float)
    w_mean = float(widths.mean()) if widths.size else float("nan")
    w_med = float(np.median(widths)) if widths.size else float("nan")

    def end_type(ref):
        eidx, end = ref
        node = edges[eidx]["nodes"][end]
        if node[0] == "J":
            return "abutting"
        pix = edges[eidx]["path"][0 if end == 0 else -1]
        return "border" if min(pix[0], pix[1], H - 1 - pix[0], W - 1 - pix[1]) < p.border_px \
            else "free"

    if ch["closed"] or ch.get("end_refs") is None:
        e1t = e2t = "closed"
    else:
        e1t, e2t = end_type(ch["end_refs"][0]), end_type(ch["end_refs"][1])

    dmin = np.minimum(np.minimum(P[:, 0], P[:, 1]),
                      np.minimum(H - 1 - P[:, 0], W - 1 - P[:, 1]))
    border_touching = bool((dmin < p.border_px).any())

    flags = []
    # localized corner = wrong-merge signature. The raw total-turning term was
    # retired as a flag 2026-08-14: it scales with chain length, so long
    # straight wires through merged junctions exceeded it without bending
    # (curv_total_max stays in params for reference only)
    if max_local > p.curv_local_max:
        flags.append("high_curvature")          # candidate wrong merge
    if length_px > 0 and total_turn / length_px > p.turn_rate_max:
        flags.append("high_turn_rate")          # bent along its whole length
    if length_px < p.short_px:
        flags.append("short")
    if np.isfinite(w_mean) and not (p.width_min_px <= w_mean <= p.width_max_px):
        flags.append("width_outlier")

    off_family = bool(np.isfinite(fdev) and abs(fdev) > p.off_family_deg)
    return {"length_px": round(length_px, 2),
            "length_nm": round(length_px * nm_per_px, 1),
            "width_mean_px": round(w_mean, 2), "width_median_px": round(w_med, 2),
            "width_mean_nm": round(w_mean * nm_per_px, 1),
            "orientation_deg": round(orient, 2) if np.isfinite(orient) else float("nan"),
            "family": family, "family_dev_deg": round(fdev, 2) if np.isfinite(fdev) else float("nan"),
            "off_family": off_family,
            "total_abs_turning_deg": round(total_turn, 1),
            "max_local_turning_deg": round(max_local, 1),
            "n_junctions_traversed": ch["n_junctions"],
            "n_edges": ch.get("n_edges", 1),
            "end1_type": e1t, "end2_type": e2t,
            "border_touching": border_touching,
            "qc_flags": ";".join(flags),
            "_family_idx": fi, "_pixels": ch["pixels"], "_end_refs": ch.get("end_refs")}


# --------------------------------------------------------------------------
# per-mask orchestration
# --------------------------------------------------------------------------

def extract_from_mask(mask: np.ndarray, nm_per_px: float, p) -> tuple[list, dict, dict]:
    """Full extraction on one boolean mask. Returns (records, summary, render)."""
    H, W = mask.shape
    raw_px = int(mask.sum())

    # --- pre-skeleton cleanup: small-component removal, then junction closing
    lab, ncomp = ndi.label(mask, structure=ONES3)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = sizes >= p.min_area
    m1 = keep[lab]
    removed_comp = int(((sizes[1:] > 0) & (sizes[1:] < p.min_area)).sum())

    close_added = 0
    m2 = m1
    if p.junction_close:
        # v2 junction kernels (dir_close in ridge_nanowire_images2): 9 px line
        # elements at the measured family orientations, so skeletons connect
        # through crossing dips instead of fragmenting
        u8 = m1.astype(np.uint8)
        m2 = m1.copy()
        for ang in p.close_angles:
            m2 |= cv2.morphologyEx(u8, cv2.MORPH_CLOSE,
                                   line_kernel(p.close_len, ang)).astype(bool)
        close_added = int(m2.sum()) - int(m1.sum())

    edt = ndi.distance_transform_edt(m2)

    # --- skeletonize, prune spurs, decompose
    sk0 = skeletonize(m2)
    sk_px0 = int(sk0.sum())
    sk, prune_log = prune_spurs(sk0, p)
    dec = decompose_skeleton(sk, p.junction_dilate)
    dec = merge_bridged_junctions(dec, p.junction_merge_px)

    # --- graph, stub pairing, chains
    G, edges, cycles, ginfo = build_graph(dec, edt, p)
    links, jstats = pair_all_junctions(G, edges, p)
    jstats["clusters_total"] = dec["n_junctions"]   # incl. phantom/no-branch clusters
    jstats["junction_merges"] = dec["junction_merges"]
    jstats["bridges_dropped"] = dec["bridges_dropped"]
    jstats["phantom_degree1_clusters"] = ginfo["phantom_degree1_clusters"]
    jstats["clusters_no_branch"] = ginfo["clusters_no_branch"]
    chains = build_chains(edges, links)
    for cyc in cycles:                          # isolated skeleton rings
        chains.append({"pixels": cyc["path"], "length_px": cyc["length_px"],
                       "widths": cyc["widths"], "n_junctions": 0, "closed": True,
                       "end_refs": None, "n_edges": 1})

    # hard drop (not flag): below one wire width a chain cannot be a real
    # wire -- removes the degenerate point-chains that padded the count.
    # Distinct from the soft `short` QC flag (p.short_px) on survivors.
    dropped_lens = [ch["length_px"] for ch in chains
                    if ch["length_px"] < p.min_wire_px]
    chains = [ch for ch in chains if ch["length_px"] >= p.min_wire_px]

    records = []
    for wid, ch in enumerate(chains):
        rec = measure_chain(ch, edges, nm_per_px, (H, W), p)
        rec["wire_id"] = wid
        records.append(rec)

    # --- summary
    def conv(rs):
        L_px = [r["length_px"] for r in rs]
        return {"count": len(rs),
                "length_px": _stats(L_px),
                "length_nm": _stats([r["length_nm"] for r in rs]),
                "width_mean_px": _stats([r["width_mean_px"] for r in rs]),
                "width_mean_nm": _stats([r["width_mean_nm"] for r in rs]),
                "family_fractions": {str(c): round(sum(r["family"] == c for r in rs) / len(rs), 4)
                                     for c in p.family_centers} if rs else {},
                "off_family_count": sum(r["off_family"] for r in rs)}

    non_border = [r for r in records if not r["border_touching"]]
    hist_edges = np.arange(0, 180.1, 2)
    ohist = np.zeros(len(hist_edges) - 1)
    for r in records:
        if np.isfinite(r["orientation_deg"]):
            ohist[min(int(r["orientation_deg"] // 2), len(ohist) - 1)] += r["length_px"]
    sm = np.convolve(np.r_[ohist[-2:], ohist, ohist[:2]],
                     np.array([1, 2, 3, 2, 1]) / 9.0, mode="same")[2:-2]
    peaks = [float(hist_edges[i] + 1) for i in range(len(sm))
             if sm[i] >= 0.2 * sm.max() and sm[i] >= sm[i - 1] and sm[i] >= sm[(i + 1) % len(sm)]] \
        if sm.max() > 0 else []

    qc = {"high_curvature": sum("high_curvature" in r["qc_flags"] for r in records),
          "high_turn_rate": sum("high_turn_rate" in r["qc_flags"] for r in records),
          "short": sum("short" in r["qc_flags"] for r in records),
          "width_outlier": sum("width_outlier" in r["qc_flags"] for r in records),
          "clean": sum(r["qc_flags"] == "" for r in records)}

    summary = {
        "shape": [H, W], "nm_per_px": round(nm_per_px, 4),
        "cleanup": {"mask_px_raw": raw_px, "components_raw": int(ncomp),
                    "small_components_removed": removed_comp,
                    "small_px_removed": raw_px - int(m1.sum()),
                    "junction_close_applied": bool(p.junction_close),
                    "close_added_px": close_added, "mask_px_final": int(m2.sum())},
        "skeleton": {"px_before_prune": sk_px0, "px_after_prune": int(sk.sum()),
                     "trace_incomplete": dec["trace_incomplete"],
                     "isolated_loops": len(cycles)},
        "spur_pruning": prune_log,
        "chains_dropped_subwire": {"count": len(dropped_lens),
                                   "length_px": _stats(dropped_lens)},
        "wires_all": conv(records),
        "wires_excl_border": conv(non_border),
        "junctions": jstats,
        "qc_flag_counts": qc,
        "orientation_hist": {"bin_deg": 2, "length_px_weighted": [round(v, 1) for v in ohist]},
        "orientation_peaks_deg": peaks,
    }
    render = {"chains": records,
              "junction_centroids": [dec["centroids"][lab]
                                     for lab in sorted(ginfo["real_junction_labels"])],
              "edges": edges, "mask_clean": m2, "skeleton": sk}
    return records, summary, render


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

CSV_COLS = ["wire_id", "length_px", "length_nm", "width_mean_px", "width_median_px",
            "width_mean_nm", "orientation_deg", "family", "family_dev_deg",
            "off_family", "total_abs_turning_deg", "max_local_turning_deg",
            "n_junctions_traversed", "n_edges", "end1_type", "end2_type",
            "border_touching", "qc_flags"]

FAMILY_COLORS = {0: (60, 100, 255), 1: (60, 220, 60), 2: (255, 60, 60)}  # blue/green/red
OFF_FAMILY_COLOR = (255, 255, 255)


def instance_color(i: int) -> np.ndarray:
    """Deterministic distinct color per wire (golden-ratio hue cycling)."""
    r, g, b = colorsys.hsv_to_rgb((i * 0.6180339887) % 1.0, 0.85, 1.0)
    return np.array([int(r * 255), int(g * 255), int(b * 255)], np.uint8)


def _paint(base: np.ndarray, chains: list, colors: list) -> np.ndarray:
    layer = np.zeros_like(base)
    for rec, col in zip(chains, colors):
        pts = np.asarray(rec["_pixels"])
        layer[pts[:, 0], pts[:, 1]] = col
    layer = cv2.dilate(layer, np.ones((3, 3), np.uint8))
    out = base.copy()
    m = layer.any(-1)
    out[m] = layer[m]
    return out


def render_overlays(bg: np.ndarray, records: list, render: dict, edges: list,
                    out_dir: Path, name: str) -> None:
    base = np.stack([bg] * 3, axis=2)

    inst = _paint(base, records, [instance_color(r["wire_id"]) for r in records])
    for r, c in render["junction_centroids"]:
        cv2.circle(inst, (int(round(c)), int(round(r))), 5, (255, 255, 0), 1)
    for rec in records:
        if rec["_end_refs"] is None:
            continue
        for (eidx, end), et in zip(rec["_end_refs"], (rec["end1_type"], rec["end2_type"])):
            if et in ("free", "border"):
                pr, pc = edges[eidx]["path"][0 if end == 0 else -1]
                cv2.circle(inst, (pc, pr), 2, (255, 255, 255), -1)
    Image.fromarray(inst).save(out_dir / f"{name}_instances.png")

    fam = _paint(base, records,
                 [FAMILY_COLORS.get(r["_family_idx"], OFF_FAMILY_COLOR)
                  if not r["off_family"] else OFF_FAMILY_COLOR for r in records])
    Image.fromarray(fam).save(out_dir / f"{name}_families.png")


def write_frame_outputs(name: str, records: list, summary: dict, render: dict,
                        bg: np.ndarray, out_root: Path, params: dict) -> pd.DataFrame:
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in records], columns=CSV_COLS)
    df.to_csv(out_dir / "wires.csv", index=False)
    summary = dict(summary)
    summary["frame"] = name
    summary["params"] = params
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_jsonable)
    render_overlays(bg, records, render, render["edges"], out_dir, name)
    return df


# --------------------------------------------------------------------------
# GT validation (existing labels only -- no new manual annotation)
# --------------------------------------------------------------------------

def paint_family_map(records: list, mask: np.ndarray) -> np.ndarray:
    """Family index per mask pixel via nearest extracted chain pixel (-1 = none)."""
    lab = np.full(mask.shape, -1, np.int16)
    for rec in records:
        pts = np.asarray(rec["_pixels"])
        lab[pts[:, 0], pts[:, 1]] = rec["_family_idx"]
    if (lab >= 0).any():
        _, (ir, ic) = ndi.distance_transform_edt(lab < 0, return_indices=True)
        fam = lab[ir, ic]
    else:
        fam = lab
    return np.where(mask, fam, -1)


def load_gt_frame(fr: dict, out_root: Path, p, _clf_cache={}) -> dict:
    """One 80k GT frame at 50k-equivalent: background, scale, per-family GT
    masks + union, and the cached v3 pipeline mask (computed once if absent,
    identical code path to production)."""
    from gt_labels import FAMILIES, parse_labeled, load_raw
    from wire_classifier import PROD_THR, load_classifier, wire_probability

    name, s = fr["name"], fr["scale_to_50k"]
    raw = load_raw(fr["raw"])
    h, w = raw.shape
    size = (int(round(w * s)), int(round(h * s)))
    img50 = cv2.resize(raw, size, interpolation=cv2.INTER_AREA)
    bg = np.clip(img50, 0, 255).astype(np.uint8)
    px_m, _ = read_pixel_size(fr["raw"])
    nm_per_px = px_m * 1e9 / s
    if abs(nm_per_px / TARGET_NM - 1) > p.scale_tol:
        raise SystemExit(f"GT frame {name}: nm/px {nm_per_px:.3f} deviates "
                         f">{p.scale_tol:.0%} from {TARGET_NM} -- scale chain broken")

    parsed = parse_labeled(fr["labeled"])
    fam50 = {fam: cv2.resize(parsed["filled"][fam].astype(np.uint8), size,
                             interpolation=cv2.INTER_NEAREST).astype(bool)
             for fam in FAMILIES}
    gt_mask = np.zeros(bg.shape, bool)
    for fam in FAMILIES:
        gt_mask |= fam50[fam]

    gt_mask_dir = out_root / "gt_pipeline"
    gt_mask_dir.mkdir(parents=True, exist_ok=True)
    mpath = gt_mask_dir / f"{name}_50k_pseudo.png"
    if mpath.exists():
        v3_mask = np.array(Image.open(mpath)) > 127
    else:
        if _clf_cache.get("clf") is None:
            _clf_cache["clf"] = load_classifier()
            try:
                import os
                _clf_cache["clf"].n_jobs = max(1, len(os.sched_getaffinity(0)))
            except AttributeError:
                pass
        prob = wire_probability(_clf_cache["clf"], img50 / 255.0)
        np.save(gt_mask_dir / f"{name}_prob.npy", prob.astype(np.float32))
        v3_mask = prob > PROD_THR
        Image.fromarray((v3_mask * 255).astype(np.uint8), mode="L").save(mpath)

    return {"name": name, "bg": bg, "nm_per_px": nm_per_px, "parsed": parsed,
            "fam50": fam50, "gt_mask": gt_mask, "v3_mask": v3_mask}


def run_gt_validation(out_root: Path, p, params: dict) -> None:
    """Measurement-level comparison: extraction on GT wire masks vs on the v3
    pipeline masks for the same two 80k frames. Manuscript material; no gates."""
    from gt_labels import FAMILIES, gt_frames

    # GT hue convention -> family center (blue ~0/180, green ~60, red ~123.5)
    hue_center = {"blue": 0.0, "green": 60.0, "red": 123.5}
    hue_fidx = {fam: int(np.argmin([abs(dev_180(hue_center[fam], c))
                                    for c in p.family_centers]))
                for fam in FAMILIES}

    frames = gt_frames()
    if not frames:
        raise SystemExit("GT frames not found -- check data/real/OneDrive_1_7-17-2026")

    validation = {"frames": {}, "params": params}
    all_gt_len, all_v3_len = [], []

    for fr in frames:
        b = load_gt_frame(fr, out_root, p)
        name, bg, nm_per_px = b["name"], b["bg"], b["nm_per_px"]
        parsed, fam50 = b["parsed"], b["fam50"]
        gt_mask, v3_mask = b["gt_mask"], b["v3_mask"]

        out = {}
        recs = {}
        for tag, mask in (("GT", gt_mask), ("v3", v3_mask)):
            records, summary, render = extract_from_mask(mask, nm_per_px, p)
            write_frame_outputs(f"{name}_{tag}", records, summary, render, bg,
                                out_root, params)
            recs[tag] = records
            out[tag] = {"wire_count_all": summary["wires_all"]["count"],
                        "wire_count_excl_border": summary["wires_excl_border"]["count"],
                        "length_nm_all": summary["wires_all"]["length_nm"],
                        "length_nm_excl_border": summary["wires_excl_border"]["length_nm"],
                        "width_mean_nm": summary["wires_all"]["width_mean_nm"],
                        "family_fractions_all": summary["wires_all"]["family_fractions"],
                        "junction_count": summary["junctions"]["count"],
                        "pairing_acceptance": summary["junctions"]["pairing_acceptance"],
                        "orientation_peaks_deg": summary["orientation_peaks_deg"]}

        gt_len = [r["length_nm"] for r in recs["GT"]]
        v3_len = [r["length_nm"] for r in recs["v3"]]
        all_gt_len.extend(gt_len)
        all_v3_len.extend(v3_len)
        ks = ks_2samp(gt_len, v3_len) if gt_len and v3_len else None

        # pixel-level orientation-family agreement where both assign a family
        gt_count = sum(fam50[f].astype(np.uint8) for f in FAMILIES)
        gt_fam = np.full(bg.shape, -1, np.int16)
        for fam in FAMILIES:
            gt_fam[fam50[fam] & (gt_count == 1)] = hue_fidx[fam]
        v3_fam = paint_family_map(recs["v3"], v3_mask)
        both = (gt_fam >= 0) & (v3_fam >= 0)
        agree = float((gt_fam[both] == v3_fam[both]).mean()) if both.any() else None

        cnt_ratio_all = out["v3"]["wire_count_all"] / max(out["GT"]["wire_count_all"], 1)
        cnt_ratio_nb = out["v3"]["wire_count_excl_border"] / max(out["GT"]["wire_count_excl_border"], 1)
        validation["frames"][name] = {
            "GT": out["GT"], "v3": out["v3"],
            "wire_count_ratio_all": round(cnt_ratio_all, 3),
            "wire_count_ratio_excl_border": round(cnt_ratio_nb, 3),
            "ks_length": {"stat": round(float(ks.statistic), 4),
                          "pvalue": float(ks.pvalue)} if ks else None,
            "pixel_family_agreement": {"agreement": round(agree, 4) if agree is not None else None,
                                       "n_pixels": int(both.sum())},
            "gt_labeled_instance_counts": parsed["counts"],
        }

        # printable table
        g, v = out["GT"], out["v3"]
        print(f"\n=== GT measurement validation: {name} ===")
        print(f"{'metric':38s} {'GT':>10s} {'v3':>10s} {'v3/GT':>8s}")
        rows = [
            ("wire count (all)", g["wire_count_all"], v["wire_count_all"]),
            ("wire count (excl border)", g["wire_count_excl_border"], v["wire_count_excl_border"]),
            ("length median nm (all)", g["length_nm_all"].get("median"), v["length_nm_all"].get("median")),
            ("length mean nm (all)", g["length_nm_all"].get("mean"), v["length_nm_all"].get("mean")),
            ("length median nm (excl border)", g["length_nm_excl_border"].get("median"),
             v["length_nm_excl_border"].get("median")),
            ("width mean nm (median over wires)", g["width_mean_nm"].get("median"),
             v["width_mean_nm"].get("median")),
            ("junction count", g["junction_count"], v["junction_count"]),
        ]
        for label, gv, vv in rows:
            if gv in (None,) or vv in (None,):
                continue
            print(f"{label:38s} {gv:10.1f} {vv:10.1f} {vv / gv if gv else float('nan'):8.3f}")
        for c in p.family_centers:
            gf = g["family_fractions_all"].get(str(c), 0.0)
            vf = v["family_fractions_all"].get(str(c), 0.0)
            print(f"{'family frac ' + str(c):38s} {gf:10.3f} {vf:10.3f}")
        if ks:
            print(f"{'KS stat on lengths (all)':38s} {ks.statistic:10.4f}  p={ks.pvalue:.3g}")
        if agree is not None:
            print(f"{'pixel family agreement':38s} {agree:10.4f}  (n={int(both.sum())})")

    if all_gt_len and all_v3_len:
        ks_all = ks_2samp(all_gt_len, all_v3_len)
        validation["combined_ks_length"] = {"stat": round(float(ks_all.statistic), 4),
                                            "pvalue": float(ks_all.pvalue)}
        print(f"\ncombined KS on lengths (both frames): {ks_all.statistic:.4f} "
              f"p={ks_all.pvalue:.3g}")

    with open(out_root / "gt_validation.json", "w") as f:
        json.dump(validation, f, indent=2, default=_jsonable)
    print(f"wrote {rel(out_root)}/gt_validation.json")


def run_close_ab(out_root: Path, p, params: dict) -> None:
    """A/B the junction closing on the two 80k GT frames (v3 masks, cached).

    Purpose: separate closing-caused width inflation from mask-caused
    inflation. Junction recall proxy: fraction of GT-derived junction
    centroids (extraction on the GT wire mask, closing ON) with a detected
    junction within p.junction_match_px in each arm. Read-only diagnostic:
    changes NO defaults; the decision is made from the table.
    """
    import copy
    from gt_labels import gt_frames

    frames = gt_frames()
    if not frames:
        raise SystemExit("GT frames not found -- check data/real/OneDrive_1_7-17-2026")

    p_close = copy.copy(p)
    p_close.junction_close = True
    p_noclose = copy.copy(p)
    p_noclose.junction_close = False

    ab = {"params": params, "junction_match_px": p.junction_match_px,
          "gt_reference": "extraction on GT wire mask, junction closing ON",
          "frames": {}}
    pooled = {"close": {}, "no_close": {}, "gt": {}}

    for fr in frames:
        b = load_gt_frame(fr, out_root, p)
        name, nm = b["name"], b["nm_per_px"]

        _, gt_sum, gt_ren = extract_from_mask(b["gt_mask"], nm, p_close)
        ref_j = np.asarray(gt_ren["junction_centroids"], float).reshape(-1, 2)
        gt_w_med = gt_sum["wires_all"]["width_mean_px"].get("median")

        arms = {}
        for tag, parm in (("close", p_close), ("no_close", p_noclose)):
            _, summ, ren = extract_from_mask(b["v3_mask"], nm, parm)
            det = np.asarray(ren["junction_centroids"], float).reshape(-1, 2)
            if len(ref_j) and len(det):
                dmin = np.sqrt(((ref_j[:, None] - det[None]) ** 2).sum(-1)).min(1)
                jrecall = float((dmin <= p.junction_match_px).mean())
            else:
                jrecall = None
            c, j = summ["cleanup"], summ["junctions"]
            wa, wb = summ["wires_all"], summ["wires_excl_border"]
            pre_close = c["mask_px_raw"] - c["small_px_removed"]
            arms[tag] = {
                "close_added_px": c["close_added_px"],
                "close_added_frac_of_pre": round(c["close_added_px"] / pre_close, 4),
                "width_median_px": wa["width_mean_px"].get("median"),
                "width_mean_px": wa["width_mean_px"].get("mean"),
                "width_median_nm": wa["width_mean_nm"].get("median"),
                "width_mean_nm": wa["width_mean_nm"].get("mean"),
                "junction_count": j["count"],
                "clusters_no_branch": j["clusters_no_branch"],
                "phantom_degree1": j["phantom_degree1_clusters"],
                "wire_count_all": wa["count"], "wire_count_excl_border": wb["count"],
                "length_median_nm": wa["length_nm"].get("median"),
                "junction_recall_proxy": round(jrecall, 4) if jrecall is not None else None,
                "chains_dropped_subwire": summ["chains_dropped_subwire"]["count"],
                "qc_flag_counts": summ["qc_flag_counts"],
            }
        wcf = (gt_w_med / arms["close"]["width_median_px"]
               if gt_w_med and arms["close"]["width_median_px"] else None)
        ab["frames"][name] = {
            "gt_reference": {"junction_count": len(ref_j),
                             "width_median_px": gt_w_med,
                             "wire_count_all": gt_sum["wires_all"]["count"]},
            "arms": arms,
            "width_correction_factor_gt_over_close": round(wcf, 4) if wcf else None,
        }
        for tag in ("close", "no_close"):
            for k in ("width_median_px", "junction_recall_proxy"):
                pooled[tag].setdefault(k, []).append(arms[tag][k])
        pooled["gt"].setdefault("width_median_px", []).append(gt_w_med)

        print(f"\n=== Junction-closing A/B: {name} (v3 mask; GT ref: "
              f"{len(ref_j)} junctions, width med {gt_w_med:.2f} px) ===")
        print(f"{'metric':34s} {'close':>14s} {'no-close':>14s}")
        rows = [
            ("close_added_px (frac of pre)",
             lambda a: f"{a['close_added_px']} ({a['close_added_frac_of_pre']:.1%})"),
            ("width median px / nm",
             lambda a: f"{a['width_median_px']:.2f} / {a['width_median_nm']:.1f}"),
            ("width mean px / nm",
             lambda a: f"{a['width_mean_px']:.2f} / {a['width_mean_nm']:.1f}"),
            ("junction count", lambda a: str(a["junction_count"])),
            ("no-branch clusters", lambda a: str(a["clusters_no_branch"])),
            ("phantom-degree1 clusters", lambda a: str(a["phantom_degree1"])),
            ("wire count all / excl-border",
             lambda a: f"{a['wire_count_all']} / {a['wire_count_excl_border']}"),
            ("length median nm", lambda a: f"{a['length_median_nm']:.1f}"),
            ("junction recall proxy",
             lambda a: f"{a['junction_recall_proxy']:.3f}"
             if a["junction_recall_proxy"] is not None else "n/a"),
            ("sub-wire chains dropped", lambda a: str(a["chains_dropped_subwire"])),
        ]
        for label, fmt in rows:
            print(f"{label:34s} {fmt(arms['close']):>14s} {fmt(arms['no_close']):>14s}")

    wc = np.mean(pooled["close"]["width_median_px"])
    wn = np.mean(pooled["no_close"]["width_median_px"])
    rc = np.mean([v for v in pooled["close"]["junction_recall_proxy"] if v is not None])
    rn = np.mean([v for v in pooled["no_close"]["junction_recall_proxy"] if v is not None])
    wg = np.mean(pooled["gt"]["width_median_px"])
    width_drop = (wc - wn) / wc if wc else 0.0
    recall_delta = rc - rn
    ab["pooled"] = {"width_median_px_close": round(float(wc), 3),
                    "width_median_px_no_close": round(float(wn), 3),
                    "width_median_px_gt": round(float(wg), 3),
                    "width_drop_frac_without_closing": round(float(width_drop), 4),
                    "junction_recall_close": round(float(rc), 4),
                    "junction_recall_no_close": round(float(rn), 4),
                    "width_correction_factor_gt_over_close": round(float(wg / wc), 4)}

    print("\nInterpretation guide:")
    print("- If no-close cuts width median substantially AND junction recall holds,")
    print("  closing drives the inflation: reduce close_len (try 5 and 7 px) or drop")
    print("  closing, and re-run this A/B with those values.")
    print("- If width median barely moves without closing, inflation is from the v3")
    print("  texture mask capturing full dark bodies plus rim (not fixable")
    print("  unsupervised): carry a documented width correction factor")
    print("  (GT-derived median width / pipeline median width on the GT frames).")
    print(f"\npooled: width med close {wc:.2f} px vs no-close {wn:.2f} px "
          f"(drop {width_drop:.1%}); GT {wg:.2f} px; junction recall "
          f"{rc:.3f} -> {rn:.3f} (delta {recall_delta:+.3f})")
    if width_drop > 0.15 and recall_delta < 0.05:
        rec = ("closing is the width-inflation driver and recall survives without "
               "it: recommend re-running this A/B with --close-len 5 and 7 "
               "(or --no-junction-close) before the full run")
    elif width_drop < 0.05:
        rec = (f"width inflation is mask-driven, not closing-driven: recommend the "
               f"documented width correction factor {wg / wc:.3f} "
               f"(GT median / pipeline median) and keeping closing as is")
    else:
        rec = ("ambiguous: closing contributes some width but not all -- weigh the "
               "recall cost in the table before choosing close_len or a "
               "correction factor")
    ab["recommendation"] = rec
    print(f"recommendation: {rec}")

    with open(out_root / "close_ab.json", "w") as f:
        json.dump(ab, f, indent=2, default=_jsonable)
    print(f"wrote {rel(out_root)}/close_ab.json")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--masks-dir", type=Path, default=ROOT / "results" / "ridge_nanowire_images2")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "real" / "NanowireImages2")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "wire_extract")
    ap.add_argument("--frames", nargs="*", default=None,
                    help="frame name substrings to process (default: all selected)")
    ap.add_argument("--junction-close", action=argparse.BooleanOptionalAction, default=True,
                    help="pre-skeleton closing with the v2 junction line kernels")
    ap.add_argument("--all-bands", action="store_true",
                    help="also process frames outside the validated rescale band")
    ap.add_argument("--validate-gt", action="store_true",
                    help="run the measurement-level GT validation on the two 80k frames")
    ap.add_argument("--close-ab", action="store_true",
                    help="A/B the junction closing on the two 80k GT frames "
                         "(diagnostic; changes no defaults). Without --frames, "
                         "processes no production frames")
    # tunables (defaults per the Phase 1 spec)
    ap.add_argument("--min-area", type=int, default=40, help="pre-skeleton min component px")
    ap.add_argument("--close-len", type=int, default=9, help="junction-closing line length px")
    ap.add_argument("--close-angles", type=str, default="0,60,123",
                    help="junction-closing line angles deg (v2 kernels)")
    ap.add_argument("--junction-dilate", type=int, default=3, help="CN>=3 cluster dilation px")
    ap.add_argument("--junction-merge-px", type=float, default=6.0,
                    help="contract junction-to-junction connector branches shorter "
                         "than this (bridge filtering; 0 disables). Needed because "
                         "the junction closing stretches crossings beyond the "
                         "cluster dilation reach")
    ap.add_argument("--spur-len", type=float, default=6.0, help="max geodesic px of a spur")
    ap.add_argument("--spur-passes", type=int, default=3, help="max spur-pruning passes")
    ap.add_argument("--spur-warn-frac", type=float, default=0.25,
                    help="warn if pruning removes more than this skeleton fraction")
    ap.add_argument("--stub-fit-px", type=int, default=7, help="pixels in the stub tangent fit")
    ap.add_argument("--width-exclude-px", type=int, default=3,
                    help="path pixels nearest each node excluded from width sampling")
    ap.add_argument("--pair-max-delta", type=float, default=25.0,
                    help="max collinearity deviation deg for an accepted stub pair")
    ap.add_argument("--pair-enum-max-degree", type=int, default=6,
                    help="exhaustive pairing enumeration up to this junction degree")
    ap.add_argument("--family-centers", type=str, default="0,60,123.5",
                    help="orientation family centers deg")
    ap.add_argument("--off-family-deg", type=float, default=20.0,
                    help="family deviation above which a wire is off_family")
    ap.add_argument("--curv-total-max", type=float, default=40.0,
                    help="RETIRED as a flag 2026-08-14 (kept for reference): "
                         "total |turning| scales with length; see --turn-rate-max")
    ap.add_argument("--curv-local-max", type=float, default=25.0,
                    help="high_curvature flag: max local turning deg threshold")
    ap.add_argument("--turn-rate-max", type=float, default=0.6,
                    help="high_turn_rate flag: total turning per px threshold (deg/px)")
    ap.add_argument("--min-wire-px", type=float, default=5.0,
                    help="hard drop (not flag) for chains shorter than this: below "
                         "one wire width a chain cannot be a real wire")
    ap.add_argument("--junction-match-px", type=float, default=5.0,
                    help="close-ab junction recall proxy: GT-junction match radius px")
    ap.add_argument("--curv-window", type=int, default=25, help="local turning window px")
    ap.add_argument("--rdp-tol", type=float, default=1.5,
                    help="RDP simplification tolerance px for turning measurement")
    ap.add_argument("--turn-junction-excl", type=int, default=5,
                    help="path pixels excised on each side of a junction join "
                         "before turning measurement (skeleton wobble at nodes)")
    ap.add_argument("--short-px", type=float, default=15.0, help="short flag threshold px")
    ap.add_argument("--width-min-px", type=float, default=3.0, help="width_outlier lower bound px")
    ap.add_argument("--width-max-px", type=float, default=30.0, help="width_outlier upper bound px")
    ap.add_argument("--border-px", type=int, default=2, help="border proximity px")
    ap.add_argument("--scale-tol", type=float, default=0.05,
                    help="allowed relative deviation of nm/px from 5.4 (in-band)")
    args = ap.parse_args(argv)
    args.masks_dir, args.raw_dir, args.out = (args.masks_dir.resolve(), args.raw_dir.resolve(),
                                              args.out.resolve())
    args.close_angles = [float(a) for a in args.close_angles.split(",")]
    args.family_centers = [float(a) for a in args.family_centers.split(",")]
    return args


def main() -> None:
    p = parse_args()
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(p).items()}
    params["curv_total_max_note"] = ("retired as a QC flag 2026-08-14: "
                                     "high_curvature = max_local only; the "
                                     "length-scaling total term moved to "
                                     "high_turn_rate (deg/px)")
    out_root = p.out
    out_root.mkdir(parents=True, exist_ok=True)

    cov_path = p.masks_dir / "ridge_coverage.json"
    if not cov_path.exists():
        raise SystemExit(f"missing {cov_path} -- run ridge_nanowire_images2.py first")
    cov = json.load(open(cov_path))
    per_frame = cov["per_frame"]

    names = sorted(per_frame)
    if p.frames:
        names = [n for n in names if any(f in n for f in p.frames)]
    elif p.close_ab or p.validate_gt:
        names = []          # GT-only diagnostic runs process no production frames
    skipped = []
    if not p.all_bands:
        skipped = [n for n in names if not per_frame[n]["scale_in_validated_band"]]
        names = [n for n in names if per_frame[n]["scale_in_validated_band"]]
    for n in skipped:
        print(f"skip {n}: outside validated rescale band (use --all-bands to include)")
    if not names and not (p.validate_gt or p.close_ab):
        raise SystemExit("no frames selected")

    agg = {"pipeline": "wire_extract Phase 1 (skeleton->graph->measurements)",
           "masks": str(rel(cov_path)),
           "partial_run": bool(p.frames), "skipped_out_of_band": skipped,
           "params": params, "per_frame": {}}
    pooled_all, pooled_nb = [], []

    for i, name in enumerate(names):
        meta = per_frame[name]
        nm_per_px = meta["pixel_size_m"] * 1e9 / meta["rescale_to_50k"]
        if meta["scale_in_validated_band"] and abs(nm_per_px / TARGET_NM - 1) > p.scale_tol:
            raise SystemExit(
                f"{name}: nm/px {nm_per_px:.3f} deviates >{p.scale_tol:.0%} from "
                f"{TARGET_NM} nm -- ridge_coverage.json scale chain is inconsistent, "
                f"regenerate masks before extracting")

        mpath = p.masks_dir / f"{name}_50k_pseudo.png"
        if not mpath.exists():
            raise SystemExit(f"missing mask {mpath}")
        mask = np.array(Image.open(mpath)) > 127
        if list(mask.shape) != list(meta["shape_50k"]):
            raise SystemExit(f"{name}: mask shape {mask.shape} != shape_50k "
                             f"{meta['shape_50k']} in ridge_coverage.json")

        bg_f, _ = load_frame_50k(p.raw_dir / f"{name}.tif")
        if bg_f.shape != mask.shape:
            raise SystemExit(f"{name}: raw frame reload shape {bg_f.shape} != mask "
                             f"{mask.shape} -- scale provenance mismatch")
        bg = np.clip(bg_f, 0, 255).astype(np.uint8)

        records, summary, render = extract_from_mask(mask, nm_per_px, p)
        summary["scale"] = {"nm_per_px": round(nm_per_px, 4),
                            "rescale_to_50k": meta["rescale_to_50k"],
                            "pixel_size_m": meta["pixel_size_m"],
                            "scale_in_validated_band": meta["scale_in_validated_band"]}
        write_frame_outputs(name, records, summary, render, bg, out_root, params)

        wa, wb = summary["wires_all"], summary["wires_excl_border"]
        agg["per_frame"][name] = {
            "wire_count_all": wa["count"], "wire_count_excl_border": wb["count"],
            "length_nm_median_all": wa["length_nm"].get("median"),
            "width_mean_nm_median": wa["width_mean_nm"].get("median"),
            "family_fractions_all": wa["family_fractions"],
            "junction_count": summary["junctions"]["count"],
            "pairing_acceptance": summary["junctions"]["pairing_acceptance"],
            "qc_flag_counts": summary["qc_flag_counts"],
            "spur_warning": summary["spur_pruning"]["warning_gt_max_frac"],
            "orientation_peaks_deg": summary["orientation_peaks_deg"],
        }
        pooled_all.extend(r["length_nm"] for r in records)
        pooled_nb.extend(r["length_nm"] for r in records if not r["border_touching"])
        spur_warn = " [SPUR WARNING >25%]" if summary["spur_pruning"]["warning_gt_max_frac"] else ""
        print(f"[{i + 1:2d}/{len(names)}] {name:26s} wires {wa['count']:4d} "
              f"(excl border {wb['count']:4d}) junctions "
              f"{summary['junctions']['count']:4d} accept "
              f"{summary['junctions']['pairing_acceptance']} "
              f"len med {wa['length_nm'].get('median', float('nan')):7.1f} nm "
              f"peaks {summary['orientation_peaks_deg']}{spur_warn}", flush=True)

    if names:
        agg["pooled_length_nm_all"] = _stats(pooled_all)
        agg["pooled_length_nm_excl_border"] = _stats(pooled_nb)
        with open(out_root / "extract_summary.json", "w") as f:
            json.dump(agg, f, indent=2, default=_jsonable)
        print(f"wrote {rel(out_root)}/extract_summary.json "
              f"({len(names)} frames)")

    if p.validate_gt:
        run_gt_validation(out_root, p, params)
    if p.close_ab:
        run_close_ab(out_root, p, params)


if __name__ == "__main__":
    main()
