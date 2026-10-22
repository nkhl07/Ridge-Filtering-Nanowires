"""Score wire masks against the 2 hand-labelled GT frames -- EVALUATION ONLY.

Why this exists: the pipeline must stay UNSUPERVISED (decided 2026-10-13).
The two labelled 80k frames (data/real/OneDrive_1_7-17-2026, parsed by
gt_labels.py) may only SCORE a mask; nothing in the label-free pipeline
(labelfree.py) may be chosen by looking at these numbers. This script is the
single place GT pixels are read for evaluation, so every method -- the
label-free candidates, the v2 classical comparison pipeline and the v1
harness self-check -- is scored by one harness with one convention:

  * GT frames are loaded EXACTLY like the removed tune_pseudo_mask.main did
    (load_raw -> to_50k INTER_AREA; parse_labeled union -> to_50k
    INTER_NEAREST) and scored with its score(); both functions are copied
    verbatim below, so numbers stay comparable with
    results/real_stats/pseudo_mask_params.json.
  * Junction zone (README_v3 sec. 6.2 -- README_v3.md lives in the CycleGAN
    project ~/Unsupervised-Nanowire-Segmentation-using-CycleGAN-Architecture-2/,
    not in this repo; every README_v3 citation below means that file):
    pixels where >= 2 GT orientation families lie within 4 px (each
    family's filled 50k mask dilated by a disk of radius 4, overlap count
    >= 2), that seed dilated by a disk of radius 10, intersected with the
    GT union. junction_recall =
    |pred & zone| / |zone|. zone_fraction = |zone| / |GT| is printed as a
    convention sanity check (README_v3 quotes ~0.19-0.20).
  * Every method has a continuous score (the shared sato response), so each
    also gets threshold-free average precision and an ORACLE coverage-matched
    operating point (threshold = quantile so predicted coverage == GT
    coverage). The oracle USES LABELS and is a diagnostic only -- never an
    operating point for the pipeline.
  * Aggregation: per frame, the mean over frames (how tune_pseudo_mask
    aggregated, kept for comparability) and pooled-pixel counts.

Methods: lf_otsu is the PRE-DECLARED label-free default (fixed before any GT
evaluation); the other lf_* thresholds and two ablations are reported next to
it, never used to pick it. v2_fixed is the label-tuned v2 classical pipeline
(sato > 0.14 + dir_close at 0/60/123), kept as a comparison.
v1_sato_thr0.10 is NOT a pipeline any more (v1 and the supervised v3 RF were
removed 2026-10-13, results archived in ~/ridge-filtering-archive/): it is
the harness self-check (removed pipeline).

Harness self-check: v1 (sato response > 0.10, no closing) must reproduce the
frame-mean recall/precision/F1 of the (sato, s1-3, thr 0.10) entry in
pseudo_mask_params.json within 2e-3. FAIL is printed loudly (after the
tables), recorded in the JSON and turns the exit status to 3 (after the JSON
is written). A "zone_convention" sub-status also compares v1's frame-mean
junction recall with README_v3 sec. 6.2's 0.574 (same row as the matched
0.517/0.484/0.500), so a wrong junction-zone convention shows up as WARN. It
does not change PASS/FAIL, and it tunes nothing in the label-free pipeline.
A contract check also confirms labelfree.sato_response equals the response r
returned by ridge_nanowire_images2.pseudo_mask (max |diff| <= 1e-6); a
mismatch is printed loudly and also turns the exit status to 3.

Run via slurm/eval_masks.sbatch (reads data/real; not on the login node).
`--selftest` runs every code path on tiny synthetic frames (no GT files) and
writes to a temp dir.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from skimage.morphology import disk
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labelfree  # noqa: E402
import ridge_nanowire_images2 as v2  # noqa: E402
from gt_labels import FAMILIES, gt_frames, load_raw, parse_labeled  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT = ROOT / "results" / "eval" / "eval_masks.json"
V1_REF_PATH = ROOT / "results" / "real_stats" / "pseudo_mask_params.json"

FAMILY_RADIUS = 4       # px @50k: families "within 4 px" of each other
ZONE_RADIUS = 10        # px @50k: junction seed dilation
MIN_FAMILIES = 2        # families that must meet to seed a junction zone
V1_THR = 0.10           # v1 sato-only threshold (label-tuned, baseline only)
HARNESS_TOL = 2e-3      # |frame-mean metric - pseudo_mask_params.json| allowed
V1_JREC_LITERAL = 0.574     # README_v3 sec. 6.2, v1 sato thr 0.10 junction recall
JREC_TOL = 5e-3             # |v1 jRec - literal| above this -> zone_convention WARN
PREDECLARED = "lf_otsu"     # pre-declared label-free default; printed first
V2_LITERALS = {"recall": 0.570, "precision": 0.505, "f1": 0.534,
               "junction_recall": 0.672}    # label-tuned, provenance unknown
V1_LABEL = "harness self-check (removed pipeline)"
CONTRACT_TOL = 1e-6     # labelfree.sato_response vs v2.pseudo_mask's r
ORACLE_NOTE = "oracle: uses labels, diagnostic only"


# --------------------------------------------------------------------------- v1-era scorer
# to_50k() and score() are copied VERBATIM from the removed tune_pseudo_mask.py
# (the v1-era scorer that produced pseudo_mask_params.json). Do not edit them:
# the harness self-check is only meaningful while they stay byte-for-byte equal.

def to_50k(img: np.ndarray, s: float, nearest: bool = False) -> np.ndarray:
    """Rescale an 80k-native array to 50k-equivalent pixel scale (s = 0.625)."""
    h, w = img.shape[:2]
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=interp)


def score(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = float((pred & gt).sum())
    rec = tp / max(gt.sum(), 1)
    prec = tp / max(pred.sum(), 1)
    f1 = 2 * rec * prec / max(rec + prec, 1e-9)
    return {"recall": rec, "precision": prec, "f1": f1,
            "pred_cov": float(pred.mean()), "gt_cov": float(gt.mean()),
            "coverage_ratio": float(gt.mean() / max(pred.mean(), 1e-9))}


# --------------------------------------------------------------------------- frames

@dataclass
class Frame:
    name: str
    scale_to_50k: float
    raw50: np.ndarray                   # float32 [0,255] at 50k-equivalent scale
    gt50: np.ndarray                    # bool GT union at 50k
    fam50: dict[str, np.ndarray]        # bool filled mask per family at 50k
    zone: np.ndarray = field(init=False)    # bool junction zone, subset of gt50

    def __post_init__(self) -> None:
        self.zone = junction_zone(self.fam50, self.gt50)

    @property
    def zone_fraction(self) -> float:
        return float(self.zone.sum() / max(self.gt50.sum(), 1))


def junction_zone(fam50: dict[str, np.ndarray], gt50: np.ndarray) -> np.ndarray:
    """README_v3 sec. 6.2: >= 2 families within 4 px, dilated 10 px, within GT."""
    k_fam = disk(FAMILY_RADIUS).astype(np.uint8)
    count = np.zeros(gt50.shape, np.uint8)
    for m in fam50.values():
        count += cv2.dilate(m.astype(np.uint8), k_fam) > 0
    seed = (count >= MIN_FAMILIES).astype(np.uint8)
    return (cv2.dilate(seed, disk(ZONE_RADIUS).astype(np.uint8)) > 0) & gt50


def build_frame(name: str, s: float, raw: np.ndarray, gt: dict) -> Frame:
    """80k-native raw + parse_labeled dict -> 50k Frame (tune_pseudo_mask convention)."""
    raw50 = to_50k(raw, s)
    gt50 = to_50k(gt["union"].astype(np.uint8), s, nearest=True).astype(bool)
    fam50 = {f: to_50k(gt["filled"][f].astype(np.uint8), s, nearest=True).astype(bool)
             for f in FAMILIES}
    return Frame(name, s, raw50, gt50, fam50)


def load_gt_frames() -> list[Frame]:
    out = []
    for fr in gt_frames():
        out.append(build_frame(fr["name"], fr["scale_to_50k"], load_raw(fr["raw"]),
                               parse_labeled(fr["labeled"])))
    if not out:
        raise SystemExit("no GT frames found (gt_labels.GT_DIR)")
    return out


# --------------------------------------------------------------------------- scoring

def _jrec(pred: np.ndarray, zone: np.ndarray) -> float:
    return float((pred & zone).sum() / max(zone.sum(), 1))


def oracle_point(cont: np.ndarray, fr: Frame) -> dict:
    """Coverage-matched threshold (USES LABELS: diagnostic only)."""
    thr = float(np.quantile(cont, 1.0 - fr.gt50.mean()))
    pred = cont > thr
    sc = score(pred, fr.gt50)
    return {"oracle_thr": thr, "oracle_recall": sc["recall"],
            "oracle_precision": sc["precision"], "oracle_f1": sc["f1"],
            "oracle_pred_cov": sc["pred_cov"],
            "oracle_junction_recall": _jrec(pred, fr.zone)}


def evaluate(fr: Frame, mask: np.ndarray, cont: np.ndarray | None) -> dict:
    """Metrics for one mask (+ optional continuous score) on one frame."""
    if mask.shape != fr.gt50.shape:
        raise ValueError(f"{fr.name}: mask {mask.shape} != GT {fr.gt50.shape}")
    mask = mask.astype(bool)
    m = dict(score(mask, fr.gt50))
    m["junction_recall"] = _jrec(mask, fr.zone)
    if cont is not None:
        m["ap"] = float(average_precision_score(fr.gt50.ravel(), cont.ravel()))
        m.update(oracle_point(cont, fr))
    counts = {"tp": int((mask & fr.gt50).sum()), "n_pred": int(mask.sum()),
              "n_gt": int(fr.gt50.sum()), "n_zone": int(fr.zone.sum()),
              "zone_tp": int((mask & fr.zone).sum()), "n_px": int(mask.size)}
    return {"metrics": m, "counts": counts}


def aggregate(per_frame: dict[str, dict]) -> tuple[dict, dict]:
    """(mean over frames of every metric, pooled-pixel recall/precision/F1/jRec)."""
    rows = [r["metrics"] for r in per_frame.values()]
    keys = [k for k in rows[0] if all(k in r for r in rows)]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    count_keys = next(iter(per_frame.values()))["counts"]
    c = {k: sum(r["counts"][k] for r in per_frame.values()) for k in count_keys}
    rec = c["tp"] / max(c["n_gt"], 1)
    prec = c["tp"] / max(c["n_pred"], 1)
    pooled = {"recall": rec, "precision": prec,
              "f1": 2 * rec * prec / max(rec + prec, 1e-9),
              "junction_recall": c["zone_tp"] / max(c["n_zone"], 1),
              "pred_cov": c["n_pred"] / max(c["n_px"], 1),
              "gt_cov": c["n_gt"] / max(c["n_px"], 1), "counts": c}
    return mean, pooled


# --------------------------------------------------------------------------- methods
# A method maps one Frame -> (mask, continuous score | None, info).

Output = tuple[np.ndarray, "np.ndarray | None", dict]


def _v1(fr: Frame) -> Output:
    """Harness self-check: v1 (removed pipeline) = sato response > 0.10, no closing."""
    resp = labelfree.sato_response(fr.raw50)
    # contract check: labelfree's response must equal the r v2.pseudo_mask returns
    _, ref = v2.pseudo_mask(fr.raw50)
    info = {"thr": V1_THR, "close": False,
            "max_abs_diff_vs_v2_pseudo_mask_r": float(np.abs(resp - ref).max())}
    return resp > V1_THR, resp, info


def _v2(fr: Frame) -> Output:
    mask, r = v2.pseudo_mask(fr.raw50)
    return mask, r, {"thr": v2.THR, "angles_deg": list(v2.CLOSE_ANGLES),
                     "close_len": v2.CLOSE_LEN}


def _lf(method: str, **kw) -> Callable[[Frame], Output]:
    def run(fr: Frame) -> Output:
        mask, resp, info = labelfree.labelfree_mask(fr.raw50, method=method, **kw)
        return mask, resp, info
    return run


METHODS: dict[str, dict] = {}


def _register(name: str, fn: Callable[[Frame], Output], kind: str, label: str) -> None:
    METHODS[name] = {"fn": fn, "kind": kind, "label": label}


_register(PREDECLARED, _lf("otsu"), "candidate",
          "PRE-DECLARED label-free default: sato + otsu + dirclose at image angles")
for _m in labelfree.LF_METHODS:
    if f"lf_{_m}" not in METHODS:
        _register(f"lf_{_m}", _lf(_m), "candidate",
                  f"label-free sato + {_m} + dirclose at image angles")
_register("lf_otsu_fixedangles", _lf("otsu", angles=tuple(v2.CLOSE_ANGLES)),
          "ablation", "lf_otsu with v2's label-era angles (isolates angle estimation)")
_register("lf_otsu_noclose", _lf("otsu", close=False), "ablation",
          "lf_otsu without directional closing")
_register("v2_fixed", _v2, "comparison_label_tuned",
          "v2 classical pseudo_mask: sato > 0.14 + dirclose 0/60/123 (label-tuned)")
_register("v1_sato_thr0.10", _v1, "harness_self_check",
          f"{V1_LABEL}: v1 sato s1-3 > 0.10, no closing")


# --------------------------------------------------------------------------- checks

def harness_check(mean: dict | None, ref_path: Path = V1_REF_PATH) -> dict:
    """Harness self-check: v1 frame means vs the tune_pseudo_mask entry (sato, s1-3, 0.10)."""
    if mean is None:
        return {"status": "NOT_RUN"}
    try:
        ref_all = json.loads(Path(ref_path).read_text())["all"]
        ref = next(r for r in ref_all if r["filter"] == "sato" and r["sigmas"] == "s1-3"
                   and abs(r["thr"] - V1_THR) < 1e-9)
    except (OSError, KeyError, StopIteration, json.JSONDecodeError) as e:
        return {"status": "FAIL", "reason": f"reference unreadable: {e!r}",
                "reference": str(ref_path)}
    diffs = {k: mean[k] - ref[k] for k in ("recall", "precision", "f1")}
    ok = all(abs(d) <= HARNESS_TOL for d in diffs.values())
    jd = mean["junction_recall"] - V1_JREC_LITERAL
    zone = {"status": "OK" if abs(jd) <= JREC_TOL else "WARN", "tol": JREC_TOL,
            "literal": V1_JREC_LITERAL, "source": "README_v3 sec. 6.2 (v1 row)",
            "harness_value": mean["junction_recall"], "diff": jd}
    return {"status": "PASS" if ok else "FAIL", "tol": HARNESS_TOL,
            "reference": str(ref_path),
            "reference_values": {k: ref[k] for k in diffs},
            "harness_values": {k: mean[k] for k in diffs}, "diffs": diffs,
            "zone_convention": zone}


def contract_check(v1_result: dict | None) -> dict:
    """labelfree.sato_response must equal the r returned by v2.pseudo_mask."""
    if v1_result is None:
        return {"status": "NOT_RUN"}
    d = max(pf["info"]["max_abs_diff_vs_v2_pseudo_mask_r"]
            for pf in v1_result["per_frame"].values())
    return {"status": "PASS" if d <= CONTRACT_TOL else "FAIL", "max_abs_diff": d,
            "tol": CONTRACT_TOL,
            "compared": "labelfree.sato_response vs ridge_nanowire_images2.pseudo_mask r"}


# --------------------------------------------------------------------------- report

def _git() -> dict:
    def g(*a: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    status = g("status", "--porcelain")
    return {"commit": g("rev-parse", "--short", "HEAD") or None,
            "dirty": bool(status), "dirty_files": status.splitlines()[:50]}


def _provenance() -> dict:
    import scipy
    import skimage
    import sklearn
    return {"git": _git(), "timestamp": datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "skimage": skimage.__version__, "cv2": cv2.__version__,
            "labelfree": {"file": str(Path(labelfree.__file__).resolve()),
                          "DEFAULT_METHOD": labelfree.DEFAULT_METHOD,
                          "LF_METHODS": list(labelfree.LF_METHODS)}}


def _f(x: float | None, w: int = 6) -> str:
    return f"{x:{w}.3f}" if x is not None else f"{'-':>{w}s}"


def print_table(results: dict) -> None:
    hdr = (f"{'method':24s} {'recall':>6s} {'prec':>6s} {'F1':>6s} {'jRec':>6s} "
           f"{'covR':>6s} {'AP':>6s} {'orF1':>6s} {'orJRc':>6s}")

    def line(tag: str, m: dict) -> str:
        return (f"{tag:24s} {_f(m['recall'])} {_f(m['precision'])} {_f(m['f1'])} "
                f"{_f(m['junction_recall'])} {_f(m['coverage_ratio'])} {_f(m.get('ap'))} "
                f"{_f(m.get('oracle_f1'))} {_f(m.get('oracle_junction_recall'))}")

    print(f"\nMEAN OVER FRAMES   (covR = GT cov / pred cov; or* = {ORACLE_NOTE})")
    print(hdr)
    for name, r in results.items():
        mark = "*" if name == PREDECLARED else " "
        print(line(f"{mark}{name}", r["mean_over_frames"]))
    print(f"  * = pre-declared label-free default ({PREDECLARED}); "
          f"v1_sato_thr0.10 = {V1_LABEL}, not a candidate")
    print("  AP/or* score the CONTINUOUS output (the shared sato response for every "
          "method), not the thresholded/closed mask")
    print("\nPER FRAME")
    print(hdr)
    for name, r in results.items():
        for fname, pf in r["per_frame"].items():
            print(line(f" {name}@{fname[-1:] if fname[-1:].isdigit() else fname}",
                       pf["metrics"]))
    print("\nPOOLED PIXELS")
    print(f"{'method':24s} {'recall':>6s} {'prec':>6s} {'F1':>6s} {'jRec':>6s}")
    for name, r in results.items():
        p = r["pooled"]
        print(f" {name:23s} {_f(p['recall'])} {_f(p['precision'])} {_f(p['f1'])} "
              f"{_f(p['junction_recall'])}")


def run(frames: list[Frame], names: list[str], out_path: Path,
        ref_path: Path = V1_REF_PATH) -> dict:
    """Run + score the selected methods, print the table, write the JSON."""
    for fr in frames:
        print(f"{fr.name}: 50k {fr.raw50.shape}  GT cov {fr.gt50.mean() * 100:.2f}%  "
              f"zone_fraction {fr.zone_fraction:.3f} (README_v3: ~0.19-0.20)", flush=True)
    if labelfree.DEFAULT_METHOD != "otsu":
        print(f"WARNING: labelfree.DEFAULT_METHOD={labelfree.DEFAULT_METHOD!r} but the "
              f"pre-declared default is otsu ({PREDECLARED})", flush=True)

    results: dict[str, dict] = {}
    for name in names:
        spec = METHODS[name]
        t0 = time.time()
        per_frame = {}
        for fr in frames:
            mask, cont, info = spec["fn"](fr)
            ev = evaluate(fr, mask, cont)
            ev["info"] = info
            per_frame[fr.name] = ev
        mean, pooled = aggregate(per_frame)
        results[name] = {"kind": spec["kind"], "label": spec["label"],
                         "per_frame": per_frame, "mean_over_frames": mean,
                         "pooled": pooled, "seconds": round(time.time() - t0, 1)}
        print(f"[{name}] F1 {mean['f1']:.3f}  jRec {mean['junction_recall']:.3f}  "
              f"({results[name]['seconds']:.0f}s)", flush=True)
        if name.startswith("lf_"):
            for fn_, pf in per_frame.items():
                i = pf["info"]
                print(f"    {fn_}: thr {i['thr']:.4f}  angles "
                      f"{[round(a, 1) for a in i['angles_deg']]} ({i['angles_source']})")
        if name == "v1_sato_thr0.10":
            d = max(pf["info"]["max_abs_diff_vs_v2_pseudo_mask_r"]
                    for pf in per_frame.values())
            print(f"    labelfree.sato_response vs ridge_nanowire_images2.pseudo_mask r: "
                  f"max |diff| {d:.2e}"
                  f"{'' if d <= CONTRACT_TOL else '  <-- CONTRACT MISMATCH'}")

    print_table(results)

    v1 = results.get("v1_sato_thr0.10")
    check = harness_check(v1["mean_over_frames"] if v1 else None, ref_path)
    contract = contract_check(v1)
    bar = "=" * 72
    if check["status"] == "PASS":
        print(f"\nHARNESS CHECK PASS [v1 = {V1_LABEL}]: matches pseudo_mask_params.json "
              f"within {HARNESS_TOL} "
              f"({ {k: round(d, 5) for k, d in check['diffs'].items()} })")
    elif check["status"] == "FAIL":
        print(f"\n{bar}\nHARNESS CHECK FAIL [v1 = {V1_LABEL}] -- scores are NOT comparable "
              f"with pseudo_mask_params.json:\n{json.dumps(check, indent=2)}\n{bar}")
    else:
        print("\nharness check not run (v1_sato_thr0.10 not selected)")
    if contract["status"] == "FAIL":
        print(f"\n{bar}\nCONTRACT CHECK FAIL -- labelfree.sato_response != "
              f"pseudo_mask r:\n{json.dumps(contract, indent=2)}\n{bar}")
    elif contract["status"] == "PASS":
        print(f"contract check PASS: sato_response == pseudo_mask r "
              f"(max |diff| {contract['max_abs_diff']:.2e} <= {CONTRACT_TOL})")
    if "zone_convention" in check:
        z = check["zone_convention"]
        print(f"zone convention {z['status']}: v1 junction recall "
              f"{z['harness_value']:.3f} vs README_v3 {z['literal']:.3f} "
              f"(diff {z['diff']:+.4f}, tol {z['tol']})"
              + ("  <-- junction-zone definition may differ" if z["status"] == "WARN" else ""))

    v2_cmp = None
    if "v2_fixed" in results:
        m = results["v2_fixed"]["mean_over_frames"]
        v2_cmp = {k: {"literal": lit, "now": m[k], "diff": m[k] - lit}
                  for k, lit in V2_LITERALS.items()}
        print("v2_fixed vs old label-tuned literals (informational, provenance unknown): "
              + "  ".join(f"{k} {d['now']:.3f} vs {d['literal']:.3f}"
                          for k, d in v2_cmp.items()))

    payload = {
        "purpose": "GT used ONLY to score; nothing here may tune the label-free pipeline",
        "provenance": _provenance(),
        "predeclared_default": PREDECLARED,
        "methods_run": names,
        "parameters": {
            "junction_zone": {"family_disk_radius": FAMILY_RADIUS,
                              "min_families": MIN_FAMILIES,
                              "zone_disk_radius": ZONE_RADIUS,
                              "definition": "README_v3 sec. 6.2; zone within GT union"},
            "v1_thr": V1_THR, "sato_sigmas": list(v2.SIGMAS),
            "v2": {"thr": v2.THR, "close_angles": list(v2.CLOSE_ANGLES),
                   "close_len": v2.CLOSE_LEN},
            "harness_tol": HARNESS_TOL, "contract_tol": CONTRACT_TOL,
            "scorer": "to_50k/score copied verbatim from the removed tune_pseudo_mask.py",
            "oracle": f"threshold = quantile(score, 1 - GT coverage); {ORACLE_NOTE}",
            "ap_and_oracle_input": "the method's continuous score (sato response), "
                                   "not its thresholded/closed mask",
            "aggregation": "mean_over_frames (tune_pseudo_mask convention) + pooled pixels",
        },
        "frames": [{"name": f.name, "scale_to_50k": f.scale_to_50k,
                    "shape_50k": list(f.gt50.shape), "gt_cov": float(f.gt50.mean()),
                    "zone_fraction": f.zone_fraction} for f in frames],
        "harness_check": {**check, "label": V1_LABEL},
        "contract_check": contract,
        "v2_literal_comparison": v2_cmp,
        "removed_pipelines": "v1 (sato thr 0.10) and v3 (supervised RF) removed "
                             "2026-10-13; results in ~/ridge-filtering-archive/",
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")
    return payload


# --------------------------------------------------------------------------- selftest

def _synthetic_frame(seed: int, s: float = 0.625, shape=(320, 448)) -> tuple:
    """Mottled background + dark wires in 3 families at 0/60/123 deg (80k native)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    bg = cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), 2.0)
    img = 150 + 12 * bg / bg.std()
    filled = {f: np.zeros(shape, np.uint8) for f in FAMILIES}
    for fam, ang in zip(FAMILIES, (123.0, 60.0, 0.0)):  # red ~122, green ~60, blue ~0
        rad = np.deg2rad(ang)
        d = np.array([np.cos(rad), -np.sin(rad)])       # line_kernel convention (rows up)
        for _ in range(4):
            c = rng.uniform([0.15 * w, 0.15 * h], [0.85 * w, 0.85 * h])
            p0, p1 = (c - 400 * d).astype(int), (c + 400 * d).astype(int)
            cv2.line(filled[fam], tuple(map(int, p0)), tuple(map(int, p1)), 1, 7)
    union = np.zeros(shape, bool)
    for m in filled.values():
        union |= m.astype(bool)
    img[union] -= 70
    img = (img - img.min()) / (img.max() - img.min() + 1e-9) * 255.0
    gt = {"filled": {f: m.astype(bool) for f, m in filled.items()}, "union": union}
    return f"synthetic_{seed}", s, img.astype(np.float32), gt


def selftest(out_path: Path | None) -> None:
    frames = [build_frame(*_synthetic_frame(seed)) for seed in (1, 2)]
    fr = frames[0]
    assert fr.raw50.shape == fr.gt50.shape == (200, 280), fr.raw50.shape
    assert fr.zone.any() and not (fr.zone & ~fr.gt50).any(), "zone must be non-empty, in GT"
    assert 0 < fr.zone_fraction < 1

    # scorer edge cases: perfect, empty, inverted
    perf = evaluate(fr, fr.gt50, fr.gt50.astype(np.float32))["metrics"]
    for k in ("recall", "precision", "f1", "junction_recall", "ap",
              "oracle_f1", "oracle_junction_recall"):
        assert abs(perf[k] - 1) < 1e-9, (k, perf[k])
    empty = evaluate(fr, np.zeros_like(fr.gt50), None)["metrics"]
    assert empty["recall"] == empty["f1"] == empty["junction_recall"] == 0
    assert "ap" not in empty and "oracle_f1" not in empty
    inv = evaluate(fr, ~fr.gt50, None)["metrics"]
    assert inv["recall"] == 0 and inv["precision"] == 0
    try:
        evaluate(fr, fr.gt50[:-1], None)
        raise AssertionError("shape mismatch not caught")
    except ValueError:
        pass

    # pooled vs mean: equal-sized frames, pooled recall is the GT-weighted mean
    pf = {f.name: evaluate(f, f.gt50, None) for f in frames}
    mean, pooled = aggregate(pf)
    assert abs(mean["f1"] - 1) < 1e-9 and abs(pooled["f1"] - 1) < 1e-9

    # v1-era scorer copied from tune_pseudo_mask: exact semantics on a known case
    sc = score(np.array([[1, 1, 0, 0]], bool), np.array([[1, 0, 1, 0]], bool))
    assert sc["recall"] == sc["precision"] == sc["f1"] == 0.5 and sc["coverage_ratio"] == 1.0
    assert to_50k(np.zeros((320, 448), np.uint8), 0.625, nearest=True).shape == (200, 280)

    # full run over every registered method
    tmp = None
    if out_path is None:
        tmp = tempfile.TemporaryDirectory()
        out_path = Path(tmp.name) / "eval_masks_selftest.json"
    # scratch reference files always live in their own temp dir, never next to --out
    ref_tmp = tempfile.TemporaryDirectory()
    ref_dir = Path(ref_tmp.name)
    missing_ref = ref_dir / "no_such_reference.json"
    print("selftest: the HARNESS CHECK FAIL below is EXPECTED (reference missing on purpose)")
    payload = run(frames, list(METHODS), out_path, ref_path=missing_ref)

    reloaded = json.loads(out_path.read_text())
    assert reloaded["methods_run"] == list(METHODS)
    assert reloaded["harness_check"]["status"] == "FAIL"        # missing reference
    assert reloaded["harness_check"]["label"] == V1_LABEL
    assert reloaded["contract_check"]["status"] == "PASS", reloaded["contract_check"]
    assert reloaded["results"]["v1_sato_thr0.10"]["kind"] == "harness_self_check"
    assert not any(k.startswith("rf") for k in reloaded["results"])
    for key in ("git", "timestamp", "python", "sklearn", "skimage"):
        assert key in reloaded["provenance"], key
    for name, r in payload["results"].items():
        for fname, ev in r["per_frame"].items():
            for k, v in ev["metrics"].items():
                assert np.isfinite(v), (name, fname, k, v)
                if k in ("recall", "precision", "f1", "junction_recall", "ap",
                         "oracle_recall", "oracle_precision", "oracle_f1",
                         "oracle_junction_recall"):
                    assert 0 <= v <= 1, (name, fname, k, v)
            m = ev["metrics"]
            assert abs(m["oracle_pred_cov"] - m["gt_cov"]) < 0.02, (name, m)
        assert set(r["pooled"]) >= {"recall", "precision", "f1", "junction_recall"}
    v1 = payload["results"]["v1_sato_thr0.10"]
    for ev in v1["per_frame"].values():
        assert ev["info"]["max_abs_diff_vs_v2_pseudo_mask_r"] <= CONTRACT_TOL
    for name in METHODS:
        if name.startswith("lf_") and name != "lf_otsu_fixedangles":
            for ev in payload["results"][name]["per_frame"].values():
                assert ev["info"]["angles_source"] == "image", name
    fixed = payload["results"]["lf_otsu_fixedangles"]["per_frame"]
    assert all(ev["info"]["angles_deg"] == [0.0, 60.0, 123.0] for ev in fixed.values())
    assert all(not ev["info"]["close"]
               for ev in payload["results"]["lf_otsu_noclose"]["per_frame"].values())
    assert payload["v2_literal_comparison"] is not None

    # contract check logic: FAIL past tolerance, NOT_RUN without v1
    bad = {"per_frame": {"x": {"info": {"max_abs_diff_vs_v2_pseudo_mask_r": 2e-6}}}}
    assert contract_check(bad)["status"] == "FAIL"
    assert contract_check(None)["status"] == "NOT_RUN"

    # harness check logic: PASS on a matching reference, FAIL past tolerance
    vm = v1["mean_over_frames"]
    ref_file = ref_dir / "ref_selftest.json"
    entry = {"filter": "sato", "sigmas": "s1-3", "thr": 0.1,
             **{k: vm[k] for k in ("recall", "precision", "f1")}}
    ref_file.write_text(json.dumps({"all": [entry]}))
    assert harness_check(vm, ref_file)["status"] == "PASS"
    entry["f1"] += 2.5e-3
    ref_file.write_text(json.dumps({"all": [entry]}))
    assert harness_check(vm, ref_file)["status"] == "FAIL"
    assert harness_check(None, ref_file)["status"] == "NOT_RUN"
    z = harness_check(vm, ref_file)["zone_convention"]
    assert z["status"] in ("OK", "WARN") and abs(z["diff"] - (
        vm["junction_recall"] - V1_JREC_LITERAL)) < 1e-12
    ref_tmp.cleanup()

    print("\nSELFTEST PASS (synthetic frames; no GT files read)")
    if tmp is not None:
        tmp.cleanup()


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(f"methods: {PREDECLARED} = pre-declared label-free default; other lf_* =\n"
                f"label-free alternatives/ablations (reported, never used to pick the\n"
                f"default); v2_fixed = v2 classical comparison (label-tuned);\n"
                f"v1_sato_thr0.10 = {V1_LABEL}.\n"
                f"exit status 3 = harness self-check or sato contract check FAILED\n"
                f"(the JSON is still written)."))
    ap.add_argument("--methods", default="",
                    help=f"comma-separated subset of: {', '.join(METHODS)} (default all; "
                         f"bare label-free names like otsu,li mean lf_otsu,lf_li). "
                         f"The harness self-check only runs if v1_sato_thr0.10 is included")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output JSON (default {OUT_DEFAULT.relative_to(ROOT)})")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic frames only; no GT files; temp output")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.out)
        return

    names = [m.strip() for m in args.methods.split(",") if m.strip()] or list(METHODS)
    names = [f"lf_{m}" if m not in METHODS and f"lf_{m}" in METHODS else m for m in names]
    unknown = [m for m in names if m not in METHODS]
    if unknown:
        ap.error(f"unknown methods {unknown}; choose from {list(METHODS)}")
    names = sorted(names, key=list(METHODS).index)      # pre-declared default first

    payload = run(load_gt_frames(), names, args.out or OUT_DEFAULT)
    if "FAIL" in (payload["harness_check"]["status"], payload["contract_check"]["status"]):
        sys.exit(3)


if __name__ == "__main__":
    main()
