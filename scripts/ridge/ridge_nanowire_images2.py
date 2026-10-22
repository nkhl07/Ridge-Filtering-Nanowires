"""Wire masks for ALL full frames in data/real/NanowireImages2.

Two pipelines (see --help):

  label-free (default)  sato + per-image auto threshold + directional closing at
                        angles estimated from the image itself (labelfree.py)
  v2  (--classical)     sato + fixed THR 0.14 + closing at fixed 0/60/123 deg;
                        label-tuned, kept as a COMPARISON pipeline only

Unsupervised rule (decided 2026-10-13): the pipeline must stay UNSUPERVISED.
The 2 hand-labelled GT frames (data/real/OneDrive_1_7-17-2026) may be used
ONLY to evaluate a mask (scripts/ridge/eval_masks.py), never to train,
threshold or choose anything, and no new manual labels are made. v2's THR and
CLOSE_ANGLES were tuned against the GT frames, which is why v2 is a comparison
and not the default.

Removed 2026-10-13: v1 (sato thr 0.10, no closing) and v3 (supervised
texture+ridge random forest trained on the GT frames). Their scripts and
results are archived in ~/ridge-filtering-archive/v1_v3_results_2026-10-13.tar
(see the README.txt there); v1 survives only as eval_masks.py's harness
self-check.

Pipeline label-free (default): v2's sato response (same sigmas, same
black_ridges, same 50k-equivalent scale) but THR is replaced by
skimage threshold_<method> on each frame's normalised response (default
otsu, pre-declared before any GT evaluation; --lf-method picks another) and
CLOSE_ANGLES by up to 3 orientation-family peaks measured on that frame. The
remaining constants (SIGMAS, CLOSE_LEN, 3 families, 50k target scale) are
declared priors carried over from the label era and frozen, recorded in
ridge_coverage.json under "priors". The masks come out already
direction-closed. No GT scores are written here -- run eval_masks.py.

Pipeline v2 (`--classical`, junction-closing -- tuned 2026-07-24 against the 2 GT frames):
sato sigma 1-3, black_ridges=True, per-frame min-max normalised response,
thr 0.14, then morphological closing with 9 px line kernels at the three
measured family orientations (0/60/123 deg, from real_stats). The closing
bridges the junction dropouts where crossing wires read as a blob (not a
ridge) and the Hessian response craters -- ~20% of GT wire pixels sit in
junction zones. Coverage-matched scores vs GT: recall 0.517->0.570,
precision 0.484->0.505, junction recall 0.574->0.672 against the v1
sato-only pipeline (thr 0.10) -- unverified 2026-07-24 literals whose
generating code was lost; eval_masks.py re-measures them. meijering and a
steerable oriented-ridge bank were both tried and scored WORSE than sato --
do not revisit without new evidence. Run on the RAW 62-frame TIFF set, not a
cache. Differences from the cache-based passes, on purpose:

  1. Scale comes from per-frame FEI metadata (tag 34682 PixelWidth, cross-
     checked against HFW/width), NOT the filename magnification. Five frames
     (12000x_59, 20000x_58, 50000x_16, 65000x_60, 6500x_57) have metadata
     pixel sizes 2x off their filename mag; the caches were built with
     filename-mag factors and are mis-scaled for those frames.
  2. The databar is cropped at NATIVE resolution (bottom 70/1536-width rows)
     BEFORE rescaling. normalized_cache rescaled databar and all, so shrunk
     databars survived the KEEP_ROWS=1024 crop in sub-1024-row frames.
  3. Frames needing upscale > 2x to reach 50k-equivalent have native wire
     widths under ~2 px (sub-resolution); they are processed anyway but
     flagged `unreliable_upscale` -- treat their masks as indicative only.
  4. The sato params were only ever VALIDATED at rescale factors 0.625-1.0
     (80k GT x0.625 tuning + 50k in-band frames). Outside that band the masks
     visibly degrade in two distinct ways: downscaled high-mag frames
     (100-200k, factor <= 0.5) over-fire on background mottle that lands at
     ridge scale (200k frames report ~30-58% coverage vs a true ~10-15%);
     upscaled low-mag frames show wires as BRIGHT rods, so black_ridges fires
     on their dark halos -- rim detections, under-counting bodies. Frames
     outside the band carry `scale_in_validated_band: false` and are excluded
     from the validated-band summary.

The scale handling below (1-4) is shared by both pipelines.

Outputs to results/ridge_labelfree/ (default) or results/ridge_v2/
(--classical); --out DIR overrides either:
  <name>_50k_pseudo.png    binary pseudo-mask at 50k-equivalent scale
  <name>_50k_overlay.png   gray frame + magenta mask (downscaled if huge)
  ridge_coverage.json      per-frame coverage, scale provenance, flags, summary

Importing this module has no side effects (labelfree.py, wire_extract.py and
the tests import its constants); directories are only created in main().
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.filters import sato

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "real" / "NanowireImages2"
OUT_LF = ROOT / "results" / "ridge_labelfree"       # default (label-free) output
OUT_V2 = ROOT / "results" / "ridge_v2"              # --classical output
# Hardcoded copy of labelfree.LF_METHODS / DEFAULT_METHOD so --help never has to
# import labelfree (which imports this module); main() asserts they agree.
LF_METHOD_CHOICES = ("otsu", "li", "triangle", "yen", "isodata")
LF_DEFAULT_METHOD = "otsu"

SIGMAS = (1, 2, 3)      # tuned sato scales (unchanged from v1)
THR = 0.14              # retuned WITH the closing; coverage ratio 0.89 vs GT
CLOSE_LEN = 9           # line-kernel length (px @50k) for junction closing
CLOSE_ANGLES = (0, 60, 123)     # measured orientation families (real_stats)
TARGET_PX = 8.288e-6 / 1536     # 50k-equivalent pixel size (m), from FEI HFW
DATABAR_ROWS = 70               # bottom rows at 1536-wide native resolution
MISMATCH_TOL = 0.05             # filename-mag vs metadata scale disagreement flag
UPSCALE_FLAG = 2.0              # above this, native wires are sub-resolution
VALIDATED_BAND = (0.625, 1.0)   # rescale factors the sato params were tuned/checked at
OVERLAY_MAX_SIDE = 8192         # overlays above this are downscaled for viewing


def read_pixel_size(tif_path: Path) -> tuple[float, Image.Image]:
    """Pixel size in metres from FEI tag 34682 (PixelWidth, else HFW/width)."""
    im = Image.open(tif_path)
    blob = str(im.tag_v2.get(34682)) if hasattr(im, "tag_v2") else ""
    num = r"([0-9][0-9.]*(?:[eE][+-]?[0-9]+)?)"   # blob has empty "PixelWidth=" lines
    m = re.search(rf"PixelWidth={num}", blob)
    if m:
        return float(m.group(1)), im
    m = re.search(rf"HFW={num}", blob)
    if not m:
        raise ValueError(f"No PixelWidth/HFW in metadata: {tif_path}")
    return float(m.group(1)) / im.size[0], im


def line_kernel(length: int, ang_deg: float) -> np.ndarray:
    """Binary line structuring element at the given orientation."""
    L = length if length % 2 else length + 1
    ker = np.zeros((L, L), np.uint8)
    cc = (L - 1) / 2
    rad = np.deg2rad(ang_deg)
    for t in np.linspace(-cc, cc, 4 * L):
        ker[int(round(cc - t * np.sin(rad))), int(round(cc + t * np.cos(rad)))] = 1
    return ker


KERNELS = [line_kernel(CLOSE_LEN, a) for a in CLOSE_ANGLES]


def dir_close(mask: np.ndarray) -> np.ndarray:
    """Union of closings along the family orientations: bridges junction gaps."""
    u8 = mask.astype(np.uint8)
    out = mask.copy()
    for ker in KERNELS:
        out |= cv2.morphologyEx(u8, cv2.MORPH_CLOSE, ker).astype(bool)
    return out


def pseudo_mask(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tuned sato pseudo-mask + junction closing. Wires are DARK bodies."""
    img01 = gray.astype(np.float32) / 255.0
    r = sato(img01, sigmas=SIGMAS, black_ridges=True)
    r = (r - r.min()) / (r.max() - r.min() + 1e-12)
    return dir_close(r > THR), r


def load_frame_50k(tif_path: Path) -> tuple[np.ndarray, dict]:
    """Native load -> databar crop -> min-max [0,255] -> 50k-equivalent rescale."""
    px, im = read_pixel_size(tif_path)
    a = np.array(im).astype(np.float32)
    h, w = a.shape
    a = a[: h - DATABAR_ROWS * w // 1536]                   # crop at native scale
    a = (a - a.min()) / (a.max() - a.min() + 1e-9) * 255.0  # load_raw convention

    factor = px / TARGET_PX
    interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    if abs(factor - 1) > 1e-3:
        a = cv2.resize(a, (int(round(a.shape[1] * factor)),
                           int(round(a.shape[0] * factor))), interpolation=interp)

    name_mag = int(tif_path.stem.split("_")[1].rstrip("×"))
    name_factor = 50000 / name_mag
    meta = {
        "native_shape": [h, w],
        "pixel_size_m": px,
        "rescale_to_50k": factor,
        "filename_mag": name_mag,
        "mag_mismatch": abs(factor - name_factor) / name_factor > MISMATCH_TOL,
        "unreliable_upscale": factor > UPSCALE_FLAG,
        "scale_in_validated_band":
            VALIDATED_BAND[0] - 1e-9 <= factor <= VALIDATED_BAND[1] + 1e-9,
    }
    return a, meta


def flag_str(meta: dict) -> str:
    flags = [f for f in ("mag_mismatch", "unreliable_upscale") if meta[f]]
    if not meta["scale_in_validated_band"]:
        flags.append("extrapolated_scale")
    return "".join(f" [{f}]" for f in flags)


def save_pair(out_dir: Path, name: str, gray: np.ndarray, mask: np.ndarray) -> float:
    """Write pseudo + overlay PNGs; overlay downscaled if huge. Returns overlay scale."""
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
        out_dir / f"{name}_50k_pseudo.png")

    rgb = np.stack([gray] * 3, axis=2).astype(np.uint8)
    rgb[mask] = (255, 0, 255)
    s = min(1.0, OVERLAY_MAX_SIDE / max(rgb.shape[:2]))
    if s < 1.0:
        rgb = cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)),
                         interpolation=cv2.INTER_AREA)
    Image.fromarray(rgb, mode="RGB").save(out_dir / f"{name}_50k_overlay.png")
    return s


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI. Runs before any data is touched, so --help / bad flags are free.

    Unknown flags error (the old hand-rolled sys.argv parsing silently ran the
    default pipeline on a typo like `--clasical`).
    """
    ap = argparse.ArgumentParser(
        description="Wire masks for every NanowireImages2 frame at 50k-equivalent scale. "
                    "Default: label-free (sato + per-image auto threshold + closing at "
                    "image-estimated angles). GT frames are evaluation-only "
                    "(scripts/ridge/eval_masks.py).")
    ap.add_argument("only", nargs="?", default="",
                    help="only frames whose path contains this substring (default: all). "
                         "Filtered runs do not rewrite ridge_coverage.json.")
    ap.add_argument("--classical", action="store_true",
                    help="v2 comparison pipeline: sato + THR 0.14 + closing at "
                         "0/60/123 deg (label-tuned; not the default)")
    ap.add_argument("--lf-method", choices=LF_METHOD_CHOICES, default=None,
                    help=f"auto-threshold method of the label-free pipeline (default "
                         f"{LF_DEFAULT_METHOD}, pre-declared; do not pick by GT score)")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output dir (default {OUT_LF.relative_to(ROOT)}, or "
                         f"{OUT_V2.relative_to(ROOT)} with --classical)")
    args = ap.parse_args(argv)
    if args.lf_method is not None and args.classical:
        ap.error("--lf-method applies to the label-free pipeline, not --classical")
    if args.lf_method is None:
        args.lf_method = LF_DEFAULT_METHOD
    if args.out is None:
        args.out = OUT_V2 if args.classical else OUT_LF
    args.out = args.out.resolve()
    return args


def _display(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def main() -> None:
    args = parse_args()
    classical, only, out_dir = args.classical, args.only, args.out
    labelfree = not classical
    paths = [Path(p) for p in sorted(glob.glob(str(SRC / "*.tif"))) if only in p]
    if not paths:
        raise SystemExit(f"No frames matching '{only}' in {SRC}")
    out_dir.mkdir(parents=True, exist_ok=True)

    lf_info: dict = {}          # label-free: info dict of the frame just masked
    if classical:
        make_mask = lambda gray: pseudo_mask(gray)[0]
        pipeline = "v2 classical sato+dirclose"
    else:
        import labelfree as lf      # lazy: labelfree imports this module
        assert tuple(lf.LF_METHODS) == LF_METHOD_CHOICES, "LF_METHOD_CHOICES out of sync"
        assert lf.DEFAULT_METHOD == LF_DEFAULT_METHOD, "LF_DEFAULT_METHOD out of sync"

        def make_mask(gray: np.ndarray) -> np.ndarray:
            mask, _, info = lf.labelfree_mask(gray, method=args.lf_method)
            lf_info.clear()
            lf_info.update(info)
            return mask

        pipeline = (f"label-free sato (auto-threshold {args.lf_method} "
                    f"+ dirclose at image-estimated angles)")
    print(f"pipeline: {pipeline}", flush=True)
    print(f"output:   {_display(out_dir)}", flush=True)

    # cheap frames first so the log fills up fast; the 4010x monster goes last
    def out_pixels(p: Path) -> float:
        px, im = read_pixel_size(p)          # metadata only, pixel data stays lazy
        return im.size[0] * im.size[1] * (px / TARGET_PX) ** 2

    paths.sort(key=out_pixels)

    per_frame: dict[str, dict] = {}
    for i, p in enumerate(paths):
        gray, meta = load_frame_50k(p)
        mask = make_mask(gray)
        meta["coverage"] = float(mask.mean())
        meta["shape_50k"] = list(gray.shape)
        meta["overlay_scale"] = save_pair(out_dir, p.stem, gray, mask)
        lf_str = ""
        if labelfree:
            meta["lf_method"] = lf_info["method"]
            meta["lf_thr"] = float(lf_info["thr"])
            meta["lf_angles_deg"] = [float(a) for a in lf_info["angles_deg"]]
            lf_str = (f" | {meta['lf_method']} thr {meta['lf_thr']:.4f} angles "
                      + "/".join(f"{a:.1f}" for a in meta["lf_angles_deg"]))
        per_frame[p.stem] = meta
        print(f"[{i + 1:2d}/{len(paths)}] {p.stem:26s} {str(meta['shape_50k']):14s} "
              f"x{meta['rescale_to_50k']:.3f} coverage {meta['coverage'] * 100:5.2f}%"
              f"{lf_str}{flag_str(meta)}", flush=True)

    def stats(covs: np.ndarray) -> dict:
        if len(covs) == 0:
            return {"n": 0}
        return {
            "n": len(covs),
            "median": float(np.median(covs)),
            "mean": float(covs.mean()),
            "p25": float(np.percentile(covs, 25)),
            "p75": float(np.percentile(covs, 75)),
            "min": float(covs.min()),
            "max": float(covs.max()),
        }

    summary = stats(np.array([m["coverage"] for m in per_frame.values()
                              if not m["unreliable_upscale"]]))
    summary_band = stats(np.array([m["coverage"] for m in per_frame.values()
                                   if m["scale_in_validated_band"]]))

    print()
    for name, meta in sorted(per_frame.items(), key=lambda kv: kv[1]["coverage"]):
        print(f"  {name:26s} pseudo coverage {meta['coverage'] * 100:5.2f}%"
              f"{flag_str(meta)}")
    for label, s in (("upscale<=2x", summary), ("validated band 0.625-1.0", summary_band)):
        if s["n"] == 0:
            print(f"\n0/{len(per_frame)} frames ({label})")
            continue
        print(f"\n{s['n']}/{len(per_frame)} frames ({label}) | "
              f"median {s['median'] * 100:.2f}% | "
              f"p25-p75 {s['p25'] * 100:.2f}-{s['p75'] * 100:.2f}% | "
              f"range {s['min'] * 100:.2f}-{s['max'] * 100:.2f}%")
    if labelfree:
        # no GT coverage targets here: choosing --lf-method by matching them
        # would be tuning on labels (GT is evaluation-only, via eval_masks.py)
        print("GT is evaluation-only for the label-free pipeline: score with "
              "scripts/ridge/eval_masks.py")
    else:
        print("references: 80k GT 12.8%/9.7% | v1 sato-only in-band median was 12.76%")

    if only:
        print(f"filtered run ('{only}'): NOT rewriting ridge_coverage.json")
        return
    if labelfree:
        # GT is evaluation-only for this pipeline: no scores are stored next to
        # the masks, only the label-free settings and the declared priors.
        scores = {
            "gt_scores": "not stored here; run scripts/ridge/eval_masks.py -> "
                         "results/eval/eval_masks.json (GT used for evaluation only)",
            "label_free": True,
            "lf_method": args.lf_method,
            "priors": {
                "sato_sigmas": list(SIGMAS),
                "black_ridges": True,
                "close_len_px": CLOSE_LEN,
                "k_families": int(getattr(lf, "K_FAMILIES", 3)),
                "target_scale": "50k-equivalent",
                "target_pixel_size_m": TARGET_PX,
                "provenance": "label-era values (v1/v2: sigmas and close length were "
                              "compared on the 2 GT frames), frozen; not re-tuned",
            },
        }
    else:
        scores = {
            "gt_scores_coverage_matched": {
                "recall": 0.570, "precision": 0.505, "f1": 0.534,
                "coverage_ratio": 0.89, "junction_recall": 0.672,
                "sato_sigmas": list(SIGMAS), "sato_thr": THR,
                "dirclose": {"length_px": CLOSE_LEN, "angles_deg": list(CLOSE_ANGLES)}},
            "gt_scores_provenance": "hardcoded literal from 2026-07-24; generating "
                                    "code lost; re-measured by scripts/ridge/eval_masks.py",
        }
    # build + serialise first, then write a temp file and swap it in: a failure
    # must never leave a truncated ridge_coverage.json for wire_extract to choke on
    text = json.dumps({
        "source": _display(SRC),
        "pipeline": pipeline,
        # both pipelines direction-close their masks; wire_extract.py reads this
        # to skip its own junction closing
        "dir_closed": True,
        **scores,
        "target_pixel_size_m": TARGET_PX,
        "scale_provenance": "FEI tag 34682 PixelWidth (NOT filename mag)",
        "validated_band_rescale": list(VALIDATED_BAND),
        "per_frame": per_frame,
        "summary_upscale_le2": summary,
        "summary_validated_band": summary_band,
    }, indent=2)
    tmp = out_dir / "ridge_coverage.json.tmp"
    tmp.write_text(text)
    os.replace(tmp, out_dir / "ridge_coverage.json")
    print(f"wrote {_display(out_dir)}/ridge_coverage.json + "
          f"{len(per_frame)} pseudo/overlay PNG pairs")


if __name__ == "__main__":
    main()
