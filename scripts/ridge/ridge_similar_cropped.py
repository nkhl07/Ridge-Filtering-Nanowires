"""Tuned sato pseudo-masks for the similar-cropped training cache.

Runs the same ridge filter as prep_handcorrect_frame.py (sato sigma 1-3,
black_ridges=True, thr 0.10, per-frame min-max normalised response) over every
frame in data/real/similar_cropped_cache (already rescaled to 50k-equivalent,
the scale the sigmas were tuned at).

Outputs to results/ridge_similar_cropped/:
  <name>_pseudo.png    binary pseudo-mask, white = wire
  <name>_overlay.png   gray frame + magenta pseudo-mask
  ridge_coverage.json  per-frame coverage + summary stats
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import sato

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "real" / "similar_cropped_cache"
OUT = ROOT / "results" / "ridge_similar_cropped"
OUT.mkdir(parents=True, exist_ok=True)

KEEP_ROWS = 1024
SIGMAS = (1, 2, 3)      # tuned: sato sigma 1-3, thr 0.10 -> coverage ratio 0.94 vs GT
THR = 0.10


def pseudo_mask(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tuned sato pseudo-mask. Wires are DARK bodies -> black_ridges=True."""
    img01 = gray.astype(np.float32) / 255.0
    r = sato(img01, sigmas=SIGMAS, black_ridges=True)
    r = (r - r.min()) / (r.max() - r.min() + 1e-12)
    return r > THR, r


def crop_databar(a: np.ndarray) -> np.ndarray:
    return a[:KEEP_ROWS] if a.shape[0] > KEEP_ROWS else a


def main() -> None:
    paths = sorted(glob.glob(str(CACHE / "*.npy")))
    if not paths:
        raise SystemExit(f"No frames in {CACHE}")

    per_frame = {}
    for p in paths:
        name = Path(p).stem
        gray = crop_databar(np.load(p))
        mask, _ = pseudo_mask(gray)
        per_frame[name] = float(mask.mean())

        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
            OUT / f"{name}_pseudo.png")
        rgb = np.stack([gray] * 3, axis=2).astype(np.uint8)
        rgb[mask] = (255, 0, 255)
        Image.fromarray(rgb, mode="RGB").save(OUT / f"{name}_overlay.png")

    covs = np.array(sorted(per_frame.values()))
    summary = {
        "n_frames": len(covs),
        "median": float(np.median(covs)),
        "mean": float(covs.mean()),
        "p25": float(np.percentile(covs, 25)),
        "p75": float(np.percentile(covs, 75)),
        "min": float(covs.min()),
        "max": float(covs.max()),
    }

    for name, cov in sorted(per_frame.items(), key=lambda kv: kv[1]):
        print(f"  {name:34s} pseudo coverage {cov * 100:5.2f}%")
    print(f"\n{len(covs)} frames | median {summary['median'] * 100:.2f}% | "
          f"p25-p75 {summary['p25'] * 100:.2f}-{summary['p75'] * 100:.2f}% | "
          f"range {summary['min'] * 100:.2f}-{summary['max'] * 100:.2f}%")
    print("references: in-band full-frame pseudo median 12.76% | 80k GT 12.8%/9.7%")

    with open(OUT / "ridge_coverage.json", "w") as f:
        json.dump({
            "cache": str(CACHE.relative_to(ROOT)),
            "sato_sigmas": list(SIGMAS),
            "sato_thr": THR,
            "per_frame_coverage": per_frame,
            "summary": summary,
        }, f, indent=2)
    print(f"wrote {OUT.relative_to(ROOT)}/ridge_coverage.json + "
          f"{len(covs)} pseudo/overlay PNG pairs")


if __name__ == "__main__":
    main()
