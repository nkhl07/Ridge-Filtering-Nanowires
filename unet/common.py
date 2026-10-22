"""Shared plumbing for unet_v1: paths, config, frame selection, provenance.

Importing this module puts scripts/ridge on sys.path so the existing pipeline
modules (labelfree, ridge_nanowire_images2, wire_extract, gt_labels) are
imported as-is -- unet/ never modifies or re-implements them.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RIDGE_DIR = ROOT / "scripts" / "ridge"
sys.path.insert(0, str(RIDGE_DIR))

from ridge_nanowire_images2 import TARGET_PX, read_pixel_size  # noqa: E402

# label values of the cleaned pseudo-labels
BG, WIRE, IGNORE = 0, 1, 255


def rel(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def select_frames(cfg: dict) -> list[tuple[Path, float]]:
    """(tif path, 50k rescale factor) of the in-band, non-excluded frames, sorted by name.

    Scale comes from FEI metadata (read_pixel_size), never the filename mag.
    The GT frames live in a different directory and are never listed here;
    the name check below is a second guard.
    """
    fcfg = cfg["frames"]
    lo, hi = fcfg["scale_band"]
    out = []
    for p in sorted((ROOT / cfg["paths"]["raw_dir"]).glob("*.tif")):
        if any(x in p.name for x in fcfg["exclude"]):
            continue
        assert not p.name.startswith("TiS3"), f"GT frame in the training pool: {p}"
        px, _ = read_pixel_size(p)
        factor = px / TARGET_PX
        if not (lo - 1e-9 <= factor <= hi + 1e-9):
            continue
        assert factor <= fcfg["max_upscale"] + 1e-9, f"{p.name}: upscale {factor:.2f}x"
        out.append((p, factor))
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git() -> dict:
    def run(*args):
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    status = run("status", "--porcelain") or ""
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status),
            "n_uncommitted_paths": len(status.splitlines())}


def write_provenance(out_dir: Path, cfg_path: Path, extra: dict | None = None) -> dict:
    """Copy the config and write provenance.json (git hash + source hashes).

    The working tree is dirty (labelfree.py etc. are uncommitted), so the
    commit hash alone does not pin the code: every imported source file is
    hashed as well.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cfg_path, out_dir / "config.yaml")
    sources = sorted((ROOT / "unet").rglob("*.py")) + [
        RIDGE_DIR / n for n in ("labelfree.py", "ridge_nanowire_images2.py",
                                "wire_extract.py", "gt_labels.py", "eval_masks.py")]
    prov = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": rel(cfg_path),
        "git": _git(),
        "source_sha256": {rel(p): _sha256(p) for p in sources if p.exists()},
        "python": platform.python_version(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "argv": sys.argv,
    }
    if extra:
        prov.update(extra)
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
    return prov
