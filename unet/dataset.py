"""unet_v1 Phase 2 data: frame-level split, random training patches, val tiles.

Inputs are the Phase 1 outputs in results/unet_v1/pseudo_labels/ (v1e,
commit ce1670a): <name>_img.npy (float32 [0,255], 50k-equivalent scale) and
<name>_label.png (0 background, 1 wire, 255 ignore). GT frames are never read.

  split_frames      train / pseudo-val split BY FRAME, stratified by
                    magnification tier (from the metadata rescale factor,
                    not the filename mag)
  PatchDataset      random crop x crop patches; even indices are drawn from
                    crops with >= min_wire_frac wire px, odd indices uniformly,
                    so a sequential batch is half wire-rich. Augmentation:
                    flips, 90 deg rotations, brightness/contrast/gamma jitter,
                    Gaussian noise -- no scale or elastic (pixel scale is
                    calibrated and stays fixed)
  val_tiles         deterministic crop x crop tiles covering a whole frame
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from unet.common import IGNORE, WIRE

# 1-channel input: x/255 normalised with the mean of ImageNet's RGB mean/std
# (smp sums the pretrained RGB stem weights into one channel)
NORM_MEAN, NORM_STD = 0.449, 0.226


@dataclass
class Frame:
    name: str
    tier: int                  # 50k-equivalent magnification tier, e.g. 65000
    img: np.ndarray            # float32 [0,1]
    label: np.ndarray          # uint8 {0, 1, 255}


def tier_of(rescale_to_50k: float) -> int:
    """Magnification tier from the metadata rescale factor (50k / factor, to 5k)."""
    return int(round(50000 / rescale_to_50k / 5000) * 5000)


def load_frames(label_dir: Path, names: list[str] | None = None) -> list[Frame]:
    summary = json.loads((label_dir / "summary.json").read_text())
    out = []
    for name, st in sorted(summary["per_frame"].items()):
        if names is not None and name not in names:
            continue
        img = np.load(label_dir / f"{name}_img.npy").astype(np.float32) / 255.0
        lab = np.array(Image.open(label_dir / f"{name}_label.png"))
        assert img.shape == lab.shape and set(np.unique(lab)) <= {0, WIRE, IGNORE}, name
        out.append(Frame(name, tier_of(st["rescale_to_50k"]), img, lab))
    return out


def split_frames(names_tiers: dict[str, int], val_frac: float, seed: int,
                 exclude: list[str] = ()) -> dict:
    """Frame-level split, stratified by tier: round(val_frac * n) val frames per
    tier (a tier with one frame stays in train). Seeded and order-independent."""
    rng = np.random.default_rng(seed)
    train, val = [], []
    for tier in sorted(set(names_tiers.values())):
        names = sorted(n for n, t in names_tiers.items() if t == tier and n not in exclude)
        k = int(round(val_frac * len(names))) if len(names) > 1 else 0
        pick = set(rng.choice(names, size=k, replace=False).tolist()) if k else set()
        val += [n for n in names if n in pick]
        train += [n for n in names if n not in pick]
    return {"train": sorted(train), "val": sorted(val),
            "tiers": {n: names_tiers[n] for n in sorted(names_tiers)},
            "val_frac": val_frac, "seed": seed}


def normalise(img01: np.ndarray) -> np.ndarray:
    return ((img01 - NORM_MEAN) / NORM_STD).astype(np.float32)


def wire_rich_corners(label: np.ndarray, crop: int, stride: int, min_frac: float) -> np.ndarray:
    """Top-left corners (on a stride grid) of crops with >= min_frac wire px."""
    H, W = label.shape
    ii = np.pad((label == WIRE).astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    rs, cs = np.arange(0, H - crop + 1, stride), np.arange(0, W - crop + 1, stride)
    r, c = np.meshgrid(rs, cs, indexing="ij")
    n = ii[r + crop, c + crop] - ii[r, c + crop] - ii[r + crop, c] + ii[r, c]
    keep = n >= min_frac * crop * crop
    return np.stack([r[keep], c[keep]], axis=1)


def augment(img: np.ndarray, lab: np.ndarray, rng: np.random.Generator,
            acfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Geometric ops on both arrays; photometric ops on the image only ([0,1] space)."""
    if rng.random() < 0.5:
        img, lab = img[:, ::-1], lab[:, ::-1]
    if rng.random() < 0.5:
        img, lab = img[::-1], lab[::-1]
    k = int(rng.integers(4))
    img, lab = np.rot90(img, k), np.rot90(lab, k)
    b, c, g = acfg["brightness"], acfg["contrast"], acfg["gamma"]
    img = np.clip(img, 0, 1) ** rng.uniform(1 / g, g)
    img = (img - 0.5) * rng.uniform(1 - c, 1 + c) + 0.5 + rng.uniform(-b, b)
    img = img + rng.normal(0, rng.uniform(0, acfg["noise_std_max"]), img.shape)
    return np.ascontiguousarray(img, np.float32), np.ascontiguousarray(lab)


class PatchDataset(Dataset):
    """`length` random patches per epoch; call set_epoch() so every epoch differs.

    Index parity sets the patch kind (even = wire-rich, odd = uniform), so with
    an unshuffled DataLoader and an even batch size each batch is half wire-rich.
    Each (seed, epoch, index) has its own RNG: reproducible with any worker count.
    Frames are drawn in proportion to their area.
    """

    def __init__(self, frames: list[Frame], crop: int, length: int, seed: int, dcfg: dict,
                 augment_cfg: dict | None):
        self.frames, self.crop, self.length, self.seed = frames, crop, length, seed
        self.acfg = augment_cfg
        self.epoch = 0
        area = np.array([f.img.size for f in frames], float)
        self.p = area / area.sum()
        self.rich = [wire_rich_corners(f.label, crop, dcfg["rich_grid_stride"],
                                       dcfg["min_wire_frac"]) for f in frames]
        has = np.array([len(r) > 0 for r in self.rich])
        assert has.any(), "no wire-rich crop in any training frame"
        self.p_rich = np.where(has, self.p, 0) / self.p[has].sum()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        cr = self.crop
        if i % 2 == 0:
            fi = rng.choice(len(self.frames), p=self.p_rich)
            r, c = self.rich[fi][rng.integers(len(self.rich[fi]))]
        else:
            fi = rng.choice(len(self.frames), p=self.p)
            H, W = self.frames[fi].img.shape
            r, c = rng.integers(H - cr + 1), rng.integers(W - cr + 1)
        f = self.frames[fi]
        img, lab = f.img[r:r + cr, c:c + cr], f.label[r:r + cr, c:c + cr]
        if self.acfg is not None:
            img, lab = augment(img, lab, rng, self.acfg)
        return (torch.from_numpy(normalise(img))[None],
                torch.from_numpy(lab.astype(np.int64))[None])


def tile_starts(n: int, crop: int) -> list[int]:
    """Non-overlapping starts covering [0, n); the last tile is flush with the edge."""
    s = list(range(0, n - crop + 1, crop))
    if s[-1] + crop < n:
        s.append(n - crop)
    return s


def val_tiles(frame: Frame, crop: int) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """All crop x crop image tiles covering the frame and their top-left corners
    (edge tiles overlap their neighbour; stitch_tiles writes each px once)."""
    H, W = frame.img.shape
    pos = [(r, c) for r in tile_starts(H, crop) for c in tile_starts(W, crop)]
    x = np.stack([normalise(frame.img[r:r + crop, c:c + crop]) for r, c in pos])
    return torch.from_numpy(x)[:, None], pos


def stitch_tiles(tiles: np.ndarray, pos: list[tuple[int, int]], shape) -> np.ndarray:
    """Inverse of val_tiles for (n, crop, crop) arrays: later tiles overwrite overlaps."""
    out = np.zeros(shape, tiles.dtype)
    cr = tiles.shape[-1]
    for t, (r, c) in zip(tiles, pos):
        out[r:r + cr, c:c + cr] = t
    return out
