"""Synthetic-only tests for unet/{dataset,losses,train_unet}.py (plain asserts).

Small arrays and a CPU model without pretrained weights -- safe on the login
node; slurm/unet_train.sbatch runs them before training.

Run from the repo root: OMP_NUM_THREADS=2 $PYTHON -m unet.test_train_unet
"""
from __future__ import annotations

import numpy as np
import torch

from unet.common import IGNORE, WIRE
from unet import dataset as D
from unet.losses import masked_hard_dice, masked_loss
from unet.train_unet import build_model

LCFG = {"bce_weight": 1.0, "dice_weight": 1.0, "cldice": {"enabled": True, "weight": 0.5, "iters": 5}}
DCFG = {"crop": 64, "samples_per_epoch": 32, "min_wire_frac": 0.01, "rich_grid_stride": 8}
ACFG = {"brightness": 0.08, "contrast": 0.15, "gamma": 1.2, "noise_std_max": 0.03}


def test_all_ignore_zero_loss_and_grad() -> None:
    torch.manual_seed(0)
    model = build_model({"encoder": "resnet34"}, weights=None)
    x = torch.randn(2, 1, 64, 64)
    y = torch.full((2, 1, 64, 64), IGNORE, dtype=torch.int64)
    terms = masked_loss(model(x), y, LCFG)          # clDice on: every term must vanish
    for k, v in terms.items():
        assert v.item() == 0.0, (k, v.item())
    terms["total"].backward()
    g = [p.grad for p in model.parameters() if p.grad is not None]
    assert g and all(float(t.abs().max()) == 0.0 for t in g)
    # control: one valid px gives a non-zero loss and gradient
    model.zero_grad()
    y[0, 0, 10, 10] = WIRE
    masked_loss(model(x), y, LCFG)["total"].backward()
    assert any(float(p.grad.abs().max()) > 0 for p in model.parameters() if p.grad is not None)


def test_ignore_px_do_not_change_loss() -> None:
    logits = torch.randn(1, 1, 32, 32)
    lab = (torch.rand(1, 1, 32, 32) > 0.7).long()
    lab[..., :8] = IGNORE
    a = masked_loss(logits, lab, LCFG)["total"]
    logits2 = logits.clone()
    logits2[..., :8] = 50.0                          # wild values under ignore
    assert torch.allclose(a, masked_loss(logits2, lab, LCFG)["total"])


def test_hard_dice() -> None:
    lab = np.array([[1, 1, 0, IGNORE]], np.uint8)
    assert masked_hard_dice(np.array([[0.9, 0.1, 0.1, 0.9]]), lab) == 2 / 3
    assert masked_hard_dice(np.zeros((1, 4)), np.array([[0, 0, IGNORE, IGNORE]], np.uint8)) == 1.0


def test_split_stratified() -> None:
    tiers = {f"a{i}": 50000 for i in range(8)} | {f"b{i}": 65000 for i in range(8)} \
        | {"c0": 70000} | {f"d{i}": 80000 for i in range(4)}
    sp = D.split_frames(tiers, 0.2, seed=0)
    assert len(sp["val"]) == 5 and len(sp["train"]) == 16
    assert sorted(sp["train"] + sp["val"]) == sorted(tiers)
    vt = [tiers[n] for n in sp["val"]]
    assert vt.count(50000) == 2 and vt.count(65000) == 2 and vt.count(80000) == 1 and "c0" in sp["train"]
    assert sp == D.split_frames(dict(reversed(list(tiers.items()))), 0.2, seed=0)  # order-free
    assert D.tier_of(0.7692305791505791) == 65000 and D.tier_of(0.6250007722007722) == 80000


def _frames():
    rng = np.random.default_rng(0)
    out = []
    for k in range(2):
        img = rng.random((150, 200)).astype(np.float32)
        lab = np.zeros((150, 200), np.uint8)
        lab[20 + 40 * k:24 + 40 * k, 30:90] = WIRE      # one short wire per frame
        lab[100:110, 150:160] = IGNORE
        out.append(D.Frame(f"f{k}", 50000, img, lab))
    return out


def test_patches_half_wire_rich_and_aligned() -> None:
    ds = D.PatchDataset(_frames(), 64, 32, seed=0, dcfg=DCFG, augment_cfg=ACFG)
    for i in range(0, 32, 2):
        x, y = ds[i]
        assert x.shape == (1, 64, 64) and y.shape == (1, 64, 64)
        assert (y == WIRE).float().mean() >= 0.01, i
        assert set(torch.unique(y).tolist()) <= {0, WIRE, IGNORE}
    # reproducible per (seed, epoch, index); a new epoch draws new patches
    assert torch.equal(ds[3][0], ds[3][0])
    a = ds[3][0]
    ds.set_epoch(1)
    assert not torch.equal(a, ds[3][0])
    # geometric augmentation moves image and label together: encode the label in the image
    f = _frames()[0]
    f.img = (f.label == WIRE).astype(np.float32)
    rng = np.random.default_rng(1)
    for _ in range(20):
        img, lab = D.augment(f.img, f.label, rng, {"brightness": 0, "contrast": 0,
                                                   "gamma": 1.0, "noise_std_max": 0})
        assert np.array_equal(img > 0.5, lab == WIRE)


def test_val_tiles_cover_frame() -> None:
    f = _frames()[0]
    x, pos = D.val_tiles(f, 64)
    assert D.tile_starts(150, 64) == [0, 64, 86]
    t = np.stack([f.img[r:r + 64, c:c + 64] for r, c in pos])
    assert np.array_equal(D.stitch_tiles(t, pos, f.img.shape), f.img)
    assert torch.allclose(x[0, 0], torch.from_numpy(D.normalise(f.img[:64, :64])))


def main() -> None:
    test_all_ignore_zero_loss_and_grad()
    test_ignore_px_do_not_change_loss()
    test_hard_dice()
    test_split_stratified()
    test_patches_half_wire_rich_and_aligned()
    test_val_tiles_cover_frame()
    print("ALL PASS")


if __name__ == "__main__":
    main()
