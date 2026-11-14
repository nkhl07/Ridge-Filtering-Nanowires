"""unet_v1 losses over non-ignore pixels only.

Every term multiplies by the valid mask (label != 255) before any reduction,
so ignore pixels contribute neither loss nor gradient, and a batch that is all
ignore gives exactly zero loss and zero gradient (test_train_unet checks this).
Soft Dice and clDice are computed over the whole batch (one ratio), which
keeps them defined on wire-free patches.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from unet.common import IGNORE

EPS = 1.0       # Dice / clDice smoothing (px)


def split_target(label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """label (int, {0,1,255}) -> (float wire target, float valid mask)."""
    valid = (label != IGNORE).float()
    return (label == 1).float() * valid, valid


def masked_bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * valid).sum() / valid.sum().clamp_min(1.0)


def masked_soft_dice(prob: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    p = prob * valid
    return 1 - (2 * (p * target).sum() + EPS) / (p.sum() + target.sum() + EPS)


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    return torch.minimum(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, 3, 1, 1)


def soft_skeleton(x: torch.Tensor, iters: int) -> torch.Tensor:
    """Shit et al. 2021 (clDice) differentiable skeleton."""
    skel = F.relu(x - _soft_dilate(_soft_erode(x)))
    for _ in range(iters):
        x = _soft_erode(x)
        d = F.relu(x - _soft_dilate(_soft_erode(x)))
        skel = skel + F.relu(d - skel * d)
    return skel


def masked_cldice(prob: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                  iters: int) -> torch.Tensor:
    p = prob * valid
    sp, st = soft_skeleton(p, iters), soft_skeleton(target, iters)
    tprec = ((sp * target).sum() + EPS) / (sp.sum() + EPS)
    tsens = ((st * p).sum() + EPS) / (st.sum() + EPS)
    return 1 - 2 * tprec * tsens / (tprec + tsens)


def masked_loss(logits: torch.Tensor, label: torch.Tensor, lcfg: dict) -> dict:
    """BCE + soft Dice (+ optional clDice), all masked; computed in float32."""
    logits = logits.float()
    target, valid = split_target(label)
    prob = torch.sigmoid(logits)
    terms = {"bce": masked_bce(logits, target, valid),
             "dice": masked_soft_dice(prob, target, valid)}
    total = lcfg["bce_weight"] * terms["bce"] + lcfg["dice_weight"] * terms["dice"]
    if lcfg["cldice"]["enabled"]:
        terms["cldice"] = masked_cldice(prob, target, valid, lcfg["cldice"]["iters"])
        total = total + lcfg["cldice"]["weight"] * terms["cldice"]
    terms["total"] = total
    return terms


def masked_hard_dice(prob, label) -> float:
    """Hard Dice at 0.5 over non-ignore px (numpy arrays); 1.0 if both are empty."""
    valid = label != IGNORE
    p, t = (prob > 0.5) & valid, (label == 1) & valid
    den = p.sum() + t.sum()
    return float(2 * (p & t).sum() / den) if den else 1.0
