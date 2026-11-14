"""unet_v1 Phase 2: train the U-Net student on the v1e pseudo-labels.

smp.Unet(resnet34, imagenet, in_channels=1, classes=1) on random 256x256
patches of the train frames (unet/dataset.py), masked BCE + soft Dice
(unet/losses.py), AdamW + per-step cosine schedule, bf16 autocast.

Pseudo-val frames (frame-level split, stratified by magnification tier) are
used ONLY for early stopping and checkpoint selection. Each val frame is
predicted tile by tile, stitched, and scored over its non-ignore px; the
reported numbers are MEAN OF PER-FRAME masked loss / masked Dice (hard, at
0.5). GT frames are never read.

Writes results/unet_v1/train/<run_name>/: config.yaml (the YAML as given),
effective_config.json (after CLI overrides), provenance.json, split.json,
history.json, metrics.csv, loss_curves.png, panels/*.png, tb/ (TensorBoard),
best.pt (best pseudo-val masked Dice), last.pt, summary.json.

Run via slurm/unet_train.sbatch (GPU). The diagnostic is --epochs 5.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import segmentation_models_pytorch as smp  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from unet.common import IGNORE, ROOT, load_config, rel, write_provenance  # noqa: E402
from unet.dataset import (NORM_MEAN, NORM_STD, PatchDataset, load_frames,  # noqa: E402
                          split_frames, stitch_tiles, val_tiles)
from unet.losses import masked_hard_dice, masked_loss  # noqa: E402

AGG = "mean of per-frame"           # label for every pseudo-val number


def build_model(mcfg: dict, weights: str | None = "imagenet") -> torch.nn.Module:
    return smp.Unet(encoder_name=mcfg["encoder"], encoder_weights=weights,
                    in_channels=1, classes=1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_frame(model, frame, crop: int, dev, amp_dtype, batch: int) -> np.ndarray:
    """Stitched sigmoid probability for a whole frame (non-overlapping tiles)."""
    x, pos = val_tiles(frame, crop)
    out = []
    for i in range(0, len(x), batch):
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out.append(model(x[i:i + batch].to(dev)).float().cpu())
    logits = torch.cat(out)[:, 0].numpy()
    return stitch_tiles(logits, pos, frame.img.shape)


def evaluate(model, frames, crop, dev, amp_dtype, lcfg, batch) -> dict:
    model.eval()
    per = {}
    for f in frames:
        logits = predict_frame(model, f, crop, dev, amp_dtype, batch)
        lab = torch.from_numpy(f.label.astype(np.int64))[None, None]
        loss = masked_loss(torch.from_numpy(logits)[None, None], lab, lcfg)
        prob = 1 / (1 + np.exp(-logits))
        per[f.name] = {"loss": float(loss["total"]), "bce": float(loss["bce"]),
                       "soft_dice_loss": float(loss["dice"]),
                       "dice": masked_hard_dice(prob, f.label),
                       "pred_wire_frac_valid": float((prob[f.label != IGNORE] > 0.5).mean()),
                       "label_wire_frac_valid": float((f.label[f.label != IGNORE] == 1).mean()),
                       "_prob": prob}
    return per


def panel(frame, prob: np.ndarray, size: int) -> np.ndarray:
    """image | pseudo-label (wire white, ignore gray, bg black) | prediction,
    on a centred size x size crop."""
    H, W = frame.img.shape
    r, c = max((H - size) // 2, 0), max((W - size) // 2, 0)
    sl = (slice(r, r + size), slice(c, c + size))
    img = (np.clip(frame.img[sl], 0, 1) * 255).astype(np.uint8)
    lab = np.where(frame.label[sl] == IGNORE, 128, frame.label[sl] * 255).astype(np.uint8)
    pr = (prob[sl] * 255).astype(np.uint8)
    sep = np.full((img.shape[0], 6), 255, np.uint8)
    return np.concatenate([img, sep, lab, sep, pr], axis=1)


def plot_curves(hist: list[dict], path: Path) -> None:
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in hist], "o-", label="train (masked)")
    ax[0].plot(ep, [h["val_loss"] for h in hist], "s-", label=f"pseudo-val (masked, {AGG})")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("BCE + soft Dice"); ax[0].legend()
    ax[1].plot(ep, [h["val_dice"] for h in hist], "s-", color="C2")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel(f"pseudo-val masked Dice\n({AGG}, thr 0.5)")
    for a in ax:
        a.grid(alpha=0.3)
        a.set_xticks(ep)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "unet_v1.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    ap.add_argument("--run-name", default=None, help="override train.run_name")
    args = ap.parse_args()
    cfg = load_config(args.config)
    eff = copy.deepcopy(cfg)
    tcfg = eff["train"]
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    if args.run_name is not None:
        tcfg["run_name"] = args.run_name
    seed_everything(cfg["seed"])

    label_dir = ROOT / tcfg["pseudo_label_dir"]
    pl_commit = json.loads((label_dir / "provenance.json").read_text())["git"]["commit"]
    assert pl_commit.startswith(tcfg["expected_pseudo_label_commit"]), \
        f"pseudo-labels in {rel(label_dir)} are from {pl_commit}, expected " \
        f"{tcfg['expected_pseudo_label_commit']}"

    out = ROOT / cfg["paths"]["out_root"] / "train" / tcfg["run_name"]
    (out / "panels").mkdir(parents=True, exist_ok=True)
    prov = write_provenance(out, args.config, {"pseudo_label_commit": pl_commit,
                                               "cli_overrides": {"epochs": args.epochs,
                                                                 "run_name": args.run_name}})
    (out / "effective_config.json").write_text(json.dumps(eff, indent=2))

    frames = load_frames(label_dir)
    sp = split_frames({f.name: f.tier for f in frames}, tcfg["split"]["val_frac"],
                      tcfg["split"]["seed"], cfg["frames"]["exclude_from_training"])
    (out / "split.json").write_text(json.dumps(sp, indent=2))
    train_f = [f for f in frames if f.name in sp["train"]]
    val_f = [f for f in frames if f.name in sp["val"]]
    print(f"train {len(train_f)} frames, pseudo-val {len(val_f)}: {sp['val']}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = getattr(torch, tcfg["amp_dtype"]) if tcfg["amp"] and dev.type == "cuda" else None
    model = build_model(tcfg["model"]).to(dev)
    torch.backends.cudnn.benchmark = True

    dcfg, bs = tcfg["data"], tcfg["batch_size"]
    assert dcfg["samples_per_epoch"] % bs == 0 and bs % 2 == 0
    ds = PatchDataset(train_f, dcfg["crop"], dcfg["samples_per_epoch"], cfg["seed"], dcfg,
                      tcfg["augment"])
    steps = dcfg["samples_per_epoch"] // bs
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["epochs"] * steps)
    tb = SummaryWriter(out / "tb")

    hist, best, since_best = [], -math.inf, 0
    for epoch in range(1, tcfg["epochs"] + 1):
        ds.set_epoch(epoch)
        # shuffle=False keeps each batch half wire-rich (index parity); every
        # epoch draws new patches through set_epoch
        dl = DataLoader(ds, batch_size=bs, shuffle=False, drop_last=True,
                        num_workers=tcfg["num_workers"], pin_memory=dev.type == "cuda")
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        t0 = time.time()
        losses = []
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x)
            loss = masked_loss(logits, y, tcfg["loss"])["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_train = time.time() - t0

        t0 = time.time()
        per = evaluate(model, val_f, dcfg["crop"], dev, amp_dtype, tcfg["loss"], bs)
        t_val = time.time() - t0
        mem = ({"max_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
                "max_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}
               if dev.type == "cuda" else {})
        h = {"epoch": epoch, "train_loss": float(np.mean(losses)),
             "val_loss": float(np.mean([p["loss"] for p in per.values()])),
             "val_dice": float(np.mean([p["dice"] for p in per.values()])),
             "val_per_frame": {n: {k: v for k, v in p.items() if not k.startswith("_")}
                               for n, p in per.items()},
             "lr_end": sched.get_last_lr()[0], "steps": len(losses),
             "sec_train": round(t_train, 1), "sec_val": round(t_val, 1), **mem}
        hist.append(h)

        tb.add_scalar("loss/train_masked", h["train_loss"], epoch)
        tb.add_scalar("loss/val_masked_mean_per_frame", h["val_loss"], epoch)
        tb.add_scalar("dice/val_masked_mean_per_frame", h["val_dice"], epoch)
        tb.add_scalar("lr", h["lr_end"], epoch)
        if epoch == 1 or epoch % tcfg["panel_every"] == 0 or epoch == tcfg["epochs"]:
            for f in val_f:
                pan = panel(f, per[f.name]["_prob"], tcfg["panel_size"])
                Image.fromarray(pan).save(out / "panels" / f"epoch{epoch:03d}_{f.name}.png")
                tb.add_image(f"panel/{f.name}", pan[None], epoch)

        ck = {"model": model.state_dict(), "epoch": epoch, "val_dice": h["val_dice"],
              "config": eff, "split": sp, "norm": [NORM_MEAN, NORM_STD]}
        torch.save(ck, out / "last.pt")
        if h["val_dice"] > best:
            best, since_best = h["val_dice"], 0
            torch.save(ck, out / "best.pt")
        else:
            since_best += 1
        print(f"epoch {epoch:3d}  train {h['train_loss']:.4f}  val loss {h['val_loss']:.4f}  "
              f"val Dice ({AGG}) {h['val_dice']:.4f}  {t_train:.0f}s train + {t_val:.0f}s val  "
              f"GPU max alloc {mem.get('max_allocated_gb', float('nan')):.2f} GB", flush=True)
        (out / "history.json").write_text(json.dumps(hist, indent=2))
        if since_best >= tcfg["early_stop_patience"]:
            print(f"early stop: no pseudo-val Dice gain in {since_best} epochs", flush=True)
            break
    tb.close()

    with open(out / "metrics.csv", "w", newline="") as fh:
        keys = [k for k in hist[0] if k != "val_per_frame"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(hist)
    plot_curves(hist, out / "loss_curves.png")
    bi = int(np.argmax([h["val_dice"] for h in hist]))
    summary = {"run_name": tcfg["run_name"], "epochs_run": len(hist),
               "aggregation": f"pseudo-val loss and Dice are the {AGG} value over val frames",
               "best_epoch": hist[bi]["epoch"], "best_val_dice": hist[bi]["val_dice"],
               "last_val_dice": hist[-1]["val_dice"],
               "sec_per_epoch_train_median": float(np.median([h["sec_train"] for h in hist])),
               "sec_per_epoch_val_median": float(np.median([h["sec_val"] for h in hist])),
               "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
               "gpu_max_allocated_gb": max((h.get("max_allocated_gb", 0) for h in hist)),
               "gpu_max_reserved_gb": max((h.get("max_reserved_gb", 0) for h in hist)),
               "split": sp, "pseudo_label_commit": pl_commit, "git": prov["git"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "split"}, indent=1))


if __name__ == "__main__":
    main()
