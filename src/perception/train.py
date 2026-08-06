"""Multi-GPU PointPillars training on Waymo Open Dataset v1.3.0.

Run from the project root (so ``src`` is importable)::

    python3 -m src.perception.train \
        --data-dir /waymo_data \
        --batch-size 4 \
        --epochs 30

Inside the container on gn34, the dataset is bound at ``/waymo_data`` and the
project at ``/project``::

    apptainer exec --nv \\
        --bind /gpfs/data/gpfs0/datasets/waymo_open_dataset_v_1_3_0:/waymo_data \\
        --bind /gpfs/data/gpfs0/aphan_group/ka_raceing_vdc:/project \\
        /beegfs/shared/singularity-images/NGC/pytorch_23.10-py3.simg \\
        bash -lc 'cd /project && python3 -m src.perception.train --data-dir /waymo_data'

Loss design (matches the model's dense pseudo-image output):
  * classification: sigmoid focal loss (alpha=0.25, gamma=2.0) over the
    ``(B, 3, fh, fw)`` grid; a feature cell is positive for class c when it
    contains >= 1 occupied pillar whose cell center falls inside a GT box of
    class c (boxes are matched to pillars via ``pillar_indices``, then mapped
    to the pseudo-image layout the model uses).
  * regression: smooth L1 (beta=1/9) on the 7 channels
    ``(dx, dy, dz, dl, dw, dh, dyaw)`` relative to per-class anchor dims and
    the matched pillar cell center, masked to positive cells only.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# Allow running as a plain script from anywhere in the repo.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.perception.pointpillars import PointPillars          # noqa: E402
from src.perception.waymo_loader import (                     # noqa: E402
    WaymoPillarDataset, collate_fn,
    GRID_X, GRID_Y, PILLAR_SIZE, MAX_PILLARS, MAX_POINTS_PER_PILLAR,
)

CLASS_NAMES = ["VEHICLE", "PEDESTRIAN", "CYCLIST"]

# Per-class anchor dims used to normalize regression targets.
CLASS_ANCHORS = {
    0: {"l": 4.6, "w": 2.0, "h": 1.6},   # VEHICLE
    1: {"l": 0.9, "w": 0.8, "h": 1.8},   # PEDESTRIAN
    2: {"l": 1.8, "w": 0.7, "h": 1.5},   # CYCLIST
}


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Binary sigmoid focal loss (SECOND/PointPillars style), mean-reduced."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * ((1.0 - p_t) ** gamma) * bce
    return loss.mean()


def output_grid(model, device, max_pillars=MAX_PILLARS,
                n_points=MAX_POINTS_PER_PILLAR):
    """Run one dummy forward on the unwrapped model to get (fh, fw)."""
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, max_pillars, n_points, 9, device=device)
        cls, _reg = model(dummy)
    return int(cls.shape[-2]), int(cls.shape[-1])


def build_targets(pillar_indices, boxes, labels, box_mask, fh, fw):
    """Map GT boxes onto the model's dense (B, 3, fh, fw) / (B, 7, fh, fw) grid.

    The model scatters tensor position k to pseudo-image cell
    ``(k // W, k % W)`` with ``W = int(MAX_PILLARS ** 0.5) + 1`` and the
    backbone downsamples by ``downsample`` (two stride-2 convs).  For every GT
    box we mark every occupied pillar cell whose center is inside the (rotated)
    box, map those cells to feature cells, and assign class + regression
    target (first box wins per cell).

    Returns ``(cls_target, reg_target, reg_mask)`` with shapes
    ``(B, 3, fh, fw)``, ``(B, 7, fh, fw)``, ``(B, 1, fh, fw)``.
    """
    W = int(MAX_PILLARS ** 0.5) + 1
    downsample = max(1, math.ceil(W / fh))
    x_min, y_min = GRID_X[0], GRID_Y[0]
    px, py = PILLAR_SIZE

    B = pillar_indices.shape[0]
    cls_t = torch.zeros(B, 3, fh, fw, device=pillar_indices.device)
    reg_t = torch.zeros(B, 7, fh, fw, device=pillar_indices.device)
    reg_m = torch.zeros(B, 1, fh, fw, device=pillar_indices.device)

    for b in range(B):
        idx = pillar_indices[b]                     # (P, 2)
        occ = idx[:, 0] >= 0
        if not occ.any():
            continue
        gx = idx[occ, 0].long()
        gy = idx[occ, 1].long()
        k = torch.nonzero(occ).squeeze(1)           # tensor position of pillar
        cx = x_min + (gx.float() + 0.5) * px        # cell centers (vehicle frame)
        cy = y_min + (gy.float() + 0.5) * py
        r = k // W
        c = k % W
        fr = torch.clamp(r // downsample, 0, fh - 1)
        fc = torch.clamp(c // downsample, 0, fw - 1)
        assigned = torch.zeros(fh, fw, dtype=torch.bool,
                               device=pillar_indices.device)

        for j in range(boxes.shape[1]):
            if not box_mask[b, j]:
                continue
            box = boxes[b, j]
            cls = int(labels[b, j])
            bx, by, bz = box[0].item(), box[1].item(), box[2].item()
            w_, l_, h_ = box[3].item(), box[4].item(), box[5].item()
            hdg = box[6].item()
            cos_h, sin_h = math.cos(hdg), math.sin(hdg)
            # pillar cell center in the box's local frame
            dx, dy = cx - bx, cy - by
            u = dx * cos_h + dy * sin_h
            v = -dx * sin_h + dy * cos_h
            inside = (u.abs() <= l_ / 2.0) & (v.abs() <= w_ / 2.0)
            mfr, mfc = fr[inside], fc[inside]
            mcx, mcy = cx[inside], cy[inside]
            free = ~assigned[mfr, mfc]
            frr, fcc = mfr[free], mfc[free]
            if frr.numel() == 0:
                continue
            assigned[frr, fcc] = True

            anc = CLASS_ANCHORS[cls]
            cls_t[b, cls, frr, fcc] = 1.0
            reg_t[b, 0, frr, fcc] = (bx - mcx[free]) / anc["l"]
            reg_t[b, 1, frr, fcc] = (by - mcy[free]) / anc["w"]
            reg_t[b, 2, frr, fcc] = bz / anc["h"]           # cell z assumed 0
            reg_t[b, 3, frr, fcc] = math.log(max(l_, 1e-3) / anc["l"])
            reg_t[b, 4, frr, fcc] = math.log(max(w_, 1e-3) / anc["w"])
            reg_t[b, 5, frr, fcc] = math.log(max(h_, 1e-3) / anc["h"])
            reg_t[b, 6, frr, fcc] = hdg
            reg_m[b, 0, frr, fcc] = 1.0

    return cls_t, reg_t, reg_m


def detection_loss(cls_pred, reg_pred, pillar_indices, boxes, labels,
                   box_mask, fh, fw, reg_weight=2.0):
    """Total detection loss: focal (cls) + smooth-L1 (reg) on positives."""
    cls_t, reg_t, reg_m = build_targets(pillar_indices, boxes, labels,
                                        box_mask, fh, fw)
    cls_loss = sigmoid_focal_loss(cls_pred, cls_t)
    n_pos = reg_m.sum().clamp(min=1.0)
    reg_loss = (F.smooth_l1_loss(reg_pred * reg_m, reg_t * reg_m,
                                 beta=1.0 / 9.0, reduction="sum")
                / n_pos)
    loss = cls_loss + reg_weight * reg_loss
    return loss, cls_loss, reg_loss, n_pos


def train_epoch(model, loader, optimizer, scheduler, device, fh, fw,
                max_batches=None, writer=None, epoch=0):
    model.train()
    total, n_cls, n_reg, steps = 0.0, 0.0, 0.0, 0
    for i, (pillars, indices, boxes, labels, box_mask) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        pillars = pillars.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        box_mask = box_mask.to(device, non_blocking=True)

        cls_pred, reg_pred = model(pillars)
        loss, cls_l, reg_l, _n_pos = detection_loss(
            cls_pred, reg_pred, indices, boxes, labels, box_mask, fh, fw)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total += loss.item()
        n_cls += cls_l.item()
        n_reg += reg_l.item()
        steps += 1
        if writer is not None and i % 20 == 0:
            writer.add_scalar("train/loss_step", loss.item(),
                              epoch * len(loader) + i)
        if i % 10 == 0:
            print(f"    step {i}: loss={loss.item():.4f} "
                  f"cls={cls_l.item():.4f} reg={reg_l.item():.4f}", flush=True)
    return total / max(steps, 1), n_cls / max(steps, 1), n_reg / max(steps, 1)


@torch.no_grad()
def validate(model, loader, device, fh, fw, max_batches=None):
    model.eval()
    total, n_cls, n_reg, steps = 0.0, 0.0, 0.0, 0
    for i, (pillars, indices, boxes, labels, box_mask) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        pillars = pillars.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        box_mask = box_mask.to(device, non_blocking=True)
        cls_pred, reg_pred = model(pillars)
        loss, cls_l, reg_l, _n = detection_loss(
            cls_pred, reg_pred, indices, boxes, labels, box_mask, fh, fw)
        total += loss.item()
        n_cls += cls_l.item()
        n_reg += reg_l.item()
        steps += 1
    model.train()
    return total / max(steps, 1), n_cls / max(steps, 1), n_reg / max(steps, 1)


def save_checkpoint(path, model, optimizer, scheduler, epoch, args):
    state = {
        "epoch": epoch,
        "model_state_dict": (model.module.state_dict()
                             if isinstance(model, nn.DataParallel)
                             else model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[checkpoint] saved {path}", flush=True)


def load_checkpoint(path, model, optimizer, scheduler, device):
    state = torch.load(path, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    return state["epoch"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/waymo_data",
                    help="Waymo dataset root (container mount /waymo_data)")
    ap.add_argument("--split-train", default="training")
    ap.add_argument("--split-val", default="validation")
    ap.add_argument("--max-files-train", type=int, default=None)
    ap.add_argument("--max-files-val", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames per split (quick tests)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="cap batches per epoch (quick tests)")
    ap.add_argument("--checkpoint-dir",
                    default=os.path.join(PROJECT_ROOT, "runs", "pointpillars",
                                         "checkpoints"))
    ap.add_argument("--log-dir",
                    default=os.path.join(PROJECT_ROOT, "runs", "pointpillars",
                                         "tb"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count()
    print(f"device={device} gpus={n_gpus}", flush=True)
    if device == "cuda" and args.batch_size % max(n_gpus, 1) != 0:
        print(f"WARNING: batch_size {args.batch_size} not divisible by "
              f"{n_gpus} GPUs — DataParallel will receive uneven splits.",
              flush=True)

    # --- data --------------------------------------------------------------
    t0 = time.time()
    train_ds = WaymoPillarDataset(args.data_dir, split=args.split_train,
                                  max_files=args.max_files_train,
                                  max_frames=args.max_frames)
    val_ds = WaymoPillarDataset(args.data_dir, split=args.split_val,
                                max_files=args.max_files_val,
                                max_frames=args.max_frames)
    print(f"[data] train={len(train_ds)} frames, val={len(val_ds)} frames "
          f"({time.time() - t0:.0f}s)", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=args.num_workers,
                              pin_memory=(device == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=args.num_workers,
                            pin_memory=(device == "cuda"))

    # --- model --------------------------------------------------------------
    model = PointPillars(num_classes=len(CLASS_NAMES)).to(device)
    fh, fw = output_grid(model, device)
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,} "
          f"output_grid={fh}x{fw}", flush=True)
    if device == "cuda" and n_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    steps_per_epoch = min(len(train_loader),
                          args.max_batches or len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=max(1, steps_per_epoch * args.epochs))

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer,
                                      scheduler, device) + 1
        print(f"[resume] resumed from epoch {start_epoch}", flush=True)

    writer = SummaryWriter(log_dir=args.log_dir)

    # --- loop ----------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        t_ep = time.time()
        tr_l, tr_c, tr_r = train_epoch(
            model, train_loader, optimizer, scheduler, device, fh, fw,
            max_batches=args.max_batches, writer=writer, epoch=epoch)
        va_l, va_c, va_r = validate(model, val_loader, device, fh, fw,
                                    max_batches=args.max_batches)
        lr = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch + 1}/{args.epochs}] "
              f"train loss={tr_l:.4f} (cls {tr_c:.4f}, reg {tr_r:.4f}) | "
              f"val loss={va_l:.4f} (cls {va_c:.4f}, reg {va_r:.4f}) | "
              f"lr={lr:.2e} | {time.time() - t_ep:.0f}s", flush=True)
        writer.add_scalar("train/loss", tr_l, epoch)
        writer.add_scalar("train/cls_loss", tr_c, epoch)
        writer.add_scalar("train/reg_loss", tr_r, epoch)
        writer.add_scalar("val/loss", va_l, epoch)
        writer.add_scalar("val/cls_loss", va_c, epoch)
        writer.add_scalar("val/reg_loss", va_r, epoch)
        writer.add_scalar("train/lr", lr, epoch)

        ckpt = os.path.join(args.checkpoint_dir, f"epoch_{epoch + 1}.pth")
        save_checkpoint(ckpt, model, optimizer, scheduler, epoch, args)

    writer.close()
    print("[done] training finished", flush=True)


if __name__ == "__main__":
    main()
