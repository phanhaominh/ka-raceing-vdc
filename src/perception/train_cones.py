"""Fine-tune the Waymo PointPillars backbone on FSAE cone data (coneScenes).

Loads the coneScenes sample dataset (points/*.bin (N,4) float32
[x, y, z, intensity] + labels/*.txt lines ``x y z dx dy dz yaw class_name``),
pillarizes with the same vectorized ``build_pillars`` as the Waymo loader,
initializes the model from the trained Waymo checkpoint
(``runs/waymo_v100/checkpoints/epoch_100.pth``; cls_head is re-initialized for
4 cone classes, everything else is loaded), and fine-tunes with the SAME
focal + smooth-L1 + per-class-anchor loss/target logic as ``train.py`` /
``train_numpy.py`` (anchors are overridden to cone sizes).

Run from the project root inside the container on gn03 (4x V100)::

    apptainer exec --nv \\
        --bind /gpfs/data/gpfs0/aphan_group/ka_raceing_vdc:/project \\
        /beegfs/shared/singularity-images/NGC/pytorch_23.10-py3.simg \\
        bash -lc 'cd /project && python3 -m src.perception.train_cones \\
            --data-dir /project/data/conescenes_sample/vargarda8 \\
            --checkpoint /project/runs/waymo_v100/checkpoints/epoch_100.pth \\
            --batch-size 4 --epochs 50 --num-workers 4'

Checkpoints are saved every ``--checkpoint-every`` (10) epochs to
``runs/cone_finetune/checkpoints/``; TensorBoard logs to ``runs/cone_finetune/tb/``.
"""

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # train.py pulls waymo_loader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.perception.pointpillars import PointPillars              # noqa: E402
from src.perception.waymo_loader import (                         # noqa: E402
    build_pillars,
    GRID_X, GRID_Y, GRID_Z, PILLAR_SIZE,
    MAX_PILLARS, MAX_POINTS_PER_PILLAR,
)
from src.perception.numpy_loader import collate_fn                # noqa: E402
import src.perception.train as train_mod                          # noqa: E402

# ---------------------------------------------------------------------------
# Cone classes / anchors (labels contain Cone_Yellow + Cone_Blue in this
# sample; Orange/Big have no samples but are kept as classes per the spec).
# ---------------------------------------------------------------------------
CONE_CLASSES = ["Cone_Yellow", "Cone_Blue", "Cone_Orange", "Cone_Big"]
CONE_ANCHORS = {
    0: {"l": 0.23, "w": 0.23, "h": 0.33},   # Cone_Yellow (measured dx=dy=0.23, dz=0.33)
    1: {"l": 0.23, "w": 0.23, "h": 0.33},   # Cone_Blue
    2: {"l": 0.23, "w": 0.23, "h": 0.33},   # Cone_Orange
    3: {"l": 0.35, "w": 0.35, "h": 0.50},   # Cone_Big (no samples here; larger)
}
# build_targets/detection_loss read train_mod.CLASS_ANCHORS and
# train_mod.NUM_CLASSES at call time.
train_mod.CLASS_ANCHORS = CONE_ANCHORS
train_mod.NUM_CLASSES = len(CONE_CLASSES)


class ConeDataset(Dataset):
    """coneScenes dataset: points/*.bin + labels/*.txt -> pillars."""

    def __init__(self, data_dir, augment=True):
        self.points_dir = os.path.join(data_dir, "points")
        self.labels_dir = os.path.join(data_dir, "labels")
        stems = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(self.points_dir) if f.endswith(".bin"))
        self.frames = [s for s in stems
                       if os.path.exists(os.path.join(self.labels_dir, s + ".txt"))]
        assert self.frames, f"no labeled .bin/.txt pairs under {data_dir}"
        self.augment = augment
        print(f"[ConeDataset] {len(self.frames)} labeled frames from {data_dir}",
              flush=True)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        stem = self.frames[idx]
        pts = np.fromfile(os.path.join(self.points_dir, stem + ".bin"),
                          dtype=np.float32).reshape(-1, 4)
        boxes, labels = self._read_labels(stem)
        if self.augment:
            pts, boxes = self._augment(pts, boxes)
        pillars, indices, _n = build_pillars(
            pts[:, :3], pts[:, 3],
            max_pillars=MAX_PILLARS, max_points_per_pillar=MAX_POINTS_PER_PILLAR,
            grid_x=GRID_X, grid_y=GRID_Y, grid_z=GRID_Z, pillar_size=PILLAR_SIZE)
        return (torch.from_numpy(pillars),
                torch.from_numpy(indices),
                torch.from_numpy(boxes),
                torch.from_numpy(labels))

    def _read_labels(self, stem):
        boxes, labels = [], []
        with open(os.path.join(self.labels_dir, stem + ".txt")) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 8:
                    continue
                x, y, z, dx, dy, dz, yaw, name = parts
                if name not in CONE_CLASSES:
                    continue
                boxes.append([float(x), float(y), float(z),
                              float(dx), float(dy), float(dz), float(yaw)])
                labels.append(CONE_CLASSES.index(name))
        if boxes:
            return (np.asarray(boxes, dtype=np.float32),
                    np.asarray(labels, dtype=np.int64))
        return (np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))

    def _augment(self, pts, boxes):
        """Random rotation around z (+-10 deg) and random x-flip, applied to
        both points and box centers/headings so pillars and targets stay
        consistent."""
        angle = (random.random() * 2.0 - 1.0) * math.radians(10.0)
        c, s = math.cos(angle), math.sin(angle)
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        pts[:, 0] = c * x - s * y
        pts[:, 1] = s * x + c * y
        if boxes.shape[0]:
            bx, by = boxes[:, 0].copy(), boxes[:, 1].copy()
            boxes[:, 0] = c * bx - s * by
            boxes[:, 1] = s * bx + c * by
            boxes[:, 6] = boxes[:, 6] + angle
        if random.random() < 0.5:
            pts[:, 0] = -pts[:, 0]
            if boxes.shape[0]:
                boxes[:, 0] = -boxes[:, 0]
                boxes[:, 6] = -boxes[:, 6]
        return pts, boxes


def load_backbone(model, ckpt_path, device):
    """Load all matching tensors from the Waymo checkpoint (skips cls_head,
    whose output channels differ: 3 Waymo classes -> 4 cone classes)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    own = model.state_dict()
    loaded, skipped = [], []
    for k, v in sd.items():
        k2 = k.replace("module.", "")
        if k2 in own and own[k2].shape == v.shape:
            own[k2] = v
            loaded.append(k2)
        else:
            skipped.append((k2, tuple(v.shape),
                            tuple(own[k2].shape) if k2 in own else None))
    model.load_state_dict(own)
    print(f"[backbone] loaded {len(loaded)}/{len(sd)} tensors from {ckpt_path}",
          flush=True)
    for k, vs, os_ in skipped:
        print(f"  skipped {k}: ckpt{vs} -> model{os_}", flush=True)
    return len(loaded), len(skipped)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/project/data/conescenes_sample/vargarda8")
    ap.add_argument("--checkpoint",
                    default=os.path.join(PROJECT_ROOT, "runs", "waymo_v100",
                                         "checkpoints", "epoch_100.pth"))
    ap.add_argument("--val-frames", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--freeze-pfn", action="store_true", default=True,
                    help="freeze PFN weights (recommended for 63 frames)")
    ap.add_argument("--augment", action="store_true", default=True,
                    help="random rotation + flip augmentation")
    ap.add_argument("--checkpoint-dir",
                    default=os.path.join(PROJECT_ROOT, "runs", "cone_finetune",
                                         "checkpoints"))
    ap.add_argument("--log-dir",
                    default=os.path.join(PROJECT_ROOT, "runs", "cone_finetune",
                                         "tb"))
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count()
    print(f"device={device} gpus={n_gpus}", flush=True)
    if device == "cuda" and args.batch_size % max(n_gpus, 1) != 0:
        print(f"WARNING: batch_size {args.batch_size} not divisible by "
              f"{n_gpus} GPUs", flush=True)

    # --- data --------------------------------------------------------------
    t0 = time.time()
    full_ds = ConeDataset(args.data_dir, augment=args.augment)
    n_val = min(args.val_frames, max(1, len(full_ds) // 5))
    n_train = len(full_ds) - n_val
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)
    print(f"[data] train={len(train_ds)} frames, val={len(val_ds)} frames "
          f"({time.time() - t0:.0f}s)", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers,
                              pin_memory=(device == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers,
                            pin_memory=(device == "cuda"))

    # --- model: Waymo backbone + new 4-class cls_head -----------------------
    model = PointPillars(num_classes=len(CONE_CLASSES)).to(device)
    n_loaded, n_skip = load_backbone(model, args.checkpoint, device)
    assert n_skip <= 2, f"expected only cls_head to be skipped, got {n_skip}"
    if args.freeze_pfn:
        for p in model.pfn.parameters():
            p.requires_grad = False
        print("[model] PFN frozen", flush=True)

    fh, fw = train_mod.output_grid(model, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] params={total:,} trainable={trainable:,} "
          f"output_grid={fh}x{fw}", flush=True)

    if device == "cuda" and n_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = min(len(train_loader),
                          args.max_batches or len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=max(1, steps_per_epoch * args.epochs))

    start_epoch = 0
    if args.resume:
        start_epoch = train_mod.load_checkpoint(
            args.resume, model, optimizer, scheduler, device) + 1
        print(f"[resume] resumed from epoch {start_epoch}", flush=True)

    writer = SummaryWriter(log_dir=args.log_dir)

    # --- fine-tune loop ------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        t_ep = time.time()
        tr_l, tr_c, tr_r = train_mod.train_epoch(
            model, train_loader, optimizer, scheduler, device, fh, fw,
            max_batches=args.max_batches, writer=writer, epoch=epoch)
        va_l, va_c, va_r = train_mod.validate(
            model, val_loader, device, fh, fw, max_batches=args.max_batches)
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

        if (epoch + 1) % args.checkpoint_every == 0:
            ckpt = os.path.join(args.checkpoint_dir, f"epoch_{epoch + 1}.pth")
            train_mod.save_checkpoint(ckpt, model, optimizer, scheduler,
                                      epoch, args)

    writer.close()
    print("[done] cone fine-tuning finished", flush=True)


if __name__ == "__main__":
    main()
