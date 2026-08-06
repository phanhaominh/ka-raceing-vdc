"""PointPillars training on the pre-pillarized numpy Waymo cache.

Adapted from ``train.py``: identical model, loss (sigmoid focal + smooth L1,
per-class-anchor regression targets on the model's 28x28 grid) and loop — only
the data pipeline changes, from TFRecord reading to fast numpy npz loads.

Run from the project root inside the container on gn03 (4x V100)::

    apptainer exec --nv \\
        --bind /gpfs/data/gpfs0/aphan_group/ka_raceing_vdc:/project \\
        /beegfs/shared/singularity-images/NGC/pytorch_23.10-py3.simg \\
        bash -lc 'cd /project && python3 -m src.perception.train_numpy \\
            --data-dir /project/data/waymo_numpy_pillars \\
            --batch-size 4 --epochs 100 --num-workers 4'

Checkpoints are saved every ``--checkpoint-every`` epochs to
``runs/waymo_v100/checkpoints/``; TensorBoard logs go to ``runs/waymo_v100/tb/``.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # train.py pulls waymo_loader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.perception.pointpillars import PointPillars          # noqa: E402
from src.perception.numpy_loader import (                     # noqa: E402
    NumpyPillarDataset, collate_fn,
)
from src.perception.train import (                            # noqa: E402
    CLASS_NAMES,
    detection_loss, output_grid, train_epoch, validate,
    save_checkpoint, load_checkpoint,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/project/data/waymo_numpy_pillars",
                    help="dir with frame_*.npz (pillars/indices/boxes/labels)")
    ap.add_argument("--val-frames", type=int, default=30,
                    help="number of frames held out for validation")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="cap batches per epoch (quick tests)")
    ap.add_argument("--checkpoint-dir",
                    default=os.path.join(PROJECT_ROOT, "runs", "waymo_v100",
                                         "checkpoints"))
    ap.add_argument("--log-dir",
                    default=os.path.join(PROJECT_ROOT, "runs", "waymo_v100",
                                         "tb"))
    ap.add_argument("--checkpoint-every", type=int, default=10)
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
    full_ds = NumpyPillarDataset(args.data_dir)
    n_val = min(args.val_frames, max(1, len(full_ds) // 5))
    n_train = len(full_ds) - n_val
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)
    print(f"[data] train={len(train_ds)} frames, val={len(val_ds)} frames "
          f"({time.time() - t0:.0f}s)", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=args.num_workers,
                              pin_memory=(device == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=args.num_workers,
                            pin_memory=(device == "cuda"))

    # --- model ---------------------------------------------------------------
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

        if (epoch + 1) % args.checkpoint_every == 0:
            ckpt = os.path.join(args.checkpoint_dir, f"epoch_{epoch + 1}.pth")
            save_checkpoint(ckpt, model, optimizer, scheduler, epoch, args)

    writer.close()
    print("[done] training finished", flush=True)


if __name__ == "__main__":
    main()
