"""Smoke test for the Waymo loader + PointPillars training stack.

Verifies, on 1 real TFRecord file:
  1. dataset indexing: ``ds[0]`` shapes (pillars (P, N, 9), indices (P, 2),
     boxes (B, 7), labels (B,))
  2. batch collation: (B, P, N, 9) / (B, P, 2) / (B, maxB, 7) + box mask
  3. multi-GPU forward pass: cls (B, 3, fh, fw), reg (B, 7, fh, fw)
  4. loss decreases over a few overfit epochs on a 4-sample subset

Run inside the container on gn34 (4x A100)::

    apptainer exec --nv \\
        --bind /gpfs/data/gpfs0/datasets/waymo_open_dataset_v_1_3_0:/waymo_data \\
        --bind /gpfs/data/gpfs0/aphan_group/ka_raceing_vdc:/project \\
        /beegfs/shared/singularity-images/NGC/pytorch_23.10-py3.simg \\
        bash -lc 'cd /project && python3 -m src.perception.test_loader \
            --data-dir /waymo_data --max-files 1 --epochs 15'
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.perception.pointpillars import PointPillars                # noqa: E402
from src.perception.waymo_loader import (                           # noqa: E402
    WaymoPillarDataset, collate_fn, pillar_counts,
    MAX_PILLARS, MAX_POINTS_PER_PILLAR,
)
from src.perception.train import (                                  # noqa: E402
    detection_loss, output_grid, CLASS_NAMES,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/waymo_data")
    ap.add_argument("--max-files", type=int, default=1)
    ap.add_argument("--n-samples", type=int, default=4,
                    help="samples for the batch-collation test")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15,
                    help="overfit epochs for the loss-decrease check")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = max(1, torch.cuda.device_count())
    assert args.batch_size % n_gpus == 0, (
        f"batch_size {args.batch_size} must be divisible by {n_gpus} GPUs")
    print(f"== smoke test: device={device} gpus={n_gpus} "
          f"data={args.data_dir}", flush=True)

    # -- 1. dataset + sample shapes -----------------------------------------
    t0 = time.time()
    ds = WaymoPillarDataset(args.data_dir, split="training",
                            max_files=args.max_files)
    print(f"[1] dataset ready: {len(ds)} frames in {len(ds.files)} file(s) "
          f"({time.time() - t0:.0f}s)", flush=True)

    pillars, indices, boxes, labels = ds[0]
    print(f"[2] sample 0: pillars={tuple(pillars.shape)} "
          f"indices={tuple(indices.shape)} boxes={tuple(boxes.shape)} "
          f"labels={tuple(labels.shape)}", flush=True)
    n_occ = int((indices[:, 0] >= 0).sum())
    print(f"    occupied pillars={n_occ}/{MAX_PILLARS} "
          f"boxes={len(boxes)} classes="
          f"{np.bincount(labels.numpy()).tolist() if len(labels) else 'none'}",
          flush=True)
    assert pillars.shape == (MAX_PILLARS, MAX_POINTS_PER_PILLAR, 9)
    assert indices.shape == (MAX_PILLARS, 2)
    assert boxes.shape[1] == 7 and pillars.dtype == torch.float32
    assert (indices[:n_occ, 0] >= 0).all() and (indices[n_occ:, 0] < 0).all()
    print("    PASS: sample shapes OK", flush=True)

    # -- 2. batch collation --------------------------------------------------
    from torch.utils.data import DataLoader, Subset
    n = min(args.n_samples, len(ds))
    loader = DataLoader(Subset(ds, range(n)), batch_size=args.batch_size,
                        collate_fn=collate_fn, num_workers=0)
    (b_pil, b_idx, b_box, b_lbl, b_mask) = next(iter(loader))
    print(f"[3] batch: pillars={tuple(b_pil.shape)} indices={tuple(b_idx.shape)} "
          f"boxes={tuple(b_box.shape)} labels={tuple(b_lbl.shape)} "
          f"mask={tuple(b_mask.shape)} (valid boxes={int(b_mask.sum())})",
          flush=True)
    assert b_pil.shape == (args.batch_size, MAX_PILLARS, MAX_POINTS_PER_PILLAR, 9)
    assert b_idx.shape == (args.batch_size, MAX_PILLARS, 2)
    assert b_box.shape[2] == 7 and b_mask.shape == b_lbl.shape
    counts = pillar_counts(b_idx)
    print(f"    occupied pillars per sample: {counts.tolist()}", flush=True)
    print("    PASS: batch collation OK", flush=True)

    # -- 3. forward pass (DataParallel) --------------------------------------
    model = PointPillars(num_classes=len(CLASS_NAMES)).to(device)
    fh, fw = output_grid(model, device)
    model = torch.nn.DataParallel(model) if device == "cuda" else model
    model.eval()
    with torch.no_grad():
        cls_p, reg_p = model(b_pil.to(device))
    print(f"[4] forward: cls={tuple(cls_p.shape)} reg={tuple(reg_p.shape)} "
          f"grid={fh}x{fw} mem="
          f"{torch.cuda.max_memory_allocated(0) / 1e9:.2f}GB (gpu0)",
          flush=True)
    assert cls_p.shape == (args.batch_size, 3, fh, fw)
    assert reg_p.shape == (args.batch_size, 7, fh, fw)
    print("    PASS: forward pass OK", flush=True)

    # -- 4. loss decreases (overfit) -----------------------------------------
    # Load the real batch ONCE through the loader, then overfit on the cached
    # tensors — the point is to verify the loss math decreases on real data,
    # not to benchmark TFRecord re-reading (which is the slow part).
    train_loader = DataLoader(Subset(ds, range(args.batch_size)),
                              batch_size=args.batch_size,
                              collate_fn=collate_fn, num_workers=0)
    p, i, bx, lb, m = next(iter(train_loader))
    p = p.to(device); i = i.to(device)
    bx = bx.to(device); lb = lb.to(device); m = m.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses, first = [], None
    model.train()
    for ep in range(args.epochs):
        cls_pred, reg_pred = model(p)
        loss, cls_l, reg_l, n_pos = detection_loss(
            cls_pred, reg_pred, i, bx, lb, m, fh, fw)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if first is None:
            first = loss.item()
        print(f"    overfit epoch {ep + 1:2d}: loss={losses[-1]:.4f} "
              f"cls={cls_l.item():.4f} reg={reg_l.item():.4f} "
              f"pos_cells={int(n_pos)}", flush=True)
    print(f"[5] loss: {first:.4f} -> {losses[-1]:.4f} "
          f"(final/last3: {np.mean(losses[-3:]):.4f})", flush=True)
    assert losses[-1] < first, "loss did not decrease!"
    print("    PASS: loss decreases", flush=True)

    print("\nALL SMOKE TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
