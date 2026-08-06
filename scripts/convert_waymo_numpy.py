"""One-time conversion: Waymo TFRecord frames -> pillar npz for fast numpy training.

The original ``data/waymo_numpy/frame_*.npz`` conversion only saved the raw
range image (ri) + boxes + labels — without the laser calibration (beam
inclinations, extrinsic, per-pixel pose) the range image cannot be projected
back to xyz, so this script re-reads the source TFRecords with the tested
extraction from ``src.perception.waymo_loader`` (correct vehicle-frame
geometry incl. per-pixel pose) and saves pre-pillarized tensors:

    frame_%06d.npz  keys:
      pillars        (12000, 100, 9) float32  standard 9-feature encoding
      pillar_indices (12000, 2)      int32    grid (x, y) cell; (-1,-1) = empty
      boxes          (B, 7)          float32  (x, y, z, w, l, h, yaw) vehicle frame
      labels         (B,)            int64    0=VEHICLE 1=PEDESTRIAN 2=CYCLIST

Frame selection (diverse, deterministic): the first ``--n-files`` sorted
training segments, ``--frames-per-file`` consecutive frames each, starting at
``--start-frame`` (mid-segment frames are denser and have more pedestrians).
The default 10 files x 15 frames x frame 60-74 yields ~4.9k boxes with a
vehicle/pedestrian/cyclist mix (file 0 is vehicle-only; file 1 is
pedestrian-rich — see probes).

Sharded usage (inside the container, run ``--nshards`` processes):

    apptainer exec --nv \\
      --bind /gpfs/data/gpfs0/datasets/waymo_open_dataset_v_1_3_0:/waymo_data \\
      --bind /gpfs/data/gpfs0/aphan_group/ka_raceing_vdc:/project \\
      /beegfs/shared/singularity-images/NGC/pytorch_23.10-py3.simg \\
      bash -lc 'cd /project && python3 scripts/convert_waymo_numpy.py \\
        --out-dir /project/data/waymo_numpy_pillars \\
        --n-files 10 --frames-per-file 15 --start-frame 60 \\
        --shard 0 --nshards 4'
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from waymo_open_dataset import dataset_pb2 as open_dataset  # noqa: E402

from src.perception import waymo_loader  # noqa: E402


def _pillar_builder():
    """A bare WaymoPillarDataset instance (config attrs only, no __init__)."""
    ds = waymo_loader.WaymoPillarDataset.__new__(waymo_loader.WaymoPillarDataset)
    ds.max_pillars = waymo_loader.MAX_PILLARS
    ds.max_points_per_pillar = waymo_loader.MAX_POINTS_PER_PILLAR
    ds.grid_x, ds.grid_y, ds.grid_z = waymo_loader.GRID_X, waymo_loader.GRID_Y, waymo_loader.GRID_Z
    ds.pillar_size = waymo_loader.PILLAR_SIZE
    ds.grid_w, ds.grid_h = waymo_loader.GRID_W, waymo_loader.GRID_H
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/waymo_data")
    ap.add_argument("--out-dir", default="/project/data/waymo_numpy_pillars")
    ap.add_argument("--split", default="training")
    ap.add_argument("--n-files", type=int, default=10)
    ap.add_argument("--frames-per-file", type=int, default=15)
    ap.add_argument("--start-frame", type=int, default=60)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    builder = _pillar_builder()
    files = sorted(glob.glob(
        os.path.join(args.data_dir, args.split, "**", "*.tfrecord"),
        recursive=True))
    assert files, f"no tfrecords under {args.data_dir}/{args.split}"
    plan = []  # (file_path, frame_index) per global frame
    for fpath in files[:args.n_files]:
        plan += [(fpath, j)
                 for j in range(args.start_frame,
                                args.start_frame + args.frames_per_file)]
    os.makedirs(args.out_dir, exist_ok=True)

    t_start = time.time()
    converted = 0
    for idx, (fpath, fi_idx) in enumerate(plan):
        if idx % args.nshards != args.shard:
            continue
        t0 = time.time()
        ds_tf = tf.data.TFRecordDataset(fpath, compression_type="")
        frame = None
        for data in ds_tf.skip(fi_idx).take(1):
            frame = open_dataset.Frame()
            frame.ParseFromString(data.numpy())
        assert frame is not None
        pts = waymo_loader._top_laser_points(frame)
        pillars, indices, _n = builder._build_pillars(pts[:, 3:6], pts[:, 1])
        boxes, labels = builder._extract_boxes(frame)
        out = os.path.join(args.out_dir, f"frame_{idx:06d}.npz")
        np.savez_compressed(
            out,
            pillars=pillars,
            pillar_indices=indices,
            boxes=boxes,
            labels=labels)
        converted += 1
        print(f"[shard {args.shard}] frame {idx:03d} "
              f"(file {idx // args.frames_per_file}, frame {fi_idx}): "
              f"{time.time() - t0:.1f}s -> {os.path.basename(out)} "
              f"({os.path.getsize(out) // 1024} KB)", flush=True)

    print(f"[shard {args.shard}] done: {converted} frames, "
          f"{time.time() - t_start:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
