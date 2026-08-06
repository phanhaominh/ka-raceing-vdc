"""Waymo Open Dataset v1.3.0 TFRecord loader for PointPillars.

Reads Waymo TFRecords directly (TensorFlow is used only for TFRecord I/O and
range-image decompression; all training-side tensors are numpy/PyTorch).

Data facts (verified on the on-disk v1.3.0 data, 2026-08):
  * ``frame.lasers[0]`` is the TOP 64-beam lidar (``LaserName.TOP == 1``) —
    the same class of sensor as the Hesai Pandar64 used in KA-RaceIng.
  * ``frame_utils.convert_range_image_to_point_cloud(...,
    keep_polar_features=True)`` returns a per-laser *list* (sorted by
    calibration name); each row is ``[range, intensity, elongation, x, y, z]``
    and is already in the **vehicle frame**.
  * ``frame.laser_labels`` boxes are also in the **vehicle frame** in v1.3.0
    (centers are small offsets around the ego vehicle), so no pose transform
    is applied.
  * v1.3.0 SDK API quirks: ``parse_range_image_and_camera_projection``
    returns a 4-tuple ``(range_images, camera_projections, seg_labels,
    range_image_top_pose)``; ``convert_range_image_to_point_cloud`` uses
    ``ri_index=0`` for the first return.

``__getitem__(idx)`` returns ``(pillars, pillar_indices, boxes, labels)``:
  * ``pillars``        (MAX_PILLARS, MAX_POINTS_PER_PILLAR, 9) float32
  * ``pillar_indices`` (MAX_PILLARS, 2) int32 — grid (x, y) cell; (-1,-1) = empty slot
  * ``boxes``          (B, 7) float32 — (x, y, z, w, l, h, yaw), vehicle frame
  * ``labels``         (B,)  int64   — 0=VEHICLE, 1=PEDESTRIAN, 2=CYCLIST

The 9 point features are the standard PointPillars encoding:
  ``(x, y, z, intensity, x_c, y_c, z_c, x_p, y_p)``
where ``(x_c, y_c, z_c)`` is the pillar centroid (over all points of the
pillar) and ``(x_p, y_p)`` the offset of the point from the pillar center.

``MAX_PILLARS`` is fixed (12000) because the PointPillars model scatters
pillars onto a dense square grid of size ``H = int(P**0.5) + 1`` and needs a
fixed P across the whole batch.  Pillars are ordered by ascending grid index,
so tensor position k maps deterministically to pseudo-image cell
``(k // H, k % H)`` — this is what the target builder in ``train.py`` relies
on.
"""

import glob
import os

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2
from waymo_open_dataset.utils import range_image_utils, transform_utils

# ---------------------------------------------------------------------------
# Grid / pillar configuration (PointPillars defaults, matches the model)
# ---------------------------------------------------------------------------
GRID_X = (-75.2, 75.2)      # meters, x
GRID_Y = (-75.2, 75.2)      # meters, y
GRID_Z = (-4.0, 2.0)        # meters, z
PILLAR_SIZE = (0.16, 0.16)  # meters, (x, y)
MAX_POINTS_PER_PILLAR = 100
MAX_PILLARS = 12000         # fixed per-sample pillar count (model contract)

GRID_W = int((GRID_X[1] - GRID_X[0]) / PILLAR_SIZE[0])  # 940
GRID_H = int((GRID_Y[1] - GRID_Y[0]) / PILLAR_SIZE[1])  # 940

NUM_FEATURES = 9

# Waymo Label.Type -> 0/1/2 class ids (TYPE_SIGN and TYPE_UNKNOWN dropped).
# In v1.3.0 the Label message lives in waymo_open_dataset.label_pb2.
WAYMO_TYPE_TO_CLASS = {
    label_pb2.Label.Type.TYPE_VEHICLE: 0,
    label_pb2.Label.Type.TYPE_PEDESTRIAN: 1,
    label_pb2.Label.Type.TYPE_CYCLIST: 2,
}


def _top_laser_points(frame):
    """Return the TOP (64-beam) lidar, first return, as (N, 6) float32
    ``[range, intensity, elongation, x, y, z]`` in the vehicle frame.

    Implements only the TOP-laser path of the SDK's
    ``convert_range_image_to_point_cloud`` (which decompresses and converts
    all five lasers, ~30 s/frame on this cluster).  Only the TOP laser's
    return-1 range image + per-pixel pose are decompressed here (~3 s/frame).
    """
    def _decompress(field, msg_cls):
        raw = tf.io.decode_compressed(field, "ZLIB").numpy()
        msg = msg_cls()
        msg.ParseFromString(raw)
        return msg

    laser = next((l for l in frame.lasers
                  if l.name == open_dataset.LaserName.TOP), frame.lasers[0])
    cal = next((c for c in frame.context.laser_calibrations
                if c.name == open_dataset.LaserName.TOP), None)
    assert cal is not None, "TOP laser calibration missing"

    ri = _decompress(laser.ri_return1.range_image_compressed,
                     open_dataset.MatrixFloat)
    range_t = tf.reshape(tf.convert_to_tensor(ri.data), ri.shape.dims)  # (H,W,6)

    # beam inclinations (reverse, as the SDK does)
    if len(cal.beam_inclinations) == 0:
        beam = range_image_utils.compute_inclination(
            tf.constant([cal.beam_inclination_min, cal.beam_inclination_max]),
            height=ri.shape.dims[0])
    else:
        beam = tf.constant(cal.beam_inclinations)
    beam = tf.reverse(beam, axis=[-1])
    extrinsic = np.reshape(np.array(cal.extrinsic.transform), [4, 4])

    # per-pixel lidar pose (TOP only)
    frame_pose = tf.convert_to_tensor(
        np.reshape(np.array(frame.pose.transform), [4, 4]))
    pixel_pose = None
    if len(laser.ri_return1.range_image_pose_compressed) > 0:
        pose_ri = _decompress(laser.ri_return1.range_image_pose_compressed,
                              open_dataset.MatrixFloat)
        pose_t = tf.reshape(tf.convert_to_tensor(pose_ri.data),
                            pose_ri.shape.dims)                    # (H,W,6)
        rot = transform_utils.get_rotation_matrix(
            pose_t[..., 0], pose_t[..., 1], pose_t[..., 2])
        pixel_pose = transform_utils.get_transform(rot, pose_t[..., 3:])  # (H,W,4,4)

    cart = range_image_utils.extract_point_cloud_from_range_image(
        tf.expand_dims(range_t[..., 0], 0),   # range channel
        tf.expand_dims(extrinsic, 0),
        tf.expand_dims(beam, 0),
        pixel_pose=(tf.expand_dims(pixel_pose, 0) if pixel_pose is not None
                    else None),
        frame_pose=tf.expand_dims(frame_pose, 0))
    cart = tf.squeeze(cart, 0)                                       # (H,W,3)
    cart = tf.concat([range_t[..., 0:3], cart], axis=-1)             # (H,W,6)
    valid = range_t[..., 0] > 0
    pts = tf.gather_nd(cart, tf.compat.v1.where(valid)).numpy()
    return np.asarray(pts, dtype=np.float32)


class WaymoPillarDataset(Dataset):
    """Map-style dataset over Waymo frames (one sample per frame).

    Args:
        data_dir: dataset root, i.e. the directory containing split dirs
            (training/ validation/ testing/).
        split: one of 'training', 'validation', 'testing'.
        max_files: only use the first N .tfrecord files (for quick tests).
        max_frames: cap the total number of frames returned.
        grid_x/grid_y/grid_z/pillar_size: pillar grid configuration.
        max_pillars/max_points_per_pillar: pillar tensor sizes.
    """

    def __init__(self, data_dir, split="training", max_files=None,
                 max_frames=None, max_pillars=MAX_PILLARS,
                 max_points_per_pillar=MAX_POINTS_PER_PILLAR,
                 grid_x=GRID_X, grid_y=GRID_Y, grid_z=GRID_Z,
                 pillar_size=PILLAR_SIZE):
        self.split = split
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.pillar_size = pillar_size
        self.max_pillars = max_pillars
        self.max_points_per_pillar = max_points_per_pillar
        self.grid_w = int((grid_x[1] - grid_x[0]) / pillar_size[0])
        self.grid_h = int((grid_y[1] - grid_y[0]) / pillar_size[1])

        self.files = sorted(glob.glob(
            os.path.join(data_dir, split, "**", "*.tfrecord"),
            recursive=True))
        if max_files is not None:
            self.files = self.files[:max_files]
        assert self.files, f"no .tfrecord files found under {data_dir}/{split}"

        # One pass per file to count frames (needed for __len__/indexing).
        # For a full training split this is a one-time ~1-2 h scan; for
        # testing use max_files to keep it fast. A cached index would remove
        # this cost entirely (future optimization).
        self._counts = []
        for i, f in enumerate(self.files):
            n = self._count_frames(f)
            self._counts.append(n)
            print(f"[WaymoPillarDataset] {os.path.basename(f)}: {n} frames "
                  f"({i + 1}/{len(self.files)})", flush=True)
        self._offsets = np.cumsum([0] + self._counts)
        self._length = int(self._offsets[-1])
        if max_frames is not None:
            self._length = min(self._length, int(max_frames))

    # -- helpers ------------------------------------------------------------
    def _count_frames(self, path):
        ds = tf.data.TFRecordDataset(path, compression_type="")
        return sum(1 for _ in ds)

    def _read_frame(self, path, frame_idx):
        ds = tf.data.TFRecordDataset(path, compression_type="")
        for data in ds.skip(frame_idx).take(1):
            frame = open_dataset.Frame()
            frame.ParseFromString(data.numpy())
            return frame
        raise IndexError(f"frame {frame_idx} not in {path}")

    # -- Dataset interface ---------------------------------------------------
    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        idx = int(idx)
        if idx < 0 or idx >= self._length:
            raise IndexError(f"index {idx} out of range ({self._length})")
        file_i = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        frame_i = int(idx - self._offsets[file_i])
        frame = self._read_frame(self.files[file_i], frame_i)
        return self._frame_to_sample(frame)

    # -- frame -> sample ------------------------------------------------------
    def _frame_to_sample(self, frame):
        pts = _top_laser_points(frame)
        xyz = pts[:, 3:6]          # (N, 3) vehicle frame
        intensity = pts[:, 1]      # (N,)
        pillars, indices, _n = self._build_pillars(xyz, intensity)
        boxes, labels = self._extract_boxes(frame)
        return (torch.from_numpy(pillars),
                torch.from_numpy(indices),
                torch.from_numpy(boxes),
                torch.from_numpy(labels))

    def _extract_boxes(self, frame):
        """VEHICLE/PEDESTRIAN/CYCLIST boxes in the vehicle frame."""
        boxes, labels = [], []
        for lbl in frame.laser_labels:
            cls = WAYMO_TYPE_TO_CLASS.get(lbl.type)
            if cls is None:
                continue
            b = lbl.box
            boxes.append((b.center_x, b.center_y, b.center_z,
                          b.width, b.length, b.height, b.heading))
            labels.append(cls)
        if boxes:
            return (np.asarray(boxes, dtype=np.float32),
                    np.asarray(labels, dtype=np.int64))
        return (np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))

    def _build_pillars(self, xyz, intensity):
        """Vectorized pillarization (delegates to module-level build_pillars)."""
        return build_pillars(xyz, intensity, self.max_pillars,
                             self.max_points_per_pillar, self.grid_x,
                             self.grid_y, self.grid_z, self.pillar_size)


def build_pillars(xyz, intensity, max_pillars=MAX_PILLARS,
                  max_points_per_pillar=MAX_POINTS_PER_PILLAR,
                  grid_x=GRID_X, grid_y=GRID_Y, grid_z=GRID_Z,
                  pillar_size=PILLAR_SIZE):
    """Vectorized pillarization of an (N, 3) point cloud + (N,) intensities.

    Points are assigned to (gx, gy) cells on the xy grid, sorted by ascending
    grid index ``gy * grid_w + gx`` and truncated to ``max_pillars`` occupied
    pillars x ``max_points_per_pillar`` points.  Feature columns per point:
    (x, y, z, intensity, x_c, y_c, z_c, x_p, y_p) where (x_c, y_c, z_c) is the
    pillar centroid over all of the pillar's points and (x_p, y_p) the offset
    of the point from the pillar center.

    Returns ``(pillars, pillar_indices, n_occupied)`` with shapes
    ``(max_pillars, max_points_per_pillar, 9)`` float32, ``(max_pillars, 2)``
    int32 ((-1,-1) = empty slot) and the number of occupied pillars.
    """
    grid_w = int((grid_x[1] - grid_x[0]) / pillar_size[0])
    grid_h = int((grid_y[1] - grid_y[0]) / pillar_size[1])
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    x_min, y_min = grid_x[0], grid_y[0]
    px, py = pillar_size

    ok = ((x >= grid_x[0]) & (x < grid_x[1]) &
          (y >= grid_y[0]) & (y < grid_y[1]) &
          (z >= grid_z[0]) & (z < grid_z[1]))
    x, y, z, intensity = x[ok], y[ok], z[ok], intensity[ok]

    out_pillars = np.zeros((max_pillars, max_points_per_pillar, 9),
                           dtype=np.float32)
    out_indices = np.full((max_pillars, 2), -1, dtype=np.int32)
    if len(x) == 0:
        return out_pillars, out_indices, 0

    gx = np.clip(np.floor((x - x_min) / px).astype(np.int64),
                 0, grid_w - 1)
    gy = np.clip(np.floor((y - y_min) / py).astype(np.int64),
                 0, grid_h - 1)
    pid = gy * grid_w + gx

    order = np.argsort(pid, kind="stable")
    pid = pid[order]
    x, y, z, intensity = x[order], y[order], z[order], intensity[order]

    # pillar id per point, counts, per-pillar starts, per-point rank
    new_pillar = np.zeros(len(pid) + 1, dtype=bool)
    new_pillar[0] = True
    new_pillar[1:-1] = pid[1:] != pid[:-1]
    pillar_idx = np.cumsum(new_pillar)[:-1] - 1
    num_pillars = int(pillar_idx[-1]) + 1
    counts = np.bincount(pillar_idx, minlength=num_pillars)
    starts = np.searchsorted(pillar_idx, np.arange(num_pillars),
                             side="left")
    rank = np.arange(len(pid)) - starts[pillar_idx]

    keep = rank < max_points_per_pillar
    keep_pid = pillar_idx[keep]
    P = min(num_pillars, max_pillars)
    sel = keep_pid < P
    kx = x[keep][sel]
    ky = y[keep][sel]
    kz = z[keep][sel]
    ki = intensity[keep][sel]
    kpid = keep_pid[sel]
    krank = rank[keep][sel]
    kgx = gx[keep][sel]
    kgy = gy[keep][sel]

    # pillar centroid over ALL points of the pillar (standard PointPillars)
    sums = np.add.reduceat(np.stack([x, y, z], axis=1), starts)
    centroids = sums / counts[:, None]              # (num_pillars, 3)
    cx = centroids[kpid, 0]
    cy = centroids[kpid, 1]
    cz = centroids[kpid, 2]

    # offset from pillar center
    xp = kx - (x_min + (kgx.astype(np.float32) + 0.5) * px)
    yp = ky - (y_min + (kgy.astype(np.float32) + 0.5) * py)

    feats = np.stack([kx, ky, kz, ki, cx, cy, cz, xp, yp], axis=1)
    flat = out_pillars.reshape(-1, 9)
    flat[kpid * max_points_per_pillar + krank] = feats

    # grid (x, y) cell per pillar (same for all points of a pillar)
    first = np.searchsorted(kpid, np.arange(P), side="left")
    out_indices[:P, 0] = kgx[first]
    out_indices[:P, 1] = kgy[first]
    return out_pillars, out_indices, P


def pillar_counts(pillar_indices):
    """Number of occupied pillars per sample, from a (B, P, 2) index tensor."""
    return (pillar_indices[..., 0] >= 0).sum(dim=-1)


def collate_fn(batch):
    """Stack a batch of ``(pillars, pillar_indices, boxes, labels)`` samples.

    Pillars/indices have fixed shape so they stack directly; boxes/labels have
    a per-sample count so they are padded to the batch maximum.

    Returns ``(pillars, pillar_indices, boxes, labels, box_mask)``:
      * pillars        (B, MAX_PILLARS, N, 9)
      * pillar_indices (B, MAX_PILLARS, 2)
      * boxes          (B, maxB, 7) float32 (padded with zeros)
      * labels         (B, maxB) int64 (padded with zeros)
      * box_mask       (B, maxB) bool — which box slots are real
    """
    pillars = torch.stack([b[0] for b in batch])
    indices = torch.stack([b[1] for b in batch])
    max_b = max(b[2].shape[0] for b in batch)
    boxes = torch.zeros(len(batch), max_b, 7, dtype=batch[0][2].dtype)
    labels = torch.zeros(len(batch), max_b, dtype=batch[0][3].dtype)
    mask = torch.zeros(len(batch), max_b, dtype=torch.bool)
    for i, b in enumerate(batch):
        nb = b[2].shape[0]
        if nb:
            boxes[i, :nb] = b[2]
            labels[i, :nb] = b[3]
            mask[i, :nb] = True
    return pillars, indices, boxes, labels, mask
