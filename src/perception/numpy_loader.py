"""Fast numpy pillar loader for PointPillars training.

Loads the pre-pillarized Waymo frames written by
``scripts/convert_waymo_numpy.py`` into ``data/waymo_numpy_pillars/``
(replacing the earlier range-image-only ``data/waymo_numpy`` cache, whose raw
``ri`` lacked the laser calibration needed to project range -> xyz):

    frame_%06d.npz keys:
      pillars        (MAX_PILLARS, MAX_POINTS_PER_PILLAR, 9) float32
                     standard 9-feature encoding
                     (x, y, z, intensity, x_c, y_c, z_c, x_p, y_p)
      pillar_indices (MAX_PILLARS, 2) int32  grid (x, y) cell; (-1,-1) = empty
      boxes          (B, 7) float32  (x, y, z, w, l, h, yaw), vehicle frame
      labels         (B,) int64      0=VEHICLE 1=PEDESTRIAN 2=CYCLIST

``__getitem__`` returns ``(pillars, pillar_indices, boxes, labels)`` — the same
interface as ``WaymoPillarDataset``, so the loss/target code in ``train.py``
works unchanged.  This module is deliberately TF-free: it only does numpy
loads + torch tensors.
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyPillarDataset(Dataset):
    """Map-style dataset over pre-pillarized ``frame_*.npz`` files."""

    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "frame_*.npz")))
        assert self.files, f"no frame_*.npz files found under {data_dir}"
        print(f"[NumpyPillarDataset] {len(self.files)} frames from {data_dir}",
              flush=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        return (torch.from_numpy(d["pillars"]),
                torch.from_numpy(d["pillar_indices"]),
                torch.from_numpy(d["boxes"]),
                torch.from_numpy(d["labels"]))


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
