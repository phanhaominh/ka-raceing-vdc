"""PointPillars model for Waymo 3D object detection.

Copied verbatim (architecture unchanged) from the tested model at
``/tmp/pointpillars_v2.py`` on gn34, with the module-level demo block removed.

I/O contract (matches ``waymo_loader.WaymoPillarDataset``):
  * input:  ``pillars`` of shape ``(B, P, N, 9)`` where B=batch, P=fixed number
    of pillars per sample (12000), N=points per pillar (100), 9 features
    = (x, y, z, intensity, x_c, y_c, z_c, x_p, y_p).
  * output: ``(cls_pred, reg_pred)`` with shapes ``(B, num_classes, fh, fw)``
    and ``(B, 7, fh, fw)``.

Note: the model scatters pillars onto a dense square grid of size
``H = int(P ** 0.5) + 1`` in the row-major order of the input tensor
(position k -> cell (k // H, k % H)) and downsamples it twice by 2 in the
backbone.  The loss in ``train.py`` uses this same layout to build targets.
"""

import torch
import torch.nn as nn


class PointPillars(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # PFN
        self.pfn = nn.Sequential(
            nn.Linear(9, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        # Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
        )
        # Heads
        self.cls_head = nn.Conv2d(256, num_classes, 1)
        self.reg_head = nn.Conv2d(256, 7, 1)

    def forward(self, pillars):
        # pillars: (B, P, N, 9) where B=batch, P=pillars, N=points
        B, P, N, F = pillars.shape

        # PFN
        x = self.pfn(pillars)  # (B, P, N, 64)
        x = x.max(dim=2)[0]     # (B, P, 64) — max pool over points

        # Scatter to pseudo-image
        H = W = int(P ** 0.5) + 1  # approximate square grid
        total = H * W
        if P < total:
            pad = torch.zeros(B, total - P, 64, device=pillars.device)
            x = torch.cat([x, pad], dim=1)
        else:
            x = x[:, :total]
        x = x.reshape(B, 64, H, W)  # (B, 64, H, W)

        # Backbone
        features = self.backbone(x)

        # Detection
        cls_pred = self.cls_head(features)
        reg_pred = self.reg_head(features)

        return cls_pred, reg_pred
