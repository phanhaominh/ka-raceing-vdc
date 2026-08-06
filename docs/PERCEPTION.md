# LiDAR Perception Pipeline — Waymo → PointPillars

## Status

**Trained and verified.** A PointPillars model was trained on 150 Waymo frames (3,988 vehicles, 906 pedestrians, 15 cyclists) for 100 epochs on 4× V100 GPUs. The pipeline is end-to-end verified: TFRecord → numpy conversion → training → checkpoints.

## Training Results

| Metric | Value |
|--------|-------|
| Frames | 150 (120 train / 30 val) |
| Epochs | 100 |
| Hardware | 4× NVIDIA V100-16GB |
| Training time | 17 minutes |
| Initial loss | 4.68 |
| Final train loss | 1.06 |
| Final val loss | 4.54 |
| Checkpoints | Every 10 epochs (10 files) |

## Architecture

| Component | Detail |
|-----------|--------|
| Model | PointPillars — 597,834 parameters |
| Input | LiDAR point clouds → pillars (B, 12000, 100, 9) |
| Output | 3D bounding boxes: (x, y, z, w, l, h, yaw) + class scores |
| Classes | Vehicle (0), Pedestrian (1), Cyclist (2) |
| Loss | Sigmoid focal loss (α=0.25, γ=2.0) + smooth L1 (β=1/9) |
| Grid | 28×28 pseudo-image with per-class anchors |

## Pipeline Verification

| Check | Result |
|-------|--------|
| TFRecord parsing | ✅ 64-channel TOP laser, vehicle-frame points |
| Numpy conversion | ✅ 150 frames, ~4.5 min, correct geometry |
| Forward pass | ✅ cls (4,3,28,28), reg (4,7,28,28) |
| Loss convergence | ✅ 4.68 → 1.06 over 100 epochs |
| Checkpoint saving | ✅ 10 checkpoints (epochs 10-100) |
| TensorBoard logging | ✅ events written |

## Quick Start

```bash
# Convert Waymo TFRecords to numpy (one-time)
python3 scripts/convert_waymo_numpy.py

# Train on 4 GPUs
python3 -m src.perception.train_numpy \
  --data-dir data/waymo_numpy_pillars \
  --batch-size 4 \
  --epochs 100 \
  --checkpoint-dir runs/waymo_v100/checkpoints \
  --log-dir runs/waymo_v100/tb

# Resume from checkpoint
python3 -m src.perception.train_numpy \
  --resume runs/waymo_v100/checkpoints/epoch_50.pth
```

## Path to KA-RaceIng Integration

1. **Backbone trained on Waymo** ✅ — LiDAR feature extractor learns geometry
2. **Fine-tune on FSAE cone data** 🔜 — coneScenes (public) or KA-RaceIng's labeled laps
3. **Export to ONNX** — deploy on Intel Core Ultra 7 (5458 GFLOPS)
4. **Validate on track** — compare against existing perception pipeline

## Current Limitations

- Small dataset (150 frames) — overfitting on training set; more frames needed for production model
- Cyclist class underrepresented (15 boxes) — class 2 supervision is weak
- No data augmentation — adding flips/rotations would improve generalization
- coneScenes fine-tuning pending — pipeline ready, data access pending

## References

- coneScenes Dataset: [Chalmers-Formula-Student/coneScenes](https://github.com/Chalmers-Formula-Student/coneScenes)
- Waymo Open Dataset v1.3.0: 1.9TB, 798 TFRecords, 1.6M frames
