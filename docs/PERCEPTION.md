# LiDAR Perception Pipeline — Waymo → PointPillars

## Overview

A 3D object detector trained on the Waymo Open Dataset v1.3.0 (1.9TB, LiDAR + 3D boxes) using the PointPillars architecture. Designed for transfer to KA-RaceIng's driverless vehicle, which uses the same Hesai 64-channel LiDAR sensor.

## Architecture

| Component | Detail |
|-----------|--------|
| Model | PointPillars — 597,834 parameters |
| Input | LiDAR point clouds → pillars (P, N, 9) |
| Output | 3D bounding boxes: (x, y, z, w, l, h, yaw) + class scores |
| Classes | Vehicle, Pedestrian, Cyclist (→ transfer to FSAE cones) |
| Training | 4× NVIDIA A100-80GB, DataParallel |
| Loss | Sigmoid focal loss (α=0.25, γ=2.0) + smooth L1 (β=1/9) |

## Verification

| Check | Result |
|-------|--------|
| TFRecord parsing | ✅ 153,830 points/frame, 64-channel TOP laser |
| Coordinate frame | ✅ Points and boxes in vehicle frame — no transform needed |
| Forward pass | ✅ cls (4,3,28,28), reg (4,7,28,28), 0.81 GB/GPU |
| Loss convergence | ✅ 6.47 → 3.96 over 30 epochs on real batch |
| Full train.py run | ✅ 1 epoch, checkpoint saved, TensorBoard written |
| Extraction speed | ✅ ~3 s/frame (optimized single-laser path) |

## Quick Start

```bash
# Requires: PyTorch container with Waymo SDK
apptainer run --nv --cleanenv \
  --bind /gpfs/data/gpfs0/datasets/waymo_open_dataset_v_1_3_0:/waymo_data \
  --bind /gpfs/data/gpfs0/aphan_group/ka_raceing_vdc:/project \
  /beegfs/shared/singularity-images/NGC/pytorch_23.10-py3.simg \
  python3 /project/src/perception/test_loader.py

# Full training (multi-GPU)
apptainer run --nv --cleanenv \
  --bind /gpfs/data/gpfs0/datasets/waymo_open_dataset_v_1_3_0:/waymo_data \
  --bind /gpfs/data/gpfs0/aphan_group/ka_raceing_vdc:/project \
  /beegfs/shared/singularity-images/NGC/pytorch_23.10-py3.simg \
  python3 /project/src/perception/train.py --data_dir /waymo_data --epochs 50
```

## Path to KA-RaceIng Integration

1. **Backbone trained** — LiDAR feature extractor learns geometry from millions of Waymo points
2. **Transfer learning** — Replace classification head (3 classes → 2: cone, no-cone)
3. **Fine-tune** — On KA-RaceIng's labeled cone data (estimated: 100-500 labeled laps)
4. **Deploy** — Export to ONNX/TorchScript for Intel Core Ultra 7 inference

## Limitations

- Detects vehicles/pedestrians/cyclists, not FSAE cones (requires fine-tuning)
- TFRecord I/O is per-frame skip-scan (~3s/frame); full training would benefit from pre-converted format
- Model architecture is PointPillars (fast, proven); CenterPoint or VoxelNet may offer better accuracy
