# LiDAR Perception Pipeline — Waymo → PointPillars

## Status

**Pipeline verified. Model training in progress.** The infrastructure works end-to-end (TFRecord → pillars → forward pass → loss decrease → checkpoint). Detection performance metrics (mAP, precision/recall) are pending completion of full training.

## Architecture

| Component | Detail |
|-----------|--------|
| Model | PointPillars |
| Input | LiDAR point clouds → pillars (P, N, 9) |
| Output | 3D bounding boxes: (x, y, z, w, l, h, yaw) + class scores |
| Classes | Vehicle, Pedestrian, Cyclist (→ transfer to FSAE cones) |
| Training | 4× NVIDIA A100-80GB, DataParallel |
| Loss | Sigmoid focal loss (α=0.25, γ=2.0) + smooth L1 (β=1/9) |

## Pipeline Verification

| Check | Result |
|-------|--------|
| TFRecord parsing | ✅ 153,830 points/frame from TOP laser |
| Coordinate frame | ✅ Points and boxes in vehicle frame — no transform needed |
| Pillar extraction | ✅ (12000, 100, 9) per sample, ~3s/frame |
| Forward pass | ✅ cls (4,3,28,28), reg (4,7,28,28), 0.81 GB/GPU |
| Loss convergence (smoke) | ✅ 6.47 → 3.96 over 30 epochs on 4-sample batch |
| Training loop | ✅ 1 epoch completes, checkpoint saves, TensorBoard writes |

## What's Pending

- **Detection metrics** (mAP, precision/recall at IoU=0.5) — requires full training completion
- **Qualitative results** — rendered point clouds with predicted boxes
- **coneScenes fine-tuning** — sample scene downloaded, loader written, pending full dataset access
- **ONNX export + inference benchmarking** — for deployment on Intel Core Ultra 7

## Quick Start

```bash
# Smoke test (verifies pipeline)
apptainer run --nv --cleanenv \
  --bind /gpfs/data/gpfs0/datasets/waymo_open_dataset_v_1_3_0:/waymo_data \
  --bind /gpfs/data/gpfs0/aphan_group/ka_raceing_vdc:/project \
  /beegfs/shared/singularity-images/NGC/pytorch_23.10-py3.simg \
  python3 /project/src/perception/test_loader.py

# Full training (multi-GPU, 50 epochs)
apptainer run --nv --cleanenv \
  --bind /gpfs/data/gpfs0/datasets/waymo_open_dataset_v_1_3_0:/waymo_data \
  --bind /gpfs/data/gpfs0/aphan_group/ka_raceing_vdc:/project \
  /beegfs/shared/singularity-images/NGC/pytorch_23.10-py3.simg \
  bash -c "cd /project && PYTHONPATH=/project python3 src/perception/train.py \
  --data-dir /waymo_data --epochs 50 --batch-size 8 --log-dir /project/runs/waymo_full"
```

## Path to KA-RaceIng Integration

1. **Backbone trained on Waymo** — LiDAR feature extractor learns geometry from millions of points
2. **Fine-tune on FSAE cone data** — coneScenes (public) or KA-RaceIng's labeled laps
3. **Export to ONNX** — deploy on Intel Core Ultra 7 (5458 GFLOPS)
4. **Validate on track** — compare against existing perception pipeline

## Limitations

- Model architecture is PointPillars (fast, proven); larger models (CenterPoint, VoxelNet) may improve accuracy
- Detection metrics not yet available — training in progress
- TFRecord I/O is per-frame skip-scan; full training benefits from pre-converted format
- coneScenes full dataset requires community contribution; sample scene available for testing
