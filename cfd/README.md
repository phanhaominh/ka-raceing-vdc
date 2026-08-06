# OpenFOAM CFD Pipeline

Automated aerodynamic sweep infrastructure for Formula Student vehicles.

## Status

✅ OpenFOAM 2512 working via Apptainer  
✅ Mesh generation with automated AoA sweep (-5° to 20°, 0.5° increments)  
✅ Slurm array deployment tested on 51 cases simultaneously  
🔜 Solver integration pending team-specific boundary conditions and solver settings  
🔜 Full-car mesh pending KA-RaceIng geometry

## What Works

- STL rotation to arbitrary angle of attack
- blockMesh → surfaceFeatureExtract → snappyHexMesh pipeline
- Two-level mesh refinement (basic + highAspectRatio)
- HPC deployment: 51 cases on 408 cores in ~5 minutes

## What's Needed

- Solver boundary conditions (U, p, nut) matching KA-RaceIng workflow
- Force coefficient extraction (Cl, Cd)
- Validation against team's existing CFD results

## Quick Test

```bash
cd cfd
bash scripts/run_aoa.sh 5 results/test_aoa5
```

## Full Sweep

```bash
sbatch scripts/slurm_sweep.sh  # 51 cases, -5° to 20° AoA
```
