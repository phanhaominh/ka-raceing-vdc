#!/bin/bash
# Run one airfoil case at a specific angle of attack
# Usage: run_aoa.sh <angle> <output_dir>

AOA=$1
OUTDIR=$2
CONTAINER=/gpfs/data/gpfs0/aphan_group/openfoam.sif
TUTORIAL=/gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/cfd/airfoilWithLayers

echo "[$(date)] Starting AoA=${AOA}°"

# Create case directory
mkdir -p $OUTDIR
cd $OUTDIR

# Copy clean tutorial
cp -r $TUTORIAL/* .

# Rotate the airfoil STL by AOA degrees (around z-axis)
apptainer exec $CONTAINER bash -c "
source /usr/lib/openfoam/openfoam2512/etc/bashrc
surfaceTransformPoints -rollPitchYaw '(0 0 $AOA)' geometry/aerofoil.stl geometry/aerofoil.stl
"

# Run the meshing pipeline
apptainer exec --bind $PWD:$PWD $CONTAINER bash -c "
source /usr/lib/openfoam/openfoam2512/etc/bashrc
cd $PWD
./Allrun 2>&1 | tee log.run
"

echo "[$(date)] AoA=${AOA}° complete"
