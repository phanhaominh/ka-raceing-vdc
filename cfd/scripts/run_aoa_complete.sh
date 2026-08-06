#!/bin/bash
AOA=$1
OUTDIR=$2
CONTAINER=/gpfs/data/gpfs0/aphan_group/openfoam.sif

echo "[$(date)] Full CFD for AoA=${AOA}°"

mkdir -p $OUTDIR
cd $OUTDIR

# 1. Mesh (from airfoilWithLayers)
cp -r /gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/cfd/airfoilWithLayers/* .

apptainer exec $CONTAINER bash -c "
source /usr/lib/openfoam/openfoam2512/etc/bashrc
surfaceTransformPoints -rollPitchYaw '(0 0 $AOA)' geometry/aerofoil.stl geometry/aerofoil.stl
" 2>&1 | tail -1

apptainer exec --bind $PWD:$PWD $CONTAINER bash -c "
source /usr/lib/openfoam/openfoam2512/etc/bashrc
cd $PWD
./Allrun 2>&1 | tail -5
"

# 2. Solver setup (from airfoil_full)
cp -r /gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/cfd/airfoil_full/0.orig .
cp -r /gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/cfd/airfoil_full/system/* system/ 2>/dev/null
cp -r /gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/cfd/airfoil_full/constant/* constant/ 2>/dev/null

# 3. Copy 0.orig to 0
cp -r 0.orig 0

# 4. Run simpleFoam
apptainer exec --bind $PWD:$PWD $CONTAINER bash -c "
source /usr/lib/openfoam/openfoam2512/etc/bashrc
cd $PWD
simpleFoam 2>&1 | tail -5
"

echo "[$(date)] AoA=${AOA}° complete"
