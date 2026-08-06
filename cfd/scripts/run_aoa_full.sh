#!/bin/bash
# Full CFD pipeline: mesh + solve + extract forces
AOA=$1
OUTDIR=$2
CONTAINER=/gpfs/data/gpfs0/aphan_group/openfoam.sif
TUTORIAL=/gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/cfd/airfoilWithLayers

echo "[$(date)] Starting full CFD for AoA=${AOA}°"

mkdir -p $OUTDIR
cd $OUTDIR
cp -r $TUTORIAL/* .

# Rotate geometry
apptainer exec $CONTAINER bash -c "
source /usr/lib/openfoam/openfoam2512/etc/bashrc
surfaceTransformPoints -rollPitchYaw '(0 0 $AOA)' geometry/aerofoil.stl geometry/aerofoil.stl
"

# Mesh
apptainer exec --bind $PWD:$PWD $CONTAINER bash -c "
source /usr/lib/openfoam/openfoam2512/etc/bashrc
cd $PWD
./Allrun 2>&1 | tee log.mesh
"

# Add solver if we have boundary conditions
if [ -d "0.org" ]; then
    cp -r 0.org 0
fi

# Run simpleFoam
apptainer exec --bind $PWD:$PWD $CONTAINER bash -c "
source /usr/lib/openfoam/openfoam2512/etc/bashrc
cd $PWD
if [ -f system/controlDict ]; then
    # Modify controlDict for quick run
    sed -i 's/endTime.*/endTime         500;/' system/controlDict
    sed -i 's/writeInterval.*/writeInterval   100;/' system/controlDict
    
    simpleFoam 2>&1 | tee log.solver
    
    # Try to extract forces
    if [ -d postProcessing ]; then
        echo 'Forces extracted'
        find postProcessing -name '*.dat' -exec cat {} \;
    fi
fi
"

echo "[$(date)] AoA=${AOA}° complete"
