#!/bin/bash
#SBATCH --job-name=cfd_airfoil
#SBATCH --partition=ais-cpu
#SBATCH --array=0-50
#SBATCH --ntasks=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/cfd/results/slurm_%A_%a.out
#SBATCH --error=/gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/cfd/results/slurm_%A_%a.err

# AoA from -5 to 20 in 0.5 degree steps
AOA=$(echo "scale=1; -5 + $SLURM_ARRAY_TASK_ID * 0.5" | bc)
OUTDIR=/gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/cfd/results/aoa_${SLURM_ARRAY_TASK_ID}

echo "[$(date)] Starting AoA=${AOA}° on $(hostname)"

bash /gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/cfd/scripts/run_aoa.sh $AOA $OUTDIR

echo "[$(date)] AoA=${AOA}° complete"
