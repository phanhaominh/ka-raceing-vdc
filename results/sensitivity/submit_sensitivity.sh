#!/bin/bash
#SBATCH --job-name=sens
#SBATCH --partition=ais-cpu
#SBATCH --array=0-499
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/results/sensitivity/slurm_%A_%a.out
#SBATCH --error=/gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/results/sensitivity/slurm_%A_%a.err

set -e
cd /gpfs/data/gpfs0/aphan_group/ka_raceing_vdc
export PYTHONUNBUFFERED=1

START=$((SLURM_ARRAY_TASK_ID * 100))
END=$((START + 100 - 1))
if [ $END -ge 50000 ]; then END=$((50000 - 1)); fi

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} starting: samples ${START}-${END} on $(hostname)"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/3.12

python3 -u -c "
from src.sweeps.sensitivity import run_batch_from_file
run_batch_from_file(range($START, $END + 1), 'results/sensitivity/lhs_samples.npy', 'results/sensitivity')
"

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} DONE"
