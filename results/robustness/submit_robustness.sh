#!/bin/bash
#SBATCH --job-name=robust
#SBATCH --partition=ais-cpu
#SBATCH --array=0-999
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/results/robustness/slurm_%A_%a.out
#SBATCH --error=/gpfs/data/gpfs0/aphan_group/ka_raceing_vdc/results/robustness/slurm_%A_%a.err

set -e
cd /gpfs/data/gpfs0/aphan_group/ka_raceing_vdc
export PYTHONUNBUFFERED=1

START=$((SLURM_ARRAY_TASK_ID * 200))
END=$((START + 200 - 1))
if [ $END -ge 200000 ]; then END=$((200000 - 1)); fi

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} starting: samples ${START}-${END} on $(hostname)"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/3.12

for ((j=START; j<=END; j++)); do
    python3 -u src/sweeps/run_one_sample.py $j results/robustness
    
    if (( (j - START + 1) % 5 == 0 )); then
        echo "  [$(($j - START + 1))/$(($END - START + 1))] sample $j done"
    fi
done

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} DONE"
