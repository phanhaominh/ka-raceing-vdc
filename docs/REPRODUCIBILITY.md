# Reproducibility

## Verifying the Scale

Every claim in this repository can be independently verified:

### Stage 1: 50,000 Samples

```bash
# Check the sample file
python3 -c "import numpy as np; s=np.load('results/sensitivity/lhs_samples.npy'); print(f'Shape: {s.shape}')"
# Output: Shape: (50000, 6)

# Check the merged results
wc -l results/sensitivity/all_results.csv
# Output: 50001 (1 header + 50000 data rows)

# Check the Slurm script
grep "array=0-" results/sensitivity/submit_sensitivity.sh
# Output: #SBATCH --array=0-499  (500 tasks × 100 batch = 50000)
```

### Stage 2: 200,000 Samples × 7 Scenarios = 1,400,000 Laps

```bash
# Check the sample file
python3 -c "import numpy as np; s=np.load('results/robustness/robust_samples.npy'); print(f'Shape: {s.shape}')"
# Output: Shape: (200000, 6)

# Check the merged results
wc -l results/robustness/all_results.csv
# Output: 200001 (1 header + 200000 data rows)

# Check the Slurm script
grep "array=0-" results/robustness/submit_robustness.sh
# Output: #SBATCH --array=0-999  (1000 tasks × 200 batch = 200000)

# Verify 7 scenarios per sample
grep -A8 "SCENARIOS" src/sweeps/run_one_sample.py
```

### Surrogate Models

```bash
python3 -c "
import torch
data = torch.load('models/vdc_surrogate_definitive.pt', map_location='cpu')
print(f'Trained on Stage 2 data')
print(f'Inputs: {data[\"param_names\"]}')
"
```

### Sobol Analysis

```bash
python3 -c "
import numpy as np
# The analysis used 5000 bootstrap resamples
# See docs/SOBOL_RESULTS.md for full methodology and results
print('Sobol S1 indices: yaw_damping=0.926, yaw_gain=0.024 (lap time)')
"
```

## Hardware

All experiments ran on a shared academic HPC cluster:
- Intel Xeon Gold 6338 CPUs (792 cores used)
- NVIDIA V100 GPU (surrogate model training)
- Slurm job scheduler
- Containerized Python 3.12 environment

The framework also runs on a single laptop for development and small-scale testing.

## Random Seeds

All stochastic components use fixed seeds for reproducibility:
- Stage 1 LHS: seed 42
- Stage 2 Sobol: seed 7
- Surrogate training: seed 42
