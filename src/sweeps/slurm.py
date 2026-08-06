"""Slurm job-array script generation for the ais-cpu partition.

Each array task runs a batch of parameter sets using the module-loaded
Python 3.12 (no container needed), writing per-batch CSVs to
``results/sensitivity/`` (Stage 1) or ``results/robustness/`` (Stage 2).

Generated scripts are submitted with::

    sbatch results/sensitivity/submit_sensitivity.sh
    sbatch results/robustness/submit_robustness.sh
"""

import os

import numpy as np

from src.sweeps.sensitivity import generate_lhs_samples, RESULTS_DIR
from src.sweeps.robustness import generate_robust_samples, ROBUSTNESS_DIR
from src.sweeps.sensitivity import PARAM_NAMES

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _script(name, outdir, ntasks, batch, n, python_call, time="24:00:00"):
    """Build the common Slurm array script body (module-loaded Python)."""
    os.makedirs(outdir, exist_ok=True)
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=ais-cpu
#SBATCH --array=0-{ntasks}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time={time}
#SBATCH --output={outdir}/slurm_%A_%a.out
#SBATCH --error={outdir}/slurm_%A_%a.err
set -e
cd {_PROJECT_ROOT}
START=$((SLURM_ARRAY_TASK_ID * {batch}))
END=$((START + {batch} - 1))
if [ $END -ge {n} ]; then END=$(({n} - 1)); fi
# Environment: module-loaded Python 3.12 (no container needed)
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/3.12
export PYTHONUNBUFFERED=1
python3 -u -c "{python_call}"
"""


def generate_sensitivity_job_array(n_samples=5000, batch_size=20,
                                   outdir=None, time="12:00:00"):
    """Generate the Stage 1 job-array script (+ the LHS sample file).

    Args:
        n_samples (int): total LHS samples.
        batch_size (int): samples per array task.

    Returns:
        str: script path.
    """
    outdir = outdir or RESULTS_DIR
    os.makedirs(outdir, exist_ok=True)

    # samples file must exist for the array tasks; generate it here too
    samples_path = os.path.join(outdir, "lhs_samples.npy")
    if not os.path.exists(samples_path):
        np.save(samples_path, generate_lhs_samples(n_samples, seed=42))
        print(f"[slurm] wrote {samples_path}")

    ntasks = int(np.ceil(n_samples / batch_size)) - 1
    py = (f"from src.sweeps.sensitivity import run_batch_from_file; "
          f"run_batch_from_file(range($START, $END + 1), "
          f"'results/sensitivity/lhs_samples.npy', 'results/sensitivity')")
    path = os.path.join(outdir, "submit_sensitivity.sh")
    with open(path, "w") as f:
        f.write(_script("sens", outdir, ntasks, batch_size, n_samples,
                        py, time))
    print(f"[slurm] wrote {path} "
          f"({n_samples} samples / {batch_size} per task = "
          f"{ntasks + 1} tasks)")
    print(f"[slurm] submit: sbatch {path}")
    return path


def generate_robustness_job_array(top_params=None, n_samples=3000,
                                  batch_size=10, outdir=None,
                                  time="24:00:00",
                                  samples_filename="robust_samples.npy",
                                  generate_samples=True):
    """Generate the Stage 2 job-array script (+ robustness sample file).

    Args:
        top_params (list[str] | None): sensitive parameters from Stage 1
            (default: first 4).
        samples_filename (str): name of the pre-generated samples file
            (e.g. ``sobol_200k_samples.npy``) in ``results/robustness/``.
        generate_samples (bool): write the sample file if it does not
            exist yet (LHS fallback; pass False when samples were
            generated upstream, e.g. Sobol with Stage-1 medians).

    Returns:
        str: script path.
    """
    outdir = outdir or ROBUSTNESS_DIR
    os.makedirs(outdir, exist_ok=True)
    top_params = top_params or PARAM_NAMES[:4]

    samples_path = os.path.join(outdir, samples_filename)
    if generate_samples and not os.path.exists(samples_path):
        np.save(samples_path,
                generate_robust_samples(top_params, n_samples, seed=7))
        print(f"[slurm] wrote {samples_path} (swept: {top_params})")

    ntasks = int(np.ceil(n_samples / batch_size)) - 1
    py = (f"from src.sweeps.robustness import run_robustness_batch_from_file; "
          f"run_robustness_batch_from_file(range($START, $END + 1), "
          f"'results/robustness/{samples_filename}', 'results/robustness')")
    path = os.path.join(outdir, "submit_robustness.sh")
    with open(path, "w") as f:
        f.write(_script("robust", outdir, ntasks, batch_size, n_samples,
                        py, time))
    print(f"[slurm] wrote {path} ({n_samples} sets x "
          f"{len(_scenario_names())} scenarios, {ntasks + 1} tasks)")
    print(f"[slurm] submit: sbatch {path}")
    return path


def _scenario_names():
    from src.sweeps.robustness import SCENARIO_NAMES
    return SCENARIO_NAMES


def merge_batches(pattern, out_path):
    """Merge per-batch CSVs into one file (header from the first file)."""
    import glob
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[merge] no files matching {pattern}")
        return None
    with open(out_path, "w", newline="") as out:
        with open(files[0]) as f0:
            header = f0.readline()
        out.write(header)
        for fp in files:
            with open(fp) as f:
                f.readline()  # skip header
                for line in f:
                    out.write(line)
    print(f"[merge] {len(files)} files -> {out_path}")
    return out_path


def monitor_jobs(partition="ais-cpu"):
    """Print monitoring / merge commands for the running sweep."""
    print(f"""
# --- monitor ---
squeue -u $USER -p {partition}
# completed batches
ls {RESULTS_DIR}/batch_*.csv 2>/dev/null | wc -l
# failed tasks
grep -l -iE 'error|traceback' {RESULTS_DIR}/slurm_*.err 2>/dev/null | wc -l
# --- merge Stage 1 results ---
python3 -c "from src.sweeps.slurm import merge_batches; \\
 merge_batches('{RESULTS_DIR}/batch_*.csv', '{RESULTS_DIR}/all_results.csv')"
""")
