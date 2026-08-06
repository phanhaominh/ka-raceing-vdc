"""Top-level Phase 3 pipeline: Stage 1 sensitivity -> Stage 2 robustness.

Run from the project root (so ``src`` is importable)::

    python3 -m src.sweeps.run_sweep --stage 1 --n-samples 1000 --local-test
    python3 -m src.sweeps.run_sweep --stage all --n-samples 5000 --submit

Stage 1 generates the LHS samples and a Slurm job array; Stage 2 ranks the
sensitive parameters from the merged Stage 1 results, generates the robust
samples and a Stage 2 array that can depend on the Stage 1 job.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

from src.sweeps import sensitivity, robustness, slurm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _sbatch(script, dependency=None):
    """Submit a script with sbatch; returns the job id (or None)."""
    cmd = ["sbatch"]
    if dependency:
        cmd += ["--dependency", f"afterany:{dependency}"]
    cmd.append(script)
    out = subprocess.run(cmd, capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip())
    if out.returncode == 0:
        jobid = out.stdout.strip().split()[-1]
        return jobid
    return None


def stage1(args):
    """LHS samples + local test + (optional) submit Stage 1 array."""
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "sensitivity"),
                exist_ok=True)
    samples = sensitivity.generate_lhs_samples(args.n_samples, seed=42)
    samples_path = os.path.join(PROJECT_ROOT, "results", "sensitivity",
                                "lhs_samples.npy")
    np.save(samples_path, samples)
    print(f"[stage1] wrote {samples_path} ({args.n_samples} samples)")

    if args.local_test:
        sensitivity.test_sensitivity(n_test=10)

    script = slurm.generate_sensitivity_job_array(
        args.n_samples, batch_size=args.batch_size)
    jobid = None
    if args.submit:
        jobid = _sbatch(script)
    print(f"[stage1] submit with: sbatch {script}")
    return jobid


def stage2(args, stage1_jobid=None):
    """Rank sensitive params, generate robust samples, submit Stage 2."""
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "robustness"),
                exist_ok=True)
    merged = os.path.join(PROJECT_ROOT, "results", "sensitivity",
                          "all_results.csv")
    if not os.path.exists(merged):
        print(f"[stage2] {merged} not found - using the local test CSV")
        test_csv = os.path.join(PROJECT_ROOT, "results", "sensitivity",
                                "batch_00000.csv")
        merged = test_csv if os.path.exists(test_csv) else None
    if merged:
        top = robustness.select_sensitive_parameters(merged, n_top=4)
    else:
        top = sensitivity.PARAM_NAMES[:4]
    print(f"[stage2] sensitive parameters: {top}")

    script = slurm.generate_robustness_job_array(
        top_params=top, n_samples=args.n_robust,
        batch_size=args.robust_batch_size)
    if args.submit:
        jobid = _sbatch(script, dependency=stage1_jobid)
        print(f"[stage2] submitted (depends on stage1 job {stage1_jobid})")
    else:
        print(f"[stage2] submit with: sbatch {script}")

    slurm.monitor_jobs()


def main():
    ap = argparse.ArgumentParser(
        description="Phase 3 VDC parameter sweep pipeline (Zhores)")
    ap.add_argument("--stage", choices=["1", "2", "all"], default="all")
    ap.add_argument("--n-samples", type=int, default=5000,
                    help="Stage 1 LHS sample count")
    ap.add_argument("--n-robust", type=int, default=3000,
                    help="Stage 2 parameter-set count")
    ap.add_argument("--batch-size", type=int, default=20,
                    help="Stage 1 samples per Slurm task")
    ap.add_argument("--robust-batch-size", type=int, default=10,
                    help="Stage 2 parameter sets per Slurm task")
    ap.add_argument("--local-test", action="store_true",
                    help="run the 10-sample local test first")
    ap.add_argument("--submit", action="store_true",
                    help="actually submit the Slurm arrays")
    args = ap.parse_args()

    if args.stage in ("1", "all"):
        jobid = stage1(args)
    else:
        jobid = None

    if args.stage in ("2", "all"):
        stage2(args, stage1_jobid=jobid)


if __name__ == "__main__":
    sys.exit(main())
