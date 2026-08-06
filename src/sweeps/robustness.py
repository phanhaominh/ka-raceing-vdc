"""Stage 2: robust optimization over the sensitive VDC parameters.

Each parameter set is evaluated across multiple grip / driver scenarios
(nominal, low/high grip, cold tires, driver variation); the objective is

    J = mean(lap_time) + 0.5 * std(lap_time) + 10.0 * P(crash)

with crash = |yaw rate| > 3 rad/s or off-track (> 2.5 m from center line).
Produces the Pareto frontier (fast vs consistent) from the evaluated sets.

Local smoke test::

    python3 -c "from src.sweeps.robustness import test_robustness; test_robustness(2)"
"""

import csv
import os

import numpy as np

from src.sweeps.sensitivity import (run_lap, params_from_sample,
                                    PARAM_NAMES, PARAM_BOUNDS, CSV_FIELDS,
                                    RESULTS_DIR, _read_csv, generate_lhs_samples)
from src.controller import VDCParams

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
ROBUSTNESS_DIR = os.path.join(_PROJECT_ROOT, "results", "robustness")

# Scenario grid: each parameter set is evaluated in all of these.
# ``warmup`` = unscored laps run first (cold-tire scenarios; identical until
# a thermal model exists, kept for pipeline structure).
SCENARIOS = [
    dict(name="nominal",      grip_scale=1.00, driver_noise=0.00,
         driver_seed=None, warmup=0),
    dict(name="low_grip",     grip_scale=0.85, driver_noise=0.00,
         driver_seed=None, warmup=0),
    dict(name="high_grip",    grip_scale=1.15, driver_noise=0.00,
         driver_seed=None, warmup=0),
    dict(name="cold_tires",   grip_scale=0.90, driver_noise=0.00,
         driver_seed=None, warmup=2),
    dict(name="driver_var_1", grip_scale=1.00, driver_noise=0.03,
         driver_seed=101, warmup=0),
    dict(name="driver_var_2", grip_scale=1.00, driver_noise=0.03,
         driver_seed=102, warmup=0),
    dict(name="driver_var_3", grip_scale=1.00, driver_noise=0.03,
         driver_seed=103, warmup=0),
]
SCENARIO_NAMES = [s["name"] for s in SCENARIOS]


# --------------------------------------------------------------------------
# Sensitivity ranking (Stage 1 -> Stage 2)
# --------------------------------------------------------------------------
_METRICS = ["lap_time", "yaw_error_rms", "energy_used"]


def _correlation_table(csv_path):
    """Pearson r of each parameter with the key metrics (valid laps only).

    Returns:
        tuple: (corr dict name->metric->r, number of valid laps).
    """
    rows = [r for r in _read_csv(csv_path)
            if not r.get("crashed") and r.get("completed")]
    corr = {}
    for name in PARAM_NAMES:
        corr[name] = {}
        for m in _METRICS:
            x = np.array([r[f"param_{name}"] for r in rows], dtype=float)
            y = np.array([r[m] for r in rows], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            c = (np.corrcoef(x[mask], y[mask])[0, 1]
                 if mask.sum() > 20 else 0.0)
            corr[name][m] = float(c) if np.isfinite(c) else 0.0
    return corr, len(rows)


def select_sensitive_parameters(csv_path, n_top=4):
    """Rank parameters by importance = max |r| across lap_time,
    yaw_error_rms and energy_used.

    Lap time alone is driver-dominated on this track (std ~0.03 s over
    50k laps), so ranking by |r(lap_time)| alone would be noise; the
    composite captures influence on any performance metric.

    Returns:
        list[str]: the ``n_top`` most sensitive PARAM_NAMES.
    """
    corr, n = _correlation_table(csv_path)
    if n < 20:
        return PARAM_NAMES[:n_top]
    importance = {name: max(abs(v) for v in corr[name].values())
                  for name in PARAM_NAMES}
    ranked = sorted(PARAM_NAMES, key=lambda nm: -importance[nm])
    return ranked[:n_top]


def analyze_stage1(csv_path, out_png=None):
    """Stage-1 sensitivity analysis: correlation table, ranking, heatmap.

    Computes Pearson r of each of the 6 VDC parameters with lap_time,
    yaw_error_rms and energy_used over the non-crashed, completed laps;
    ranks by importance (max |r| across the three metrics); optionally
    saves a 6x3 heatmap to ``out_png``.

    Returns:
        tuple: (top4, medians, corr) where corr[name][metric] = r and
        medians[name] = Stage-1 median of the parameter.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corr, n_rows = _correlation_table(csv_path)
    rows = [r for r in _read_csv(csv_path)
            if not r.get("crashed") and r.get("completed")]
    medians = {name: float(np.median([r[f"param_{name}"] for r in rows]))
               for name in PARAM_NAMES}
    importance = {name: max(abs(v) for v in corr[name].values())
                  for name in PARAM_NAMES}
    ranked = sorted(PARAM_NAMES, key=lambda nm: -importance[nm])
    top4 = ranked[:4]

    print(f"\n[analyze_stage1] {n_rows} valid laps loaded")
    print("Rank | Parameter          | lap_time | yaw_rms | energy  "
          "| importance(max|r|)")
    for i, name in enumerate(ranked, 1):
        c = corr[name]
        print(f"{i:>4} | {name:<19s} | {c['lap_time']:+.4f} | "
              f"{c['yaw_error_rms']:+.4f} | {c['energy_used']:+.4f} | "
              f"{importance[name]:.4f}")

    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        mat = np.array([[corr[n][m] for m in _METRICS] for n in ranked])
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(_METRICS)))
        ax.set_xticklabels(_METRICS)
        ax.set_yticks(range(len(ranked)))
        ax.set_yticklabels(ranked)
        ax.set_title(f"VDC Parameter Sensitivity - Stage 1 ({n_rows} laps)")
        for i in range(len(ranked)):
            for j in range(len(_METRICS)):
                ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if abs(mat[i, j]) > 0.5 else "black")
        fig.colorbar(im, ax=ax, label="Pearson r")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"[analyze_stage1] heatmap -> {out_png}")

    return top4, medians, corr


def generate_sobol_samples(top_params, n_samples, fixed_values, seed=123):
    """Sobol sequence over ``top_params``; the rest fixed at values.

    Quasi-random (low-discrepancy) coverage of the swept dimensions with
    ``n_samples`` points; non-swept parameters take the values in
    ``fixed_values`` (e.g. the Stage-1 medians).

    Returns:
        ndarray (n_samples, 6) in PARAM_NAMES order.
    """
    from scipy.stats import qmc

    d = len(top_params)
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(n_samples)))
    u = sampler.random_base2(m)[:n_samples]
    out = np.zeros((n_samples, len(PARAM_NAMES)))
    for i, name in enumerate(PARAM_NAMES):
        if name in top_params:
            j = top_params.index(name)
            lo, hi = PARAM_BOUNDS[name]
            out[:, i] = lo + u[:, j] * (hi - lo)
        else:
            out[:, i] = fixed_values.get(name, 0.0)
    return out


def generate_robust_samples(top_params, n_samples, seed=7):
    """LHS over the top parameters; the rest pinned to VDCParams defaults.

    Returns:
        ndarray (n_samples, 6) in PARAM_NAMES order.
    """
    defaults = VDCParams()
    rng = np.random.default_rng(seed)
    d = len(top_params)
    u = np.empty((n_samples, d))
    for j in range(d):
        u[:, j] = (rng.permutation(n_samples) + rng.random(n_samples)) \
            / n_samples
    out = np.zeros((n_samples, len(PARAM_NAMES)))
    for i, name in enumerate(PARAM_NAMES):
        if name in top_params:
            j = top_params.index(name)
            lo, hi = PARAM_BOUNDS[name]
            out[:, i] = lo + u[:, j] * (hi - lo)
        else:
            out[:, i] = getattr(defaults, name)
    return out


# --------------------------------------------------------------------------
# Scenario evaluation
# --------------------------------------------------------------------------
def evaluate_set(params_dict, scenarios=SCENARIOS):
    """Run all scenarios for one parameter set; return objective metrics.

    Returns:
        dict: mean_lap, std_lap, p_crash, J, lap_<scenario> per scenario.
    """
    laps, crashed = [], []
    per_scen = {}
    for sc in scenarios:
        r = run_lap(params_dict, grip_scale=sc["grip_scale"],
                    driver_noise=sc["driver_noise"],
                    driver_seed=sc["driver_seed"],
                    warmup_laps=sc["warmup"])
        laps.append(r["lap_time"])
        crashed.append(r["crashed"])
        per_scen[sc["name"]] = r["lap_time"]
    laps = np.asarray(laps, dtype=float)
    p_crash = float(np.mean(crashed))
    mean_lap, std_lap = float(np.mean(laps)), float(np.std(laps))
    return dict(mean_lap=mean_lap, std_lap=std_lap, p_crash=p_crash,
                J=float(mean_lap + 0.5 * std_lap + 10.0 * p_crash),
                **{f"lap_{k}": v for k, v in per_scen.items()})


ROBUST_FIELDS = (["sample_id"] + [f"param_{n}" for n in PARAM_NAMES] +
                 ["mean_lap", "std_lap", "p_crash", "J"] +
                 [f"lap_{s}" for s in SCENARIO_NAMES])


def run_robustness_batch(sample_indices, samples, output_dir=None):
    """Evaluate a batch of parameter sets; write one CSV per batch."""
    output_dir = output_dir or ROBUSTNESS_DIR
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for idx in sample_indices:
        sample = samples[idx]
        row = {"sample_id": int(idx)}
        for i, name in enumerate(PARAM_NAMES):
            row[f"param_{name}"] = float(sample[i])
        row.update(evaluate_set(params_from_sample(sample)))
        rows.append(row)
    path = os.path.join(output_dir, f"robust_{int(sample_indices[0]):05d}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROBUST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def run_robustness_batch_from_file(sample_indices, samples_path,
                                   output_dir=None):
    samples = np.load(samples_path)
    return run_robustness_batch(sample_indices, samples, output_dir)


# --------------------------------------------------------------------------
# Pareto frontier
# --------------------------------------------------------------------------
def pareto_frontier(csv_path, out_path=None):
    """Non-dominated parameter sets on (mean_lap, std_lap), both minimized.

    Returns:
        list[dict]: the Pareto-optimal rows.
    """
    rows = [r for r in _read_csv(csv_path)]
    mean = np.array([r["mean_lap"] for r in rows])
    std = np.array([r["std_lap"] for r in rows])
    dominated = np.zeros(len(rows), dtype=bool)
    for i in range(len(rows)):
        for j in range(len(rows)):
            if i == j:
                continue
            if (mean[j] <= mean[i] and std[j] <= std[i]
                    and (mean[j] < mean[i] or std[j] < std[i])):
                dominated[i] = True
                break
    pareto = [rows[i] for i in range(len(rows)) if not dominated[i]]
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ROBUST_FIELDS)
            w.writeheader()
            w.writerows(pareto)
    return pareto


# --------------------------------------------------------------------------
# Local smoke test
# --------------------------------------------------------------------------
def test_robustness(n_sets=2):
    """Local smoke test: evaluate n_sets parameter sets across scenarios."""
    samples = generate_robust_samples(PARAM_NAMES[:3], n_sets, seed=1)
    path = run_robustness_batch(list(range(n_sets)), samples)
    rows = _read_csv(path)
    for r in rows:
        print(f"  set {r['sample_id']}: J={r['J']:.2f} "
              f"mean={r['mean_lap']:.2f}s std={r['std_lap']:.2f}s "
              f"P(crash)={r['p_crash']:.2f}")
    pareto = pareto_frontier(path)
    print(f"[test_robustness] wrote {path} ({len(rows)} sets, "
          f"{len(pareto)} pareto-optimal)")
    return path
