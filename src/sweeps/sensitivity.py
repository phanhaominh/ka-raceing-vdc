"""Stage 1: Latin-Hypercube sensitivity analysis over the 6 VDC parameters.

Each sample runs one full lap (autocross-like track, pure-pursuit driver,
VDC controller) and reports lap time, yaw-error RMS, peak slip, energy use
and crash status.  Results are written as per-batch CSVs (one per Slurm
array task) with a consistent schema shared with Stage 2.

Local test::

    python3 -c "from src.sweeps.sensitivity import test_sensitivity; test_sensitivity(10)"
"""

import contextlib
import csv
import os

import numpy as np

from src.vehicle import (KIT25E_PARAMS, VehicleState, step,
                         compute_slip_angles, _solve_kappa)
from src.controller import VDCParams, VDCController
from src.sweeps.track import build_track, PurePursuitDriver, TRACK_WIDTH

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "sensitivity")
VALIDATION_DIR = os.path.join(_PROJECT_ROOT, "results", "validation")

# Six swept VDC parameters: name -> (VDCParams field, [lo, hi])
PARAM_BOUNDS = {
    "yaw_gain": (500.0, 2000.0),
    "yaw_damping": (0.0, 60.0),
    "tc_slip_target": (0.05, 0.20),
    "tc_aggressiveness": (0.1, 1.0),
    "regen_ratio": (0.0, 1.0),
    "brake_bias_front": (0.55, 0.70),
}
PARAM_NAMES = list(PARAM_BOUNDS)

DT = 0.003
MAX_LAP_TIME = 90.0
REGEN_EFFICIENCY = 0.8   # fraction of regen power recovered to the battery
CRASH_DEV = TRACK_WIDTH / 2.0 + 1.0   # 2.5 m off center line = off-track
CRASH_YAW = 3.0                       # rad/s spin-out threshold
FINISH_MARGIN = 1.0                   # m from the path end = lap complete

_TRACK = None


def get_track():
    """Lazily build the shared track."""
    global _TRACK
    if _TRACK is None:
        _TRACK = build_track()
    return _TRACK


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def generate_lhs_samples(n_samples, seed=42):
    """Latin Hypercube samples over the 6 VDC parameters (numpy-only).

    Each parameter dimension is stratified into ``n_samples`` bins and one
    sample drawn per bin (randomized permutation), giving good coverage of
    the 6-D box with ``n_samples`` points.  Reproducible via ``seed``.

    Returns:
        ndarray (n_samples, 6): scaled parameter values in PARAM_NAMES order.
    """
    rng = np.random.default_rng(seed)
    u = np.empty((n_samples, len(PARAM_NAMES)))
    for j in range(len(PARAM_NAMES)):
        perm = rng.permutation(n_samples)
        u[:, j] = (perm + rng.random(n_samples)) / n_samples
    bounds = np.array([PARAM_BOUNDS[n] for n in PARAM_NAMES], dtype=float)
    return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])


def params_from_sample(sample):
    """Convert a sample row into a VDCParams-compatible dict."""
    return {name: float(sample[i]) for i, name in enumerate(PARAM_NAMES)}


# --------------------------------------------------------------------------
# Grip scaling (used by robustness scenarios)
# --------------------------------------------------------------------------
@contextlib.contextmanager
def grip_scaling(scale):
    """Temporarily scale every tire force by ``scale``.

    Wraps the ``get_combined`` / ``get_fx`` names in the ``src.vehicle``
    module namespace (both are looked up at call time, including inside the
    slip-ratio solver), so the whole model -- vehicle and VDC controller --
    consistently sees scaled grip.  Restores the originals on exit.
    """
    import src.vehicle as veh

    if abs(scale - 1.0) < 1e-9:
        yield
        return
    orig_comb, orig_fx = veh.get_combined, veh.get_fx

    def _combined(sa, k, fz, cam=0.0):
        fx, fy = orig_comb(sa, k, fz, cam)
        return fx * scale, fy * scale

    veh.get_combined = _combined
    veh.get_fx = lambda k, fz: orig_fx(k, fz) * scale
    try:
        yield
    finally:
        veh.get_combined = orig_comb
        veh.get_fx = orig_fx


# --------------------------------------------------------------------------
# Lap simulation
# --------------------------------------------------------------------------
def run_lap(params_dict, grip_scale=1.0, driver_noise=0.0, driver_seed=None,
            warmup_laps=0, dt=DT, max_time=MAX_LAP_TIME, track=None):
    """Run one scored lap with the given VDC parameters.

    Args:
        params_dict (dict): VDCParams field values.
        grip_scale (float): multiplies all tire forces (scenarios).
        driver_noise (float): std-dev of gaussian noise added to throttle,
            brake and steering (driver-variation scenarios).
        driver_seed (int | None): RNG seed for the driver noise.
        warmup_laps (int): unscored laps run first (cold-tire scenarios;
            identical until a thermal model exists).
        dt (float): integration step [s].
        max_time (float): safety ceiling [s].
        track (Track | None): track to drive; None = default autocross
            track (used by the endurance module for its own circuit).

    Returns:
        dict: lap metrics (see :func:`run_batch` for the CSV schema).
    """
    track = track if track is not None else get_track()
    vp = KIT25E_PARAMS
    vdc = VDCController(vp, VDCParams(**params_dict))
    driver = PurePursuitDriver(track)
    rng = np.random.default_rng(driver_seed) if driver_seed is not None \
        else None

    def lap_once():
        """One out-and-back lap; returns (metrics, completed, crashed)."""
        state = VehicleState(x=float(track.path_x[0]),
                             y=float(track.path_y[0]),
                             heading=_start_heading(track))
        yaw_sq, yaw_n = 0.0, 0
        peak_slip, energy, net_energy = 0.0, 0.0, 0.0
        max_dev, max_prog = 0.0, 0.0
        t = 0.0
        crashed = completed = False
        for i in range(int(max_time / dt)):
            t = i * dt
            steer, thr, brk, v_tgt, r_tgt, idx, dev = driver.control(state, dt)
            max_dev = max(max_dev, dev)
            if rng is not None:
                steer += float(rng.normal(0.0, driver_noise * 120.0))
                thr = float(np.clip(thr + rng.normal(0.0, driver_noise),
                                    0.0, 1.0))
                brk = float(np.clip(brk + rng.normal(0.0, driver_noise),
                                    0.0, 1.0))

            T_motor, bp = vdc.compute_torques(state, steer, thr, brk,
                                              r_tgt, dt)
            state = step(state, vp, steer, thr, bp, T_motor, dt)

            # metrics
            e_r = r_tgt - state.yaw_rate
            yaw_sq += e_r * e_r
            yaw_n += 1
            omega = state.vx / vp.tire_rolling_radius
            pow_abs = np.abs(T_motor) * abs(omega)
            energy += float(np.sum(pow_abs)) * dt
            gross = float(np.sum(np.clip(T_motor, 0.0, None) * abs(omega))) * dt
            regen = float(np.sum(np.clip(-T_motor, 0.0, None) * abs(omega))) * dt
            net_energy += gross - REGEN_EFFICIENCY * regen
            if i % 20 == 0:  # slip metric at reduced rate (cheaper)
                _, Fz = compute_slip_angles(state, vp, steer)
                for j in range(4):
                    k = _solve_kappa(
                        T_motor[j] - bp[j] * (vp.brake_torque_max_total / 4.0),
                        Fz[j], vp.tire_rolling_radius)
                    peak_slip = max(peak_slip, abs(k))
            max_prog = max(max_prog, float(track.path_s[idx]))
            if abs(state.yaw_rate) > CRASH_YAW or dev > CRASH_DEV:
                crashed = True
                break
            if max_prog >= track.total - FINISH_MARGIN:
                completed = True
                break
        return dict(
            lap_time=float(t if (completed or crashed) else max_time),
            completed=int(completed), crashed=int(crashed),
            yaw_error_rms=float(np.sqrt(yaw_sq / max(yaw_n, 1))),
            peak_slip=float(peak_slip), energy_used=float(energy),
            net_energy_used=float(net_energy),
            max_deviation=float(max_dev),
        ), completed, crashed

    with grip_scaling(grip_scale):
        for _ in range(warmup_laps):
            lap_once()  # unscored
        metrics, completed, crashed = lap_once()

    metrics["mean_speed"] = round(track.total / max(metrics["lap_time"], 1e-6),
                                  2)
    return metrics


def _start_heading(track):
    """Initial heading along the first track segment."""
    return float(np.arctan2(track.path_y[1] - track.path_y[0],
                            track.path_x[1] - track.path_x[0]))


# --------------------------------------------------------------------------
# Batch execution + CSV
# --------------------------------------------------------------------------
CSV_FIELDS = (["sample_id"] + [f"param_{n}" for n in PARAM_NAMES] +
              ["lap_time", "completed", "crashed", "yaw_error_rms",
               "peak_slip", "energy_used", "max_deviation", "mean_speed"])


def run_batch(sample_indices, samples, output_dir=None):
    """Run a batch of LHS samples and write one CSV per batch.

    Args:
        sample_indices (sequence): indices into ``samples``.
        samples (ndarray (N,6)): LHS sample array.
        output_dir (str | None): CSV destination (default RESULTS_DIR).

    Returns:
        str: path of the written CSV.
    """
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for idx in sample_indices:
        sample = samples[idx]
        metrics = run_lap(params_from_sample(sample))
        row = {"sample_id": int(idx)}
        for i, name in enumerate(PARAM_NAMES):
            row[f"param_{name}"] = float(sample[i])
        row.update(metrics)
        rows.append(row)
    path = os.path.join(output_dir, f"batch_{int(sample_indices[0]):05d}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def run_batch_from_file(sample_indices, samples_path, output_dir=None):
    """Slurm task entry: load the LHS array and run a batch of indices."""
    samples = np.load(samples_path)
    return run_batch(sample_indices, samples, output_dir)


def run_one():
    """Container entrypoint: read SAMPLE_INDICES / SAMPLES_PATH from env."""
    env = os.environ
    start, end = (int(x) for x in env["SAMPLE_INDICES"].split("-"))
    run_batch_from_file(range(start, end + 1), env["SAMPLES_PATH"],
                        env.get("OUTPUT_DIR"))


# --------------------------------------------------------------------------
# Local test
# --------------------------------------------------------------------------
def test_sensitivity(n_test=10, seed=42):
    """Run n_test LHS samples locally; write CSV + correlation heatmap.

    Returns:
        tuple: (csv_path, png_path).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = generate_lhs_samples(n_test, seed)
    csv_path = run_batch(list(range(n_test)), samples,
                         output_dir=RESULTS_DIR)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    png_path = os.path.join(VALIDATION_DIR, "sensitivity_test.png")

    rows = _read_csv(csv_path)
    names = [f"param_{n}" for n in PARAM_NAMES] + ["lap_time", "yaw_error_rms",
                                                   "peak_slip", "energy_used"]
    data = np.array([[r[k] for k in names] for r in rows], dtype=float)
    corr = np.corrcoef(data, rowvar=False)
    mask = np.isfinite(corr)
    corr = np.where(mask, corr, 0.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Sensitivity correlations ({n_test} LHS samples)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    n_crash = sum(r["crashed"] for r in rows)
    laps = [r["lap_time"] for r in rows]
    print(f"[test_sensitivity] wrote {csv_path} "
          f"({len(rows)} rows, {n_crash} crashed, "
          f"lap times {min(laps):.1f}-{max(laps):.1f} s)")
    print(f"[test_sensitivity] heatmap -> {png_path}")
    return csv_path, png_path


def _read_csv(path):
    """Read a results CSV into a list of dicts (values as float)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows
