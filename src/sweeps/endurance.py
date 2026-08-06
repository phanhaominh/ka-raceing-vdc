"""Endurance simulation: a ~23 km multi-lap run with tire degradation and
battery state-of-charge tracking, reusing the VDC controller and the
pure-pursuit driver from the autocross pipeline.

Key question addressed by :func:`endurance_test`: do the traction-control
parameters (``tc_slip_target``, ``tc_aggressiveness``) - statistically
inert on a single autocross lap - matter over a full endurance distance as
tires degrade?  The test compares an autocross calibration (tight slip
target, aggressive cut) against an endurance calibration (looser slip
target, gentler cut) over the same 11-lap run, with all other parameters
held equal.

Runnable as::

    python3 -c "from src.sweeps.endurance import endurance_test; endurance_test()"
"""

import os

import numpy as np

from src.sweeps.track import TRACK_WAYPOINTS, build_track
from src.sweeps.sensitivity import run_lap, grip_scaling

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
VALIDATION_DIR = os.path.join(_PROJECT_ROOT, "results", "validation")

# Endurance layout: 4 concatenated autocross sectors -> ~2.3 km circuit
LAPS_PER_SECTOR = 4
LAP_OFFSET_X = 540.0       # m between sector starts (covers the ~552 m lap)
N_ENDURANCE_LAPS = 11      # 11 x ~2.3 km ~= 24.9 km
DEGRADATION_RATE = 0.018   # grip lost per completed lap (~82% after 10)
GRIP_FLOOR = 0.60

BATTERY_CAPACITY_J = 25.2e6   # 7 kWh usable accumulator
BATTERY_POWER_W = 116000.0    # peak discharge (matches max_motor_power_total)
BATTERY_LOW_SOC = 0.20        # below this the power derates linearly


def _make_endurance_waypoints():
    """Concatenate 4 copies of the autocross track, offset in X."""
    parts = []
    for i in range(LAPS_PER_SECTOR):
        shifted = TRACK_WAYPOINTS.copy()
        shifted[:, 0] += i * LAP_OFFSET_X
        parts.append(shifted)
    return np.vstack(parts)


ENDURANCE_WAYPOINTS = _make_endurance_waypoints()

_ENDURANCE_TRACK = None


def build_endurance_track():
    """Build (once) the ~2.3 km endurance circuit."""
    global _ENDURANCE_TRACK
    if _ENDURANCE_TRACK is None:
        _ENDURANCE_TRACK = build_track(ENDURANCE_WAYPOINTS)
    return _ENDURANCE_TRACK


def tire_grip_scale(lap_number, rate=DEGRADATION_RATE):
    """Grip multiplier after ``lap_number`` completed laps (0-based).

    ``lap 0`` = fresh tires (1.0); each lap loses ``rate`` of grip down to
    ``GRIP_FLOOR``.  0.018/lap -> ~0.82 after 10 laps.
    """
    return max(GRIP_FLOOR, 1.0 - rate * lap_number)


class EnduranceBattery:
    """Battery state-of-charge tracker with low-SoC power derating.

    Energy accounting uses the net (regen-recovered) energy per lap; the
    available power is patched into the vehicle's motor power limit so the
    powertrain is power-limited once SoC drops below ``low_soc``.
    """

    def __init__(self, capacity_j=BATTERY_CAPACITY_J,
                 power_w=BATTERY_POWER_W, low_soc=BATTERY_LOW_SOC):
        self.capacity = capacity_j
        self.power_limit = power_w
        self.low_soc = low_soc
        self.net_energy = 0.0

    @property
    def soc(self):
        """State of charge [0-1]."""
        return max(0.0, 1.0 - self.net_energy / self.capacity)

    @property
    def available_power(self):
        """Total motor power available [W]; derated below low_soc."""
        s = self.soc
        if s > self.low_soc:
            return self.power_limit
        return self.power_limit * max(0.2, s / self.low_soc)

    @property
    def depleted(self):
        return self.soc <= 0.0

    def consume(self, energy_j):
        """Record net energy drawn for one lap [J]."""
        self.net_energy += max(0.0, energy_j)


def endurance_lap(params_dict, track, lap_number, grip_scale, battery,
                  seed=None, dt=0.003, max_time=400.0):
    """Run one endurance lap with degraded tires and battery power limit.

    The grip scale is applied through the module's :func:`grip_scaling`
    context (so the whole model, including the kappa solver, sees the
    reduced grip); the battery's available power is temporarily patched
    into ``KIT25E_PARAMS.max_motor_power_total`` for the lap and restored
    afterwards (the simulation is single-threaded).

    Returns the :func:`run_lap` metrics plus ``lap_number``, ``grip_scale``,
    ``battery_soc`` and ``power_limited``.
    """
    import src.vehicle as veh

    orig_power = veh.KIT25E_PARAMS.max_motor_power_total
    try:
        veh.KIT25E_PARAMS.max_motor_power_total = battery.available_power
        with grip_scaling(grip_scale):
            r = run_lap(params_dict, track=track, grip_scale=1.0,
                        driver_seed=seed, dt=dt, max_time=max_time)
    finally:
        veh.KIT25E_PARAMS.max_motor_power_total = orig_power

    battery.consume(r["net_energy_used"])
    r["lap_number"] = lap_number
    r["grip_scale"] = grip_scale
    r["battery_soc"] = battery.soc
    r["power_limited"] = battery.available_power < 0.95 * battery.power_limit
    return r


def run_endurance(params_dict, track=None, n_laps=N_ENDURANCE_LAPS,
                  degradation_rate=DEGRADATION_RATE, seed=42):
    """Simulate a full endurance run with tire degradation + battery.

    Args:
        params_dict (dict): VDCParams field values.
        track (Track | None): circuit to drive (default endurance circuit).
        n_laps (int): number of laps.
        degradation_rate (float): grip lost per lap.
        seed (int): base RNG seed (per-lap seed = seed + lap).

    Returns:
        dict: per-lap results, finish status, totals, SoC, grip trace.
    """
    track = track or build_endurance_track()
    battery = EnduranceBattery()
    laps, grip_trace = [], []
    total_time = total_energy = 0.0
    dnf_reason = None

    for lap in range(n_laps):
        grip = tire_grip_scale(lap, degradation_rate)
        grip_trace.append(grip)
        r = endurance_lap(params_dict, track, lap, grip, battery,
                          seed=seed + lap)
        laps.append(r)
        total_time += r["lap_time"]
        total_energy += r["net_energy_used"]
        if r["crashed"]:
            dnf_reason = "crash"
            break
        if not r["completed"]:
            dnf_reason = "did_not_complete"
            break
        if battery.depleted:
            dnf_reason = "battery"
            break

    return dict(
        laps=laps,
        finished=dnf_reason is None,
        total_distance=track.total * len(laps),
        total_time=total_time,
        total_energy=total_energy,
        final_soc=battery.soc,
        grip_trace=grip_trace,
        dnf_reason=dnf_reason,
    )


# --------------------------------------------------------------------------
# Comparison test
# --------------------------------------------------------------------------
def endurance_test():
    """Compare autocross vs endurance calibration over a ~23 km run.

    Both calibrations share the yaw-loop gains (Stage 1 showed those
    dominate single-lap time); they differ only in the TC parameters -
    the point of the experiment.  Prints a per-lap comparison and saves
    ``results/validation/endurance_comparison.png``.

    Runnable as::

        python3 -c "from src.sweeps.endurance import endurance_test; endurance_test()"
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    track = build_endurance_track()
    n_laps = N_ENDURANCE_LAPS

    # Autocross calibration: tight TC (aggressive slip control)
    ac = dict(yaw_gain=1000.0, yaw_damping=20.0, tc_slip_target=0.12,
              tc_aggressiveness=0.8, regen_ratio=0.8, brake_bias_front=0.60)
    # Endurance calibration: looser TC (let degraded tires slip more)
    end = dict(yaw_gain=1000.0, yaw_damping=20.0, tc_slip_target=0.16,
               tc_aggressiveness=0.4, regen_ratio=0.8, brake_bias_front=0.60)

    print(f"[endurance_test] circuit {track.total:.0f} m, {n_laps} laps "
          f"(~{track.total * n_laps / 1000:.1f} km)")
    print("[endurance_test] running autocross calibration ...")
    ra = run_endurance(ac, track, n_laps)
    print("[endurance_test] running endurance calibration ...")
    re_ = run_endurance(end, track, n_laps)

    # --- summary table -------------------------------------------------------
    print("\n===== Endurance comparison (~%.1f km, tire degradation) ====="
          % (track.total * n_laps / 1000))
    print(f"{'Metric':<28s} {'Autocross':>12s} {'Endurance':>12s}")
    print("-" * 54)
    print(f"{'Finished':<28s} {str(ra['finished']):>12s} "
          f"{str(re_['finished']):>12s}")
    if ra["dnf_reason"] or re_["dnf_reason"]:
        print(f"{'DNF reason':<28s} {str(ra['dnf_reason']):>12s} "
              f"{str(re_['dnf_reason']):>12s}")
    print(f"{'Total time [s]':<28s} {ra['total_time']:>12.1f} "
          f"{re_['total_time']:>12.1f}")
    print(f"{'Total energy (net) [MJ]':<28s} {ra['total_energy'] / 1e6:>12.2f} "
          f"{re_['total_energy'] / 1e6:>12.2f}")
    print(f"{'Final SoC':<28s} {ra['final_soc']:>11.1%} "
          f"{re_['final_soc']:>11.1%}")
    print(f"{'Final grip':<28s} {ra['grip_trace'][-1]:>11.1%} "
          f"{re_['grip_trace'][-1]:>11.1%}")

    # --- per-lap table --------------------------------------------------------
    print(f"\n{'Lap':>4s} {'grip':>6s} {'AC lap [s]':>10s} "
          f"{'End lap [s]':>10s} {'AC kJ':>8s} {'End kJ':>8s} "
          f"{'AC peak k':>9s} {'End peak k':>9s}")
    n = max(len(ra["laps"]), len(re_["laps"]))
    for i in range(n):
        la = ra["laps"][i] if i < len(ra["laps"]) else {}
        le = re_["laps"][i] if i < len(re_["laps"]) else {}
        g = ra["grip_trace"][i] if i < len(ra["grip_trace"]) else 1.0
        print(f"{i:4d} {g:6.3f} "
              f"{la.get('lap_time', float('nan')):>10.2f} "
              f"{le.get('lap_time', float('nan')):>10.2f} "
              f"{la.get('net_energy_used', 0) / 1e3:>8.0f} "
              f"{le.get('net_energy_used', 0) / 1e3:>8.0f} "
              f"{la.get('peak_slip', float('nan')):>9.3f} "
              f"{le.get('peak_slip', float('nan')):>9.3f}")

    # persist per-lap data (lap, grip, lap time, net energy, SoC, peak slip)
    import csv as _csv
    laps_csv = os.path.join(VALIDATION_DIR, "endurance_laps.csv")
    with open(laps_csv, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["lap", "grip", "calib", "lap_time_s", "net_energy_kJ",
                    "soc", "peak_slip"])
        for name, res in [("autocross", ra), ("endurance", re_)]:
            for l in res["laps"]:
                w.writerow([l["lap_number"], round(l["grip_scale"], 4),
                            name, round(l["lap_time"], 2),
                            round(l["net_energy_used"] / 1e3, 1),
                            round(l["battery_soc"], 4),
                            round(l["peak_slip"], 4)])
    print(f"Saved per-lap data: {laps_csv}")

    # --- TC engagement check (the key question) --------------------------------
    print("\nTC engagement check (peak longitudinal slip vs threshold):")
    for name, res, thr in [("autocross", ra, ac["tc_slip_target"]),
                           ("endurance", re_, end["tc_slip_target"])]:
        peaks = [l["peak_slip"] for l in res["laps"]]
        engaged = sum(1 for p in peaks if p > thr)
        print(f"  {name:<10s} peak_slip {max(peaks):.3f} vs "
              f"threshold {thr:.2f} -> TC would cut in {engaged}/"
              f"{len(peaks)} laps")

    # --- plot ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    lap_nums = list(range(1, n_laps + 1))
    axes[0].plot(lap_nums[:len(ra["laps"])], [l["lap_time"] for l in ra["laps"]],
                 "b-o", label="Autocross calib.")
    axes[0].plot(lap_nums[:len(re_["laps"])], [l["lap_time"] for l in re_["laps"]],
                 "r-s", label="Endurance calib.")
    axes[0].set_xlabel("Lap")
    axes[0].set_ylabel("Lap time [s]")
    axes[0].set_title("Lap Time Progression")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(lap_nums[:len(ra["grip_trace"])], ra["grip_trace"], "b-o",
                 label="tire grip")
    axes[1].set_xlabel("Lap")
    axes[1].set_ylabel("Grip scale [-]")
    axes[1].set_title("Tire Degradation")
    axes[1].grid(True)

    axes[2].plot(lap_nums[:len(ra["laps"])], [l["peak_slip"] for l in ra["laps"]],
                 "b-o", label="Autocross peak slip")
    axes[2].plot(lap_nums[:len(re_["laps"])], [l["peak_slip"] for l in re_["laps"]],
                 "r-s", label="Endurance peak slip")
    axes[2].axhline(ac["tc_slip_target"], color="b", ls=":", label="AC TC thr")
    axes[2].axhline(end["tc_slip_target"], color="r", ls=":", label="End TC thr")
    axes[2].set_xlabel("Lap")
    axes[2].set_ylabel("Peak slip ratio [-]")
    axes[2].set_title("Peak Longitudinal Slip")
    axes[2].legend(fontsize=8)
    axes[2].grid(True)

    fig.suptitle(f"KIT25e Endurance - {track.total * n_laps / 1000:.1f} km, "
                 "tire degradation + battery")
    fig.tight_layout()
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    out_path = os.path.join(VALIDATION_DIR, "endurance_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")
    return ra, re_


if __name__ == "__main__":
    endurance_test()
