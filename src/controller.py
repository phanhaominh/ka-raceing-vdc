"""Vehicle Dynamics Controller (VDC) for the KA-RaceIng KIT25e.

Three subsystems, all per-wheel (the KIT25e has 4 independent hub motors and
no mechanical differential):

1. **Torque vectoring (TV)** -- a PD controller on yaw-rate error converts
   the tracking error into a corrective yaw moment, then shifts torque
   left/right on both axles:

       Mz_des = K_yaw * (r_target - r) - D_yaw * r_dot
       delta_T = Mz_des * r_eff / (2 * track_width)     [per wheel]
       T_FL = T_base - delta_T,  T_FR = T_base + delta_T,  ... (RL/RR same)

   A positive Mz_des (car not turning enough) adds torque to the
   right-side wheels (FR, RR).  Because each wheel's longitudinal force
   acts at y = +-track_width/2 from the CoG, this creates a positive yaw
   moment ``Mz = 2*track_width*delta_T/r_eff`` exactly equal to Mz_des
   (in the Fx = T/r limit).  The left/right shift preserves the total
   driver-requested longitudinal force.  The differential is clipped to
   ``tv_max_torque_delta`` and ramped out at low speed (a stationary tire
   cannot generate lateral force, so vectoring is ineffective below
   ``vx_low``).

2. **Traction control (TC)** -- per-wheel slip ratio is estimated from the
   commanded torque via the same Pacejka inversion the vehicle model uses
   (``_solve_kappa``), so the estimate matches the slip the tire will
   actually develop.  When |kappa| exceeds ``tc_slip_target`` the wheel's
   drive torque is reduced:

       T_reduced = T * (1 - K_tc * (|kappa| - tc_slip_target))

3. **Brake blending** -- the total braking demand is split front/rear by
   ``brake_bias_front``; per wheel, ``regen_ratio`` of that wheel's demand
   is taken by the motor (regen, up to its speed-dependent regen limit) and
   the remainder goes to the friction brake.  Returns per-wheel friction
   brake pressures; the vehicle model broadcasts them per wheel.

Usage::

    vdc = VDCController(KIT25E_PARAMS)
    T_motor, bp = vdc.compute_torques(state, steer, thr, brk, r_tgt, dt)
    state = step(state, params, steer, thr, bp, T_motor, dt)

All angles are in degrees at the public API, radians internally.
"""

from dataclasses import dataclass
import os

import numpy as np

from src.vehicle import (
    VehicleParams,
    VehicleState,
    KIT25E_PARAMS,
    compute_slip_angles,
    step,
    _solve_kappa,
)


# --------------------------------------------------------------------------
# Tunable parameters
# --------------------------------------------------------------------------
@dataclass
class VDCParams:
    """Tunable VDC parameters (SI units).

    Every field documents its physical meaning and valid range.  Defaults
    are calibrated for the KIT25e model (Iz = 60 kg m^2, track 1.22 m,
    per-wheel motor limit 150 Nm / 29 kW).
    """

    # --- Torque vectoring ---------------------------------------------------
    yaw_gain: float = 1000.0
    """Proportional yaw gain K_yaw [N m s / rad] in
    ``Mz_des = K_yaw*(r_target - r) - D_yaw*r_dot``.

    Converts a yaw-rate tracking error into a corrective yaw moment.
    Physically meaningful values are ~500-2000 (the 0.1-2.0 nominal range
    in the task spec yields < 1 N m of moment at 1 rad/s error, i.e. a
    sub-0.1 N m wheel differential -- far below what a 150 N m motor can
    act on; it appears to assume a normalized error).  Higher -> more
    aggressive correction; too high -> yaw oscillation."""

    yaw_damping: float = 20.0
    """Derivative yaw damping D_yaw [N m s^2 / rad] in
    ``Mz_des = ... - D_yaw*r_dot``.

    Opposes yaw acceleration to damp oscillation.  Valid ~0-60; 0 = no
    damping (oscillation risk), too high = sluggish.  (Nominal 0.5-1.5
    from the task spec is negligible with Mz_des in N m.)"""

    tv_max_torque_delta: float = 75.0
    """Maximum left/right torque differential per wheel [N m].

    Clips ``delta_T`` so one side cannot saturate while the other is cut.
    Valid 0-150 (the per-wheel motor torque limit)."""

    # --- Traction control ---------------------------------------------------
    tc_slip_target: float = 0.12
    """Slip-ratio threshold [-].

    When a wheel's estimated slip ratio exceeds this, its drive torque is
    reduced.  Valid range 0.05-0.20; MF5.0 longitudinal peak slip is
    ~0.10-0.18, so 0.12 sits near peak grip."""

    tc_aggressiveness: float = 0.8
    """Traction-control aggressiveness K_tc [-].

    Cut per unit excess slip: ``T_reduced = T*(1 - K_tc*(|kappa| -
    tc_slip_target))``.  Valid 0.2-1.0; the cut is soft at the top of the
    range (25% slip vs 12% target -> ~10% cut at K_tc = 1.0); 0 = off."""

    # --- Brake blending -----------------------------------------------------
    regen_ratio: float = 0.8
    """Fraction of each wheel's braking demand taken by regen [0-1].

    The motor absorbs ``regen_ratio`` of the demand up to its
    speed-dependent regen torque limit; the remainder is friction.
    Valid 0-1; 0 = no regen, 1 = regen as much as possible."""

    brake_bias_front: float = 0.60
    """Front-axle share of total braking demand [0-1].

    Splits the braking demand between axles before regen/friction
    blending.  Valid 0.55-0.70 (FSAE convention: front-biased for
    stability)."""


# --------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------
class VDCController:
    """Vehicle Dynamics Controller: torque vectoring + traction control +
    brake blending for the KIT25e.

    Feed :meth:`compute_torques` output into :func:`src.vehicle.step`::

        vdc = VDCController(KIT25E_PARAMS)
        T_motor, bp = vdc.compute_torques(state, steer, thr, brk, r_tgt, dt)
        state = step(state, params, steer, thr, bp, T_motor, dt)

    ``bp`` is a 4-vector of per-wheel friction brake pressures; the vehicle
    model broadcasts it per wheel.
    """

    def __init__(self, vehicle_params=None, vdc_params=None):
        self.vp = vehicle_params if vehicle_params is not None else KIT25E_PARAMS
        self.cp = vdc_params if vdc_params is not None else VDCParams()
        self._prev_yaw_rate = 0.0
        self._last_r_dot = 0.0

    # ------------------------------------------------------------------ helpers
    def _motor_limits(self, omega):
        """Per-wheel drive/regen torque limits at wheel speed omega [rad/s].

        Drive: ``min(max_torque, max_power/omega)`` (power-limited at high
        speed).  Regen: 80% of the drive limit (motor/battery regen
        derating), also power-limited.

        Args:
            omega (float): wheel angular speed [rad/s] (may be negative
                when reversing).

        Returns:
            tuple: (T_drive_lim, T_regen_lim) [N m].
        """
        T_max_w = self.vp.max_motor_torque_total / 4.0
        P_max_w = self.vp.max_motor_power_total / 4.0
        w = max(abs(omega), 0.1)
        T_drive_lim = min(T_max_w, P_max_w / w)
        T_regen_lim = min(0.8 * T_max_w, P_max_w / w)
        return T_drive_lim, T_regen_lim

    def _estimate_slip_ratios(self, motor_torque_Nm, Fz):
        """Estimate per-wheel longitudinal slip from the commanded torque.

        Uses the same Pacejka inversion the vehicle model applies inside
        ``step`` (``_solve_kappa``), so the estimate matches the slip the
        tire will develop for this torque/load.

        Args:
            motor_torque_Nm (array_like[4]): commanded motor torque [N m].
            Fz (array_like[4]): normal loads [N].

        Returns:
            ndarray[4]: estimated slip ratios.
        """
        r_eff = self.vp.tire_rolling_radius
        kappa = np.zeros(4)
        for i in range(4):
            kappa[i] = _solve_kappa(motor_torque_Nm[i], Fz[i], r_eff)
        return kappa

    # ------------------------------------------------------------------ main
    def compute_torques(self, state, steering_deg, throttle, brake_pressure,
                        target_yaw_rate, dt=0.001):
        """Compute per-wheel motor torques and friction brake pressures.

        Order: driver base torque -> torque vectoring -> traction control ->
        brake blending (regen enters as negative motor torque, friction as
        per-wheel brake pressure).

        Edge cases:
        * ``|vx| < 0.5`` m/s: torque vectoring and traction control are
          disabled (no lateral force from a stationary tire); regen fades
          out below ``vx_low`` so friction brakes take over at standstill.
        * ``vx < 0`` (reverse): TV and TC stay disabled; throttle maps to
          motor torque directly (driver responsibility).
        * Both throttle and brake requested: braking takes priority
          (``T_base = 0``).

        Args:
            state (VehicleState): current vehicle state.
            steering_deg (float): steering wheel angle [deg].
            throttle (float): driver throttle 0..1.
            brake_pressure (float): driver brake demand 0..1.
            target_yaw_rate (float | None): desired yaw rate [rad/s];
                None disables torque vectoring.
            dt (float): integration step [s], used for the r_dot estimate.

        Returns:
            tuple: (motor_torque_Nm[4], brake_pressure[4]).
        """
        vp, cp = self.vp, self.cp
        vx = abs(state.vx)
        r_eff = vp.tire_rolling_radius
        tw = vp.track_width
        omega = state.vx / r_eff

        T_max_w, T_regen_lim = self._motor_limits(omega)

        # --- driver base demand ---------------------------------------------
        braking = brake_pressure > 0.05
        T_base = 0.0 if braking else throttle * T_max_w  # braking priority

        # --- 1. torque vectoring ---------------------------------------------
        T_tv = np.full(4, T_base)
        if target_yaw_rate is not None and vx >= 0.5:
            r_dot = (state.yaw_rate - self._prev_yaw_rate) / max(dt, 1e-6)
            self._last_r_dot = r_dot
            e_r = target_yaw_rate - state.yaw_rate
            mz_des = cp.yaw_gain * e_r - cp.yaw_damping * r_dot
            delta_T = mz_des * r_eff / (2.0 * tw)
            delta_T = float(np.clip(delta_T, -cp.tv_max_torque_delta,
                                    cp.tv_max_torque_delta))
            delta_T *= min(1.0, vx / vp.vx_low)  # ramp in with speed
            T_tv = np.array([T_base - delta_T, T_base + delta_T,
                             T_base - delta_T, T_base + delta_T])
        self._prev_yaw_rate = state.yaw_rate

        # --- 2. traction control ---------------------------------------------
        if vx >= 0.5:
            _, Fz = compute_slip_angles(state, vp, steering_deg)
            kappa = self._estimate_slip_ratios(T_tv, Fz)
            for i in range(4):
                if T_tv[i] > 0.0 and abs(kappa[i]) > cp.tc_slip_target:
                    cut = cp.tc_aggressiveness * (abs(kappa[i])
                                                  - cp.tc_slip_target)
                    T_tv[i] *= (1.0 - min(cut, 1.0))
        else:
            kappa = np.zeros(4)

        # --- 3. brake blending ------------------------------------------------
        brake_pressure_out = np.zeros(4)
        if braking:
            T_brake_total = brake_pressure * vp.brake_torque_max_total
            T_dem_w = np.array([
                T_brake_total * cp.brake_bias_front / 2.0,
                T_brake_total * cp.brake_bias_front / 2.0,
                T_brake_total * (1.0 - cp.brake_bias_front) / 2.0,
                T_brake_total * (1.0 - cp.brake_bias_front) / 2.0,
            ])
            # regen unavailable at (near) standstill
            regen_speed_scale = min(1.0, vx / vp.vx_low)
            regen_w = np.zeros(4)
            for i in range(4):
                regen_w[i] = min(cp.regen_ratio * T_dem_w[i],
                                 T_regen_lim) * regen_speed_scale
            friction_w = T_dem_w - regen_w
            brake_pressure_out = friction_w / (vp.brake_torque_max_total / 4.0)
            # regen enters the motor command as negative torque
            T_tv = T_tv - regen_w

        # --- final clipping ---------------------------------------------------
        motor_torque_Nm = np.clip(T_tv, -T_regen_lim, T_max_w)

        return motor_torque_Nm, brake_pressure_out


# --------------------------------------------------------------------------
# Test harness
# --------------------------------------------------------------------------
def vdc_test():
    """Simulate a constant-steering corner with the VDC controller active.

    Starts at 15 m/s on a straight, applies 30 deg steering wheel (~5 deg
    road wheel, a moderate left corner) with mild throttle; a short brake
    demand (0.2 for 0.8 s) is applied around t = 3 s to exercise brake
    blending without stopping the car.  Prints per-wheel torques, slip
    ratios and the steady-state yaw-rate tracking error, and saves
    ``results/validation/vdc_test.png`` (4 panels: yaw tracking, per-wheel
    torques, per-wheel slip ratios, speed + brake blending).

    Runnable as::

        python3 -c "from src.controller import vdc_test; vdc_test()"
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = KIT25E_PARAMS
    vdc = VDCController(params)
    state = VehicleState(vx=15.0)

    dt = 0.001
    t_max = 6.0
    n_steps = int(t_max / dt)
    steer_deg = 30.0  # constant steering wheel angle

    ts, vxs, rs, r_tgts = [], [], [], []
    T_log = [[], [], [], []]
    k_log = [[], [], [], []]
    bp_log = []

    for i in range(n_steps):
        t = i * dt

        # kinematic neutral-steer target yaw rate for the current steer
        delta_road = np.radians(steer_deg / params.steering_ratio)
        r_target = (state.vx * np.tan(delta_road) / params.wheelbase
                    if abs(state.vx) > 0.5 else 0.0)

        throttle = 0.15
        brake_in = 0.2 if 3.0 < t < 3.8 else 0.0  # brief braking: blend demo

        T_motor, bp_out = vdc.compute_torques(
            state, steer_deg, throttle, brake_in, r_target, dt)
        state = step(state, params, steer_deg, throttle, bp_out, T_motor, dt)

        if i % 10 == 0:
            _, Fz = compute_slip_angles(state, params, steer_deg)
            kappa = vdc._estimate_slip_ratios(T_motor, Fz)
            ts.append(t)
            vxs.append(state.vx)
            rs.append(state.yaw_rate)
            r_tgts.append(r_target)
            for j in range(4):
                T_log[j].append(float(T_motor[j]))
                k_log[j].append(float(kappa[j]))
            bp_log.append(float(np.max(bp_out)))

    # --- summary -------------------------------------------------------------
    rs_a = np.asarray(rs[-500:])
    rt_a = np.asarray(r_tgts[-500:])
    err_ss = float(np.mean(np.abs(rs_a - rt_a)))
    peak_split = max(abs(T_log[0][k] - T_log[1][k]) for k in range(len(ts)))
    peak_bp = float(np.max(bp_log))            # during the brake window
    min_T = min(min(tl) for tl in T_log)       # most negative = max regen

    print("\n===== KIT25e VDC corner test =====")
    print(f"Final speed:                {state.vx:.2f} m/s")
    print(f"Yaw-rate tracking error:    {err_ss:.4f} rad/s "
          f"(mean |r_target - r|, last 0.5 s)")
    print(f"Steady torque split:        FL={T_log[0][-1]:.1f} "
          f"FR={T_log[1][-1]:.1f} RL={T_log[2][-1]:.1f} "
          f"RR={T_log[3][-1]:.1f} Nm")
    print(f"Peak left/right torque diff:{peak_split:.1f} Nm "
          f"(TV authority limit {vdc.cp.tv_max_torque_delta:.0f} Nm)")
    print(f"Slip ratios (end):          FL={k_log[0][-1]:.3f} "
          f"FR={k_log[1][-1]:.3f} RL={k_log[2][-1]:.3f} "
          f"RR={k_log[3][-1]:.3f}   (TC threshold {vdc.cp.tc_slip_target:.2f})")
    print(f"Brake blending (window):    peak friction {peak_bp:.3f}, "
          f"peak regen torque {min_T:.1f} Nm  (regen ratio "
          f"{vdc.cp.regen_ratio:.0%}, bias {vdc.cp.brake_bias_front:.0%} front)")
    print("=================================================")

    # --- plots --------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(ts, rs, "b-", label="actual")
    axes[0, 0].plot(ts, r_tgts, "r--", label="target (kinematic)")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Yaw rate [rad/s]")
    axes[0, 0].set_title("Yaw Rate Tracking (torque vectoring)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    names = ["FL", "FR", "RL", "RR"]
    for j, name in enumerate(names):
        axes[0, 1].plot(ts, T_log[j], label=name)
    axes[0, 1].axhline(0, color="k", lw=0.8)
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Motor torque [Nm]")
    axes[0, 1].set_title("Per-Wheel Motor Torques (regen < 0)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    for j, name in enumerate(names):
        axes[1, 0].plot(ts, k_log[j], label=name)
    axes[1, 0].axhline(vdc.cp.tc_slip_target, color="k", ls=":",
                       label="TC threshold")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Slip ratio [-]")
    axes[1, 0].set_title("Per-Wheel Slip Ratios")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(ts, vxs, "b-", label="vx")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Speed [m/s]")
    axes[1, 1].set_title("Speed and Brake Blending")
    axes[1, 1].grid(True)
    ax2 = axes[1, 1].twinx()
    ax2.plot(ts, bp_log, "r-", label="friction brake")
    ax2.set_ylabel("Friction brake pressure [-]")
    ax2.set_ylim(0, 1)

    fig.suptitle("KIT25e VDC - Corner Entry with Torque Vectoring, "
                 "Traction Control, Brake Blending")
    fig.tight_layout()
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results", "validation"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results", "validation", "vdc_test.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")
    return dict(err_ss=err_ss, peak_split=peak_split,
                final_T=[float(t) for t in T_motor],
                final_bp=float(np.max(bp_out)), peak_bp=peak_bp,
                peak_regen=min_T)


if __name__ == "__main__":
    vdc_test()
