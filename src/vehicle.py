"""3-DOF rigid-body vehicle dynamics model for the KA-RaceIng KIT25e.

Implements a planar (longitudinal / lateral / yaw) vehicle model coupled to
the Pacejka MF5.0 tire model in ``src.tire``.

Conventions
-----------
* Body frame: x forward, y to the LEFT, z up.
* ``yaw_rate`` > 0 = turning LEFT (CCW viewed from above).
* ``vy`` > 0 = lateral velocity to the LEFT.
* ``steering_deg`` is the STEERING WHEEL angle; road-wheel angle is
  ``steering_deg / steering_ratio``; positive = steer LEFT.
* Tire forces from ``src.tire`` are in the vehicle convention: positive slip
  angle -> positive Fy (leftward), positive kappa -> positive Fx (forward).
* Weight transfer uses the CG inertial accelerations (``state.ax/ay``) from
  the *previous* timestep (one-step lag, standard for explicit coupling).

Aero: downforce proportional to v^2, split front/rear per ``Cl_front`` /
``Cl_rear``, acting at the CoG (no pitch/roll sensitivity).  Drag opposes the
velocity vector, also at the CoG.

Out of scope: suspension kinematics, roll/pitch dynamics, driver model, VDC
controller, track model.  Integration is Forward Euler.
"""

from dataclasses import dataclass
from dataclasses import replace
import os

import numpy as np

from src.tire import get_combined, get_fx

GRAVITY = 9.81          # m/s^2
AIR_DENSITY = 1.225     # kg/m^3

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "validation")

# Longitudinal slip stiffness used only as a first-guess in the slip-ratio
# solver; the solver itself inverts the Pacejka curve so the tire force
# matches the wheel torque demand (see ``_solve_kappa``).
KAPPA_LIN = 0.01                # slip ratio below which the linear branch holds
KAPPA_SPIN = 0.25               # slip ratio used when a wheel is traction-limited


# --------------------------------------------------------------------------
# Parameters / state
# --------------------------------------------------------------------------
@dataclass
class VehicleParams:
    """Vehicle parameters for the KIT25e, SI units unless noted.

    All values are the KIT25e reference set (FSG 2025 registration data plus
    team estimates); fields marked "estimated" are initial guesses that the
    skidpad validation can nudge (see ``skidpad_test``).
    """

    # --- mass & inertia ---
    mass: float = 244.0                 # kg  (car 176 + driver 68)
    wheelbase: float = 1.530            # m
    track_width: float = 1.220          # m  (front = rear)
    cog_height: float = 0.280           # m  (estimated)
    weight_dist_front: float = 0.50     # -  (50/50 with driver)
    yaw_inertia: float = 60.0           # kg*m^2 (estimated)

    # --- tires / steering ---
    tire_rolling_radius: float = 0.228  # m  (from .tir, load-adjusted)
    steering_ratio: float = 6.0         # -  (steering wheel : road wheel)
    max_steering_deg: float = 120.0     # deg, steering wheel lock (estimated)

    # --- powertrain / brakes ---
    max_motor_torque_total: float = 600.0    # Nm at wheels, 4 motors
    max_motor_power_total: float = 116000.0  # W  (4 x 29 kW)
    brake_torque_max_total: float = 600.0    # Nm at wheels, all 4 corners

    # --- aero ---
    Cl_total: float = 6.2               # -  (positive = downforce)
    Cl_front: float = 2.5               # -  downforce share, front
    Cl_rear: float = 3.7                # -  downforce share, rear
    Cd: float = 1.5                     # -  (estimated)
    aero_ref_area: float = 1.2          # m^2 (estimated)

    # --- misc ---
    vx_low: float = 1.0                 # m/s  (VXLOW: lateral force scaled
                                        #       down below this speed)


KIT25E_PARAMS = VehicleParams()


@dataclass
class VehicleState:
    """3-DOF vehicle state (SI units, radians).

    ``ax`` / ``ay`` carry the CG inertial accelerations in the body frame
    from the *previous* step; they feed the weight-transfer calculation and
    are updated by :func:`step`.
    """

    vx: float = 0.0        # forward speed [m/s]
    vy: float = 0.0        # lateral speed [m/s] (+ = left)
    yaw_rate: float = 0.0  # yaw rate [rad/s] (+ = left turn)
    x: float = 0.0         # global position X [m]
    y: float = 0.0         # global position Y [m]
    heading: float = 0.0   # global heading [rad]
    ax: float = 0.0        # previous-step CG accel, body frame x [m/s^2]
    ay: float = 0.0        # previous-step CG accel, body frame y [m/s^2]


# --------------------------------------------------------------------------
# Kinematics and loads
# --------------------------------------------------------------------------
def compute_slip_angles(state, params, steering_deg):
    """Compute per-wheel slip angles and normal loads.

    Slip angle of wheel i: ``alpha_i = delta_i - atan2(vy_w, vx_w)`` where
    ``vy_w, vx_w`` are the velocity components at the wheel in the body frame
    and ``delta_i`` the road-wheel steer (front axle only).  Positive slip
    angle -> the wheel is turned further into the corner than the velocity,
    producing positive (leftward) Fy in the tire model.

    Normal loads include static load, longitudinal and lateral weight
    transfer from the previous step's CG accelerations, and aero downforce
    split front/rear by ``Cl_front``/``Cl_rear``.

    Args:
        state (VehicleState): current state.
        params (VehicleParams): vehicle parameters.
        steering_deg (float): steering wheel angle [deg] (+ = left).

    Returns:
        tuple: (slip_angles_deg [4], normal_loads_N [4]) for
        FL, FR, RL, RR.
    """
    a = params.wheelbase * params.weight_dist_front      # CG -> front axle
    b = params.wheelbase * (1.0 - params.weight_dist_front)  # CG -> rear axle
    tw = params.track_width

    # Wheel CG-relative positions: (x forward, y left)
    pos_x = np.array([a, a, -b, -b])
    pos_y = np.array([tw / 2.0, -tw / 2.0, tw / 2.0, -tw / 2.0])

    # Velocity at each wheel (rigid body)
    vx_w = state.vx - state.yaw_rate * pos_y
    vy_w = state.vy + state.yaw_rate * pos_x

    # Road-wheel steer for the front axle
    delta_rad = np.radians(steering_deg / params.steering_ratio)
    steer = np.array([delta_rad, delta_rad, 0.0, 0.0])

    alpha_rad = steer - np.arctan2(vy_w, np.maximum(np.abs(vx_w), 1e-3))
    slip_angles_deg = np.degrees(alpha_rad)

    # --- normal loads -----------------------------------------------------
    Fz_total = params.mass * GRAVITY
    Fz_f = Fz_total * params.weight_dist_front
    Fz_r = Fz_total * (1.0 - params.weight_dist_front)

    # Longitudinal transfer (previous-step ax): braking -> front gains
    dFz_long = params.mass * state.ax * params.cog_height / params.wheelbase
    Fz = np.array([
        Fz_f / 2.0 - dFz_long / 2.0,   # FL
        Fz_f / 2.0 - dFz_long / 2.0,   # FR
        Fz_r / 2.0 + dFz_long / 2.0,   # RL
        Fz_r / 2.0 + dFz_long / 2.0,   # RR
    ])

    # Aero downforce, split by Cl_front / Cl_rear (acts at CoG -> no moment)
    v_sq = state.vx ** 2 + state.vy ** 2
    F_down = 0.5 * AIR_DENSITY * params.Cl_total * params.aero_ref_area * v_sq
    Fz[0] += F_down * params.Cl_front / params.Cl_total / 2.0
    Fz[1] += F_down * params.Cl_front / params.Cl_total / 2.0
    Fz[2] += F_down * params.Cl_rear / params.Cl_total / 2.0
    Fz[3] += F_down * params.Cl_rear / params.Cl_total / 2.0

    # Lateral transfer per axle (previous-step ay), distributed by CURRENT
    # axle load share (static + aero): with rear-biased aero the rear axle
    # transfers proportionally more load when cornering.  ay > 0 = left turn
    # -> right-side wheels gain load (car rolls to the outside).
    Fz_axle_total = np.sum(Fz)
    dFz_lat_total = params.mass * state.ay * params.cog_height / tw
    dFz_lat_f = dFz_lat_total * (Fz[0] + Fz[1]) / Fz_axle_total
    dFz_lat_r = dFz_lat_total * (Fz[2] + Fz[3]) / Fz_axle_total
    Fz[0] -= dFz_lat_f / 2.0   # FL
    Fz[1] += dFz_lat_f / 2.0   # FR
    Fz[2] -= dFz_lat_r / 2.0   # RL
    Fz[3] += dFz_lat_r / 2.0   # RR

    # Minimum load so the tire model never sees zero/negative Fz
    Fz = np.maximum(Fz, 50.0)

    return slip_angles_deg, Fz


def compute_tire_forces(slip_angles_deg, slip_ratios, normal_loads_N):
    """Evaluate the MF5.0 tire model at all four wheels.

    Args:
        slip_angles_deg (array_like[4]): slip angles [deg] (FL, FR, RL, RR).
        slip_ratios (array_like[4]): longitudinal slip kappa (FL, FR, RL, RR).
        normal_loads_N (array_like[4]): vertical loads [N] (FL, FR, RL, RR).

    Returns:
        tuple: (Fx[4], Fy[4]) tire forces in N, vehicle convention.
    """
    Fx = np.zeros(4)
    Fy = np.zeros(4)
    for i in range(4):
        Fx[i], Fy[i] = get_combined(
            slip_angles_deg[i], slip_ratios[i], normal_loads_N[i], 0.0
        )
    return Fx, Fy


def _solve_kappa(T_net, Fz, r_eff):
    """Solve the slip ratio whose tire force matches the wheel torque.

    Finds kappa on the rising branch of the pure longitudinal Pacejka curve
    (0 .. kappa_peak) such that ``|Fx(kappa, Fz)| * r_eff == |T_net|``.  If
    the torque demand exceeds the tire's peak force the wheel is
    traction-limited (spinning): returns ``KAPPA_SPIN`` (a high slip ratio so
    slip-based traction control can act); the tire then delivers less than
    the demanded force, as with a real spinning wheel.

    Uses a linear-stiffness fast path for small slips (cruising / skidpad)
    and a peak scan + bisection for launch conditions.

    Args:
        T_net (float): net wheel torque [Nm] (+ drive, - brake).
        Fz (float): normal load [N].
        r_eff (float): tire rolling radius [m].

    Returns:
        float: slip ratio kappa, signed like ``T_net``.
    """
    sign = 1.0 if T_net >= 0.0 else -1.0
    demand = abs(T_net) / max(r_eff, 1e-6)
    if demand < 1e-6 or Fz < 50.0:
        return 0.0

    # Fast path: numerical linear stiffness at zero slip
    Fx_lo = abs(get_fx(0.0, Fz))
    Fx_hi = abs(get_fx(KAPPA_LIN, Fz))
    K_lin = (Fx_hi - Fx_lo) / KAPPA_LIN
    kappa0 = demand / max(K_lin, 1e-6)
    if kappa0 <= KAPPA_LIN:
        return sign * kappa0

    # Peak scan on [0, KAPPA_SPIN]
    ks = np.linspace(0.0, KAPPA_SPIN, 100)
    fxs = np.array([abs(get_fx(k, Fz)) for k in ks])
    i_peak = int(np.argmax(fxs))
    kappa_peak = float(ks[i_peak])
    fx_peak = float(fxs[i_peak])

    if demand >= fx_peak:
        return sign * KAPPA_SPIN  # traction-limited: wheel spins

    # Bisection on the rising branch [0, kappa_peak]
    lo, hi = 0.0, kappa_peak
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if abs(get_fx(sign * mid, Fz)) < demand:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-5:
            break
    return sign * 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Vehicle-level forces
# --------------------------------------------------------------------------
def compute_vehicle_forces(state, params, steering_deg, throttle,
                           brake_pressure, motor_torque_Nm=None):
    """Compute total body-frame forces and yaw moment on the vehicle.

    Includes tire forces (via :func:`compute_tire_forces`), aero drag, and
    low-speed lateral force scaling below ``params.vx_low`` (VXLOW behaviour
    from the tire property file).  Aero downforce is already inside the
    normal loads.

    Args:
        state (VehicleState): current state.
        params (VehicleParams): vehicle parameters.
        steering_deg (float): steering wheel angle [deg].
        throttle (float): 0..1 pedal position.
        brake_pressure (float): 0..1 brake input.
        motor_torque_Nm (array_like[4] | None): per-wheel drive torque [Nm];
            if None, derived from ``throttle`` with power limiting.

    Returns:
        tuple: (total_Fx, total_Fy, total_yaw_moment) in N / N / Nm,
        body frame (x forward, y left, yaw + = left turn).
    """
    slip_angles_deg, Fz = compute_slip_angles(state, params, steering_deg)

    # --- per-wheel torque -> slip ratio ------------------------------------
    if motor_torque_Nm is None:
        T_max_w = params.max_motor_torque_total / 4.0
        P_max_w = params.max_motor_power_total / 4.0
        omega = max(abs(state.vx), 0.5) / params.tire_rolling_radius
        T_drive_w = min(throttle * T_max_w, P_max_w / omega)
        motor_torque_Nm = np.full(4, T_drive_w)
    T_drive = np.asarray(motor_torque_Nm, dtype=float)

    T_brake_w = brake_pressure * params.brake_torque_max_total / 4.0
    T_net = T_drive - T_brake_w

    # Slip ratios consistent with the torque demand.  The former linear
    # stiffness estimate (kappa = T/(r*K)) was physically inconsistent: the
    # tire then delivered Fx != T/r and traction-limited wheelspin could not
    # be represented (the launch torque never reached the friction limit).
    # Now kappa is solved per wheel on the Pacejka curve so that
    # Fx(kappa, Fz) * r == T_net up to the friction limit; beyond it the
    # wheel is pinned at KAPPA_SPIN (spinning) and delivers less force.
    kappa = np.zeros(4)
    for j in range(4):
        kappa[j] = _solve_kappa(T_net[j], Fz[j], params.tire_rolling_radius)

    # --- tire forces --------------------------------------------------------
    Fx_tire, Fy_tire = compute_tire_forces(slip_angles_deg, kappa, Fz)

    # Low-speed scaling of lateral force (VXLOW): no spurious Fy at standstill
    v = float(np.hypot(state.vx, state.vy))
    f_lat = min(1.0, v / max(params.vx_low, 1e-3))
    Fy_tire = Fy_tire * f_lat

    # --- aero drag (at CoG, opposes velocity) -------------------------------
    F_drag = 0.5 * AIR_DENSITY * params.Cd * params.aero_ref_area * v * v
    F_aero_x = -F_drag * (state.vx / max(v, 1e-3))
    F_aero_y = -F_drag * (state.vy / max(v, 1e-3))

    total_Fx = float(np.sum(Fx_tire) + F_aero_x)
    total_Fy = float(np.sum(Fy_tire) + F_aero_y)

    # --- yaw moment from tire forces (aero acts at CoG -> no moment) --------
    a = params.wheelbase * params.weight_dist_front
    b = params.wheelbase * (1.0 - params.weight_dist_front)
    tw = params.track_width
    pos_x = np.array([a, a, -b, -b])
    pos_y = np.array([tw / 2.0, -tw / 2.0, tw / 2.0, -tw / 2.0])
    total_yaw_moment = float(np.sum(pos_x * Fy_tire - pos_y * Fx_tire))

    return total_Fx, total_Fy, total_yaw_moment


def step(state, params, steering_deg, throttle, brake_pressure,
         motor_torque_Nm=None, dt=0.001):
    """Advance the vehicle state by ``dt`` using Forward Euler.

    Equations of motion (body frame, y left, z up):

        vx_dot = Fx/m + r*vy
        vy_dot = Fy/m - r*vx
        r_dot  = Mz/Iz

    ``state.ax`` / ``state.ay`` are set to the CG inertial accelerations
    (Fx/m, Fy/m) for use by the weight-transfer model on the next step.

    Args:
        state (VehicleState): current state.
        params (VehicleParams): vehicle parameters.
        steering_deg (float): steering wheel angle [deg].
        throttle (float): 0..1.
        brake_pressure (float): 0..1.
        motor_torque_Nm (array_like[4] | None): per-wheel drive torque; None
            -> derived from throttle.
        dt (float): integration step [s], default 1 ms.

    Returns:
        VehicleState: new state.
    """
    Fx, Fy, Mz = compute_vehicle_forces(
        state, params, steering_deg, throttle, brake_pressure, motor_torque_Nm
    )

    m = params.mass
    Iz = params.yaw_inertia
    r = state.yaw_rate

    ax_inertial = Fx / m
    ay_inertial = Fy / m

    vx_dot = ax_inertial + r * state.vy
    vy_dot = ay_inertial - r * state.vx
    r_dot = Mz / Iz

    cos_h = np.cos(state.heading)
    sin_h = np.sin(state.heading)

    new_state = VehicleState(
        vx=state.vx + vx_dot * dt,
        vy=state.vy + vy_dot * dt,
        yaw_rate=r + r_dot * dt,
        x=state.x + (state.vx * cos_h - state.vy * sin_h) * dt,
        y=state.y + (state.vx * sin_h + state.vy * cos_h) * dt,
        heading=state.heading + r * dt,
        ax=ax_inertial,
        ay=ay_inertial,
    )
    return new_state


# --------------------------------------------------------------------------
# Skidpad validation
# --------------------------------------------------------------------------
def skidpad_test():
    """FSG-style constant-radius skidpad simulation.

    Drives the KIT25e model on a circle of radius ``R = 15.25 m`` (FSG rules)
    with a proportional yaw-rate controller (equivalent to maintaining the
    radius, since ``R = v/r``) plus a PI speed controller targeting the lap
    time ``t = 5.14 s`` (``v = 2*pi*R/t``).  Runs until the yaw rate
    stabilizes, then prints the steady-state lateral acceleration, lap time,
    and understeer gradient, and saves
    ``results/validation/skidpad_trace.png``.

    If the car oversteers (understeer gradient <= 0) the CoG height is
    lowered / front aero increased; if the lap time is off by more than
    0.5 s the relevant knob is adjusted and the sim re-run (max 6 attempts).

    Returns:
        dict: summary of the final run.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R_target = 15.25      # m, FSG skidpad radius
    target_lap = 5.14     # s, KIT25e FSG 2025 result
    v_target = 2.0 * np.pi * R_target / target_lap  # ~18.64 m/s

    dt = 0.001
    t_max = 40.0
    n_steps = int(t_max / dt)

    # Controller gains (tuned for this vehicle)
    Kp_v, Ki_v = 0.08, 0.10   # speed PI: throttle
    Kp_r, Ki_r = 6.0, 4.0     # yaw-rate PI + Ackermann feedforward

    def run_once(params):
        """Run one skidpad attempt; return (summary, logs)."""
        state = VehicleState()
        int_v = 0.0
        int_r = 0.0
        t_converged = t_max

        # Path-geometry integration for the true (position-derived) radius
        dist = 0.0
        dheading = 0.0
        px, py = state.x, state.y
        ph = state.heading

        ts, vxs, vys, rs, steers, thrs, rest, rpath = [], [], [], [], [], [], [], []

        for i in range(n_steps):
            t = i * dt
            v = float(np.hypot(state.vx, state.vy))

            # --- speed controller (PI) ---
            e_v = v_target - v
            int_v = float(np.clip(int_v + e_v * dt, -5.0, 5.0))
            throttle = float(np.clip(Kp_v * e_v + Ki_v * int_v, 0.0, 1.0))

            # --- steering controller: feedforward + PI on yaw-rate error ---
            # Feedforward supplies the kinematic Ackermann steer (L/R); the
            # integral supplies the understeer-gradient extra at speed; P
            # handles the transient.  r_des = v/R <=> holding radius R.
            r_des = v / R_target
            e_r = r_des - state.yaw_rate
            int_r = float(np.clip(int_r + Ki_r * e_r * dt, -80.0, 80.0))
            steer_ff = (params.steering_ratio * params.wheelbase / R_target
                        * 180.0 / np.pi * min(1.0, v / 5.0))
            steering = float(np.clip(steer_ff + Kp_r * e_r + int_r,
                                     -params.max_steering_deg,
                                     params.max_steering_deg))

            state = step(state, params, steering, throttle, 0.0, None, dt)

            # path geometry
            dist += float(np.hypot(state.x - px, state.y - py))
            px, py = state.x, state.y
            dheading += state.heading - ph
            ph = state.heading

            if i % 20 == 0:  # log every 20 ms
                R_est = v / max(abs(state.yaw_rate), 1e-3)
                R_path = dist / max(abs(dheading), 1e-6)
                ts.append(t)
                vxs.append(state.vx)
                vys.append(state.vy)
                rs.append(state.yaw_rate)
                steers.append(steering)
                thrs.append(throttle)
                rest.append(R_est)
                rpath.append(R_path)

            # convergence: yaw rate stable over a 4 s window after 20 s
            if t > 20.0 and i % 2000 == 0 and len(rs) >= 200:
                if max(rs[-200:]) - min(rs[-200:]) < 0.005:
                    t_converged = t
                    break

        # steady-state metrics from the last 3 s of the log
        seg = slice(-150, None) if len(rs) >= 150 else slice(None)
        v_ss = float(np.mean(vxs[seg]))
        r_ss = float(np.mean(rs[seg]))
        steer_ss = float(np.mean(steers[seg]))
        R_est_ss = v_ss / max(abs(r_ss), 1e-6)
        R_path_ss = float(np.mean(rpath[seg]))
        a_lat = v_ss * abs(r_ss)
        a_lat_g = a_lat / GRAVITY
        lap_time = 2.0 * np.pi * R_path_ss / max(v_ss, 1e-3)   # actual path
        lap_time_est = 2.0 * np.pi * R_est_ss / max(v_ss, 1e-3)

        # Understeer gradient: K_us = (delta_road - L/R) / a_y  [deg/g]
        delta_road_rad = np.radians(steer_ss / params.steering_ratio)
        k_us = (delta_road_rad - params.wheelbase / R_est_ss) \
            * 180.0 / np.pi / max(a_lat_g, 1e-3)

        summary = dict(
            t_converged=t_converged, v_ss=v_ss, r_ss=r_ss,
            R_est=R_est_ss, R_path=R_path_ss, a_lat=a_lat, a_lat_g=a_lat_g,
            lap_time=lap_time, lap_time_est=lap_time_est, K_us=k_us,
            steer_ss=steer_ss,
        )
        logs = dict(t=ts, vx=vxs, vy=vys, r=rs, steer=steers,
                    throttle=thrs, R_est=rest, R_path=rpath)
        return summary, logs

    # --- tuning loop ---------------------------------------------------------
    params = replace(KIT25E_PARAMS)
    knobs = dict()
    summary, logs = None, None
    for attempt in range(6):
        p = replace(params, **knobs)
        summary, logs = run_once(p)
        print(f"[attempt {attempt + 1}] "
              f"lap(path)={summary['lap_time']:.3f}s "
              f"R_est={summary['R_est']:.2f}m R_path={summary['R_path']:.2f}m "
              f"K_us={summary['K_us']:.2f}deg/g a_lat={summary['a_lat_g']:.2f}g")

        # Genuine oversteer (K_us <= 0): spec remedies - lower CoG height,
        # then raise front aero.  Positive K_us (understeer) needs no
        # adjustment.
        if summary["K_us"] <= 0.0:
            if knobs.get("cog_height", p.cog_height) > 0.20:
                knobs["cog_height"] = knobs.get("cog_height", p.cog_height) - 0.01
                print("  -> oversteer: lowering cog_height to "
                      f"{knobs['cog_height']:.3f} m")
                continue
            cf = knobs.get("Cl_front", p.Cl_front)
            if cf < 3.9:
                knobs["Cl_front"] = cf + 0.1
                print("  -> oversteer: raising Cl_front to "
                      f"{knobs['Cl_front']:.2f}")
                continue
            cr = knobs.get("Cl_rear", p.Cl_rear)
            if cr < 4.5:
                knobs["Cl_rear"] = cr + 0.1
                print("  -> oversteer: raising Cl_rear (more rear grip) to "
                      f"{knobs['Cl_rear']:.2f}")
                continue

        # can't hold the circle / steering saturated -> more front grip
        if summary["R_path"] > R_target * 1.10:
            knobs["Cl_front"] = knobs.get("Cl_front", p.Cl_front) + 0.1
            print("  -> running wide: raising Cl_front to "
                  f"{knobs['Cl_front']:.2f}")
            continue

        # lap time too slow -> cut drag (car speed-limited by drag/aero)
        if abs(summary["lap_time"] - target_lap) > 0.5:
            knobs["Cd"] = knobs.get("Cd", p.Cd) - 0.1
            print("  -> lap time off: lowering Cd to "
                  f"{knobs['Cd']:.2f}")
            continue

        break  # all criteria met

    # --- print final metrics -------------------------------------------------
    p = replace(params, **knobs)
    print("\n===== KIT25e skidpad result (R = 15.25 m, target 5.14 s) =====")
    print(f"Steady-state speed:        {summary['v_ss']:6.2f} m/s")
    print(f"Steady-state yaw rate:     {summary['r_ss']:6.3f} rad/s")
    print(f"Radius (v/r):              {summary['R_est']:6.2f} m  "
          f"(position-derived: {summary['R_path']:.2f} m)")
    print(f"Lateral acceleration:      {summary['a_lat']:6.2f} m/s^2  "
          f"({summary['a_lat_g']:.2f} g)")
    print(f"Lap time (path radius):    {summary['lap_time']:6.3f} s   "
          f"(target 5.14 s, delta {summary['lap_time'] - target_lap:+.3f} s)")
    print(f"Lap time (v/r radius):     {summary['lap_time_est']:6.3f} s")
    print(f"Understeer gradient:       {summary['K_us']:6.2f} deg/g"
          + ("   (UNDERSTEER - OK)" if summary["K_us"] > 0.0
             else "   (OVERSTEER)"))
    print(f"Steering wheel @ steady:   {summary['steer_ss']:6.1f} deg "
          f"(road wheel {summary['steer_ss'] / p.steering_ratio:.1f} deg)")
    print(f"Converged at t = {summary['t_converged']:.1f} s "
          f"(tuning: {knobs if knobs else 'none'})")
    print("================================================================")

    # --- trace plot -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    t_arr = np.asarray(logs["t"])
    axes[0].plot(t_arr, logs["vx"], "b-", label="vx")
    axes[0].plot(t_arr, logs["vy"], "b--", label="vy")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Speed [m/s]")
    axes[0].set_title("Speed vs Time")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t_arr, logs["r"], "r-")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Yaw rate [rad/s]")
    axes[1].set_title("Yaw Rate vs Time")
    axes[1].grid(True)

    axes[2].plot(t_arr, logs["R_est"], "g-", label="R = v/r")
    axes[2].plot(t_arr, logs["R_path"], "g--", label="R from path")
    axes[2].axhline(R_target, color="k", ls=":", label=f"target {R_target} m")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Radius [m]")
    axes[2].set_title("Path Radius vs Time")
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle(f"KIT25e Skidpad (R={R_target:.2f} m) - "
                 f"lap {summary['lap_time']:.2f} s, "
                 f"{summary['a_lat_g']:.2f} g, "
                 f"K_us {summary['K_us']:.1f} deg/g")
    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "skidpad_trace.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")
    return summary


def acceleration_test(launch_torque_factor=None):
    """Simulate a 0-75 m straight-line launch and validate against the
    KIT25e FSG 2025 acceleration result (3.57 s, 115 km/h trap speed).

    Full throttle, no steering, no braking, straight line.  Per-wheel motor
    torque is ``min(launch_torque_factor * T_max_w, P_max_w / omega)`` so the
    powertrain is torque-limited at launch and power-limited at high speed.
    The slip-ratio solver inside :func:`compute_vehicle_forces` delivers the
    torque demand as tire force up to the friction limit; a wheel whose
    solved slip ratio exceeds 20% is treated as spinning and a simple
    traction control cuts its torque back to the 20%-slip level.

    ``launch_torque_factor`` scales the rated (power-derived) 600 Nm wheel
    torque to represent PMSM hub-motor peak-torque headroom at launch.  If
    None (default), it is auto-tuned from 1.0 so the 0-75 m time lands inside
    3.57 +/- 0.2 s; the factor-1.0 (rated torque) result is always reported
    for reference.

    Prints time to 75 m, speed at 75 m, peak longitudinal acceleration and
    wheelspin events; saves ``results/validation/accel_trace.png``.

    Args:
        launch_torque_factor (float | None): torque multiplier; None =
            auto-tune.

    Returns:
        dict: summary of the final run.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = KIT25E_PARAMS
    target_time = 3.57     # s, KIT25e FSG 2025 acceleration result
    target_dist = 75.0     # m, FSG acceleration distance
    dt = 0.001
    t_max = 8.0
    n_steps = int(t_max / dt)
    T_max_w = params.max_motor_torque_total / 4.0   # 150 Nm continuous/wheel
    P_max_w = params.max_motor_power_total / 4.0    # 29 kW/wheel
    r_eff = params.tire_rolling_radius

    def run_once(factor):
        """Run one 0-75 m launch; return (summary, logs)."""
        state = VehicleState()
        ts, dists, vxs, axs = [], [], [], []
        kappas = [[], [], [], []]
        fzs = [[], [], [], []]
        wheelspin_events = 0
        t = 0.0
        for i in range(n_steps):
            t = i * dt
            _, Fz = compute_slip_angles(state, params, 0.0)
            omega = max(state.vx, 0.5) / r_eff
            T_w = np.full(4, min(factor * T_max_w, P_max_w / omega))

            # Traction control: cut torque to any wheel slipping > 20%
            k_now = np.zeros(4)
            for j in range(4):
                k_j = _solve_kappa(T_w[j], Fz[j], r_eff)
                if abs(k_j) > 0.20:
                    wheelspin_events += 1
                    T_w[j] = abs(get_fx(0.20, Fz[j])) * r_eff
                    k_j = _solve_kappa(T_w[j], Fz[j], r_eff)
                k_now[j] = k_j

            state = step(state, params, 0.0, 1.0, 0.0,
                         motor_torque_Nm=T_w, dt=dt)

            if i % 10 == 0:
                ts.append(t)
                dists.append(state.x)
                vxs.append(state.vx)
                axs.append(state.ax)
                for j in range(4):
                    kappas[j].append(float(k_now[j]))
                    fzs[j].append(float(Fz[j]))

            if state.x >= target_dist:
                break

        peak_ax = float(np.max(axs))
        summary = dict(
            t_75=t, v_75=float(state.vx), peak_ax=peak_ax,
            wheelspin_events=wheelspin_events, t_end=t, dist=float(state.x),
            final_Fz=Fz, final_kappa=k_now,
        )
        logs = dict(ts=ts, dists=dists, vxs=vxs, axs=axs,
                    kappas=kappas, fzs=fzs)
        return summary, logs

    # --- torque factor: fixed, or auto-tuned to 3.57 +/- 0.2 s -------------
    factor = launch_torque_factor if launch_torque_factor is not None else 1.0
    raw_summary = None
    summary = logs = None
    for attempt in range(10):
        summary, logs = run_once(factor)
        if attempt == 0:
            raw_summary = summary
        print(f"[attempt {attempt + 1}] factor={factor:.2f} "
              f"t75={summary['t_75']:.3f}s "
              f"v75={summary['v_75'] * 3.6:.0f} km/h "
              f"peak_ax={summary['peak_ax'] / GRAVITY:.2f} g "
              f"wheelspin={summary['wheelspin_events']}")
        if launch_torque_factor is not None:
            break
        if summary["t_75"] > target_time + 0.2:
            factor = min(factor + 0.05, 1.6)
            continue
        if summary["t_75"] < target_time - 0.2:
            factor = max(factor - 0.05, 0.8)
            continue
        break

    # --- final metrics ------------------------------------------------------
    print("\n===== KIT25e Acceleration 0-75 m (target 3.57 s) =====")
    print(f"Launch torque factor:      {factor:.2f} x 600 Nm "
          f"({factor * params.max_motor_torque_total:.0f} Nm total)")
    print(f"Time to 75 m:              {summary['t_75']:.3f} s   "
          f"(target {target_time} s, delta "
          f"{summary['t_75'] - target_time:+.3f} s)")
    print(f"Speed at 75 m:             {summary['v_75']:.2f} m/s  "
          f"({summary['v_75'] * 3.6:.1f} km/h)   (real: 115 km/h)")
    print(f"Peak longitudinal accel:   {summary['peak_ax']:.2f} m/s^2  "
          f"({summary['peak_ax'] / GRAVITY:.2f} g)")
    print(f"Wheelspin events:          {summary['wheelspin_events']}  "
          f"(slip > 20%, traction control cut)")
    print(f"Final slip ratios:         FL={summary['final_kappa'][0]:.3f} "
          f"FR={summary['final_kappa'][1]:.3f} "
          f"RL={summary['final_kappa'][2]:.3f} "
          f"RR={summary['final_kappa'][3]:.3f}")
    print(f"Final front loads:         FL={summary['final_Fz'][0]:.0f} N  "
          f"FR={summary['final_Fz'][1]:.0f} N")
    print(f"Final rear loads:          RL={summary['final_Fz'][2]:.0f} N  "
          f"RR={summary['final_Fz'][3]:.0f} N")
    if raw_summary is not None and launch_torque_factor is None:
        print(f"(reference at rated 600 Nm, factor 1.0: "
              f"{raw_summary['t_75']:.3f} s - outside the band; 600 Nm is the "
              f"power-derived continuous rating, real hub motors launch with "
              f"higher peak torque)")
    print("=====================================================")

    # --- plots --------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(logs["dists"], logs["vxs"], "b-")
    axes[0, 0].set_xlabel("Distance [m]")
    axes[0, 0].set_ylabel("Speed [m/s]")
    axes[0, 0].set_title("Speed vs Distance")
    axes[0, 0].grid(True)

    axes[0, 1].plot(logs["ts"], np.asarray(logs["axs"]) / GRAVITY, "r-")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Acceleration [g]")
    axes[0, 1].set_title("Longitudinal Acceleration vs Time")
    axes[0, 1].grid(True)

    for j, name in enumerate(["FL", "FR", "RL", "RR"]):
        axes[1, 0].plot(logs["ts"], logs["kappas"][j], label=name)
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Slip ratio kappa [-]")
    axes[1, 0].set_title("Slip Ratio per Wheel")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    for j, name in enumerate(["FL", "FR", "RL", "RR"]):
        axes[1, 1].plot(logs["ts"], logs["fzs"][j], label=name)
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Normal load [N]")
    axes[1, 1].set_title("Normal Load per Wheel")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    fig.suptitle(f"KIT25e 0-75 m Acceleration - "
                 f"{summary['t_75']:.2f} s, {summary['v_75'] * 3.6:.0f} km/h, "
                 f"launch x{factor:.2f}")
    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "accel_trace.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")
    return summary


if __name__ == "__main__":
    skidpad_test()
