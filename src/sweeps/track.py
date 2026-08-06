"""Autocross-like test track and a pure-pursuit driver model.

The track is a waypoint polyline (start straight -> slalom -> hairpin ->
long straight -> sweeper -> second hairpin -> finish straight),
interpolated to a dense path with cumulative distance, **signed** curvature
(needed for the VDC yaw-rate target: ``r = v*kappa`` must flip sign for
left/right corners) and a curvature-limited target speed.  The driver is a
pure-pursuit path follower with a PI speed controller and a steering-rate
limit; it returns, per timestep, the steering wheel angle [deg], throttle,
brake, target speed and target yaw rate that feed the VDC + vehicle model.

The lap is ~540 m and takes ~30-40 s at FSAE autocross speeds, exercising
the VDC across straights, transitions and corners.  Track curvature is kept
moderate (kappa <= ~0.3 1/m) so the curvature-limited yaw rate stays below
the 3.0 rad/s crash threshold even at the limit.
"""

import numpy as np

GRAVITY = 9.81
MU_SPEED_LIMIT = 1.6    # lateral mu used for the curvature speed limit
TRACK_WIDTH = 3.0       # m, corridor half-width used for crash detection
STEER_RATE_LIMIT = 250.0  # deg/s max steering wheel rate (driver realism)

# (x, y, target_speed_mps) waypoints
TRACK_WAYPOINTS = np.array([
    [0.0, 0.0, 14.0],        # start straight, 60 m
    [60.0, 0.0, 22.0],
    [65.0, -3.0, 9.0],       # slalom: 4 gates, 10 m spacing, +-3 m
    [75.0, 3.0, 9.0],
    [85.0, -3.0, 9.0],
    [95.0, 3.0, 9.0],
    [100.0, -3.0, 8.0],
    [104.0, -9.0, 8.0],      # right hairpin (gentle kinks, kappa<=0.3)
    [110.0, -13.0, 14.0],
    [190.0, -13.0, 24.0],    # long straight, 80 m
    [192.0, -5.0, 13.0],     # left sweeper (R~12 m)
    [202.0, 5.0, 13.0],
    [214.0, 5.0, 16.0],
    [294.0, 5.0, 24.0],      # straight, 80 m
    [296.0, -1.0, 8.0],      # right hairpin (R~6 m)
    [302.0, -8.0, 14.0],
    [522.0, -8.0, 26.0],     # finish straight, 220 m
])


class Track:
    """Dense interpolated path with curvature and target-speed arrays."""

    def __init__(self, path_x, path_y, path_s, curv, v_target, total):
        self.path_x = path_x
        self.path_y = path_y
        self.path_s = path_s        # cumulative distance along the path [m]
        self.curv = curv            # SIGNED curvature [1/m] (smoothed)
        self.v_target = v_target    # target speed [m/s] (waypoint+curvature)
        self.total = total          # total lap length [m]
        self.width = TRACK_WIDTH


def _smooth(arr, window):
    """Moving average with edge padding (keeps array length)."""
    if window < 2:
        return arr.copy()
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    out = np.convolve(padded, kernel, mode="valid")
    return out[: len(arr)]


def build_track(waypoints=TRACK_WAYPOINTS, spacing=0.5):
    """Interpolate the waypoint polyline into a dense :class:`Track`.

    Args:
        waypoints (ndarray (N,3)): x, y, target speed per node.
        spacing (float): nominal spacing of path points [m].

    Returns:
        Track: interpolated path with signed curvature and speed profile.
    """
    wp = np.asarray(waypoints, dtype=float)
    pts = wp[:, :2]
    v_node = wp[:, 2]
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    s_node = np.concatenate([[0.0], np.cumsum(seg_len)])

    n = max(3, int(np.ceil(s_node[-1] / spacing)))
    path_s = np.linspace(0.0, s_node[-1], n)

    # linear interpolation of x, y along the polyline
    idx = np.clip(np.searchsorted(s_node, path_s, side="right") - 1,
                  0, len(seg_len) - 1)
    frac = np.zeros(n)
    ok = seg_len[idx] > 1e-9
    frac[ok] = (path_s[ok] - s_node[idx[ok]]) / seg_len[idx[ok]]
    path_x = pts[idx, 0] + frac * seg[idx, 0]
    path_y = pts[idx, 1] + frac * seg[idx, 1]

    # light path smoothing rounds the waypoint kinks into corners
    path_x = _smooth(path_x, 3)
    path_y = _smooth(path_y, 3)
    ds_seg = np.hypot(np.diff(path_x), np.diff(path_y))
    path_s = np.concatenate([[0.0], np.cumsum(ds_seg)])
    total = float(path_s[-1])

    # SIGNED curvature from wrapped heading change per unit distance
    head = np.arctan2(np.diff(path_y), np.diff(path_x))
    dhead = np.angle(np.exp(1j * np.diff(head)))
    ds = np.hypot(np.diff(path_x[1:]), np.diff(path_y[1:])) + 1e-9
    curv = np.concatenate([[0.0], dhead / ds, [0.0]])
    curv = _smooth(curv, 5)

    # speed profile: waypoint target capped by the curvature limit (|kappa|)
    v_cap = np.sqrt(MU_SPEED_LIMIT * GRAVITY / np.maximum(np.abs(curv), 0.02))
    v_target = np.minimum(np.interp(path_s, s_node, v_node), v_cap)
    v_target = np.clip(_smooth(v_target, 10), 4.0, 30.0)

    return Track(path_x, path_y, path_s, curv, v_target, total)


def closest_point(state, track):
    """Index and distance of the path point nearest to the vehicle."""
    d2 = ((track.path_x - state.x) ** 2 + (track.path_y - state.y) ** 2)
    i = int(np.argmin(d2))
    return i, float(np.sqrt(d2[i]))


class PurePursuitDriver:
    """Pure-pursuit path follower with a PI speed controller.

    Steering: find the path point ``lookahead`` meters ahead of the closest
    point, compute the road-wheel angle to intercept it (classic pure
    pursuit: ``delta = atan(2*L_wb*sin(alpha)/L_d)``), convert to steering
    wheel angle, and apply a steering-rate limit (realistic driver).  Speed:
    PI on the error vs the curvature-limited target.  Yaw-rate target for
    the VDC: ``r = v * kappa_signed`` at the closest point.
    """

    def __init__(self, track, lookahead=3.5, ratio=6.0,
                 max_steer_deg=120.0, wheelbase=1.53):
        self.track = track
        self.lookahead_base = lookahead
        self.ratio = ratio
        self.max_steer = max_steer_deg
        self.wheelbase = wheelbase
        self.int_v = 0.0
        self._steer_prev = 0.0
        self.Kp_v, self.Ki_v, self.Kb = 0.15, 0.05, 0.55

    def control(self, state, dt=0.002):
        """Compute driver commands for the current state.

        Returns:
            tuple: (steering_wheel_deg, throttle, brake, v_target,
            r_target, path_idx, track_deviation_m).
        """
        tr = self.track
        v = float(np.hypot(state.vx, state.vy))
        idx, dev = closest_point(state, tr)

        # speed-dependent lookahead
        L = float(np.clip(self.lookahead_base + 0.35 * v, 3.0, 10.0))
        s_la = min(tr.path_s[idx] + L, tr.total)
        i2 = int(np.clip(np.searchsorted(tr.path_s, s_la), 0,
                         len(tr.path_s) - 1))
        x_la, y_la = float(tr.path_x[i2]), float(tr.path_y[i2])

        dx, dy = x_la - state.x, y_la - state.y
        cos_h, sin_h = np.cos(state.heading), np.sin(state.heading)
        xb = dx * cos_h + dy * sin_h      # lookahead in vehicle frame
        yb = -dx * sin_h + dy * cos_h     # + = left

        if xb > 0.01:
            alpha = np.arctan2(yb, xb)
            delta_road = np.arctan(2.0 * self.wheelbase * np.sin(alpha)
                                   / max(L, 1e-3))
        else:  # lookahead beside/behind the car: steer hard toward it
            delta_road = np.sign(yb) * 0.4
        steer_des = float(np.clip(np.degrees(delta_road) * self.ratio,
                                  -self.max_steer, self.max_steer))
        # steering rate limit
        max_delta = STEER_RATE_LIMIT * dt
        steer = float(np.clip(steer_des, self._steer_prev - max_delta,
                              self._steer_prev + max_delta))
        self._steer_prev = steer

        # speed target: minimum over the upcoming horizon (brake for
        # corners ahead, like a real driver)
        spacing = tr.total / max(len(tr.path_x) - 1, 1)
        horizon = float(np.clip(2.0 * v, 12.0, 40.0))
        j_end = min(len(tr.v_target), idx + int(horizon / spacing) + 1)
        v_tgt = float(np.min(tr.v_target[idx:j_end]))
        e = v_tgt - v
        self.int_v = float(np.clip(self.int_v + e * dt, -3.0, 3.0))
        throttle = float(np.clip(self.Kp_v * e + self.Ki_v * self.int_v,
                                 0.0, 1.0))
        brake = float(np.clip(-self.Kb * e, 0.0, 1.0)) if e < 0.0 else 0.0

        r_target = float(np.clip(v * tr.curv[idx], -3.0, 3.0))
        return steer, throttle, brake, v_tgt, r_target, idx, dev
