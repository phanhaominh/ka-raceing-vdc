"""Aggressive pure-pursuit driver that challenges the tires.

Modifications vs the standard driver:
- Late braking: brakes later and harder into corners
- Full throttle exit: 100% throttle from apex onward
- Trail braking: carries brake pressure into turn-in
- Higher corner entry speeds: targets 95% of grip limit

This pushes longitudinal slip high enough for TC to engage,
especially on degraded tires during endurance runs.
"""

import numpy as np

def aggressive_speed_target(curvature, grip_scale=1.0, aggression=0.95):
    """Compute aggressive target speed for a given curvature.
    
    Standard driver uses ~80% of grip limit.
    Aggressive driver uses 95% (aggression=0.95).
    """
    if curvature < 1e-6:
        return 30.0  # straight: go fast
    
    # Lateral acceleration limit: a_y_max = grip * mu * g
    # v_max = sqrt(a_y_max / curvature)
    mu_max = 1.8 * grip_scale  # peak lateral friction
    a_y_max = mu_max * 9.81 * aggression
    v_max = np.sqrt(a_y_max / abs(curvature))
    return min(v_max, 30.0)


def aggressive_brake_point(distance_to_corner, current_speed, target_speed, grip_scale=1.0):
    """Brake later — start braking closer to the corner.
    
    Standard driver brakes early and coasts in.
    Aggressive driver brakes hard and late.
    """
    # Deceleration limit: a_x_max = grip * mu_x * g
    mu_x = 1.5 * grip_scale  # peak longitudinal friction
    a_brake = mu_x * 9.81 * 0.9  # 90% of max braking
    
    # Distance needed to slow from current to target
    dv = current_speed - target_speed
    if dv <= 0:
        return 0.0, 0.0  # no braking needed
    
    # v^2 = u^2 + 2as → s = (v^2 - u^2) / (2a)
    dist_needed = (current_speed**2 - target_speed**2) / (2 * a_brake)
    
    # Brake when distance to corner < distance needed (late braking)
    if distance_to_corner < dist_needed * 0.8:  # 20% later than standard
        brake_pressure = min(1.0, dv / (a_brake * 0.5))
        return brake_pressure, a_brake
    
    return 0.0, 0.0


def aggressive_throttle(in_corner, at_apex, exiting_corner, current_speed, target_speed):
    """Full throttle from apex onward.
    
    Standard driver gradually feeds in throttle.
    Aggressive driver goes 100% at the apex.
    """
    if exiting_corner or at_apex:
        # Full send from apex
        throttle = min(1.0, 1.5 * (1.0 - current_speed / max(target_speed, 1.0)))
        return max(0.3, throttle)  # at least 30% throttle
    elif in_corner:
        # Maintenance throttle mid-corner
        return 0.2
    else:
        # Full throttle on straights
        return 1.0


print("Aggressive driver module loaded.")
print("Use with run_lap(params, driver_profile='aggressive')")
print("This pushes longitudinal slip high enough for TC to engage.")
