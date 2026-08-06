"""Pacejka Magic Formula 5.0 (MF5.0) tire model.

Implements pure lateral (Fy), pure longitudinal (Fx) and friction-ellipse
combined-slip force generation for the Hoosier 16.0x7.5-10 R20 tire used on
the KA-RaceIng KIT25e Formula Student car.

Coefficients are parsed at runtime from the TNO MF5.0 property file
``models/hoosier_r20.tir`` (``PROPERTY_FILE_FORMAT =
'MF_05'``, ``TYRESIDE = 'LEFT'``, ``FNOMIN = 1000.0 N``, ``LFZO = 0.8``).

Conventions
-----------
* All *internal* math uses radians; the public API accepts angles in degrees.
* ``Fz`` (normal load) and forces are in Newton.
* Slip ratio ``kappa`` is dimensionless; positive = traction/driving,
  negative = braking (standard MF-Tyre sign convention).
* The .tir file is a LEFT-side measurement: its coefficient set yields a
  negative cornering stiffness (positive slip angle -> negative Fy in the
  TNO bench frame).  For use in a vehicle-dynamics framework this module
  normalizes the output sign so that *positive slip angle -> positive Fy*.
  Magnitudes and load-dependence are unchanged by this normalization.

Out of scope: aligning moment (Mz), overturning moment, rolling resistance.
Only numpy/scipy are required (plus matplotlib for ``validate()``).
"""

import os

import numpy as np

# --------------------------------------------------------------------------
# Module-level paths / state
# --------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIR_PATH = os.path.join(_PROJECT_ROOT, "models", "hoosier_r20.tir")

_coeffs = None  # lazy-loaded coefficient dict (see _load_coeffs)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
def parse_tir(filepath):
    """Parse a TNO MF5.0 ``.tir`` property file into a flat coefficient dict.

    Handles the standard TNO layout:

    * lines starting with ``$`` are comments and are skipped,
    * ``[SECTION_NAME]`` headers set the current section (used for metadata
      bookkeeping only; coefficients are stored flat),
    * ``NAME = VALUE`` pairs are parsed; trailing ``$ inline comments`` are
      stripped; values that do not parse as floats (e.g. ``'LEFT'``,
      ``'MF_05'``) are stored as strings.

    Args:
        filepath (str): path to the ``.tir`` file.

    Returns:
        dict: every coefficient and parameter by its exact upper-case name
        from the file (e.g. ``PCY1``, ``PDX1``, ``FNOMIN``, ``LFZO``,
        ``TYRESIDE``, ``LONGVL`` ...).
    """
    coeffs = {}
    section = None
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("$"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip()
                coeffs.setdefault("_sections", []).append(section)
                continue
            if "=" not in line:
                continue  # e.g. [SHAPE] table rows
            name, _, value = line.partition("=")
            name = name.strip()
            value = value.split("$", 1)[0].strip()  # strip inline comment
            if not name or not value:
                continue
            try:
                coeffs[name] = float(value)
            except ValueError:
                coeffs[name] = value  # keep strings (TYRESIDE, FILE_TYPE, ...)
    return coeffs


def _load_coeffs():
    """Lazily parse the .tir file once; returns the cached coefficient dict."""
    global _coeffs
    if _coeffs is None:
        _coeffs = parse_tir(TIR_PATH)
    return _coeffs


# --------------------------------------------------------------------------
# Internal pure-force kernels (radians)
# --------------------------------------------------------------------------
def _compute_pure_fy(alpha_rad, Fz, gamma_rad, c):
    """Pacejka MF5.0 pure lateral force, all angles in radians.

    Implements the exact MF-Tyre 5.x lateral formulation:

        Fy = Dy*sin(Cy*arctan(By*ay - Ey*(By*ay - arctan(By*ay)))) + SVy

    with load/camber-dependent factors per the TNO definition.  ``PKY1 < 0``
    in the LEFT-side file produces a negative cornering stiffness; the final
    output is normalized (flipped) so that positive slip angle yields
    positive Fy, matching the vehicle-dynamics convention.
    """
    # Nominal load scaled by LFZO (MF5.0): Fz0' = LFZO*FNOMIN
    Fz0 = c["FNOMIN"] * c.get("LFZO", 1.0)
    dfz = (Fz - Fz0) / Fz0

    # Camber scaling (MF5.0 uses gamma* = LGAY*gamma)
    gamma_star = gamma_rad * c.get("LGAY", 1.0)

    # Shape factor
    Cy = c["PCY1"] * c.get("LCY", 1.0)

    # Peak factor (friction)
    muy = (c["PDY1"] + c["PDY2"] * dfz) * (
        1.0 - c["PDY3"] * gamma_star ** 2
    ) * c.get("LMUY", 1.0)
    Dy = muy * Fz

    # Cornering stiffness (MF5.0 sin(2*arctan) form)
    Kya = (
        c["PKY1"]
        * Fz0
        * np.sin(2.0 * np.arctan(Fz / (c["PKY2"] * Fz0)))
        * (1.0 - c["PKY3"] * np.abs(gamma_star))
        * c.get("LKY", 1.0)
    )
    By = Kya / (Cy * Dy + 1e-12)

    # Curvature factor (camber- and sign-of-alpha dependent), Ey <= 1
    Ey = (c["PEY1"] + c["PEY2"] * dfz) * (
        1.0 - (c["PEY3"] + c["PEY4"] * gamma_star) * np.sign(alpha_rad)
    ) * c.get("LEY", 1.0)
    Ey = np.minimum(Ey, 1.0)

    # Horizontal shift (incl. camber-induced, scaled by LKYG)
    SHy = (c["PHY1"] + c["PHY2"] * dfz) * c.get("LHY", 1.0) + c[
        "PHY3"
    ] * gamma_star * c.get("LKYG", 1.0)
    alpha_y = alpha_rad + SHy

    # Vertical shift
    SVy = (
        Fz
        * (
            (c["PVY1"] + c["PVY2"] * dfz) * c.get("LVY", 1.0)
            + (c["PVY3"] + c["PVY4"] * dfz)
            * gamma_star
            * c.get("LKYG", 1.0)
        )
        * c.get("LMUY", 1.0)
    )

    # Magic formula
    arg = By * alpha_y
    Fy = Dy * np.sin(Cy * np.arctan(arg - Ey * (arg - np.arctan(arg)))) + SVy

    # Sign normalization: the LEFT-side TNO file has PKY1 < 0 (positive slip
    # angle -> negative Fy in the bench frame). Flip to the vehicle
    # convention (positive slip angle -> positive Fy).
    if c["PKY1"] < 0:
        Fy = -Fy
    return Fy


def _compute_pure_fx(kappa, Fz, c):
    """Pacejka MF5.0 pure longitudinal force.

    Implements the exact MF-Tyre 5.x longitudinal formulation:

        Fx = Dx*sin(Cx*arctan(Bx*kx - Ex*(Bx*kx - arctan(Bx*kx)))) + SVx

    ``kappa`` positive = traction, negative = braking.
    """
    Fz0 = c["FNOMIN"] * c.get("LFZO", 1.0)
    dfz = (Fz - Fz0) / Fz0

    # Shape factor
    Cx = c["PCX1"] * c.get("LCX", 1.0)

    # Peak factor (friction); PDX3 is the camber sensitivity of Mux, which is
    # out of scope here (get_fx has no camber input), so it is not applied.
    mux = (c["PDX1"] + c["PDX2"] * dfz) * c.get("LMUX", 1.0)
    Dx = mux * Fz

    # Slip stiffness
    Kxk = (
        Fz
        * (c["PKX1"] + c["PKX2"] * dfz)
        * np.exp(c["PKX3"] * dfz)
        * c.get("LKX", 1.0)
    )
    Bx = Kxk / (Cx * Dx + 1e-12)

    # Curvature factor (sign-of-kappa dependent), Ex <= 1
    Ex = (c["PEX1"] + c["PEX2"] * dfz + c["PEX3"] * dfz ** 2) * (
        1.0 - c["PEX4"] * np.sign(kappa)
    ) * c.get("LEX", 1.0)
    Ex = np.minimum(Ex, 1.0)

    # Horizontal / vertical shifts
    SHx = (c["PHX1"] + c["PHX2"] * dfz) * c.get("LHX", 1.0)
    kappa_x = kappa + SHx
    SVx = (
        Fz
        * (c["PVX1"] + c["PVX2"] * dfz)
        * c.get("LVX", 1.0)
        * c.get("LMUX", 1.0)
    )

    # Magic formula
    arg = Bx * kappa_x
    Fx = Dx * np.sin(Cx * np.arctan(arg - Ex * (arg - np.arctan(arg)))) + SVx
    return Fx


# --------------------------------------------------------------------------
# Public API (degrees in, Newton out)
# --------------------------------------------------------------------------
def get_fy(slip_angle_deg, normal_load_N, camber_angle_deg=0.0):
    """Pure lateral force Fy for a given slip angle and normal load.

    Args:
        slip_angle_deg (float|ndarray): slip angle in degrees (positive =
            positive Fy, vehicle convention).
        normal_load_N (float|ndarray): vertical load Fz in Newton.
        camber_angle_deg (float, optional): camber angle in degrees. Defaults
            to 0.

    Returns:
        float|ndarray: lateral force Fy in Newton.
    """
    c = _load_coeffs()
    alpha_rad = np.radians(slip_angle_deg)
    gamma_rad = np.radians(camber_angle_deg)
    return _compute_pure_fy(alpha_rad, normal_load_N, gamma_rad, c)


def get_fx(slip_ratio, normal_load_N):
    """Pure longitudinal force Fx for a given slip ratio and normal load.

    Args:
        slip_ratio (float|ndarray): longitudinal slip kappa, dimensionless,
            typically in [-1, 1] (positive = traction, negative = braking).
        normal_load_N (float|ndarray): vertical load Fz in Newton.

    Returns:
        float|ndarray: longitudinal force Fx in Newton.
    """
    c = _load_coeffs()
    return _compute_pure_fx(slip_ratio, normal_load_N, c)


def get_combined(slip_angle_deg, slip_ratio, normal_load_N, camber_angle_deg=0.0):
    """Combined-slip forces using friction-ellipse weighting.

    Computes the pure forces Fx0, Fy0 and scales them so the pair stays
    inside the friction ellipse whose semi-axes are the pure peaks
    (Dx = mux*Fz, Dy = muy*Fz):

        rho = sqrt((Fx0/Dx)^2 + (Fy0/Dy)^2)
        scale = min(1, 1/rho)

    When only one slip input is active the ellipse reduces to the pure curve
    (scale == 1); under combined slip both components are reduced
    proportionally, preserving their directions.

    Args:
        slip_angle_deg (float|ndarray): slip angle in degrees.
        slip_ratio (float|ndarray): longitudinal slip kappa (dimensionless).
        normal_load_N (float|ndarray): vertical load Fz in Newton.
        camber_angle_deg (float, optional): camber angle in degrees.

    Returns:
        tuple: (Fx, Fy) combined forces in Newton.
    """
    c = _load_coeffs()
    Fz0 = c["FNOMIN"] * c.get("LFZO", 1.0)
    dfz = (normal_load_N - Fz0) / Fz0
    gamma_rad = np.radians(camber_angle_deg)

    # Pure forces
    Fx0 = _compute_pure_fx(slip_ratio, normal_load_N, c)
    Fy0 = _compute_pure_fy(
        np.radians(slip_angle_deg), normal_load_N, gamma_rad, c
    )

    # Friction-ellipse semi-axes (pure peaks)
    mux = (c["PDX1"] + c["PDX2"] * dfz) * c.get("LMUX", 1.0)
    muy = (c["PDY1"] + c["PDY2"] * dfz) * (
        1.0 - c["PDY3"] * gamma_rad ** 2
    ) * c.get("LMUY", 1.0)
    Dx = np.maximum(mux * normal_load_N, 1e-9)
    Dy = np.maximum(muy * normal_load_N, 1e-9)

    # Ellipse utilization
    rho = np.sqrt((Fx0 / Dx) ** 2 + (Fy0 / Dy) ** 2)
    scale = np.minimum(1.0, 1.0 / (rho + 1e-12))

    return scale * Fx0, scale * Fy0


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate():
    """Generate validation plots and print key tire metrics.

    Produces ``results/validation/tire_curves.png`` with:

    * left panel: pure lateral force Fy vs slip angle for loads
      200 / 400 / 600 / 800 / 1000 N (camber = 0),
    * right panel: pure longitudinal force Fx vs slip ratio for the same
      loads.

    Prints the peak lateral friction coefficient and peak slip angle at 500 N,
    and the cornering stiffness at 500 N (numerical central-difference slope
    at zero slip angle).  Raises if any NaN/Inf appears in the outputs.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless login node
    import matplotlib.pyplot as plt

    c = _load_coeffs()

    loads = [200.0, 400.0, 600.0, 800.0, 1000.0]
    alpha_range = np.linspace(-20.0, 20.0, 401)
    kappa_range = np.linspace(-1.0, 1.0, 401)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Fy vs slip angle at multiple loads -------------------------------
    for Fz in loads:
        Fy = get_fy(alpha_range, Fz, 0.0)
        ax1.plot(alpha_range, Fy, label=f"{Fz:.0f} N")
    ax1.set_xlabel("Slip angle [deg]")
    ax1.set_ylabel("Lateral force Fy [N]")
    ax1.set_title("Hoosier R20 - Pure Lateral Force (MF5.0)")
    ax1.legend(title="Fz")
    ax1.grid(True)

    # --- Fx vs slip ratio at multiple loads --------------------------------
    for Fz in loads:
        Fx = get_fx(kappa_range, Fz)
        ax2.plot(kappa_range, Fx, label=f"{Fz:.0f} N")
    ax2.set_xlabel("Slip ratio kappa [-]")
    ax2.set_ylabel("Longitudinal force Fx [N]")
    ax2.set_title("Hoosier R20 - Pure Longitudinal Force (MF5.0)")
    ax2.legend(title="Fz")
    ax2.grid(True)

    fig.suptitle("Pacejka MF5.0 - Hoosier 16.0x7.5-10 R20 (KIT25e)")
    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "tire_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")

    # --- Key metrics at 500 N load -----------------------------------------
    Fz_ref = 500.0
    Fy_500 = get_fy(alpha_range, Fz_ref, 0.0)
    Fx_500 = get_fx(kappa_range, Fz_ref)

    if not np.all(np.isfinite(Fy_500)) or not np.all(np.isfinite(Fx_500)):
        raise RuntimeError("NaN/Inf detected in tire force outputs - check coefficients")

    peak_mu = np.max(np.abs(Fy_500)) / Fz_ref
    idx_peak = int(np.argmax(np.abs(Fy_500)))
    peak_alpha = abs(alpha_range[idx_peak])

    # Cornering stiffness: central-difference slope at alpha = 0
    h_deg = 0.5
    dFy = get_fy(h_deg, Fz_ref, 0.0) - get_fy(-h_deg, Fz_ref, 0.0)
    calpha_rad = dFy / (2.0 * np.radians(h_deg))  # N/rad
    calpha_deg = dFy / (2.0 * h_deg)  # N/deg

    print("\n===== Key metrics @ Fz = 500 N (camber = 0) =====")
    print(f"Peak lateral friction coefficient: {peak_mu:.3f} (peak Fy {np.max(np.abs(Fy_500)):.1f} N)")
    print(f"Peak slip angle: {peak_alpha:.2f} deg")
    print(f"Cornering stiffness: {calpha_rad:,.0f} N/rad  ({calpha_deg:.1f} N/deg)")
    print("=================================================")

    # Overall peak friction across the validation load sweep
    mus = [np.max(np.abs(get_fy(alpha_range, Fz, 0.0))) / Fz for Fz in loads]
    print("Peak lateral mu by load:", ", ".join(f"{Fz:.0f}N:{mu:.2f}" for Fz, mu in zip(loads, mus)))
    return out_path


if __name__ == "__main__":
    coeffs = parse_tir(TIR_PATH)
    n = len(coeffs)
    print(f"Parsed {n} entries from {TIR_PATH}")
    print("FNOMIN:", coeffs.get("FNOMIN"), "| LFZO:", coeffs.get("LFZO"),
          "| Fz0':", coeffs["FNOMIN"] * coeffs.get("LFZO", 1.0))
    print("TYRESIDE:", coeffs.get("TYRESIDE"), "| LONGVL:", coeffs.get("LONGVL"))
    validate()
