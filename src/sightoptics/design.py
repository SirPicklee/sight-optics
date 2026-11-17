"""Simple lens design helpers: singlets and cemented achromatic doublets."""

from __future__ import annotations

import math

from .glass import WL_D, Glass
from .raytrace import SequentialSystem, SphericalSurface


def achromat_powers(f: float, crown: Glass, flint: Glass) -> tuple[float, float]:
    """Thin-lens achromat condition φ1/V1 + φ2/V2 = 0 with φ1 + φ2 = 1/f.

    Returns the focal lengths (f_crown, f_flint) [mm].
    """
    v1, v2 = crown.vd, flint.vd
    phi = 1.0 / f
    phi1 = phi * v1 / (v1 - v2)
    phi2 = -phi * v2 / (v1 - v2)
    return 1.0 / phi1, 1.0 / phi2


def cemented_achromat(f: float, semi: float, crown: Glass, flint: Glass, c1: float,
                      t_crown: float = 6.0, t_flint: float = 3.0) -> SequentialSystem:
    """Cemented doublet (crown first) with front curvature c1 = 1/R1 as the bending parameter.

    R2 (cemented surface) keeps the crown power, R3 keeps the flint power (thin-lens values).
    """
    f1, f2 = achromat_powers(f, crown, flint)
    n1, n2 = crown.nd, flint.nd
    c2 = c1 - 1.0 / (f1 * (n1 - 1.0))
    c3 = c2 - 1.0 / (f2 * (n2 - 1.0))
    r = [math.inf if abs(c) < 1e-15 else 1.0 / c for c in (c1, c2, c3)]
    return SequentialSystem([
        SphericalSurface(r[0], t_crown, crown, semi),
        SphericalSurface(r[1], t_flint, flint, semi),
        SphericalSurface(r[2], 0.0, 1.0, semi),
    ])


def planoconvex(f: float, semi: float, glass: Glass, t: float = 6.0) -> SequentialSystem:
    """Plano-convex singlet, curved side towards the object (thin-lens R = (n-1) f)."""
    R = (glass.nd - 1.0) * f
    return SequentialSystem([SphericalSurface(R, t, glass, semi), SphericalSurface(math.inf, 0.0, 1.0, semi)])


def scale_to_efl(system: SequentialSystem, f_target: float, wavelength: float = WL_D) -> SequentialSystem:
    """Scale radii and thicknesses so that the EFL equals f_target exactly (apertures unchanged)."""
    k = f_target / system.to_paraxial(wavelength).efl
    return SequentialSystem([
        SphericalSurface(s.R * k, s.thickness * k, s.n, s.semi_diameter) for s in system.surfaces
    ], system.n_object)


def correct_color(system: SequentialSystem, wl_short: float, wl_long: float) -> SequentialSystem:
    """Adjust the curvature of the last surface so that the paraxial foci of two
    wavelengths coincide (thick-lens refinement of the thin-lens achromat)."""
    from scipy.optimize import brentq

    surfs = list(system.surfaces)
    c0 = 0.0 if math.isinf(surfs[-1].R) else 1.0 / surfs[-1].R

    def build(c):
        last = surfs[-1]
        R = math.inf if abs(c) < 1e-15 else 1.0 / c
        return SequentialSystem(surfs[:-1] + [SphericalSurface(R, last.thickness, last.n, last.semi_diameter)],
                                system.n_object)

    def focal_diff(c):
        s = build(c)
        return s.to_paraxial(wl_short).bfl - s.to_paraxial(wl_long).bfl

    c = brentq(focal_diff, c0 - 0.005, c0 + 0.005, xtol=1e-14)
    return build(c)


def achromat_design(f: float, semi: float, crown: Glass, flint: Glass, c1: float,
                    t_crown: float = 6.0, t_flint: float = 3.0) -> SequentialSystem:
    """Full recipe: thin-lens achromat -> exact F/C colour correction -> scale to EFL = f."""
    from .glass import WL_C, WL_F

    s = cemented_achromat(f, semi, crown, flint, c1, t_crown, t_flint)
    s = correct_color(s, WL_F, WL_C)
    return scale_to_efl(s, f)



def bending_solutions(f: float, semi: float, crown: Glass, flint: Glass,
                      c1_min: float = 0.0, c1_max: float = 0.016, n_scan: int = 33) -> list[float]:
    """Front curvatures c1 that minimise the monochromatic (d-line) RMS spot of the
    colour-corrected cemented doublet: scan, then refine every local minimum."""
    import numpy as np
    from scipy.optimize import minimize_scalar

    from .spot import best_focus

    def mono_rms(c1):
        return best_focus(achromat_design(f, semi, crown, flint, c1), semi, wavelengths=[WL_D], n=15)[1]

    grid = np.linspace(c1_min, c1_max, n_scan)
    vals = np.array([mono_rms(c) for c in grid])
    sols = []
    for i in range(1, n_scan - 1):
        if vals[i] < vals[i - 1] and vals[i] < vals[i + 1]:
            res = minimize_scalar(mono_rms, bounds=(grid[i - 1], grid[i + 1]), method="bounded",
                                  options={"xatol": 1e-8})
            sols.append(float(res.x))
    return sols