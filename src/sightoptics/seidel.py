"""Third-order (Seidel) aberration coefficients from a paraxial marginal and chief ray trace.

Notation follows Welford, "Aberrations of Optical Systems":
    marginal ray (y, u), chief ray (ybar, ubar), slopes u = dy/dz
    A = n (u + y c)      refraction invariant, marginal ray
    Abar = n (ubar + ybar c)
    H = n (ubar y - u ybar)   Lagrange invariant
    S_I   = -Σ A²    y Δ(u/n)        spherical aberration
    S_II  = -Σ A Abar y Δ(u/n)       coma
    S_III = -Σ Abar² y Δ(u/n)        astigmatism
    S_IV  = -Σ H² c Δ(1/n)           Petzval field curvature
    S_V   =  Σ (Abar/A)(S_III + S_IV) distortion (per surface)
    C_I   =  Σ A y Δ(δn/n)           longitudinal colour
    C_II  =  Σ Abar y Δ(δn/n)        lateral colour
The aperture stop is at the first surface; object at infinity.
Conversion to image-space blur measures was checked sign by sign against the
exact ray tracer (see measure_real and tests/test_seidel.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .glass import WL_C, WL_D, WL_F, refractive_index
from .raytrace import SequentialSystem

NAMES = ("S_I spherical", "S_II coma", "S_III astigmatism", "S_IV Petzval", "S_V distortion",
         "C_I axial colour", "C_II lateral colour")


@dataclass
class SeidelResult:
    surfaces: np.ndarray  # shape (n_surfaces, 7): S_I..S_V, C_I, C_II per surface
    n_image: float
    u_image: float  # marginal ray slope in image space
    image_height: float  # paraxial chief ray height on the paraxial image plane
    lagrange: float

    @property
    def totals(self) -> np.ndarray:
        return self.surfaces.sum(axis=0)

    def transverse(self) -> dict[str, float]:
        """Seidel sums converted to image-space blur measures [mm] (Welford ch. 8)."""
        S1, S2, S3, S4, S5, C1, C2 = self.totals
        n, u = self.n_image, self.u_image
        return {
            "LSA (marginal focus - paraxial focus)": -S1 / (2 * n * u**2),
            "tangential coma (transverse)": 3 * S2 / (2 * n * u),
            "astigmatism: T focus - S focus (longitudinal)": -S3 / (n * u**2),
            "Petzval surface sag (longitudinal)": -S4 / (2 * n * u**2),
            "sagittal field curvature (longitudinal)": -(S3 + S4) / (2 * n * u**2),
            "tangential field curvature (longitudinal)": -(3 * S3 + S4) / (2 * n * u**2),
            "distortion (transverse)": S5 / (2 * n * u),
            "distortion [%]": 100 * (S5 / (2 * n * u)) / self.image_height if self.image_height else 0.0,
            "axial colour F-C (longitudinal)": -C1 / (n * u**2),
            "lateral colour F-C (transverse)": C2 / (n * u),
        }


def seidel(system: SequentialSystem, semi: float, field_deg: float,
           wavelength: float = WL_D, wl_short: float = WL_F, wl_long: float = WL_C) -> SeidelResult:
    """Seidel coefficients for an object at infinity, stop at the first surface."""
    y, u = semi, 0.0  # marginal ray
    yb, ub = 0.0, math.tan(math.radians(field_deg))  # chief ray through stop centre
    n = refractive_index(system.n_object, wavelength)
    dn = refractive_index(system.n_object, wl_short) - refractive_index(system.n_object, wl_long)
    H = n * (ub * y - u * yb)
    rows = []
    for k, s in enumerate(system.surfaces):
        c = s.c
        n2 = refractive_index(s.n, wavelength)
        dn2 = refractive_index(s.n, wl_short) - refractive_index(s.n, wl_long)
        A = n * (u + y * c)
        Ab = n * (ub + yb * c)
        u2 = (n * u - y * c * (n2 - n)) / n2
        ub2 = (n * ub - yb * c * (n2 - n)) / n2
        d_un = u2 / n2 - u / n
        s1 = -A**2 * y * d_un
        s2 = -A * Ab * y * d_un
        s3 = -Ab**2 * y * d_un
        s4 = -H**2 * c * (1 / n2 - 1 / n)
        s5 = (Ab / A) * (s3 + s4) if A != 0 else 0.0
        d_dn = dn2 / n2 - dn / n
        c1 = A * y * d_dn
        c2 = Ab * y * d_dn
        rows.append((s1, s2, s3, s4, s5, c1, c2))
        n, dn, u, ub = n2, dn2, u2, ub2
        if k < len(system.surfaces) - 1:
            y += s.thickness * u
            yb += s.thickness * ub
    # paraxial image plane: where the marginal ray crosses the axis
    l_img = -y / u
    h_img = yb + l_img * ub
    return SeidelResult(np.array(rows), n, u, h_img, H)


# ---------------------------------------------------------------------------
# Exact ray-trace measurements of the same quantities (for validation)
# ---------------------------------------------------------------------------

def _ray(system, x0, y0, field_deg, wavelength, z0=-20.0):
    """Trace one parallel ray that crosses the stop plane (z = 0) at (x0, y0)."""
    a = math.radians(field_deg)
    d = np.array([[0.0, math.sin(a), math.cos(a)]])
    pos = np.array([[x0, y0, 0.0]]) + (z0 / d[0, 2]) * d
    p, k = system.trace(pos, d, wavelength=wavelength)
    return p[0], k[0]


def _at_z(p, k, z):
    return p + (z - p[2]) / k[2] * k


def _focus_z(system, field_deg, dx, dy, wavelength, z_ref):
    """z where a ray offset by (dx, dy) in the pupil re-crosses the chief ray (in that offset direction)."""
    p0, k0 = _ray(system, 0.0, 0.0, field_deg, wavelength)
    p1, k1 = _ray(system, dx, dy, field_deg, wavelength)
    comp = 1 if dy else 0
    g = lambda z: _at_z(p1, k1, z)[comp] - _at_z(p0, k0, z)[comp]
    z1 = z_ref + 1.0
    return z_ref - g(z_ref) * (z1 - z_ref) / (g(z1) - g(z_ref))


def measure_real(system: SequentialSystem, semi: float, field_deg: float, wavelength: float = WL_D,
                 wl_short: float = WL_F, wl_long: float = WL_C, eps: float = 1e-3) -> dict[str, float]:
    """Exact ray-trace values of the quantities returned by SeidelResult.transverse()."""
    z_img = system.last_vertex + system.to_paraxial(wavelength).bfl
    pm, km = _ray(system, 0.0, semi, 0.0, wavelength)
    lsa = (pm[2] - pm[1] * km[2] / km[1]) - z_img
    yc = _at_z(*_ray(system, 0.0, 0.0, field_deg, wavelength), z_img)[1]
    yu = _at_z(*_ray(system, 0.0, semi, field_deg, wavelength), z_img)[1]
    yl = _at_z(*_ray(system, 0.0, -semi, field_deg, wavelength), z_img)[1]
    y_par = system.to_paraxial(wavelength).efl * math.tan(math.radians(field_deg))
    zt = _focus_z(system, field_deg, 0.0, eps, wavelength, z_img) - z_img
    zs = _focus_z(system, field_deg, eps, 0.0, wavelength, z_img) - z_img
    y_short = _at_z(*_ray(system, 0.0, 0.0, field_deg, wl_short), z_img)[1]
    y_long = _at_z(*_ray(system, 0.0, 0.0, field_deg, wl_long), z_img)[1]
    return {
        "LSA (marginal focus - paraxial focus)": lsa,
        "tangential coma (transverse)": (yu + yl) / 2 - yc,
        "astigmatism: T focus - S focus (longitudinal)": zt - zs,
        "sagittal field curvature (longitudinal)": zs,
        "tangential field curvature (longitudinal)": zt,
        "distortion (transverse)": yc - y_par,
        "axial colour F-C (longitudinal)": system.to_paraxial(wl_short).bfl - system.to_paraxial(wl_long).bfl,
        "lateral colour F-C (transverse)": y_short - y_long,
    }