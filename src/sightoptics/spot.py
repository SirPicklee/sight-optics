"""Spot diagrams, RMS spot size, best focus and chromatic focal shift.

The aperture stop is assumed to be at the first surface (true for a telescope
objective). Field angles are in degrees, wavelengths in micrometres, lengths in mm.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from .glass import WL_C, WL_D, WL_F
from .raytrace import SequentialSystem

VISIBLE = (WL_F, WL_D, WL_C)


def pupil_grid(semi: float, n: int = 21) -> np.ndarray:
    """Square grid of (x, y) points inside a circle of radius semi; shape (N, 2)."""
    g = np.linspace(-semi, semi, n)
    x, y = np.meshgrid(g, g)
    keep = x**2 + y**2 <= semi**2 * (1 + 1e-12)
    return np.column_stack((x[keep], y[keep]))


def collimated_bundle(semi: float, field_deg: float = 0.0, n: int = 21, z0: float = -20.0):
    """Parallel rays filling the entrance pupil (at z = 0) for a field angle in the y-z plane."""
    a = math.radians(field_deg)
    xy = pupil_grid(semi, n)
    d = np.array([0.0, math.sin(a), math.cos(a)])
    pos = np.column_stack((xy[:, 0], xy[:, 1], np.zeros(len(xy)))) + (z0 / d[2]) * d
    return pos, np.tile(d, (len(xy), 1))


def trace_to_plane(system: SequentialSystem, pos, dirs, z_image: float, wavelength: float = WL_D) -> np.ndarray:
    """Trace rays and return their (x, y) hits on the plane z = z_image (dead rays removed)."""
    p, d = system.trace(pos, dirs, wavelength=wavelength)
    t = (z_image - p[:, 2]) / d[:, 2]
    hit = p + t[:, None] * d
    hit = hit[~np.isnan(hit).any(axis=1)]
    return hit[:, :2]


def rms_radius(xy: np.ndarray, centre: np.ndarray | None = None) -> float:
    """RMS distance of spot points from their centroid (or a given centre)."""
    c = xy.mean(axis=0) if centre is None else centre
    return float(np.sqrt(np.mean(np.sum((xy - c) ** 2, axis=1))))


def spot(system: SequentialSystem, semi: float, z_image: float, field_deg: float = 0.0,
         wavelengths: Sequence[float] = VISIBLE, n: int = 21) -> dict[float, np.ndarray]:
    """Spot points for each wavelength on the image plane."""
    pos, dirs = collimated_bundle(semi, field_deg, n)
    return {wl: trace_to_plane(system, pos, dirs, z_image, wl) for wl in wavelengths}


def polychromatic_rms(spots: dict[float, np.ndarray]) -> float:
    """RMS radius of all wavelengths together, about their common centroid."""
    return rms_radius(np.vstack(list(spots.values())))


def best_focus(system: SequentialSystem, semi: float, field_deg: float = 0.0,
               wavelengths: Sequence[float] = VISIBLE, n: int = 21, search_mm: float = 5.0) -> tuple[float, float]:
    """Image plane z that minimises the polychromatic RMS spot. Returns (z_best, rms)."""
    z_par = system.last_vertex + system.to_paraxial().bfl
    pos, dirs = collimated_bundle(semi, field_deg, n)

    def cost(z):
        return rms_radius(np.vstack([trace_to_plane(system, pos, dirs, z, wl) for wl in wavelengths]))

    res = minimize_scalar(cost, bounds=(z_par - search_mm, z_par + search_mm), method="bounded",
                          options={"xatol": 1e-6})
    return float(res.x), float(res.fun)


def chromatic_focal_shift(system: SequentialSystem, wavelengths) -> np.ndarray:
    """Paraxial back focal length at each wavelength [mm]."""
    return np.array([system.to_paraxial(wl).bfl for wl in wavelengths])