"""Real (exact) sequential ray tracing through spherical surfaces.

Rays are traced in 3D and vectorised with numpy: every array has shape (N, 3).
A ray that misses a surface, falls outside its clear aperture or undergoes
total internal reflection is marked dead by setting it to NaN.

Prescription convention (same as Zemax / Code V lens data editors):
    each surface has a radius R (math.inf = flat), the thickness to the NEXT
    surface, the material AFTER the surface (a number or a Glass) and a clear
    semi-diameter. Wavelengths are in micrometres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .glass import WL_D, Glass, refractive_index
from .paraxial import ParaxialSystem, Space
from .paraxial import Surface as ParaxialSurface


@dataclass(frozen=True)
class SphericalSurface:
    R: float
    thickness: float
    n: float | Glass
    semi_diameter: float = math.inf
    dy: float = 0.0  # decenter of the surface in y [mm] (manufacturing error; 0 = perfectly centred)

    @property
    def c(self) -> float:
        """Curvature 1/R (0 for a flat surface)."""
        return 0.0 if math.isinf(self.R) else 1.0 / self.R


def sag(c: float, r: np.ndarray) -> np.ndarray:
    """Sag z(r) of a sphere with curvature c at radial height r."""
    r = np.asarray(r, dtype=float)
    return c * r**2 / (1.0 + np.sqrt(1.0 - (c * r) ** 2))


def intersect(pos: np.ndarray, dirs: np.ndarray, c: float, z_vertex: float, dy: float = 0.0) -> np.ndarray:
    """Intersection points of rays with a sphere (vertex at z_vertex, curvature c).

    Solves c|p + tK|^2 - 2(p_z + t K_z) = 0 in vertex coordinates and takes the
    root nearest to the vertex with the numerically stable form t = F / (B + sqrt(B^2 - cF)).
    """
    p = pos - np.array([0.0, dy, z_vertex])
    pk = np.einsum("ij,ij->i", p, dirs)
    pp = np.einsum("ij,ij->i", p, p)
    b = dirs[:, 2] - c * pk
    f = c * pp - 2.0 * p[:, 2]
    disc = b**2 - c * f
    with np.errstate(invalid="ignore", divide="ignore"):
        t = f / (b + np.sqrt(disc))
    t = np.where(disc < 0, np.nan, t)
    return pos + t[:, None] * dirs


def surface_normal(points: np.ndarray, c: float, z_vertex: float, dy: float = 0.0) -> np.ndarray:
    """Unit normal of the sphere at the given points, oriented towards +z."""
    q = points - np.array([0.0, dy, z_vertex])
    nrm = np.column_stack((-c * q[:, 0], -c * q[:, 1], 1.0 - c * q[:, 2]))
    return nrm / np.linalg.norm(nrm, axis=1, keepdims=True)


def refract(dirs: np.ndarray, normals: np.ndarray, n1: float, n2: float) -> np.ndarray:
    """Vector form of Snell's law. Rays with total internal reflection become NaN."""
    mu = n1 / n2
    cos_i = np.einsum("ij,ij->i", dirs, normals)
    normals = np.where(cos_i[:, None] < 0, -normals, normals)
    cos_i = np.abs(cos_i)
    radicand = 1.0 - mu**2 * (1.0 - cos_i**2)
    with np.errstate(invalid="ignore"):
        cos_t = np.sqrt(radicand)
    out = mu * dirs + (cos_t - mu * cos_i)[:, None] * normals
    out[radicand < 0] = np.nan
    return out


@dataclass
class SequentialSystem:
    """Ordered list of spherical surfaces; the first vertex sits at z = 0."""

    surfaces: Sequence[SphericalSurface]
    n_object: float = 1.0

    @property
    def vertex_z(self) -> np.ndarray:
        th = [0.0] + [s.thickness for s in self.surfaces[:-1]]
        return np.cumsum(th)

    @property
    def last_vertex(self) -> float:
        return float(self.vertex_z[-1])

    def trace(self, pos: np.ndarray, dirs: np.ndarray, keep_path: bool = False, wavelength: float = WL_D):
        """Trace rays through all surfaces.

        Returns final positions (on the last surface), final directions and,
        if keep_path, a list of the intersection points at every surface.
        """
        pos = np.array(pos, dtype=float)
        dirs = np.array(dirs, dtype=float)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        n1 = self.n_object
        path = [pos.copy()]
        for s, zv in zip(self.surfaces, self.vertex_z):
            n2 = refractive_index(s.n, wavelength)
            pos = intersect(pos, dirs, s.c, zv, s.dy)
            r = np.hypot(pos[:, 0], pos[:, 1])
            pos[r > s.semi_diameter] = np.nan
            nrm = surface_normal(pos, s.c, zv, s.dy)
            dirs = refract(dirs, nrm, n1, n2)
            n1 = n2
            path.append(pos.copy())
        return (pos, dirs, path) if keep_path else (pos, dirs)

    def to_paraxial(self, wavelength: float = WL_D) -> ParaxialSystem:
        """Equivalent ABCD model of the same prescription (last thickness excluded)."""
        els, n1 = [], self.n_object
        for i, s in enumerate(self.surfaces):
            n2 = refractive_index(s.n, wavelength)
            els.append(ParaxialSurface(s.R, n1, n2))
            if i < len(self.surfaces) - 1:
                els.append(Space(s.thickness))
            n1 = n2
        return ParaxialSystem(els)


def meridional_fan(heights: np.ndarray, angle_rad: float = 0.0, z0: float = -20.0):
    """Parallel rays in the y-z plane starting at z0 with field angle angle_rad."""
    h = np.asarray(heights, dtype=float)
    pos = np.column_stack((np.zeros_like(h), h, np.full_like(h, z0)))
    d = np.array([0.0, math.sin(angle_rad), math.cos(angle_rad)])
    return pos, np.tile(d, (h.size, 1))


def axis_crossing(pos: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """z coordinate where each meridional ray crosses the optical axis (y = 0)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return pos[:, 2] - pos[:, 1] * dirs[:, 2] / dirs[:, 1]