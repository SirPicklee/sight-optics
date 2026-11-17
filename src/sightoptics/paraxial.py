"""Paraxial (first-order) ray tracing with ray-transfer (ABCD) matrices.

Ray state: (y, u) = (height [mm], angle [rad]), measured in the local medium.
Light travels left to right. Radius R > 0 when the centre of curvature is to the right.
System matrix = M_n @ ... @ M_2 @ M_1 (first element hit is right-most).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence, Union

import numpy as np


@dataclass(frozen=True)
class Space:
    """Free propagation over distance d [mm]."""

    d: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1.0, self.d], [0.0, 1.0]])

    @property
    def length(self) -> float:
        return self.d


@dataclass(frozen=True)
class ThinLens:
    """Ideal thin lens of focal length f [mm]."""

    f: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1.0, 0.0], [-1.0 / self.f, 1.0]])

    @property
    def length(self) -> float:
        return 0.0


@dataclass(frozen=True)
class Surface:
    """Spherical refracting surface from index n1 to n2 with radius R [mm] (math.inf = flat)."""

    R: float
    n1: float
    n2: float

    @property
    def matrix(self) -> np.ndarray:
        power = 0.0 if math.isinf(self.R) else (self.n2 - self.n1) / self.R
        return np.array([[1.0, 0.0], [-power / self.n2, self.n1 / self.n2]])

    @property
    def length(self) -> float:
        return 0.0


Element = Union[Space, ThinLens, Surface]


@dataclass
class ParaxialSystem:
    """Ordered list of paraxial elements."""

    elements: Sequence[Element] = field(default_factory=list)

    @property
    def matrix(self) -> np.ndarray:
        m = np.eye(2)
        for el in self.elements:
            m = el.matrix @ m
        return m

    @property
    def abcd(self) -> tuple[float, float, float, float]:
        (a, b), (c, d) = self.matrix
        return float(a), float(b), float(c), float(d)

    @property
    def total_length(self) -> float:
        return sum(el.length for el in self.elements)

    def is_afocal(self, tol: float = 1e-12) -> bool:
        return abs(self.abcd[2]) < tol

    @property
    def efl(self) -> float:
        """Effective focal length -1/C (assumes same medium in and out)."""
        c = self.abcd[2]
        return math.inf if c == 0 else -1.0 / c

    @property
    def bfl(self) -> float:
        """Back focal length: last element -> rear focal point, -A/C."""
        a, _, c, _ = self.abcd
        return math.inf if c == 0 else -a / c

    @property
    def ffl(self) -> float:
        """Front focal length: front focal point -> first element, -D/C."""
        _, _, c, d = self.abcd
        return math.inf if c == 0 else -d / c

    @property
    def angular_magnification(self) -> float:
        """For an afocal system: u_out / u_in = D (negative = inverted)."""
        if not self.is_afocal():
            raise ValueError("system is not afocal")
        return self.abcd[3]

    def stop_image_distance(self) -> float:
        """Distance after the last element where the image of the first element
        (taken as the aperture stop) is formed: -B/D. For a telescope this is the eye relief."""
        _, b, _, d = self.abcd
        return -b / d

    def trace(self, y0: float, u0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trace a ray; return z positions, heights and angles after every element."""
        z, ray = 0.0, np.array([y0, u0], dtype=float)
        zs, ys, us = [z], [ray[0]], [ray[1]]
        for el in self.elements:
            ray = el.matrix @ ray
            z += el.length
            zs.append(z)
            ys.append(ray[0])
            us.append(ray[1])
        return np.array(zs), np.array(ys), np.array(us)


def kepler(f_objective: float, f_eyepiece: float) -> ParaxialSystem:
    """Thin-lens Kepler telescope: objective -> f_o + f_e -> eyepiece."""
    return ParaxialSystem([ThinLens(f_objective), Space(f_objective + f_eyepiece), ThinLens(f_eyepiece)])


def thick_lens(R1: float, R2: float, t: float, n: float, n_medium: float = 1.0) -> ParaxialSystem:
    """Singlet lens in a medium: surface -> glass thickness t -> surface."""
    return ParaxialSystem([Surface(R1, n_medium, n), Space(t), Surface(R2, n, n_medium)])