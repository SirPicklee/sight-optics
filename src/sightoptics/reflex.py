"""Reflex (red-dot) sight optics.

The shooter looks through a window towards +z. An LED sits on the eye side, below the line of
sight. Its light is reflected back towards the eye by a curved, partially reflecting (dichroic)
surface. If the LED is at the focus, the reflected rays leave parallel and the dot appears at
infinity, superimposed on the target. Where the reflected rays are NOT parallel, the dot moves
on the target when the head moves: parallax.

The window is an off-axis section of a rotationally symmetric system whose axis passes through
the LED, so the LED stays out of the line of sight. Lengths in mm, angles in rad.

Supported windows (vertex of the first surface at z = 0, light from the LED travels +z):
  * a bare spherical mirror (front-surface)
  * a cemented "Mangin doublet": glass shell A | mirrored interface B | C. The LED light
    refracts at A, reflects at B, refracts at A again. Scene light goes straight through A, B, C.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .glass import Glass, refractive_index

MOA = math.radians(1 / 60)
WL_LED = 0.650  # red LED [µm]


def _intersect(pos, dirs, R, zv):
    """Nearest forward intersection with a sphere of signed radius R (centre at zv + R) or a plane."""
    if math.isinf(R):
        t = (zv - pos[:, 2]) / dirs[:, 2]
    else:
        c = np.array([0.0, 0.0, zv + R])
        q = pos - c
        b = np.einsum("ij,ij->i", dirs, q)
        disc = b**2 - (np.einsum("ij,ij->i", q, q) - R**2)
        with np.errstate(invalid="ignore"):
            root = np.sqrt(disc)
        t1, t2 = -b - root, -b + root
        t = np.where(t1 > 1e-9, t1, t2)
    return pos + t[:, None] * dirs


def _normal(points, R, zv):
    if math.isinf(R):
        return np.tile([0.0, 0.0, 1.0], (len(points), 1))
    n = points - np.array([0.0, 0.0, zv + R])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def _refract(dirs, normals, n1, n2):
    cos_i = np.einsum("ij,ij->i", dirs, normals)
    normals = np.where(cos_i[:, None] < 0, -normals, normals)
    cos_i = np.abs(cos_i)
    mu = n1 / n2
    with np.errstate(invalid="ignore"):
        cos_t = np.sqrt(1 - mu**2 * (1 - cos_i**2))
    return mu * dirs + (cos_t - mu * cos_i)[:, None] * normals


def _reflect(dirs, normals):
    return dirs - 2 * np.einsum("ij,ij->i", dirs, normals)[:, None] * normals


@dataclass(frozen=True)
class ReflexWindow:
    """R_A: eye-side surface, R_B: mirrored surface, R_C: target-side surface.
    For a bare mirror use glass=None (only R_B is used, at z = 0)."""

    R_B: float
    glass: Glass | float | None = None
    R_A: float = math.inf
    R_C: float = math.inf
    t1: float = 3.0  # A -> B
    t2: float = 3.0  # B -> C

    def n(self, wl):
        return 1.0 if self.glass is None else refractive_index(self.glass, wl)

    # ---- dot path: LED -> (A) -> B mirror -> (A) -> eye ----
    def reflect_rays(self, pos, dirs, wl=WL_LED):
        if self.glass is None:
            p = _intersect(pos, dirs, self.R_B, 0.0)
            return p, _reflect(dirs, _normal(p, self.R_B, 0.0))
        n = self.n(wl)
        p = _intersect(pos, dirs, self.R_A, 0.0)
        d = _refract(dirs, _normal(p, self.R_A, 0.0), 1.0, n)
        p = _intersect(p, d, self.R_B, self.t1)
        d = _reflect(d, _normal(p, self.R_B, self.t1))
        p = _intersect(p, d, self.R_A, 0.0)
        d = _refract(d, _normal(p, self.R_A, 0.0), n, 1.0)
        return p, d

    # ---- scene path: target -> C -> B -> A -> eye (traced eye -> target, same thing) ----
    def transmit_rays(self, pos, dirs, wl=0.5876):
        if self.glass is None:
            return pos, dirs
        n = self.n(wl)
        p = _intersect(pos, dirs, self.R_A, 0.0)
        d = _refract(dirs, _normal(p, self.R_A, 0.0), 1.0, n)
        p = _intersect(p, d, self.R_C, self.t1 + self.t2)
        d = _refract(d, _normal(p, self.R_C, self.t1 + self.t2), n, 1.0)
        return p, d

    def focus_distance(self, h: float = 0.01) -> float:
        """Paraxial LED distance: send a near-axis parallel ray in from the eye side and see where
        the reflected ray crosses the axis (reciprocity: LED there -> parallel output)."""
        pos = np.array([[0.0, h, -50.0]])
        p, d = self.reflect_rays(pos, np.array([[0.0, 0.0, 1.0]]))
        z = p[0, 2] - p[0, 1] * d[0, 2] / d[0, 1]
        return -z

    @property
    def transmission_power(self) -> float:
        """Thick-lens power [1/mm] of the window for the scene (A and C, glass in between)."""
        if self.glass is None:
            return 0.0
        n = self.n(0.5876)
        pa = 0.0 if math.isinf(self.R_A) else (n - 1) / self.R_A
        pc = 0.0 if math.isinf(self.R_C) else (1 - n) / self.R_C
        return pa + pc - (self.t1 + self.t2) / n * pa * pc


def zero_power_rc(R_A: float, glass, t_total: float) -> float:
    """R_C that makes the window afocal for the scene: φ_A + φ_C - (t/n) φ_A φ_C = 0."""
    n = refractive_index(glass, 0.5876)
    pa = (n - 1) / R_A
    pc = -pa / (1 - t_total / n * pa)
    return (1 - n) / pc


def window_points(center_y: float, radius: float, n: int = 15) -> np.ndarray:
    """Grid of (x, y) points filling the circular viewing window."""
    g = np.linspace(-radius, radius, n)
    x, y = np.meshgrid(g, g)
    keep = x**2 + y**2 <= radius**2
    return np.column_stack((x[keep], y[keep] + center_y))


def dot_directions(window: ReflexWindow, led_z: float, pts: np.ndarray, led_xy=(0.0, 0.0)):
    """Angles (ax, ay) [rad] at which the eye sees the dot through each window point."""
    led = np.array([led_xy[0], led_xy[1], led_z])
    # aim each LED ray at the window point on the plane z = 0 (close enough to the surface)
    tgt = np.column_stack((pts, np.zeros(len(pts))))
    d = tgt - led
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    _, out = window.reflect_rays(np.tile(led, (len(pts), 1)), d)
    view = -out  # the eye looks opposite to the light direction
    return np.column_stack((view[:, 0] / view[:, 2], view[:, 1] / view[:, 2]))


def parallax(window: ReflexWindow, led_z: float, pts: np.ndarray, led_xy=(0.0, 0.0)) -> tuple[float, float]:
    """(RMS, max) dot movement [rad] over the window, after zeroing on the window centre average."""
    a = dot_directions(window, led_z, pts, led_xy)
    dev = np.linalg.norm(a - a.mean(axis=0), axis=1)
    return float(np.sqrt(np.mean(dev**2))), float(dev.max())


def best_led_z(window: ReflexWindow, pts: np.ndarray, span: float = 5.0, metric=None) -> float:
    """LED axial position that minimises RMS parallax (the factory 'focus' adjustment).
    metric: parallax (dot only, default) or aiming_error (dot relative to the target)."""
    metric = metric or parallax
    z0 = -window.focus_distance()
    res = minimize_scalar(lambda z: metric(window, z, pts)[0], bounds=(z0 - span, z0 + span),
                          method="bounded", options={"xatol": 1e-6})
    return float(res.x)


def mangin_for_focus(f: float, R_A: float, glass, t1: float = 3.0, t2: float = 3.0) -> ReflexWindow:
    """Cemented Mangin window with LED focus distance f: R_B solved, R_C set for zero scene power."""
    R_C = zero_power_rc(R_A, glass, t1 + t2)

    def err(R_B):
        return ReflexWindow(R_B, glass, R_A, R_C, t1, t2).focus_distance() - f

    # walk R_B from a strong to a weak mirror until the focus passes f, then refine
    lo = -0.5 * f
    for hi in -0.5 * f * 1.05 ** np.arange(1, 120):
        e = err(hi)
        if e > 0:
            break
        lo = hi
    else:
        raise ValueError("no mirror radius gives the requested focus")
    R_B = brentq(err, lo, hi, xtol=1e-10)
    return ReflexWindow(R_B, glass, R_A, R_C, t1, t2)

def scene_directions(window: ReflexWindow, pts: np.ndarray) -> np.ndarray:
    """Angles (ax, ay) [rad] at which a target at infinity straight ahead is seen through each window point.

    A line of sight along +z leaves the window deviated by dev; to see the target the eye must look
    the other way, so the apparent target direction is -dev (first order)."""
    pos = np.column_stack((pts, np.full(len(pts), -30.0)))
    _, d = window.transmit_rays(pos, np.tile([0.0, 0.0, 1.0], (len(pts), 1)))
    return -np.column_stack((d[:, 0] / d[:, 2], d[:, 1] / d[:, 2]))


def aiming_error(window: ReflexWindow, led_z: float, pts: np.ndarray, led_xy=(0.0, 0.0)) -> tuple[float, float]:
    """(RMS, max) movement [rad] of the dot RELATIVE TO THE TARGET over the window, after zeroing.
    This is the parallax the shooter actually experiences: dot errors and window prism together."""
    e = dot_directions(window, led_z, pts, led_xy) - scene_directions(window, pts)
    dev = np.linalg.norm(e - e.mean(axis=0), axis=1)
    return float(np.sqrt(np.mean(dev**2))), float(dev.max())


def optimise_mangin(f: float, glass, pts: np.ndarray, R_A_bounds=(-150.0, -50.0), t1: float = 3.0, t2: float = 3.0):
    """Bend the eye-side surface R_A to minimise the RMS aiming error (dot relative to target),
    re-focusing the LED for every candidate."""

    def cost(R_A):
        w = mangin_for_focus(f, R_A, glass, t1, t2)
        return aiming_error(w, best_led_z(w, pts, metric=aiming_error), pts)[0]

    res = minimize_scalar(cost, bounds=R_A_bounds, method="bounded", options={"xatol": 1e-4})
    return mangin_for_focus(f, res.x, glass, t1, t2)