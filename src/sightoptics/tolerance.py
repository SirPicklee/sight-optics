"""Tolerance analysis: sensitivity, RSS and Monte Carlo for a lens prescription.

Every manufactured lens differs from the drawing. Each tolerance is a +/- limit on one parameter:
    radius     relative error of a surface radius (e.g. 0.002 = 0.2 %)
    thickness  error of the thickness after a surface [mm]
    index      error of the glass index after a surface (melt-to-melt variation)
    decenter   lateral offset of a surface [mm] (wedge / centring error)
Performance is judged after the assembly compensator (refocusing the reticle), on axis:
    rms   wavefront error with piston, tilt and defocus removed [waves]  -> image sharpness
    boresight  line-of-sight shift [rad]  (removed later by zeroing, but must stay in turret range)
    focus      refocus needed [mm]        (must stay in the focus adjustment range)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .glass import WL_D, Glass, OffsetGlass
from .raytrace import SequentialSystem
from .wavefront import wavefront


@dataclass(frozen=True)
class Tolerance:
    kind: str  # "radius", "thickness", "index", "decenter"
    surface: int
    limit: float  # +/- limit

    @property
    def name(self) -> str:
        unit = {"radius": "%", "thickness": "mm", "index": "", "decenter": "mm"}[self.kind]
        lim = self.limit * 100 if self.kind == "radius" else self.limit
        return f"{self.kind} S{self.surface + 1} (±{lim:g}{unit})"


def perturb(system: SequentialSystem, tols: list[Tolerance], values: np.ndarray) -> SequentialSystem:
    """Copy of the system with every tolerance applied at the given value (same units as limit)."""
    surfs = list(system.surfaces)
    for tol, v in zip(tols, values):
        s = surfs[tol.surface]
        if tol.kind == "radius":
            surfs[tol.surface] = replace(s, R=s.R * (1 + v))
        elif tol.kind == "thickness":
            surfs[tol.surface] = replace(s, thickness=s.thickness + v)
        elif tol.kind == "index":
            base = s.n.base if isinstance(s.n, OffsetGlass) else s.n
            dn0 = s.n.dn if isinstance(s.n, OffsetGlass) else 0.0
            surfs[tol.surface] = replace(s, n=OffsetGlass(base, dn0 + v))
        elif tol.kind == "decenter":
            surfs[tol.surface] = replace(s, dy=s.dy + v)
        else:
            raise ValueError(tol.kind)
    return SequentialSystem(surfs, system.n_object)


def doublet_tolerances(system: SequentialSystem, radius: float, thickness: float, index: float,
                       decenter: float) -> list[Tolerance]:
    """Standard tolerance set for a lens: every radius and decenter, every glass thickness and index."""
    out = []
    for i, s in enumerate(system.surfaces):
        out.append(Tolerance("radius", i, radius))
        out.append(Tolerance("decenter", i, decenter))
        if isinstance(s.n, (Glass, OffsetGlass)):
            out.append(Tolerance("thickness", i, thickness))
            out.append(Tolerance("index", i, index))
    return out


def evaluate(system: SequentialSystem, semi: float, z_image: float, wavelength: float = WL_D,
             n: int = 48, refocus: bool = True) -> dict[str, float]:
    """On-axis performance: fit piston + tilt (+ defocus if refocus) to the wavefront, keep the residual.

    refocus=True models the assembly compensator (the reticle/focus is adjusted on every unit);
    refocus=False is a fixed, glued-in reticle position."""
    wf = wavefront(system, semi, 0.0, z_image, wavelength, n)
    m = wf.mask
    g = (np.arange(n) - n / 2 + 0.5) * wf.dp / semi
    x, y = np.meshgrid(g, g)
    x, y, W = x[m], y[m], wf.W[m]
    A = np.column_stack((np.ones_like(x), x, y, x**2 + y**2))
    coef, *_ = np.linalg.lstsq(A, W, rcond=None)
    if refocus:  # remove piston, tilt and defocus
        resid = W - A @ coef
    else:  # remove piston and tilt only; defocus stays in the error
        resid = W - A[:, :3] @ np.linalg.lstsq(A[:, :3], W, rcond=None)[0]
    rms = float(np.sqrt(np.mean(resid**2)))
    lam = wavelength * 1e-3
    na = semi / system.to_paraxial(wavelength).efl
    # wavefront tilt -> angular shift of the image; chief-ray offset on the image plane -> more shift
    efl = system.to_paraxial(wavelength).efl
    boresight = float(np.hypot(wf.image_point[0] / efl + coef[1] * lam / semi,
                               wf.image_point[1] / efl + coef[2] * lam / semi))
    focus = float(2 * coef[3] * lam / na**2)  # W020 = dz NA^2 / 2
    return {"rms": rms, "strehl": math.exp(-(2 * math.pi * rms) ** 2), "boresight": boresight, "focus": focus}


def sensitivity(system: SequentialSystem, tols: list[Tolerance], semi: float, z_image: float) -> list[dict]:
    """Change of every metric when ONE tolerance is pushed to +limit (others nominal)."""
    nominal = evaluate(system, semi, z_image)
    rows = []
    for k, tol in enumerate(tols):
        v = np.zeros(len(tols))
        v[k] = tol.limit
        e = evaluate(perturb(system, tols, v), semi, z_image)
        rows.append({
            "tolerance": tol,
            # RMS wavefront errors add in quadrature: report the added variance as an RMS
            "d_rms": math.copysign(math.sqrt(abs(e["rms"] ** 2 - nominal["rms"] ** 2)), e["rms"] - nominal["rms"]),
            "d_boresight": e["boresight"] - nominal["boresight"],
            "d_focus": e["focus"] - nominal["focus"],
        })
    return rows


def rss_prediction(nominal: dict, sens: list[dict], uniform: bool = True) -> dict[str, float]:
    """Expected (1-sigma) spread from the sensitivities: errors add in quadrature (root sum square).
    For a uniform distribution on ±limit the standard deviation is limit/sqrt(3)."""
    k = 1 / math.sqrt(3) if uniform else 1.0
    rms = math.sqrt(nominal["rms"] ** 2 + sum((k * r["d_rms"]) ** 2 for r in sens))
    return {
        "rms": rms,
        "boresight": math.sqrt(sum((k * r["d_boresight"]) ** 2 for r in sens)),
        "focus": math.sqrt(sum((k * r["d_focus"]) ** 2 for r in sens)),
    }


def monte_carlo(system: SequentialSystem, tols: list[Tolerance], semi: float, z_image: float,
                n_trials: int = 500, seed: int = 0, refocus: bool = True) -> dict[str, np.ndarray]:
    """Build n_trials virtual lenses with every tolerance drawn uniformly in ±limit."""
    rng = np.random.default_rng(seed)
    limits = np.array([t.limit for t in tols])
    out = {"rms": [], "strehl": [], "boresight": [], "focus": []}
    for _ in range(n_trials):
        e = evaluate(perturb(system, tols, rng.uniform(-limits, limits)), semi, z_image, refocus=refocus)
        for key in out:
            out[key].append(e[key])
    return {k: np.array(v) for k, v in out.items()}