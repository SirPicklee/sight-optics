"""Optical glass dispersion with the Sellmeier equation.

n^2(λ) = 1 + Σ B_i λ² / (λ² − C_i),   λ in micrometres.
Coefficients from the Schott optical glass catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fraunhofer lines [µm]
WL_F = 0.4861327  # blue (hydrogen)
WL_D = 0.5875618  # yellow (helium d-line), design wavelength
WL_C = 0.6562725  # red (hydrogen)


@dataclass(frozen=True)
class Glass:
    name: str
    B: tuple[float, float, float]
    C: tuple[float, float, float]

    def n(self, wavelength_um):
        """Refractive index at the given wavelength(s) [µm]."""
        l2 = np.asarray(wavelength_um, dtype=float) ** 2
        n2 = 1.0 + sum(b * l2 / (l2 - c) for b, c in zip(self.B, self.C))
        return np.sqrt(n2)

    @property
    def nd(self) -> float:
        return float(self.n(WL_D))

    @property
    def vd(self) -> float:
        """Abbe number V_d = (n_d - 1) / (n_F - n_C)."""
        return float((self.n(WL_D) - 1.0) / (self.n(WL_F) - self.n(WL_C)))


N_BK7 = Glass("N-BK7", (1.03961212, 0.231792344, 1.01046945), (0.00600069867, 0.0200179144, 103.560653))
N_F2 = Glass("N-F2", (1.39757037, 0.159201403, 1.2686543), (0.00995906143, 0.0546931752, 119.248346))


@dataclass(frozen=True)
class OffsetGlass:
    """A glass melt whose index differs from the catalogue by a constant dn (manufacturing tolerance)."""

    base: Glass
    dn: float

    @property
    def name(self) -> str:
        return self.base.name

    def n(self, wavelength_um):
        return self.base.n(wavelength_um) + self.dn


def refractive_index(material, wavelength_um: float) -> float:
    """Index of a material that is a plain number, a Glass or an OffsetGlass."""
    return float(material.n(wavelength_um)) if hasattr(material, "n") else float(material)