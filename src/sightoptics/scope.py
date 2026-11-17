"""First-order model of a variable-magnification (zoom) rifle scope.

Layout (thin lenses, light from left to right):

    objective --f_o--> FFP --a--> erector 1 --t--> erector 2 --b--> SFP --f_e--> eyepiece --ER--> eye
                       (first focal plane)                         (second focal plane)

The two erector lenses relay the inverted image at the FFP to the SFP with magnification m_e < 0,
which turns the image upright again. FFP -> SFP distance L = a + t + b is fixed by the tube;
zooming moves both erector lenses (a "variator" and a "compensator") along cam curves so the image
stays on the SFP while m_e changes. Total magnification M = (f_o / f_e) * |m_e|.
The objective is the aperture stop; lengths in mm, angles in radians unless stated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import fsolve

from .paraxial import ParaxialSystem, Space, ThinLens


def relay(a: float, t: float, b: float, f1: float, f2: float) -> ParaxialSystem:
    return ParaxialSystem([Space(a), ThinLens(f1), Space(t), ThinLens(f2), Space(b)])


@dataclass
class ZoomScope:
    f_objective: float
    d_objective: float
    f_erector: float  # both erector lenses (equal focal lengths)
    erector_length: float  # FFP -> SFP distance L
    f_eyepiece: float
    field_stop: float  # diameter of the field stop at the SFP
    m_range: tuple[float, float] = (-1.0, -3.0)  # erector magnification at min and max zoom
    _cam: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the cam table by continuation: solve, step m_e a little, solve again from there."""
        L, f = self.erector_length, self.f_erector
        guess = np.array([0.3 * L, 0.4 * L])
        rows = []
        for m in np.linspace(self.m_range[0], self.m_range[1], 201):
            sol = self._solve(m, guess)
            if sol is None:
                raise ValueError(f"no erector solution for m_e = {m:.3f}; change f_erector or erector_length")
            guess = sol
            rows.append((m, sol[0], sol[1]))
        self._cam = np.array(rows)

    def _solve(self, m: float, guess) -> np.ndarray | None:
        L, f = self.erector_length, self.f_erector

        def eq(v):
            a, t = v
            A, B, _, _ = relay(a, t, L - a - t, f, f).abcd
            return [B, A - m]  # B = 0: FFP imaged onto SFP;  A = m: required magnification

        sol, _, ier, _ = fsolve(eq, guess, full_output=True, xtol=1e-12)
        a, t = sol
        if ier != 1 or min(a, t, L - a - t) < 1.0:
            return None
        return sol

    # ---- zoom ----
    @property
    def base_magnification(self) -> float:
        return self.f_objective / self.f_eyepiece

    @property
    def magnification_range(self) -> tuple[float, float]:
        return tuple(self.base_magnification * abs(m) for m in self.m_range)

    def erector_magnification(self, M: float) -> float:
        return -M / self.base_magnification

    def erector_positions(self, M: float) -> tuple[float, float, float]:
        """(a, t, b): FFP -> erector 1, erector 1 -> erector 2, erector 2 -> SFP."""
        m = self.erector_magnification(M)
        lo, hi = sorted(self.m_range)
        if not lo - 1e-9 <= m <= hi + 1e-9:
            raise ValueError(f"M = {M} outside the zoom range {self.magnification_range}")
        order = np.argsort(self._cam[:, 0])
        guess = [np.interp(m, self._cam[order, 0], self._cam[order, i]) for i in (1, 2)]
        a, t = self._solve(m, guess)
        return a, t, self.erector_length - a - t

    def system(self, M: float) -> ParaxialSystem:
        a, t, b = self.erector_positions(M)
        f = self.f_erector
        return ParaxialSystem([
            ThinLens(self.f_objective), Space(self.f_objective + a), ThinLens(f), Space(t), ThinLens(f),
            Space(b + self.f_eyepiece), ThinLens(self.f_eyepiece),
        ])

    @property
    def length(self) -> float:
        """Objective -> eyepiece (optical length, thin lenses)."""
        return self.f_objective + self.erector_length + self.f_eyepiece

    # ---- user-facing numbers ----
    def eye_relief(self, M: float) -> float:
        return self.system(M).stop_image_distance()

    def exit_pupil(self, M: float) -> float:
        return self.d_objective / M

    def real_fov(self, M: float) -> float:
        """Full real field of view [rad]; the field stop sits at the SFP."""
        h_ffp = self.field_stop / 2 / abs(self.erector_magnification(M))
        return 2 * math.atan(h_ffp / self.f_objective)

    def fov_at_100m(self, M: float) -> float:
        return 2 * 100.0 * math.tan(self.real_fov(M) / 2)

    @property
    def apparent_fov(self) -> float:
        return 2 * math.atan(self.field_stop / 2 / self.f_eyepiece)

    # ---- reticle ----
    def reticle_subtension(self, M: float, spacing_mm: float, plane: str) -> float:
        """Object-space angle [rad] between two reticle marks spacing_mm apart.

        FFP: the reticle is magnified together with the target -> constant angle.
        SFP: the reticle is behind the zoom -> its angle on the target changes with M.
        """
        if plane.upper() == "FFP":
            return spacing_mm / self.f_objective
        if plane.upper() == "SFP":
            return spacing_mm / (self.f_objective * abs(self.erector_magnification(M)))
        raise ValueError("plane must be 'FFP' or 'SFP'")

    def sfp_spacing_for(self, angle_rad: float, calibrated_M: float) -> float:
        """SFP mark spacing [mm] that subtends angle_rad at the calibration magnification."""
        return angle_rad * self.f_objective * abs(self.erector_magnification(calibrated_M))

    # ---- parallax ----
    def parallax_error(self, target_m: float, parallax_free_m: float, pupil_height: float) -> float:
        """Apparent shift of the reticle on the target [mm at the target] when the eye looks through
        the entrance-pupil height pupil_height, with the reticle focused for parallax_free_m.

        Exact thin-lens geometry: the target images at s', the reticle sits at s0'; the ray through
        pupil height h crosses the reticle plane at h (1 - s0'/s'), which subtends (that / f_o)
        in object space.
        """
        f = self.f_objective
        s_img = 1.0 / (1.0 / f - 1.0 / (target_m * 1e3))
        s_ret = 1.0 / (1.0 / f - 1.0 / (parallax_free_m * 1e3)) if math.isfinite(parallax_free_m) else f
        y_ret = pupil_height * (1.0 - s_ret / s_img)
        return y_ret / f * target_m * 1e3