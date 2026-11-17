"""First-order (thin-lens) optics for telescopic sights.

Sign convention (Gaussian, real-is-positive):
    1/s_o + 1/s_i = 1/f
    s_o > 0 : real object to the left of the lens
    s_i > 0 : real image to the right of the lens
All lengths in millimetres, angles in degrees unless stated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def thin_lens_image(f: float, s_o: float) -> float:
    """Image distance for a thin lens. Returns inf if the object is at the focal point."""
    if f == 0:
        raise ValueError("focal length must be non-zero")
    if math.isinf(s_o):
        return f
    if math.isclose(s_o, f):
        return math.inf
    return 1.0 / (1.0 / f - 1.0 / s_o)


def lateral_magnification(s_o: float, s_i: float) -> float:
    """Lateral (transverse) magnification m = -s_i / s_o."""
    return -s_i / s_o


@dataclass(frozen=True)
class KeplerTelescope:
    """Afocal Kepler telescope: objective + eyepiece, separated by f_obj + f_eye.

    Parameters
    ----------
    f_objective : objective focal length [mm]
    f_eyepiece : eyepiece focal length [mm]
    d_objective : objective clear aperture (entrance pupil) [mm]
    field_stop : field stop diameter at the common focal plane [mm]
    """

    f_objective: float
    f_eyepiece: float
    d_objective: float
    field_stop: float

    def __post_init__(self) -> None:
        for name in ("f_objective", "f_eyepiece", "d_objective", "field_stop"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def magnification(self) -> float:
        """Angular magnification |M| = f_obj / f_eye (image is inverted)."""
        return self.f_objective / self.f_eyepiece

    @property
    def length(self) -> float:
        """Lens separation for an afocal system [mm]."""
        return self.f_objective + self.f_eyepiece

    @property
    def exit_pupil(self) -> float:
        """Exit pupil diameter D / M [mm]."""
        return self.d_objective / self.magnification

    @property
    def eye_relief(self) -> float:
        """Distance from eyepiece to exit pupil (image of objective) [mm]."""
        return thin_lens_image(self.f_eyepiece, self.length)

    @property
    def real_fov(self) -> float:
        """Full real (object-space) field of view [deg]."""
        return 2.0 * math.degrees(math.atan(self.field_stop / (2.0 * self.f_objective)))

    @property
    def apparent_fov(self) -> float:
        """Full apparent (eye-side) field of view [deg]."""
        return 2.0 * math.degrees(math.atan(self.field_stop / (2.0 * self.f_eyepiece)))

    @property
    def fov_m_at_100m(self) -> float:
        """Linear field of view at 100 m [m] (common catalogue figure)."""
        return 2.0 * 100.0 * math.tan(math.radians(self.real_fov / 2.0))

    @property
    def twilight_factor(self) -> float:
        """Twilight factor sqrt(M * D) (DIN 58388)."""
        return math.sqrt(self.magnification * self.d_objective)

    @property
    def relative_brightness(self) -> float:
        """Relative brightness = exit_pupil^2 [mm^2]."""
        return self.exit_pupil ** 2

    def summary(self) -> dict[str, float]:
        return {
            "magnification": self.magnification,
            "length_mm": self.length,
            "exit_pupil_mm": self.exit_pupil,
            "eye_relief_mm": self.eye_relief,
            "real_fov_deg": self.real_fov,
            "apparent_fov_deg": self.apparent_fov,
            "fov_m_at_100m": self.fov_m_at_100m,
            "twilight_factor": self.twilight_factor,
            "relative_brightness": self.relative_brightness,
        }