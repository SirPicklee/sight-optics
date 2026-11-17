"""Wavefront error (OPD), point spread function and MTF by wave optics.

1) Trace a square grid of rays across the entrance pupil, accumulating optical path length.
2) Measure the OPD of every ray on a reference sphere centred on the image point and passing
   through the exit pupil -> wavefront error map W(x, y).
3) Pupil function P = A exp(2πi W/λ);  PSF = |FFT(P)|²;  OTF = FFT(PSF);  MTF = |OTF|.
Stop at the first surface, object at infinity, lengths in mm, wavelengths in µm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .glass import WL_C, WL_D, WL_F, refractive_index
from .raytrace import SequentialSystem, intersect, refract, surface_normal


def trace_opl(system: SequentialSystem, pos, dirs, wavelength: float = WL_D):
    """Trace rays and accumulate optical path length from a plane wavefront through the origin."""
    pos = np.array(pos, dtype=float)
    dirs = np.array(dirs, dtype=float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    n1 = refractive_index(system.n_object, wavelength)
    opl = n1 * np.einsum("ij,ij->i", pos, dirs)  # start on the plane wavefront x·d = 0
    for s, zv in zip(system.surfaces, system.vertex_z):
        n2 = refractive_index(s.n, wavelength)
        new = intersect(pos, dirs, s.c, zv, s.dy)
        r = np.hypot(new[:, 0], new[:, 1])
        new[r > s.semi_diameter] = np.nan
        opl = opl + n1 * np.linalg.norm(new - pos, axis=1)
        dirs = refract(dirs, surface_normal(new, s.c, zv, s.dy), n1, n2)
        pos, n1 = new, n2
    return pos, dirs, opl


def exit_pupil_z(system: SequentialSystem, wavelength: float = WL_D) -> float:
    """Axial position of the exit pupil (paraxial image of the stop at the first surface)."""
    return system.last_vertex + system.to_paraxial(wavelength).stop_image_distance()


def _pupil_coords(semi: float, n: int):
    dp = 2.0 * semi / n
    g = (np.arange(n) - n / 2 + 0.5) * dp
    x, y = np.meshgrid(g, g)
    return x, y, dp


@dataclass
class Wavefront:
    W: np.ndarray  # OPD map [waves], NaN outside the pupil, mean (piston) removed
    mask: np.ndarray  # True inside the pupil
    dp: float  # pupil sample spacing [mm]
    wavelength: float  # [µm]
    image_point: np.ndarray  # reference sphere centre (x, y, z)

    @property
    def rms(self) -> float:
        return float(np.nanstd(self.W))

    @property
    def pv(self) -> float:
        return float(np.nanmax(self.W) - np.nanmin(self.W))

    @property
    def max_step(self) -> float:
        """Largest OPD change between neighbouring pupil samples [waves].
        Above ~0.5 wave the phase is under-sampled (aliasing): increase n."""
        return float(max(np.nanmax(np.abs(np.diff(self.W, axis=0))), np.nanmax(np.abs(np.diff(self.W, axis=1)))))


def wavefront(system: SequentialSystem, semi: float, field_deg: float, z_image: float,
              wavelength: float = WL_D, n: int = 64, image_point: np.ndarray | None = None) -> Wavefront:
    """Wavefront error map on the exit-pupil reference sphere centred on the image point.

    image_point defaults to where the chief ray of this wavelength hits the plane z = z_image.
    """
    a = math.radians(field_deg)
    d = np.array([0.0, math.sin(a), math.cos(a)])
    x, y, dp = _pupil_coords(semi, n)
    mask = x**2 + y**2 <= semi**2
    pts = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    pts = np.vstack((pts, [0.0, 0.0, 0.0]))  # last ray = chief ray
    z0 = -20.0
    pos = pts + (z0 / d[2]) * d
    p, k, opl = trace_opl(system, pos, np.tile(d, (len(pos), 1)), wavelength)

    pc, kc = p[-1], k[-1]
    if image_point is None:
        image_point = pc + (z_image - pc[2]) / kc[2] * kc
    z_ep = exit_pupil_z(system, wavelength)
    e = pc + (z_ep - pc[2]) / kc[2] * kc  # chief ray on the exit pupil plane
    R = np.linalg.norm(image_point - e)

    # distance along each ray to the reference sphere (root nearest to the exit pupil)
    q = p - image_point
    b = np.einsum("ij,ij->i", k, q)
    c = np.einsum("ij,ij->i", q, q) - R**2
    root = np.sqrt(b**2 - c)
    t1, t2 = -b - root, -b + root
    t = np.where(np.abs(t1) < np.abs(t2), t1, t2)
    total = opl + t  # image space is air (n = 1)

    lam_mm = wavelength * 1e-3
    W = ((total[:-1] - total[-1]) / lam_mm).reshape(x.shape)
    W[~mask] = np.nan
    W -= np.nanmean(W)
    return Wavefront(W, mask & ~np.isnan(W), dp, wavelength, image_point)


def required_samples(system: SequentialSystem, semi: float, field_deg: float, z_image: float,
                     wavelengths: Sequence[float], n0: int = 64, n_max: int = 512, max_step: float = 0.5) -> int:
    """Smallest pupil grid (n0, 2·n0, ...) whose OPD step between samples stays below max_step waves."""
    n = n0
    while n < n_max and max(wavefront(system, semi, field_deg, z_image, wl, n).max_step for wl in wavelengths) > max_step:
        n *= 2
    return n


@dataclass
class PSF:
    psf: np.ndarray  # normalised so that a perfect pupil has peak 1 (value at peak = Strehl)
    dx: float  # image-plane sample spacing [mm]
    otf: np.ndarray  # complex OTF, fftshifted, OTF(0) = 1
    freq: np.ndarray  # 1D frequency axis of the shifted OTF [cycles/mm]

    @property
    def strehl(self) -> float:
        return float(self.psf.max())

    def mtf_slices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Positive-frequency MTF slices: (freq, sagittal = along x, tangential = along y)."""
        c = len(self.freq) // 2
        f = self.freq[c:]
        return f, np.abs(self.otf[c, c:]), np.abs(self.otf[c:, c])

    def otf_slices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = len(self.freq) // 2
        return self.freq[c:], self.otf[c, c:], self.otf[c:, c]


def psf_from_wavefront(wf: Wavefront, efl: float, pad: int = 8) -> PSF:
    """Fraunhofer PSF and OTF from a wavefront map (entrance-pupil sampling scaled by the EFL)."""
    n = wf.W.shape[0]
    nfft = n * pad
    pupil = np.where(wf.mask, np.exp(2j * np.pi * np.nan_to_num(wf.W)), 0.0)
    field = np.fft.fftshift(np.fft.fft2(pupil, s=(nfft, nfft)))
    psf = np.abs(field) ** 2 / wf.mask.sum() ** 2
    dx = wf.wavelength * 1e-3 * efl / (nfft * wf.dp)
    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
    otf /= otf[nfft // 2, nfft // 2]
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, dx))
    return PSF(psf, dx, otf, freq)


def diffraction_mtf(freq: np.ndarray, cutoff: float) -> np.ndarray:
    """MTF of a perfect circular pupil: (2/π)(φ - cos φ sin φ), φ = acos(ν/ν_c)."""
    v = np.clip(np.asarray(freq, dtype=float) / cutoff, 0.0, 1.0)
    phi = np.arccos(v)
    return (2 / np.pi) * (phi - np.cos(phi) * np.sin(phi))


# Approximate CIE photopic eye sensitivity V(λ) at the F, d and C lines (interpolated from the CIE table)
PHOTOPIC = {WL_F: 0.18, WL_D: 0.78, WL_C: 0.077}


def polychromatic_mtf(system: SequentialSystem, semi: float, field_deg: float, z_image: float,
                      weights: dict[float, float], freq: np.ndarray, n: int = 64, pad: int = 8):
    """Weighted polychromatic MTF (sagittal, tangential) on a common frequency axis.

    weights: {wavelength_um: weight}. The complex OTFs are averaged with these weights, all
    referenced to the same image point (the d-line chief ray), so lateral colour costs contrast.
    """
    efl = system.to_paraxial().efl
    ref = wavefront(system, semi, field_deg, z_image, WL_D, n).image_point
    sag = np.zeros_like(freq, dtype=complex)
    tan = np.zeros_like(freq, dtype=complex)
    total = sum(weights.values())
    for wl, w in weights.items():
        f, s, t = psf_from_wavefront(wavefront(system, semi, field_deg, z_image, wl, n, ref), efl, pad).otf_slices()
        sag += w * (np.interp(freq, f, s.real) + 1j * np.interp(freq, f, s.imag))
        tan += w * (np.interp(freq, f, t.real) + 1j * np.interp(freq, f, t.imag))
    return np.abs(sag) / total, np.abs(tan) / total