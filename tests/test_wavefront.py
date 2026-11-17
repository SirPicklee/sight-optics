import math

import numpy as np
import pytest

from sightoptics import (
    N_BK7,
    N_F2,
    WL_D,
    Wavefront,
    achromat_design,
    diffraction_mtf,
    exit_pupil_z,
    meridional_fan,
    planoconvex,
    psf_from_wavefront,
    required_samples,
    scale_to_efl,
    wavefront,
)

F, SEMI = 200.0, 20.0
LAM_MM = WL_D * 1e-3


@pytest.fixture(scope="module")
def doublet():
    return achromat_design(F, SEMI, N_BK7, N_F2, 0.01108)  # bending solution #2


def perfect(n=64, semi=SEMI):
    dp = 2 * semi / n
    g = (np.arange(n) - n / 2 + 0.5) * dp
    x, y = np.meshgrid(g, g)
    mask = x**2 + y**2 <= semi**2
    return Wavefront(np.where(mask, 0.0, np.nan), mask, dp, WL_D, np.zeros(3))


def test_perfect_pupil_mtf_matches_analytic_formula():
    p = psf_from_wavefront(perfect(), F)
    f, sag, tan = p.mtf_slices()
    cutoff = 1 / (LAM_MM * F / (2 * SEMI))
    assert np.max(np.abs(sag - diffraction_mtf(f, cutoff))) < 0.01
    assert np.max(np.abs(tan - diffraction_mtf(f, cutoff))) < 0.01


def test_perfect_pupil_has_strehl_one():
    assert psf_from_wavefront(perfect(), F).strehl == pytest.approx(1.0, abs=1e-9)


def test_airy_first_zero_position():
    p = psf_from_wavefront(perfect(), F, pad=16)
    c = p.psf.shape[0] // 2
    profile = p.psf[c, c:]
    first_min = np.argmax(np.diff(profile) > 0)  # first index where the profile starts rising again
    r = first_min * p.dx
    assert r == pytest.approx(1.22 * LAM_MM * F / (2 * SEMI), rel=0.05)


def test_exit_pupil_is_where_chief_ray_crosses_axis(doublet):
    a = math.radians(0.01)
    pos, dirs = meridional_fan(np.array([-20.0 * math.tan(a)]), a)
    p, k = doublet.trace(pos, dirs)
    z_cross = p[0, 2] - p[0, 1] * k[0, 2] / k[0, 1]
    assert exit_pupil_z(doublet) == pytest.approx(z_cross, abs=1e-4)


def test_pure_defocus_rms_is_w020_over_sqrt12(doublet):
    # tiny aperture: aberrations negligible, only defocus remains
    semi, dz = 2.0, 0.5
    z_par = doublet.last_vertex + doublet.to_paraxial().bfl
    wf = wavefront(doublet, semi, 0.0, z_par + dz, WL_D)
    na = semi / F
    w020 = dz * na**2 / 2 / LAM_MM
    assert wf.rms == pytest.approx(w020 / math.sqrt(12), rel=0.02)


def test_small_aperture_at_paraxial_focus_is_nearly_perfect(doublet):
    z_par = doublet.last_vertex + doublet.to_paraxial().bfl
    assert wavefront(doublet, 2.0, 0.0, z_par, WL_D).rms < 1e-3


def test_marechal_approximation_for_small_errors(doublet):
    z_par = doublet.last_vertex + doublet.to_paraxial().bfl
    wf = wavefront(doublet, SEMI, 0.0, z_par - 0.075, WL_D)
    strehl = psf_from_wavefront(wf, F).strehl
    assert wf.rms < 0.07  # λ/14: diffraction limited
    assert strehl == pytest.approx(math.exp(-(2 * math.pi * wf.rms) ** 2), abs=0.03)


def test_doublet_pupil_is_well_sampled(doublet):
    z_par = doublet.last_vertex + doublet.to_paraxial().bfl
    assert wavefront(doublet, SEMI, 0.0, z_par - 0.065, WL_D, n=64).max_step < 0.5


def test_large_aberrations_need_a_finer_pupil_grid():
    singlet = scale_to_efl(planoconvex(F, SEMI, N_BK7), F)
    z = singlet.last_vertex + singlet.to_paraxial().bfl - 1.0
    n = required_samples(singlet, SEMI, 0.0, z, [WL_D])
    assert n > 64
    assert wavefront(singlet, SEMI, 0.0, z, WL_D, n).max_step < 0.5