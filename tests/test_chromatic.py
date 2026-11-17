import math

import numpy as np
import pytest

from sightoptics import (
    N_BK7,
    N_F2,
    WL_C,
    WL_D,
    WL_F,
    SequentialSystem,
    SphericalSurface,
    achromat_design,
    achromat_powers,
    best_focus,
    cemented_achromat,
    chromatic_focal_shift,
    collimated_bundle,
    correct_color,
    meridional_fan,
    planoconvex,
    pupil_grid,
    rms_radius,
    scale_to_efl,
    trace_to_plane,
)

F, SEMI = 200.0, 20.0


def test_catalogue_values_bk7():
    assert N_BK7.nd == pytest.approx(1.5168, abs=1e-4)
    assert N_BK7.vd == pytest.approx(64.17, abs=0.05)


def test_catalogue_values_f2():
    assert N_F2.nd == pytest.approx(1.6200, abs=1e-4)
    assert N_F2.vd == pytest.approx(36.43, abs=0.05)


def test_normal_dispersion_blue_bends_more():
    assert N_BK7.n(WL_F) > N_BK7.n(WL_D) > N_BK7.n(WL_C)


def test_plain_number_material_ignores_wavelength():
    s = SequentialSystem([SphericalSurface(100.0, 8.0, 1.5), SphericalSurface(-100.0, 0.0, 1.0)])
    pos, dirs = meridional_fan(np.array([10.0]))
    a, _ = s.trace(pos, dirs, wavelength=WL_F)
    b, _ = s.trace(pos, dirs, wavelength=WL_C)
    assert a == pytest.approx(b)


def test_thin_lens_achromat_condition():
    f1, f2 = achromat_powers(F, N_BK7, N_F2)
    assert 1 / f1 + 1 / f2 == pytest.approx(1 / F)
    assert (1 / f1) / N_BK7.vd + (1 / f2) / N_F2.vd == pytest.approx(0.0, abs=1e-12)
    assert f1 > 0 > f2  # positive crown, negative flint


def test_correct_color_makes_f_and_c_foci_coincide():
    s = correct_color(cemented_achromat(F, SEMI, N_BK7, N_F2, 0.011), WL_F, WL_C)
    assert s.to_paraxial(WL_F).bfl == pytest.approx(s.to_paraxial(WL_C).bfl, abs=1e-9)


def test_scale_to_efl():
    s = scale_to_efl(planoconvex(F, SEMI, N_BK7), F)
    assert s.to_paraxial().efl == pytest.approx(F)


def test_singlet_longitudinal_colour_is_f_over_v():
    s = scale_to_efl(planoconvex(F, SEMI, N_BK7), F)
    bfl_f, bfl_c = chromatic_focal_shift(s, [WL_F, WL_C])
    assert bfl_c - bfl_f == pytest.approx(F / N_BK7.vd, rel=0.05)


def test_pupil_grid_inside_circle():
    xy = pupil_grid(5.0, 11)
    assert np.all(np.hypot(xy[:, 0], xy[:, 1]) <= 5.0 + 1e-9)
    assert len(xy) > 50


def test_rms_radius_known_value():
    pts = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    assert rms_radius(pts) == pytest.approx(1.0)


def test_best_focus_not_worse_than_paraxial_focus():
    s = scale_to_efl(planoconvex(F, SEMI, N_BK7), F)
    z_par = s.last_vertex + s.to_paraxial().bfl
    pos, dirs = collimated_bundle(SEMI, 0.0, 15)
    rms_par = rms_radius(trace_to_plane(s, pos, dirs, z_par, WL_D))
    _, rms_best = best_focus(s, SEMI, wavelengths=[WL_D], n=15)
    assert rms_best <= rms_par


def test_doublet_beats_singlet_by_a_wide_margin():
    singlet = scale_to_efl(planoconvex(F, SEMI, N_BK7), F)
    doublet = achromat_design(F, SEMI, N_BK7, N_F2, 0.011)
    _, r_s = best_focus(singlet, SEMI, n=15)
    _, r_d = best_focus(doublet, SEMI, n=15)
    assert r_d < r_s / 5