import math

import numpy as np
import pytest

from sightoptics import KeplerTelescope, ParaxialSystem, Space, ThinLens, kepler, thick_lens


def test_space_and_lens_determinant_is_one():
    sys_ = ParaxialSystem([ThinLens(80.0), Space(37.0), ThinLens(-45.0), Space(12.0)])
    assert np.linalg.det(sys_.matrix) == pytest.approx(1.0)


def test_single_thin_lens_efl():
    assert ParaxialSystem([ThinLens(120.0)]).efl == pytest.approx(120.0)


def test_two_thin_lenses_combination_formula():
    f1, f2, d = 100.0, 50.0, 30.0
    expected = 1.0 / (1.0 / f1 + 1.0 / f2 - d / (f1 * f2))
    sys_ = ParaxialSystem([ThinLens(f1), Space(d), ThinLens(f2)])
    assert sys_.efl == pytest.approx(expected)


def test_parallel_ray_crosses_axis_at_bfl():
    sys_ = ParaxialSystem([ThinLens(100.0), Space(30.0), ThinLens(50.0)])
    _, ys, us = sys_.trace(5.0, 0.0)
    assert -ys[-1] / us[-1] == pytest.approx(sys_.bfl)


def test_kepler_is_afocal_and_inverting():
    sys_ = kepler(200.0, 25.0)
    assert sys_.is_afocal()
    assert sys_.angular_magnification == pytest.approx(-8.0)


def test_kepler_eye_relief_matches_first_order():
    sys_ = kepler(200.0, 25.0)
    ref = KeplerTelescope(200.0, 25.0, 40.0, 20.0)
    assert sys_.stop_image_distance() == pytest.approx(ref.eye_relief)


def test_kepler_exit_beam_diameter_is_exit_pupil():
    sys_ = kepler(200.0, 25.0)
    _, ys, _ = sys_.trace(20.0, 0.0)  # marginal ray, D/2 = 20 mm
    assert 2 * abs(ys[-1]) == pytest.approx(40.0 / 8.0)


def test_thick_lens_matches_thick_lensmaker_equation():
    R1, R2, t, n = 100.0, -100.0, 8.0, 1.5168
    inv_f = (n - 1) * (1 / R1 - 1 / R2 + (n - 1) * t / (n * R1 * R2))
    assert thick_lens(R1, R2, t, n).efl == pytest.approx(1 / inv_f)


def test_flat_surface_has_no_power():
    lens = thick_lens(math.inf, math.inf, 10.0, 1.5)
    assert lens.abcd[2] == pytest.approx(0.0)