import math

import pytest

from sightoptics import KeplerTelescope, lateral_magnification, thin_lens_image


def test_thin_lens_2f_2f():
    # Object at 2f -> image at 2f, magnification -1
    s_i = thin_lens_image(100.0, 200.0)
    assert s_i == pytest.approx(200.0)
    assert lateral_magnification(200.0, s_i) == pytest.approx(-1.0)


def test_thin_lens_object_at_infinity_images_at_focus():
    assert thin_lens_image(50.0, math.inf) == pytest.approx(50.0)


def test_thin_lens_object_at_focus_images_at_infinity():
    assert math.isinf(thin_lens_image(50.0, 50.0))


@pytest.fixture
def scope_8x40():
    return KeplerTelescope(f_objective=200.0, f_eyepiece=25.0, d_objective=40.0, field_stop=20.0)


def test_magnification_and_length(scope_8x40):
    assert scope_8x40.magnification == pytest.approx(8.0)
    assert scope_8x40.length == pytest.approx(225.0)


def test_exit_pupil(scope_8x40):
    assert scope_8x40.exit_pupil == pytest.approx(5.0)


def test_eye_relief_formula(scope_8x40):
    # ER = f_e * (f_o + f_e) / f_o
    assert scope_8x40.eye_relief == pytest.approx(25.0 * 225.0 / 200.0)


def test_apparent_fov_is_about_m_times_real_fov(scope_8x40):
    # tan(apparent/2) = M * tan(real/2) exactly for a thin-lens Kepler telescope
    t_real = math.tan(math.radians(scope_8x40.real_fov / 2))
    t_app = math.tan(math.radians(scope_8x40.apparent_fov / 2))
    assert t_app == pytest.approx(scope_8x40.magnification * t_real)


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        KeplerTelescope(f_objective=-100.0, f_eyepiece=25.0, d_objective=40.0, field_stop=20.0)