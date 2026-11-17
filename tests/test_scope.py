import math

import numpy as np
import pytest

from sightoptics import ZoomScope, relay

FS = 2 * 45 * math.tan(math.radians(11))  # field stop for a 22° apparent field


@pytest.fixture(scope="module")
def scope():
    return ZoomScope(f_objective=135, d_objective=40, f_erector=25, erector_length=80, f_eyepiece=45, field_stop=FS)


def test_zoom_range(scope):
    assert scope.magnification_range == pytest.approx((3.0, 9.0))


@pytest.mark.parametrize("M", [3.0, 4.0, 5.5, 7.0, 9.0])
def test_afocal_and_upright_at_every_zoom(scope, M):
    sys_ = scope.system(M)
    assert sys_.is_afocal(tol=1e-10)
    assert sys_.angular_magnification == pytest.approx(M)  # positive = upright image


@pytest.mark.parametrize("M", [3.0, 6.0, 9.0])
def test_erector_keeps_image_on_sfp(scope, M):
    a, t, b = scope.erector_positions(M)
    A, B, _, _ = relay(a, t, b, scope.f_erector, scope.f_erector).abcd
    assert B == pytest.approx(0.0, abs=1e-9)
    assert A == pytest.approx(scope.erector_magnification(M))
    assert a + t + b == pytest.approx(scope.erector_length)


def test_cam_curves_are_smooth(scope):
    Ms = np.linspace(3, 9, 61)
    pos = np.array([scope.erector_positions(M)[:2] for M in Ms])
    lens2 = pos[:, 0] + pos[:, 1]
    assert np.max(np.abs(np.diff(pos[:, 0]))) < 2.0
    assert np.max(np.abs(np.diff(lens2))) < 2.0


def test_exit_pupil_from_ray_trace(scope):
    M = 6.0
    _, ys, _ = scope.system(M).trace(20.0, 0.0)
    assert 2 * abs(ys[-1]) == pytest.approx(scope.exit_pupil(M))


def test_eye_relief_is_where_chief_ray_crosses_axis(scope):
    M = 9.0
    _, ys, us = scope.system(M).trace(0.0, 0.01)
    assert -ys[-1] / us[-1] == pytest.approx(scope.eye_relief(M))


def test_real_fov_scales_inversely_with_magnification(scope):
    assert scope.fov_at_100m(3.0) / scope.fov_at_100m(9.0) == pytest.approx(3.0, rel=1e-3)


def test_ffp_reticle_subtension_is_constant(scope):
    a = [scope.reticle_subtension(M, 0.135, "FFP") for M in (3, 6, 9)]
    assert a == pytest.approx([1e-3] * 3)  # 0.135 mm at f = 135 mm -> 1 mrad


def test_sfp_reticle_is_correct_only_at_calibration_power(scope):
    s = scope.sfp_spacing_for(1e-3, calibrated_M=9.0)
    assert scope.reticle_subtension(9.0, s, "SFP") == pytest.approx(1e-3)
    assert scope.reticle_subtension(3.0, s, "SFP") == pytest.approx(3e-3)


def test_no_parallax_at_the_parallax_free_distance(scope):
    assert scope.parallax_error(100, 100, 20.0) == pytest.approx(0.0, abs=1e-9)


def test_parallax_rule_of_thumb(scope):
    # error at the target ≈ h |1 - R / R0|
    assert abs(scope.parallax_error(300, 100, 20.0)) == pytest.approx(20.0 * 2, rel=0.01)