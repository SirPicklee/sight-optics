import math

import numpy as np
import pytest

from sightoptics import (
    MOA,
    N_BK7,
    ReflexWindow,
    aiming_error,
    best_led_z,
    dot_directions,
    mangin_for_focus,
    optimise_mangin,
    parallax,
    window_points,
    zero_power_rc,
)
from sightoptics.reflex import _normal, _reflect

F = 75.0
PTS = window_points(15.0, 11.0, 11)


def test_reflection_law():
    d = np.array([[0.0, math.sin(0.3), math.cos(0.3)]])
    n = np.array([[0.0, 0.0, -1.0]])
    r = _reflect(d, n)
    assert r[0] == pytest.approx([0.0, math.sin(0.3), -math.cos(0.3)])


def test_mirror_focus_is_half_the_radius():
    assert ReflexWindow(-2 * F).focus_distance() == pytest.approx(F, rel=1e-6)


def test_mirror_normal_points_to_centre():
    n = _normal(np.array([[0.0, 0.0, 0.0]]), -150.0, 0.0)
    assert n[0] == pytest.approx([0.0, 0.0, 1.0])


def test_tiny_window_on_axis_has_no_parallax():
    pts = window_points(0.0, 0.5, 5)
    rms, _ = parallax(ReflexWindow(-2 * F), -F, pts)
    assert rms / MOA < 1e-3


def test_refocusing_the_led_reduces_mirror_parallax():
    m = ReflexWindow(-2 * F)
    assert parallax(m, best_led_z(m, PTS), PTS)[0] < parallax(m, -F, PTS)[0]


def test_led_lateral_shift_moves_dot_by_shift_over_f():
    m = ReflexWindow(-2 * F)
    pts = window_points(0.0, 1.0, 5)
    a0 = dot_directions(m, -F, pts).mean(axis=0)
    a1 = dot_directions(m, -F, pts, led_xy=(0.0, 0.1)).mean(axis=0)
    assert abs(a1[1] - a0[1]) == pytest.approx(0.1 / F, rel=0.01)


def test_zero_power_window():
    rc = zero_power_rc(-80.0, N_BK7, 6.0)
    w = ReflexWindow(-120.0, N_BK7, -80.0, rc, 3.0, 3.0)
    assert w.transmission_power == pytest.approx(0.0, abs=1e-15)


def test_mangin_hits_requested_focus():
    w = mangin_for_focus(F, -80.0, N_BK7)
    assert w.focus_distance() == pytest.approx(F, rel=1e-6)


def test_optimised_mangin_beats_bare_mirror_by_far():
    m = ReflexWindow(-2 * F)
    w = optimise_mangin(F, N_BK7, PTS, (-100.0, -60.0))
    e_m = aiming_error(m, best_led_z(m, PTS, metric=aiming_error), PTS)[0]
    e_w = aiming_error(w, best_led_z(w, PTS, metric=aiming_error), PTS)[0]
    assert e_w < e_m / 20