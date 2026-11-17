import math

import numpy as np
import pytest

from sightoptics import (
    SequentialSystem,
    SphericalSurface,
    axis_crossing,
    intersect,
    meridional_fan,
    refract,
    surface_normal,
    thick_lens,
)

N_BK7 = 1.5168


def biconvex(t=8.0, semi=20.0):
    return SequentialSystem([
        SphericalSurface(100.0, t, N_BK7, semi),
        SphericalSurface(-100.0, 0.0, 1.0, semi),
    ])


def test_normal_incidence_is_not_bent():
    d = np.array([[0.0, 0.0, 1.0]])
    n = np.array([[0.0, 0.0, 1.0]])
    assert refract(d, n, 1.0, 1.5) == pytest.approx(d)


def test_snell_law_holds():
    theta = math.radians(30)
    d = np.array([[0.0, math.sin(theta), math.cos(theta)]])
    out = refract(d, np.array([[0.0, 0.0, 1.0]]), 1.0, 1.5)
    assert 1.0 * math.sin(theta) == pytest.approx(1.5 * out[0, 1])
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)


def test_total_internal_reflection_kills_ray():
    theta = math.radians(60)  # above critical angle 41.8 deg for n = 1.5 -> 1.0
    d = np.array([[0.0, math.sin(theta), math.cos(theta)]])
    out = refract(d, np.array([[0.0, 0.0, 1.0]]), 1.5, 1.0)
    assert np.all(np.isnan(out))


def test_intersection_with_flat_plane():
    pos, dirs = meridional_fan(np.array([3.0]), math.radians(10), z0=-10.0)
    hit = intersect(pos, dirs, 0.0, 5.0)
    assert hit[0, 2] == pytest.approx(5.0)
    assert hit[0, 1] == pytest.approx(3.0 + 15.0 * math.tan(math.radians(10)))


def test_intersection_point_lies_on_sphere():
    R, zv = 50.0, 2.0
    pos, dirs = meridional_fan(np.linspace(-20, 20, 9), math.radians(5))
    hit = intersect(pos, dirs, 1 / R, zv)
    centre = np.array([0.0, 0.0, zv + R])
    assert np.linalg.norm(hit - centre, axis=1) == pytest.approx(np.full(9, R))


def test_normal_on_axis_points_along_z():
    n = surface_normal(np.array([[0.0, 0.0, 0.0]]), 1 / 50.0, 0.0)
    assert n[0] == pytest.approx([0.0, 0.0, 1.0])


def test_near_axis_ray_focuses_at_paraxial_bfl():
    sys_ = biconvex()
    pos, dirs = sys_.trace(*meridional_fan(np.array([1e-3])))
    bfl_real = axis_crossing(pos, dirs)[0] - sys_.last_vertex
    assert bfl_real == pytest.approx(sys_.to_paraxial().bfl, rel=1e-6)


def test_to_paraxial_matches_thick_lens():
    assert biconvex().to_paraxial().efl == pytest.approx(thick_lens(100, -100, 8, N_BK7).efl)


def test_directions_stay_unit_vectors():
    sys_ = biconvex()
    _, dirs = sys_.trace(*meridional_fan(np.linspace(-15, 15, 7), math.radians(3)))
    assert np.linalg.norm(dirs, axis=1) == pytest.approx(np.ones(7))


def test_ray_outside_clear_aperture_is_vignetted():
    sys_ = biconvex(semi=10.0)
    pos, _ = sys_.trace(*meridional_fan(np.array([5.0, 15.0])))
    assert not np.isnan(pos[0]).any()
    assert np.isnan(pos[1]).all()


def test_marginal_rays_focus_short_of_paraxial_focus():
    # Positive spherical aberration: edge rays cross the axis before the paraxial focus
    sys_ = biconvex()
    pos, dirs = sys_.trace(*meridional_fan(np.array([15.0])))
    assert axis_crossing(pos, dirs)[0] - sys_.last_vertex < sys_.to_paraxial().bfl


def test_planoconvex_curved_side_first_is_better():
    h = np.array([15.0])
    a = SequentialSystem([SphericalSurface(51.68, 6.0, N_BK7), SphericalSurface(math.inf, 0.0, 1.0)])
    b = SequentialSystem([SphericalSurface(math.inf, 6.0, N_BK7), SphericalSurface(-51.68, 0.0, 1.0)])
    lsa = []
    for s in (a, b):
        pos, dirs = s.trace(*meridional_fan(h))
        lsa.append(abs(axis_crossing(pos, dirs)[0] - s.last_vertex - s.to_paraxial().bfl))
    assert lsa[0] < lsa[1]