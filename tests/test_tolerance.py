import math

import numpy as np
import pytest

from sightoptics import (
    N_BK7,
    N_F2,
    WL_D,
    OffsetGlass,
    Tolerance,
    achromat_design,
    doublet_tolerances,
    evaluate,
    monte_carlo,
    perturb,
    rss_prediction,
    sensitivity,
)
from sightoptics.glass import refractive_index

F, SEMI = 200.0, 20.0


@pytest.fixture(scope="module")
def lens():
    d = achromat_design(F, SEMI, N_BK7, N_F2, 0.01108)
    z = d.last_vertex + d.to_paraxial().bfl - 0.0645
    return d, z


def test_offset_glass_shifts_index():
    assert refractive_index(OffsetGlass(N_BK7, 0.001), WL_D) == pytest.approx(N_BK7.nd + 0.001, abs=1e-12)


def test_zero_perturbation_changes_nothing(lens):
    d, z = lens
    tols = doublet_tolerances(d, 0.002, 0.05, 0.0005, 0.02)
    a = evaluate(d, SEMI, z)
    b = evaluate(perturb(d, tols, np.zeros(len(tols))), SEMI, z)
    assert b["rms"] == pytest.approx(a["rms"], abs=1e-12)


def test_radius_tolerance_scales_radius(lens):
    d, _ = lens
    p = perturb(d, [Tolerance("radius", 0, 0.01)], np.array([0.01]))
    assert p.surfaces[0].R == pytest.approx(d.surfaces[0].R * 1.01)


def test_shifting_the_whole_lens_moves_the_line_of_sight_by_dy_over_f(lens):
    d, z = lens
    tols = [Tolerance("decenter", i, 0.05) for i in range(3)]
    e = evaluate(perturb(d, tols, np.full(3, 0.05)), SEMI, z)
    assert e["boresight"] == pytest.approx(0.05 / F, rel=0.02)


def test_refocus_removes_pure_defocus(lens):
    d, z = lens
    shifted = evaluate(d, SEMI, z + 0.3, refocus=True)
    fixed = evaluate(d, SEMI, z + 0.3, refocus=False)
    assert shifted["rms"] < 0.03
    assert fixed["rms"] > 0.3
    assert abs(shifted["focus"]) == pytest.approx(0.3, rel=0.05)


def test_monte_carlo_is_reproducible(lens):
    d, z = lens
    tols = doublet_tolerances(d, 0.002, 0.05, 0.0005, 0.02)
    a = monte_carlo(d, tols, SEMI, z, 20, seed=1)
    b = monte_carlo(d, tols, SEMI, z, 20, seed=1)
    assert a["rms"] == pytest.approx(b["rms"])


def test_rss_predicts_monte_carlo_boresight(lens):
    d, z = lens
    tols = doublet_tolerances(d, 0.002, 0.05, 0.0005, 0.02)
    sens = sensitivity(d, tols, SEMI, z)
    rss = rss_prediction(evaluate(d, SEMI, z), sens)
    mc = monte_carlo(d, tols, SEMI, z, 400, seed=2)
    assert math.sqrt(np.mean(mc["boresight"] ** 2)) == pytest.approx(rss["boresight"], rel=0.15)


def test_compensator_is_essential(lens):
    d, z = lens
    tols = doublet_tolerances(d, 0.002, 0.05, 0.0005, 0.02)
    with_comp = monte_carlo(d, tols, SEMI, z, 100, seed=3, refocus=True)
    without = monte_carlo(d, tols, SEMI, z, 100, seed=3, refocus=False)
    assert (with_comp["strehl"] >= 0.8).mean() > 0.9
    assert (without["strehl"] >= 0.8).mean() < 0.3