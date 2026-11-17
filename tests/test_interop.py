"""Cross-check against rayoptics. Skipped automatically if rayoptics is not installed."""

import numpy as np
import pytest

pytest.importorskip("rayoptics")

from sightoptics import N_BK7, N_F2, WL_C, WL_D, WL_F, achromat_design, chromatic_focal_shift, pupil_grid, seidel, trace_to_plane, collimated_bundle  # noqa: E402
from sightoptics.interop import bfl_at, first_order, intercepts, third_order, to_rayoptics  # noqa: E402

F, SEMI, FIELD, IMG = 200.0, 20.0, 2.5, 194.0
WL_NM = {WL_F: 486.1327, WL_D: 587.5618, WL_C: 656.2725}


@pytest.fixture(scope="module")
def models():
    d = achromat_design(F, SEMI, N_BK7, N_F2, 0.011)
    return d, to_rayoptics(d, 2 * SEMI, FIELD, IMG)


def test_efl_and_bfl_match(models):
    d, opm = models
    fo = first_order(opm)
    assert fo["efl"] == pytest.approx(d.to_paraxial().efl, rel=1e-9)
    assert fo["bfl"] == pytest.approx(d.to_paraxial().bfl, rel=1e-9)


def test_focus_at_each_wavelength_matches(models):
    d, opm = models
    ours = chromatic_focal_shift(d, [WL_F, WL_D, WL_C])
    theirs = [bfl_at(opm, WL_NM[w]) for w in (WL_F, WL_D, WL_C)]
    assert ours == pytest.approx(theirs, abs=1e-4)


def test_seidel_sums_match(models):
    d, opm = models
    assert seidel(d, SEMI, FIELD).totals[:5] == pytest.approx(third_order(opm), rel=1e-5)


@pytest.mark.parametrize("field_index, field_deg", [(0, 0.0), (1, FIELD)])
def test_real_ray_intercepts_match(models, field_index, field_deg):
    d, opm = models
    xy = pupil_grid(0.95 * SEMI, 7)
    pos, dirs = collimated_bundle(0.95 * SEMI, field_deg, 7)
    z_img = d.last_vertex + IMG
    for wl, nm in WL_NM.items():
        ours = trace_to_plane(d, pos, dirs, z_img, wl)
        theirs = intercepts(opm, xy / SEMI, field_index, nm)
        assert np.max(np.abs(ours - theirs)) < 1e-5  # 10 nm