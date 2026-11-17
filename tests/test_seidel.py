import pytest

from sightoptics import (
    N_BK7,
    N_F2,
    SequentialSystem,
    SphericalSurface,
    achromat_design,
    measure_real,
    planoconvex,
    scale_to_efl,
    seidel,
)

F, SEMI = 200.0, 20.0
SMALL_SEMI, SMALL_FIELD = 2.0, 0.5  # third-order regime


@pytest.fixture(scope="module")
def doublet():
    return achromat_design(F, SEMI, N_BK7, N_F2, 0.011)


@pytest.fixture(scope="module")
def compare(doublet):
    return seidel(doublet, SMALL_SEMI, SMALL_FIELD).transverse(), measure_real(doublet, SMALL_SEMI, SMALL_FIELD)


@pytest.mark.parametrize("key", [
    "LSA (marginal focus - paraxial focus)",
    "tangential coma (transverse)",
    "sagittal field curvature (longitudinal)",
    "tangential field curvature (longitudinal)",
    "distortion (transverse)",
    "lateral colour F-C (transverse)",
])
def test_seidel_matches_real_rays_in_third_order_regime(compare, key):
    predicted, real = compare
    assert predicted[key] == pytest.approx(real[key], rel=0.02)


def test_axial_colour_of_singlet_matches_paraxial():
    s = scale_to_efl(planoconvex(F, SEMI, N_BK7), F)
    key = "axial colour F-C (longitudinal)"
    assert seidel(s, SEMI, 0.0).transverse()[key] == pytest.approx(measure_real(s, SEMI, 0.0)[key], rel=0.02)


def test_spherical_aberration_scales_with_fourth_power_of_aperture(doublet):
    s1 = seidel(doublet, 5.0, 0.0).totals[0]
    s2 = seidel(doublet, 10.0, 0.0).totals[0]
    assert s2 / s1 == pytest.approx(16.0)


def test_coma_is_linear_in_field(doublet):
    c1 = seidel(doublet, SEMI, 1.0).totals[1]
    c2 = seidel(doublet, SEMI, 2.0).totals[1]
    assert c2 / c1 == pytest.approx(2.0, rel=1e-3)  # tan(2°)/tan(1°) ≈ 2


def test_petzval_of_thin_lens_is_h2_phi_over_n():
    n = 1.5168
    s = SequentialSystem([SphericalSurface(100.0, 1e-3, n), SphericalSurface(-100.0, 0.0, 1.0)])
    r = seidel(s, 10.0, 2.0)
    phi = 1.0 / s.to_paraxial().efl
    assert r.totals[3] == pytest.approx(r.lagrange**2 * phi / n, rel=1e-4)


def test_bending_does_not_change_petzval_or_astigmatism():
    a = seidel(achromat_design(F, SEMI, N_BK7, N_F2, 0.004), SEMI, 2.5).totals
    b = seidel(achromat_design(F, SEMI, N_BK7, N_F2, 0.011), SEMI, 2.5).totals
    assert a[3] == pytest.approx(b[3], rel=0.02)  # Petzval
    assert a[2] == pytest.approx(b[2], rel=0.15)  # astigmatism (thin-lens: H² φ)
    assert a[1] * b[1] < 0  # coma changes sign between the two bendings