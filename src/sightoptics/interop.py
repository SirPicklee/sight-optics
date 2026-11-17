"""Bridge to rayoptics (https://github.com/mjhoptics/ray-optics) for independent validation.

rayoptics is an optional dependency: pip install rayoptics
Only Glass objects with Schott catalogue names and air (n = 1) are supported.
"""

from __future__ import annotations

import math

import numpy as np

from .glass import Glass
from .raytrace import SequentialSystem


def to_rayoptics(system: SequentialSystem, epd: float, field_deg: float, image_distance: float):
    """Build a rayoptics OpticalModel of the same prescription.

    Object at infinity, stop at the first surface, fields [0, field_deg], wavelengths F, d, C.
    image_distance: distance from the last surface to the image plane [mm].
    """
    from rayoptics.environment import FieldSpec, OpticalModel, PupilSpec, WvlSpec

    opm = OpticalModel()
    sm, osp = opm["seq_model"], opm["optical_spec"]
    osp["pupil"] = PupilSpec(osp, key=["object", "epd"], value=epd)
    osp["fov"] = FieldSpec(osp, key=["object", "angle"], value=field_deg, flds=[0.0, 1.0], is_relative=True)
    osp["wvls"] = WvlSpec([("F", 1.0), ("d", 1.0), ("C", 1.0)], ref_wl=1)
    opm.radius_mode = True
    sm.gaps[0].thi = 1e10  # object "at infinity"

    last = len(system.surfaces) - 1
    for i, s in enumerate(system.surfaces):
        R = 0.0 if math.isinf(s.R) else s.R  # rayoptics: radius 0 = flat
        thi = image_distance if i == last else s.thickness
        if isinstance(s.n, Glass):
            sm.add_surface([R, thi, s.n.name, "Schott"], sd=s.semi_diameter)
        elif float(s.n) == 1.0:
            sm.add_surface([R, thi], sd=s.semi_diameter)
        else:
            raise ValueError("only Schott Glass objects or air are supported")
        if i == 0:
            sm.set_stop()
    opm.update_model()
    return opm


def first_order(opm) -> dict[str, float]:
    fod = opm["analysis_results"]["parax_data"].fod
    return {"efl": float(fod.efl), "bfl": float(fod.bfl), "fno": float(fod.fno)}


def bfl_at(opm, wavelength_nm: float, rel_height: float = 1e-4) -> float:
    """Back focal length at any wavelength, from a near-axis real ray traced by rayoptics.

    (first_order() is only used at the reference wavelength.)
    """
    from rayoptics.raytr import trace

    fld = opm["optical_spec"]["fov"].fields[0]
    ray, _, _ = trace.trace_base(opm, [0.0, rel_height], fld, wavelength_nm)
    p, d = ray[-2][0], ray[-2][1]  # point on, and direction after, the last lens surface
    return float(p[2] - p[1] * d[2] / d[1])


def third_order(opm) -> np.ndarray:
    """Seidel sums S_I .. S_V."""
    from rayoptics.environment import compute_third_order

    to = compute_third_order(opm)
    return to.loc["sum"].to_numpy(dtype=float)[:5]


def intercepts(opm, pupil_rel: np.ndarray, field_index: int, wavelength_nm: float) -> np.ndarray:
    """Real-ray (x, y) intercepts on the image surface for relative pupil coordinates (N, 2)."""
    from rayoptics.raytr import trace

    fld = opm["optical_spec"]["fov"].fields[field_index]
    out = []
    for px, py in pupil_rel:
        ray, _, _ = trace.trace_base(opm, [float(px), float(py)], fld, wavelength_nm)
        out.append(ray[-1][0][:2])
    return np.array(out)