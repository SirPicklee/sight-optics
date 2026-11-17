"""Exp06: independent validation of sightoptics against rayoptics.

The same cemented doublet (8x40 objective, f = 200 mm, F/5) is built in both codes.
1) First-order data: EFL, BFL at F/d/C, F-number.
2) Seidel sums S_I .. S_V.
3) Real rays: image-plane intercepts and RMS spot for 2 fields x 3 wavelengths.
4) Speed: vectorised numpy tracing vs ray-by-ray.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sightoptics import (
    N_BK7,
    N_F2,
    VISIBLE,
    WL_C,
    WL_D,
    WL_F,
    achromat_design,
    bending_solutions,
    best_focus,
    chromatic_focal_shift,
    collimated_bundle,
    pupil_grid,
    rms_radius,
    seidel,
    trace_to_plane,
)
from sightoptics.interop import bfl_at, first_order, intercepts, third_order, to_rayoptics

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
F, SEMI, FIELD, N_GRID = 200.0, 20.0, 2.5, 21
PUPIL = 0.95 * SEMI  # stay inside the clear aperture at the field edge
WL_NM = {WL_F: 486.1327, WL_D: 587.5618, WL_C: 656.2725}
NAME = {WL_F: "F", WL_D: "d", WL_C: "C"}

doublet = achromat_design(F, SEMI, N_BK7, N_F2, bending_solutions(F, SEMI, N_BK7, N_F2)[-1])
z_img, _ = best_focus(doublet, SEMI)
opm = to_rayoptics(doublet, 2 * SEMI, FIELD, z_img - doublet.last_vertex)
diffs = {}


def pct(a, b):
    return 100 * abs(a - b) / abs(b)


# ---------- 1) First order ----------
print("1) First-order data")
print(f"   {'quantity':<12}{'sightoptics':>14}{'rayoptics':>14}{'diff %':>11}")
fo = first_order(opm)
par = doublet.to_paraxial()
rows = [("EFL", par.efl, fo["efl"]), ("F-number", par.efl / (2 * SEMI), fo["fno"])]
for wl in VISIBLE:
    rows.append((f"BFL {NAME[wl]}", chromatic_focal_shift(doublet, [wl])[0], bfl_at(opm, WL_NM[wl])))
for name, a, b in rows:
    diffs[name] = pct(a, b)
    print(f"   {name:<12}{a:>14.6f}{b:>14.6f}{diffs[name]:>11.2e}")

# ---------- 2) Seidel ----------
print("\n2) Seidel sums [mm]")
print(f"   {'sum':<12}{'sightoptics':>14}{'rayoptics':>14}{'diff %':>11}")
ours_s = seidel(doublet, SEMI, FIELD).totals[:5]
theirs_s = third_order(opm)
for name, a, b in zip(["S_I", "S_II", "S_III", "S_IV", "S_V"], ours_s, theirs_s):
    diffs[name] = pct(a, b)
    print(f"   {name:<12}{a:>14.6e}{b:>14.6e}{diffs[name]:>11.2e}")

# ---------- 3) Real rays ----------
print(f"\n3) Real rays on the image plane ({N_GRID}x{N_GRID} pupil grid, pupil radius {PUPIL} mm)")
print(f"   {'field':>6}{'wl':>4}{'rays':>6}{'max |Δ| [nm]':>14}{'RMS ours [µm]':>15}{'RMS rayopt [µm]':>17}{'diff %':>10}")
xy_rel = pupil_grid(PUPIL, N_GRID) / SEMI
spots = {}
for fi, fld in enumerate((0.0, FIELD)):
    pos, dirs = collimated_bundle(PUPIL, fld, N_GRID)
    for wl in VISIBLE:
        ours = trace_to_plane(doublet, pos, dirs, z_img, wl)
        theirs = intercepts(opm, xy_rel, fi, WL_NM[wl])
        spots[(fld, wl)] = (ours, theirs)
        r1, r2 = rms_radius(ours), rms_radius(theirs)
        key = f"RMS {fld:.1f}° {NAME[wl]}"
        diffs[key] = pct(r1, r2)
        dmax = np.max(np.abs(ours - theirs)) * 1e6
        print(f"   {fld:>5.1f}°{NAME[wl]:>4}{len(ours):>6}{dmax:>14.3f}{r1*1e3:>15.4f}{r2*1e3:>17.4f}{diffs[key]:>10.2e}")

# ---------- 4) Speed ----------
pos, dirs = collimated_bundle(PUPIL, FIELD, N_GRID)
t0 = time.perf_counter()
for _ in range(20):
    trace_to_plane(doublet, pos, dirs, z_img, WL_D)
t_ours = (time.perf_counter() - t0) / 20
t0 = time.perf_counter()
intercepts(opm, xy_rel, 1, WL_NM[WL_D])
t_theirs = time.perf_counter() - t0
n = len(xy_rel)
print(f"\n4) Speed for {n} rays: sightoptics {t_ours*1e3:.2f} ms, rayoptics (ray by ray) {t_theirs*1e3:.1f} ms"
      f"  -> {t_theirs / t_ours:.0f}x")
print("   (rayoptics also computes optical path and keeps the full ray history, so this is not a like-for-like benchmark)")

worst = max(diffs, key=diffs.get)
print(f"\nLargest relative difference: {worst} = {diffs[worst]:.2e} %")

# ---------- Figure ----------
colors = {WL_F: "tab:blue", WL_D: "goldenrod", WL_C: "tab:red"}
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
for wl in VISIBLE:
    ours, theirs = spots[(FIELD, wl)]
    c = np.vstack([s[0] for k, s in spots.items() if k[0] == FIELD]).mean(axis=0)
    ax[0].scatter((ours[:, 0] - c[0]) * 1e3, (ours[:, 1] - c[1]) * 1e3, s=14, facecolors="none",
                  edgecolors=colors[wl], label=f"sightoptics {NAME[wl]}")
    ax[0].scatter((theirs[:, 0] - c[0]) * 1e3, (theirs[:, 1] - c[1]) * 1e3, s=6, marker="x", color="k",
                  lw=0.5, label="rayoptics" if wl == WL_C else None)
ax[0].set_aspect("equal")
ax[0].set_xlabel("x [µm]")
ax[0].set_ylabel("y [µm]")
ax[0].set_title(f"Spot at {FIELD}°: circles = sightoptics, x = rayoptics")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)

names = list(diffs)
vals = np.array([max(diffs[k], 1e-12) for k in names])
ax[1].barh(names, vals, color="tab:green")
ax[1].set_xscale("log")
ax[1].axvline(1e-2, color="k", ls="--", lw=1, label="0.01 %")
ax[1].set_xlabel("relative difference [%] (log scale)")
ax[1].set_title("sightoptics vs rayoptics")
ax[1].invert_yaxis()
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, axis="x", which="both")
fig.tight_layout()
out = RESULTS / "exp06_rayoptics.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")