"""Exp04: chromatic aberration, lens bending and the cemented achromatic doublet.

Design target: 8x40 scope objective, f = 200 mm, D = 40 mm (F/5).
1) Glass data from the Sellmeier equation.
2) Plano-convex singlet: longitudinal colour vs the f/V rule.
3) Cemented N-BK7 / N-F2 doublet: thin-lens powers, then scan the bending (c1 = 1/R1)
   and find the bendings with minimum spherical aberration.
4) Spot diagrams singlet vs doublet, on axis and at the edge of the field (2.5 deg),
   compared with the Airy disk and with what the eye can resolve through an 8x eyepiece.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from sightoptics import (
    N_BK7,
    N_F2,
    VISIBLE,
    WL_C,
    WL_D,
    WL_F,
    achromat_design,
    achromat_powers,
    best_focus,
    chromatic_focal_shift,
    planoconvex,
    polychromatic_rms,
    rms_radius,
    scale_to_efl,
    spot,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
F, SEMI, FIELD, M = 200.0, 20.0, 2.5, 8.0
N_GRID = 21

airy_um = 1.22 * WL_D * F / (2 * SEMI)  # Airy radius [µm]
eye_um = math.radians(1 / 60) / M * F * 1000  # 1 arcmin eye limit, referred to the objective focal plane [µm]

# ---------- 1) Glass data ----------
print("1) Glass data (Sellmeier)")
print(f"   {'glass':<7}{'n_F':>9}{'n_d':>9}{'n_C':>9}{'V_d':>8}")
for g in (N_BK7, N_F2):
    print(f"   {g.name:<7}{float(g.n(WL_F)):>9.5f}{g.nd:>9.5f}{float(g.n(WL_C)):>9.5f}{g.vd:>8.2f}")

# ---------- 2) Singlet ----------
singlet = scale_to_efl(planoconvex(F, SEMI, N_BK7), F)
bfl_f, bfl_d, bfl_c = chromatic_focal_shift(singlet, [WL_F, WL_D, WL_C])
print("\n2) Plano-convex N-BK7 singlet, f = 200 mm, F/5")
print(f"   focus(C) - focus(F) = {bfl_c - bfl_f:.3f} mm   (rule of thumb f/V = {F / N_BK7.vd:.3f} mm)")

# ---------- 3) Doublet bending ----------
f1, f2 = achromat_powers(F, N_BK7, N_F2)
print("\n3) Cemented achromat N-BK7 + N-F2")
print(f"   thin-lens powers: f_crown = {f1:.2f} mm, f_flint = {f2:.2f} mm")


def mono_rms(c1):
    return best_focus(achromat_design(F, SEMI, N_BK7, N_F2, c1), SEMI, wavelengths=[WL_D], n=15)[1]


c1_grid = np.linspace(0.0, 0.016, 33)
rms_grid = np.array([mono_rms(c) for c in c1_grid])
minima = [i for i in range(1, len(c1_grid) - 1) if rms_grid[i] < rms_grid[i - 1] and rms_grid[i] < rms_grid[i + 1]]
solutions = []
for i in minima:
    res = minimize_scalar(mono_rms, bounds=(c1_grid[i - 1], c1_grid[i + 1]), method="bounded", options={"xatol": 1e-8})
    solutions.append(res.x)

print(f"   bending solutions with minimum spherical aberration (d-line):")
print(f"   {'#':>3}{'R1':>9}{'R2':>9}{'R3':>10}{'RMS d [µm]':>12}{'RMS FdC':>9}{'RMS 2.5deg':>11}")
designs = []
for k, c1 in enumerate(solutions, 1):
    d = achromat_design(F, SEMI, N_BK7, N_F2, c1)
    _, r_mono = best_focus(d, SEMI, wavelengths=[WL_D])
    _, r_poly = best_focus(d, SEMI)
    _, r_off = best_focus(d, SEMI, field_deg=FIELD)
    designs.append((r_off, d))
    R = [s.R for s in d.surfaces]
    print(f"   {k:>3}{R[0]:>9.2f}{R[1]:>9.2f}{R[2]:>10.1f}{r_mono*1e3:>12.2f}{r_poly*1e3:>9.2f}{r_off*1e3:>11.2f}")

doublet = min(designs, key=lambda x: x[0])[1]
cf = chromatic_focal_shift(doublet, [WL_F, WL_D, WL_C])
print(f"   chosen design: smallest off-axis spot. F/C focus equal; d-line focus off by {(cf[1]-cf[0])*1e3:.1f} µm (secondary spectrum)")

# ---------- 4) Spot diagrams ----------
print(f"\n4) Spot sizes on a FLAT reticle plane (best on-axis polychromatic focus)")
print(f"   Airy radius = {airy_um:.2f} µm, eye limit through {M:.0f}x eyepiece = {eye_um:.2f} µm")
print(f"   {'system':<9}{'field':>7}{'RMS F':>8}{'RMS d':>8}{'RMS C':>8}{'RMS all':>9}  [µm]")
spots = {}
for name, s in (("singlet", singlet), ("doublet", doublet)):
    z_img, _ = best_focus(s, SEMI)
    for fld in (0.0, FIELD):
        sp = spot(s, SEMI, z_img, fld, VISIBLE, N_GRID)
        spots[(name, fld)] = sp
        r = [rms_radius(sp[wl]) * 1e3 for wl in VISIBLE]
        print(f"   {name:<9}{fld:>6.1f}°{r[0]:>8.2f}{r[1]:>8.2f}{r[2]:>8.2f}{polychromatic_rms(sp)*1e3:>9.2f}")

# ---------- Figure 1: spot diagrams ----------
colors = {WL_F: "tab:blue", WL_D: "goldenrod", WL_C: "tab:red"}
fig, ax = plt.subplots(2, 2, figsize=(9, 9))
for r, name in enumerate(("singlet", "doublet")):
    for c, fld in enumerate((0.0, FIELD)):
        a = ax[r, c]
        sp = spots[(name, fld)]
        centre = np.vstack(list(sp.values())).mean(axis=0)
        for wl, pts in sp.items():
            p = (pts - centre) * 1e3
            a.scatter(p[:, 0], p[:, 1], s=3, color=colors[wl], label=f"{wl*1000:.0f} nm")
        for rad, ls, lab in ((airy_um, "-", "Airy disk"), (eye_um, "--", "eye limit (8x)")):
            a.add_patch(plt.Circle((0, 0), rad, fill=False, color="k", ls=ls, lw=1, label=lab))
        lim = max(1.15 * np.abs(np.vstack([(p - centre) * 1e3 for p in sp.values()])).max(), 1.5 * eye_um)
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_aspect("equal")
        a.set_title(f"{name}, field {fld:.1f}°   RMS = {polychromatic_rms(sp)*1e3:.1f} µm")
        a.set_xlabel("x [µm]")
        a.set_ylabel("y [µm]")
        a.grid(alpha=0.3)
ax[0, 0].legend(fontsize=7, loc="upper right")
fig.suptitle("Spot diagrams, f = 200 mm F/5 objective (note different scales)")
fig.tight_layout()
out1 = RESULTS / "exp04_spots.png"
fig.savefig(out1, dpi=150)

# ---------- Figure 2: bending + chromatic focal shift ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
ax[0].semilogy(c1_grid, rms_grid * 1e3, "o-", ms=3)
for c1 in solutions:
    ax[0].axvline(c1, color="green", ls=":", lw=1)
ax[0].axhline(airy_um, color="k", lw=1, label="Airy radius")
ax[0].set_xlabel("front curvature c1 = 1/R1 [1/mm]")
ax[0].set_ylabel("RMS spot, d-line, best focus [µm]")
ax[0].set_title("Lens bending: same power, same colour correction")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3, which="both")

wl = np.linspace(0.45, 0.70, 60)
for s, name in ((singlet, "singlet"), (doublet, "achromatic doublet")):
    shift = chromatic_focal_shift(s, wl) - chromatic_focal_shift(s, [WL_D])[0]
    ax[1].plot(wl * 1000, shift * 1000, label=name)
for w in VISIBLE:
    ax[1].axvline(w * 1000, color=colors[w], ls=":", lw=1)
ax[1].set_xlabel("wavelength [nm]")
ax[1].set_ylabel("focus shift vs d-line [µm]")
ax[1].set_title("Chromatic focal shift")
ax[1].set_ylim(-400, 1200)
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)
inset = ax[1].inset_axes([0.45, 0.5, 0.5, 0.42])
inset.plot(wl * 1000, (chromatic_focal_shift(doublet, wl) - cf[1]) * 1000, color="tab:orange")
inset.set_title("doublet zoom", fontsize=7)
inset.tick_params(labelsize=6)
inset.grid(alpha=0.3)
fig.tight_layout()
out2 = RESULTS / "exp04_achromat.png"
fig.savefig(out2, dpi=150)
print(f"\nSaved {out1}\nSaved {out2}")