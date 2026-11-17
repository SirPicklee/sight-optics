"""Exp03: own real ray tracer vs paraxial theory.

1) Biconvex N-BK7 lens: real ray focus vs paraxial BFL as ray height grows
   -> longitudinal spherical aberration (LSA), check LSA ~ h^2.
2) Plano-convex lens, same glass and focal length, two orientations:
   curved side to the collimated beam vs flat side to the beam.
3) Ray diagrams of both orientations + LSA curves.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sightoptics import SequentialSystem, SphericalSurface, axis_crossing, meridional_fan, sag

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
N_BK7 = 1.5168
SEMI = 20.0


def lsa_curve(system, heights):
    pos, dirs = system.trace(*meridional_fan(heights))
    bfl_real = axis_crossing(pos, dirs) - system.last_vertex
    return bfl_real, bfl_real - system.to_paraxial().bfl


# ---------- 1) Biconvex lens ----------
biconvex = SequentialSystem([SphericalSurface(100.0, 8.0, N_BK7, SEMI), SphericalSurface(-100.0, 0.0, 1.0, SEMI)])
par = biconvex.to_paraxial()
heights = np.array([0.01, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0])
bfl_real, lsa = lsa_curve(biconvex, heights)

print("1) Biconvex N-BK7 (R = +/-100, t = 8): paraxial EFL = %.3f mm, BFL = %.3f mm" % (par.efl, par.bfl))
print(f"   {'h [mm]':>7}{'real BFL':>11}{'LSA [mm]':>10}{'LSA/h^2':>10}")
for h, b, l in zip(heights, bfl_real, lsa):
    print(f"   {h:>7.2f}{b:>11.4f}{l:>10.4f}{l / h**2:>10.5f}")

# ---------- 2) Plano-convex orientations ----------
R_pc = 51.68  # gives f ~ 100 mm with N-BK7
curved_first = SequentialSystem([SphericalSurface(R_pc, 6.0, N_BK7, SEMI), SphericalSurface(math.inf, 0.0, 1.0, SEMI)])
flat_first = SequentialSystem([SphericalSurface(math.inf, 6.0, N_BK7, SEMI), SphericalSurface(-R_pc, 0.0, 1.0, SEMI)])

print("\n2) Plano-convex N-BK7, f = %.2f mm, ray height h = 15 mm" % curved_first.to_paraxial().efl)
print(f"   {'orientation':<16}{'EFL':>8}{'BFL':>9}{'LSA@15mm':>10}")
for name, s in (("curved first", curved_first), ("flat first", flat_first)):
    _, l = lsa_curve(s, np.array([15.0]))
    p = s.to_paraxial()
    print(f"   {name:<16}{p.efl:>8.2f}{p.bfl:>9.2f}{l[0]:>10.3f}")

# ---------- 3) Figure ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
fan = np.linspace(-SEMI * 0.95, SEMI * 0.95, 15)


def draw_lens(a, system):
    r = np.linspace(-SEMI, SEMI, 100)
    z1 = system.vertex_z[0] + sag(system.surfaces[0].c, r)
    z2 = system.vertex_z[1] + sag(system.surfaces[1].c, r)
    a.fill_betweenx(r, z1, z2, color="lightsteelblue", alpha=0.7)


for a, s, title in ((ax[0], curved_first, "Plano-convex, curved side first"), (ax[1], flat_first, "Plano-convex, flat side first")):
    pos, dirs, path = s.trace(*meridional_fan(fan), keep_path=True)
    z_end = s.last_vertex + 110.0
    t = (z_end - pos[:, 2]) / dirs[:, 2]
    end = pos + t[:, None] * dirs
    for i in range(fan.size):
        zs = [p[i, 2] for p in path] + [end[i, 2]]
        ys = [p[i, 1] for p in path] + [end[i, 1]]
        a.plot(zs, ys, color="tab:red", lw=0.7)
    draw_lens(a, s)
    f_par = s.last_vertex + s.to_paraxial().bfl
    a.axvline(f_par, color="k", ls=":", lw=1, label="paraxial focus")
    a.set_xlim(-20, z_end)
    a.set_ylim(-SEMI - 2, SEMI + 2)
    a.set_title(title)
    a.set_xlabel("z [mm]")
    a.set_ylabel("y [mm]")
    a.legend(fontsize=8, loc="lower left")
    a.grid(alpha=0.3)
    # zoom inset near focus
    ins = a.inset_axes([0.62, 0.62, 0.36, 0.34])
    for i in range(fan.size):
        ins.plot([pos[i, 2], end[i, 2]], [pos[i, 1], end[i, 1]], color="tab:red", lw=0.6)
    ins.axvline(f_par, color="k", ls=":", lw=1)
    ins.set_xlim(f_par - 8, f_par + 2)
    ins.set_ylim(-1.5, 1.5)
    ins.tick_params(labelsize=6)

h = np.linspace(0.5, SEMI * 0.95, 60)
for s, name in ((curved_first, "curved first"), (flat_first, "flat first"), (biconvex, "biconvex R=+/-100")):
    _, l = lsa_curve(s, h)
    ax[2].plot(h, l, label=name)
ax[2].set_xlabel("ray height h [mm]")
ax[2].set_ylabel("LSA = real focus - paraxial focus [mm]")
ax[2].set_title("Longitudinal spherical aberration")
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.3)

fig.tight_layout()
out = RESULTS / "exp03_real_raytrace.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")