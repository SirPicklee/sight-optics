"""Exp09: reflex (red-dot) sight.

Window: 22 mm clear aperture, centred 15 mm above the LED axis (off-axis section), LED focus 75 mm.
1) Dot size: LED diameter -> dot size in MOA.
2) Bare spherical mirror: parallax at the paraxial focus and after re-focusing the LED.
3) Cemented Mangin window (N-BK7, dichroic interface): bend the eye-side surface to cancel parallax,
   with the outer surfaces chosen so the window has zero power for the scene.
4) What the shooter sees: dot movement on the target for a head movement across the window.
5) LED position tolerance: axial (focus -> parallax) and lateral (-> zero shift).
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    scene_directions,
    window_points,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
F, Y_WIN, R_WIN = 75.0, 15.0, 11.0
pts = window_points(Y_WIN, R_WIN, 21)

# ---------- 1) Dot size ----------
print(f"1) Dot size with an LED at f = {F:.0f} mm")
print(f"   {'dot [MOA]':>10}{'LED diameter [µm]':>20}{'dot at 100 m [cm]':>20}")
for moa in (1, 2, 3, 6):
    print(f"   {moa:>10}{moa * MOA * F * 1e3:>20.1f}{moa * MOA * 100 * 100:>20.2f}")

# ---------- 2) Spherical mirror ----------
mirror = ReflexWindow(-2 * F)
z_m = best_led_z(mirror, pts, metric=aiming_error)
print(f"\n2) Bare spherical mirror R = {mirror.R_B:.0f} mm, window {2*R_WIN:.0f} mm centred {Y_WIN:.0f} mm off axis")
print(f"   {'LED position':<28}{'z [mm]':>9}{'RMS [MOA]':>11}{'max [MOA]':>11}")
for name, z in (("paraxial focus (f = R/2)", -F), ("re-focused (best RMS)", z_m)):
    r, mx = aiming_error(mirror, z, pts)
    print(f"   {name:<28}{z:>9.2f}{r / MOA:>11.3f}{mx / MOA:>11.3f}")

# ---------- 3) Mangin window ----------
print("\n3) Cemented Mangin window, N-BK7 3 mm + 3 mm: bending the eye-side surface R_A")
print(f"   {'R_A':>8}{'R_B (mirror)':>14}{'R_C':>9}{'aim RMS [MOA]':>15}")
scan = np.linspace(-150, -55, 39)
scan_err = []
for ra in scan:
    w = mangin_for_focus(F, ra, N_BK7)
    e = aiming_error(w, best_led_z(w, pts, metric=aiming_error), pts)[0] / MOA
    scan_err.append(e)
for ra in (-150, -120, -100, -90, -70, -60):
    w = mangin_for_focus(F, ra, N_BK7)
    e = aiming_error(w, best_led_z(w, pts, metric=aiming_error), pts)[0] / MOA
    print(f"   {ra:>8.0f}{w.R_B:>14.1f}{w.R_C:>9.1f}{e:>15.3f}")
mangin = optimise_mangin(F, N_BK7, pts, (-100.0, -60.0))
z_w = best_led_z(mangin, pts, metric=aiming_error)
dot_only = parallax(mangin, z_w, pts)
aim = aiming_error(mangin, z_w, pts)
scene = scene_directions(mangin, pts)
print(f"   optimum: R_A = {mangin.R_A:.2f}, R_B = {mangin.R_B:.2f}, R_C = {mangin.R_C:.2f} mm, "
      f"scene power = {mangin.transmission_power:.1e} 1/mm, LED at {-z_w:.2f} mm")
print(f"   dot alone:          RMS {dot_only[0] / MOA:.3f}, max {dot_only[1] / MOA:.3f} MOA")
print(f"   window prism alone: spread {np.ptp(scene[:, 1]) / MOA:.3f} MOA across the window")
print(f"   dot vs target:      RMS {aim[0] / MOA:.3f}, max {aim[1] / MOA:.3f} MOA  <- what the shooter sees")

# ---------- 4) On the target ----------
print("\n4) Largest dot movement on the target for a head movement over the whole window [cm]")
print(f"   {'distance':>9}{'mirror':>9}{'Mangin':>9}")
for r in (25, 50, 100, 200):
    print(f"   {r:>7} m{aiming_error(mirror, z_m, pts)[1] * r * 100:>9.2f}{aim[1] * r * 100:>9.2f}")

# ---------- 5) LED tolerance ----------
print("\n5) LED position tolerance (Mangin window)")
dzs = np.linspace(-0.3, 0.3, 61)
tol_curve = np.array([aiming_error(mangin, z_w + dz, pts)[1] / MOA for dz in dzs])
ok = dzs[tol_curve <= 1.0]
print(f"   axial: max parallax <= 1 MOA for LED within {ok.min()*1e3:+.0f} / {ok.max()*1e3:+.0f} µm of best focus")
print(f"   lateral: 0.1 mm LED shift moves the dot by {0.1 / F / MOA:.1f} MOA at every eye position "
      "(a zero shift, removed by the windage/elevation adjustment, not parallax)")

# ---------- Figure ----------
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
for a_, w, z, title in ((ax[0, 0], mirror, z_m, "Bare spherical mirror"), (ax[0, 1], mangin, z_w, "Mangin window (optimised)")):
    e = dot_directions(w, z, pts) - scene_directions(w, pts)
    e = (e - e.mean(axis=0)) / MOA
    mag = np.linalg.norm(e, axis=1)
    sc = a_.scatter(pts[:, 0], pts[:, 1], c=mag, cmap="viridis", s=40, vmin=0)
    a_.quiver(pts[:, 0], pts[:, 1], e[:, 0], e[:, 1], angles="xy", color="w", width=0.003)
    a_.add_patch(plt.Circle((0, Y_WIN), R_WIN, fill=False, color="k"))
    a_.plot(0, 0, "r*", ms=12, label="LED axis")
    a_.set_aspect("equal")
    a_.set_xlim(-R_WIN - 3, R_WIN + 3)
    a_.set_ylim(-2, Y_WIN + R_WIN + 3)
    a_.set_title(f"{title}: dot vs target, max {mag.max():.2f} MOA")
    a_.set_xlabel("eye position x in the window [mm]")
    a_.set_ylabel("eye position y [mm]")
    a_.legend(fontsize=8, loc="lower right")
    fig.colorbar(sc, ax=a_, label="dot movement [MOA]")

ax[1, 0].semilogy(scan, scan_err, "o-", ms=3)
ax[1, 0].axhline(aiming_error(mirror, z_m, pts)[0] / MOA, color="tab:red", ls="--", label="bare spherical mirror")
ax[1, 0].axvline(mangin.R_A, color="green", ls=":", label=f"optimum R_A = {mangin.R_A:.1f} mm")
ax[1, 0].set_xlabel("eye-side surface radius R_A [mm]")
ax[1, 0].set_ylabel("RMS dot-vs-target error [MOA]")
ax[1, 0].set_title("Bending the Mangin window")
ax[1, 0].legend(fontsize=8)
ax[1, 0].grid(alpha=0.3, which="both")

ax[1, 1].plot(dzs * 1e3, tol_curve)
ax[1, 1].axhline(1.0, color="k", ls="--", lw=1, label="1 MOA")
ax[1, 1].set_xlabel("LED axial error from best focus [µm]")
ax[1, 1].set_ylabel("max dot-vs-target error [MOA]")
ax[1, 1].set_title("LED focus tolerance")
ax[1, 1].legend(fontsize=8)
ax[1, 1].grid(alpha=0.3)
fig.tight_layout()
out = RESULTS / "exp09_reflex.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")