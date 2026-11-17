"""Exp08: 3-9x40 variable-magnification rifle scope (first-order design).

1) Trade study: eyepiece and erector focal lengths, erector length -> eye relief and tube length.
   Pick the shortest design that keeps >= 90 mm eye relief at every magnification.
2) The chosen design across the zoom: cam positions, eye relief, exit pupil, field of view.
3) FFP vs SFP reticle: what a "1 mrad" mark really means at each magnification.
4) Parallax: reticle-on-target error vs target distance.
5) Turret click values at common distances.
6) Clear apertures the lenses need (what has to fit in the tube).
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sightoptics import ZoomScope

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
D, M_MIN, M_MAX, AFOV_DEG, ER_MIN = 40.0, 3.0, 9.0, 22.0, 90.0


def make(fe, fr, L):
    fs = 2 * fe * math.tan(math.radians(AFOV_DEG / 2))
    return ZoomScope(f_objective=M_MIN * fe, d_objective=D, f_erector=fr, erector_length=L, f_eyepiece=fe, field_stop=fs)


# ---------- 1) Trade study ----------
print(f"1) Trade study (3-9x{D:.0f}, apparent field {AFOV_DEG}°), requirement: eye relief >= {ER_MIN} mm")
print(f"   {'f_eye':>6}{'f_erect':>8}{'L':>6}{'f_obj':>7}{'length':>8}{'ER@3x':>8}{'ER@9x':>8}  ok")
trade = []
for fe in (40, 45, 50, 55):
    for fr in (20, 25, 30, 35):
        for L in (70, 80, 100, 120):
            try:
                s = make(fe, fr, L)
            except ValueError:
                continue
            er = [s.eye_relief(M) for M in np.linspace(M_MIN, M_MAX, 13)]
            ok = min(er) >= ER_MIN
            trade.append((fe, fr, L, s, min(er), max(er), ok))
            print(f"   {fe:>6}{fr:>8}{L:>6}{s.f_objective:>7.0f}{s.length:>8.0f}{er[0]:>8.1f}{er[-1]:>8.1f}  {'yes' if ok else ''}")
fe, fr, L, scope, *_ = min((t for t in trade if t[6]), key=lambda t: (t[3].length, t[5] - t[4]))
print(f"   -> chosen: f_eye = {fe}, f_erector = {fr}, L = {L}, f_objective = {scope.f_objective:.0f}, "
      f"optical length = {scope.length:.0f} mm")

# ---------- 2) Across the zoom ----------
Ms = np.linspace(M_MIN, M_MAX, 61)
print("\n2) Chosen design across the zoom range")
print(f"   {'M':>5}{'m_erect':>9}{'lens1 [mm]':>11}{'lens2 [mm]':>11}{'ER [mm]':>9}{'EP [mm]':>9}{'FOV m@100m':>12}{'TF':>6}")
for M in (3, 4, 5, 6, 7, 8, 9):
    a, t, b = scope.erector_positions(M)
    print(f"   {M:>5.1f}{scope.erector_magnification(M):>9.3f}{a:>11.2f}{a + t:>11.2f}{scope.eye_relief(M):>9.1f}"
          f"{scope.exit_pupil(M):>9.2f}{scope.fov_at_100m(M):>12.2f}{math.sqrt(M * D):>6.1f}")
print(f"   apparent field = {math.degrees(scope.apparent_fov):.1f}° (constant, field stop at the SFP)")

# ---------- 3) FFP vs SFP ----------
mrad = 1e-3
s_ffp = mrad * scope.f_objective
s_sfp = scope.sfp_spacing_for(mrad, calibrated_M=M_MAX)
print(f"\n3) Reticle with '1 mrad' marks: FFP spacing {s_ffp*1e3:.0f} µm, SFP spacing {s_sfp*1e3:.0f} µm (calibrated at {M_MAX:.0f}x)")
print(f"   {'M':>5}{'FFP mark [mrad]':>17}{'SFP mark [mrad]':>17}")
for M in (3, 4.5, 6, 9):
    print(f"   {M:>5.1f}{scope.reticle_subtension(M, s_ffp, 'FFP')*1e3:>17.3f}{scope.reticle_subtension(M, s_sfp, 'SFP')*1e3:>17.3f}")
R, hold = 400.0, 3.0  # target distance [m], holdover needed [mrad]
for M in (4.5, 6.0):
    used = hold * scope.reticle_subtension(M, s_sfp, "SFP") * 1e3
    print(f"   SFP at {M:.1f}x: holding the 3rd mark gives {used:.1f} mrad instead of {hold:.0f} -> "
          f"miss by {(used - hold) * R / 1e3 * 100:.0f} cm at {R:.0f} m")

# ---------- 4) Parallax ----------
R0 = 100.0
dist = np.linspace(25, 600, 200)
h = D / 2  # eye at the edge of the exit pupil == edge of the entrance pupil
err = np.array([abs(scope.parallax_error(r, R0, h)) for r in dist])
print(f"\n4) Parallax, reticle parallax-free at {R0:.0f} m, eye at the edge of the exit pupil")
print(f"   {'target [m]':>11}{'error at target [mm]':>22}{'[mrad]':>8}")
for r in (25, 50, 100, 200, 400, 600):
    e = abs(scope.parallax_error(r, R0, h))
    print(f"   {r:>11.0f}{e:>22.1f}{e / r:>8.3f}")
print("   (independent of magnification: the exit pupil shrinks as D/M, so the largest eye offset maps to D/2)")

# ---------- 5) Clicks ----------
print("\n5) Turret click values [cm at the target]")
print(f"   {'click':<12}" + "".join(f"{r:>8.0f} m" for r in (100, 200, 300, 500)))
for name, rad in (("0.1 mrad", 1e-4), ("1/4 MOA", math.radians(1 / 60) / 4)):
    print(f"   {name:<12}" + "".join(f"{rad * r * 100:>10.2f}" for r in (100, 200, 300, 500)))

# ---------- 6) Clear apertures ----------
print("\n6) Required clear semi-diameters [mm] (largest ray height of the full bundle, on axis and at the field edge)")
names = ("objective", "erector 1", "erector 2", "eyepiece")
print(f"   {'M':>5}" + "".join(f"{n:>12}" for n in names))
for M in (3, 6, 9):
    sys_ = scope.system(M)
    half = scope.real_fov(M) / 2
    idx = [i + 1 for i, e in enumerate(sys_.elements) if type(e).__name__ == "ThinLens"]
    hmax = np.zeros(len(idx))
    for y0 in np.linspace(-D / 2, D / 2, 21):
        for u0 in (0.0, half, -half):
            _, ys, _ = sys_.trace(y0, u0)
            hmax = np.maximum(hmax, np.abs(ys[idx]))
    print(f"   {M:>5.1f}" + "".join(f"{h:>12.1f}" for h in hmax))
eyepiece_semi = scope.eye_relief(M_MIN) * math.tan(scope.apparent_fov / 2) + scope.exit_pupil(M_MIN) / 2
print(f"   eyepiece rule: ER x tan(AFOV/2) + EP/2 = {eyepiece_semi:.1f} mm at {M_MIN:.0f}x"
      " -> long eye relief and a wide apparent field cannot both fit a ~40 mm eyepiece")

# ---------- Figure 1: design curves ----------
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
pos = np.array([scope.erector_positions(M) for M in Ms])
ax[0, 0].plot(Ms, pos[:, 0], label="erector lens 1 (from FFP)")
ax[0, 0].plot(Ms, pos[:, 0] + pos[:, 1], label="erector lens 2 (from FFP)")
ax[0, 0].axhline(0, color="k", lw=1)
ax[0, 0].axhline(L, color="k", lw=1)
ax[0, 0].text(M_MIN, 1.5, "FFP", fontsize=8)
ax[0, 0].text(M_MIN, L - 4, "SFP", fontsize=8)
ax[0, 0].set_xlabel("magnification M")
ax[0, 0].set_ylabel("position [mm]")
ax[0, 0].set_title("Cam curves of the erector")
ax[0, 0].legend(fontsize=8)
ax[0, 0].grid(alpha=0.3)

er = [scope.eye_relief(M) for M in Ms]
ax[0, 1].plot(Ms, er, "tab:blue", label="eye relief")
ax[0, 1].axhline(ER_MIN, color="tab:blue", ls="--", lw=1, label=f"requirement {ER_MIN:.0f} mm")
ax[0, 1].set_xlabel("magnification M")
ax[0, 1].set_ylabel("eye relief [mm]", color="tab:blue")
a2 = ax[0, 1].twinx()
a2.plot(Ms, [scope.exit_pupil(M) for M in Ms], "tab:orange", label="exit pupil")
a2.axhspan(5, 7, color="navy", alpha=0.1)
a2.axhspan(2, 3, color="gold", alpha=0.2)
a2.set_ylabel("exit pupil [mm] (bands: eye pupil night / day)", color="tab:orange")
ax[0, 1].set_title("Eye relief and exit pupil")
ax[0, 1].legend(fontsize=8, loc="upper right")
ax[0, 1].grid(alpha=0.3)

ax[1, 0].plot(Ms, [scope.reticle_subtension(M, s_ffp, "FFP") * 1e3 for M in Ms], label="FFP reticle")
ax[1, 0].plot(Ms, [scope.reticle_subtension(M, s_sfp, "SFP") * 1e3 for M in Ms], label=f"SFP reticle (calibrated at {M_MAX:.0f}x)")
ax[1, 0].set_xlabel("magnification M")
ax[1, 0].set_ylabel("value of one '1 mrad' mark on the target [mrad]")
ax[1, 0].set_title("FFP vs SFP reticle")
ax[1, 0].legend(fontsize=8)
ax[1, 0].grid(alpha=0.3)

ax[1, 1].plot(dist, err / 10, label=f"parallax-free at {R0:.0f} m")
ax[1, 1].plot(dist, [abs(scope.parallax_error(r, 300.0, h)) / 10 for r in dist], label="adjusted (AO) to 300 m")
ax[1, 1].set_xlabel("target distance [m]")
ax[1, 1].set_ylabel("max. reticle shift on target [cm]")
ax[1, 1].set_title("Parallax error (eye at the edge of the exit pupil)")
ax[1, 1].legend(fontsize=8)
ax[1, 1].grid(alpha=0.3)
fig.tight_layout()
out1 = RESULTS / "exp08_zoom_scope.png"
fig.savefig(out1, dpi=150)

# ---------- Figure 2: layout at 3x and 9x ----------
fig, ax = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
for a_, M in zip(ax, (M_MIN, M_MAX)):
    sys_ = scope.system(M)
    half = scope.real_fov(M) / 2
    z_end = sys_.total_length + scope.eye_relief(M) + 20
    for y0, u0, col in ((D / 2, 0, "tab:blue"), (-D / 2, 0, "tab:blue"), (0, half, "tab:red"),
                        (D / 2, half, "tab:red"), (-D / 2, half, "tab:red")):
        zs, ys, us = sys_.trace(y0, u0)
        zs = np.append(zs, z_end)
        ys = np.append(ys, ys[-1] + us[-1] * (z_end - sys_.total_length))
        a_.plot(zs, ys, color=col, lw=0.8)
    zl = np.cumsum([0] + [e.length for e in sys_.elements])
    lens_z = [zl[i] for i, e in enumerate(sys_.elements) if type(e).__name__ == "ThinLens"]
    for z, name in zip(lens_z, ("objective", "erector 1", "erector 2", "eyepiece")):
        hh = D / 2 + 2 if name == "objective" else 14
        a_.annotate("", xy=(z, hh), xytext=(z, -hh), arrowprops=dict(arrowstyle="<->", lw=1.5))
        a_.text(z, -hh - 4 if name == "erector 2" else hh + 1, name, ha="center", fontsize=8)
    for z, name in ((scope.f_objective, "FFP"), (scope.f_objective + L, "SFP")):
        a_.axvline(z, color="gray", ls=":", lw=1)
        a_.text(z, -D / 2 - 6, name, ha="center", fontsize=8)
    z_eye = sys_.total_length + scope.eye_relief(M)
    a_.axvline(z_eye, color="green", lw=1.5)
    a_.text(z_eye, D / 2 - 4, f"eye\nER {scope.eye_relief(M):.0f} mm", ha="center", fontsize=8, color="green")
    a_.axhline(0, color="k", lw=0.5)
    a_.set_ylim(-D / 2 - 9, D / 2 + 8)
    a_.set_title(f"{M:.0f}x: blue = on-axis marginal rays, red = edge-of-field bundle")
    a_.set_ylabel("y [mm]")
    a_.grid(alpha=0.3)
ax[1].set_xlabel("z [mm]")
fig.tight_layout()
out2 = RESULTS / "exp08_layout.png"
fig.savefig(out2, dpi=150)
print(f"\nSaved {out1}\nSaved {out2}")