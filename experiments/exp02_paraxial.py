"""Exp02: paraxial matrix optics.

1) Kepler 8x40: matrix method vs first-order formulas (Step 1).
2) Thick vs thin lens: how much does lens thickness change the focal length?
3) Ray diagram of the telescope: marginal rays, chief rays, exit pupil location.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sightoptics import KeplerTelescope, kepler, thick_lens

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

# ---------- 1) Matrix vs first-order ----------
F_O, F_E, D, STOP = 200.0, 25.0, 40.0, 20.0
ref = KeplerTelescope(F_O, F_E, D, STOP)
sys_ = kepler(F_O, F_E)
A, B, C, Dm = sys_.abcd

_, ys, _ = sys_.trace(D / 2, 0.0)
ep_matrix = 2 * abs(ys[-1])

print("1) Kepler 8x40 - system matrix")
print(f"   [[A, B], [C, D]] = [[{A:.4f}, {B:.2f}], [{C:.2e}, {Dm:.4f}]]   det = {A*Dm - B*C:.6f}")
print(f"   {'quantity':<22}{'matrix':>10}{'formula':>10}")
print(f"   {'angular magnification':<22}{Dm:>10.3f}{-ref.magnification:>10.3f}")
print(f"   {'exit pupil [mm]':<22}{ep_matrix:>10.3f}{ref.exit_pupil:>10.3f}")
print(f"   {'eye relief [mm]':<22}{sys_.stop_image_distance():>10.3f}{ref.eye_relief:>10.3f}")

# ---------- 2) Thick vs thin lens ----------
n_bk7 = 1.5168
print("\n2) Biconvex N-BK7 lens, R1 = +100, R2 = -100 mm")
print(f"   {'t [mm]':>7}{'thin f':>10}{'thick EFL':>11}{'diff %':>8}{'BFL':>9}")
f_thin = 1 / ((n_bk7 - 1) * (1 / 100 - 1 / -100))
for t in (0.0, 4.0, 8.0, 15.0, 30.0):
    lens = thick_lens(100.0, -100.0, t, n_bk7)
    diff = 100 * (lens.efl - f_thin) / f_thin
    print(f"   {t:>7.1f}{f_thin:>10.2f}{lens.efl:>11.2f}{diff:>8.2f}{lens.bfl:>9.2f}")

# ---------- 3) Ray diagram ----------
half_fov = np.radians(ref.real_fov / 2)
fig, ax = plt.subplots(figsize=(11, 4.5))
z_end = sys_.total_length + 1.6 * sys_.stop_image_distance()


def draw(y0, u0, **kw):
    zs, ys, us = sys_.trace(y0, u0)
    zs = np.append(zs, z_end)
    ys = np.append(ys, ys[-1] + us[-1] * (z_end - sys_.total_length))
    ax.plot(zs, ys, **kw)


for y0 in (-D / 2, 0.0, D / 2):  # on-axis bundle
    draw(y0, 0.0, color="tab:blue", lw=1)
for y0 in (-D / 2, D / 2):  # off-axis bundle, marginal rays
    draw(y0, half_fov, color="tab:red", lw=1)
draw(0.0, half_fov, color="tab:red", lw=1.8, ls="--", label="chief ray (edge of field)")
ax.plot([], [], color="tab:blue", label="on-axis bundle")
ax.plot([], [], color="tab:red", label="edge-of-field bundle")

for z, h, name in ((0, D / 2 + 3, "objective"), (sys_.total_length, 10, "eyepiece")):
    ax.annotate("", xy=(z, h), xytext=(z, -h), arrowprops=dict(arrowstyle="<->", lw=2))
    ax.text(z, h + 1.5, name, ha="center")
ax.axvline(F_O, color="gray", ls=":", lw=1)
ax.text(F_O, -D / 2 - 5, "field stop /\nreticle plane", ha="center", fontsize=8)
z_ep = sys_.total_length + sys_.stop_image_distance()
ax.axvline(z_ep, color="green", lw=1.5)
ax.text(z_ep, D / 2, f"exit pupil\nER = {sys_.stop_image_distance():.1f} mm", ha="center", fontsize=8, color="green")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("z [mm]")
ax.set_ylabel("ray height y [mm]")
ax.set_title("Kepler 8x40 - paraxial ray trace")
ax.legend(loc="lower left", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
out = RESULTS / "exp02_paraxial.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")