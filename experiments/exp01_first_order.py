"""Exp01: first-order design trade-offs of a Kepler telescope.

1) Print the first-order summary of a few classic configurations.
2) Plot exit pupil vs magnification for D = 40 mm against the eye pupil (day/night).
3) Plot eye relief vs eyepiece focal length.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sightoptics import KeplerTelescope

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

configs = {
    "4x32": KeplerTelescope(f_objective=120.0, f_eyepiece=30.0, d_objective=32.0, field_stop=18.0),
    "8x40": KeplerTelescope(f_objective=200.0, f_eyepiece=25.0, d_objective=40.0, field_stop=20.0),
    "10x50": KeplerTelescope(f_objective=250.0, f_eyepiece=25.0, d_objective=50.0, field_stop=20.0),
    "12x56": KeplerTelescope(f_objective=300.0, f_eyepiece=25.0, d_objective=56.0, field_stop=20.0),
}

header = f"{'cfg':>6} {'M':>5} {'L[mm]':>7} {'EP[mm]':>7} {'ER[mm]':>7} {'FOV[deg]':>9} {'AFOV[deg]':>10} {'m@100m':>7} {'TF':>6}"
print(header)
print("-" * len(header))
for name, t in configs.items():
    s = t.summary()
    print(
        f"{name:>6} {s['magnification']:5.1f} {s['length_mm']:7.1f} {s['exit_pupil_mm']:7.2f} "
        f"{s['eye_relief_mm']:7.1f} {s['real_fov_deg']:9.2f} {s['apparent_fov_deg']:10.1f} "
        f"{s['fov_m_at_100m']:7.2f} {s['twilight_factor']:6.1f}"
    )

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) exit pupil vs magnification, D = 40 mm
mags = np.linspace(2, 20, 200)
ax[0].plot(mags, 40.0 / mags, label="exit pupil (D = 40 mm)")
ax[0].axhspan(2, 3, alpha=0.2, color="gold", label="eye pupil, daylight (2-3 mm)")
ax[0].axhspan(5, 7, alpha=0.2, color="navy", label="eye pupil, night (5-7 mm)")
ax[0].set_xlabel("Magnification M")
ax[0].set_ylabel("Exit pupil [mm]")
ax[0].set_title("Exit pupil vs magnification")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)

# (b) eye relief vs eyepiece focal length, f_obj = 200 mm
f_eyes = np.linspace(8, 40, 200)
er = [KeplerTelescope(200.0, fe, 40.0, 20.0).eye_relief for fe in f_eyes]
ax[1].plot(f_eyes, er, label="thin-lens eye relief")
ax[1].axhline(90, ls="--", color="red", label="rifle-scope recoil target (~90 mm)")
ax[1].set_xlabel("Eyepiece focal length f_e [mm]")
ax[1].set_ylabel("Eye relief [mm]")
ax[1].set_title("Eye relief vs eyepiece focal length (f_obj = 200 mm)")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)

fig.tight_layout()
out = RESULTS / "exp01_first_order.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")