"""Exp07: wavefront error, PSF and MTF of the 8x40 objective (wave optics).

1) Validation: perfect pupil vs the analytic diffraction MTF.
2) Doublet on axis: focus scan of Strehl ratio (d-line and eye-weighted).
3) Which focus for the reticle? Geometric best RMS vs best eye-weighted MTF.
4) MTF at the frequency the eye can use through an 8x eyepiece, on axis and at 2.5 deg.
5) Resolution in object space: diffraction limit of the objective vs the eye behind 8x.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from sightoptics import (
    N_BK7,
    N_F2,
    PHOTOPIC,
    WL_D,
    Wavefront,
    achromat_design,
    bending_solutions,
    best_focus,
    diffraction_mtf,
    planoconvex,
    polychromatic_mtf,
    psf_from_wavefront,
    required_samples,
    scale_to_efl,
    wavefront,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
F, SEMI, FIELD, M = 200.0, 20.0, 2.5, 8.0
LAM_MM = WL_D * 1e-3
CUTOFF = 2 * SEMI / (LAM_MM * F)  # 1/(λ F#) [cycles/mm]
# Eye: ~30 cycles/degree (20/20 vision). Through an Mx scope this is M*30 cycles per degree of object space,
# which on the objective focal plane is (M*30) / (f * tan 1°) cycles/mm.
EYE_FREQ = M * 30 / (F * math.tan(math.radians(1)))
freq = np.linspace(0, CUTOFF, 200)

doublet = achromat_design(F, SEMI, N_BK7, N_F2, bending_solutions(F, SEMI, N_BK7, N_F2)[-1])
singlet = scale_to_efl(planoconvex(F, SEMI, N_BK7), F)
z_par = doublet.last_vertex + doublet.to_paraxial().bfl

# ---------- 1) Validation ----------
wf = wavefront(doublet, SEMI, 0.0, z_par, WL_D)
perfect = Wavefront(np.where(wf.mask, 0.0, np.nan), wf.mask, wf.dp, WL_D, wf.image_point)
p0 = psf_from_wavefront(perfect, F)
f0, s0, _ = p0.mtf_slices()
print("1) Validation, perfect pupil (no aberrations)")
print(f"   max |MTF_numeric - MTF_analytic| = {np.max(np.abs(s0 - diffraction_mtf(f0, CUTOFF))):.4f}, "
      f"Strehl = {p0.strehl:.4f}, cutoff = {CUTOFF:.0f} cycles/mm")


# ---------- 2) Focus scan ----------
def strehl_d(dz):
    return psf_from_wavefront(wavefront(doublet, SEMI, 0.0, z_par + dz, WL_D), F, pad=4).strehl


def eye_mtf(system, z, field=0.0, n=64):
    s, t = polychromatic_mtf(system, SEMI, field, z, PHOTOPIC, np.array([EYE_FREQ]), n=n, pad=4)
    return float((s[0] + t[0]) / 2)


dzs = np.linspace(-0.2, 0.1, 31)
strehls = np.array([strehl_d(dz) for dz in dzs])
mtf_eye_scan = np.array([eye_mtf(doublet, z_par + dz) for dz in dzs])
res_d = minimize_scalar(lambda dz: -strehl_d(dz), bounds=(-0.15, 0.0), method="bounded")
dz_d = res_d.x
wf_d = wavefront(doublet, SEMI, 0.0, z_par + dz_d, WL_D)
print("\n2) Doublet on axis, d-line")
print(f"   best focus {dz_d*1e3:+.1f} µm from paraxial focus: RMS WFE = {wf_d.rms:.4f} waves "
      f"(λ/{1/wf_d.rms:.0f}), P-V = {wf_d.pv:.3f} waves")
print(f"   Strehl = {-res_d.fun:.3f}   Maréchal estimate exp(-(2πσ)²) = {math.exp(-(2*math.pi*wf_d.rms)**2):.3f}")

# ---------- 3) Reticle focus ----------
z_geo, _ = best_focus(doublet, SEMI)
res_eye = minimize_scalar(lambda dz: -eye_mtf(doublet, z_par + dz), bounds=(-0.15, 0.05), method="bounded")
z_eye = z_par + res_eye.x
print(f"\n3) Where to put the reticle? (eye frequency on the focal plane = {EYE_FREQ:.1f} cycles/mm)")
print(f"   {'focus choice':<38}{'dz [µm]':>9}{'eye-weighted MTF':>18}")
for name, z in (("paraxial d-line focus", z_par), ("best geometric RMS (F, d, C equal)", z_geo),
                ("best d-line Strehl", z_par + dz_d), ("best eye-weighted MTF", z_eye)):
    print(f"   {name:<38}{(z - z_par)*1e3:>9.1f}{eye_mtf(doublet, z):>18.3f}")

# ---------- 4) MTF at the eye frequency ----------
print(f"\n4) MTF at {EYE_FREQ:.0f} cycles/mm (what the eye uses at 8x), reticle at the eye-weighted focus")
print(f"   {'system':<10}{'field':>7}{'sagittal':>10}{'tangential':>12}{'pupil grid':>12}")
z_s_eye = minimize_scalar(lambda z: -eye_mtf(singlet, z, n=128), bounds=(z_par - 3, z_par + 3), method="bounded").x
curves = {}
for name, s, z in (("doublet", doublet, z_eye), ("singlet", singlet, z_s_eye)):
    for fld in (0.0, FIELD):
        semi = 0.95 * SEMI if fld else SEMI
        n = required_samples(s, semi, fld, z, list(PHOTOPIC))  # avoid aliasing of large wavefront errors
        sag, tan = polychromatic_mtf(s, semi, fld, z, PHOTOPIC, freq, n=n, pad=4)
        curves[(name, fld)] = (sag, tan)
        print(f"   {name:<10}{fld:>6.1f}°{np.interp(EYE_FREQ, freq, sag):>10.3f}{np.interp(EYE_FREQ, freq, tan):>12.3f}{n:>9}x{n}")
print(f"   diffraction limit        {diffraction_mtf(EYE_FREQ, CUTOFF):>10.3f}")

# ---------- 5) Object-space resolution ----------
rayleigh = 1.22 * LAM_MM / (2 * SEMI)  # rad
eye = math.radians(1 / 60) / M
print("\n5) Angular resolution in object space")
print(f"   objective, Rayleigh 1.22 λ/D : {math.degrees(rayleigh)*3600:5.2f} arcsec  ({rayleigh*1e6:.1f} µrad)")
print(f"   eye (1 arcmin) behind {M:.0f}x     : {math.degrees(eye)*3600:5.2f} arcsec  ({eye*1e6:.1f} µrad)")
print(f"   -> the system is eye-limited by a factor of {eye / rayleigh:.1f}")

# ---------- Figure ----------
fig, ax = plt.subplots(2, 2, figsize=(12, 9.5))

p_d = psf_from_wavefront(wf_d, F, pad=8)
c = p_d.psf.shape[0] // 2
h = 20
ext = np.array([-h, h, -h, h]) * p_d.dx * 1e3
im = ax[0, 0].imshow(np.log10(p_d.psf[c - h:c + h, c - h:c + h] + 1e-6), extent=ext, cmap="inferno", vmin=-4, vmax=0)
ax[0, 0].add_patch(plt.Circle((0, 0), 1.22 * LAM_MM * F / (2 * SEMI) * 1e3, fill=False, color="c", ls="--", label="Airy first dark ring"))
ax[0, 0].set_title(f"PSF, d-line, best focus (Strehl {p_d.strehl:.2f}), log scale")
ax[0, 0].set_xlabel("x [µm]")
ax[0, 0].set_ylabel("y [µm]")
ax[0, 0].legend(fontsize=8, loc="upper right")
fig.colorbar(im, ax=ax[0, 0], label="log10(PSF / perfect peak)")

ax[0, 1].plot(dzs * 1e3, strehls, label="Strehl, d-line")
ax[0, 1].plot(dzs * 1e3, mtf_eye_scan, label=f"eye-weighted MTF @ {EYE_FREQ:.0f} c/mm")
ax[0, 1].axhline(0.8, color="k", ls=":", lw=1, label="Strehl 0.8 (diffraction limited)")
for z, lab, col in ((z_geo, "geometric best", "tab:red"), (z_eye, "eye-weighted best", "tab:green")):
    ax[0, 1].axvline((z - z_par) * 1e3, color=col, ls="--", lw=1, label=lab)
ax[0, 1].set_xlabel("image plane shift from paraxial d-line focus [µm]")
ax[0, 1].set_title("Focus scan on axis")
ax[0, 1].legend(fontsize=8)
ax[0, 1].grid(alpha=0.3)

ax[1, 0].plot(freq, diffraction_mtf(freq, CUTOFF), "k-", label="diffraction limit")
ax[1, 0].plot(freq, curves[("doublet", 0.0)][0], label="doublet, eye-weighted")
ax[1, 0].plot(freq, curves[("singlet", 0.0)][0], label="singlet, eye-weighted")
ax[1, 0].axvline(EYE_FREQ, color="gray", ls="--", lw=1, label="eye limit at 8x")
ax[1, 0].set_xlabel("spatial frequency on the focal plane [cycles/mm]")
ax[1, 0].set_ylabel("MTF")
ax[1, 0].set_title("On-axis MTF")
ax[1, 0].legend(fontsize=8)
ax[1, 0].grid(alpha=0.3)

for (name, fld), (sag, tan) in curves.items():
    if fld == 0 or name != "doublet":
        continue
    ax[1, 1].plot(freq, sag, "b-", label=f"doublet {fld}°, sagittal")
    ax[1, 1].plot(freq, tan, "r-", label=f"doublet {fld}°, tangential")
ax[1, 1].plot(freq, curves[("doublet", 0.0)][0], "g-", label="doublet on axis")
ax[1, 1].plot(freq, diffraction_mtf(freq, CUTOFF), "k-", lw=0.8, label="diffraction limit")
ax[1, 1].axvline(EYE_FREQ, color="gray", ls="--", lw=1)
ax[1, 1].set_xlim(0, 150)
ax[1, 1].set_xlabel("spatial frequency on the focal plane [cycles/mm]")
ax[1, 1].set_ylabel("MTF")
ax[1, 1].set_title(f"Edge of field ({FIELD}°) on the flat reticle plane")
ax[1, 1].legend(fontsize=8)
ax[1, 1].grid(alpha=0.3)

fig.tight_layout()
out = RESULTS / "exp07_mtf.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")