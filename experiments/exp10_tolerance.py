"""Exp10: tolerance analysis of the 8x40 achromatic objective.

1) Nominal on-axis performance.
2) Sensitivity: push one tolerance at a time to its limit -> which parameters matter?
3) Monte Carlo: 500 virtual lenses per tolerance grade, with and without the focus compensator.
   RSS prediction vs Monte Carlo.
4) What the mechanics must provide: focus compensator range and boresight allowance.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from sightoptics import (
    MOA,
    N_BK7,
    N_F2,
    achromat_design,
    bending_solutions,
    doublet_tolerances,
    evaluate,
    monte_carlo,
    rss_prediction,
    sensitivity,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
F, SEMI, N_TRIALS = 200.0, 20.0, 500
LAMBDA_14 = 1 / 14  # RMS wavefront error for Strehl 0.8 (diffraction limited)
GRADES = {  # radius [relative], thickness [mm], index, decenter [mm]
    "precision": (0.002, 0.05, 0.0005, 0.02),
    "commercial": (0.005, 0.10, 0.0010, 0.05),
    "economy": (0.010, 0.20, 0.0020, 0.10),
}

doublet = achromat_design(F, SEMI, N_BK7, N_F2, bending_solutions(F, SEMI, N_BK7, N_F2)[-1])
z_par = doublet.last_vertex + doublet.to_paraxial().bfl
# best d-line focus from Exp07 is ~65 µm in front of the paraxial focus; find it here with the same metric
dzs = np.linspace(-0.12, 0.0, 121)
z_img = z_par + dzs[np.argmin([evaluate(doublet, SEMI, z_par + dz, refocus=False)["rms"] for dz in dzs])]

# ---------- 1) Nominal ----------
nom = evaluate(doublet, SEMI, z_img)
print("1) Nominal design (on axis, d-line)")
print(f"   RMS wavefront error {nom['rms']:.4f} waves, Strehl {nom['strehl']:.3f}; spec: RMS <= λ/14 = {LAMBDA_14:.3f}")

# ---------- 2) Sensitivity ----------
tols = doublet_tolerances(doublet, *GRADES["precision"])
sens = sensitivity(doublet, tols, SEMI, z_img)
print("\n2) Sensitivity, precision grade (one tolerance at +limit, refocused)")
print(f"   {'tolerance':<28}{'ΔRMS [waves]':>13}{'Δboresight [MOA]':>18}{'Δfocus [µm]':>13}")
for r in sorted(sens, key=lambda r: -abs(r["d_rms"])):
    print(f"   {r['tolerance'].name:<28}{r['d_rms']:>13.4f}{r['d_boresight'] / MOA:>18.3f}{r['d_focus'] * 1e3:>13.1f}")

# ---------- 3) Monte Carlo ----------
print(f"\n3) Monte Carlo, {N_TRIALS} lenses per grade, tolerances uniform in ±limit")
print(f"   {'grade':<11}{'compensator':>12}{'RMS 50%':>9}{'RMS 95%':>9}{'yield':>8}{'bore 95% [MOA]':>16}{'focus 95% [mm]':>16}")
results = {}
for grade, lims in GRADES.items():
    t = doublet_tolerances(doublet, *lims)
    for comp in (True, False):
        mc = monte_carlo(doublet, t, SEMI, z_img, N_TRIALS, seed=0, refocus=comp)
        results[(grade, comp)] = mc
        y = (mc["rms"] <= LAMBDA_14).mean() * 100
        print(f"   {grade:<11}{'refocus' if comp else 'fixed':>12}{np.percentile(mc['rms'], 50):>9.3f}"
              f"{np.percentile(mc['rms'], 95):>9.3f}{y:>7.0f}%{np.percentile(mc['boresight'], 95) / MOA:>16.2f}"
              f"{np.percentile(np.abs(mc['focus']), 95):>16.2f}")

print("\n   RSS prediction vs Monte Carlo (1-sigma), with compensator")
print(f"   {'grade':<11}{'bore RSS':>10}{'bore MC':>10}{'focus RSS':>11}{'focus MC':>10}   [MOA, mm]")
for grade, lims in GRADES.items():
    t = doublet_tolerances(doublet, *lims)
    rss = rss_prediction(nom, sensitivity(doublet, t, SEMI, z_img))
    mc = results[(grade, True)]
    print(f"   {grade:<11}{rss['boresight'] / MOA:>10.2f}{np.sqrt(np.mean(mc['boresight'] ** 2)) / MOA:>10.2f}"
          f"{rss['focus']:>11.3f}{np.std(mc['focus']):>10.3f}")

# ---------- 4) Mechanics ----------
mc = results[("commercial", True)]
print("\n4) Requirements on the mechanics (commercial grade, 99.7% of units)")
print(f"   focus compensator range: ±{np.percentile(np.abs(mc['focus']), 99.7):.2f} mm")
print(f"   boresight to be absorbed by the turrets: {np.percentile(mc['boresight'], 99.7) / MOA:.1f} MOA")

# ---------- Figure ----------
colors = {"radius": "tab:blue", "thickness": "tab:orange", "index": "tab:green", "decenter": "tab:red"}
fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
order = sorted(sens, key=lambda r: abs(r["d_rms"]))
ax[0, 0].barh([r["tolerance"].name for r in order], [abs(r["d_rms"]) for r in order],
              color=[colors[r["tolerance"].kind] for r in order])
ax[0, 0].set_xlabel("|ΔRMS wavefront error| [waves] at +limit")
ax[0, 0].set_title("Sensitivity (precision grade, refocused)")
ax[0, 0].grid(alpha=0.3, axis="x")
ax[0, 0].legend(handles=[Patch(color=c, label=k) for k, c in colors.items()], fontsize=8)

bins = np.linspace(0, 0.12, 49)
for grade in GRADES:
    ax[0, 1].hist(results[(grade, True)]["rms"], bins=bins, alpha=0.5, label=grade)
ax[0, 1].axvline(LAMBDA_14, color="k", ls="--", label="λ/14 (Strehl 0.8)")
ax[0, 1].set_xlabel("RMS wavefront error [waves]")
ax[0, 1].set_ylabel("lenses")
ax[0, 1].set_title("With focus compensator")
ax[0, 1].legend(fontsize=8)

bins = np.logspace(-2, 1.3, 50)
for grade in GRADES:
    ax[1, 0].hist(results[(grade, False)]["rms"], bins=bins, alpha=0.5, label=grade)
ax[1, 0].axvline(LAMBDA_14, color="k", ls="--", label="λ/14")
ax[1, 0].set_xscale("log")
ax[1, 0].set_xlabel("RMS wavefront error [waves]")
ax[1, 0].set_ylabel("lenses")
ax[1, 0].set_title("Fixed reticle position (no compensator)")
ax[1, 0].legend(fontsize=8)

for grade in GRADES:
    m = results[(grade, True)]
    ax[1, 1].scatter(m["focus"], m["boresight"] / MOA, s=5, alpha=0.5, label=grade)
ax[1, 1].set_xlabel("refocus needed [mm]")
ax[1, 1].set_ylabel("boresight shift [MOA]")
ax[1, 1].set_title("What the mechanics must absorb")
ax[1, 1].legend(fontsize=8)
ax[1, 1].grid(alpha=0.3)
fig.tight_layout()
out = RESULTS / "exp10_tolerance.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")