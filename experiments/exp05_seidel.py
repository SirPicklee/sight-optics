"""Exp05: Seidel (third-order) aberrations.

1) Seidel sums for the singlet and the two bending solutions of the achromat.
2) Validation: Seidel prediction vs exact ray trace, small aperture/field (third-order regime)
   and full aperture/field (where higher orders appear).
3) Which aberrations can lens bending fix, and which can it not?
Figures: per-surface contributions, S_I and S_II vs bending, field curves T/S vs field angle.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sightoptics import (
    N_BK7,
    N_F2,
    SEIDEL_NAMES,
    achromat_design,
    bending_solutions,
    measure_real,
    planoconvex,
    scale_to_efl,
    seidel,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
F, SEMI, FIELD = 200.0, 20.0, 2.5

singlet = scale_to_efl(planoconvex(F, SEMI, N_BK7), F)
sols = bending_solutions(F, SEMI, N_BK7, N_F2)
doublets = [achromat_design(F, SEMI, N_BK7, N_F2, c1) for c1 in sols]
systems = [("singlet", singlet)] + [(f"doublet #{k}", d) for k, d in enumerate(doublets, 1)]
chosen = doublets[-1]

# ---------- 1) Seidel sums ----------
print(f"1) Seidel sums [mm], semi-aperture {SEMI} mm, field {FIELD} deg")
short = ["S_I", "S_II", "S_III", "S_IV", "S_V", "C_I", "C_II"]
print(f"   {'system':<12}" + "".join(f"{s:>11}" for s in short))
for name, s in systems:
    t = seidel(s, SEMI, FIELD).totals
    print(f"   {name:<12}" + "".join(f"{v:>11.2e}" for v in t))

# ---------- 2) Validation ----------
for semi, fld, label in ((2.0, 0.5, "small aperture & field (third-order regime)"),
                         (0.95 * SEMI, FIELD, "full aperture & field")):
    pred = seidel(chosen, semi, fld).transverse()
    real = measure_real(chosen, semi, fld)
    print(f"\n2) Doublet #{len(doublets)}: Seidel vs real rays, {label}: h = {semi:.1f} mm, field = {fld} deg")
    print(f"   {'quantity [µm]':<46}{'Seidel':>10}{'real':>10}")
    for k in real:
        print(f"   {k:<46}{pred[k]*1e3:>10.3f}{real[k]*1e3:>10.3f}")

# ---------- 3) What bending can and cannot do ----------
print("\n3) Effect of bending (Seidel blur measures at full aperture, 2.5 deg) [µm]")
keys = ["LSA (marginal focus - paraxial focus)", "tangential coma (transverse)",
        "astigmatism: T focus - S focus (longitudinal)", "Petzval surface sag (longitudinal)"]
print(f"   {'system':<12}{'R1':>9}" + "".join(f"{k.split(' (')[0][:14]:>16}" for k in keys))
for name, s in systems:
    tr = seidel(s, SEMI, FIELD).transverse()
    print(f"   {name:<12}{s.surfaces[0].R:>9.1f}" + "".join(f"{tr[k]*1e3:>16.1f}" for k in keys))

# ---------- Figures ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

# (a) per-surface contributions of the chosen doublet
res = seidel(chosen, SEMI, FIELD)
rows = res.surfaces[:, :5]
x = np.arange(5)
w = 0.2
for i in range(rows.shape[0]):
    ax[0].bar(x + (i - 1.5) * w, rows[i], w, label=f"surface {i + 1}")
ax[0].bar(x + 1.5 * w, rows.sum(axis=0), w, color="k", label="total")
ax[0].set_xticks(x, [n.split(" ", 1)[1] for n in SEIDEL_NAMES[:5]], rotation=20)
ax[0].axhline(0, color="k", lw=0.5)
ax[0].set_ylabel("Seidel coefficient [mm]")
ax[0].set_title(f"Doublet #{len(doublets)}: contribution of each surface")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3, axis="y")

# (b) S_I and S_II vs bending
c1 = np.linspace(0.0, 0.016, 81)
S = np.array([seidel(achromat_design(F, SEMI, N_BK7, N_F2, c), SEMI, FIELD).totals for c in c1])
ax[1].plot(c1, S[:, 0], label="S_I spherical")
ax[1].plot(c1, S[:, 1], label="S_II coma")
ax[1].plot(c1, S[:, 2], label="S_III astigmatism")
ax[1].plot(c1, S[:, 3], label="S_IV Petzval")
ax[1].set_ylim(-0.05, 0.05)
for k, c in enumerate(sols, 1):
    ax[1].axvline(c, color="green", ls=":", lw=1)
    ax[1].text(c, 0.042, f" #{k}", color="green")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_xlabel("front curvature c1 = 1/R1 [1/mm]")
ax[1].set_ylabel("Seidel sum [mm]")
ax[1].set_title("What bending changes (and what it cannot)")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)

# (c) field curves of the chosen doublet
fields = np.linspace(0.1, 3.0, 25)
zt, zs, pt, ps = [], [], [], []
for fld in fields:
    m = measure_real(chosen, SEMI, fld)
    tr = seidel(chosen, SEMI, fld).transverse()
    zt.append(m["tangential field curvature (longitudinal)"])
    zs.append(m["sagittal field curvature (longitudinal)"])
    pt.append(tr["tangential field curvature (longitudinal)"])
    ps.append(tr["sagittal field curvature (longitudinal)"])
ax[2].plot(np.array(zt) * 1e3, fields, "r-", label="T (real rays)")
ax[2].plot(np.array(zs) * 1e3, fields, "b-", label="S (real rays)")
ax[2].plot(np.array(pt) * 1e3, fields, "ko", ms=3, mfc="none", label="Seidel prediction")
ax[2].plot(np.array(ps) * 1e3, fields, "ko", ms=3, mfc="none")
ax[2].axvline(0, color="k", lw=1, label="flat reticle plane")
ax[2].set_xlabel("focus shift from paraxial image plane [µm]")
ax[2].set_ylabel("field angle [deg]")
ax[2].set_title("Field curves: why the edge of the field is blurred")
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.3)

fig.tight_layout()
out = RESULTS / "exp05_seidel.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")