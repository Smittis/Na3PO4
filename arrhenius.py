"""
Arrhenius plot of Na+ ionic conductivity in Na3PO4.

Plots log(sigma*T) vs 1000/T and fits activation energy Ea
to the superionic region (800-1200 K).
"""

import numpy as np
import matplotlib.pyplot as plt

# ---- Data ----
T      = np.array([600,   800,   1000,  1200])   # K
sigma_Sm  = np.array([0.017, 4.272, 17.836, 32.302])  # S/m
sigma     = sigma_Sm * 10                              # mS/cm

k_B    = 8.617333e-5   # eV/K

# Arrhenius: sigma*T = A * exp(-Ea / kB*T)
# ln(sigma*T) = ln(A) - Ea/(kB) * (1/T)
inv_T     = 1000.0 / T          # 1000/T  for x-axis
ln_sigmaT = np.log(sigma * T)

# ---- Fit to superionic points (800-1200 K) ----
mask  = T >= 800
slope, intercept = np.polyfit(1.0 / T[mask], ln_sigmaT[mask], 1)
Ea    = -slope * k_B             # eV
x_fit = np.linspace(1.0 / 1300, 1.0 / 700, 100)
y_fit = slope * x_fit + intercept

# ---- Plot ----
fig, ax = plt.subplots(figsize=(7, 5))

ax.semilogy(inv_T[T >= 800],  sigma[T >= 800],
            "o", color="crimson", ms=8, label="superionic (800–1200 K)")
ax.semilogy(inv_T[T < 800],   sigma[T < 800],
            "s", color="steelblue", ms=8, label="solid (600 K)")

# Arrhenius fit line (in sigma units: exp(y_fit) / T_for_x)
T_fit   = 1000.0 / (x_fit * 1000)   # recover T from 1/T
sig_fit = np.exp(y_fit) / T_fit
ax.semilogy(1000 * x_fit, sig_fit, "--", color="crimson", lw=1.5,
            label=f"Arrhenius fit  $E_a$ = {Ea:.2f} eV")

# Value labels on each data point
labels = ["0.17 mS/cm", "42.7 mS/cm", "178 mS/cm", "323 mS/cm"]
offsets = [(-0.08, 1.8), (-0.06, 1.8), (-0.06, 1.8), (-0.06, 1.8)]
colors  = ["steelblue", "crimson", "crimson", "crimson"]
for i, (x, y, lbl, (dx, dy), c) in enumerate(zip(inv_T, sigma, labels, offsets, colors)):
    ax.annotate(lbl, xy=(x, y), xytext=(x + dx, y * dy),
                fontsize=8, color=c, ha="center")

ax.set_xlabel("1000 / T  (K⁻¹)")
ax.set_ylabel("σ  (mS/cm)")
ax.set_title(r"Na$^+$ conductivity in Na$_3$PO$_4$  —  Arrhenius plot")
ax.legend()
ax.set_ylim(0.05, 2000)
ax.grid(True, which="both", ls=":", alpha=0.4)

# secondary x-axis showing T in K
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
tick_T   = [700, 800, 900, 1000, 1200]
tick_pos = [1000 / t for t in tick_T]
ax2.set_xticks(tick_pos)
ax2.set_xticklabels([f"{t} K" for t in tick_T])

plt.tight_layout()
plt.savefig("arrhenius.png", dpi=150)
print(f"Activation energy Ea = {Ea:.3f} eV")
print("Plot saved to arrhenius.png")
