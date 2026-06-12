"""
Compute Na+ diffusion coefficient and ionic conductivity from LAMMPS MSD output.

Usage:
    python conductivity.py msd_Na_800K.dat

The Nernst-Einstein equation used:
    sigma = (N_Na * z^2 * e^2 * D) / (V * k_B * T)

where D is extracted from the slope of MSD vs time (MSD = 6 D t in 3D).
"""

import sys
import numpy as np

# ---- System parameters (adjust if you change the supercell) ----
N_Na      = 96          # Na atoms in 2x2x2 supercell (12 per unit cell * 8)
z         = 1           # formal charge of Na+ in units of e
T         = 800.0       # K
V_ang3    = 3672.3585   # Å³  (equilibrated NPT box volume from thermo output)

# ---- Physical constants ----
e         = 1.60217663e-19   # C
k_B       = 1.380649e-23     # J/K
V_m3      = V_ang3 * 1e-30   # m³

# ---- Load MSD file ----
fname = sys.argv[1] if len(sys.argv) > 1 else "msd_Na_800K.dat"
data  = np.loadtxt(fname, comments="#")

steps     = data[:, 0]
msd_total = data[:, 4]          # Å²
time_ps   = steps * 0.002       # ps  (timestep = 0.002 ps)
time_s    = time_ps * 1e-12     # s

# ---- Fit linear region (middle 60% of trajectory) ----
n = len(time_s)
i0, i1 = int(0.2 * n), int(0.8 * n)
slope, intercept = np.polyfit(time_s[i0:i1], msd_total[i0:i1], 1)

# MSD = 6 D t in 3D → D = slope / 6
D_ang2_s = slope / 6                       # Å²/s
D_m2_s   = D_ang2_s * 1e-20               # m²/s  (1 Å² = 1e-20 m²)
D_cm2_s  = D_m2_s * 1e4                   # cm²/s

# ---- Nernst-Einstein conductivity ----
sigma = (N_Na * z**2 * e**2 * D_m2_s) / (V_m3 * k_B * T)   # S/m
sigma_mS_cm = sigma * 10                                     # mS/cm

print(f"Fit range:           steps {int(steps[i0])} – {int(steps[i1])}")
print(f"Diffusion coeff D:   {D_m2_s:.3e} m²/s  ({D_cm2_s:.3e} cm²/s)")
print(f"Conductivity σ:      {sigma:.3f} S/m  ({sigma_mS_cm:.3f} mS/cm)")
print()
print("Note: Nernst-Einstein assumes uncorrelated Na+ hops (Haven ratio = 1).")
print("      For a correlated estimate use the Green-Kubo current autocorrelation.")

# ---- Optional plot ----
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(time_ps, msd_total, label="MSD Na")
    ax.plot(time_ps[i0:i1],
            slope * time_s[i0:i1] + intercept,         # Å²
            "r--", label=f"fit  D={D_m2_s:.2e} m²/s")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("MSD (Å²)")
    ax.set_title(f"Na MSD at 800 K   σ = {sigma_mS_cm:.2f} mS/cm")
    ax.legend()
    plt.tight_layout()
    plt.savefig("msd_800K.png", dpi=150)
    print("Plot saved to msd_800K.png")
except ImportError:
    pass
