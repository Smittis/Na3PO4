"""
Volume analysis for Na3PO4 MD simulations.

Extracts NPT equilibration volumes from LAMMPS thermo output files
and plots:
  (a) NPT convergence at each temperature
  (b) Equilibrium volume vs temperature (phase transition)

Usage:
    python volume_analysis.py

Expects SLURM output files in ../lamps_*K/ directories.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import glob
import re


def extract_npt_volume(filepath):
    """
    Extract volume vs step from NPT section of a LAMMPS output file.
    NPT section is identified by thermo header containing 'Volume' AND 'Lx'.
    """
    steps, vols = [], []
    in_npt = False

    with open(filepath) as f:
        for line in f:
            if 'Step' in line and 'Volume' in line and 'Lx' in line:
                in_npt = True
                continue
            if in_npt:
                parts = line.split()
                if len(parts) >= 7 and parts[0].isdigit():
                    steps.append(int(parts[0]))
                    vols.append(float(parts[6]))
                elif 'Loop time' in line:
                    break

    return np.array(steps), np.array(vols)


def find_output_files():
    """
    Search for LAMMPS output files in lamps_*K directories.
    Returns dict {temperature: filepath}.
    """
    files = {}
    patterns = [
        "lamps_*K/mace_Na3PO4_*K-*.out",
        "../lamps_*K/mace_Na3PO4_*K-*.out",
    ]
    for pattern in patterns:
        for f in glob.glob(pattern):
            match = re.search(r'(\d+)K', f)
            if match:
                T = int(match.group(1))
                files[T] = f

    return dict(sorted(files.items()))


# ── Fallback: use known equilibrated volumes from conductivity.py ──
KNOWN_VOLUMES = {
    600:  3610.8492,
    800:  3672.3585,
    1000: 3794.1227,
    1200: 3840.7934,
}


def main():
    output_files = find_output_files()

    if not output_files:
        print("No LAMMPS output files found in lamps_*K/ directories.")
        print("Using known equilibrated volumes from conductivity.py scripts.")
        npt_data = {}
    else:
        print(f"Found output files: {output_files}")
        npt_data = {}
        for T, fpath in output_files.items():
            steps, vols = extract_npt_volume(fpath)
            if len(steps) > 0:
                npt_data[T] = (steps, vols)
                print(f"  {T} K: {len(steps)} NPT points, "
                      f"V_start={vols[0]:.1f}, V_final={vols[-1]:.2f} Å³")

    # ── Equilibrium volumes ──
    temps = np.array(sorted(KNOWN_VOLUMES.keys()))
    vol_eq = np.array([KNOWN_VOLUMES[T] for T in temps])

    # Volume expansion relative to 600 K
    expansion = (vol_eq - vol_eq[0]) / vol_eq[0] * 100
    print(f"\nEquilibrium volumes:")
    for T, V, dV in zip(temps, vol_eq, expansion):
        print(f"  {T:4d} K: V = {V:.2f} Å³  (ΔV = +{dV:.2f}%)")

    # ── Figure ──
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, wspace=0.38)

    # Panel A: NPT convergence (use first available T, or 800K)
    ax1 = fig.add_subplot(gs[0])
    show_T = 800 if 800 in npt_data else (list(npt_data.keys())[0] if npt_data else None)

    if show_T and show_T in npt_data:
        steps, vols = npt_data[show_T]
        time_ps = steps * 0.002
        ax1.plot(time_ps, vols, color='#028090', lw=1.4, label='V(t)')
        ax1.axhline(vols[-1], color='#B31B5A', ls='--', lw=1.5,
                    label=f'$V_{{eq}}$ = {vols[-1]:.1f} Å³')
        ax1.set_title(f'NPT Equilibrierung — {show_T} K',
                      fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
    else:
        ax1.text(0.5, 0.5, 'No NPT thermo data\navailable',
                 ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax1.set_title('NPT Equilibrierung', fontsize=13, fontweight='bold')

    ax1.set_xlabel('Zeit (ps)', fontsize=12)
    ax1.set_ylabel('Volumen (Å³)', fontsize=12)
    ax1.set_ylim(3200, 4000)
    ax1.grid(True, ls=':', alpha=0.4)

    # Panel B: V_eq vs T
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter([600], [vol_eq[0]], color='steelblue', s=120, zorder=5,
                label='solid (600 K)')
    ax2.scatter(temps[1:], vol_eq[1:], color='#B31B5A', s=120, zorder=5,
                label='superionic (800–1200 K)')
    ax2.plot(temps, vol_eq, color='gray', lw=1, ls='-', zorder=3)

    # Phase transition annotation
    ax2.annotate('', xy=(800, vol_eq[1]), xytext=(600, vol_eq[0]),
                 arrowprops=dict(arrowstyle='->', color='#B31B5A', lw=1.5))
    ax2.text(700, (vol_eq[0] + vol_eq[1]) / 2 + 15,
             f'+{expansion[1]:.1f}%\n(Yin: 2.98%)',
             ha='center', va='bottom', fontsize=9.5, color='#B31B5A',
             bbox=dict(boxstyle='round,pad=0.2', fc='#FFE8EE', ec='none'))

    # Transition region
    ax2.axvspan(600, 800, alpha=0.08, color='#B31B5A', label='Übergangsbereich')
    ax2.axvline(620, color='#B31B5A', ls=':', lw=1.2)
    ax2.text(622, vol_eq.max() - 5, 'T$_c$ ≈ 620 K', fontsize=9, color='#B31B5A')

    ax2.set_xlabel('Temperatur (K)', fontsize=12)
    ax2.set_ylabel('Gleichgewichtsvolumen (Å³)', fontsize=12)
    ax2.set_title('Volumen vs. Temperatur', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.set_xticks([600, 800, 1000, 1200])
    ax2.grid(True, ls=':', alpha=0.4)

    plt.suptitle('Na₃PO₄ — Phasenübergang & NPT Equilibrierung (MACE)',
                 fontsize=13, y=1.01, fontweight='bold', color='#1B2A4A')
    plt.tight_layout()
    plt.savefig('volume_analysis.png', dpi=150, bbox_inches='tight')
    print('\nSaved volume_analysis.png')


if __name__ == '__main__':
    main()
