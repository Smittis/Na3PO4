"""
PO4 orientational autocorrelation function (OACF) for Na3PO4 MD trajectories.

C(t) = <P2(cos theta(t))>

where theta is the angle between a P-O bond vector at time origin 0 and at
lag time t, and P2 is the second Legendre polynomial: P2(x) = (3x^2 - 1) / 2.

Interpretation:
  C(t) ~ 1  ->  PO4 tetrahedra frozen (solid-like, low T)
  C(t) -> 0  ->  PO4 tetrahedra freely rotating (superionic-like, high T)

Atom types in dump files:  1 = Na,  2 = O,  3 = P

Run from the Na3PO4 parent directory:
    python oacf.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── dump file parser ──────────────────────────────────────────────────────────

def parse_dump(filepath):
    """
    Read a LAMMPS dump file (format: id type q x y z).
    Returns list of (timestep, box_lengths_array, atoms_array).
    atoms_array columns: id, type, x, y, z  (charge dropped).
    """
    frames = []
    with open(filepath) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if "ITEM: TIMESTEP" in lines[i]:
            ts      = int(lines[i + 1])
            natoms  = int(lines[i + 3])
            box     = np.array([
                float(lines[i + 5].split()[1]) - float(lines[i + 5].split()[0]),
                float(lines[i + 6].split()[1]) - float(lines[i + 6].split()[0]),
                float(lines[i + 7].split()[1]) - float(lines[i + 7].split()[0]),
            ])
            atoms = []
            for j in range(natoms):
                p = lines[i + 9 + j].split()
                # id type q x y z  ->  keep id, type, x, y, z
                atoms.append([int(p[0]), int(p[1]),
                               float(p[3]), float(p[4]), float(p[5])])
            frames.append((ts, box, np.array(atoms)))
            i += 9 + natoms
        else:
            i += 1
    return frames


# ── topology: find which 4 O atoms belong to each P ──────────────────────────

def build_topology(atoms, box, cutoff=2.0):
    """
    For each P atom (type 3) find its 4 covalently bonded O atoms (type 2).
    Returns:
        atoms_sorted  --  atoms sorted by id, shape (N, 5)
        topology      --  int array (n_PO4, 5): [p_idx, o1, o2, o3, o4]
                          indices into atoms_sorted
    """
    order = np.argsort(atoms[:, 0])
    a = atoms[order]                          # sorted by atom id

    p_idx = np.where(a[:, 1] == 3)[0]
    o_idx = np.where(a[:, 1] == 2)[0]
    o_pos = a[o_idx, 2:5]

    topology = []
    for pi in p_idx:
        p_pos = a[pi, 2:5]
        dr    = o_pos - p_pos
        dr   -= np.round(dr / box) * box      # minimum image
        dist  = np.linalg.norm(dr, axis=1)
        near4 = np.argsort(dist)[:4]
        assert dist[near4[-1]] < cutoff, (
            f"4th nearest O at {dist[near4[-1]]:.2f} Å exceeds cutoff {cutoff} Å"
        )
        topology.append([pi] + [o_idx[k] for k in near4])

    return a, np.array(topology, dtype=int)


# ── bond vectors for every frame ──────────────────────────────────────────────

def bond_vectors_all_frames(frames, topology):
    """
    Compute normalised P-O bond vectors for every frame.
    Returns array of shape (n_frames, n_PO4, 4, 3).
    """
    n_frames = len(frames)
    n_po4    = len(topology)
    vecs     = np.zeros((n_frames, n_po4, 4, 3))

    for fi, (_, box, atoms) in enumerate(frames):
        order = np.argsort(atoms[:, 0])
        pos   = atoms[order, 2:5]             # sorted positions

        for ti, row in enumerate(topology):
            p_pos = pos[row[0]]
            for bi, oi in enumerate(row[1:]):
                dr  = pos[oi] - p_pos
                dr -= np.round(dr / box) * box
                vecs[fi, ti, bi] = dr / np.linalg.norm(dr)

    return vecs


# ── OACF ─────────────────────────────────────────────────────────────────────

def p2(x):
    return (3.0 * x**2 - 1.0) / 2.0


def compute_oacf(vecs):
    """
    C(t) = <P2(cos theta(t))> averaged over all time origins, PO4 units,
    and the 4 P-O bonds per tetrahedron.
    vecs: (n_frames, n_PO4, 4, 3)
    Returns C array of length n_frames // 2.
    """
    n = vecs.shape[0]
    max_lag = n // 2
    C = np.zeros(max_lag)

    for lag in range(max_lag):
        ref  = vecs[:n - lag]           # (n_origins, n_PO4, 4, 3)
        cur  = vecs[lag:]               # (n_origins, n_PO4, 4, 3)
        dots = np.sum(ref * cur, axis=-1)   # (n_origins, n_PO4, 4)
        C[lag] = np.mean(p2(dots))

    return C


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    temperatures  = [600, 800, 1000, 1200]
    dt_ps         = 0.002 * 2000          # timestep × dump interval = 4 ps
    colors        = ["steelblue", "seagreen", "darkorange", "crimson"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for T, color in zip(temperatures, colors):
        dump_file = Path(f"lamps_{T}K/dump.prod_{T}K")
        if not dump_file.exists():
            print(f"{T} K: dump file not found, skipping")
            continue

        print(f"{T} K: loading {dump_file} ...", flush=True)
        frames = parse_dump(dump_file)
        print(f"       {len(frames)} frames loaded")

        if len(frames) < 5:
            print(f"       too few frames, skipping")
            continue

        _, box0, atoms0 = frames[0]
        atoms_sorted, topology = build_topology(atoms0, box0)
        print(f"       {len(topology)} PO4 units")

        vecs = bond_vectors_all_frames(frames, topology)
        C    = compute_oacf(vecs)
        time = np.arange(len(C)) * dt_ps

        ax.plot(time, C, label=f"{T} K", color=color, lw=2)
        print(f"       C(0) = {C[0]:.3f},  C(t_max) = {C[-1]:.3f}")

    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("Time lag (ps)")
    ax.set_ylabel(r"$C(t) = \langle P_2(\cos\theta) \rangle$")
    ax.set_title(r"PO$_4$ orientational autocorrelation — Na$_3$PO$_4$")
    ax.legend()
    ax.set_ylim(-0.3, 1.05)
    plt.tight_layout()
    plt.savefig("oacf_PO4.png", dpi=150)
    print("\nSaved oacf_PO4.png")


if __name__ == "__main__":
    main()
