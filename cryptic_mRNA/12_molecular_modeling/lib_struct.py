"""SWCNT geometry + controlled receptor-PDBQT writing (§9).

The receptor PDBQT is written with full column control (every carbon typed `A`,
charge 0.000) rather than relying on Open Babel defaults (§9.3).
"""
import numpy as np

def tube_geometry(n, m, a_cc=1.42):
    d = a_cc * np.sqrt(3 * (n**2 + n*m + m**2)) / np.pi
    theta = np.degrees(np.arctan(np.sqrt(3) * m / (2*n + m)))
    return {"analytic_diameter_A": float(d), "chiral_angle_deg": float(theta)}

def build_tube(n, m, target_length_A=90.0, a_cc=1.42):
    from ase.build import nanotube
    unit = nanotube(n, m, length=1, bond=a_cc)
    cell_len = unit.cell.lengths()[2]
    n_cells = max(1, int(np.ceil(target_length_A / cell_len)))
    tube = nanotube(n, m, length=n_cells, bond=a_cc)
    tube.center()
    # clear the degenerate 2D periodic cell so downstream writers don't invert it
    tube.set_pbc(False)
    tube.set_cell([0, 0, 0])
    return tube

def measured_diameter(positions):
    xy = positions[:, :2] - positions[:, :2].mean(axis=0)
    return float(2 * np.sqrt((xy**2).sum(axis=1)).mean())

def write_xyz(positions, path, symbol="C"):
    with open(path, "w") as fh:
        fh.write(f"{len(positions)}\nSWCNT\n")
        for p in positions:
            fh.write(f"{symbol} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

def write_pdb(positions, path, resname="CNT"):
    with open(path, "w") as fh:
        for i, p in enumerate(positions, 1):
            fh.write(f"ATOM  {i:5d}  C   {resname} A{1:4d}    "
                     f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00           C\n")
        fh.write("END\n")

def write_carbon_pdbqt(positions, path, resname="CNT", atom_type="A", charge=0.000):
    """Rigid all-carbon receptor PDBQT, fully column-controlled (§9.3)."""
    with open(path, "w") as fh:
        fh.write("REMARK  SWCNT rigid receptor; all carbons type "
                 f"{atom_type}, charge {charge:.3f} (graphitic sp2, documented).\n")
        for i, p in enumerate(positions, 1):
            fh.write(
                f"ATOM  {i:5d}  C   {resname} A{1:4d}    "
                f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00    "
                f"{charge:6.3f} {atom_type:<2s}\n")
        fh.write("TER\n")

def validate_pdbqt_typing(path):
    types, charges, n = set(), set(), 0
    with open(path) as fh:
        for l in fh:
            if l.startswith(("ATOM", "HETATM")):
                n += 1
                toks = l.split()
                types.add(toks[-1])
                try:
                    charges.add(round(float(toks[-2]), 3))
                except Exception:
                    pass
    return {"n_atoms": n, "atom_types": sorted(types),
            "charges": sorted(charges),
            "all_type_A": types == {"A"},
            "all_charge_zero": charges <= {0.0}}
