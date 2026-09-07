#!/usr/bin/env python3
"""Build + validate the 5 core SWCNT chiralities (§9).

All tubes built to IDENTICAL axial length (build long, trim to a common centered
window) with a_cc = 1.42 Å, open ends. Validate analytic vs measured diameter
(<0.5 Å). Write PDB/PDBQT/XYZ + metadata + typing sidecar (carbons type A,
charge 0.000) + geometry_validation.tsv. Receptor PDBQT is fully column-controlled
(not Open Babel defaults).
"""
import os, sys, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import ROOT, load_configs, write_json, utc_now, sha256_file
import lib_struct as ls

OUT = os.path.join(ROOT, "results", "swcnt")
os.makedirs(OUT, exist_ok=True)

def trim_to_length(tube, L):
    """Trim atoms to a centered axial window of length L (Å). Open ends."""
    pos = tube.get_positions()
    zc = pos[:, 2].mean()
    keep = np.abs(pos[:, 2] - zc) <= L / 2.0
    return tube[keep]

def main():
    cfg = load_configs()
    sc = cfg["config"]["swcnt"]
    a_cc = sc["a_cc_angstrom"]
    common_L = max(84.0, sc["target_length_angstrom"])   # identical for all tubes
    tol = sc["diameter_tolerance_angstrom"]
    chiralities = cfg["config"]["chiralities"]

    rows = []
    for n, m in chiralities:
        # build long enough that after trimming we keep >= common_L
        tube = ls.build_tube(n, m, target_length_A=common_L + 25.0, a_cc=a_cc)
        tube = trim_to_length(tube, common_L)
        pos = tube.get_positions()
        geo = ls.tube_geometry(n, m, a_cc)
        measured_d = ls.measured_diameter(pos)
        axial = float(pos[:, 2].ptp())
        diff = abs(measured_d - geo["analytic_diameter_A"])
        stem = os.path.join(OUT, f"{n}_{m}")
        ls.write_pdb(pos, stem + ".pdb")
        ls.write_xyz(pos, stem + ".xyz")
        ls.write_carbon_pdbqt(pos, stem + ".pdbqt",
                              atom_type=sc["carbon_atom_type"], charge=sc["carbon_charge"])
        typ = ls.validate_pdbqt_typing(stem + ".pdbqt")
        struct_sha = sha256_file(stem + ".pdbqt")
        end_excl = cfg["config"]["swcnt"]["end_exclusion_angstrom"]
        central = axial - 2 * end_excl
        # typing sidecar
        with open(stem + ".typing.txt", "w") as fh:
            fh.write(f"SWCNT ({n},{m}) receptor typing\n")
            fh.write(f"atom_type=A (graphitic sp2 -> aromatic carbon)\n")
            fh.write(f"partial_charge=0.000 (no validated charge model for defect-free sidewall;\n")
            fh.write(f"  Gasteiger on a periodic carbon lattice is not meaningful)\n")
            fh.write(f"n_atoms={typ['n_atoms']}  all_type_A={typ['all_type_A']}  "
                     f"all_charge_zero={typ['all_charge_zero']}\n")
            fh.write(f"a_cc={a_cc} A  axial_length={axial:.3f} A  open ends (trimmed)\n")
            fh.write(f"structure_sha256={struct_sha}\n")
            fh.write(f"builder=ASE nanotube; tool versions in provenance/\n")
        # per-tube metadata
        meta = {
            "chirality_n": n, "chirality_m": m,
            "analytic_diameter_A": round(geo["analytic_diameter_A"], 4),
            "measured_diameter_A": round(measured_d, 4),
            "diameter_diff_A": round(diff, 4),
            "chiral_angle_deg": round(geo["chiral_angle_deg"], 3),
            "axial_length_A": round(axial, 3),
            "number_of_carbon_atoms": len(tube),
            "end_treatment": "open (trimmed, no H caps)",
            "central_sidewall_length_A": round(central, 3),
            "carbon_atom_type": sc["carbon_atom_type"],
            "carbon_charge": sc["carbon_charge"],
            "structure_sha256": struct_sha,
            "diameter_within_tolerance": bool(diff < tol),
        }
        with open(stem + ".metadata.tsv", "w") as fh:
            fh.write("\t".join(meta.keys()) + "\n")
            fh.write("\t".join(str(v) for v in meta.values()) + "\n")
        rows.append(meta)
        print(f"({n},{m}): d_meas={measured_d:.2f} d_analytic={geo['analytic_diameter_A']:.2f} "
              f"(diff {diff:.3f}) axial={axial:.1f}A atoms={len(tube)} "
              f"typeA={typ['all_type_A']} q0={typ['all_charge_zero']}")

    # geometry_validation.tsv
    cols = list(rows[0].keys())
    with open(os.path.join(OUT, "geometry_validation.tsv"), "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    all_ok = all(r["diameter_within_tolerance"] for r in rows)
    lengths = sorted({round(r["axial_length_A"], 2) for r in rows})
    # identical within atomic-discretization tolerance at the open (trimmed) ends
    identical_len = (max(lengths) - min(lengths)) < 1.0
    all_min_len = all(r["axial_length_A"] >= 80.0 for r in rows)
    write_json(os.path.join(ROOT, "results", "gates", "swcnt_geometry.json"),
               {"gate": "SWCNT_geometry_validation",
                "passed": bool(all_ok and identical_len and all_min_len),
                "all_diameters_within_0.5A": all_ok,
                "design_window_length_A": common_L,
                "identical_axial_length_within_1A": identical_len,
                "all_at_least_80A": all_min_len,
                "axial_lengths_A": lengths,
                "length_spread_A": round(max(lengths) - min(lengths), 2),
                "length_note": "common 90 Å design window; <1 Å spread is open-end atomic "
                               "discretization, not a real length difference",
                "n_chiralities": len(rows), "utc": utc_now()})
    print(f"\nGeometry gate: all_diam_ok={all_ok} identical_len(<1A)={identical_len} "
          f">=80A={all_min_len} (lengths={lengths})")

if __name__ == "__main__":
    main()
