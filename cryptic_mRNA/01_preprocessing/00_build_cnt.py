#!/usr/bin/env python3
"""
00_build_cnt.py  —  Build the (n,m) SWCNT, periodic along z, and emit:
    system/cnt.pdb           coordinates centred in x,y at the box centre
    system/cnt.itp           GROMACS moleculetype 'CNT' (type CNTC, charge 0,
                             intra-tube neighbour exclusions; frozen at run time)
    system/cnt_meta.json     axis, radius, length, n_atoms (for downstream steps)

Mirrors Harvey et al. 2017: nanotube = sp2 carbon, spans the box in z, frozen.
Reads config/md_config.yaml.
"""
import json, sys, os
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG  = yaml.safe_load(open(ROOT / "config" / "md_config.yaml"))
OUT  = ROOT / "system"; OUT.mkdir(exist_ok=True)

def main():
    from ase.build import nanotube
    nt   = CFG["nanotube"]
    box  = CFG["box"]
    n, m = nt["n"], nt["m"]
    reps = nt["n_unitcells"]

    tube = nanotube(n, m, length=reps)          # ASE: length = number of unit cells
    pos  = tube.get_positions()                 # Angstrom
    zrep = tube.cell[2, 2]                       # axial period (A) -> becomes box z

    # Centre tube axis on (box_x/2, box_y/2); base z at 0 so it tiles under PBC.
    cx, cy = box["x_nm"] * 10 / 2, box["y_nm"] * 10 / 2
    pos[:, 0] += cx - pos[:, 0].mean()
    pos[:, 1] += cy - pos[:, 1].mean()
    pos[:, 2] -= pos[:, 2].min()
    radius = float(np.mean(np.hypot(pos[:, 0] - cx, pos[:, 1] - cy)))
    natom  = len(pos)

    # --- cnt.pdb -------------------------------------------------------------
    with open(OUT / "cnt.pdb", "w") as fh:
        fh.write("TITLE     (%d,%d) SWCNT, %d unit cells, frozen sp2 carbon\n" % (n, m, reps))
        fh.write("CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1           1\n"
                 % (box["x_nm"]*10, box["y_nm"]*10, zrep))
        for i, (x, y, z) in enumerate(pos, 1):
            fh.write("ATOM  %5d %-4s CNT A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n"
                     % (i % 100000, "CA", 1, x, y, z))
        fh.write("END\n")

    # --- neighbour exclusions (pairs within 1.8 A ~ bonded; 1-3 within 2.6 A) --
    # Rigid frozen tube: exclude close intra-tube pairs so no spurious huge LJ.
    from scipy.spatial import cKDTree
    tree  = cKDTree(pos)
    pairs = tree.query_pairs(r=2.6)             # covers 1-2 (~1.42A) and 1-3 (~2.46A)
    excl  = {}
    for i, j in pairs:
        excl.setdefault(i + 1, []).append(j + 1)

    # --- cnt.itp -------------------------------------------------------------
    with open(OUT / "cnt.itp", "w") as fh:
        fh.write("; (%d,%d) SWCNT, %d cells, %d C atoms. Frozen sp2 carbon (Harvey 2017).\n"
                 % (n, m, reps, natom))
        fh.write('#include "%s/forcefield/cnt_atomtypes.itp"\n\n' % ROOT.as_posix())
        fh.write("[ moleculetype ]\n; name   nrexcl\nCNT      1\n\n")
        fh.write("[ atoms ]\n;   nr  type  resnr residue atom  cgnr   charge     mass\n")
        for i in range(1, natom + 1):
            fh.write("%6d  CNTC     1    CNT    CA  %6d   0.0000  12.0110\n" % (i, i))
        fh.write("\n[ exclusions ]\n")
        for i in sorted(excl):
            fh.write("%6d " % i + " ".join("%d" % j for j in sorted(excl[i])) + "\n")
        fh.write("\n")

    meta = dict(n=n, m=m, n_unitcells=reps, n_atoms=natom,
                radius_A=radius, axis="z", zrep_A=float(zrep),
                box_nm=[box["x_nm"], box["y_nm"], round(zrep/10, 4)])
    json.dump(meta, open(OUT / "cnt_meta.json", "w"), indent=2)
    print("[00_build_cnt] (%d,%d) x%d -> %d C atoms, radius %.3f A, axial %.3f nm"
          % (n, m, reps, natom, radius, zrep/10))
    print("[00_build_cnt] IMPORTANT: set box.z_nm in config to %.4f (tube axial period)"
          % (zrep/10))
    print("[00_build_cnt] wrote system/cnt.pdb, cnt.itp, cnt_meta.json")

if __name__ == "__main__":
    sys.exit(main())
