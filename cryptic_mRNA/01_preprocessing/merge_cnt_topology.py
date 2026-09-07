#!/usr/bin/env python3
"""
merge_cnt_topology.py — insert the frozen CNT molecule into a pdb2gmx topology
and prepend the CNT coordinates to the nucleic-acid .gro.

Usage:
  merge_cnt_topology.py topol.top nucleic.gro system/cnt.pdb system/cnt.itp \
                        out.top out.gro

- Adds  #include "<abs>/system/cnt.itp"  after the forcefield include line.
- Appends  'CNT   1'  to the [ molecules ] section (CNT listed FIRST in coords).
- Writes out.gro = CNT atoms (from cnt.pdb) followed by all nucleic-acid atoms.
"""
import sys
from pathlib import Path

def read_pdb_xyz(p):
    at = []
    for ln in open(p):
        if ln.startswith(("ATOM", "HETATM")):
            name = ln[12:16].strip()
            x = float(ln[30:38]) / 10.0
            y = float(ln[38:46]) / 10.0
            z = float(ln[46:54]) / 10.0
            at.append((name, x, y, z))
    return at

def main(top_in, gro_in, cnt_pdb, cnt_itp, top_out, gro_out):
    cnt = read_pdb_xyz(cnt_pdb)
    inc = '#include "%s"\n' % Path(cnt_itp).resolve().as_posix()

    lines = open(top_in).read().splitlines(keepends=True)
    out, added_inc, in_mol = [], False, False
    for ln in lines:
        # add CNT include right after the main forcefield.itp include
        if (not added_inc) and ln.strip().startswith('#include') and 'forcefield.itp' in ln:
            out.append(ln); out.append(inc); added_inc = True; continue
        out.append(ln)
    if not added_inc:                      # fallback: put include at very top
        out = [inc] + out
    # append CNT to [ molecules ]
    text = "".join(out).rstrip() + "\n"
    if "[ molecules ]" not in text:
        text += "\n[ molecules ]\n; Compound   #mols\n"
    text += "CNT            1\n"
    open(top_out, "w").write(text)

    # ---- gro: CNT first, then nucleic acids ----
    g = open(gro_in).read().splitlines()
    title, n_na = g[0], int(g[1])
    na_atoms = g[2:2 + n_na]
    box = g[2 + n_na]
    ntot = len(cnt) + n_na
    with open(gro_out, "w") as fh:
        fh.write("STMN2-CE CNT+sensor merged\n")
        fh.write("%5d\n" % ntot)
        ai = 0
        for _, x, y, z in cnt:
            ai += 1
            fh.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"
                     % (1, "CNT", "CA", ai % 100000, x, y, z))
        for ln in na_atoms:
            ai += 1
            # renumber atom serial (cols 15-20) keeping residue block intact
            fh.write(ln[:15] + "%5d" % (ai % 100000) + ln[20:] + "\n")
        fh.write(box + "\n")
    print("[merge] wrote %s (+CNT include, CNT in [molecules]) and %s (%d atoms)"
          % (top_out, gro_out, ntot))

if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    main(*sys.argv[1:])
