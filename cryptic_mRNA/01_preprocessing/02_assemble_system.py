#!/usr/bin/env python3
"""
02_assemble_system.py — place the sensor next to the frozen (9,4) tube and emit
the two starting structures Harvey compared:
    system/complex_hyb.pdb    tube + sensor ssDNA + STMN2-CE target  (Hyb)
    system/complex_unhyb.pdb  tube + sensor ssDNA only               (Unhyb)

The (GT)15 backbone is translated to sit ~6 Å outside the tube surface, roughly
parallel to the tube axis (z), with the capture:target duplex projecting away
from the wall. The final adsorbed wrap is produced by the REMD equilibration
(scripts 30_remd), exactly as Harvey let the strand adsorb under replica exchange.

Reads system/cnt.pdb, system/cnt_meta.json, system/sensor_charmm.pdb.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SYS  = ROOT / "system"

def read_pdb(p):
    recs = []
    for ln in open(p):
        if ln.startswith(("ATOM", "HETATM")):
            recs.append([ln, np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])])
    return recs

def write_pdb(path, blocks, title):
    with open(path, "w") as fh:
        fh.write("TITLE     %s\n" % title)
        i = 0
        for recs in blocks:
            for ln, xyz in recs:
                i += 1
                fh.write(ln[:6] + "%5d" % (i % 100000) + ln[11:30]
                         + "%8.3f%8.3f%8.3f" % (xyz[0], xyz[1], xyz[2]) + ln[54:])
            fh.write("TER\n")
        fh.write("END\n")

def main():
    meta = json.load(open(SYS / "cnt_meta.json"))
    cx, cy = meta["box_nm"][0]*10/2, meta["box_nm"][1]*10/2
    R = meta["radius_A"]
    cnt = read_pdb(SYS / "cnt.pdb")
    sensor = read_pdb(SYS / "sensor_charmm.pdb")

    # split sensor into DNA (strand A, 52 nt) vs RNA target — by residue serial:
    # amber2charmm keeps chain/residue columns; RNA residues are RA/RU/RG/RC.
    rna_res = {"RA", "RU", "RG", "RC"}
    dna = [r for r in sensor if r[0][17:20].strip() not in rna_res]
    rna = [r for r in sensor if r[0][17:20].strip() in rna_res]

    allxyz = np.array([r[1] for r in sensor])
    # translate so the sensor's DNA centroid sits at (cx, cy+R+6, mid-z), i.e. the
    # (GT)15 lies just outside the sidewall; then rotate long axis onto z.
    dna_xyz = np.array([r[1] for r in dna])
    c0 = dna_xyz.mean(0)
    # principal axis of the DNA strand -> align to z
    u, s, vt = np.linalg.svd(dna_xyz - c0)
    axis = vt[0]
    z = np.array([0, 0, 1.0])
    v = np.cross(axis, z); sn = np.linalg.norm(v)
    if sn > 1e-6:
        v /= sn; ang = np.arccos(np.clip(axis @ z, -1, 1))
        K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        Rm = np.eye(3) + np.sin(ang)*K + (1-np.cos(ang))*(K@K)
    else:
        Rm = np.eye(3)
    def place(recs):
        out = []
        for ln, xyz in recs:
            p = Rm @ (xyz - c0)
            p += np.array([cx, cy + R + 8.0, meta["box_nm"][2]*10/2])
            out.append([ln, p])
        return out
    dna_p = place(dna); rna_p = place(rna)

    write_pdb(SYS / "complex_hyb.pdb",   [cnt, dna_p, rna_p],
              "(9,4) CNT + STMN2-CE sensor ssDNA + target RNA (Hyb)")
    write_pdb(SYS / "complex_unhyb.pdb", [cnt, dna_p],
              "(9,4) CNT + STMN2-CE sensor ssDNA only (Unhyb)")
    # sensor-only PDBs (no CNT) for pdb2gmx:
    write_pdb(SYS / "sensor_hyb_only.pdb",   [dna_p, rna_p], "sensor+target for pdb2gmx")
    write_pdb(SYS / "sensor_unhyb_only.pdb", [dna_p],        "sensor ssDNA for pdb2gmx")
    print("[02] placed sensor at r=R+8A outside (9,4) wall (R=%.2f A)." % R)
    print("[02] wrote complex_{hyb,unhyb}.pdb and sensor_{hyb,unhyb}_only.pdb")
    print("[02] Next: scripts/03_solvate_ions.sh <hyb|unhyb>")

if __name__ == "__main__":
    main()
